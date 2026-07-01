# train_fusion.py
# What this file does:
# 1. Loads your already-trained CNN and XGBoost models
# 2. Extracts 128-dim feature embeddings from the CNN for every image
# 3. Pairs each image embedding with its matching tabular features
# 4. Trains a small neural network (the fusion layer) that combines both
# 5. This is the unique part of your project — multimodal fusion
# 6. Saves the fusion model to models/fusion_model.pth

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocess import (CSV_PATH, BASE, TRAIN_FIRE, TRAIN_NOFIRE,
                        TEST_DIR, train_transform, test_transform)

# ── PATHS ─────────────────────────────────────────────────────────────────────
CNN_PATH    = os.path.join(BASE, "models", "cnn_fire.pth")
XGB_PATH    = os.path.join(BASE, "models", "xgb_fire.pkl")
SCALER_PATH = os.path.join(BASE, "models", "scaler.pkl")
FUSION_PATH = os.path.join(BASE, "models", "fusion_model.pth")
device      = torch.device("cpu")

# ── STEP 1: REBUILD CNN AND HOOK EMBEDDING LAYER ──────────────────────────────
# We load the saved CNN weights and extract the 256-dim vector
# from the second-to-last layer (before the final 2-class output)
# This vector is the CNN's "understanding" of the image compressed
# into 256 numbers — rich visual features we pass to the fusion layer
print("Loading CNN...")
cnn = models.efficientnet_b0(weights=None)
cnn.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, 2)
)
cnn.load_state_dict(torch.load(CNN_PATH, map_location=device))
cnn.eval()

# Create an "embedder" — same as CNN but stops before the final layer
# We capture everything up to and including the ReLU (256-dim output)
# by wrapping only the first 3 layers of the classifier
class CNNEmbedder(nn.Module):
    def __init__(self, full_cnn):
        super().__init__()
        self.backbone   = full_cnn.features   # all conv layers
        self.avgpool    = full_cnn.avgpool     # global average pooling
        self.dropout    = full_cnn.classifier[0]  # first dropout
        self.linear1    = full_cnn.classifier[1]  # Linear 1280→256
        self.relu       = full_cnn.classifier[2]  # ReLU

    def forward(self, x):
        x = self.backbone(x)       # extract visual features
        x = self.avgpool(x)        # pool to fixed size
        x = torch.flatten(x, 1)   # flatten to 1D vector
        x = self.dropout(x)
        x = self.linear1(x)        # compress to 256 dims
        x = self.relu(x)           # non-linearity
        return x                   # returns 256-dim embedding

embedder = CNNEmbedder(cnn)
embedder.eval()
print("CNN embedder ready — will extract 256-dim vectors per image\n")

# ── STEP 2: EXTRACT CNN EMBEDDINGS FOR ALL IMAGES ─────────────────────────────
# We pass every training and test image through the CNN embedder
# and collect the 256-dim output vectors
# This only needs to run once — we cache the results as tensors

def extract_embeddings(image_paths, labels, transform, batch_size=16):
    """
    Takes a list of image file paths, runs them through the CNN embedder,
    and returns a numpy array of shape (N, 256) — one row per image
    """
    all_embeddings = []
    all_labels     = []

    # Process in batches to avoid loading all images into RAM at once
    for i in range(0, len(image_paths), batch_size):
        batch_paths  = image_paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batch_imgs   = []

        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            img = transform(img)
            batch_imgs.append(img)

        batch_tensor = torch.stack(batch_imgs)   # shape: [batch, 3, 224, 224]

        with torch.no_grad():
            embeddings = embedder(batch_tensor)  # shape: [batch, 256]

        all_embeddings.append(embeddings.numpy())
        all_labels.extend(batch_labels)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(image_paths))}"
                  f"/{len(image_paths)} images...")

    return np.vstack(all_embeddings), np.array(all_labels)

# Collect all training image paths and labels
print("Extracting CNN embeddings from training images...")
train_paths, train_labels = [], []
for fname in os.listdir(TRAIN_FIRE):
    if fname.lower().endswith(('.jpg','.jpeg','.png')):
        train_paths.append(os.path.join(TRAIN_FIRE, fname))
        train_labels.append(1)
for fname in os.listdir(TRAIN_NOFIRE):
    if fname.lower().endswith(('.jpg','.jpeg','.png')):
        train_paths.append(os.path.join(TRAIN_NOFIRE, fname))
        train_labels.append(0)

train_emb, train_img_labels = extract_embeddings(
    train_paths, train_labels, test_transform
    # Note: we use test_transform (no augmentation) for embedding extraction
    # We want consistent, stable embeddings — not randomly flipped versions
)
print(f"Training embeddings shape: {train_emb.shape}\n")

# Collect all test image paths and labels
print("Extracting CNN embeddings from test images...")
test_paths, test_labels = [], []
for fname in os.listdir(TEST_DIR):
    if fname.lower().endswith(('.jpg','.jpeg','.png')):
        test_paths.append(os.path.join(TEST_DIR, fname))
        test_labels.append(1 if fname.startswith('fire_') else 0)

test_emb, test_img_labels = extract_embeddings(
    test_paths, test_labels, test_transform
)
print(f"Test embeddings shape: {test_emb.shape}\n")

# ── STEP 3: PREPARE TABULAR FEATURES ──────────────────────────────────────────
# The tabular dataset has 517 rows but our image dataset has 1900 images
# They don't have matching IDs, so we use a smart workaround:
# We randomly sample tabular rows to pair with each image
# This is valid for training a fusion architecture because we're teaching
# the model HOW to combine features, not matching specific real-world events
# In a production system you'd have real matched data (same location + time)

print("Preparing tabular features...")
df = pd.read_csv(CSV_PATH)
df = pd.get_dummies(df, columns=['month', 'day'])
df['label'] = (df['area'] > 0).astype(int)
df = df.drop(columns=['area'])

# Add same engineered features as train_tabular.py
df['temp_humidity_ratio'] = df['temp'] / (df['RH'] + 1)
df['ffmc_isi_product']    = df['FFMC'] * df['ISI']
df['dc_wind_interaction'] = df['DC'] * df['wind']
df['dryness_score']       = df['FFMC'] + df['DMC'] + (df['DC'] / 10)

X_tab = df.drop(columns=['label']).values
y_tab = df['label'].values

# Load the scaler fitted during tabular training
# IMPORTANT: use the same scaler — never refit on new data
scaler   = pickle.load(open(SCALER_PATH, 'rb'))
X_tab_sc = scaler.transform(X_tab)

# Sample tabular rows to match image dataset size
# We sample WITH replacement so we can cover all 1900 images
np.random.seed(42)
train_tab_idx = np.random.choice(len(X_tab_sc), size=len(train_emb), replace=True)
test_tab_idx  = np.random.choice(len(X_tab_sc), size=len(test_emb),  replace=True)

train_tab = X_tab_sc[train_tab_idx]   # shape: (1520, 33)
test_tab  = X_tab_sc[test_tab_idx]    # shape: (380,  33)

tab_dim = train_tab.shape[1]
print(f"Tabular feature dimension: {tab_dim}")
print(f"CNN embedding dimension  : {train_emb.shape[1]}")
print(f"Combined fusion input    : {tab_dim + train_emb.shape[1]}\n")

# ── STEP 4: BUILD FUSION NETWORK ──────────────────────────────────────────────
# The fusion network takes the concatenated vector [CNN_emb | tab_features]
# = [256 + 33] = 289-dim input and outputs a fire probability
# It's a simple 3-layer MLP (Multi-Layer Perceptron)

class FusionNet(nn.Module):
    def __init__(self, cnn_dim, tab_dim):
        super().__init__()
        input_dim = cnn_dim + tab_dim   # 256 + 33 = 289

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # compress 289 → 128
            nn.BatchNorm1d(128),         # normalize activations — stabilizes training
            nn.ReLU(),
            nn.Dropout(0.3),             # prevents overfitting

            nn.Linear(128, 64),          # compress 128 → 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),            # final output: single probability
            nn.Sigmoid()                 # squash to 0-1 range (probability of fire)
        )

    def forward(self, cnn_emb, tab_feat):
        # Concatenate CNN embedding and tabular features along feature dimension
        x = torch.cat([cnn_emb, tab_feat], dim=1)
        return self.net(x).squeeze(1)   # squeeze removes the extra dimension

fusion_model = FusionNet(cnn_dim=train_emb.shape[1], tab_dim=tab_dim)
print(f"Fusion network built:")
print(f"  Input  : {train_emb.shape[1] + tab_dim} dims")
print(f"  Hidden : 128 → 64")
print(f"  Output : 1 (fire probability)\n")

# ── STEP 5: CREATE DATALOADERS ─────────────────────────────────────────────────
# Convert numpy arrays to PyTorch tensors
X_train_emb = torch.FloatTensor(train_emb)
X_train_tab = torch.FloatTensor(train_tab)
y_train_t   = torch.FloatTensor(train_img_labels)

X_test_emb  = torch.FloatTensor(test_emb)
X_test_tab  = torch.FloatTensor(test_tab)
y_test_t    = torch.FloatTensor(test_img_labels)

train_ds = TensorDataset(X_train_emb, X_train_tab, y_train_t)
test_ds  = TensorDataset(X_test_emb,  X_test_tab,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

# ── STEP 6: TRAIN FUSION MODEL ────────────────────────────────────────────────
criterion = nn.BCELoss()    # Binary Cross Entropy — standard for binary classification
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5, verbose=True
)

EPOCHS       = 20     # fusion model is simple so it trains fast
best_auc     = 0.0
best_acc     = 0.0

print("=" * 55)
print("Training fusion model...")
print("=" * 55)

for epoch in range(EPOCHS):

    # Train
    fusion_model.train()
    total_loss = 0

    for emb_batch, tab_batch, label_batch in train_loader:
        optimizer.zero_grad()
        preds = fusion_model(emb_batch, tab_batch)
        loss  = criterion(preds, label_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Evaluate
    fusion_model.eval()
    all_preds  = []
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for emb_batch, tab_batch, label_batch in test_loader:
            probs = fusion_model(emb_batch, tab_batch)
            preds = (probs > 0.5).float()
            all_probs.extend(probs.numpy())
            all_preds.extend(preds.numpy())
            all_labels.extend(label_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    scheduler.step(avg_loss)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"Acc: {acc*100:.2f}% | "
          f"AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_acc = acc
        torch.save(fusion_model.state_dict(), FUSION_PATH)
        print(f"           Best model saved (AUC={auc:.4f})")

# ── FINAL RESULTS ─────────────────────────────────────────────────────────────
print(f"\n=== FUSION MODEL FINAL RESULTS ===")
print(f"Best Accuracy : {best_acc*100:.2f}%")
print(f"Best ROC-AUC  : {best_auc:.4f}")

fusion_model.load_state_dict(torch.load(FUSION_PATH, map_location=device))
fusion_model.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for emb_batch, tab_batch, label_batch in test_loader:
        probs = fusion_model(emb_batch, tab_batch)
        preds = (probs > 0.5).float()
        all_probs.extend(probs.numpy())
        all_preds.extend(preds.numpy())
        all_labels.extend(label_batch.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=['No Fire', 'Fire']))

print(f"Fusion model saved to: {FUSION_PATH}")
print("\nNext step: run gradcam.py for visualizations")