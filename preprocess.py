# preprocess.py
# What this file does:
# 1. Loads your Training images and applies augmentation (flipping, rotating etc.)
#    to artificially increase variety so the model doesn't overfit
# 2. Loads your Testing images correctly (the Testing folder has mixed filenames
#    instead of subfolders, so we detect fire vs nofire from the filename itself)
# 3. Loads your tabular CSV and cleans + scales it
# 4. Prints a summary so you can confirm everything loaded correctly

import os
from PIL import Image                          # opens image files
from torch.utils.data import Dataset           # base class for PyTorch datasets
from torchvision import transforms             # image transformations
import pandas as pd                            # for loading the CSV
from sklearn.preprocessing import StandardScaler  # scales numbers to same range
import torch

# ── PATHS ────────────────────────────────────────────────────────────────────
# These point to your exact folder structure — no need to change anything
_LOCAL = r"C:\Users\natal\OneDrive\Desktop\WildFire AI\Wildfire Pred"
BASE   = _LOCAL if os.path.exists(_LOCAL) else os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "data", "tabular", "forestfires.csv")
TRAIN_FIRE = os.path.join(BASE, "data", "images", "Forest Fire Dataset", "Training", "fire")
TRAIN_NOFIRE = os.path.join(BASE, "data", "images", "Forest Fire Dataset", "Training", "nofire")
TEST_DIR   = os.path.join(BASE, "data", "images", "Forest Fire Dataset", "Testing")
CSV_PATH   = os.path.join(BASE, "data", "tabular", "forestfires.csv")

# ── IMAGE TRANSFORMS ─────────────────────────────────────────────────────────
# Training transform: includes augmentation (random flips/rotations)
# Why augmentation? With ~1500 training images, the model needs to see
# the same image in different orientations so it learns the concept of
# "fire" not just the exact pixels it trained on
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),              # EfficientNet needs 224x224
    transforms.RandomHorizontalFlip(p=0.5),    # randomly mirror 50% of images
    transforms.RandomVerticalFlip(p=0.2),      # randomly flip upside down 20%
    transforms.RandomRotation(degrees=15),     # rotate randomly up to 15 degrees
    transforms.ColorJitter(                    # randomly tweak brightness/contrast
        brightness=0.3,                        # simulates different lighting conditions
        contrast=0.3,
        saturation=0.3
    ),
    transforms.ToTensor(),                     # converts image pixels 0-255 → 0.0-1.0
    transforms.Normalize(                      # subtract ImageNet mean, divide by std
        mean=[0.485, 0.456, 0.406],            # these are standard values used by
        std=[0.229, 0.224, 0.225]              # all pretrained torchvision models
    )
])

# Testing transform: NO augmentation — we want consistent, clean evaluation
# We never augment test data because we want to measure real performance
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── TRAINING DATASET ─────────────────────────────────────────────────────────
# Training folder has two subfolders: fire/ and nofire/
# We assign label 1 = fire, label 0 = nofire
class TrainFireDataset(Dataset):
    def __init__(self, fire_dir, nofire_dir, transform=None):
        self.transform = transform
        self.samples = []

        # Load all fire images → label 1
        for fname in os.listdir(fire_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append((os.path.join(fire_dir, fname), 1))

        # Load all nofire images → label 0
        for fname in os.listdir(nofire_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append((os.path.join(nofire_dir, fname), 0))

    def __len__(self):
        # PyTorch calls this to know how many samples exist
        return len(self.samples)

    def __getitem__(self, idx):
        # PyTorch calls this to get one sample by index
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')  # convert ensures 3 channels (no alpha)
        if self.transform:
            img = self.transform(img)
        return img, label

# ── TESTING DATASET ───────────────────────────────────────────────────────────
# Testing folder has ALL images in one flat folder
# We detect the label from the filename: fire_XXXX.jpg → 1, nofire_XXXX.jpg → 0
class TestFireDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.transform = transform
        self.samples = []

        for fname in os.listdir(test_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(test_dir, fname)
                # filename starts with 'fire_' → label 1, 'nofire_' → label 0
                label = 1 if fname.startswith('fire_') else 0
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ── TABULAR DATA ──────────────────────────────────────────────────────────────
# The UCI Forest Fires CSV has weather + vegetation features
# We clean it, convert categorical columns, and scale all numbers to 0-1 range
def load_tabular():
    df = pd.read_csv(CSV_PATH)

    print("\n--- Tabular data preview ---")
    print(df.head())                           # shows first 5 rows
    print(f"\nShape: {df.shape}")              # (rows, columns)
    print(f"Columns: {list(df.columns)}")

    # The 'month' and 'day' columns are text (jan, feb... / mon, tue...)
    # We convert them to numbers using one-hot encoding
    # Why? ML models can't work with raw text like "jan" — they need numbers
    df = pd.get_dummies(df, columns=['month', 'day'])

    # 'area' is the target — how many hectares burned
    # We convert it to binary: 0 = no fire (area == 0), 1 = fire (area > 0)
    # This makes it a classification problem matching our CNN labels
    df['label'] = (df['area'] > 0).astype(int)
    df = df.drop(columns=['area'])             # remove original area column

    # Separate features (X) from labels (y)
    X = df.drop(columns=['label'])
    y = df['label']

    # Scale all feature numbers to have mean=0, std=1
    # Why? Features like temperature (25°C) and wind (5 km/h) are on very
    # different scales. Scaling puts them all on equal footing so no single
    # feature dominates just because of its unit size
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nTabular features after encoding: {X_scaled.shape[1]} columns")
    print(f"Fire cases: {y.sum()} | No-fire cases: {(y==0).sum()}")

    return X_scaled, y.values, scaler, list(X.columns)

# ── MAIN: run this file to verify everything loads correctly ──────────────────
if __name__ == "__main__":

    # Test image datasets
    train_dataset = TrainFireDataset(TRAIN_FIRE, TRAIN_NOFIRE, transform=train_transform)
    test_dataset  = TestFireDataset(TEST_DIR, transform=test_transform)

    # Count fire vs nofire in training set
    train_labels = [s[1] for s in train_dataset.samples]
    train_fire   = sum(train_labels)
    train_nofire = len(train_labels) - train_fire

    # Count fire vs nofire in test set
    test_labels  = [s[1] for s in test_dataset.samples]
    test_fire    = sum(test_labels)
    test_nofire  = len(test_labels) - test_fire

    print("=== IMAGE DATASET ===")
    print(f"Training samples : {len(train_dataset)}  (fire={train_fire}, nofire={train_nofire})")
    print(f"Testing  samples : {len(test_dataset)}   (fire={test_fire},  nofire={test_nofire})")

    # Test that one image actually loads without errors
    img, label = train_dataset[0]
    print(f"\nSample image tensor shape : {img.shape}")   # should be [3, 224, 224]
    print(f"Sample label              : {label}")          # should be 0 or 1

    # Test tabular data
    print("\n=== TABULAR DATASET ===")
    X, y, scaler, cols = load_tabular()
    print(f"\nFinal X shape : {X.shape}")
    print(f"Final y shape : {y.shape}")

    print("\n All data loaded successfully. Ready for training.")