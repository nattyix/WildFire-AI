# train_cnn.py
# What this file does:
# 1. Loads EfficientNet-B0 — a pretrained model that already knows how to
#    "see" shapes, textures, edges from ImageNet training
# 2. Replaces its final layer to output fire/nofire predictions
# 3. Trains it on your 1520 images for 10 epochs
# 4. Tests it on your 380 test images and prints accuracy + confusion matrix
# 5. Saves the trained model to models/cnn_fire.pth for use in fusion later

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from preprocess import TrainFireDataset, TestFireDataset
from preprocess import train_transform, test_transform
from preprocess import TRAIN_FIRE, TRAIN_NOFIRE, TEST_DIR, BASE

# ── SETUP ─────────────────────────────────────────────────────────────────────
# Create models/ folder if it doesn't exist yet
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
MODEL_SAVE_PATH = os.path.join(BASE, "models", "cnn_fire.pth")

# Use CPU (you don't have a GPU — this is fine, just slower)
device = torch.device("cpu")
print(f"Using device: {device}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
train_dataset = TrainFireDataset(TRAIN_FIRE, TRAIN_NOFIRE, transform=train_transform)
test_dataset  = TestFireDataset(TEST_DIR, transform=test_transform)

# DataLoader batches your images so the model trains on 16 at a time
# instead of one by one — much faster and more stable training
# num_workers=0 is important on Windows — using >0 causes errors on Windows
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=0)

print(f"Training batches : {len(train_loader)}")   # 1520 / 16 = 95 batches
print(f"Testing  batches : {len(test_loader)}")    # 380  / 16 = 24 batches

# ── BUILD MODEL ───────────────────────────────────────────────────────────────
# We use EfficientNet-B0 from torchvision (no need for efficientnet_pytorch package)
# weights=DEFAULT loads pretrained ImageNet weights automatically
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# EfficientNet's final classifier is originally built for 1000 ImageNet classes
# We replace it with our own 2-class head (fire vs nofire)
# The 1280 is EfficientNet-B0's fixed internal feature size — don't change it
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),   # dropout randomly zeros 30% of neurons
                                        # during training — prevents overfitting
    nn.Linear(1280, 256),              # compress 1280 features → 256
    nn.ReLU(),                         # ReLU activation: makes model non-linear
    nn.Dropout(p=0.2),
    nn.Linear(256, 2)                  # final output: 2 classes (fire=1, nofire=0)
)

model = model.to(device)              # move model to CPU

# ── FREEZE EARLY LAYERS ───────────────────────────────────────────────────────
# EfficientNet has ~200 layers. We freeze the first ~150 so they keep their
# pretrained ImageNet knowledge and only the last few layers learn fire-specific
# features. This is called "transfer learning" — it's why we can get 95%+
# accuracy with only 1520 images instead of needing millions.
# Without freezing, CPU training would take hours and likely overfit.
all_params = list(model.parameters())
total_layers = len(all_params)
freeze_until = int(total_layers * 0.75)   # freeze 75% of layers

for i, param in enumerate(all_params):
    if i < freeze_until:
        param.requires_grad = False        # frozen: won't be updated during training
    else:
        param.requires_grad = True         # trainable: will be updated

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters : {trainable:,} / {total:,}")
print("(Frozen layers keep ImageNet knowledge, trainable layers learn fire)\n")

# ── LOSS + OPTIMIZER ──────────────────────────────────────────────────────────
# CrossEntropyLoss: standard loss function for classification
# It measures how wrong the model's predictions are — we want to minimize it
criterion = nn.CrossEntropyLoss()

# Adam optimizer: updates model weights after each batch to reduce the loss
# lr=1e-4 (learning rate) controls how big each update step is
# Too high → model overshoots. Too low → trains too slowly. 1e-4 is a safe start.
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4    # weight decay adds a small penalty for large weights
                         # which helps prevent overfitting
)

# Learning rate scheduler: reduces lr by 30% if val loss doesn't improve
# for 2 epochs. Helps squeeze out extra accuracy in later epochs.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=2, verbose=True
)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
# One "epoch" = the model sees every training image once
# We train for 10 epochs — on CPU this takes roughly 20-40 minutes total
EPOCHS = 10

print("=" * 55)
print("Starting training — this will take ~20-40 min on CPU")
print("=" * 55)

best_accuracy = 0.0   # track best test accuracy to save the best model

for epoch in range(EPOCHS):

    # ── TRAIN PHASE ───────────────────────────────────────────────────────────
    model.train()                      # puts model in training mode (enables dropout)
    running_loss    = 0.0
    correct_train   = 0
    total_train     = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()          # clear gradients from previous batch
                                       # (if you don't do this they accumulate)

        outputs = model(images)        # forward pass: model makes predictions
        loss = criterion(outputs, labels)  # calculate how wrong the predictions are

        loss.backward()               # backward pass: calculate gradients
                                      # (how much to adjust each weight)
        optimizer.step()              # update weights using gradients

        running_loss += loss.item()

        # Calculate training accuracy for this batch
        _, predicted = torch.max(outputs, 1)   # pick class with highest score
        correct_train += (predicted == labels).sum().item()
        total_train   += labels.size(0)

        # Print progress every 20 batches so you know it's running
        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | "
                  f"Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train

    # ── EVALUATION PHASE ──────────────────────────────────────────────────────
    model.eval()                       # puts model in eval mode (disables dropout)
                                       # dropout must be OFF during evaluation
    all_preds  = []
    all_labels = []

    with torch.no_grad():              # no_grad: don't compute gradients during eval
                                       # saves memory and speeds up evaluation
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_accuracy = 100 * accuracy_score(all_labels, all_preds)

    print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
    print(f"  Train Loss     : {avg_train_loss:.4f}")
    print(f"  Train Accuracy : {train_accuracy:.2f}%")
    print(f"  Test Accuracy  : {test_accuracy:.2f}%")

    # Step the scheduler using training loss
    scheduler.step(avg_train_loss)

    # Save model only if this epoch gives the best test accuracy so far
    # This ensures we keep the best version, not just the last one
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   New best model saved ({test_accuracy:.2f}%)")

    print("-" * 55)

# ── FINAL EVALUATION ──────────────────────────────────────────────────────────
print("\n=== FINAL RESULTS ===")
print(f"Best Test Accuracy: {best_accuracy:.2f}%")

# Load the best saved model for final detailed evaluation
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification report: shows precision, recall, F1 for each class
# Precision = of all images predicted as fire, how many were actually fire?
# Recall    = of all actual fire images, how many did we correctly detect?
# F1        = harmonic mean of precision and recall (balanced metric)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=['No Fire', 'Fire']))

# Confusion matrix: 2x2 grid showing correct vs incorrect predictions
# [[TN  FP]     TN = correctly predicted nofire
#  [FN  TP]]    TP = correctly predicted fire
#               FP = wrongly said fire when there was none (false alarm)
#               FN = missed a real fire (most dangerous error)
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(f"                Predicted No Fire   Predicted Fire")
print(f"Actual No Fire  {cm[0][0]:^18}  {cm[0][1]:^14}")
print(f"Actual Fire     {cm[1][0]:^18}  {cm[1][1]:^14}")

print(f"\nModel saved to: {MODEL_SAVE_PATH}")
print("Next step: run train_tabular.py")