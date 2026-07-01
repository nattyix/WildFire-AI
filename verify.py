# verify.py
# Tests your saved model on 4 images you manually downloaded from Google
# These are completely outside the training dataset — a true real-world test

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

BASE        = r"C:\Users\natal\OneDrive\Desktop\Wildfire Pred"
MODEL_PATH  = os.path.join(BASE, "models", "cnn_fire.pth")
VERIFY_DIR  = os.path.join(BASE, "verify_images")

# Images you saved manually — label is in the filename prefix
# fire1.jpg, fire2.jpg → actual fire (label 1)
# nofire1.jpg, nofire2.jpg → no fire (label 0)
TEST_IMAGES = [
    ("fire1.jpg",   1, "Google fire image 1"),
    ("fire2.jpg",   1, "Google fire image 2"),
    ("nofire1.jpg", 0, "Google forest image 1"),
    ("nofire2.jpg", 0, "Google forest image 2"),
]

# Rebuild exact same model architecture as train_cnn.py
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, 2)
)

# Load the saved weights
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()   # eval mode: disables dropout for consistent predictions

# Same transform as test_transform in preprocess.py
# MUST match exactly — if you change this, predictions will be garbage
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("=" * 50)
print("Testing on unseen Google Images")
print("=" * 50 + "\n")

correct = 0
total   = 0

for fname, true_label, description in TEST_IMAGES:
    path = os.path.join(VERIFY_DIR, fname)

    # Check file exists before trying to open it
    if not os.path.exists(path):
        print(f"  File not found: {fname} — skipping\n")
        continue

    img    = Image.open(path).convert('RGB')
    tensor = transform(img).unsqueeze(0)   # unsqueeze adds batch dimension
                                            # model expects [batch, channels, H, W]
                                            # single image becomes [1, 3, 224, 224]

    with torch.no_grad():
        output     = model(tensor)
        probs      = torch.softmax(output, dim=1)   # raw scores → probabilities
        pred       = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100

    pred_str = "FIRE"    if pred == 1        else "NO FIRE"
    true_str = "FIRE"    if true_label == 1  else "NO FIRE"
    result   = "CORRECT" if pred == true_label else "WRONG"

    print(f"{description} ({fname})")
    print(f"  Predicted : {pred_str} — {confidence:.1f}% confidence")
    print(f"  Actual    : {true_str}  {result}\n")

    if pred == true_label:
        correct += 1
    total += 1

if total > 0:
    print("=" * 50)
    print(f"Real-world accuracy: {correct}/{total} = {100*correct/total:.0f}%")
    print("=" * 50)

    if correct == total:
        print("\nModel generalizes well — 100% was real!")
    elif correct / total >= 0.75:
        print("\nMild overfitting — still good, minor fixes needed")
    else:
        print("\n Overfitting confirmed — model needs adjustments")