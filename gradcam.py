# gradcam.py
# What this file does:
# 1. Loads your trained CNN model
# 2. Picks sample fire and nofire images from your test set
# 3. Generates Grad-CAM heatmaps — colored overlays showing WHERE
#    the model looked when making its decision
# 4. Saves side-by-side comparison images (original vs heatmap)
#    to your outputs/ folder — these go straight into your paper/dashboard

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from preprocess import TEST_DIR, BASE

# ── PATHS ─────────────────────────────────────────────────────────────────────
CNN_PATH   = os.path.join(BASE, "models",  "cnn_fire.pth")
OUTPUT_DIR = os.path.join(BASE, "outputs", "gradcam")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cpu")

# ── LOAD CNN ──────────────────────────────────────────────────────────────────
print("Loading CNN model...")
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, 2)
)
model.load_state_dict(torch.load(CNN_PATH, map_location=device))
model.eval()

# ── GRAD-CAM IMPLEMENTATION ───────────────────────────────────────────────────
# Grad-CAM works by:
# 1. Running a forward pass and recording the feature maps at the last conv layer
# 2. Running a backward pass and recording the gradients at that same layer
# 3. Weighting each feature map by its average gradient
# 4. Summing all weighted feature maps → produces a heatmap
# 5. Overlaying the heatmap on the original image

# We use "hooks" to intercept the layer's output and gradients
# A hook is like a spy that quietly records values as they flow through

class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None   # will store gradients from backward pass
        self.activations = None   # will store feature maps from forward pass

        # EfficientNet's last conv block is model.features[-1]
        # This is the deepest convolutional layer — it has the most
        # semantic understanding of what's in the image
        target_layer = model.features[-1]

        # Register forward hook — fires during forward pass
        # Captures the feature maps (activations) coming OUT of the layer
        target_layer.register_forward_hook(self._save_activation)

        # Register backward hook — fires during backward pass
        # Captures the gradients flowing BACK through the layer
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output shape: [batch, channels, H, W]
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] shape: [batch, channels, H, W]
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx=None):
        """
        img_tensor: preprocessed image tensor [1, 3, 224, 224]
        class_idx : which class to explain (None = predicted class)
        returns   : heatmap as numpy array [224, 224], values 0-1
        """
        # Forward pass
        output = self.model(img_tensor)

        # If no class specified, explain the predicted class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero all existing gradients
        self.model.zero_grad()

        # Backward pass — only for the target class score
        # This tells us: which pixels most increased the fire/nofire score?
        score = output[0, class_idx]
        score.backward()

        # Global average pool the gradients over spatial dimensions
        # Result: one weight per channel — how important was each channel?
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

        # ReLU: only keep positive values (pixels that increased the score)
        # Negative values mean the pixel pushed AGAINST the predicted class
        cam = torch.relu(cam)

        # Resize heatmap to match original image size
        cam = torch.nn.functional.interpolate(
            cam, size=(224, 224), mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().numpy()   # [224, 224]

        # Normalize to 0-1 so we can apply a colormap
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx

# ── IMAGE PREPROCESSING ───────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def denormalize(tensor):
    """
    Reverses the normalization so we can display the original image
    Normalization: x = (x - mean) / std
    Denormalization: x = x * std + mean
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor.squeeze(0) * std + mean
    img  = img.permute(1, 2, 0).numpy()          # [C,H,W] → [H,W,C]
    img  = np.clip(img, 0, 1)
    return img

# ── SELECT SAMPLE IMAGES ──────────────────────────────────────────────────────
# Pick 4 fire and 4 nofire images from the test set to visualize
fire_samples   = []
nofire_samples = []

for fname in sorted(os.listdir(TEST_DIR)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    path = os.path.join(TEST_DIR, fname)
    if fname.startswith('fire_') and len(fire_samples) < 4:
        fire_samples.append((path, fname, 1))
    elif fname.startswith('nofire_') and len(nofire_samples) < 4:
        nofire_samples.append((path, fname, 0))
    if len(fire_samples) == 4 and len(nofire_samples) == 4:
        break

all_samples = fire_samples + nofire_samples
print(f"Selected {len(fire_samples)} fire + {len(nofire_samples)} nofire images\n")

# ── GENERATE GRAD-CAM VISUALIZATIONS ─────────────────────────────────────────
gradcam = GradCAM(model)

label_names = {0: 'No Fire', 1: 'Fire'}
class_colors = {0: 'Blues', 1: 'Reds'}   # fire = red heatmap, nofire = blue

print("Generating Grad-CAM heatmaps...")

for path, fname, true_label in all_samples:

    # Load and preprocess image
    img_pil    = Image.open(path).convert('RGB').resize((224, 224))
    img_tensor = transform(img_pil).unsqueeze(0)   # add batch dim → [1,3,224,224]

    # Generate heatmap
    cam, pred_class = gradcam.generate(img_tensor)

    # Get prediction confidence
    with torch.no_grad():
        output    = model(img_tensor)
        probs     = torch.softmax(output, dim=1)
        confidence = probs[0][pred_class].item() * 100

    # Convert original image to numpy for display
    original_np = denormalize(img_tensor)

    # Apply colormap to heatmap (jet = blue→green→red spectrum)
    heatmap_colored = cm.jet(cam)[:, :, :3]   # drop alpha channel → [H,W,3]

    # Overlay: blend original image with heatmap
    # alpha controls transparency: 0.55 original + 0.45 heatmap
    overlay = 0.55 * original_np + 0.45 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    # ── PLOT: 3 panels side by side ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('#1a1a2e')   # dark background — matches dashboard style

    # Panel 1: Original image
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image', color='white', fontsize=11, pad=8)
    axes[0].axis('off')

    # Panel 2: Grad-CAM heatmap only
    axes[1].imshow(original_np)
    axes[1].imshow(cam, cmap='jet', alpha=0.5)
    axes[1].set_title('Grad-CAM Heatmap', color='white', fontsize=11, pad=8)
    axes[1].axis('off')

    # Panel 3: Full overlay with prediction label
    axes[2].imshow(overlay)
    pred_str = label_names[pred_class]
    true_str = label_names[true_label]
    correct  = pred_class == true_label

    # Color the prediction text: green if correct, red if wrong
    title_color = '#00ff88' if correct else '#ff4444'
    axes[2].set_title(
        f'Predicted: {pred_str} ({confidence:.1f}%)\nActual: {true_str}',
        color=title_color, fontsize=10, pad=8
    )
    axes[2].axis('off')

    # Add colorbar to show heatmap intensity scale
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='jet',
                                norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attention intensity', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.suptitle(
        f'Grad-CAM Analysis — {fname}',
        color='white', fontsize=12, y=1.02
    )
    plt.tight_layout()

    # Save
    save_name = fname.replace('.jpg', '_gradcam.png').replace('.jpeg', '_gradcam.png')
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_name}  [{true_str} → Predicted: {pred_str} {confidence:.1f}%]")

# ── COMBINED SUMMARY FIGURE ───────────────────────────────────────────────────
# Creates one big figure with all 8 images — perfect for your paper/presentation
print("\nGenerating combined summary figure...")

fig, axes = plt.subplots(4, 3, figsize=(14, 18))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('Grad-CAM Visual Explanations — Wildfire Detection Model',
             color='white', fontsize=14, y=1.01)

for row_idx, (path, fname, true_label) in enumerate(all_samples[:4]):
    img_pil    = Image.open(path).convert('RGB').resize((224, 224))
    img_tensor = transform(img_pil).unsqueeze(0)
    cam, pred_class = gradcam.generate(img_tensor)

    with torch.no_grad():
        output     = model(img_tensor)
        probs      = torch.softmax(output, dim=1)
        confidence = probs[0][pred_class].item() * 100

    original_np     = denormalize(img_tensor)
    heatmap_colored = cm.jet(cam)[:, :, :3]
    overlay         = np.clip(0.55 * original_np + 0.45 * heatmap_colored, 0, 1)

    axes[row_idx][0].imshow(original_np)
    axes[row_idx][0].set_title(
        f'{fname[:20]}', color='white', fontsize=8)
    axes[row_idx][0].axis('off')

    axes[row_idx][1].imshow(original_np)
    axes[row_idx][1].imshow(cam, cmap='jet', alpha=0.5)
    axes[row_idx][1].set_title('Grad-CAM', color='white', fontsize=8)
    axes[row_idx][1].axis('off')

    correct = pred_class == true_label
    color   = '#00ff88' if correct else '#ff4444'
    axes[row_idx][2].imshow(overlay)
    axes[row_idx][2].set_title(
        f'Pred: {label_names[pred_class]} {confidence:.0f}%',
        color=color, fontsize=8)
    axes[row_idx][2].axis('off')

for ax_row in axes:
    for ax in ax_row:
        ax.set_facecolor('#1a1a2e')

plt.tight_layout()
summary_path = os.path.join(OUTPUT_DIR, "gradcam_summary.png")
plt.savefig(summary_path, dpi=120, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f"Summary figure saved: {summary_path}")

print(f"\n All Grad-CAM outputs saved to: {OUTPUT_DIR}")
print("Next step: run app.py (Streamlit dashboard)")