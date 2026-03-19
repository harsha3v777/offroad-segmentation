"""
app.py  –  Flask inference server for Offroad Segmentation
===========================================================
Requirements:
    pip install flask torch torchvision pillow numpy

Place next to:
    segmentation_head.pth
    model_config.json

Run:
    python app.py
Then open http://localhost:5000
"""

import io
import json
import os
import base64

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
HEAD_PATH  = os.path.join(BASE_DIR, "segmentation_head.pth")
CFG_PATH   = os.path.join(BASE_DIR, "model_config.json")

# ── Load config ────────────────────────────────────────────────────────────────
with open(CFG_PATH) as f:
    CFG = json.load(f)

N_CLASSES    = CFG["n_classes"]
IMAGE_W      = CFG["image_w"]
IMAGE_H      = CFG["image_h"]
N_EMBEDDING  = CFG["n_embedding"]
CLASS_NAMES  = CFG["class_names"]
CLASS_COLORS = np.array(CFG["class_colors"], dtype=np.uint8)  # (N,3)  RGB

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Model definition (must match training code exactly) ───────────────────────
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ── Load backbone (DINOv2) ─────────────────────────────────────────────────────
BACKBONE_SIZE = CFG["backbone_size"]
_arch_map = {
    "small": "vits14",
    "base":  "dinov2_vitb14_reg",
    "large": "dinov2_vitl14_reg",
    "giant": "dinov2_vitg14_reg",
}
print("Loading DINOv2 backbone …")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.eval().to(DEVICE)
print("Backbone ready.")

# ── Load segmentation head ─────────────────────────────────────────────────────
tokenW = IMAGE_W // 14
tokenH = IMAGE_H // 14

head = SegmentationHeadConvNeXt(
    in_channels=N_EMBEDDING,
    out_channels=N_CLASSES,
    tokenW=tokenW,
    tokenH=tokenH,
)
head.load_state_dict(torch.load(HEAD_PATH, map_location=DEVICE, weights_only=False)["model_state"])
head.eval().to(DEVICE)
print("Segmentation head ready.")

# ── Image transforms ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_H, IMAGE_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Helpers ────────────────────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def colorize_mask(class_mask: np.ndarray) -> Image.Image:
    """class_mask: (H,W) int array → RGB PIL image"""
    rgb = CLASS_COLORS[class_mask]          # (H,W,3)
    return Image.fromarray(rgb, mode="RGB")


def blend(original: Image.Image, mask_rgb: Image.Image, alpha=0.55) -> Image.Image:
    orig_resized = original.resize(mask_rgb.size, Image.BILINEAR).convert("RGB")
    return Image.blend(orig_resized, mask_rgb, alpha)


def class_pixel_stats(class_mask: np.ndarray):
    total = class_mask.size
    stats = []
    for i, name in enumerate(CLASS_NAMES):
        count = int((class_mask == i).sum())
        pct   = round(count / total * 100, 2)
        if pct > 0:
            stats.append({
                "id": i,
                "name": name,
                "color": CLASS_COLORS[i].tolist(),
                "percent": pct,
            })
    stats.sort(key=lambda x: x["percent"], reverse=True)
    return stats


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file  = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    orig_size = image.size        # (W, H)

    # Inference
    inp = transform(image).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    with torch.no_grad():
        feats  = backbone.forward_features(inp)["x_norm_patchtokens"]
        logits = head(feats)
        logits = F.interpolate(logits, size=(IMAGE_H, IMAGE_W),
                               mode="bilinear", align_corners=False)
        pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (H,W)

    # Outputs
    mask_img   = colorize_mask(pred)
    overlay    = blend(image, mask_img)
    stats      = class_pixel_stats(pred)

    return jsonify({
        "original":  pil_to_b64(image.resize((IMAGE_W, IMAGE_H))),
        "mask":      pil_to_b64(mask_img),
        "overlay":   pil_to_b64(overlay),
        "stats":     stats,
    })


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7860)
