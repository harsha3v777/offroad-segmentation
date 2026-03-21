
Live Demo - https://huggingface.co/spaces/harsha3777/offroad-segmentation

# OffRoad Vision — Terrain Segmentation

> Semantic segmentation of off-road terrain scenes using a frozen DINOv2 backbone and a custom ConvNeXt-style head. Built for the **Duality AI Track — Hack Energy 2.0** hackathon. Deployed live as a web application on Hugging Face Spaces.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-orange?style=flat-square)](https://huggingface.co/spaces/harsha3777/offroad-segmentation)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red?style=flat-square)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green?style=flat-square)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

---

## Table of Contents

- [Demo](#demo)
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)
- [Team](#team)

---

## Demo

Upload any off-road scene image and the model instantly segments every terrain type pixel by pixel.

| Original | Segmentation Mask | Overlay |
|----------|-------------------|---------|
| Raw input image | Color-coded terrain classes | Mask blended with original |

**Live app:** https://huggingface.co/spaces/harsha3777/offroad-segmentation

---

## Overview

Off-road environments are highly unstructured — a mix of trees, rocks, dry grass, bushes, logs, and sky all blended together. Semantic segmentation assigns a terrain class to every single pixel in the image, which is critical for:

- Autonomous off-road vehicle navigation
- Agricultural and mining robots
- Drone landing zone detection
- Environmental monitoring and land cover mapping
- Search and rescue robotics

This project trains a segmentation model on the Duality AI off-road dataset and deploys it as a real-time web application where users can upload images and instantly visualize terrain segmentation results.

---

## Dataset

Provided by **Duality AI** as part of Hack Energy 2.0.

| Split | Images |
|-------|--------|
| Train | 2,857 |
| Validation | 317 |
| Test | 1,002 |

### Terrain Classes (11 total)

| ID | Class | Color |
|----|-------|-------|
| 0 | Background | Black |
| 1 | Trees | Forest Green |
| 2 | Lush Bushes | Bright Green |
| 3 | Dry Grass | Tan |
| 4 | Dry Bushes | Brown |
| 5 | Ground Clutter | Olive |
| 6 | Flowers | Pink |
| 7 | Logs | Dark Brown |
| 8 | Rocks | Grey |
| 9 | Landscape | Olive Drab |
| 10 | Sky | Sky Blue |

> **Bug fix:** The original Duality-provided script was missing class 600 (Flowers). We identified and corrected this, bringing the total from 10 to 11 classes.

---

## Model Architecture

The model is a two-stage pipeline:

```
Input Image (476×266)
       │
       ▼
┌─────────────────────┐
│   DINOv2 Backbone   │  ← Frozen, not trained
│   (dinov2_vits14)   │
│   384-dim patches   │
└─────────┬───────────┘
          │  patch tokens (B, N, 384)
          ▼
┌─────────────────────┐
│  ConvNeXt Seg Head  │  ← Trained (2.4M params)
│  Conv7×7 → GELU     │
│  DWConv7×7 → GELU   │
│  Conv1×1 → 11cls    │
└─────────┬───────────┘
          │  logits (B, 11, H/14, W/14)
          ▼
   Bilinear Upsample
          │
          ▼
  Predicted Mask (476×266)
```

### DINOv2 Backbone

- Model: `dinov2_vits14` (ViT-Small with 14×14 patches)
- Pretrained by Facebook Research using self-supervised learning
- Output: 384-dimensional patch token embeddings
- Status: **Frozen** — weights are not updated during training

### ConvNeXt Segmentation Head

- Stem: `Conv2d(384 → 128, kernel=7, padding=3)` + GELU
- Block: `DepthwiseConv2d(128→128, kernel=7)` + GELU + `Conv2d(128→128, kernel=1)` + GELU
- Classifier: `Conv2d(128 → 11, kernel=1)`
- Trainable parameters: **2,432,907**

---

## Training

### Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 25 |
| Batch size | 2 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Scheduler | Cosine Annealing |
| Loss | Weighted Cross Entropy |
| Hardware | Google Colab Tesla T4 GPU |
| Image size | 476 × 266 |

### Class Weights (Weighted CE Loss)

Rare classes were given higher weights to prevent the model from ignoring them:

```python
class_weights = [0.5, 1.0, 1.5, 2.0, 2.0, 3.0, 5.0, 5.0, 2.0, 0.5, 0.5]
#                bg  tree  bush  dry  dryb  gnd  flwr  log  rock land  sky
```

---

## Results

| Metric | Score |
|--------|-------|
| Best Validation mIoU | **0.3125** |
| Validation Pixel Accuracy | **67.19%** |
| Best Epoch | 21 / 25 |
| Test mIoU (3 sample images) | 0.2248 avg |

---

## Project Structure

```
offroad-segmentation/
├── app.py                  # Flask inference server
├── index.html              # Frontend web interface
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container for deployment
├── model_config.json       # Model config (classes, colors, dims)
├── segmentation_head.pth   # Trained model weights
└── README.md
```

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- Git

### Clone the repository

```bash
git clone https://github.com/harsha3777/offroad-segmentation.git
cd offroad-segmentation
```

### Install dependencies

```bash
pip install flask torch torchvision pillow numpy gunicorn
```

### Download model weights

Download `segmentation_head.pth` and `model_config.json` from the releases section or your Google Drive and place them in the project root.

---

## Running Locally

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

Upload any off-road image to get instant terrain segmentation with:
- Color-coded segmentation mask
- Overlay of mask on original image
- Pixel distribution chart for all terrain classes

---

## Training From Scratch

To retrain the model on the Duality AI dataset, open `Duality.ipynb` in Google Colab:

1. Mount Google Drive
2. Unzip the Duality AI dataset into `DualityHackathon/dataset/`
3. Run all cells sequentially
4. Best model is saved automatically to `DualityHackathon/runs/best_model.pth`

The notebook covers:
- Dataset loading and preprocessing
- DINOv2 backbone loading
- ConvNeXt head definition
- Training loop with validation
- Metric tracking (mIoU, pixel accuracy)
- Sample prediction visualization

---

## Deployment

The app is containerized with Docker and deployed on Hugging Face Spaces.

### Build and run with Docker

```bash
docker build -t offroad-segmentation .
docker run -p 7860:7860 offroad-segmentation
```

### Hugging Face Spaces

The app runs automatically on push to the Hugging Face Space repository. The Dockerfile handles all dependencies and starts the gunicorn server on port 7860.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backbone | DINOv2 (Facebook Research) |
| Segmentation head | Custom ConvNeXt (PyTorch) |
| Training | Google Colab T4 GPU |
| Backend | Flask + Gunicorn |
| Frontend | HTML / CSS / JavaScript |
| Containerization | Docker |
| Hosting | Hugging Face Spaces |
| Version control | GitHub |

---

## Future Work

- **Larger backbone** — Upgrade from DINOv2-small to DINOv2-base or large for better feature quality
- **Advanced decoder** — Replace the ConvNeXt head with SegFormer or DeepLabV3+ for sharper boundaries
- **Data augmentation** — Add random flips, color jitter, and random crops during training
- **Backbone fine-tuning** — Unfreeze and fine-tune DINOv2 on terrain data with a very small learning rate
- **Full test set evaluation** — Run inference on all 1,002 test images and submit to the hackathon leaderboard
- **Real-time video** — Extend the app to process live video streams from drone or vehicle cameras
- **Mobile deployment** — Quantize and export to ONNX for edge device inference

---

## Team

**Team Name:** OffRoad Vision
**Hackathon:** Hack Energy 2.0 — Duality AI Track

| Name | Role |
|------|------|
| Harsha Vennapusa | Model training, deployment, full-stack development |

---

## Acknowledgements

- [Duality AI](https://duality.ai) for the off-road segmentation dataset and hackathon
- [Facebook Research](https://github.com/facebookresearch/dinov2) for the DINOv2 backbone
- [Hugging Face](https://huggingface.co) for free model hosting

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.live web application.
Thank you for watching. The live demo is available at our Hugging Face Space, and all code is on GitHub. Feel free to try it out!"
