# Pretrained Model Checkpoints

This directory contains the trained neural network model weights for reproducing the IEEE BigData 2026 Solar Filament Segmentation Challenge results.

---

## Model Checkpoint Inventory

| Filename | Architecture | Size | Public PQ | Private PQ | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`best_filament_model.pth`** | **CS-FilamentNet (ResNet-34 + MDSC)** | **81.30 MB** | **0.62** | **0.61** | **Primary Champion Model (Rank 6 Worldwide)** |
| **`baseline_unet_036.pth`** | **Lightweight Standard U-Net** | **7.39 MB** | **0.36** | **0.34** | **Baseline Comparison Model** |

---

## 1. Primary Model: CS-FilamentNet (`best_filament_model.pth`)

- **Backbone**: ResNet-34 adapted to 1-channel solar H-alpha observations.
- **Bottleneck**: Multi-Directional Strip Convolutions (paired 1x15 and 15x1 elongated kernels with dilation rate = 2).
- **Decoder**: Squeeze-and-Excitation (SE) channel attention with bilinear skip-connection fusion.
- **Training Strategy**: Coordinate-guided patch sampling (75% foreground ratio) and Compound Focal Tversky (alpha = 0.70, beta = 0.30) plus Soft Dice loss.
- **Leaderboard Results**: 0.62 Public PQ, 0.61 Private PQ (Rank 6 Worldwide).
- **Loading in PyTorch**:
```python
import torch
from src.models import CSFilamentNet

model = CSFilamentNet(pretrained=False)
model.load_state_dict(torch.load("weights/best_filament_model.pth", map_location="cpu"))
model.eval()
```

---

## 2. Baseline Model: Lightweight U-Net (`baseline_unet_036.pth`)

- **Architecture**: Compact 4-stage convolutional U-Net.
- **Size**: 7.39 MB (ultra-lightweight footprint).
- **Leaderboard Results**: 0.36 Public PQ / 0.34 Private PQ.
- **Purpose**: Provides a fast baseline for benchmarking and comparison against the curvilinear strip convolution architecture.
