"""
Compound Loss Functions for Curvilinear Solar Filament Segmentation
Heavily weights False Negatives (alpha=0.70, beta=0.30) to preserve narrow filament spines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompoundFocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss.
    TI = (TP + eps) / (TP + alpha * FN + beta * FP + eps)
    Loss = (1 - TI) ** gamma
    """
    def __init__(self, alpha: float = 0.70, beta: float = 0.30, gamma: float = 1.33, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        p_flat = probs.view(-1)
        t_flat = targets.view(-1)
        
        tp = (p_flat * t_flat).sum()
        fp = (p_flat * (1.0 - t_flat)).sum()
        fn = ((1.0 - p_flat) * t_flat).sum()
        
        tversky_index = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        focal_tversky = torch.pow((1.0 - tversky_index), self.gamma)
        return focal_tversky


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss for continuous boundary alignment."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_flat = probs.view(-1)
        t_flat = targets.view(-1)
        
        intersection = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice


class CombinedFilamentLoss(nn.Module):
    """
    Compound Loss: 50% Focal Tversky (alpha=0.70, beta=0.30) + 50% Soft Dice.
    """
    def __init__(self, alpha: float = 0.70, beta: float = 0.30, gamma: float = 1.33):
        super().__init__()
        self.tversky = CompoundFocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_tversky = self.tversky(logits, targets)
        l_dice = self.dice(logits, targets)
        return 0.5 * l_tversky + 0.5 * l_dice
