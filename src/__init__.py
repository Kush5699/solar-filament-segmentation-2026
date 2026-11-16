"""
CS-FilamentNet: Curvilinear Strip U-Net and Multi-Annotator Consensus Fusion
IEEE BigData 2026 Solar Filament Segmentation Challenge (Rank 6 Solution)
"""

from .models import CSFilamentNet
from .dataset import MAGFiLODataset, get_train_val_loaders
from .losses import CompoundFocalTverskyLoss, SoftDiceLoss
from .inference import GaussianWindowInferer, extract_instances_from_mask, encode_rle
from .consensus import MultiAnnotatorConsensusEngine

__all__ = [
    "CSFilamentNet",
    "MAGFiLODataset",
    "get_train_val_loaders",
    "CompoundFocalTverskyLoss",
    "SoftDiceLoss",
    "GaussianWindowInferer",
    "extract_instances_from_mask",
    "encode_rle",
    "MultiAnnotatorConsensusEngine",
]
