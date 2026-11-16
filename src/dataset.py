"""
MAGFiLO Dataset and Coordinate-Guided Patch Sampler
Handles extreme 99.5% class imbalance via foreground-biased sampling.
"""

import os
import json
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class MAGFiLODataset(Dataset):
    """
    MAGFiLO Solar Filament Dataset with Coordinate-Guided Patch Extraction.
    
    Parameters:
        images_dir: Directory containing full-disk H-alpha images.
        annotations_json: Path to MAGFiLO COCO-format JSON annotations file.
        patch_size: Size of square patches (default: 512).
        fg_prob: Probability of sampling a patch centered on a foreground filament (default: 0.75).
        jitter: Maximum spatial pixel jitter added to foreground patch centers (default: 40).
        is_train: Whether training augmentations should be applied.
    """
    def __init__(
        self,
        images_dir: str,
        annotations_json: Optional[str] = None,
        preloaded_data: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
        patch_size: int = 512,
        fg_prob: float = 0.75,
        jitter: int = 40,
        is_train: bool = True,
        samples_per_epoch: int = 1500
    ):
        self.patch_size = patch_size
        self.fg_prob = fg_prob
        self.jitter = jitter
        self.is_train = is_train
        self.samples_per_epoch = samples_per_epoch
        
        if preloaded_data is not None:
            self.samples = preloaded_data
        elif annotations_json is not None:
            self.samples = self._load_from_json(images_dir, annotations_json)
        else:
            self.samples = []

    def _load_from_json(self, images_dir: str, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        images_map = {img['id']: img for img in data['images']}
        annos_by_image = {}
        for ann in data['annotations']:
            annos_by_image.setdefault(ann['image_id'], []).append(ann)
            
        loaded = []
        for img_id, img_info in images_map.items():
            img_path = os.path.join(images_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            h, w = img.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            for ann in annos_by_image.get(img_id, []):
                for seg in ann.get('segmentation', []):
                    pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], 1)
                    
            fg_coords = np.argwhere(mask > 0)
            loaded.append((img, mask, fg_coords))
            
        return loaded

    def __len__(self) -> int:
        return self.samples_per_epoch if self.is_train else len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx = random.randint(0, len(self.samples) - 1) if self.is_train else idx % len(self.samples)
        img, mask, fg_coords = self.samples[sample_idx]
        h, w = img.shape
        ps = self.patch_size
        
        # Coordinate-guided patch center selection
        use_fg = (random.random() < self.fg_prob) and (len(fg_coords) > 0)
        if use_fg:
            center_y, center_x = fg_coords[random.randint(0, len(fg_coords) - 1)]
            center_y += random.randint(-self.jitter, self.jitter)
            center_x += random.randint(-self.jitter, self.jitter)
            top = max(0, min(h - ps, center_y - ps // 2))
            left = max(0, min(w - ps, center_x - ps // 2))
        else:
            top = random.randint(0, max(0, h - ps))
            left = random.randint(0, max(0, w - ps))
            
        patch_img = img[top:top + ps, left:left + ps].copy()
        patch_mask = mask[top:top + ps, left:left + ps].copy()
        
        # Training augmentations
        if self.is_train:
            if random.random() > 0.5:
                patch_img = np.fliplr(patch_img)
                patch_mask = np.fliplr(patch_mask)
            if random.random() > 0.5:
                patch_img = np.flipud(patch_img)
                patch_mask = np.flipud(patch_mask)
            k = random.randint(0, 3)
            if k > 0:
                patch_img = np.rot90(patch_img, k)
                patch_mask = np.rot90(patch_mask, k)
                
        # Normalize to [0, 1]
        img_tensor = torch.from_numpy(patch_img).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(patch_mask).float().unsqueeze(0)
        
        return img_tensor, mask_tensor


def get_train_val_loaders(
    images_dir: str,
    annotations_json: str,
    batch_size: int = 8,
    num_workers: int = 2,
    val_split: float = 0.15
) -> Tuple[DataLoader, DataLoader]:
    """Prepares stratified train and validation PyTorch DataLoaders."""
    full_ds = MAGFiLODataset(images_dir, annotations_json, is_train=True)
    n_total = len(full_ds.samples)
    n_val = max(1, int(n_total * val_split))
    
    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)
    
    val_samples = [full_ds.samples[i] for i in indices[:n_val]]
    train_samples = [full_ds.samples[i] for i in indices[n_val:]]
    
    train_ds = MAGFiLODataset(
        images_dir="",
        preloaded_data=train_samples,
        is_train=True,
        samples_per_epoch=len(train_samples) * 10
    )
    val_ds = MAGFiLODataset(
        images_dir="",
        preloaded_data=val_samples,
        is_train=False,
        samples_per_epoch=len(val_samples)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
