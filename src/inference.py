"""
Inference Pipeline: 2D Gaussian Window Blending, TTA, and Instance Extraction
Produces submission-ready disjoint instances and Kaggle-compliant RLE masks.
"""

from typing import List, Tuple, Dict
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def make_gaussian_window(patch_size: int = 512, sigma: float = 128.0) -> np.ndarray:
    """Generates a 2D Gaussian kernel to smoothly blend overlapping inference windows."""
    ax = np.arange(patch_size) - (patch_size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))
    kernel = kernel / np.max(kernel)
    return kernel.astype(np.float32)


class GaussianWindowInferer:
    """Executes smooth sliding-window inference with 2-fold TTA on 2048x2048 solar disks."""
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        patch_size: int = 512,
        stride: int = 384,
        sigma: float = 128.0,
        use_tta: bool = True
    ):
        self.model = model.eval()
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.use_tta = use_tta
        self.window = make_gaussian_window(patch_size, sigma)

    @torch.no_grad()
    def predict_full_disk(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        ps = self.patch_size
        stride = self.stride
        
        prob_accum = np.zeros((h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)
        
        y_steps = list(range(0, h - ps + 1, stride))
        if y_steps[-1] != h - ps:
            y_steps.append(h - ps)
            
        x_steps = list(range(0, w - ps + 1, stride))
        if x_steps[-1] != w - ps:
            x_steps.append(w - ps)
            
        for y in y_steps:
            for x in x_steps:
                patch = img[y:y + ps, x:x + ps].astype(np.float32) / 255.0
                t_patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Forward pass
                logits = self.model(t_patch)
                probs = torch.sigmoid(logits)
                
                # 2-fold TTA (Horizontal Flip)
                if self.use_tta:
                    t_patch_flip = torch.flip(t_patch, dims=[3])
                    logits_flip = self.model(t_patch_flip)
                    probs_flip = torch.flip(torch.sigmoid(logits_flip), dims=[3])
                    probs = 0.5 * (probs + probs_flip)
                    
                pred_patch = probs.squeeze().cpu().numpy()
                prob_accum[y:y + ps, x:x + ps] += pred_patch * self.window
                weight_accum[y:y + ps, x:x + ps] += self.window
                
        # Normalize overlapping blends
        weight_accum[weight_accum == 0] = 1.0
        final_probs = prob_accum / weight_accum
        return final_probs


def extract_instances_from_mask(
    prob_map: np.ndarray,
    prob_threshold: float = 0.45,
    min_area: int = 150,
    max_instances: int = 8
) -> List[Tuple[np.ndarray, float]]:
    """
    Extracts high-confidence disjoint filament instances from continuous probability maps.
    Scores components by Confidence = Mean_Probability * (Area ** 0.20).
    """
    binary_mask = (prob_map > prob_threshold).astype(np.uint8)
    
    # Mild morphological clean to sever single-pixel bridges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    candidates = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
            
        comp_mask = (labels == lbl)
        mean_p = float(prob_map[comp_mask].mean())
        score = mean_p * (area ** 0.20)
        candidates.append((comp_mask, score))
        
    # Sort descending by physical confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:max_instances]


def encode_rle(mask: np.ndarray) -> str:
    """
    Fortran-order binary Run-Length Encoding (RLE) matching Kaggle / MAGFiLO format.
    """
    dots = mask.flatten(order='F')
    runs = np.where(dots[1:] != dots[:-1])[0] + 2
    runs = np.concatenate([[1] if dots[0] else [], runs, [len(dots) + 1] if dots[-1] else []])
    starts = runs[::2]
    lengths = runs[1::2] - starts
    return ' '.join(f"{s} {l}" for s, l in zip(starts, lengths))
