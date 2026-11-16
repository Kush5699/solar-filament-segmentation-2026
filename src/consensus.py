"""
Multi-Annotator Consensus Fusion Engine
Resolves annotator ambiguity in the MAGFiLO benchmark and generates final submission records.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from .inference import encode_rle, extract_instances_from_mask


class MultiAnnotatorConsensusEngine:
    """
    Consensus fusion engine combining neural predictions with multi-annotator priors.
    Ensures 100% disjoint instances and valid Panoptic Quality compliance.
    """
    def __init__(self, prob_threshold: float = 0.45, max_instances: int = 8):
        self.prob_threshold = prob_threshold
        self.max_instances = max_instances

    def process_disk_predictions(
        self,
        image_id: str,
        prob_map: np.ndarray,
        consensus_prior: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Blends raw model probabilities with multi-annotator consensus priors.
        Allocates boundary pixels greedily to guarantee disjoint instance assignments.
        """
        if consensus_prior is not None:
            combined_probs = 0.65 * prob_map + 0.35 * consensus_prior
        else:
            combined_probs = prob_map
            
        instances = extract_instances_from_mask(
            combined_probs,
            prob_threshold=self.prob_threshold,
            max_instances=self.max_instances
        )
        
        # Enforce strict pixel disjointness
        h, w = prob_map.shape
        allocated = np.zeros((h, w), dtype=bool)
        records = []
        
        for inst_id, (mask, score) in enumerate(instances, 1):
            disjoint_mask = mask & (~allocated)
            if disjoint_mask.sum() < 50:
                continue
                
            allocated |= disjoint_mask
            rle_str = encode_rle(disjoint_mask)
            
            records.append({
                "image_id": image_id,
                "filament_id": f"{image_id}_{inst_id:03d}",
                "segmentation_rle": rle_str,
                "confidence": round(float(score), 4)
            })
            
        return records

    def build_submission_dataframe(self, all_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts raw prediction records to Kaggle submission DataFrame."""
        df = pd.DataFrame(all_records)
        if not df.empty:
            df = df[["image_id", "filament_id", "segmentation_rle"]]
        else:
            df = pd.DataFrame(columns=["image_id", "filament_id", "segmentation_rle"])
        return df
