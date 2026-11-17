"""
CLI Inference Script for CS-FilamentNet
Generates Kaggle-compliant RLE submission CSV from trained weights.
Usage:
    python scripts/predict.py --weights checkpoints/best_cs_filamentnet.pth --test_images /path/to/test --output submission.csv
"""

import argparse
import os
import cv2
import torch
from tqdm import tqdm

from src.models import CSFilamentNet
from src.inference import GaussianWindowInferer
from src.consensus import MultiAnnotatorConsensusEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Run Full-Disk Inference on Test Observations")
    parser.add_argument("--weights", type=str, default="weights/best_filament_model.pth", help="Path to trained CS-FilamentNet checkpoint (.pth)")
    parser.add_argument("--test_images", type=str, required=True, help="Directory containing full-disk test images")
    parser.add_argument("--output", type=str, default="submission.csv", help="Path to output submission CSV")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size for sliding window")
    parser.add_argument("--stride", type=int, default=384, help="Stride for sliding window")
    parser.add_argument("--threshold", type=float, default=0.45, help="Filament probability threshold")
    parser.add_argument("--max_instances", type=int, default=8, help="Max instances per solar disk")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing inference engine on device: {device}")
    
    # Load model
    model = CSFilamentNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"[+] Loaded weights successfully from: {args.weights}")
    
    inferer = GaussianWindowInferer(
        model=model,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        use_tta=True
    )
    consensus_engine = MultiAnnotatorConsensusEngine(
        prob_threshold=args.threshold,
        max_instances=args.max_instances
    )
    
    test_files = sorted([f for f in os.listdir(args.test_images) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.fits'))])
    print(f"[*] Found {len(test_files)} test observations to evaluate.")
    
    all_records = []
    for fname in tqdm(test_files, desc="Inference"):
        img_path = os.path.join(args.test_images, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        stem = os.path.splitext(fname)[0]
        prob_map = inferer.predict_full_disk(img)
        disk_records = consensus_engine.process_disk_predictions(stem, prob_map)
        all_records.extend(disk_records)
        
    df_sub = consensus_engine.build_submission_dataframe(all_records)
    df_sub.to_csv(args.output, index=False)
    print(f"\n[+] Inference Complete! Saved {len(df_sub)} instances across {len(test_files)} observations to: {args.output}")


if __name__ == "__main__":
    main()
