"""
CLI Training Script for CS-FilamentNet
Usage:
    python scripts/train.py --images_dir /path/to/images --annotations /path/to/train.json --epochs 10 --batch_size 8
"""

import argparse
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models import CSFilamentNet
from src.dataset import get_train_val_loaders
from src.losses import CombinedFilamentLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train CS-FilamentNet on MAGFiLO Solar Observations")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing H-alpha training images")
    parser.add_argument("--annotations", type=str, required=True, help="Path to MAGFiLO COCO JSON annotations")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Path to save trained weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training CS-FilamentNet on device: {device}")
    
    # Data loaders
    train_loader, val_loader = get_train_val_loaders(
        images_dir=args.images_dir,
        annotations_json=args.annotations,
        batch_size=args.batch_size
    )
    
    # Model, Loss, Optimizer
    model = CSFilamentNet(pretrained=True).to(device)
    criterion = CombinedFilamentLoss(alpha=0.70, beta=0.30, gamma=1.33).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.output_dir, "best_cs_filamentnet.pth")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= max(1, len(train_loader))
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
        val_loss /= max(1, len(val_loader))
        print(f"[-] Epoch {epoch:02d} Complete | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"    -> Checkpoint saved! New best val loss: {best_val_loss:.4f} -> {best_ckpt_path}")
            
    print(f"\n[+] Training Complete! Best model exported to: {best_ckpt_path}")


if __name__ == "__main__":
    main()
