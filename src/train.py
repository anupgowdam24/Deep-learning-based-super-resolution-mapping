"""
train.py

Training pipeline for Super-Resolution Land-Cover Mapping using the unified
MultiTileDataset across all five Sentinel-2 tiles.
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MultiTileDataset
from model import SRUNet
from loss import SRLoss


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Output directories
    save_dir = args.experiment_dir if args.experiment_dir else args.checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Datasets
    print("Loading training dataset...")
    train_dataset = MultiTileDataset(
        args.data_dir, split='train',
        patch_size_10m=args.patch_size_10m,
        stride_10m=args.stride_10m,
        oversample_bareland=args.oversample_bareland,
        bareland_threshold=args.bareland_threshold,
        bareland_factor=args.bareland_factor,
        use_indices=args.use_indices
    )

    print("Loading validation dataset...")
    val_dataset = MultiTileDataset(
        args.data_dir, split='val',
        patch_size_10m=args.patch_size_10m,
        stride_10m=args.stride_10m,
        oversample_bareland=False,
        use_indices=args.use_indices
    )

    # Class weights from original non-duplicate training pixels
    class_weights = train_dataset.get_class_weights(mode=args.weight_mode, power=args.weight_power).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Dataset summary: Train={len(train_dataset)} patches, Val={len(val_dataset)} patches")
    print(f"Class weights ({args.weight_mode}): {class_weights.cpu().numpy().round(4)}")

    in_channels = 7 if args.use_indices else 4
    model = SRUNet(in_channels=in_channels, num_classes=5, upscale_factor=3).to(device)

    criterion = SRLoss(
        class_weights=class_weights,
        consistency_lambda=args.consistency_lambda,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        use_dice=args.use_dice
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)

    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, args.model_name)

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_train_loss = 0.0
        train_samples = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss, clf_loss, cons_loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)

        epoch_train_loss /= max(train_samples, 1)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                if args.max_val_batches and batch_idx >= args.max_val_batches:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss, _, _ = criterion(logits, labels)

                epoch_val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)

        epoch_val_loss /= max(val_samples, 1)
        val_losses.append(epoch_val_loss)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] - Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path}")

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='royalblue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', color='crimson')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss Curves: {os.path.basename(best_model_path)}')
    plt.grid(True, alpha=0.3)
    loss_curve_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(loss_curve_path, dpi=120)
    plt.close()

    print(f"Training complete. Best model saved at: {best_model_path}")
    return best_model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--experiment_dir', type=str, default=None, help='Specific training experiment folder')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model_name', type=str, default='best_model.pth')

    # Model & Features
    parser.add_argument('--use_indices', action='store_true', help='Use 7-channel input (RGB+NIR+NDVI+NDWI+BSI)')

    # Data sampling
    parser.add_argument('--patch_size_10m', type=int, default=96)
    parser.add_argument('--stride_10m', type=int, default=96)
    parser.add_argument('--oversample_bareland', action='store_true', default=True)
    parser.add_argument('--no_oversample_bareland', dest='oversample_bareland', action='store_false')
    parser.add_argument('--bareland_threshold', type=float, default=0.05)
    parser.add_argument('--bareland_factor', type=int, default=3)

    # Loss & Weights
    parser.add_argument('--weight_mode', type=str, default='sqrt', choices=['sqrt', 'inverse', 'power', 'none'])
    parser.add_argument('--weight_power', type=float, default=0.5)
    parser.add_argument('--consistency_lambda', type=float, default=0.3)
    parser.add_argument('--use_focal', action='store_true')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--use_dice', action='store_true')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Fast debugging limits
    parser.add_argument('--max_train_batches', type=int, default=None)
    parser.add_argument('--max_val_batches', type=int, default=None)

    args = parser.parse_args()
    train(args)
