"""
evaluate.py

Evaluation module for the Super-Resolution Land-Cover Mapping model on the
held-out test split.

Calculates:
  - Per-class Accuracy (Recall), Precision, F1 Score, and IoU for all 5 classes:
      0: Water
      1: Built-up
      2: Vegetation
      3: Cropland
      4: Bare Land
  - Average of the 5 per-class accuracy (recall) values
  - Physical consistency Mean Absolute Error (MAE) at 30m resolution
  - Saves a Markdown report and optional patch visual comparison
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt

from dataset import MultiTileDataset
from model import SRUNet

CLASS_NAMES = ['Water', 'Built-up', 'Vegetation', 'Cropland', 'Bare Land']


def evaluate_model(model, test_loader, device):
    """
    Runs model evaluation on test DataLoader and returns comprehensive metrics.
    """
    model.eval()

    all_preds_10m = []
    all_labels_10m = []
    all_preds_30m_comp = []
    all_labels_30m_comp = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # 30m composition agreement
            pred_30m_comp = F.avg_pool2d(probs, kernel_size=3, stride=3)
            labels_one_hot = F.one_hot(labels, num_classes=5).permute(0, 3, 1, 2).float()
            true_30m_comp = F.avg_pool2d(labels_one_hot, kernel_size=3, stride=3)

            all_preds_10m.append(preds.cpu().numpy())
            all_labels_10m.append(labels.cpu().numpy())
            all_preds_30m_comp.append(pred_30m_comp.cpu().numpy())
            all_labels_30m_comp.append(true_30m_comp.cpu().numpy())

    flat_preds = np.concatenate(all_preds_10m).flatten()
    flat_labels = np.concatenate(all_labels_10m).flatten()

    # Per-class metrics
    labels_list = [0, 1, 2, 3, 4]
    recalls = recall_score(flat_labels, flat_preds, average=None, labels=labels_list, zero_division=0)
    precisions = precision_score(flat_labels, flat_preds, average=None, labels=labels_list, zero_division=0)
    f1s = f1_score(flat_labels, flat_preds, average=None, labels=labels_list, zero_division=0)
    ious = jaccard_score(flat_labels, flat_preds, average=None, labels=labels_list, zero_division=0)

    # Average of 5 per-class accuracies (recalls)
    avg_accuracy = float(np.mean(recalls))

    # Physical Consistency MAE
    preds_comp_flat = np.concatenate(all_preds_30m_comp).reshape(-1, 5)
    labels_comp_flat = np.concatenate(all_labels_30m_comp).reshape(-1, 5)
    consistency_mae = float(np.mean(np.abs(preds_comp_flat - labels_comp_flat)))

    metrics = {
        'classes': CLASS_NAMES,
        'accuracy_recall': [float(r) for r in recalls],
        'precision': [float(p) for p in precisions],
        'f1': [float(f) for f in f1s],
        'iou': [float(i) for i in ious],
        'average_accuracy': avg_accuracy,
        'consistency_mae': consistency_mae,
        'per_class': {}
    }

    for idx, cname in enumerate(CLASS_NAMES):
        metrics['per_class'][cname] = {
            'recall': float(recalls[idx]),
            'precision': float(precisions[idx]),
            'f1': float(f1s[idx]),
            'iou': float(ious[idx])
        }

    return metrics


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Evaluation device: {device}")

    # Load state dict
    ckpt_path = os.path.join(args.checkpoint_dir, args.model_name)
    if not os.path.exists(ckpt_path):
        ckpt_path = args.model_name  # Direct path
    state_dict = torch.load(ckpt_path, map_location=device)

    # Detect in_channels
    in_channels = state_dict['inc.conv.0.weight'].shape[1] if 'inc.conv.0.weight' in state_dict else state_dict['inc.double_conv.0.weight'].shape[1]
    use_indices = (in_channels == 7)

    # Load test dataset
    test_dataset = MultiTileDataset(
        args.data_dir, split='test',
        patch_size_10m=args.patch_size_10m,
        stride_10m=args.stride_10m,
        use_indices=use_indices
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build model
    model = SRUNet(in_channels=in_channels, num_classes=5, upscale_factor=3).to(device)
    model.load_state_dict(state_dict)

    # Evaluate
    metrics = evaluate_model(model, test_loader, device)

    # Output report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"metrics_report_{os.path.basename(args.model_name).replace('.pth', '')}.md")
    with open(report_path, 'w') as f:
        f.write(f"# Evaluation Report: {os.path.basename(args.model_name)}\n\n")
        f.write(f"**Test Split**: {len(test_dataset)} patches across 5 tiles\n\n")
        f.write("## Per-Class Metrics (10m Resolution)\n\n")
        f.write("| Class | Accuracy (Recall) | Precision | F1 Score | IoU |\n")
        f.write("|---|---|---|---|---|\n")
        for i, c in enumerate(CLASS_NAMES):
            f.write(f"| **{c}** | {metrics['accuracy_recall'][i]:.4f} | {metrics['precision'][i]:.4f} | {metrics['f1'][i]:.4f} | {metrics['iou'][i]:.4f} |\n")

        f.write(f"\n## Summary Metrics\n\n")
        f.write(f"- **Average Per-Class Accuracy (Recall)**: {metrics['average_accuracy']*100:.2f}%\n")
        f.write(f"- **Physical Consistency MAE (30m composition)**: {metrics['consistency_mae']:.4f}\n")
        f.write(f"- **Target Reached (>= 85.0%)**: {'YES' if metrics['average_accuracy'] >= 0.85 else 'NO'}\n")

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY ({os.path.basename(args.model_name)})")
    print("=" * 60)
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:12s} | Recall: {metrics['accuracy_recall'][i]:.4f} | Prec: {metrics['precision'][i]:.4f} | F1: {metrics['f1'][i]:.4f} | IoU: {metrics['iou'][i]:.4f}")
    print("-" * 60)
    print(f"Average Accuracy (Recall): {metrics['average_accuracy']*100:.2f}% (Target: 85.0%)")
    print(f"Physical Consistency MAE:  {metrics['consistency_mae']:.4f}")
    print("=" * 60)
    print(f"Report saved to: {report_path}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model_name', type=str, default='best_model.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patch_size_10m', type=int, default=96)
    parser.add_argument('--stride_10m', type=int, default=96)

    args = parser.parse_args()
    evaluate(args)
