"""
src/ensemble_eval.py

Ensemble evaluation utility for Super-Resolution Land-Cover Mapping.
Takes an arbitrary list of SRUNet checkpoints, averages their softmax probability
outputs at 10m resolution (before taking argmax), and computes:
  - Per-class Recall (Accuracy), Precision, F1-score, and IoU (Classes 0-4)
  - Mean IoU (mIoU) across all five classes
  - Macro average recall (accuracy) across all five classes
  - Physical consistency MAE at 30m composition agreement
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MultiTileDataset, compute_spectral_indices
from model import SRUNet

CLASS_NAMES = ['Water', 'Built-up', 'Vegetation', 'Cropland', 'Bare Land']


def load_model_from_checkpoint(ckpt_path, device):
    """
    Loads SRUNet model from checkpoint, automatically detecting in_channels (4 vs 7).
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    state_dict = torch.load(ckpt_path, map_location=device)
    
    # Detect in_channels from inc block weights
    if 'inc.conv.0.weight' in state_dict:
        in_channels = state_dict['inc.conv.0.weight'].shape[1]
    elif 'inc.double_conv.0.weight' in state_dict:
        in_channels = state_dict['inc.double_conv.0.weight'].shape[1]
    else:
        in_channels = 4
        
    model = SRUNet(in_channels=in_channels, num_classes=5, upscale_factor=3).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, in_channels


def compute_metrics_from_confusion_matrix(conf_matrix, total_abs_error_30m, total_pixels_30m):
    """
    Computes per-class recall, precision, f1, iou, mIoU, macro recall, and MAE.
    conf_matrix shape: (5, 5), rows = true class, cols = predicted class.
    """
    num_classes = conf_matrix.shape[0]
    recalls = np.zeros(num_classes, dtype=float)
    precisions = np.zeros(num_classes, dtype=float)
    f1s = np.zeros(num_classes, dtype=float)
    ious = np.zeros(num_classes, dtype=float)

    for c in range(num_classes):
        tp = float(conf_matrix[c, c])
        fn = float(np.sum(conf_matrix[c, :]) - tp)
        fp = float(np.sum(conf_matrix[:, c]) - tp)

        recalls[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1s[c] = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
        ious[c] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    mean_iou = float(np.mean(ious))
    macro_recall = float(np.mean(recalls))
    consistency_mae = float(total_abs_error_30m / total_pixels_30m) if total_pixels_30m > 0 else 0.0

    metrics = {
        'classes': CLASS_NAMES,
        'accuracy_recall': [float(r) for r in recalls],
        'precision': [float(p) for p in precisions],
        'f1': [float(f) for f in f1s],
        'iou': [float(i) for i in ious],
        'mean_iou': mean_iou,
        'average_accuracy': macro_recall,
        'consistency_mae': consistency_mae,
        'confusion_matrix': conf_matrix.tolist(),
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


def evaluate_ensemble(checkpoint_paths, test_loader=None, data_dir='data', batch_size=64, device=None, verbose=True):
    """
    Evaluates an ensemble of checkpoints on the test split.
    Averages their softmax probabilities before taking argmax.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    if verbose:
        print(f"Loading {len(checkpoint_paths)} models onto {device}...")
        
    models_info = []
    for ckpt in checkpoint_paths:
        model, in_channels = load_model_from_checkpoint(ckpt, device)
        models_info.append((model, in_channels, os.path.basename(ckpt)))
        if verbose:
            print(f"  - Loaded {os.path.basename(ckpt)} (in_channels={in_channels})")

    if test_loader is None:
        if verbose:
            print(f"Loading test split from {data_dir}...")
        test_dataset = MultiTileDataset(
            data_dir, split='test',
            patch_size_10m=96, stride_10m=96,
            use_indices=False
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    conf_matrix = np.zeros((5, 5), dtype=np.int64)
    total_abs_error_30m = 0.0
    total_pixels_30m = 0

    num_models = len(models_info)

    with torch.no_grad():
        for batch_idx, (inputs_4b, labels_10m) in enumerate(test_loader):
            inputs_4b = inputs_4b.to(device)
            labels_10m = labels_10m.to(device)
            B, _, H10, W10 = labels_10m.shape[0], 5, labels_10m.shape[1], labels_10m.shape[2]

            # Precompute 7-band inputs if any model needs it
            inputs_7b = None
            has_7b = any(in_ch == 7 for _, in_ch, _ in models_info)
            if has_7b:
                inputs_7b = compute_spectral_indices(inputs_4b)

            # Sum softmax probabilities across models
            probs_sum = torch.zeros((B, 5, H10, W10), dtype=torch.float32, device=device)

            for model, in_channels, _ in models_info:
                inp = inputs_7b if in_channels == 7 else inputs_4b
                logits = model(inp)
                probs = F.softmax(logits, dim=1)
                probs_sum += probs

            probs_ens = probs_sum / float(num_models)
            preds_10m = torch.argmax(probs_ens, dim=1)

            # 30m physical consistency MAE
            pred_30m_comp = F.avg_pool2d(probs_ens, kernel_size=3, stride=3) # (B, 5, H30, W30)
            labels_one_hot = F.one_hot(labels_10m, num_classes=5).permute(0, 3, 1, 2).float()
            true_30m_comp = F.avg_pool2d(labels_one_hot, kernel_size=3, stride=3)

            abs_diff = torch.abs(pred_30m_comp - true_30m_comp)
            total_abs_error_30m += abs_diff.sum().item()
            total_pixels_30m += abs_diff.numel()

            # Update confusion matrix
            flat_labels = labels_10m.view(-1).cpu().numpy()
            flat_preds = preds_10m.view(-1).cpu().numpy()
            binc = np.bincount(5 * flat_labels + flat_preds, minlength=25)
            conf_matrix += binc.reshape((5, 5))

    metrics = compute_metrics_from_confusion_matrix(conf_matrix, total_abs_error_30m, total_pixels_30m)
    metrics['models'] = [info[2] for info in models_info]
    metrics['checkpoint_paths'] = checkpoint_paths

    if verbose:
        print_evaluation_summary(metrics)

    return metrics


def print_evaluation_summary(metrics):
    models_str = ", ".join(metrics.get('models', []))
    print("\n" + "=" * 68)
    print(f"ENSEMBLE EVALUATION SUMMARY ({models_str})")
    print("=" * 68)
    for cname in CLASS_NAMES:
        rec = metrics['per_class'][cname]['recall'] * 100.0
        prec = metrics['per_class'][cname]['precision'] * 100.0
        f1 = metrics['per_class'][cname]['f1']
        iou = metrics['per_class'][cname]['iou'] * 100.0
        print(f"  {cname:<12} | Recall: {rec:6.2f}% | Prec: {prec:6.2f}% | F1: {f1:.4f} | IoU: {iou:6.2f}%")
    print("-" * 68)
    print(f"Mean IoU (mIoU):              {metrics['mean_iou']*100.0:.2f}% ({metrics['mean_iou']:.4f})")
    print(f"Average Accuracy (Recall):    {metrics['average_accuracy']*100.0:.2f}%")
    print(f"Physical Consistency MAE:     {metrics['consistency_mae']:.4f}")
    print("=" * 68 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SRUNet checkpoint ensemble on test split.")
    parser.add_argument('--checkpoints', nargs='+', required=True, help="List of checkpoint file paths to ensemble.")
    parser.add_argument('--data_dir', type=str, default='data', help="Path to data directory.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--output_json', type=str, default=None, help="Optional path to save metrics JSON.")
    args = parser.parse_args()

    metrics = evaluate_ensemble(args.checkpoints, data_dir=args.data_dir, batch_size=args.batch_size)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.output_json}")
