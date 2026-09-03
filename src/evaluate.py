import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SRDataset
from model import SRUNet
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    ckpt_path = os.path.join(args.checkpoint_dir, args.model_name)
    state_dict = torch.load(ckpt_path, map_location=device)
    in_channels = state_dict['inc.double_conv.0.weight'].shape[1]
    use_indices = (in_channels == 7)
    
    test_dataset = SRDataset(args.data_dir, split='test', use_indices=use_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = SRUNet(in_channels=in_channels, num_classes=5, upscale_factor=3).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    all_preds_10m = []
    all_labels_10m = []
    all_preds_30m_comp = []
    all_labels_30m_comp = []
    
    saved_viz = False
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # For consistency MAE
            pred_30m_composition = F.avg_pool2d(probs, kernel_size=3, stride=3)
            labels_one_hot = F.one_hot(labels, num_classes=5).permute(0, 3, 1, 2).float()
            true_30m_composition = F.avg_pool2d(labels_one_hot, kernel_size=3, stride=3)
            
            all_preds_10m.append(preds.cpu().numpy())
            all_labels_10m.append(labels.cpu().numpy())
            all_preds_30m_comp.append(pred_30m_composition.cpu().numpy())
            all_labels_30m_comp.append(true_30m_composition.cpu().numpy())
            
            # Save one qualitative visual
            if not saved_viz:
                idx = 0
                viz_input = inputs[idx].cpu().numpy()
                viz_pred = preds[idx].cpu().numpy()
                viz_label = labels[idx].cpu().numpy()
                
                rgb = np.stack([viz_input[2], viz_input[1], viz_input[0]], axis=-1)
                rgb = np.clip(rgb * 3.0, 0, 1) # simple brightness scaling
                
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(rgb)
                plt.title('30m Input (RGB)')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(viz_pred, cmap='tab10', vmin=0, vmax=4)
                plt.title('10m Prediction')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(viz_label, cmap='tab10', vmin=0, vmax=4)
                plt.title('10m True Label')
                plt.axis('off')
                
                os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
                viz_filename = f"patch_comparison_{args.model_name.replace('.pth', '')}.png"
                plt.savefig(os.path.join(args.output_dir, 'visualizations', viz_filename))
                plt.close()
                saved_viz = True
                
    # Flatten arrays
    flat_preds = np.concatenate(all_preds_10m).flatten()
    flat_labels = np.concatenate(all_labels_10m).flatten()
    
    # Calculate metrics
    classes = ['Water', 'Built-up', 'Vegetation', 'Cropland', 'Bare Land']
    precision = precision_score(flat_labels, flat_preds, average=None, labels=[0,1,2,3,4], zero_division=0)
    recall = recall_score(flat_labels, flat_preds, average=None, labels=[0,1,2,3,4], zero_division=0)
    f1 = f1_score(flat_labels, flat_preds, average=None, labels=[0,1,2,3,4], zero_division=0)
    iou = jaccard_score(flat_labels, flat_preds, average=None, labels=[0,1,2,3,4], zero_division=0)
    
    # Consistency MAE
    preds_comp_flat = np.concatenate(all_preds_30m_comp).reshape(-1, 5)
    labels_comp_flat = np.concatenate(all_labels_30m_comp).reshape(-1, 5)
    consistency_mae = np.mean(np.abs(preds_comp_flat - labels_comp_flat))
    
    # Write report
    report_filename = f"metrics_report_{args.model_name.replace('.pth', '')}.md"
    report_path = os.path.join(args.output_dir, report_filename)
    with open(report_path, 'w') as f:
        f.write(f"# Model Evaluation Metrics ({args.model_name})\n\n")
        f.write("## Per-class Metrics (10m)\n\n")
        f.write("| Class | Accuracy (Recall) | Precision | F1 Score | IoU |\n")
        f.write("|---|---|---|---|---|\n")
        for i, c in enumerate(classes):
            f.write(f"| {c} | {recall[i]:.4f} | {precision[i]:.4f} | {f1[i]:.4f} | {iou[i]:.4f} |\n")
            
        f.write(f"\n## Physical Consistency\n")
        f.write(f"- Mean Absolute Error (MAE) at 30m composition: {consistency_mae:.4f}\n")
        
    print(f"Evaluation complete for {args.model_name}. Metrics saved to {report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to data folder')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='Directory containing saved model')
    parser.add_argument('--output_dir', type=str, default='../outputs', help='Directory to save outputs')
    parser.add_argument('--model_name', type=str, default='best_model_v2.pth', help='Model checkpoint file to evaluate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    evaluate(args)
