import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import SRDataset, get_class_weights, MultiTileDataset, get_class_weights_combined
from model import SRUNet
from loss import SRLoss
import matplotlib.pyplot as plt

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("Note: GPU unavailable, training on CPU. Expect slower runtime.")

    # Datasets and Loaders
    if args.use_second_tile:
        data_dir_v2 = os.path.join(args.data_dir, 'v2')
        train_dataset = MultiTileDataset(
            args.data_dir, data_dir_v2,
            oversample_bareland=args.oversample_bareland,
            use_indices=args.use_indices
        )
        class_weights = get_class_weights_combined(train_dataset, mode=args.weight_mode).to(device)
    else:
        train_dataset = SRDataset(args.data_dir, split='train', oversample_bareland=args.oversample_bareland, use_indices=args.use_indices)
        class_weights = get_class_weights(train_dataset, mode=args.weight_mode).to(device)

    val_dataset = SRDataset(args.data_dir, split='val', oversample_bareland=False, use_indices=args.use_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train patches: {len(train_dataset)}, Val patches: {len(val_dataset)}")
    print(f"Class weights ({args.weight_mode}): {class_weights.cpu().numpy()}")
    
    # Model
    model = SRUNet(in_channels=args.in_channels, num_classes=5, upscale_factor=3).to(device)
    
    # Loss & Optimizer
    criterion = SRLoss(class_weights=class_weights, consistency_lambda=args.consistency_lambda,
                       use_focal=args.use_focal, focal_gamma=args.focal_gamma,
                       use_dice=args.use_dice)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        epoch_train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss, clf_loss, cons_loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation Phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss, clf_loss, cons_loss = criterion(logits, labels)
                epoch_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_path = os.path.join(args.checkpoint_dir, args.model_name)
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model to {save_path}")
            
    # Plot loss curves
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    model_stem = args.model_name.replace('.pth', '')
    loss_curve_name = f"loss_curves_{model_stem}.png"
    plt.savefig(os.path.join(args.output_dir, loss_curve_name))
    plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to data folder')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='Directory to save model')
    parser.add_argument('--output_dir', type=str, default='../outputs', help='Directory to save outputs')
    parser.add_argument('--model_name', type=str, default='best_model_v8.pth', help='Filename for saved model checkpoint')
    parser.add_argument('--in_channels', type=int, default=4, help='Number of input channels')
    parser.add_argument('--no_indices', dest='use_indices', action='store_false', help='Disable spectral index channels')
    parser.set_defaults(use_indices=False)
    parser.add_argument('--use_second_tile', action='store_true', help='Include second tile from data/v2 in training')
    parser.add_argument('--weight_mode', type=str, default='sqrt', choices=['sqrt', 'inverse'], help='Class weighting mode')
    parser.add_argument('--oversample_bareland', action='store_true', help='Oversample patches with >=10%% Bare Land')
    parser.add_argument('--use_focal', action='store_true', help='Use Focal Loss instead of Cross Entropy')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss')
    parser.add_argument('--use_dice', action='store_true', help='Use combined 0.5*Weighted_CE + 0.5*Dice Loss')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--consistency_lambda', type=float, default=0.3, help='Weight for consistency loss')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
