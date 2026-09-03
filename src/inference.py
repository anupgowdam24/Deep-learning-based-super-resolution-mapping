import os
import argparse
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
import numpy as np
from model import SRUNet
import math

def process_tile(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SRUNet(in_channels=4, num_classes=5, upscale_factor=3).to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_model.pth'), map_location=device))
    model.eval()

    full_tile_path = os.path.join(args.data_dir, 'sentinel2_4band_10m_full_tile.tif')
    output_path = os.path.join(args.output_dir, 'full_tile_prediction.tif')
    
    with rasterio.open(full_tile_path) as src:
        meta = src.meta.copy()
        width_10m = src.width
        height_10m = src.height
        transform = src.transform
        
        meta.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw'
        )
        
        # We need to process in chunks to save memory.
        # Let's read blocks of 10m data, average to 30m, predict, and write back.
        # Model input must be multiple of 8 at 30m. Let's use 30m patch size of 240x240 (720x720 at 10m).
        patch_size_30 = 240
        patch_size_10 = patch_size_30 * 3
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            for y0 in range(0, height_10m, patch_size_10):
                for x0 in range(0, width_10m, patch_size_10):
                    # Read 10m chunk
                    # Make sure we don't go out of bounds, but for the model we need to pad to patch_size_10
                    window = Window(x0, y0, min(patch_size_10, width_10m - x0), min(patch_size_10, height_10m - y0))
                    chunk_10m = src.read(window=window) # (4, H, W)
                    
                    # Pad to patch_size_10 if necessary
                    pad_h = patch_size_10 - chunk_10m.shape[1]
                    pad_w = patch_size_10 - chunk_10m.shape[2]
                    
                    if pad_h > 0 or pad_w > 0:
                        chunk_10m = np.pad(chunk_10m, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
                        
                    # Block average to 30m
                    # chunk_10m is (4, 720, 720)
                    chunk_10m_tensor = torch.from_numpy(chunk_10m).float() / 10000.0
                    chunk_30m_tensor = F.avg_pool2d(chunk_10m_tensor, kernel_size=3, stride=3).unsqueeze(0) # (1, 4, 240, 240)
                    
                    chunk_30m_tensor = chunk_30m_tensor.to(device)
                    
                    with torch.no_grad():
                        logits = model(chunk_30m_tensor) # (1, 5, 720, 720)
                        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8) # (720, 720)
                        
                    # Add 1 back to labels because original classes were 1-5
                    preds = preds + 1
                    
                    # Crop back to original window size
                    valid_h = window.height
                    valid_w = window.width
                    preds_valid = preds[:valid_h, :valid_w]
                    
                    dst.write(preds_valid, 1, window=window)
                    
    print(f"Inference complete. Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to data folder')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='Directory containing saved model')
    parser.add_argument('--output_dir', type=str, default='../outputs', help='Directory to save outputs')
    
    args = parser.parse_args()
    process_tile(args)
