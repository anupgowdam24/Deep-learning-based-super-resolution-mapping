"""
inference.py

Chunk-based inference script for the Super-Resolution Land-Cover Mapping model.
Processes large Sentinel-2 GeoTIFFs in spatial windows to generate 10m land-cover
predictions without memory overflow.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window

from model import SRUNet
from dataset import compute_spectral_indices


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Inference device: {device}")

    # Load checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, args.model_name)
    if not os.path.exists(ckpt_path):
        ckpt_path = args.model_name
    state_dict = torch.load(ckpt_path, map_location=device)

    in_channels = state_dict['inc.conv.0.weight'].shape[1] if 'inc.conv.0.weight' in state_dict else state_dict['inc.double_conv.0.weight'].shape[1]
    use_indices = (in_channels == 7)

    # Initialize model
    model = SRUNet(in_channels=in_channels, num_classes=5, upscale_factor=3).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        crs = src.crs
        src_res = src.res[0]
        width = src.width
        height = src.height

        # Determine processing parameters based on whether input is 10m or 30m
        is_10m_input = (abs(src_res - 10.0) < 1.0)
        print(f"Input image: {width}x{height}, {src.count} bands, CRS={crs}, Res={src.res}, 10m_mode={is_10m_input}")

        # Destination metadata
        dst_transform = src.transform
        dst_width = width if is_10m_input else width * 3
        dst_height = height if is_10m_input else height * 3

        meta.update({
            'driver': 'GTiff',
            'dtype': rasterio.uint8,
            'count': 1,
            'width': dst_width,
            'height': dst_height,
            'crs': crs,
            'transform': dst_transform,
            'nodata': 0,
            'compress': 'lzw',
            'tiled': True
        })

        chunk_size_30m = args.chunk_size_30m  # multiple of 8, e.g. 64 or 128
        chunk_size_10m = chunk_size_30m * 3

        with rasterio.open(output_path, 'w', **meta) as dst:
            if is_10m_input:
                for y0 in range(0, height, chunk_size_10m):
                    for x0 in range(0, width, chunk_size_10m):
                        w_h = min(chunk_size_10m, height - y0)
                        w_w = min(chunk_size_10m, width - x0)
                        win = Window(x0, y0, w_w, w_h)

                        raw_10m = src.read(window=win)  # (C, H, W)
                        if raw_10m.shape[1] == 0 or raw_10m.shape[2] == 0:
                            continue

                        # Pad to multiple of chunk_size_10m
                        pad_h = chunk_size_10m - raw_10m.shape[1]
                        pad_w = chunk_size_10m - raw_10m.shape[2]
                        if pad_h > 0 or pad_w > 0:
                            raw_10m = np.pad(raw_10m, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

                        # 3x3 block-average to 30m
                        t_10m = torch.from_numpy(raw_10m[:4]).float() / 10000.0
                        t_30m = F.avg_pool2d(t_10m.unsqueeze(0), kernel_size=3, stride=3).squeeze(0)

                        if use_indices:
                            t_30m = compute_spectral_indices(t_30m)

                        t_30m = t_30m.unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = model(t_30m)
                            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                        # Remap 0..4 back to 1..5
                        preds = preds + 1
                        preds_valid = preds[:w_h, :w_w]
                        dst.write(preds_valid, 1, window=win)
            else:
                # Direct 30m input
                for y0 in range(0, height, chunk_size_30m):
                    for x0 in range(0, width, chunk_size_30m):
                        w_h = min(chunk_size_30m, height - y0)
                        w_w = min(chunk_size_30m, width - x0)
                        win_30 = Window(x0, y0, w_w, w_h)
                        win_10 = Window(x0 * 3, y0 * 3, w_w * 3, w_h * 3)

                        raw_30m = src.read(window=win_30)
                        if raw_30m.shape[1] == 0 or raw_30m.shape[2] == 0:
                            continue

                        pad_h = chunk_size_30m - raw_30m.shape[1]
                        pad_w = chunk_size_30m - raw_30m.shape[2]
                        if pad_h > 0 or pad_w > 0:
                            raw_30m = np.pad(raw_30m, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

                        t_30m = torch.from_numpy(raw_30m[:4]).float() / 10000.0
                        if use_indices:
                            t_30m = compute_spectral_indices(t_30m)

                        t_30m = t_30m.unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = model(t_30m)
                            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                        preds = preds + 1
                        preds_valid = preds[:w_h * 3, :w_w * 3]
                        dst.write(preds_valid, 1, window=win_10)

    print(f"Inference complete: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/sentinel2_4band_10m_crop.tif')
    parser.add_argument('--output_path', type=str, default='outputs/prediction_10m.tif')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name', type=str, default='best_model.pth')
    parser.add_argument('--chunk_size_30m', type=int, default=64)

    args = parser.parse_args()
    run_inference(args)
