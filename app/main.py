import os
import io
import base64
import torch
import torch.nn.functional as F
import rasterio
from rasterio.io import MemoryFile
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add src to path to import model & helpers
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import SRUNet
from dataset import compute_spectral_indices

app = FastAPI(title="SRM Pipeline API")

# Mount static files for frontend
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Load models globally (Ensemble v4 + v7)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

v4_path = os.path.join(ckpt_dir, 'best_model_v4.pth')
v7_path = os.path.join(ckpt_dir, 'best_model_v7.pth')

model_v4 = SRUNet(in_channels=4, num_classes=5, upscale_factor=3).to(device)
if os.path.exists(v4_path):
    model_v4.load_state_dict(torch.load(v4_path, map_location=device))
model_v4.eval()

model_v7 = None
if os.path.exists(v7_path):
    model_v7 = SRUNet(in_channels=7, num_classes=5, upscale_factor=3).to(device)
    model_v7.load_state_dict(torch.load(v7_path, map_location=device))
    model_v7.eval()

# Color map for the 5 classes:
# 0: Water (Blue) [0, 0, 255]
# 1: Built-up (Red) [255, 0, 0]
# 2: Vegetation (Green) [0, 128, 0]
# 3: Cropland (Yellow) [255, 255, 0]
# 4: Bare Land (Brown) [165, 42, 42]
COLOR_MAP = np.array([
    [0, 0, 255],     # 0: Water
    [255, 0, 0],     # 1: Built-up
    [0, 128, 0],     # 2: Vegetation
    [255, 255, 0],   # 3: Cropland
    [165, 42, 42]    # 4: Bare Land
], dtype=np.uint8)

def image_to_base64(img_array):
    """Convert numpy RGB array to base64 string"""
    plt.imsave('temp.png', img_array)
    with open('temp.png', 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    if os.path.exists('temp.png'):
        os.remove('temp.png')
    return f"data:image/png;base64,{encoded}"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    with MemoryFile(contents) as memfile:
        with memfile.open() as dataset:
            img = dataset.read() # (C, H, W)
            
    num_channels, h, w = img.shape
    
    # Ensure 4-band format
    if num_channels >= 4:
        img_4b = img[:4].astype(np.float32)
    elif num_channels == 3:
        # RGB only: add dummy NIR channel as mean of RGB
        nir = img.mean(axis=0, keepdims=True).astype(np.float32)
        img_4b = np.concatenate([img, nir], axis=0)
    else:
        # Single channel: duplicate across 4 channels
        img_4b = np.repeat(img[:1], 4, axis=0).astype(np.float32)
        
    img_tensor = torch.from_numpy(img_4b).float()
    
    # Handle normalization
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 10000.0
        
    # If dimensions > 500, average down from 10m to 30m
    if h > 500 or w > 500:
        img_tensor = F.avg_pool2d(img_tensor.unsqueeze(0), kernel_size=3, stride=3).squeeze(0)
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        
    # Tiled inference in 32x32 (30m) patches to prevent striping/grid artifacts
    patch_30 = 32
    H30, W30 = img_tensor.shape[1], img_tensor.shape[2]
    
    pred_map = np.zeros((H30 * 3, W30 * 3), dtype=np.uint8)
    
    with torch.no_grad():
        for y0 in range(0, H30, patch_30):
            for x0 in range(0, W30, patch_30):
                patch_h = min(patch_30, H30 - y0)
                patch_w = min(patch_30, W30 - x0)
                
                sub = img_tensor[:, y0:y0+patch_h, x0:x0+patch_w]
                pad_h = patch_30 - patch_h
                pad_w = patch_30 - patch_w
                
                if pad_h > 0 or pad_w > 0:
                    sub_4 = F.pad(sub, (0, pad_w, 0, pad_h), mode='constant', value=0)
                else:
                    sub_4 = sub
                    
                sub_4_b = sub_4.unsqueeze(0).to(device)
                
                p4 = F.softmax(model_v4(sub_4_b), dim=1)
                
                if model_v7 is not None:
                    sub_7_b = compute_spectral_indices(sub_4_b)
                    p7 = F.softmax(model_v7(sub_7_b), dim=1)
                    probs_ens = 0.5 * p4 + 0.5 * p7
                else:
                    probs_ens = p4
                    
                preds = torch.argmax(probs_ens, dim=1).squeeze(0).cpu().numpy()
                preds_valid = preds[:patch_h*3, :patch_w*3]
                
                pred_map[y0*3:(y0+patch_h)*3, x0*3:(x0+patch_w)*3] = preds_valid

    # Colorize prediction
    pred_rgb = COLOR_MAP[pred_map]
    
    # Prepare input RGB visualization (Bands 2,1,0 -> R,G,B for Sentinel-2)
    vis_tensor = img_tensor[:3, :H30, :W30].cpu().numpy()
    input_rgb = np.stack([vis_tensor[2], vis_tensor[1], vis_tensor[0]], axis=-1)
    
    # Scale for RGB display
    if input_rgb.max() <= 1.0:
        input_rgb = input_rgb * 3.0
    else:
        input_rgb = input_rgb / 255.0 * 3.0
    input_rgb = np.clip(input_rgb, 0, 1)
    
    return JSONResponse({
        "input_image": image_to_base64(input_rgb),
        "prediction_image": image_to_base64(pred_rgb)
    })
