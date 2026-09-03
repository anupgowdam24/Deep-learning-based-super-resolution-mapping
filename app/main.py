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

# Add src to path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import SRUNet

app = FastAPI(title="SRM Pipeline API")

# Mount static files for frontend
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Load model globally
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = SRUNet(in_channels=4, num_classes=5, upscale_factor=3).to(device)
checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model.pth')
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Color map for the 5 classes: Water (Blue), Built-up (Red), Vegetation (Green), Cropland (Yellow), Bare Land (Brown)
# Note: output classes are 1-5, but internally 0-4.
# Let's map 0-4 to RGB colors.
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
    os.remove('temp.png')
    return f"data:image/png;base64,{encoded}"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    with MemoryFile(contents) as memfile:
        with memfile.open() as dataset:
            img = dataset.read() # (4, H, W)
            
    # Normalize
    img_tensor = torch.from_numpy(img).float() / 10000.0
    
    # Check if we need to pad to multiple of 8 (for 30m -> U-Net requires divisible by 8)
    h, w = img_tensor.shape[1], img_tensor.shape[2]
    
    # If the user uploads the 10m image instead of 30m, let's just average it down to 30m
    # To keep it simple, we assume if dimensions are > 500 it might be the 10m one.
    if h > 500 or w > 500:
         img_tensor = F.avg_pool2d(img_tensor.unsqueeze(0), kernel_size=3, stride=3).squeeze(0)
         h, w = img_tensor.shape[1], img_tensor.shape[2]
         
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
    # Crop to max 512x512 to prevent OOM
    MAX_SIZE = 512
    if img_tensor.shape[1] > MAX_SIZE or img_tensor.shape[2] > MAX_SIZE:
        img_tensor = img_tensor[:, :MAX_SIZE, :MAX_SIZE]
        
    h_infer, w_infer = img_tensor.shape[1], img_tensor.shape[2]
    img_tensor = img_tensor.unsqueeze(0).to(device) # (1, 4, H_pad, W_pad)
    
    with torch.no_grad():
        logits = model(img_tensor) # (1, 5, H_pad*3, W_pad*3)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    # Crop back padding (or max size)
    preds = preds[: (min(h, h_infer) * 3), : (min(w, w_infer) * 3)]
        
    # Colorize
    pred_rgb = COLOR_MAP[preds]
    
    # Prepare input RGB visualization (Bands 2,1,0 -> R,G,B for S2)
    # img_tensor is (1, 4, h_infer, w_infer). Let's extract bands 2,1,0
    vis_tensor = img_tensor[0, :3, :min(h, h_infer), :min(w, w_infer)].cpu().numpy()
    input_rgb = np.stack([vis_tensor[2], vis_tensor[1], vis_tensor[0]], axis=-1)
    
    # Scale for visualization
    input_rgb = input_rgb.astype(np.float32) * 3.0
    input_rgb = np.clip(input_rgb, 0, 1)
    
    return JSONResponse({
        "input_image": image_to_base64(input_rgb),
        "prediction_image": image_to_base64(pred_rgb)
    })
