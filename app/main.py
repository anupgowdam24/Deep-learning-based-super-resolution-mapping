import os
import io
import base64
import torch
import torch.nn.functional as F
import rasterio
from rasterio.io import MemoryFile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add src to path to import model & helpers
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import SRUNet
from dataset import compute_spectral_indices

app = FastAPI(title="SRM Pipeline API")

# Enable CORS for local React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Load models globally (Winning Forward Selection Ensemble: 8, 1, 4, 6)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

ENSEMBLE_CHECKPOINT_NAMES = [
    'training_8_best.pth',
    'training_1_best.pth',
    'training_4_best.pth',
    'training_6_best.pth'
]

loaded_ensemble = []
for ckpt_name in ENSEMBLE_CHECKPOINT_NAMES:
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        in_ch = state_dict['inc.conv.0.weight'].shape[1] if 'inc.conv.0.weight' in state_dict else 4
        m = SRUNet(in_channels=in_ch, num_classes=5, upscale_factor=3).to(device)
        m.load_state_dict(state_dict)
        m.eval()
        loaded_ensemble.append((m, in_ch, ckpt_name))
        print(f"Loaded ensemble model: {ckpt_name} (in_channels={in_ch})")
    else:
        print(f"Warning: Checkpoint not found: {ckpt_path}")

print(f"App initialized with {len(loaded_ensemble)} ensemble models.")

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

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_images')
SAMPLES_METADATA = [
    {
        "id": "sample_1_bare_rock",
        "name": "Bare Rock / Quarry",
        "description": "High-reflectance rock outcrops and quarries",
        "filename": "sample_1_bare_rock.tif"
    },
    {
        "id": "sample_2_fallow_fields",
        "name": "Fallow Fields & Cropland",
        "description": "Agricultural parcels, fallow soil & reservoir",
        "filename": "sample_2_fallow_fields.tif"
    },
    {
        "id": "sample_3_urban_water",
        "name": "Urban & Water Reservoir",
        "description": "Dense urban structures and water body",
        "filename": "sample_3_urban_water.tif"
    },
    {
        "id": "sample_4_dense_vegetation",
        "name": "Dense Forest Canopy",
        "description": "Dense forest and mixed vegetation canopy",
        "filename": "sample_4_dense_vegetation.tif"
    }
]

@app.on_event("startup")
async def startup_event():
    """Pre-warm MPS/CUDA shaders with a dummy forward pass to eliminate first-request lag."""
    print(f"Pre-warming {len(loaded_ensemble)} models on {device}...")
    dummy = torch.zeros((1, 4, 32, 32), device=device)
    dummy_7b = compute_spectral_indices(dummy)
    with torch.no_grad():
        for model, in_ch, _ in loaded_ensemble:
            inp = dummy_7b if in_ch == 7 else dummy
            _ = model(inp)
    print("Warmup complete! API ready for instant inference.")

def image_to_base64(img_array):
    """Convert numpy RGB array to base64 string using in-memory buffer"""
    buf = io.BytesIO()
    plt.imsave(buf, img_array, format='png')
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def stretch_rgb_percentile(rgb_array):
    """
    Robust 2%-98% percentile contrast stretch per channel for crisp natural-color satellite display.
    Handles raw 16-bit reflectance, 8-bit imagery, and float reflectance.
    """
    stretched = np.zeros_like(rgb_array, dtype=np.float32)
    for c in range(3):
        channel = rgb_array[..., c]
        valid = channel[channel > 0]
        if len(valid) > 20:
            p2 = float(np.percentile(valid, 2))
            p98 = float(np.percentile(valid, 98))
            if p98 > p2:
                stretched[..., c] = np.clip((channel - p2) / (p98 - p2), 0.0, 1.0)
            else:
                stretched[..., c] = np.clip(channel, 0.0, 1.0)
        else:
            stretched[..., c] = np.clip(channel, 0.0, 1.0)
    return stretched

def normalize_and_format_input(img):
    """
    Validates and formats arbitrary raster array (C, H, W) to (4, H, W) normalized reflectance in [0, 1].
    Rejects label masks (values 0-5) with a helpful explanatory error.
    """
    num_channels = img.shape[0]
    
    # Check for accidental label mask upload (single channel with integer class values <= 10)
    if num_channels == 1 and float(np.max(img)) <= 10.0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file appears to be a land-cover classification label mask (classes 0-5), not optical satellite imagery. Please upload a 4-band Sentinel-2 TIFF image (B02, B03, B04, B08)."
        )
        
    # Format to 4 bands
    if num_channels >= 4:
        img_4b = img[:4].astype(np.float32)
    elif num_channels == 3:
        # RGB only: add dummy NIR channel as mean of RGB
        nir = img.mean(axis=0, keepdims=True).astype(np.float32)
        img_4b = np.concatenate([img, nir], axis=0).astype(np.float32)
    else:
        # Single channel optical grayscale: repeat across 4 channels
        img_4b = np.repeat(img[:1], 4, axis=0).astype(np.float32)
        
    # Normalization:
    max_val = float(np.max(img_4b))
    if img.dtype == np.uint8 or (max_val > 1.0 and max_val <= 255.0 and img.dtype != np.float32):
        img_4b = img_4b / 255.0
    elif max_val > 10.0:
        # Raw 16-bit reflectance (0-10000)
        img_4b = np.clip(img_4b / 10000.0, 0.0, 1.0)
    elif max_val > 1.0:
        # Slight float overshoot
        img_4b = np.clip(img_4b / max_val, 0.0, 1.0)
        
    return img_4b

def run_ensemble_inference(img_4b):
    """
    Executes batched inference (batch_size=32) across the ensemble and returns
    base64-encoded input RGB (with 2-98% percentile stretch) and 10m predicted map.
    """
    img_tensor = torch.from_numpy(img_4b).float()
    h, w = img_tensor.shape[1], img_tensor.shape[2]
    
    # If dimensions > 500, average down from 10m to 30m
    if h > 500 or w > 500:
        img_tensor = F.avg_pool2d(img_tensor.unsqueeze(0), kernel_size=3, stride=3).squeeze(0)
    
    H30, W30 = img_tensor.shape[1], img_tensor.shape[2]
    patch_30 = 32
    
    # Slice patches
    patches_list = []
    patch_coords = []
    for y0 in range(0, H30, patch_30):
        for x0 in range(0, W30, patch_30):
            patch_h = min(patch_30, H30 - y0)
            patch_w = min(patch_30, W30 - x0)
            sub = img_tensor[:, y0:y0+patch_h, x0:x0+patch_w]
            pad_h = patch_30 - patch_h
            pad_w = patch_30 - patch_w
            if pad_h > 0 or pad_w > 0:
                sub_pad = F.pad(sub, (0, pad_w, 0, pad_h), mode='constant', value=0)
            else:
                sub_pad = sub
            patches_list.append(sub_pad)
            patch_coords.append((y0, x0, patch_h, patch_w))
            
    patches_tensor = torch.stack(patches_list, dim=0) # (N, 4, 32, 32)
    pred_map = np.zeros((H30 * 3, W30 * 3), dtype=np.uint8)
    
    # Batched inference with batch_size = 32
    batch_size = 32
    has_7b = any(in_ch == 7 for _, in_ch, _ in loaded_ensemble)
    
    with torch.no_grad():
        for i in range(0, len(patches_list), batch_size):
            batch_4b = patches_tensor[i:i+batch_size].to(device)
            batch_7b = compute_spectral_indices(batch_4b) if has_7b else None
            
            probs_sum = torch.zeros((batch_4b.shape[0], 5, patch_30 * 3, patch_30 * 3), dtype=torch.float32, device=device)
            for model, in_ch, _ in loaded_ensemble:
                inp = batch_7b if in_ch == 7 else batch_4b
                probs_sum += F.softmax(model(inp), dim=1)
                
            probs_ens = probs_sum / float(len(loaded_ensemble)) if len(loaded_ensemble) > 0 else probs_sum
            batch_preds = torch.argmax(probs_ens, dim=1).cpu().numpy()
            
            for j in range(batch_preds.shape[0]):
                y0, x0, ph, pw = patch_coords[i + j]
                pred_map[y0*3:(y0+ph)*3, x0*3:(x0+pw)*3] = batch_preds[j, :ph*3, :pw*3]
                
    # Colorize prediction
    pred_rgb = COLOR_MAP[pred_map]
    
    # Prepare input RGB visualization with 2%-98% percentile stretch
    vis_tensor = img_tensor[:3, :H30, :W30].cpu().numpy()
    # Sentinel-2 band order: 0=Blue, 1=Green, 2=Red. Display RGB: Red, Green, Blue
    raw_rgb = np.stack([vis_tensor[2], vis_tensor[1], vis_tensor[0]], axis=-1)
    input_rgb = stretch_rgb_percentile(raw_rgb)
    
    return image_to_base64(input_rgb), image_to_base64(pred_rgb)

@app.get("/samples")
async def list_samples():
    """Returns pre-loaded sample scenes for instant 1-click testing."""
    return JSONResponse(SAMPLES_METADATA)

@app.post("/predict_sample/{sample_id}")
async def predict_sample(sample_id: str):
    """Executes instant inference on a server-side sample without upload overhead."""
    match = next((s for s in SAMPLES_METADATA if s["id"] == sample_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
    sample_path = os.path.join(SAMPLES_DIR, match["filename"])
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Sample file '{sample_path}' not found.")
        
    with rasterio.open(sample_path) as src:
        img = src.read()
        
    img_4b = normalize_and_format_input(img)
    input_b64, pred_b64 = run_ensemble_inference(img_4b)
    return JSONResponse({
        "input_image": input_b64,
        "prediction_image": pred_b64,
        "sample": match
    })

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Standard file upload inference endpoint with validation and contrast stretch."""
    contents = await file.read()
    
    with MemoryFile(contents) as memfile:
        with memfile.open() as dataset:
            img = dataset.read() # (C, H, W)
            
    img_4b = normalize_and_format_input(img)
    input_b64, pred_b64 = run_ensemble_inference(img_4b)
    
    return JSONResponse({
        "input_image": input_b64,
        "prediction_image": pred_b64
    })
