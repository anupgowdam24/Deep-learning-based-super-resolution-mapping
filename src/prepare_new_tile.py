"""
prepare_new_tile.py

Preprocessing script for the second training tile from the new Sentinel-2 SAFE acquisition.
Steps:
  1. Auto-select the best ~1000x1000 pixel (10km^2) window from the new tile based on BSI
  2. Crop all four 10m bands to selected window -> sentinel2_v2_4band_10m.tif
  3. Create synthetic 30m via 3x3 block averaging -> sentinel2_v2_4band_synthetic_30m.tif
  4. Download ESA WorldCover 2021 tile for the sub-window extent
  5. Reproject/crop WorldCover to match the 10m grid
  6. Remap WorldCover class codes to 1-5 encoding -> worldcover_v2_5class_10m.tif
  7. Print class pixel counts to verify Bare Land coverage
"""

import os
import glob
import urllib.request
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import warnings
warnings.filterwarnings('ignore')

SAFE_DIR = r"C:\Users\anupg\Downloads\S2A_MSIL2A_20260628T051241_N0512_R019_T43PHR_20260628T101512.SAFE"
OUTPUT_DIR = r"C:\Users\anupg\OneDrive\Desktop\Competion\project\data\v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Band paths
bands = {}
for b in ['B02', 'B03', 'B04', 'B08']:
    path = glob.glob(os.path.join(SAFE_DIR, '**', f'*{b}_10m.jp2'), recursive=True)[0]
    bands[b] = path
    print(f"Found {b}: {os.path.basename(path)}")

print("\n=== Step 1: Auto-selecting best Bare Land window ===")
# Read B03 (Green), B04 (Red), B08 (NIR) at 60m resolution for scanning
# We use 30m bands downsampled 3x more via a window to speed up scan
SCAN_DOWNSAMPLE = 30  # scan at 300m effectively (every 30th pixel at 10m)
WINDOW_PIXELS = 1000  # ~10km x 10km at 10m
SCAN_STEP = 500       # scan every 500 pixels = ~5km step

best_bsi = -999
best_window = None

with rasterio.open(bands['B02']) as b02_src, \
     rasterio.open(bands['B03']) as b03_src, \
     rasterio.open(bands['B04']) as b04_src, \
     rasterio.open(bands['B08']) as b08_src:
    
    full_h = b02_src.height
    full_w = b02_src.width
    
    # Scan coarsely over whole tile
    print(f"  Scanning {full_w}x{full_h} tile at {SCAN_DOWNSAMPLE}x downsample...")
    for y in range(0, full_h - WINDOW_PIXELS, SCAN_STEP * 3):
        for x in range(0, full_w - WINDOW_PIXELS, SCAN_STEP * 3):
            win = rasterio.windows.Window(x, y, WINDOW_PIXELS, WINDOW_PIXELS)
            # Read at reduced scale by reading every Nth pixel
            green = b03_src.read(1, window=win)[::SCAN_DOWNSAMPLE, ::SCAN_DOWNSAMPLE].astype(np.float32)
            red = b04_src.read(1, window=win)[::SCAN_DOWNSAMPLE, ::SCAN_DOWNSAMPLE].astype(np.float32)
            nir = b08_src.read(1, window=win)[::SCAN_DOWNSAMPLE, ::SCAN_DOWNSAMPLE].astype(np.float32)
            
            # Skip mostly empty (cloud/nodata) windows
            if np.mean(red) < 200:
                continue
            
            # BSI = ((red + green) - nir) / ((red + green) + nir + 1e-6)
            bsi = np.mean(((red + green) - nir) / ((red + green) + nir + 1e-6))
            
            if bsi > best_bsi:
                best_bsi = bsi
                best_window = (x, y)
    
    print(f"  Best window: x={best_window[0]}, y={best_window[1]}, BSI={best_bsi:.4f}")
    
    # Validate window is fully within tile (trim if needed)
    x_off, y_off = best_window
    x_off = min(x_off, full_w - WINDOW_PIXELS)
    y_off = min(y_off, full_h - WINDOW_PIXELS)
    
    crop_win = rasterio.windows.Window(x_off, y_off, WINDOW_PIXELS, WINDOW_PIXELS)
    
    print(f"\n=== Step 2: Cropping 10m bands ===")
    # Read all four bands at selected window
    band_data = {}
    with rasterio.open(bands['B02']) as src:
        transform_10m = src.window_transform(crop_win)
        crs = src.crs
        # bounds returns (left, bottom, right, top) plain tuple
        _bounds = src.window_bounds(crop_win)
        b_left, b_bottom, b_right, b_top = _bounds
        band_data['B02'] = src.read(1, window=crop_win).astype(np.uint16)
    for b in ['B03', 'B04', 'B08']:
        with rasterio.open(bands[b]) as src:
            band_data[b] = src.read(1, window=crop_win).astype(np.uint16)
    
    stack_10m = np.stack([band_data['B02'], band_data['B03'], band_data['B04'], band_data['B08']], axis=0)
    print(f"  Shape: {stack_10m.shape}, dtype: {stack_10m.dtype}")
    print(f"  Bounds: left={b_left}, bottom={b_bottom}, right={b_right}, top={b_top}")

out_10m = os.path.join(OUTPUT_DIR, 'sentinel2_v2_4band_10m.tif')
with rasterio.open(out_10m, 'w', driver='GTiff', count=4, dtype='uint16',
                   width=WINDOW_PIXELS, height=WINDOW_PIXELS, crs=crs,
                   transform=transform_10m) as dst:
    dst.write(stack_10m)
print(f"  Saved: {out_10m}")

print("\n=== Step 3: Creating synthetic 30m via 3x3 block averaging ===")
# Block-average 1000x1000 10m -> 333x333 30m
h_30m = WINDOW_PIXELS // 3
w_30m = WINDOW_PIXELS // 3
stack_10m_f = stack_10m.astype(np.float32)
# Trim to exactly divisible size
trim_h = h_30m * 3
trim_w = w_30m * 3
stack_10m_f = stack_10m_f[:, :trim_h, :trim_w]
# Reshape and mean
stack_30m = stack_10m_f.reshape(4, h_30m, 3, w_30m, 3).mean(axis=(2, 4)).astype(np.uint16)
print(f"  Shape: {stack_30m.shape}")

transform_30m = rasterio.transform.from_bounds(
    b_left, b_bottom, b_right, b_top,
    w_30m, h_30m
)

out_30m = os.path.join(OUTPUT_DIR, 'sentinel2_v2_4band_synthetic_30m.tif')
with rasterio.open(out_30m, 'w', driver='GTiff', count=4, dtype='uint16',
                   width=w_30m, height=h_30m, crs=crs,
                   transform=transform_30m) as dst:
    dst.write(stack_30m)
print(f"  Saved: {out_30m}")

print("\n=== Step 4: Downloading ESA WorldCover 2021 tile ===")
# WorldCover 2021 v200 on AWS: 
# https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_<lat><lon>_Map.tif
# The new tile is in zone T43PHR (UTM 43N, ~76-80E, ~13-23N)
# We need to convert the crop bounds to lat/lon to get the right WorldCover tile

# Bounds in UTM 32643: left=x_off*10+799980, bottom, right, top (approx)
from rasterio.warp import transform_bounds as warp_transform_bounds
lon_min, lat_min, lon_max, lat_max = warp_transform_bounds(
    crs, CRS.from_epsg(4326),
    b_left, b_bottom, b_right, b_top
)
print(f"  Crop region in WGS84: lon {lon_min:.2f}-{lon_max:.2f}, lat {lat_min:.2f}-{lat_max:.2f}")

# WorldCover tiles are 3°x3° named by SW corner (floor to multiple of 3)
# Format: S##N or N##E (lat floor / lon floor, multiples of 3)
lat_tile = int(np.floor(lat_min / 3) * 3)
lon_tile = int(np.floor(lon_min / 3) * 3)

lat_str = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{abs(lat_tile):02d}"
lon_str = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{abs(lon_tile):03d}"
wc_fname = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
wc_url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{wc_fname}"
wc_local = os.path.join(OUTPUT_DIR, wc_fname)

print(f"  WorldCover tile: {wc_fname}")
print(f"  URL: {wc_url}")

if not os.path.exists(wc_local):
    print(f"  Downloading... (this may take a moment)")
    try:
        urllib.request.urlretrieve(wc_url, wc_local)
        print(f"  Downloaded: {wc_local}")
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        # Try a slightly different lat tile
        lat_tile2 = int(np.floor(lat_max / 3) * 3)
        lon_tile2 = int(np.floor(lon_max / 3) * 3)
        lat_str2 = f"N{lat_tile2:02d}" if lat_tile2 >= 0 else f"S{abs(lat_tile2):02d}"
        lon_str2 = f"E{lon_tile2:03d}" if lon_tile2 >= 0 else f"W{abs(lon_tile2):03d}"
        wc_fname2 = f"ESA_WorldCover_10m_2021_v200_{lat_str2}{lon_str2}_Map.tif"
        wc_url2 = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{wc_fname2}"
        wc_local2 = os.path.join(OUTPUT_DIR, wc_fname2)
        print(f"  Trying alternate tile: {wc_fname2}")
        urllib.request.urlretrieve(wc_url2, wc_local2)
        wc_local = wc_local2
        print(f"  Downloaded: {wc_local}")
else:
    print(f"  Already exists: {wc_local}")

print("\n=== Step 5: Reprojecting WorldCover to match 10m grid ===")
# WorldCover is in EPSG:4326 at 10m (0.000090° per pixel)
# We need to reproject/crop to match our UTM 32643 10m grid

wc_out = os.path.join(OUTPUT_DIR, 'worldcover_v2_5class_10m_raw.tif')

with rasterio.open(wc_local) as wc_src:
    print(f"  WorldCover CRS: {wc_src.crs}, shape: {wc_src.width}x{wc_src.height}")
    
    # Reproject WorldCover to match our 10m UTM crop exactly
    dst_transform = transform_10m
    dst_crs = crs
    dst_width = WINDOW_PIXELS
    dst_height = WINDOW_PIXELS
    
    wc_reprojected = np.zeros((1, dst_height, dst_width), dtype=np.uint8)
    reproject(
        source=rasterio.band(wc_src, 1),
        destination=wc_reprojected,
        src_transform=wc_src.transform,
        src_crs=wc_src.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )

with rasterio.open(wc_out, 'w', driver='GTiff', count=1, dtype='uint8',
                   width=dst_width, height=dst_height, crs=dst_crs,
                   transform=dst_transform) as dst:
    dst.write(wc_reprojected)

print(f"  Saved raw WorldCover: {wc_out}")

print("\n=== Step 6: Remapping WorldCover classes to 1-5 encoding ===")
# ESA WorldCover 2021 class codes:
#  10 = Tree cover  -> class 3 (Vegetation)
#  20 = Shrubland   -> class 3 (Vegetation)
#  30 = Grassland   -> class 3 (Vegetation)
#  40 = Cropland    -> class 4 (Cropland)
#  50 = Built-up    -> class 2 (Built-up)
#  60 = Bare/sparse -> class 5 (Bare Land)
#  70 = Snow/ice    -> class 5 (Bare Land, treat as unvegetated)
#  80 = Water bodies-> class 1 (Water)
#  90 = Herbaceous wetland -> class 3 (Vegetation)
#  95 = Mangroves   -> class 3 (Vegetation)
# 100 = Moss/lichen -> class 3 (Vegetation)

wc_remap = {
    0: 0,   # No data -> ignore (set to 0, will be excluded from training)
    10: 3,  # Tree cover -> Vegetation
    20: 3,  # Shrubland -> Vegetation
    30: 3,  # Grassland -> Vegetation
    40: 4,  # Cropland -> Cropland
    50: 2,  # Built-up -> Built-up
    60: 5,  # Bare/sparse -> Bare Land
    70: 5,  # Snow/ice -> Bare Land (edge case)
    80: 1,  # Water -> Water
    90: 3,  # Herbaceous wetland -> Vegetation
    95: 3,  # Mangroves -> Vegetation
    100: 3, # Moss/lichen -> Vegetation
}

wc_data = wc_reprojected[0]
wc_mapped = np.zeros_like(wc_data, dtype=np.uint8)
for src_cls, dst_cls in wc_remap.items():
    wc_mapped[wc_data == src_cls] = dst_cls

# Check for unmapped values
unmapped = np.unique(wc_data[wc_mapped == 0])
unmapped = unmapped[unmapped != 0]
if len(unmapped) > 0:
    print(f"  WARNING: Unmapped WorldCover values: {unmapped}")

wc_final_out = os.path.join(OUTPUT_DIR, 'worldcover_v2_5class_10m_aligned.tif')
with rasterio.open(wc_final_out, 'w', driver='GTiff', count=1, dtype='uint8',
                   width=dst_width, height=dst_height, crs=dst_crs,
                   transform=dst_transform) as dst:
    dst.write(wc_mapped[np.newaxis, :, :])

print(f"  Saved: {wc_final_out}")

print("\n=== Step 7: Verification - Class pixel counts ===")
print(f"  Input region: lon {lon_min:.3f}-{lon_max:.3f}, lat {lat_min:.3f}-{lat_max:.3f}")
unique, counts = np.unique(wc_mapped, return_counts=True)
class_names = {0: 'NoData', 1: 'Water', 2: 'Built-up', 3: 'Vegetation', 4: 'Cropland', 5: 'Bare Land'}
total_valid = counts[unique > 0].sum() if len(unique[unique > 0]) > 0 else 1
for cls, cnt in zip(unique, counts):
    name = class_names.get(int(cls), f'Unknown({cls})')
    pct = cnt / total_valid * 100
    print(f"  Class {cls} ({name}): {cnt:,} pixels ({pct:.1f}%)")

print("\n=== All preprocessing complete ===")
print(f"Output files in: {OUTPUT_DIR}")
print(f"  10m 4-band: sentinel2_v2_4band_10m.tif ({WINDOW_PIXELS}x{WINDOW_PIXELS})")
print(f"  30m synthetic: sentinel2_v2_4band_synthetic_30m.tif ({w_30m}x{h_30m})")
print(f"  Labels: worldcover_v2_5class_10m_aligned.tif ({WINDOW_PIXELS}x{WINDOW_PIXELS})")
