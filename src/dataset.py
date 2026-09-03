import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import random

def compute_spectral_indices(img_4b):
    """
    Computes NDVI, NDWI, and BSI (simplified) from 4-band image (C, H, W) or (B, C, H, W).
    Bands: 0=Blue, 1=Green, 2=Red, 3=NIR.
    Returns: 7-band array/tensor stacked along channel dimension.
    """
    if isinstance(img_4b, torch.Tensor):
        if img_4b.dim() == 3:
            blue, green, red, nir = img_4b[0], img_4b[1], img_4b[2], img_4b[3]
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndwi = (green - nir) / (green + nir + 1e-6)
            bsi = ((red + green) - nir) / ((red + green) + nir + 1e-6)
            indices = torch.stack([ndvi, ndwi, bsi], dim=0)
            return torch.cat([img_4b, indices], dim=0)
        else:
            blue, green, red, nir = img_4b[:, 0], img_4b[:, 1], img_4b[:, 2], img_4b[:, 3]
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndwi = (green - nir) / (green + nir + 1e-6)
            bsi = ((red + green) - nir) / ((red + green) + nir + 1e-6)
            indices = torch.stack([ndvi, ndwi, bsi], dim=1)
            return torch.cat([img_4b, indices], dim=1)
    else:
        if img_4b.ndim == 3:
            blue, green, red, nir = img_4b[0], img_4b[1], img_4b[2], img_4b[3]
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndwi = (green - nir) / (green + nir + 1e-6)
            bsi = ((red + green) - nir) / ((red + green) + nir + 1e-6)
            indices = np.stack([ndvi, ndwi, bsi], axis=0)
            return np.concatenate([img_4b, indices], axis=0)
        else:
            blue, green, red, nir = img_4b[:, 0], img_4b[:, 1], img_4b[:, 2], img_4b[:, 3]
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndwi = (green - nir) / (green + nir + 1e-6)
            bsi = ((red + green) - nir) / ((red + green) + nir + 1e-6)
            indices = np.stack([ndvi, ndwi, bsi], axis=1)
            return np.concatenate([img_4b, indices], axis=1)

class SRDataset(Dataset):
    def __init__(self, data_dir, split='train', patch_size_10m=96, stride_10m=48,
                 oversample_bareland=True, bareland_threshold=0.10, bareland_factor=3,
                 use_indices=True):
        self.data_dir = data_dir
        self.split = split
        self.patch_size_10m = patch_size_10m
        self.stride_10m = stride_10m
        self.patch_size_30m = patch_size_10m // 3
        self.stride_30m = stride_10m // 3
        self.oversample_bareland = oversample_bareland if split == 'train' else False
        self.bareland_threshold = bareland_threshold
        self.bareland_factor = bareland_factor
        self.use_indices = use_indices
        
        self.path_10m = os.path.join(data_dir, 'sentinel2_4band_10m_crop.tif')
        self.path_30m = os.path.join(data_dir, 'sentinel2_4band_synthetic_30m.tif')
        self.path_label = os.path.join(data_dir, 'worldcover_5class_10m_aligned.tif')
        
        # Load data into memory (they are small enough)
        with rasterio.open(self.path_10m) as src:
            self.img_10m = src.read() # (4, H, W)
            
        with rasterio.open(self.path_30m) as src:
            self.img_30m = src.read() # (4, H/3, W/3)
            
        with rasterio.open(self.path_label) as src:
            self.label = src.read(1) # (H, W)
            
        # Grid splits - 5x5 blocks. 10m equivalent dimensions are roughly 1011x1008
        self.block_size_10m = 201
        self.block_size_30m = 67
        
        # Manually assigned checkerboard-like split (roughly 70/15/15)
        # 16 train, 4 val, 5 test out of 25 blocks
        self.split_assignment = {
            (0,0): 'train', (0,1): 'train', (0,2): 'val',   (0,3): 'train', (0,4): 'train',
            (1,0): 'train', (1,1): 'train', (1,2): 'train', (1,3): 'test',  (1,4): 'train',
            (2,0): 'train', (2,1): 'test',  (2,2): 'train', (2,3): 'train', (2,4): 'val',
            (3,0): 'val',   (3,1): 'train', (3,2): 'train', (3,3): 'train', (3,4): 'train',
            (4,0): 'train', (4,1): 'train', (4,2): 'test',  (4,3): 'val',   (4,4): 'test'
        }
        
        self.patches = self._extract_patches()
        
    def _extract_patches(self):
        patches = []
        # Iterate over blocks
        for bx in range(5):
            for by in range(5):
                if self.split_assignment[(bx, by)] != self.split:
                    continue
                    
                # Block bounds in 10m
                x_start_10m = bx * self.block_size_10m
                y_start_10m = by * self.block_size_10m
                
                # Sliding window within the block
                x_patches = (self.block_size_10m - self.patch_size_10m) // self.stride_10m + 1
                y_patches = (self.block_size_10m - self.patch_size_10m) // self.stride_10m + 1
                
                for px in range(x_patches):
                    for py in range(y_patches):
                        px_start = x_start_10m + px * self.stride_10m
                        py_start = y_start_10m + py * self.stride_10m
                        
                        px_start_30 = px_start // 3
                        py_start_30 = py_start // 3
                        
                        # Check Bare Land fraction (raw class 5)
                        label_sub = self.label[py_start:py_start+self.patch_size_10m, px_start:px_start+self.patch_size_10m]
                        bare_count = np.sum(label_sub == 5)
                        bare_frac = bare_count / float(self.patch_size_10m * self.patch_size_10m)
                        is_bare = bare_frac >= self.bareland_threshold
                        
                        patch_info = {
                            'x_10m': px_start, 'y_10m': py_start,
                            'x_30m': px_start_30, 'y_30m': py_start_30,
                            'is_bareland': is_bare
                        }
                        patches.append(patch_info)
                        
                        # Oversample if train and meets threshold
                        if self.split == 'train' and self.oversample_bareland and is_bare:
                            for _ in range(self.bareland_factor):
                                patches.append({
                                    'x_10m': px_start, 'y_10m': py_start,
                                    'x_30m': px_start_30, 'y_30m': py_start_30,
                                    'is_bareland': True,
                                    'is_duplicate': True
                                })
        return patches
        
    def __len__(self):
        return len(self.patches)
        
    def __getitem__(self, idx):
        patch = self.patches[idx]
        x10, y10 = patch['x_10m'], patch['y_10m']
        x30, y30 = patch['x_30m'], patch['y_30m']
        
        img_30m_patch = self.img_30m[:, y30:y30+self.patch_size_30m, x30:x30+self.patch_size_30m].astype(np.float32)
        label_patch = self.label[y10:y10+self.patch_size_10m, x10:x10+self.patch_size_10m].astype(np.int64)
        
        # Map classes 1-5 to 0-4
        label_patch = label_patch - 1
        
        # Normalization
        img_30m_patch = img_30m_patch / 10000.0
        
        # Augmentation (only for train)
        if self.split == 'train':
            # Geometric Augmentations
            if random.random() > 0.5:
                img_30m_patch = np.flip(img_30m_patch, axis=2).copy()
                label_patch = np.flip(label_patch, axis=1).copy()
            if random.random() > 0.5:
                img_30m_patch = np.flip(img_30m_patch, axis=1).copy()
                label_patch = np.flip(label_patch, axis=0).copy()
            rot = random.choice([0, 1, 2, 3])
            if rot > 0:
                img_30m_patch = np.rot90(img_30m_patch, k=rot, axes=(1, 2)).copy()
                label_patch = np.rot90(label_patch, k=rot, axes=(0, 1)).copy()
                
            # Extra Photometric Augmentation specifically for Bare Land patches
            if patch.get('is_bareland', False):
                # Random brightness jitter (+/- 15%)
                brightness = random.uniform(0.85, 1.15)
                img_30m_patch = img_30m_patch * brightness
                
                # Random contrast jitter (+/- 15%)
                contrast = random.uniform(0.85, 1.15)
                mean = img_30m_patch.mean(axis=(1, 2), keepdims=True)
                img_30m_patch = (img_30m_patch - mean) * contrast + mean
                
                # Random channel-wise gain (+/- 5%)
                gain = np.random.uniform(0.95, 1.05, size=(4, 1, 1)).astype(np.float32)
                img_30m_patch = img_30m_patch * gain
                
                # Gaussian noise
                noise = np.random.normal(0.0, 0.005, size=img_30m_patch.shape).astype(np.float32)
                img_30m_patch = np.clip(img_30m_patch + noise, 0.0, None)
                
        if self.use_indices:
            img_30m_patch = compute_spectral_indices(img_30m_patch)
            
        return torch.from_numpy(img_30m_patch), torch.from_numpy(label_patch)

def get_class_weights(dataset, mode='sqrt'):
    """Calculate class weights from the training set.
    mode='inverse': full inverse frequency weight = 1 / count
    mode='sqrt': square-root inverse frequency weight = 1 / sqrt(count)
    Ignores oversampled duplicate patches so weights stay based on original pixel counts.
    """
    class_counts = np.zeros(5, dtype=np.float64)

    for i in range(len(dataset)):
        if hasattr(dataset, 'patches') and i < len(dataset.patches) and dataset.patches[i].get('is_duplicate', False):
            continue
        _, label = dataset[i]
        label = label.numpy()
        for c in range(5):
            class_counts[c] += np.sum(label == c)
            
    if mode == 'sqrt':
        weights = 1.0 / (np.sqrt(class_counts) + 1e-6)
    else:
        total = np.sum(class_counts)
        weights = total / (class_counts + 1e-6)
        
    weights = weights / np.sum(weights) * 5.0 # Normalize weights
    return torch.from_numpy(weights).float()


class MultiTileDataset(torch.utils.data.Dataset):
    """Combines two SRDatasets: the primary dataset (train split) and a secondary
    tile dataset that is always treated as training data only.
    The secondary tile is loaded from a separate data directory (data_dir_v2).
    Val/test splits come exclusively from the primary dataset.
    """
    def __init__(self, data_dir_primary, data_dir_v2, patch_size_10m=96, stride_10m=48,
                 oversample_bareland=True, bareland_threshold=0.10, bareland_factor=3,
                 use_indices=True):
        # Primary train split (same spatial split as always)
        self.primary = SRDataset(
            data_dir_primary, split='train',
            patch_size_10m=patch_size_10m, stride_10m=stride_10m,
            oversample_bareland=oversample_bareland,
            bareland_threshold=bareland_threshold, bareland_factor=bareland_factor,
            use_indices=use_indices
        )
        # Secondary tile — always 'train', uses the v2 data directory
        # SRDataset only uses split='train' so it won't look for val/test blocks,
        # but v2 is a single tile (not a 5x5 grid) so we use a special loader below.
        self.secondary = _SecondTileDataset(
            data_dir_v2, patch_size_10m=patch_size_10m, stride_10m=stride_10m,
            oversample_bareland=oversample_bareland,
            bareland_threshold=bareland_threshold, bareland_factor=bareland_factor,
            use_indices=use_indices
        )
        print(f"MultiTileDataset: primary train patches={len(self.primary)}, secondary patches={len(self.secondary)}")
        self._primary_len = len(self.primary)
    
    def __len__(self):
        return len(self.primary) + len(self.secondary)
    
    def __getitem__(self, idx):
        if idx < self._primary_len:
            return self.primary[idx]
        else:
            return self.secondary[idx - self._primary_len]


class _SecondTileDataset(torch.utils.data.Dataset):
    """Loads patches from a single-tile v2 data directory.
    The directory contains:
      sentinel2_v2_4band_10m.tif   (H x W, typically 1000x1000)
      sentinel2_v2_4band_synthetic_30m.tif  (H/3 x W/3, typically 333x333)
      worldcover_v2_5class_10m_aligned.tif  (H x W, classes 1-5, 0 = nodata)
    Patches that are entirely nodata (label 0) are excluded.
    """
    def __init__(self, data_dir, patch_size_10m=96, stride_10m=48,
                 oversample_bareland=True, bareland_threshold=0.10, bareland_factor=3,
                 use_indices=True):
        self.data_dir = data_dir
        self.patch_size_10m = patch_size_10m
        self.stride_10m = stride_10m
        self.patch_size_30m = patch_size_10m // 3
        self.oversample_bareland = oversample_bareland
        self.bareland_threshold = bareland_threshold
        self.bareland_factor = bareland_factor
        self.use_indices = use_indices

        self.path_10m = os.path.join(data_dir, 'sentinel2_v2_4band_10m.tif')
        self.path_30m = os.path.join(data_dir, 'sentinel2_v2_4band_synthetic_30m.tif')
        self.path_label = os.path.join(data_dir, 'worldcover_v2_5class_10m_aligned.tif')

        with rasterio.open(self.path_10m) as src:
            self.img_10m = src.read()  # (4, H, W)
        with rasterio.open(self.path_30m) as src:
            self.img_30m = src.read()  # (4, H/3, W/3)
        with rasterio.open(self.path_label) as src:
            self.label = src.read(1)  # (H, W), classes 1-5, 0=nodata

        self.patches = self._extract_patches()
        print(f"  SecondTileDataset: {len(self.patches)} patches from {data_dir}")

    def _extract_patches(self):
        patches = []
        H, W = self.label.shape
        x_patches = (W - self.patch_size_10m) // self.stride_10m + 1
        y_patches = (H - self.patch_size_10m) // self.stride_10m + 1

        for py in range(y_patches):
            for px in range(x_patches):
                px_start = px * self.stride_10m
                py_start = py * self.stride_10m

                label_sub = self.label[py_start:py_start+self.patch_size_10m,
                                       px_start:px_start+self.patch_size_10m]

                # Skip patches with any nodata (class 0) or entirely invalid
                nodata_frac = np.sum(label_sub == 0) / float(self.patch_size_10m ** 2)
                if nodata_frac > 0.05:  # skip if more than 5% nodata
                    continue

                bare_count = np.sum(label_sub == 5)
                bare_frac = bare_count / float(self.patch_size_10m ** 2)
                is_bare = bare_frac >= self.bareland_threshold

                patch_info = {
                    'x_10m': px_start, 'y_10m': py_start,
                    'x_30m': px_start // 3, 'y_30m': py_start // 3,
                    'is_bareland': is_bare
                }
                patches.append(patch_info)

                # Oversample Bare Land patches
                if self.oversample_bareland and is_bare:
                    for _ in range(self.bareland_factor):
                        patches.append({
                            'x_10m': px_start, 'y_10m': py_start,
                            'x_30m': px_start // 3, 'y_30m': py_start // 3,
                            'is_bareland': True,
                            'is_duplicate': True
                        })
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        x10, y10 = patch['x_10m'], patch['y_10m']
        x30, y30 = patch['x_30m'], patch['y_30m']

        img_30m_patch = self.img_30m[:, y30:y30+self.patch_size_30m,
                                      x30:x30+self.patch_size_30m].astype(np.float32)
        label_patch = self.label[y10:y10+self.patch_size_10m,
                                  x10:x10+self.patch_size_10m].astype(np.int64)
        # Remap: classes 1-5 -> 0-4; nodata (0) -> 0 (Water, handled by low count)
        label_patch = np.clip(label_patch - 1, 0, 4)

        img_30m_patch = img_30m_patch / 10000.0

        # Geometric augmentations (always for training)
        if random.random() > 0.5:
            img_30m_patch = np.flip(img_30m_patch, axis=2).copy()
            label_patch = np.flip(label_patch, axis=1).copy()
        if random.random() > 0.5:
            img_30m_patch = np.flip(img_30m_patch, axis=1).copy()
            label_patch = np.flip(label_patch, axis=0).copy()
        rot = random.choice([0, 1, 2, 3])
        if rot > 0:
            img_30m_patch = np.rot90(img_30m_patch, k=rot, axes=(1, 2)).copy()
            label_patch = np.rot90(label_patch, k=rot, axes=(0, 1)).copy()

        # Extra Photometric augmentation for Bare Land patches
        if patch.get('is_bareland', False):
            brightness = random.uniform(0.85, 1.15)
            img_30m_patch = img_30m_patch * brightness
            contrast = random.uniform(0.85, 1.15)
            mean = img_30m_patch.mean(axis=(1, 2), keepdims=True)
            img_30m_patch = (img_30m_patch - mean) * contrast + mean
            gain = np.random.uniform(0.95, 1.05, size=(4, 1, 1)).astype(np.float32)
            img_30m_patch = img_30m_patch * gain
            noise = np.random.normal(0.0, 0.005, size=img_30m_patch.shape).astype(np.float32)
            img_30m_patch = np.clip(img_30m_patch + noise, 0.0, None)

        if self.use_indices:
            img_30m_patch = compute_spectral_indices(img_30m_patch)

        return torch.from_numpy(img_30m_patch), torch.from_numpy(label_patch)


def get_class_weights_combined(multi_dataset, mode='sqrt'):
    """Compute class weights from the combined MultiTileDataset.
    Counts pixels from primary (non-duplicate) patches AND secondary (non-duplicate) patches.
    This correctly accounts for all original training pixels across both tiles.
    """
    class_counts = np.zeros(5, dtype=np.float64)

    def count_from_dataset(ds):
        for i in range(len(ds.patches)):
            if ds.patches[i].get('is_duplicate', False):
                continue
            # Read directly from arrays (fast path, no __getitem__ augmentation)
            patch = ds.patches[i]
            x10, y10 = patch['x_10m'], patch['y_10m']
            label_patch = ds.label[y10:y10+ds.patch_size_10m, x10:x10+ds.patch_size_10m]
            # Map WorldCover raw 1-5 to 0-4
            if hasattr(ds, 'img_10m'):  # SRDataset: labels are raw 1-5
                mapped = np.clip(label_patch.astype(np.int64) - 1, 0, 4)
            else:  # already mapped
                mapped = label_patch.astype(np.int64)
            for c in range(5):
                class_counts[c] += np.sum(mapped == c)

    count_from_dataset(multi_dataset.primary)
    count_from_dataset(multi_dataset.secondary)

    print(f"Combined class pixel counts (original non-duplicate patches):")
    names = ['Water', 'Built-up', 'Vegetation', 'Cropland', 'Bare Land']
    for c, (name, cnt) in enumerate(zip(names, class_counts)):
        print(f"  Class {c} ({name}): {int(cnt):,}")

    if mode == 'sqrt':
        weights = 1.0 / (np.sqrt(class_counts) + 1e-6)
    else:
        total = np.sum(class_counts)
        weights = total / (class_counts + 1e-6)

    weights = weights / np.sum(weights) * 5.0
    return torch.from_numpy(weights).float()



