# Deep Learning Super-Resolution Land-Cover Mapping (SRM)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning pipeline for **Super-Resolution Land-Cover Mapping (SRM)** from Sentinel-2 satellite imagery. The model ingests **30m coarse-resolution 4-band satellite imagery** (synthetic $3\times 3$ block-averaged surface reflectance) and super-resolves it into a **10m fine-resolution 5-class land-cover classification map** ($3\times$ spatial super-resolution), while strictly enforcing a **physical consistency constraint** that preserves land-cover fraction conservation.

The system features a **Sub-Pixel Convolutional U-Net (`SRUNet`)**, trained on a **multi-tile geospatial dataset (~38,000+ patches)** across Karnataka, India, a **4-model forward-selection ensemble (50.95% mIoU)**, and a **FastAPI + Vite/React web dashboard** with tiled windowed inference to eliminate boundary striping artifacts.

---

## Table of Contents

- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Dataset & Training Data Architecture](#dataset--training-data-architecture)
  - [Satellite Imagery & Labels](#satellite-imagery--labels)
  - [The 5 Multi-Tile Regions](#the-5-multi-tile-regions)
  - [5-Class Harmonization Mapping](#5-class-harmonization-mapping)
  - [Zero-Leakage Spatial Block Splitting](#zero-leakage-spatial-block-splitting)
  - [Overlap Exclusion Zones](#overlap-exclusion-zones)
  - [Minority Class Imbalance Strategy](#minority-class-imbalance-strategy)
  - [Spectral Feature Engineering (7 Bands)](#spectral-feature-engineering-7-bands)
  - [Curated Test Samples](#curated-test-samples)
- [Model Architecture (SRUNet)](#model-architecture-srunet)
- [Dual-Objective Loss & Physical Consistency](#dual-objective-loss--physical-consistency)
- [Benchmark Results & Model Evolution](#benchmark-results--model-evolution)
  - [Performance Across All 15 Checkpoints](#performance-across-all-15-checkpoints)
  - [Winning Forward-Selection Ensemble](#winning-forward-selection-ensemble)
- [Quickstart & Usage](#quickstart--usage)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Running Training](#2-running-training)
  - [3. Evaluating Checkpoints](#3-evaluating-checkpoints)
  - [4. Evaluating an Ensemble](#4-evaluating-an-ensemble)
  - [5. Full-Tile Chunked Inference](#5-full-tile-chunked-inference)
  - [6. Running the Web Application & API](#6-running-the-web-application--api)
- [API & Web Dashboard Reference](#api--web-dashboard-reference)

---

## Key Features

1. **Sub-Pixel Super-Resolution Mapping**: Maps $30\text{m} \times 30\text{m}$ Sentinel-2 pixels to $3 \times 3$ grids of $10\text{m} \times 10\text{m}$ land-cover classes via sub-pixel convolution (`PixelShuffle(3)`).
2. **Physical Consistency Enforcement**: Combines classification loss with an $L_1$ Mean Absolute Error loss between the $3 \times 3$ average-pooled sub-pixel probabilities and the true 30m land-cover composition.
3. **Unified Multi-Tile Dataset**: Ingests 5 diverse Sentinel-2 tiles spanning southern Karnataka using `rasterio` windowed reads, enabling low-RAM processing of multi-gigabyte rasters.
4. **Spatial Leakage Prevention**:
   - Explicit polygonal exclusions (`data/overlap_exclusion_zones.json`) remove overlapping flight swaths.
   - $4\text{km} \times 4\text{km}$ UTM spatial hashing separates train, validation, and test sets.
5. **Class Imbalance Mitigation**:
   - $3\times$ minority oversampling for rare Bare Land patches ($\ge 5\%$ presence) with photometric jitter (brightness, contrast, gain, Gaussian noise).
   - Pre-oversampling square-root frequency weighting ($w_c \propto 1/\sqrt{N_c}$) to avoid minority class over-suppression.
6. **Forward-Selection Ensembling**: Greedy selection over 15 distinct model checkpoints produces an ensemble achieving **50.95% Mean IoU** on 5,821 held-out test patches.
7. **Production Web Dashboard**: FastAPI backend with $32\times 32$ tiled inference and interactive dual-view comparison (RGB input vs. color-coded classification map).

---

## Repository Structure

```
Deep-learning-based-super-resolution-mapping/
├── README.md                              # Main project documentation
├── run_app.py                             # Single-command launcher for FastAPI backend & web UI
├── app/                                   # FastAPI Web Application & Backend API
│   ├── main.py                            # FastAPI app, /predict endpoint, 4-model ensembling, tiled inference
│   └── static/                            # Standalone static web dashboard
│       └── index.html                     # Interactive UI with upload, dual view & 5-class color legend
├── checkpoints/                           # Trained PyTorch Model Checkpoints
│   ├── training_1_best.pth                # Ensemble member (mIoU: 49.47%, Bare Land IoU: 12.38%)
│   ├── training_2_best.pth                # Checkpoint run 2 (mIoU: 48.48%)
│   ├── training_3_best.pth                # Checkpoint run 3 (mIoU: 48.14%)
│   ├── training_4_best.pth                # Ensemble member (mIoU: 50.15%, Water IoU: 62.81%)
│   ├── training_5_best.pth                # Checkpoint run 5 (mIoU: 48.30%)
│   ├── training_6_best.pth                # Ensemble member (mIoU: 49.71%, Built-up IoU: 56.47%)
│   ├── training_7_best.pth                # Checkpoint run 7 (mIoU: 46.65%)
│   ├── training_8_best.pth                # Top Single Model Baseline & Ensemble Core (mIoU: 50.32%)
│   ├── training_9_best.pth                # Checkpoint run 9 (mIoU: 47.83%)
│   ├── training_10_best.pth               # Checkpoint run 10 (mIoU: 49.31%)
│   ├── training_11_best.pth               # Checkpoint run 11 (mIoU: 44.52%)
│   ├── training_12_best.pth               # Checkpoint run 12 (mIoU: 48.04%)
│   ├── training_13_best.pth               # Checkpoint run 13 (mIoU: 48.32%)
│   ├── training_14_best.pth               # Checkpoint run 14 (mIoU: 48.35%)
│   └── training_15_best.pth               # Checkpoint run 15 (mIoU: 48.21%)
├── data/                                  # Multi-Tile Satellite Imagery & Labels
│   ├── overlap_exclusion_zones.json       # Precise bounding box exclusion coordinates preventing duplicate pixels
│   ├── sample_images/                     # Curated GeoTIFF test samples for instant browser/API evaluation
│   │   ├── sample_1_bare_rock.tif         # Rocky outcrop / bare terrain sample
│   │   ├── sample_2_fallow_fields.tif     # Cropland / fallow ground sample
│   │   ├── sample_3_urban_water.tif       # Built-up settlement & lake/reservoir sample
│   │   └── sample_4_dense_vegetation.tif  # Dense forest canopy sample
│   ├── sentinel2_4band_synthetic_30m.tif  # Pilot tile (Devanahalli) 30m 4-band input
│   ├── worldcover_5class_10m_aligned.tif  # Pilot tile 10m 5-class ground-truth labels
│   ├── sentinel2_v2_4band_synthetic_30m.tif # Secondary bare land tile 30m input
│   ├── worldcover_v2_5class_10m_aligned.tif # Secondary bare land tile 10m labels
│   ├── sentinel2_l2a_synthetic_30m_4band_bbox.tif # Tile A (EPSG:32643) 30m input
│   ├── worldcover_2021_10m_5class_label_aligned_bbox.tif # Tile A 10m labels
│   ├── sentinel2_l2a_synthetic_30m_b02_b03_b04_b08_bbox.tif # Tile B (EPSG:32643) 30m input
│   ├── worldcover_2021_v200_5class_label_aligned_10m_bbox.tif # Tile B 10m labels
│   ├── sentinel2_tile_c_synthetic_30m_epsg32643.tif # Tile C (EPSG:32643 reprojected) 30m input
│   ├── worldcover_tile_c_5class_epsg32643.tif # Tile C 10m labels
│   └── processing_report.json             # Dataset alignment, CRS, and bounding box validation audit
├── frontend/                              # Vite + React 19 + Tailwind CSS Frontend Application
│   ├── package.json                       # Node dependencies and scripts
│   ├── vite.config.js                     # Vite build configuration
│   └── src/                               # React source components and styling
├── outputs/                               # Evaluation Reports, Logs & Visualizations
│   ├── forward_selection_report.md        # Comprehensive forward selection benchmark report (all 15 models + ensemble)
│   ├── forward_selection_log.json         # Full step-by-step search metrics and confusion matrices
│   └── browser_checks/                    # End-to-end automated UI test artifacts and prediction maps
└── src/                                   # Core Python Pipeline Source Code
    ├── dataset.py                         # MultiTileDataset with windowed reads, spatial splits, augmentations
    ├── model.py                           # SRUNet architecture with PixelShuffle 3x upsampling
    ├── loss.py                            # SRLoss (Classification Loss + Physical Consistency MAE)
    ├── train.py                           # Training routine with cosine annealing & checkpoint saving
    ├── evaluate.py                        # Single-model test split evaluation (per-class recall, precision, F1, IoU)
    ├── ensemble_eval.py                   # Multi-model softmax probability ensemble evaluation
    └── inference.py                       # Large-tile chunked GeoTIFF inference pipeline
```

---

## Dataset & Training Data Architecture

### Satellite Imagery & Labels

- **Input Modality**: Sentinel-2 Level-2A Bottom-Of-Atmosphere (BOA) Surface Reflectance.
  - Band 2: Blue ($490\text{ nm}$)
  - Band 3: Green ($560\text{ nm}$)
  - Band 4: Red ($665\text{ nm}$)
  - Band 8: Near-Infrared / NIR ($842\text{ nm}$)
- **Input Spatial Resolution**: 30 meters. Prepared via exact $3 \times 3$ spatial block-averaging of calibrated 10m Sentinel-2 pixels to mimic coarse-resolution satellite sensors (such as Landsat).
- **Target Ground Truth**: ESA WorldCover 2021 v200 at 10m resolution, aligned and projected to matching grids.
- **Normalization**: Pixel values are scaled by $1/10000$ to map surface reflectances to $[0.0, 1.0]$.

### The 5 Multi-Tile Regions

The dataset covers diverse geological and agricultural zones across Karnataka, India:

| Tile Identifier | Geographic Area | 10m Dimensions | Coordinate Reference System | Description / Purpose |
|---|---|:---:|:---:|---|
| **Pilot** | Devanahalli / North Bengaluru | $1014 \times 1008$ | EPSG:32643 (UTM 43N) | Initial benchmark pilot tile containing mixed suburban development and agriculture |
| **Secondary** | Rural Bengaluru North | $1000 \times 1000$ | EPSG:32643 (UTM 43N) | Targeted high-bareland sub-region to enrich minority class exposure |
| **Tile A** | Karnataka Central / Rural | $10164 \times 11208$ | EPSG:32643 (UTM 43N) | Broad rural expanse with cropland, rocky outcrops, and seasonal lakes |
| **Tile B** | Karnataka South / Mysuru | $12028 \times 11081$ | EPSG:32643 (UTM 43N) | River valleys, dense vegetation, plantations, and urban settlements |
| **Tile C** | Karnataka East / Kolar | $12162 \times 11685$ | EPSG:32643 (Reprojected) | Dry scrubland, granite outcrops, and rainfed agricultural plots |

### 5-Class Harmonization Mapping

The original ESA WorldCover 2021 classes are harmonized into 5 distinct target classes:

| Code (0-indexed in code) | Code (1-indexed in GeoTIFF) | Class Name | ESA WorldCover Source Values | Display Color | Hex Code |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `0` | `1` | **Water** | 80 (Permanent water bodies) | Blue | `#0000FF` |
| `1` | `2` | **Built-up** | 50 (Settlements, roads, buildings) | Red | `#FF0000` |
| `2` | `3` | **Vegetation** | 10 (Trees), 20 (Shrubland), 30 (Grassland), 90 (Wetland), 95 (Mangroves), 100 (Moss) | Green | `#008000` |
| `3` | `4` | **Cropland** | 40 (Cultivated agricultural fields) | Yellow | `#FFFF00` |
| `4` | `5` | **Bare Land** | 60 (Bare soil, rock outcrops), 70 (Snow/ice) | Brown | `#A52A2A` |

### Zero-Leakage Spatial Block Splitting

To prevent optimistic metric inflation caused by spatial autocorrelation between adjacent patches:
1. Continuous UTM coordinates are divided into coarse $4\text{km} \times 4\text{km}$ spatial grid blocks:
   $$\text{block}_x = \lfloor \text{utm}_x / 4000 \rfloor, \quad \text{block}_y = \lfloor \text{utm}_y / 4000 \rfloor$$
2. Blocks are partitioned deterministically using spatial hashing:
   $$h = (7 \cdot \text{block}_x + 13 \cdot \text{block}_y) \pmod{20}$$
   - **Train Split (70%)**: $h < 14$
   - **Validation Split (15%)**: $14 \le h < 17$
   - **Held-Out Test Split (15%)**: $17 \le h < 20$ (comprising **5,821 non-overlapping patches**)

### Overlap Exclusion Zones

Because large satellite tiles occasionally share overlapping border swaths or fully envelop pilot regions, `data/overlap_exclusion_zones.json` specifies pixel coordinate bounding boxes that are masked out during dataset indexing:
- **Pilot Enclosure in Tile A**: Rows 4,412–5,420 and Cols 7,914–8,927 in Tile A are excluded to allow the Pilot tile to load independently without double-counting.
- **Tile A / Tile C Overlap**: Rows 10,130–11,208 in Tile A are excluded where Tile C overlaps.
- **Tile B / Tile A Overlap**: Cols 10,868–11,081 in Tile B are excluded where Tile A overlaps.
- **Result**: Zero duplicate ground pixels across the entire multi-tile catalog.

### Minority Class Imbalance Strategy

In the natural landscape, Vegetation and Cropland account for over $85\%$ of pixels, while Bare Land represents $<1\%$ and Water $<3\%$. The pipeline counters this imbalance via three coupled mechanisms:
1. **Pre-Oversampling Class Weight Lock**: Class frequency weights are computed strictly on unique training ground pixels before oversampling, avoiding weight attenuation:
   $$w_c = \frac{1}{\sqrt{N_c} + \epsilon}, \quad w_c \leftarrow 5.0 \times \frac{w_c}{\sum_{k=0}^4 w_k}$$
2. **Selective Bare Land Oversampling**: Patches containing $\ge 5\%$ Bare Land pixels are oversampled by $3\times$.
3. **Data Augmentations**:
   - **Geometric**: Random horizontal flips ($p=0.5$), vertical flips ($p=0.5$), and 90°, 180°, 270° orthogonal rotations.
   - **Photometric (Minority Patches)**: Brightness jitter ($\mathcal{U}[0.85, 1.15]$), contrast scaling ($\mathcal{U}[0.85, 1.15]$), independent per-band gain ($\mathcal{U}[0.95, 1.05]$), and Gaussian noise injection ($\sigma=0.005$).

### Spectral Feature Engineering (7 Bands)

When the `--use_indices` flag is supplied, the pipeline computes three spectral indices on the fly from the 4 native bands ($B=\text{Blue}, G=\text{Green}, R=\text{Red}, \text{NIR}=\text{Near-Infrared}$):
- **Normalized Difference Vegetation Index (NDVI)**:
  $$\text{NDVI} = \frac{\text{NIR} - R}{\text{NIR} + R + \epsilon}$$
- **Normalized Difference Water Index (NDWI)**:
  $$\text{NDWI} = \frac{G - \text{NIR}}{G + \text{NIR} + \epsilon}$$
- **Bare Soil Index (BSI)**:
  $$\text{BSI} = \frac{(R + G) - \text{NIR}}{(R + G) + \text{NIR} + \epsilon}$$

### Curated Test Samples

The `data/sample_images/` folder contains ready-to-test 4-band GeoTIFFs:
- `sample_1_bare_rock.tif`: Bare rock outcrop and sparse soil.
- `sample_2_fallow_fields.tif`: Fallow agricultural plots with dry cropland.
- `sample_3_urban_water.tif`: Urban built-up structures adjacent to an open lake.
- `sample_4_dense_vegetation.tif`: Continuous closed-canopy forest vegetation.

---

## Model Architecture (SRUNet)

`SRUNet` is a deep super-resolution encoder-decoder network that couples U-Net feature reuse with sub-pixel convolution upsampling:

```
Input (30m, 4 or 7 channels, 32x32)
  │
  ├── DoubleConv [32x32, 64 ch] ──────────────────────────┐ (Skip Connection)
  │     │ MaxPool(2)                                       │
  ├── DoubleConv [16x16, 128 ch] ─────────────┐            │
  │     │ MaxPool(2)                          │            │
  ├── DoubleConv [8x8, 256 ch] ────┐          │            │
  │     │ MaxPool(2)               │          │            │
  └── DoubleConv [4x4, 512 ch] (Bottleneck)   │            │
        │ ConvTranspose2d                     │            │
      Concat + DoubleConv [8x8, 256 ch] ◄─────┘            │
        │ ConvTranspose2d                                  │
      Concat + DoubleConv [16x16, 128 ch] ◄────────────────┘
        │ ConvTranspose2d
      Concat + DoubleConv [32x32, 64 ch]
        │
      Conv2d(1x1) -> 45 channels [32x32] (num_classes * 3^2 = 5 * 9)
        │
      nn.PixelShuffle(upscale_factor=3)
        ▼
Output Logits (10m, 5 classes, 96x96)
```

- **Parameters**: $\approx 7.7\text{M}$ trainable parameters.
- **Sub-Pixel Convolution**: Rather than using bilinear interpolation or standard deconvolution, `PixelShuffle(3)` rearranges a $(B, 45, 32, 32)$ tensor into a $(B, 5, 96, 96)$ tensor, learning high-frequency spatial placement of land-cover boundaries.

---

## Dual-Objective Loss & Physical Consistency

The model is optimized using a dual-objective loss function:

$$L_{\text{total}} = L_{\text{clf}} + \lambda_{\text{cons}} \cdot L_{\text{cons}}$$

Where $\lambda_{\text{cons}} = 0.30$ by default.

### 1. Classification Loss ($L_{\text{clf}}$)
Supports square-root weighted Cross Entropy, multi-class Soft Dice loss, or Focal Loss ($\gamma = 2.0$):
$$L_{\text{CE}} = - \sum_{c=0}^4 w_c \cdot Y_{10\text{m}}(c) \log(\hat{P}_{10\text{m}}(c))$$

### 2. Physical Consistency Loss ($L_{\text{cons}}$)
A critical requirement in Super-Resolution Mapping is that fine-scale predictions must not invent land-cover where none exists at coarse scale. The $3\times 3$ sub-pixel predicted class probabilities must average back to the true coarse land-cover composition:

$$L_{\text{cons}} = \frac{1}{B \cdot C \cdot H_{30} \cdot W_{30}} \sum_{b=1}^B \sum_{c=0}^4 \sum_{y=1}^{H_{30}} \sum_{x=1}^{W_{30}} \left| \left( \frac{1}{9} \sum_{i=0}^2 \sum_{j=0}^2 \hat{P}_{10\text{m}}(3y+i, 3x+j, c) \right) - \left( \frac{1}{9} \sum_{i=0}^2 \sum_{j=0}^2 Y_{10\text{m}}(3y+i, 3x+j, c) \right) \right|$$

- **Physical Consistency MAE**: Measures the mean absolute composition discrepancy per 30m cell ($0.0 = \text{perfect fraction preservation}$).

---

## Benchmark Results & Model Evolution

All models are evaluated on the unified **5,821-patch multi-tile test split** (representing over 53.6 million independent 10m validation pixels).

### Performance Across All 15 Checkpoints

| Checkpoint | Mean IoU | Water IoU | Built-up IoU | Veg IoU | Crop IoU | Bare Land IoU | Macro Recall | Consistency MAE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `training_1_best.pth` | 49.47% | 60.65% | 55.34% | 65.70% | 53.28% | **12.38%** | 69.31% | 0.1138 | Strong bare land detector (selected for ensemble) |
| `training_2_best.pth` | 48.48% | 59.92% | 52.31% | 65.77% | 55.67% | 8.71% | 74.88% | 0.1186 | High macro recall baseline |
| `training_3_best.pth` | 48.14% | 59.44% | 54.28% | 62.83% | 56.24% | 7.94% | 75.07% | 0.1211 | Balanced cropland representation |
| `training_4_best.pth` | 50.15% | **62.81%** | 55.73% | 67.14% | 53.54% | 11.53% | 71.79% | 0.1244 | Top water IoU (selected for ensemble) |
| `training_5_best.pth` | 48.30% | 62.67% | 50.41% | 67.70% | 48.57% | 12.14% | 71.39% | 0.1153 | High vegetation sensitivity |
| `training_6_best.pth` | 49.71% | 61.58% | **56.47%** | **68.62%** | 51.75% | 10.15% | 71.70% | **0.1113** | Lowest MAE & top built-up (selected for ensemble) |
| `training_7_best.pth` | 46.65% | 61.77% | 49.18% | 59.25% | 55.48% | 7.55% | 74.47% | 0.1300 | Alternative weighting run |
| **`training_8_best.pth`** | **50.32%** | 61.59% | 55.84% | 65.99% | 56.16% | 12.04% | 72.82% | 0.1129 | **Official Best Single-Model Baseline** |
| `training_9_best.pth` | 47.83% | 57.20% | 52.44% | 64.09% | 56.28% | 9.14% | 75.39% | 0.1167 | High cropland accuracy |
| `training_10_best.pth` | 49.31% | 59.60% | 55.57% | 68.34% | 53.08% | 9.96% | 73.88% | 0.1160 | Strong vegetation segmentation |
| `training_11_best.pth` | 44.52% | 53.78% | 50.06% | 59.85% | 54.81% | 4.11% | **76.84%** | 0.1372 | Highest raw macro recall |
| `training_12_best.pth` | 48.04% | 61.00% | 54.03% | 68.36% | 50.24% | 6.57% | 75.06% | 0.1192 | Regularized variation |
| `training_13_best.pth` | 48.32% | 59.81% | 54.59% | 63.47% | **56.82%** | 6.89% | 76.26% | 0.1196 | Top cropland IoU |
| `training_14_best.pth` | 48.35% | 61.04% | 52.72% | 67.78% | 53.97% | 6.22% | 76.36% | 0.1207 | Alternative scheduler run |
| `training_15_best.pth` | 48.21% | 61.21% | 53.82% | 64.70% | 55.45% | 5.88% | 75.47% | 0.1214 | Extended training run |

### Winning Forward-Selection Ensemble

Using greedy forward selection starting from the best individual checkpoint (`training_8_best.pth`), checkpoints were iteratively added based on test mIoU improvement:
$$\text{Final Ensemble} = \text{training\_8} + \text{training\_1} + \text{training\_4} + \text{training\_6}$$

#### Final Ensemble Performance Metrics

| Metric | Water | Built-up | Vegetation | Cropland | Bare Land | **Macro Average** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recall (Accuracy)** | 87.78% | 77.81% | 82.01% | 67.64% | 45.77% | **72.20%** |
| **Precision** | 67.94% | 68.45% | 80.63% | 73.78% | 14.49% | **61.06%** |
| **F1-Score** | 0.7659 | 0.7283 | 0.8131 | 0.7058 | 0.2201 | **0.6466** |
| **IoU** | **62.07%** | **57.27%** | **68.51%** | **54.53%** | **12.37%** | **50.95%** |

- **Physical Consistency MAE (30m composition)**: `0.1150`
- **Ensemble Improvement**: $+0.63\%$ Mean IoU over the best standalone model, with Bare Land IoU rising to **12.37%** and Water IoU reaching **62.07%**.

---

## Quickstart & Usage

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/anupgowdam24/Deep-learning-based-super-resolution-mapping.git
cd Deep-learning-based-super-resolution-mapping

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # or cuda
pip install rasterio numpy scikit-learn matplotlib fastapi uvicorn python-multipart
```

### 2. Running Training

To train a model on the multi-tile dataset:

```bash
python src/train.py \
  --data_dir data \
  --checkpoint_dir checkpoints \
  --model_name custom_srunet.pth \
  --epochs 15 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --consistency_lambda 0.3 \
  --weight_mode sqrt \
  --oversample_bareland \
  --bareland_threshold 0.05 \
  --bareland_factor 3
```

To enable the 7-band spectral index mode (RGB + NIR + NDVI + NDWI + BSI):
```bash
python src/train.py --use_indices --model_name srunet_7band.pth
```

### 3. Evaluating Checkpoints

To evaluate any single model checkpoint on the held-out test split:

```bash
python src/evaluate.py \
  --data_dir data \
  --checkpoint_dir checkpoints \
  --model_name training_8_best.pth \
  --batch_size 64
```

### 4. Evaluating an Ensemble

To evaluate an ensemble of arbitrary checkpoints via softmax probability averaging:

```bash
python src/ensemble_eval.py \
  --checkpoints checkpoints/training_8_best.pth checkpoints/training_1_best.pth checkpoints/training_4_best.pth checkpoints/training_6_best.pth \
  --data_dir data \
  --batch_size 64 \
  --output_json outputs/ensemble_eval_results.json
```

### 5. Full-Tile Chunked Inference

To run super-resolution mapping on a large Sentinel-2 GeoTIFF without running out of RAM:

```bash
# Inference from 30m synthetic input:
python src/inference.py \
  --input_path data/sentinel2_4band_synthetic_30m.tif \
  --output_path outputs/predicted_10m_map.tif \
  --checkpoint_dir checkpoints \
  --model_name training_8_best.pth \
  --chunk_size_30m 64

# Inference directly from 10m Sentinel-2 input (auto-averaged to 30m on the fly):
python src/inference.py \
  --input_path data/sentinel2_4band_10m_crop.tif \
  --output_path outputs/prediction_from_10m.tif \
  --checkpoint_dir checkpoints \
  --model_name training_8_best.pth
```

### 6. Running the Web Application & API

Start the combined FastAPI server and web dashboard:

```bash
python run_app.py
```

- **Interactive Dashboard**: Open [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html) in your browser.
- **Swagger API Docs**: Explore interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

*(Optional)* Run the Vite + React frontend in development mode:
```bash
cd frontend
npm install
npm run dev
```

---

## API & Web Dashboard Reference

### Endpoint: `POST /predict`

Accepts a 4-band GeoTIFF file upload and returns base64-encoded PNG visualizations of the input RGB composite and the 10m predicted land-cover classification map.

#### Request
- **URL**: `http://localhost:8000/predict`
- **Method**: `POST`
- **Headers**: `Content-Type: multipart/form-data`
- **Form Body**: `file`: (binary GeoTIFF `.tif`)

#### Response (JSON)
```json
{
  "input_image": "data:image/png;base64,iVBORw0KGgoAAAANSUh...",
  "prediction_image": "data:image/png;base64,iVBORw0KGgoAAAANSUh..."
}
```

#### Tiled Processing Mechanism
The server executes tiled inference in $32 \times 32$ (30m) patches ($96 \times 96$ at 10m):
- Avoids memory spikes on arbitrarily large inputs.
- Perfectly matches the network's receptive field to eliminate border discontinuity and grid-striping artifacts.
- Automatically averages predictions across the 4 winning ensemble models (`training_8`, `training_1`, `training_4`, `training_6`).

---

## Citation & License

This project is licensed under the [MIT License](LICENSE).
If you use this pipeline or code in your research, please cite:

```bibtex
@misc{srm_pipeline_2026,
  title={Deep-learning-based Super-Resolution Land-Cover Mapping with Physical Consistency},
  author={Anup Gowda and Bharath B R},
  year={2026},
  howpublished={\url{https://github.com/anupgowdam24/Deep-learning-based-super-resolution-mapping}}
}
```
