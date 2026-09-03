# SIH26142 Deep Learning Super-Resolution Mapping (SRM) Pipeline

This repository contains the solution for the SIH26142 SRM pipeline.
The goal is to build a deep learning model that super-resolves a 30m-resolution 4-band Sentinel-2 image into a 10m-resolution land-cover classification map (5 classes). 
A critical requirement is that the 10m predictions average back physically consistently to the 30m true composition.

## Project Structure
```
project/
  data/                 <- input TIFF files (symlinked)
  src/                  <- python scripts (dataset, model, loss, train, evaluate, inference)
  checkpoints/          <- model weights saved here
  outputs/              <- evaluation metrics, visualizations, and full tile prediction
  README.md
```

## Running the Pipeline

All scripts should be executed from the `src` directory.

### 1. Training

To train the model, run:
```bash
cd src
python train.py --epochs 100 --batch_size 16 --learning_rate 0.001 --consistency_lambda 0.3
```
- A spatial 5x5 block grid is used to assign patches to train, val, and test splits (roughly 70/15/15 ratio) to prevent data leakage.
- Inverse class frequency weights are used to handle the class imbalance.
- The `consistency_lambda` parameter controls the trade-off between the Classification Loss and Consistency MAE Loss.
- The best model (lowest validation loss) will be saved to `checkpoints/best_model.pth`.

### 2. Evaluation

To evaluate the trained model on the held-out test split, run:
```bash
python evaluate.py
```
- Calculates per-class Precision, Recall, F1, and IoU.
- Calculates Consistency MAE on the test split.
- Generates a qualitative visualization comparing 30m input, 10m prediction, and 10m true label map, saved in `outputs/visualizations/patch_comparison.png`.
- Output report is saved to `outputs/metrics_report.md`.

### 3. Inference on Full Tile

To run inference on the full 10980x10980 tile:
```bash
python inference.py
```
- This script processes the full tile in chunks, averages the 10m input to 30m on the fly, runs the SR U-Net model, and writes the 10m predictions out to `outputs/full_tile_prediction.tif`.

---

## Model Version History & Evolution

Below is the complete evolution of model iterations tested to solve class imbalance and optimize physical consistency:

| Version | Checkpoint | Key Configuration / Changes | Mean IoU | Consistency MAE | Water IoU | Built-up IoU | Veg IoU | Crop IoU | Bare Land IoU | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|
| **v1** | `best_model.pth` | Full Inverse Frequency Weights ($w_c \propto 1/\text{count}$) | 0.1190 | 0.2897 | 0.0000 | 0.0672 | 0.1858 | 0.3271 | 0.0149 | Superseded (Severe over-penalty gap) |
| **v2** | `best_model_v2.pth` | Gentler Sqrt Weights ($w_c \propto 1/\sqrt{\text{count}}$), $\lambda=0.3$ | 0.3405 | **0.1516** | 0.1966 | 0.3579 | 0.5140 | 0.5877 | 0.0462 | Strong baseline |
| **v3** | `best_model_v3.pth` | 3x Bare Land Oversampling + Photometric Aug, post-oversampling weights | 0.3291 | 0.1638 | 0.1921 | 0.3615 | 0.4880 | 0.5798 | 0.0241 | Bugged (Weight attenuation) |
| **v4** | `best_model_v4.pth` | **Fixed Pre-Oversampling Weight Lock**, 3x Bare Land Oversample + Aug | 0.3409 | 0.1543 | **0.2007** | 0.3795 | 0.4885 | 0.5857 | **0.0501** | **FINAL RECOMMENDED (Best Minority IoU)** |
| **v5** | `best_model_v5.pth` | Same as v4 with higher consistency weight ($\lambda=0.5$) | **0.3442** | 0.1539 | 0.1939 | **0.3974** | 0.5050 | **0.5915** | 0.0331 | **Alternative Top Model (Best Mean IoU)** |
| **v6** | `best_model_v6.pth` | Focal Loss ($\gamma=2.0$) + Sqrt Weights, 3x Oversampling | 0.3080 | 0.1939 | 0.1920 | 0.3720 | 0.3664 | 0.5694 | 0.0399 | Lower performance |
| **v7** | `best_model_v7.pth` | **7 Channels (RGB+NIR+NDVI+NDWI+BSI)**, Pre-Oversample Weights | 0.3184 | 0.1628 | 0.1708 | 0.3455 | **0.5202** | 0.5139 | 0.0415 | **Reduced Bare Land-Cropland Confusion (-54.8%)** |
| **v8** | `best_model_v8.pth` | **Multi-Tile Training (+8.8k Real Bare Land Pixels)**, Sqrt Weights | 0.3088 | 0.1785 | 0.1870 | 0.3040 | 0.4807 | 0.5265 | 0.0457 | **Highest Bare Land Recall (0.1192, +14.9%)** |
| **v9** | `best_model_v9.pth` | **0.5*Weighted_CE + 0.5*Dice Loss**, Sqrt Weights, 3x Oversampling | 0.3203 | **0.1546** | 0.1995 | 0.3406 | 0.4988 | 0.5244 | 0.0379 | Smooth probability calibration |
| **Ensemble** | `v4 + v7 Softmax` | **Probability Average of v4 (4-channel) + v7 (7-channel)** | **0.3465** | 0.1576 | 0.1987 | **0.3840** | 0.5201 | 0.5787 | **0.0508** | **BEST OVERALL MODEL (Highest Mean & Bare Land IoU)** |

### Final Recommendations
- **TOP RECOMMENDED MODEL (`Ensemble v4 + v7`)**: Achieves the **highest overall Mean IoU (0.3465)**, **highest Bare Land IoU (0.0508)**, and **highest Bare Land Precision (0.1122)** by combining 4-band spatial feature learning with 7-band spectral index discrimination.
- **Recommended Single Model (`best_model_v4.pth`)**: Top single-checkpoint model for minority class balance (**Water IoU 0.2007**, **Bare Land IoU 0.0501**, MAE 0.1543).
- **Targeted Discrimination Model (`best_model_v7.pth`)**: Recommended for maximum Cropland precision (0.7943) and reducing Cropland-Bare Land cross-contamination (-54.8%).

