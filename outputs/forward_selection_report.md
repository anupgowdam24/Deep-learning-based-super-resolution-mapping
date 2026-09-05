# Forward Selection Search: SRUNet Checkpoint Ensembling

## 1. Individual Performance of All 15 Checkpoints

| Checkpoint | Mean IoU | Water IoU | Built-up IoU | Veg IoU | Crop IoU | Bare Land IoU | Avg Recall | MAE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| training_1_best.pth | 49.47% | 60.65% | 55.34% | 65.70% | 53.28% | 12.38% | 69.31% | 0.1138 |
| training_2_best.pth | 48.48% | 59.92% | 52.31% | 65.77% | 55.67% | 8.71% | 74.88% | 0.1186 |
| training_3_best.pth | 48.14% | 59.44% | 54.28% | 62.83% | 56.24% | 7.94% | 75.07% | 0.1211 |
| training_4_best.pth | 50.15% | 62.81% | 55.73% | 67.14% | 53.54% | 11.53% | 71.79% | 0.1244 |
| training_5_best.pth | 48.30% | 62.67% | 50.41% | 67.70% | 48.57% | 12.14% | 71.39% | 0.1153 |
| training_6_best.pth | 49.71% | 61.58% | 56.47% | 68.62% | 51.75% | 10.15% | 71.70% | 0.1113 |
| training_7_best.pth | 46.65% | 61.77% | 49.18% | 59.25% | 55.48% | 7.55% | 74.47% | 0.1300 |
| **training_8_best.pth** (START) | **50.32%** | 61.59% | 55.84% | 65.99% | 56.16% | 12.04% | 72.82% | 0.1129 |
| training_9_best.pth | 47.83% | 57.20% | 52.44% | 64.09% | 56.28% | 9.14% | 75.39% | 0.1167 |
| training_10_best.pth | 49.31% | 59.60% | 55.57% | 68.34% | 53.08% | 9.96% | 73.88% | 0.1160 |
| training_11_best.pth | 44.52% | 53.78% | 50.06% | 59.85% | 54.81% | 4.11% | 76.84% | 0.1372 |
| training_12_best.pth | 48.04% | 61.00% | 54.03% | 68.36% | 50.24% | 6.57% | 75.06% | 0.1192 |
| training_13_best.pth | 48.32% | 59.81% | 54.59% | 63.47% | 56.82% | 6.89% | 76.26% | 0.1196 |
| training_14_best.pth | 48.35% | 61.04% | 52.72% | 67.78% | 53.97% | 6.22% | 76.36% | 0.1207 |
| training_15_best.pth | 48.21% | 61.21% | 53.82% | 64.70% | 55.45% | 5.88% | 75.47% | 0.1214 |

**Starting Model**: `training_8_best.pth` (Highest individual mIoU: 50.32%)

---

## 2. Forward Selection Step-by-Step Log

| Step | Candidate Tested | Decision | Prev mIoU | New mIoU | Delta mIoU | Class Delta Range | Reason |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| 1 | `training_1_best.pth` | **KEPT** | 50.32% | 50.35% | +0.03% | [-0.67%, +0.81%] | mIoU improved by +0.03% (50.32% -> 50.35%), worst class delta: -0.67% |
| 2 | `training_2_best.pth` | **DISCARDED** | 50.35% | 50.08% | -0.28% | [-0.93%, +0.57%] | mIoU decreased by -0.28% (50.35% -> 50.08%) |
| 3 | `training_3_best.pth` | **DISCARDED** | 50.35% | 50.08% | -0.28% | [-1.25%, +1.00%] | mIoU decreased by -0.28% (50.35% -> 50.08%) |
| 4 | `training_4_best.pth` | **KEPT** | 50.35% | 50.80% | +0.45% | [-0.31%, +1.90%] | mIoU improved by +0.45% (50.35% -> 50.80%), worst class delta: -0.31% |
| 5 | `training_5_best.pth` | **DISCARDED** | 50.80% | 50.59% | -0.21% | [-1.18%, +0.57%] | mIoU decreased by -0.21% (50.80% -> 50.59%) |
| 6 | `training_6_best.pth` | **KEPT** | 50.80% | 50.95% | +0.15% | [-0.65%, +0.88%] | mIoU improved by +0.15% (50.80% -> 50.95%), worst class delta: -0.65% |
| 7 | `training_7_best.pth` | **DISCARDED** | 50.95% | 50.37% | -0.58% | [-2.69%, +1.22%] | mIoU decreased by -0.58% (50.95% -> 50.37%) |
| 8 | `training_9_best.pth` | **DISCARDED** | 50.95% | 50.75% | -0.20% | [-0.68%, +0.80%] | mIoU decreased by -0.20% (50.95% -> 50.75%) |
| 9 | `training_10_best.pth` | **DISCARDED** | 50.95% | 50.74% | -0.20% | [-0.45%, +0.08%] | mIoU decreased by -0.20% (50.95% -> 50.74%) |
| 10 | `training_11_best.pth` | **DISCARDED** | 50.95% | 50.34% | -0.61% | [-1.77%, +1.10%] | mIoU decreased by -0.61% (50.95% -> 50.34%) |
| 11 | `training_12_best.pth` | **DISCARDED** | 50.95% | 50.59% | -0.36% | [-1.15%, +0.17%] | mIoU decreased by -0.36% (50.95% -> 50.59%) |
| 12 | `training_13_best.pth` | **DISCARDED** | 50.95% | 50.76% | -0.19% | [-1.05%, +0.98%] | mIoU decreased by -0.19% (50.95% -> 50.76%) |
| 13 | `training_14_best.pth` | **DISCARDED** | 50.95% | 50.67% | -0.28% | [-1.21%, +0.21%] | mIoU decreased by -0.28% (50.95% -> 50.67%) |
| 14 | `training_15_best.pth` | **DISCARDED** | 50.95% | 50.73% | -0.22% | [-1.34%, +0.51%] | mIoU decreased by -0.22% (50.95% -> 50.73%) |

---

## 3. Final Chosen Ensemble Performance

**Selected Checkpoints**: `training_8_best.pth`, `training_1_best.pth`, `training_4_best.pth`, `training_6_best.pth`

| Metric | Water | Built-up | Vegetation | Cropland | Bare Land | **Mean / Macro** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recall (Accuracy)** | 87.78% | 77.81% | 82.01% | 67.64% | 45.77% | **72.20%** |
| **Precision** | 67.94% | 68.45% | 80.63% | 73.78% | 14.49% | -- |
| **F1-Score** | 0.7659 | 0.7283 | 0.8131 | 0.7058 | 0.2201 | **0.6466** |
| **IoU** | 62.07% | 57.27% | 68.51% | 54.53% | 12.37% | **50.95%** |

**Physical Consistency MAE (30m composition)**: `0.1150`

