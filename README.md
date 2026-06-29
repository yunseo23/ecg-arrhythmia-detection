# ECG Arrhythmia Detection — Binary CNN-Attention Model

## Overview
Supraventricular ectopic beats (Class S) are rare (2.54%) but clinically critical. This project detects them using a CNN-Attention binary classifier built on a data-centric ECG pipeline, without synthetic data augmentation.

## Results
| Metric | Multi-class | Binary CNN-Attention |
| Precision (Class S) | 0.1062 | **0.7578 (+7.14×)** |
| Recall (Class S) | 0.4055 | **0.7143** |
| F1-Score (Class S) | 0.1684 | **0.7354 (+4.37×)** |

141.6K parameters · < 10 ms inference · MIT-BIH Database (109,461 beats)

## Model
```
Input (300 samples) → CNN ×3 (residual) → Multi-Head Attention (8 heads) → Global Average Pooling → Dense → Sigmoid
```
