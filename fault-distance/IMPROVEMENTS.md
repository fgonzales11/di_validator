# Training Report and Improvement Analysis

## Date: 2026-04-30

---

## 1. Results Summary

### 1.1. Experiments

| Model | Parameters | Epochs completed | Best val_MAE | Test MAE | Test Max Error |
|---|---|---|---|---|---|
| CNN1D | 3 518 145 | 47 (ES) | 1.99 km | 1.99 km | 9.82 km |
| ResNet1D | 1 209 601 | 20 (ES) | 11.96 km | — | — |
| CNN1D (2nd) | 3 518 145 | 49 (ES) | 3.47 km | 2.99 km | 8.22 km |

*ES — Early Stopping (patience=15).*

**Conclusion**: CNN1D performed significantly better than ResNet1D on this dataset. ResNet1D began overfitting rapidly after epoch 5 (val_loss increased from 0.07 to 0.85), suggesting that an architecture with skip connections is excessive for 63 training examples and requires either strong regularization or more data.

---

## 2. Learning Curve Analysis (CNN1D, Best Run)

### 2.1. Train vs Validation Loss

- **Train loss** decreases monotonically from 2.04 (epoch 1) to ~0.01 (epoch 47).
- **Val loss** quickly falls to ~0.02 by epoch 10, then fluctuates between 0.003 and 0.03.
- The **gap** between train and val loss increases after epoch 20: train ~0.01, val ~0.005–0.02. This is a classic sign of **mild overfitting**.
- **Early stopping** worked correctly, stopping training before severe overfitting.

### 2.2. Validation MAE

- Best result: **MAE = 1.99 km** at epoch 31.
- After epoch 35, MAE starts to rise (3.0–5.5 km), confirming that the model is beginning to "memorize" noise in the training set.

### 2.3. Convergence Speed

- **Rapid phase** (epochs 1–10): a sharp drop in loss.
- **Fine-tuning phase** (epochs 10–30): slow improvement.
- **Degradation phase** (epochs 30–47): rising val_loss and MAE.

---

## 3. Test Error Analysis

### 3.1. Error Distribution

- **MAE**: 1.99 km (first run) / 2.99 km (second run)
- **Median error**: ~3.6 km
- **95th percentile**: ~6.0 km
- **99th percentile**: ~7.2 km

### 3.2. Problematic Distances

| File | True | Predicted | Error | Comment |
|---|---|---|---|---|
| 1A_0.5km.csv | 0.5 km | 10.3 km | 9.8 km | Catastrophic outlier |
| 1A_48km.csv | 48.0 km | 43.5 km | 4.5 km | Underestimation at long distances |
| 1A_28.5km.csv | 28.5 km | 32.0 km | 3.5 km | Overestimation at medium distances |

**Observation**: the model tends to **overestimate** predictions for medium distances (20–35 km) and **underestimate** them for long distances (>40 km). A catastrophic failure occurs at very short distances (<1 km).

### 3.3. Systematic Bias

- Mean error ≈ **−0.54 km** — a small negative bias.
- The error histogram resembles a normal distribution with heavy tails (σ ≈ 3.9 km).

---

## 4. Possible Improvements

### 4.1. Data (Data-Centric Approach) — **Highest Priority**

1. **Expand the training set**
   - Current: 63 training examples. This is critically small for 3.5M parameters.
   - Target: at least 300–500 examples (can be generated in RTDS/MATLAB).

2. **Data augmentation**
   - Add **Gaussian noise** to signals: std = 0.01–0.05 of the amplitude.
   - **Random time shift** (phase shift): ±1–2 periods (50 Hz).
   - **Amplitude scaling**: multiply by a factor of 0.9–1.1.
   - **Mixup / CutMix** for oscillograms (mix two signals with weighted labels).
   - Expected effect: less overfitting and better generalization.

3. **K-Fold Cross-Validation**
   - Current: a fixed 80/20 split (63/16). The estimate is unstable.
   - Recommendation: **5-Fold CV** with averaged metrics.
   - For the thesis: report mean MAE ± std across folds.

4. **Stratification**
   - Ensure that every fold evenly represents all distance ranges (0–10, 10–20, 20–30, 30–40, 40–50 km).
   - The current dataset is already nearly uniform (0.5 km spacing), but a random split can introduce imbalance.

### 4.2. Features

5. **Symmetrical components (Fortescue)**
   - Add 6 channels: I0, I1, I2, U0, U1, U2 (after FFT-based Fortescue).
   - The input will have 12 channels.
   - Expected effect: direct physical interpretability and better separation of fault conditions.

6. **Time-frequency features**
   - Add the spectrum (STFT / CWT) as extra channels.
   - Alternatively, use a wavelet transform to extract transients.

### 4.3. Model Architecture

7. **Reduce model capacity**
   - A CNN1D with 3.5M parameters for 63 examples has ~55K parameters per example. This is too many.
   - Recommendations:
     - `num_filters=32` instead of 64 → ~0.9M parameters.
     - Reduce the FC layer: 256→128→1 instead of 256→128→1 (already minimal).
     - Add **Global Average Pooling (GAP)** instead of Flatten + Dense to reduce the parameter count substantially.

8. **Regularization**
   - Increase `weight_decay` from 1e-5 to **1e-3 or 1e-4**.
   - Increase `dropout` from 0.3 to **0.5**.
   - Add **Dropout2d** (spatial dropout) before FC.
   - Use **Label Smoothing** (epsilon=0.1) for regression.

9. **Attention mechanisms**
   - **Squeeze-and-Excitation (SE) blocks** are already in ResNet1D and can be added to CNN1D.
   - **Temporal Attention** — weights over time to highlight fault inception.

10. **Ensembling**
    - Train 5 models with different random seeds / K-Fold splits.
    - Average their predictions. Expected MAE reduction: 10–20%.

### 4.4. Training Optimization

11. **Hyperparameter tuning (Grid/Random Search)**

    | Parameter | Range | Step |
    |---|---|---|
    | Learning rate | [1e-4, 5e-4, 1e-3, 3e-3, 1e-2] | — |
    | Batch size | [8, 16, 32] | — |
    | Dropout | [0.2, 0.3, 0.5] | — |
    | Weight decay | [1e-5, 1e-4, 1e-3] | — |
    | Loss function | [MSE, MAE, SmoothL1] | — |

12. **Advanced schedulers**
    - **CosineAnnealingWarmRestarts** (T_0=10, T_mult=2) — restarts with increasing periods.
    - **ReduceLROnPlateau** (patience=5, factor=0.5) — adaptive LR reduction on a plateau.
    - Combination: warm-up (5 epochs, linear) + cosine annealing.

13. **Earlier stopping**
    - The current patience=15 is too high for 50 epochs. **patience=7–10** is recommended.
    - Also track MAE rather than loss (val_loss can fluctuate due to outliers).

### 4.5. Analysis and Interpretability

14. **SHAP / Integrated Gradients**
    - Identify the time samples and channels that matter most for predictions.
    - Visualize attention weights.

15. **Outlier analysis**
    - 1A_0.5km.csv: a prediction of 10.3 km for a true distance of 0.5 km. This is not a random error.
    - Possible causes:
      - The signal at 0.5 km differs substantially from the others (less attenuation, a different shape).
      - There are few training examples below 2 km (only 1.0, 1.5, 2.0, 2.5 km = 4 examples).
    - **Solution**: add more examples at 0.5–2.0 km, or use weighted sampling.

---

## 5. Recommended Thesis Plan

### Stage 1: Baseline (Already Completed)

- [x] CNN1D with 6 channels
- [x] Standard normalization
- [x] Publication-quality plots (DPI=300, originally in Russian)
- [x] GPU training

### Stage 2: Data Improvements (1–2 Weeks)

- [ ] Add augmentation (noise, shift, scale)
- [ ] Implement 5-Fold CV
- [ ] Add more training examples (if possible)

### Stage 3: Model Improvements (1 Week)

- [ ] Add symmetrical components (12 channels)
- [ ] Reduce model capacity (num_filters=32, GAP)
- [ ] Strengthen regularization (dropout=0.5, weight_decay=1e-3)

### Stage 4: Optimization (3–5 Days)

- [ ] Grid search over lr, batch_size, and dropout
- [ ] Ensemble of 5 models
- [ ] Compare CNN1D vs DilatedCNN1D vs ResNet1D (with regularization)

### Stage 5: Finalization

- [ ] Comparative table of all experiments
- [ ] Plots: training, predictions, CDF, and errors by distance range
- [ ] Conclusions and recommendations for operation

---

## 6. Technical Notes

### Code Fixes Applied

1. `dataset.py`: corrected `SIGNAL_COLS` to `S1) BUS1UA` (with a space) for CSV compatibility.
2. `test.py`: corrected `TEST_DIR` → `data/data_test_csv`, `DEVICE` → `cuda`.
3. `utils/plots.py`: fully rewritten with DPI=300, Russian labels at the time, Times New Roman, grids, CDF, and percentiles.
4. `train.py`: added a call to `plot_metrics_summary`.

### Reproduce the Run

```powershell
cd Fault-Distance
python train.py --model cnn1d --epochs 100 --batch-size 16 --lr 0.001 --device cuda --data-dir data/data_training
python test.py
```

---

*Report generated automatically from an analysis of CNN1D and ResNet1D training.*
