# Overview: pipeline mathematics — from COMTRADE to the model input tensor

## Input data
- Format: COMTRADE (`.cfg` + `.dat`)
- Contents: current and voltage waveforms for phases A, B, C
- Sampling frequency: `fs` [Hz] (typically 2000–5000 Hz)
- Duration: ~200–400 ms

## Pipeline (processing stages)

```
COMTRADE (.cfg + .dat)
    ↓  tools/data_comtrade_to_csv.py
CSV file (columns: time, CT1IA, CT1IB, CT1IC, S1) BUS1UA, S1) BUS1UB, S1) BUS1UC, distance_km, fs_hz)
    ↓  data/dataset.py  (FaultDataset.__init__)
1. Center using the pre-fault history (center_by_prehistory)
2. Remove the DC component (remove_dc_period)
3. Detect fault inception and crop (detect_t0_and_crop)  [optional]
4. Pad / trim to SEQ_LENGTH
5. Compute symmetrical components (sliding_window_symseq)  [optional]
6. Normalize (standard or p.u.)
    ↓
Tensor (NUM_CHANNELS, SEQ_LENGTH) → PyTorch model
```

## Documentation files

| File | Contents |
|---|---|
| `00_overview.md` | This file — pipeline overview |
| `01_comtrade_to_csv.md` | COMTRADE → CSV conversion |
| `02_centering.md` | Centering using the pre-fault history |
| `03_dc_removal.md` | DC component removal (moving average) |
| `04_t0_detection.md` | Fault inception detection (RMS-based) |
| `05_windowing.md` | Cropping the [pre_fault, post_fault] window |
| `06_symmetrical_components.md` | Symmetrical components (sliding-window DFT + Fortescue) |
| `07_normalization.md` | Normalization: standard (z-score) vs p.u. |
| `08_tensor_format.md` | Final tensor format |

---

## Default parameters (from config.py)

| Parameter | Value | Description |
|---|---|---|
| `SEQ_LENGTH` | 400 | Window length in samples |
| `SAMPLING_FREQ_HZ` | 5000 (fallback) | Sampling frequency |
| `MAINS_FREQ_HZ` | 50 | Mains frequency |
| `T0_PRE_MS` | 50 | Pre-fault history [ms] |
| `T0_POST_MS` | 150 | Post-fault history [ms] |
| `T0_ETA_I` | 0.5 | Current rise threshold |
| `T0_ETA_U` | 0.85 | Voltage drop threshold |
| `NORMALIZATION_MODE` | 'standard' | 'standard' or 'pu' |

---

**Note:** All formulas are provided in `01_*.md` – `08_*.md`.
