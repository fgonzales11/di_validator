# Fault-Distance — AI Agent Reference

> This project predicts the distance to a short-circuit fault from current and voltage oscillograms using 1D-CNN / ResNet1D (PyTorch).
> Project language: Python; comments, documentation, runtime messages, and visualizations are in English.

---

## Project Overview

Each CSV file contains one oscillogram (one fault event). The model takes a tensor of shape `(B, NUM_CHANNELS, SEQ_LENGTH)` and predicts a scalar: the distance to the fault in kilometers (regression).

### Input CSV Format

Required columns:

- `distance_km` — label (the same in every row of the file)
- `CT1IA`, `CT1IB`, `CT1IC` — instantaneous phase A/B/C currents [A]
- `S1) BUS1UA`, `S1) BUS1UB`, `S1) BUS1UC` — instantaneous phase A/B/C voltages [kV]
- `fs_hz` — sampling frequency (written by `comtrade_to_csv.py`; older files use the `SAMPLING_FREQ_HZ` fallback)

The recommended number of rows per file is `SEQ_LENGTH = 400`.

### Technology Stack

- **PyTorch** >= 2.0.0 (main framework)
- **NumPy, Pandas, SciPy, scikit-learn** — data handling and preprocessing
- **Matplotlib, Seaborn** — visualization (DPI=300, English labels)
- **PyYAML** — experiment configuration
- **pytest** — testing
- **tqdm** — progress bars
- **comtrade** (optional) — reading COMTRADE files

The project does not use `pyproject.toml`, `setup.py`, or `package.json`; dependencies are defined in `requirements.txt`, and scripts run directly with `python <script>.py`.

---

## Directory Structure

```
Fault-Distance/
├── .agents/
│   └── skills/               # AI skills (auto-orchestrator, ml-engineer, web-search, ...)
├── .kilo/                    # Kilo agent metadata
├── .sixth/                   # Sixth agent metadata
├── .vscode/
│   └── settings.json         # VS Code settings
├── checkpoints/              # Saved weights (.pth)
├── configs/                  # Experiment YAML configurations
│   ├── base.yaml             # Base configuration (do not run directly)
│   ├── augment_train_cnn1d.yaml
│   ├── augment_train_resnet1d.yaml
│   └── activation_smoke.yaml
├── data/
│   ├── __init__.py
│   ├── augmentation.py       # TimeShift + GaussianNoise + AugmentationPipeline
│   ├── dataset.py            # FaultDataset + DataLoaderFactory
│   ├── fault_classifier.py   # (removed) fault classification is no longer used
│   ├── fault_inception.py    # Fault inception (t0) detection and cropping
│   ├── preprocessing.py      # Filters, sliding-window symseq, DataPreprocessor
│   ├── csv_all/              # Full CSV dataset (~100 files, training + testing)
│   ├── data_test/            # COMTRADE test files (.cfg + .dat)
│   ├── data_test_csv/        # Test CSV files
│   └── data_training/        # Training CSV files
├── logs/                     # Training logs (run_YYYYMMDD_HHMMSS/)
│   └── run_.../              # training.log, .png, activations/ (optional)
├── logs_smoke/               # Smoke-test logs
│   └── run_.../
│       └── activations/      # epoch_curves/, snapshots/, epoch_stats.csv
├── output/                   # Generated artifacts (not committed)
│   ├── thesis/               # Thesis plots
│   ├── pipeline/             # Preprocessing stage visualizations
│   └── symseq/               # Exported symmetrical-component COMTRADE files
├── models/
│   ├── __init__.py
│   ├── blocks.py             # SEBlock1D, ResBlock1D, InvertedResBlock1D
│   ├── cnn1d.py              # CNN1D, DilatedCNN1D, CNN1DRegressor
│   └── resnet1d.py           # FaultResNet1D (ResNet + SE blocks)
├── plots_pipeline/           # Preprocessing step visualizations
│   ├── __init__.py
│   ├── demo.py
│   ├── plot_augmentation.py
│   ├── plot_engine.py
│   ├── plot_preprocessing.py
│   └── plot_signals.py
├── scripts/
│   ├── augment_and_train.py  # Full pipeline: split → augment → train
│   ├── compare_models.py     # Compare multiple architectures on the same data
│   ├── health_check_training.py  # Preflight check (forward, backward, checkpoint, inference)
│   └── visualize_augmentation.py
├── symseq/                   # Symmetrical components (Fortescue)
│   ├── __init__.py
│   ├── adapter.py            # Batch adapter: (B,6,N) → symseq features
│   ├── core.py               # abc_to_seq / seq_to_abc / batch transforms
│   ├── fourier.py            # Phasor estimation with FFT
│   ├── power_systems.py      # symseq_from_waveforms
│   └── tests/                # test_core.py, test_fourier.py, test_adapter.py
├── tests/                    # pytest tests
│   ├── fault_inception/      # Manual t0 checks (check_t0.py, oscillograms/, README.md)
│   ├── test_activation_recorder.py
│   ├── test_augmentation.py
│   └── test_preprocessing_pipeline.py
├── tmp/                      # Temporary files
│   └── temp_smoke_train_run.py
├── tools/
│   ├── __init__.py
│   ├── README
│   ├── comtrade_to_csv.py    # COMTRADE → CSV converter
│   ├── debug_fortescue.py
│   ├── example_usage.py
│   ├── inspect_symseq.py
│   └── symseq_to_comtrade.py
├── utils/
│   ├── __init__.py
│   ├── activation_export.py  # Activation export (PNG/CSV)
│   ├── activation_recorder.py# Layer activation recording (ActivationRecorder)
│   ├── column_detector.py    # Automatic CSV column name detection
│   ├── logger.py             # TrainingLogger
│   ├── comparison_plots.py   # Comparative plots for multiple models
│   ├── metrics.py            # MAE, MSE, RMSE, R², MAPE
│   ├── plots.py              # Thesis plots (English labels, DPI=300)
│   └── probe_selection.py    # Probe batch selection
├── AGENTS.md                 # This file
├── IMPROVEMENTS.md           # Planned improvements
├── README.md                 # Main project documentation

├── config.py                 # Config dataclass + YAML loader
├── implementation_plan.md    # Thesis implementation plan
├── inference.py              # Inference on a single CSV
├── requirements.txt          # Python dependencies
├── test.py                   # Batch testing
└── train.py                  # Main training script
```

---

## Configuration System

Configuration is defined in `config.py` (the `Config` class) and can be overridden through:

1. **Python API:** `get_config(NUM_EPOCHS=200, BATCH_SIZE=64)`
2. **train.py CLI:** `python train.py --model resnet1d --epochs 100 --batch-size 32`
3. **YAML:** `python scripts/augment_and_train.py --config configs/augment_train_cnn1d.yaml`
4. **YAML + CLI override:** `--set training.num_epochs=200 model.dropout=0.1`

Priority (lowest to highest): defaults → `configs/base.yaml` → experiment YAML → CLI `--set` overrides.

Key `Config` fields:

- `MODEL_TYPE`: `'cnn1d' | 'dilated_cnn1d' | 'resnet1d'`
- `NUM_CHANNELS`: 6 (phase channels) or 12 (phase channels + symmetrical components)
- `SEQ_LENGTH`: 400
- `NORMALIZATION_MODE`: `'standard'` (StandardScaler + MinMaxScaler) or `'pu'` (physical normalization)
- `SYMSEQ_ENABLED`: True → the dataset automatically supplements the 6 channels with symmetrical components
- `T0_ENABLED`: True → automatic cropping around fault inception
- `REMOVE_DC_ENABLED`: True → remove the aperiodic component (moving average over one grid period)

---

## Preprocessing and Augmentation

### Signal Preprocessing

- **Remove DC period** (`REMOVE_DC_ENABLED`) — the **primary method** for removing the aperiodic component: subtract the moving average over one grid period (20 ms at 50 Hz). It introduces no phase distortion. It is preceded by `center_by_prehistory` (centering over the first 20 ms).
- **Fault inception (t0)** (`T0_ENABLED`) — a two-stage algorithm (D4 + cycle difference) for detecting fault inception and cropping the `[pre_fault_ms, post_fault_ms]` window.
- **Symmetrical components** (`SYMSEQ_ENABLED`) — a sliding-window DFT produces time-varying symmetrical-component magnitudes `|I1|,|I2|,|I0|,|U1|,|U2|,|U0|`, which are concatenated with the 6 phase channels (12 in total).

### Augmentation (`data/augmentation.py`)

- **TimeShift**: shift the oscillogram left/right, padding with the first/last row; preserve `SEQ_LENGTH` rows.
- **GaussianNoise**: add white noise at SNR levels of `[1, 5, 10, 20, 40]` dB.
- **AugmentationPipeline**: create `2 × 5 shifts × 5 SNR = 50` augmented copies of each source file.

---

## Models

| Model | Input | Description |
|---|---|---|
| `cnn1d` | `(B, C, 400)` | 3 Conv1d→BN→ReLU→MaxPool→Dropout blocks, then FC 256→128→1 |
| `dilated_cnn1d` | `(B, C, 400)` | Dilated convolutions (dilations=[1,2,4,8]), larger receptive field |
| `resnet1d` | `(B, C, 400)` | ResNet with SE blocks, depth=1..4, GAP + head |

`C` = `NUM_CHANNELS` (6 or 12, depending on `SYMSEQ_ENABLED`).

---

## Build and Run Commands

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Training

```bash
# Basic run
python train.py

# With parameters
python train.py --model cnn1d --epochs 100 --batch-size 32 --lr 0.001

# YAML configuration
python scripts/augment_and_train.py --config configs/augment_train_cnn1d.yaml

# YAML + override
python scripts/augment_and_train.py --config configs/augment_train_cnn1d.yaml \
    --set training.num_epochs=200
```

### Inference

```bash
python inference.py --model checkpoints/best_model.pth --csv data/data_training/1A_0.5km.csv --has-labels --device cpu
```

### Batch Testing

```bash
python test.py
```

Before running, place the test CSV files in `data/data_test_csv/` and make sure that `CHECKPOINT` in `test.py` points to the required .pth file.

### Model Comparison

```bash
# Compare CNN1D and ResNet1D over 2 epochs (smoke test)
python scripts/compare_models.py --models cnn1d resnet1d --epochs 2

# Full comparison of all three architectures
python scripts/compare_models.py --models cnn1d resnet1d dilated_cnn1d --epochs 100 --config configs/base.yaml

# With an external test set
python scripts/compare_models.py --models cnn1d resnet1d --epochs 50 --test-dir data/data_test_csv/
```

The script trains each model sequentially on the same data, runs inference, generates comparative plots, and selects the best model by MAE. Results are saved to:

- `logs/comparison_YYYYMMDD_HHMMSS/` — logs, plots, checkpoints
- `output/thesis/model_comparison/` — copies of comparative plots
- `output/thesis/best_model/` — checkpoint and plots for the best model

### Preflight Health Check

```bash
python scripts/health_check_training.py --config configs/base.yaml
```

Checks: one forward/backward step, checkpoint saving/loading, and inference on one CSV.

### COMTRADE → CSV Conversion

```bash
# Configure tools/data_comtrade_config.ini, then run:
python tools/data_comtrade_to_csv.py
```

### Data Augmentation

```bash
python data/augmentation.py --input data/data_training --output data/data_augmented
```

### Tests

```bash
pytest tests/ -v
```

---

## Activation Export (Thesis Feature)

Enable in YAML:

```yaml
activation_export:
  enabled: true
  layers: auto
  capture_epochs: ["first", "best", "every_n:5"]
  export_formats: ["png", "csv"]
```

Results are saved to `logs/run_.../activations/`:

- `snapshots/` — per-layer PNG activation curves for the probe sample.
- `snapshots.csv` — scalar statistics (mean, std, rms, energy, ...).
- `epoch_stats.csv` — per-epoch aggregation of layer metrics.
- `epoch_curves/` — plots of epoch metrics for each layer.

The probe is selected deterministically from the validation set (`first_val_sample` by default).

---

## Code Organization and Conventions

### Imports

- `scripts/` and `tests/` often use `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))` to import root modules.
- Models are imported directly: `from models.cnn1d import CNN1D, DilatedCNN1D`.

### Code Style

- **Docstrings**: Google style or NumPy style; use English for comments, documentation, runtime messages, and visualizations.
- **Type hints**: used extensively in new modules (`utils/activation_recorder.py`, `utils/probe_selection.py`, etc.).
- **Naming**: `CamelCase` for classes, `snake_case` for functions/variables, `UPPER_CASE` for configuration constants.
- **Visualization**: `matplotlib.use('Agg')`; DPI=300; English axis labels and legends; a consistent palette using `_COLOR_*` constants in `utils/plots.py`.

### Normalization

- **Standard** (`NORMALIZATION_MODE='standard'`): per-channel `StandardScaler` for signals, `MinMaxScaler` for distance → `[0, 1]`.
- **p.u.** (`NORMALIZATION_MODE='pu'`): physical normalization using base quantities. Divide currents by `Ibase = S_base_MVA·10⁶ / (√3·Unom·10³)`, voltages by `Ubase = Unom·10³ / √3`, and distance by `LINE_L_KM`. No fitted scalers are required, making it suitable for inference on new lines.

### Checkpoints

Saved in `checkpoints/` with the following contents:

```python
{
    'epoch': int,
    'model_state_dict': state_dict,
    'optimizer_state_dict': state_dict,
    'config': Config,
    'scalers': {'signal': [...], 'distance': MinMaxScaler}
}
```

When loading in `inference.py` and `test.py`, the scalers are used to inverse-transform predictions.

### Logs and Artifacts

Every `train.py` run creates a unique directory:

```
logs/run_YYYYMMDD_HHMMSS/
├── training_YYYYMMDD_HHMMSS.log
├── training_history.png
├── predictions.png
└── metrics_summary.png
```

Enabling `activation_export` adds:

```
logs/run_.../activations/
├── snapshots/                # Per-layer PNG activation curves
├── snapshots.csv             # Scalar stats (mean, std, rms, energy, ...)
├── epoch_stats.csv           # Per-epoch aggregation of layer metrics
└── epoch_curves/             # Plots of epoch metrics for each layer
```

Smoke tests (`logs_smoke/`) have a similar structure, usually with fewer epochs.

---

## Testing

Tests are written with **pytest**.

- `tests/test_preprocessing_pipeline.py` — checks `remove_dc_period`, fault type classification, 12-channel tensor construction, and sliding-window symseq variability.
- `tests/test_augmentation.py` — checks time shifts, Gaussian noise, and the full augmentation pipeline (file creation and `distance_km` preservation).
- `tests/test_activation_recorder.py` — checks activation recording, PNG/CSV export, and statistics.
- `symseq/tests/test_core.py`, `test_fourier.py`, `test_adapter.py` — symmetrical-component tests (Fortescue).
- `tests/test_column_detector.py` — checks automatic CSV column name detection.
- `tests/fault_inception/` — manual t0 detection checks (`check_t0.py` + oscillograms in `oscillograms/`).

Run:

```bash
pytest tests/ -v
```

Some tests (`TestTwelveChannelDataset`) require real CSV files in `data/csv_all/` and are skipped (`pytest.skip`) when the directory is missing.

---

## Security and Constraints

- The project contains no network services, secrets, or sensitive data.
- All data paths are relative to the working directory.
- `.gitignore` excludes: `checkpoints/`, `logs/`, `*.pth`, `*.csv`, `data/*.csv`.
- When working with COMTRADE files, make sure that `tools/data_comtrade_config.ini` contains the correct `line_length_km`.

---

## Common Pitfalls

1. **Inference channels:** if the model was trained with `SYMSEQ_ENABLED=True` and the input CSV has only 6 channels, `inference.py` and `test.py` automatically add sliding-window symseq channels. If the channel counts still differ, the script raises `ValueError`.
2. **`SIGNAL_COLS` mismatch:** older versions used names without a space (`S1)BUS1UA`). The current code expects `S1) BUS1UA` (with a space); see `data/dataset.py`.
3. **fs_hz:** new CSV files created by `comtrade_to_csv.py` contain an `fs_hz` column. If it is missing, use `cfg.SAMPLING_FREQ_HZ` (fallback: 2000 Hz). Do not hardcode the sampling frequency in algorithms; read it from the CSV or cfg.
4. **YAML and `NUM_CHANNELS`:** many YAML configurations specify `data.num_channels: 6`. If `SYMSEQ_ENABLED=True`, `load_config()` automatically adjusts `NUM_CHANNELS` to 12.
5. **Augmentation pipeline:** `scripts/augment_and_train.py` creates an intermediate `*_staging` directory. Make sure there is enough disk space: every source file produces 50 augmented copies.
