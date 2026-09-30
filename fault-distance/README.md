# Fault-Distance: 1D-CNN for Fault Distance Estimation

This project predicts the distance to a short-circuit fault from current and voltage oscillograms using 1D-CNN / ResNet1D.

---

## Project Structure

```
Fault-Distance/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # PyTorch Dataset + DataLoaderFactory
│   ├── preprocessing.py    # Preprocessing utilities
│   └── data_training/      # ← Oscillogram CSV files (keep separate from .py files)
│       ├── 1A_0.5km.csv
│       ├── 1A_1.0km.csv
│       └── ...
├── models/
│   ├── cnn1d.py
│   ├── resnet1d.py
│   └── blocks.py
├── utils/
├── config.py
├── train.py
└── README.md
```

---

## Input Data Format

Each CSV file contains **one oscillogram** (one fault event).

### CSV File Structure

| Column | Description | Unit |
|---|---|---|
| `distance_km` | Distance to the fault (the same in every row of the file) | km |
| `CT1IA` | Instantaneous phase A current | A |
| `CT1IB` | Instantaneous phase B current | A |
| `CT1IC` | Instantaneous phase C current | A |
| `S1) BUS1UA` | Instantaneous phase A voltage | kV |
| `S1) BUS1UB` | Instantaneous phase B voltage | kV |
| `S1) BUS1UC` | Instantaneous phase C voltage | kV |

The number of rows equals the number of time samples (`SEQ_LENGTH = 400` is recommended).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place CSV files in
data/data_training/

# 3. Start training
python train.py

# Or specify parameters
python train.py --model cnn1d --epochs 100 --batch-size 32
python train.py --model resnet1d --epochs 200
```

---

## Model Architectures

| Model | Input tensor | Description |
|---|---|---|
| `cnn1d` | `(B, 6, 400)` | 3-block 1D-CNN with fast training |
| `dilated_cnn1d` | `(B, 6, 400)` | Dilated convolutions with a large receptive field |
| `resnet1d` | `(B, 6, 400)` | ResNet with SE blocks and the best accuracy |

---

## Key Parameters (`config.py`)

```python
NUM_CHANNELS = 6       # Ia, Ib, Ic, Ua, Ub, Uc
SEQ_LENGTH   = 400     # Time samples per file
DATA_DIR     = 'data/data_training'  # CSV directory (separate from .py files)
MODEL_TYPE   = 'cnn1d'
BATCH_SIZE   = 32
NUM_EPOCHS   = 100
```

---

## Normalization

Currents (`~0.07–260 A`) and voltages (`~100 kV`) have different scales.
A **per-channel `StandardScaler`** normalizes each of the 6 channels independently.
Distance is normalized with `MinMaxScaler` → `[0, 1]`.
