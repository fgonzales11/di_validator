"""
Demo script: visualize every preprocessing stage.
Load one CSV from data/data_training/ and plot the following in sequence:
1. Original oscillogram
2. DC component removal
3. Butterworth filter
4. Fault inception detection (t0)
5. Symmetrical components
6. Normalization
7. Time shift (augmentation)
8. Gaussian noise (augmentation)
"""

import os
import sys
import numpy as np
import pandas as pd

# Add the project root to the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from data.preprocessing import remove_dc_period, sliding_window_symseq, DataPreprocessor
from data.fault_inception import FaultInceptionParams, detect_t0_single_phase, _fourth_order_difference, _cycle_difference_index
from data.augmentation import TimeShiftAugmentation, GaussianNoiseAugmentation

from plots_pipeline.plot_signals import plot_oscillogram_6ch, plot_fft_spectrum
from plots_pipeline.plot_preprocessing import (
    plot_dc_removal, plot_t0_detection, plot_symseq, plot_normalization
)
from plots_pipeline.plot_augmentation import plot_time_shift, plot_noise_levels

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CFG = Config()
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_training')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'pipeline')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose one file for the demonstration
DEMO_FILE = os.path.join(DATA_DIR, '1A_1.5km.csv')
if not os.path.exists(DEMO_FILE):
    # Try to find any CSV file
    import glob
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    DEMO_FILE = csv_files[0]

print(f"Demo file: {os.path.basename(DEMO_FILE)}")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DEMO_FILE)
fs = float(df['fs_hz'].iloc[0]) if 'fs_hz' in df.columns else 2000.0
T = len(df)
time_ms = np.arange(T) / fs * 1000.0

signal_cols = ['CT1IA', 'CT1IB', 'CT1IC', 'S1) BUS1UA', 'S1) BUS1UB', 'S1) BUS1UC']
sig_raw = df[signal_cols].values.astype(np.float32).T  # (6, T)

print(f"Sampling frequency: {fs:.0f} Hz")
print(f"Duration: {T} samples = {time_ms[-1]:.1f} ms")
print(f"Signal shape: {sig_raw.shape}")

# ---------------------------------------------------------------------------
# Stage 0: Original oscillogram
# ---------------------------------------------------------------------------
print("\n[Stage 0] Original oscillogram...")
plot_oscillogram_6ch(
    time_ms, sig_raw,
    channel_labels=['I_A', 'I_B', 'I_C', 'U_A', 'U_B', 'U_C'],
    units=['A', 'A', 'A', 'kV', 'kV', 'kV'],
    title='Original fault oscillogram',
    output_path=os.path.join(OUTPUT_DIR, '00_original_oscillogram.png')
)

# FFT spectrum for IA
print("[Stage 0] Phase A current spectrum...")
plot_fft_spectrum(
    time_ms, sig_raw[0], fs,
    title='Phase A current spectrum (original signal)',
    output_path=os.path.join(OUTPUT_DIR, '00_fft_spectrum_ia.png'),
    color='#E6B800'
)

# ---------------------------------------------------------------------------
# Stage 1: DC component removal
# ---------------------------------------------------------------------------
print("\n[Stage 1] DC component removal...")
sig_dc_removed = remove_dc_period(sig_raw, fs=fs, f_net=50.0)
plot_dc_removal(
    time_ms, sig_raw, sig_dc_removed, fs,
    output_path=os.path.join(OUTPUT_DIR, '01_dc_removal.png')
)

# ---------------------------------------------------------------------------
# Stage 2: Fault inception detection (t0)
# ---------------------------------------------------------------------------
print("\n[Stage 3] Fault inception detection (t0)...")
params = FaultInceptionParams(fs_hz=fs, mains_hz=50.0)
current_ia = sig_dc_removed[0, :]  # phase A current after DC removal

d4 = _fourth_order_difference(current_ia)
di, _ = _cycle_difference_index(current_ia, params)
t0_idx = detect_t0_single_phase(current_ia, params)
if t0_idx is None:
    t0_idx = T // 2
    print(f"  [WARN] t0 not detected; using the midpoint: {t0_idx}")
else:
    print(f"  t0 = sample {t0_idx} ({time_ms[t0_idx]:.1f} ms)")

plot_t0_detection(
    time_ms, current_ia, d4, di, t0_idx, fs,
    output_path=os.path.join(OUTPUT_DIR, '02_t0_detection.png')
)

# ---------------------------------------------------------------------------
# Stage 3: Symmetrical components
# ---------------------------------------------------------------------------
print("\n[Stage 3] Symmetrical components...")
symseq = sliding_window_symseq(sig_dc_removed, fs=fs, f0=50.0, window_cycles=1)
plot_symseq(
    time_ms, symseq,
    output_path=os.path.join(OUTPUT_DIR, '03_symmetrical_components.png')
)

# ---------------------------------------------------------------------------
# Stage 4: Normalization
# ---------------------------------------------------------------------------
print("\n[Stage 4] Normalization (standard)...")
sig_normalized = sig_dc_removed.copy()
for ch in range(sig_normalized.shape[0]):
    sig_normalized[ch] = DataPreprocessor.normalize_signal(sig_normalized[ch], method='standard')

plot_normalization(
    time_ms, sig_dc_removed, sig_normalized, method='standard',
    output_path=os.path.join(OUTPUT_DIR, '04_normalization.png')
)

# ---------------------------------------------------------------------------
# Stage 6: Augmentation — time shift
# ---------------------------------------------------------------------------
print("\n[Stage 6] Augmentation: time shift...")
# Time shifting requires a DataFrame
df_for_shift = df.copy()
shifter = TimeShiftAugmentation(seq_length=T)
df_shifted_left = shifter.shift_left(df_for_shift, shift_amount=20)
df_shifted_right = shifter.shift_right(df_for_shift, shift_amount=20)

sig_shift_left = df_shifted_left[signal_cols].values.astype(np.float32).T
sig_shift_right = df_shifted_right[signal_cols].values.astype(np.float32).T

plot_time_shift(
    time_ms, sig_raw, sig_shift_left, sig_shift_right, shift_amount=20,
    output_path=os.path.join(OUTPUT_DIR, '05_time_shift_augmentation.png')
)

# ---------------------------------------------------------------------------
# Stage 6: Augmentation — Gaussian noise
# ---------------------------------------------------------------------------
print("\n[Stage 6] Augmentation: Gaussian noise...")
noise_aug = GaussianNoiseAugmentation(seq_length=T, num_channels=6)
noisy_dict = {}
for snr_db in [1, 5, 10, 20, 40]:
    df_noisy = noise_aug.add_gaussian_noise(df_for_shift, snr_db=snr_db, random_state=42)
    noisy_dict[snr_db] = df_noisy[signal_cols].values.astype(np.float32).T

plot_noise_levels(
    time_ms, sig_raw, noisy_dict,
    output_path=os.path.join(OUTPUT_DIR, '06_gaussian_noise_augmentation.png')
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Demo complete!")
print(f"Plots saved to: {OUTPUT_DIR}")
print(f"{'='*60}")

# List the generated files
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")
