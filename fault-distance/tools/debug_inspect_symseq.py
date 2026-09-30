"""
Visual verification of symmetrical components using a real CSV file.

Method: sliding window = exactly 1 period (win = round(fs / f0) samples).
For each window: FFT → phasors Xa,Xb,Xc → Fortescue matrix → X0,X1,X2.
window=False: a rectangular window is valid because win = one full period.

fs is read from the CSV fs_hz column if present, otherwise from FS_FALLBACK.

Run: python inspect_symseq.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.column_detector import (
    detect_signal_columns,
    detect_distance_column,
)
from symseq.power_systems import symseq_from_waveforms

# ─── SETTINGS ────────────────────────────────────────────────────────────────
CSV_PATH    = "data/data_training/2AB_10km.csv"
F0          = 50.0    # Hz — grid frequency (fixed for this task)
FS_FALLBACK = 1000.0  # Hz — used if the CSV has no fs_hz column
# ──────────────────────────────────────────────────────────────────────────────


# ─── Loading ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Sampling frequency: from CSV or fallback
if "fs_hz" in df.columns:
    fs = float(df["fs_hz"].iloc[0])
    print(f"fs read from CSV      : {fs:.1f} Hz")
else:
    fs = FS_FALLBACK
    print(f"fs missing from CSV; using fallback: {fs:.1f} Hz")

# Window size = one full period → no leakage without a window function
win = int(round(fs / F0))
print(f"Window (1 cycle)      : {win} samples = {1000/F0:.1f} ms")

dist_col = detect_distance_column(list(df.columns))
col_map  = detect_signal_columns(list(df.columns), distance_col=dist_col)

# Metadata columns are excluded from the signal
service_cols = {dist_col, "fs_hz"}

sig_cols = [
    col_map["Ia"], col_map["Ib"], col_map["Ic"],
    col_map["Ua"], col_map["Ub"], col_map["Uc"],
]

sig = df[sig_cols].values.astype(float)   # (N, 6)
N   = len(sig)
dist_val = float(df[dist_col].iloc[0])

print(f"File                  : {os.path.basename(CSV_PATH)}")
print(f"Distance              : {dist_val:.1f} km")
print(f"Rows in CSV           : {N}")
print(f"Channels              : {col_map}")
print()

if N < win:
    print(f"[ERROR] Too few samples ({N}) for window {win}")
    sys.exit(1)


# ─── Sliding-window FFT → symmetrical components ───────────────────────────────
n_steps = N - win + 1   # number of window positions

I0_mag = np.zeros(n_steps)
I1_mag = np.zeros(n_steps)
I2_mag = np.zeros(n_steps)
U0_mag = np.zeros(n_steps)
U1_mag = np.zeros(n_steps)
U2_mag = np.zeros(n_steps)

for i in range(n_steps):
    sl = slice(i, i + win)   # window [i : i+win]

    ri = symseq_from_waveforms(
        sig[sl, 0], sig[sl, 1], sig[sl, 2],
        fs=fs, f0=F0, window=False,   # one full period → rectangular window
    )
    ru = symseq_from_waveforms(
        sig[sl, 3], sig[sl, 4], sig[sl, 5],
        fs=fs, f0=F0, window=False,
    )

    I0_mag[i] = ri["X0_mag"]
    I1_mag[i] = ri["X1_mag"]
    I2_mag[i] = ri["X2_mag"]
    U0_mag[i] = ru["X0_mag"]
    U1_mag[i] = ru["X1_mag"]
    U2_mag[i] = ru["X2_mag"]

# Time axis: center of each window, in ms
# i=0 → window center = win/2 samples from the start
t_center_ms = (np.arange(n_steps) + win / 2) / fs * 1000.0
t_full_ms   = np.arange(N) / fs * 1000.0


# ─── Verification test (synthetic data) ─────────────────────────────────────────────
print("=== Sanity check (synthetic positive sequence) ===")
t_syn = np.arange(win) / fs
A_syn = 100.0
rs = symseq_from_waveforms(
    A_syn * np.sin(2 * np.pi * F0 * t_syn),
    A_syn * np.sin(2 * np.pi * F0 * t_syn - 2 * np.pi / 3),
    A_syn * np.sin(2 * np.pi * F0 * t_syn + 2 * np.pi / 3),
    fs=fs, f0=F0, window=False,
)
ok1 = abs(rs["X1_mag"] - A_syn) < 0.5
ok2 = rs["X2_mag"] < 1.0
ok0 = rs["X0_mag"] < 1.0
print(f"  I1 = {rs['X1_mag']:.4f}  (expected {A_syn:.1f})  {'✓' if ok1 else '✗ ERROR'}")
print(f"  I2 = {rs['X2_mag']:.6f}  (expected ≈ 0)     {'✓' if ok2 else '✗ ERROR'}")
print(f"  I0 = {rs['X0_mag']:.6f}  (expected ≈ 0)     {'✓' if ok0 else '✗ ERROR'}")
print()

# Summary values over the entire sliding series (post-fault maximum)
print("=== Sliding window — peak values ===")
print(f"  max I1 = {I1_mag.max():.4f} A   |  max I2 = {I2_mag.max():.4f} A   |  max I0 = {I0_mag.max():.4f} A")
print(f"  max U1 = {U1_mag.max():.4f} kV  |  max U2 = {U2_mag.max():.4f} kV  |  max U0 = {U0_mag.max():.4f} kV")
print(f"  I2/I1 maximum = {(I2_mag / (I1_mag + 1e-12)).max():.4f}  (expected for 2AB ≈ 1.0)")


# ─── Plots ──────────────────────────────────────────────────────────────────
fname = os.path.basename(CSV_PATH)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle(
    f"Symmetrical components (sliding window {win} samples = {1000/F0:.0f} ms)"
    f"  |  {fname}  |  {dist_val:.1f} km",
    fontsize=11,
)

# ── Row 0: original signals ────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(t_full_ms, sig[:, 0], label="Ia", color="tab:blue",   lw=0.9)
ax.plot(t_full_ms, sig[:, 1], label="Ib", color="tab:orange", lw=0.9)
ax.plot(t_full_ms, sig[:, 2], label="Ic", color="tab:green",  lw=0.9)
ax.set_title("Original currents (A)")
ax.set_xlabel("Time (ms)")
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[0, 1]
ax.plot(t_full_ms, sig[:, 3], label="Ua", color="tab:blue",   lw=0.9)
ax.plot(t_full_ms, sig[:, 4], label="Ub", color="tab:orange", lw=0.9)
ax.plot(t_full_ms, sig[:, 5], label="Uc", color="tab:green",  lw=0.9)
ax.set_title("Original voltages (kV)")
ax.set_xlabel("Time (ms)")
ax.legend(fontsize=8)
ax.grid(True)

# ── Row 1: current components ──────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(t_center_ms, I1_mag, label="I1 positive sequence",   color="tab:blue",   lw=1.2)
ax.plot(t_center_ms, I2_mag, label="I2 negative sequence", color="tab:red",    lw=1.2)
ax.plot(t_center_ms, I0_mag, label="I0 zero sequence",  color="tab:purple", lw=1.2)
ax.set_title("Current symmetrical component magnitudes (A)")
ax.set_xlabel("Time (ms)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.grid(True)

# ── Row 1: voltage components ─────────────────────────────────────────
ax = axes[1, 1]
ax.plot(t_center_ms, U1_mag, label="U1 positive sequence",   color="tab:blue",   lw=1.2)
ax.plot(t_center_ms, U2_mag, label="U2 negative sequence", color="tab:red",    lw=1.2)
ax.plot(t_center_ms, U0_mag, label="U0 zero sequence",  color="tab:purple", lw=1.2)
ax.set_title("Voltage symmetrical component magnitudes (kV)")
ax.set_xlabel("Time (ms)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.grid(True)

# ── Row 2: unbalance ratios ────────────────────────────────────────
eps = 1e-12
I_unbal = I2_mag / (I1_mag + eps)
I0_ratio = I0_mag / (I1_mag + eps)

ax = axes[2, 0]
ax.plot(t_center_ms, I_unbal,  label="I2/I1 unbalance", color="tab:red",    lw=1.2)
ax.plot(t_center_ms, I0_ratio, label="I0/I1 zero sequence",     color="tab:purple", lw=1.2)
ax.axhline(1.0, color="gray", lw=0.8, ls="--", label="level 1.0")
ax.set_title("Current unbalance ratios")
ax.set_xlabel("Time (ms)")
ax.set_ylim(0, max(3.0, I_unbal.max() * 1.1))
ax.legend(fontsize=8)
ax.grid(True)

# ── Row 2: empty right panel, used for the text summary ───────────────
ax = axes[2, 1]
ax.axis("off")
summary = (
    f"File: {fname}\n"
    f"Distance: {dist_val:.1f} km\n"
    f"fs = {fs:.1f} Hz\n"
    f"Window: {win} samples ({1000/F0:.0f} ms)\n"
    f"Rows: {N}   Steps: {n_steps}\n\n"
    f"Peak values:\n"
    f"  I1 = {I1_mag.max():.3f} A\n"
    f"  I2 = {I2_mag.max():.3f} A\n"
    f"  I0 = {I0_mag.max():.3f} A\n"
    f"  U1 = {U1_mag.max():.4f} kV\n"
    f"  U2 = {U2_mag.max():.4f} kV\n"
    f"  U0 = {U0_mag.max():.4f} kV\n\n"
    f"Sanity check:\n"
    f"  I1={'✓' if ok1 else '✗'}  I2={'✓' if ok2 else '✗'}  I0={'✓' if ok0 else '✗'}"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
out = "tools/output_tools/symseq_inspect.png"
plt.savefig(out, dpi=120)
print(f"\nPlot saved: {out}")
