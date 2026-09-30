::: {.cell .markdown}
# Complete COMTRADE Processing Pipeline for Fault Location

## Overview

This notebook implements a complete processing pipeline for fault waveforms:

1.  Parse COMTRADE .cfg and .dat files
2.  Butterworth filtering (DC removal)
3.  Fault inception detection
4.  Signal windowing
5.  Calculate symmetrical components (Fortescue)
6.  Fault type classification
7.  Normalization to p.u.
8.  Build the final tensor (12 channels: phase quantities + symmetrical
    components)

**After each stage, plots with labeled quantities are generated to verify
correctness.**
:::

::::: {.cell .code execution_count="2"}
``` python
!pip install numpy pandas matplotlib scipy comtrade
```

::: {.output .stream .stdout}
    Requirement already satisfied: numpy in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (2.2.6)
    Requirement already satisfied: pandas in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (2.3.3)
    Requirement already satisfied: matplotlib in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (3.10.8)
    Requirement already satisfied: scipy in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (1.15.3)
    Requirement already satisfied: comtrade in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (0.1.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (from pandas) (2025.3)
    Requirement already satisfied: pytz>=2020.1 in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (from pandas) (2026.1.post1)
    Requirement already satisfied: cycler>=0.10 in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (from matplotlib) (4.62.1)
    Requirement already satisfied: pyparsing>=3 in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (from matplotlib) (3.3.2)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (from matplotlib) (1.3.2)
    Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (from matplotlib) (1.5.0)
    Requirement already satisfied: pillow>=8 in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (from matplotlib) (12.1.1)
    Requirement already satisfied: packaging>=20.0 in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (from matplotlib) (25.0)
    Requirement already satisfied: six>=1.5 in c:\users\muhachev-ll\appdata\local\programs\python\python310\lib\site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
:::

::: {.output .stream .stderr}

    [notice] A new release of pip is available: 23.0.1 -> 26.0.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
:::
:::::

:::: {.cell .code execution_count="3"}
``` python
# =============================================================================
# Import libraries
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import struct
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from comtrade import Comtrade

# Configure plots
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['grid.alpha'] = 0.3

print("✓ Libraries loaded")
```

::: {.output .stream .stdout}
    ✓ Libraries loaded
:::
::::

::: {.cell .markdown}
## Section 1: Parse COMTRADE Files

### 1.1 COMTRADE Structure {#11-comtrade-structure}

COMTRADE (IEEE Std C37.111) consists of two files:

- **.cfg** --- configuration (metadata)
- **.dat** --- data (waveforms)
:::

:::: {.cell .code execution_count="4"}
``` python

import os
import glob
from pathlib import Path

# Path to the data folder
DATA_FOLDER = 'data'  # Change to your path, for example: '/content/data' or 'C:/Data'

def find_comtrade_files(folder_path):
    """Find all pairs of .cfg and .dat files in the folder."""
    cfg_files = glob.glob(os.path.join(folder_path, '*.cfg'))
    comtrade_pairs = []
    
    for cfg_path in cfg_files:
        base_name = cfg_path[:-4]  # remove .cfg
        dat_path = base_name + '.dat'
        
        if os.path.exists(dat_path):
            comtrade_pairs.append({
                'cfg': cfg_path,
                'dat': dat_path,
                'name': os.path.basename(base_name)
            })
        else:
            print(f"⚠️  Warning: no .dat file found for {cfg_path}")
    
    return comtrade_pairs

# Find files
comtrade_files = find_comtrade_files(DATA_FOLDER)

print(f"✓ Found {len(comtrade_files)} COMTRADE file pair(s):")
for i, pair in enumerate(comtrade_files, 1):
    cfg_path = pair['cfg']
    print(f"  {i}. {pair['name']}")
    print(f"     cfg: {cfg_path}")
    print(f"     dat: {pair['dat']}")
```

::: {.output .stream .stdout}
    ✓ Found 1 COMTRADE file pair(s):
      1. 3.1.1(K3, 1A)
         cfg: data\3.1.1(K3, 1A).cfg
         dat: data\3.1.1(K3, 1A).dat
:::
::::

:::: {.cell .code execution_count="5"}
``` python
def inspect_channels(rec):
    """List all detected phase channels with their CFG channel numbers."""
    print(f"{'No.':<5} | {'Channel name':<20} | {'Type':<5} | {'Side':<10}")
    print("-" * 50)
    
    found_info = {}
    for i, name in enumerate(rec.analog_channel_ids):
        n_up = name.upper()
        
        # Filter by phase and type
        phase = next((p for p in ['A', 'B', 'C', '\u0410', '\u0412', '\u0421'] if p in n_up), None)
        is_u = any(x in n_up for x in ['U', '\u0423', 'V'])
        is_i = any(x in n_up for x in ['I', '\u0418'])
        
        if phase and (is_u or is_i):
            ch_num = i + 1 # Channel number as in CFG
            side = "Secondary" if "SEC" in n_up else "Primary"
            ch_type = "U" if is_u else "I"
            
            print(f"{ch_num:<5} | {name:<20} | {ch_type}{phase:<4} | {side}")
            found_info[ch_num] = name
            
    return found_info

rec = Comtrade()
rec.load(cfg_path)
channel_map = inspect_channels(rec)
```

::: {.output .stream .stdout}
    No.   | Channel name         | Type  | Side
    --------------------------------------------------
    1     | CT1IAprim            | IA    | Primary
    2     | CT1IBprim            | IB    | Primary
    3     | CT1ICprim            | IC    | Primary
    4     | CT2IAprim            | IA    | Primary
    5     | CT2IBprim            | IB    | Primary
    6     | CT2ICprim            | IC    | Primary
    7     | CT3IAprim            | IA    | Primary
    8     | CT3IBprim            | IB    | Primary
    9     | CT3ICprim            | IC    | Primary
    10    | CT4IAprim            | IA    | Primary
    11    | CT4IBprim            | IB    | Primary
    12    | CT4ICprim            | IC    | Primary
    13    | CT1IAsec             | IA    | Secondary
    14    | CT1IBsec             | IB    | Secondary
    15    | CT1ICsec             | IC    | Secondary
    16    | LINE2IA_subA         | UA    | Primary
    17    | LINE2IB_subA         | UA    | Primary
    18    | LINE2IC_subA         | UA    | Primary
    19    | LINE2IA_subB         | UA    | Primary
    20    | LINE2IB_subB         | UB    | Primary
    21    | LINE2IC_subB         | UB    | Primary
    22    | CT2IAsec             | IA    | Secondary
    23    | CT2IBsec             | IB    | Secondary
    24    | CT2ICsec             | IC    | Secondary
    25    | CT3IAsec             | IA    | Secondary
    26    | CT3IBsec             | IB    | Secondary
    27    | CT3ICsec             | IC    | Secondary
    28    | CT4IAsec             | IA    | Secondary
    29    | CT4IBsec             | IB    | Secondary
    30    | CT4ICsec             | IC    | Secondary
    31    | S1) VT1UAprim        | UA    | Primary
    32    | S1) VT1UBprim        | UB    | Primary
    33    | S1) VT1UCprim        | UC    | Primary
    34    | S1) VT2UAprim        | UA    | Primary
    35    | S1) VT2UBprim        | UB    | Primary
    36    | S1) VT2UCprim        | UC    | Primary
    37    | S1) VT1UAsec         | UA    | Secondary
    38    | S1) VT1UBsec         | UB    | Secondary
    39    | S1) VT1UCsec         | UC    | Secondary
    45    | S1) VT2UAsec         | UA    | Secondary
    46    | S1) VT2UBsec         | UB    | Secondary
    47    | S1) VT2UCsec         | UC    | Secondary
    59    | VT_ANGLE             | UA    | Primary
:::
::::

:::: {.cell .code execution_count="6"}
``` python
def parse_comtrade(rec, i_prim_nums: list, i_sec_nums: list, u_prim_nums: list, u_sec_nums: list):
    """Build signal matrices from the channel numbers selected by the user."""
    
    def get_data_by_nums(nums):
        # Create an empty 3xN matrix. Return zeros if no channel numbers are provided.
        mat = np.zeros((3, len(rec.time)))
        for i, num in enumerate(nums[:3]): # Use only the first 3 channel numbers (A, B, C)
            mat[i] = rec.analog[num - 1] # -1 because array indices start at 0
        return mat

    i_prim = get_data_by_nums(i_prim_nums)
    i_sec  = get_data_by_nums(i_sec_nums)
    u_prim = get_data_by_nums(u_prim_nums)
    u_sec  = get_data_by_nums(u_sec_nums)
    
    print("✓ Arrays I_prim, I_sec, U_prim, U_sec created successfully.")
    return i_prim, i_sec, u_prim, u_sec

# Example usage:
i_p_idx = [1, 2, 3]    # For example: CT1IAprim, CT1IBprim, CT1ICprim
i_s_idx = [13, 14, 15] # For example: CT1IAsec, CT1IBsec, CT1ICsec
u_p_idx = [31, 32, 33] # For example: S1) VT1UAprim, VT1UBprim, VT1UCprim
u_s_idx = [37, 38, 39] # For example: S1) VT1UAsec, VT1UBsec, VT1UCsec

I_prim, I_sec, U_prim, U_sec = parse_comtrade(
    rec, 
    i_prim_nums=i_p_idx, 
    i_sec_nums=i_s_idx, 
    u_prim_nums=u_p_idx, 
    u_sec_nums=u_s_idx)

# === Build a shared data context for subsequent sections ===
IA_filtered = I_prim[0]
IB_filtered = I_prim[1]
IC_filtered = I_prim[2]
UA_filtered = U_prim[0]
UB_filtered = U_prim[1]
UC_filtered = U_prim[2]

# Line parameters and nominal values
Unom_kv = 110.0          # Nominal line voltage, kV
S_base_MVA = 100.0       # Base power for calculating nominal current, MVA
Inom_kA = S_base_MVA / (np.sqrt(3) * Unom_kv)  # Nominal current, kA

data = {
    'fs': rec.cfg.sample_rates[0][0],
    'frequency': rec.cfg.frequency,
    'Inom': Inom_kA,
    'fault_time': None,          # Set manually to check detection accuracy (s)
    'fault_distance_km': None,   # Set manually for the p.u. calculation (km)
}
print("✓ Data context created")
```

::: {.output .stream .stdout}
    ✓ Arrays I_prim, I_sec, U_prim, U_sec created successfully.
    ✓ Data context created
:::
::::

:::: {.cell .code execution_count="7"}
``` python
def plot_phases(rec, i_data, u_data, i_nums, u_nums, title="Waveforms"):
    """
    Plot phases A, B, C with channel names and units from the CFG file.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    
    time_ms = np.array(rec.time) * 1000
    colors = ['#d62728', '#2ca02c', '#1f77b4'] # A, B, C
    
    # Helper function to retrieve metadata (name and unit)
    def get_meta(nums, idx_in_list):
        try:
            ch_idx = nums[idx_in_list] - 1
            channel = rec.cfg.analog_channels[ch_idx]
            return channel.name, channel.uu
        except:
            return f"Channel {idx_in_list}", "?"

    for row in range(3):
        # --- CURRENTS (left column) ---
        name_i, unit_i = get_meta(i_nums, row)
        ax_i = axes[row, 0]
        ax_i.plot(time_ms, i_data[row], color=colors[row], lw=1.5)
        ax_i.set_ylabel(f"[{unit_i}]")
        ax_i.set_title(f"{name_i}", loc='left', fontweight='bold', fontsize=11)
        ax_i.grid(True, alpha=0.3)
        
        # --- VOLTAGES (right column) ---
        name_u, unit_u = get_meta(u_nums, row)
        ax_u = axes[row, 1]
        ax_u.plot(time_ms, u_data[row], color=colors[row], lw=1.5)
        ax_u.set_ylabel(f"[{unit_u}]")
        ax_u.set_title(f"{name_u}", loc='left', fontweight='bold', fontsize=11)
        ax_u.grid(True, alpha=0.3)

    # Shared X-axis labels
    axes[2, 0].set_xlabel("Time [ms]")
    axes[2, 1].set_xlabel("Time [ms]")
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# Usage (ensure i_p_idx and u_p_idx are defined in the cell above)
plot_phases(rec, I_prim, U_prim, i_p_idx, u_p_idx, title="Primary quantities")
```

::: {.output .display_data}
![](a09e53cfdb23a210bebaf069fe774c9cc3f0eaf1.png)
:::
::::

::: {.cell .markdown}
## Section 2: Butterworth Filtering (DC Removal)
:::

:::: {.cell .code execution_count="8"}
``` python
from scipy import signal
import pandas as pd

ORDER = 4
FC = 25.0

def butterworth_highpass(data, fs, fc, order=4):
    """
    Butterworth high-pass filter (HPF).
    Remove the constant DC offset and slowly varying aperiodic components.
    For complete suppression of the fault DC component, fc < 5 Hz is recommended.
    """
    nyq = 0.5 * fs
    wn = fc / nyq
    sos = signal.butter(order, wn, btype='highpass', output='sos')
    filtered = signal.sosfiltfilt(sos, data, axis=1)
    return filtered, sos

def remove_dc_period(data, fs, f_net):
    """
    Remove the aperiodic component using a centered moving average
    over one grid-frequency period (20 ms at 50 Hz).
    This method introduces no phase distortion and minimizes edge artifacts.
    """
    period = int(fs / f_net)
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        s = pd.Series(data[i])
        aper = s.rolling(window=period, center=True, min_periods=1).mean()
        half = period // 2
        aper.iloc[:half] = aper.iloc[half:period].mean()
        aper.iloc[-half:] = aper.iloc[-period:-half].mean()
        filtered[i] = data[i] - aper.values
    return filtered

try:
    FS = rec.cfg.sample_rates[0][0]
    print(f"Sampling frequency: {FS}")
except (AttributeError, IndexError):
    print("X Sampling frequency not found")

# Centering on pre-fault history (the first 20 ms) before filtering is
# standard practice for removing CT/VT zero offsets
pre_window = int(0.02 * FS)
for i in range(3):
    I_prim[i] -= np.mean(I_prim[i, :pre_window])
    U_prim[i] -= np.mean(U_prim[i, :pre_window])

# === Select the filtering method ===
# Recommended: remove_dc_period (no edge artifacts)
# Alternative: butterworth_highpass (FC=25 Hz — DC offset only)
I_prim_filt = remove_dc_period(I_prim, FS, rec.cfg.frequency)
U_prim_filt = remove_dc_period(U_prim, FS, rec.cfg.frequency)
sos_I = sos_U = None  # SOS are not used for the moving-average method

# Update the filtered variables for subsequent sections
IA_filtered = I_prim_filt[0]
IB_filtered = I_prim_filt[1]
IC_filtered = I_prim_filt[2]
UA_filtered = U_prim_filt[0]
UB_filtered = U_prim_filt[1]
UC_filtered = U_prim_filt[2]

print("✓ Filtering complete (moving average over one period).")
```

::: {.output .stream .stdout}
    Sampling frequency: 5000.0
    ✓ Filtering complete (moving average over one period).
:::
::::

::: {.cell .markdown}
## Butterworth Filter Visualization
:::

:::: {.cell .code execution_count="9"}
``` python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
t_ms = np.array(rec.time) * 1000

# 1. Phase A current: before and after filtering
axes[0, 0].plot(t_ms, I_prim[0], 'gray', alpha=0.5, label='Original')
axes[0, 0].plot(t_ms, I_prim_filt[0], 'r-', label='Filtered')
axes[0, 0].set_ylabel('IA, kA')
axes[0, 0].set_title('Phase A current: before and after filtering')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. DC component (moving average over a 20 ms period)
window_size = int(0.02 * FS)
dc_before = np.convolve(I_prim[0], np.ones(window_size)/window_size, mode='same')
dc_after = np.convolve(I_prim_filt[0], np.ones(window_size)/window_size, mode='same')

axes[1, 0].plot(t_ms, dc_before, 'gray', alpha=0.7, label='DC before')
axes[1, 0].plot(t_ms, dc_after, 'b-', label='DC after')
axes[1, 0].set_ylabel('Mean, kA')
axes[1, 0].set_title('DC component (IA)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 3. Butterworth filter magnitude response (for reference)
sos_ref = signal.butter(ORDER, FC/(0.5*FS), btype='highpass', output='sos')
w, h = signal.sosfreqz(sos_ref, worN=2000, fs=FS)
axes[0, 1].semilogx(w, 20*np.log10(np.abs(h) + 1e-10), 'b-', linewidth=2)
axes[0, 1].axvline(x=FC, color='red', linestyle='--', label=f'fc = {FC} Hz')
axes[0, 1].axvline(x=50, color='green', linestyle='--', label='50 Hz')
axes[0, 1].set_ylabel('Magnitude, dB')
axes[0, 1].set_title('Reference Butterworth HPF magnitude response')
axes[0, 1].legend()
axes[0, 1].grid(True, which='both')
axes[0, 1].set_ylim([-60, 5])

# 4. ALL CURRENTS after filtering
axes[1, 1].plot(t_ms, I_prim_filt[0], 'r-', label='IA', lw=1)
axes[1, 1].plot(t_ms, I_prim_filt[1], 'g-', label='IB', lw=1)
axes[1, 1].plot(t_ms, I_prim_filt[2], 'b-', label='IC', lw=1)
axes[1, 1].set_ylabel('Current, kA')
axes[1, 1].set_xlabel('Time, ms')
axes[1, 1].set_title('All phases after filtering')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](58d36698090c215803140daa2395e9dda38f12ae.png)
:::
::::

::: {.cell .markdown}
## Save Filtered Currents and Voltages Back to COMTRADE
:::

:::: {.cell .code execution_count="10"}
``` python
import numpy as np
from datetime import datetime

file_name = "filtered_data"

# 1. Build the CFG header in strict accordance with IEEE C37.111-1999
station_name = "FILTERED"
rec_dev_id = "PY_SCRIPT"
rev_year = "1999"
nA = 6
nD = 0
tt = nA + nD

cfg_lines = [
    f"{station_name},{rec_dev_id},{rev_year}",
    f"{tt},{nA}A,{nD}D",
]

all_signals = [
    np.array(I_prim_filt[0]), np.array(I_prim_filt[1]), np.array(I_prim_filt[2]),
    np.array(U_prim_filt[0]), np.array(U_prim_filt[1]), np.array(U_prim_filt[2])
]
names = ["IA_filt", "IB_filt", "IC_filt", "UA_filt", "UB_filt", "UC_filt"]
phases = ["A", "B", "C", "A", "B", "C"]
units = ["kA", "kA", "kA", "kV", "kV", "kV"]

for i in range(6):
    sig = all_signals[i]
    ch_id = names[i]
    ph = phases[i]
    ccbm = ""
    uu = units[i]
    a = 1.0
    b = 0.0
    skew = 0
    min_val = int(np.floor(np.min(sig)))
    max_val = int(np.ceil(np.max(sig)))
    primary = 1
    secondary = 1
    ps = "P"
    cfg_lines.append(f"{i+1},{ch_id},{ph},{ccbm},{uu},{a},{b},{skew},{min_val},{max_val},{primary},{secondary},{ps}")

lf = int(rec.cfg.frequency)
nrates = 1
npts = len(all_signals[0])
start_dt = datetime.now().strftime("%d/%m/%Y,%H:%M:%S.%f")[:-3]
trigger_dt = start_dt

cfg_lines.extend([
    str(lf),
    str(nrates),
    f"{int(FS)},{npts}",
    start_dt,
    trigger_dt,
    "ASCII",
    "1"
])

with open(f"{file_name}.cfg", "w", newline="") as f:
    f.write("\n".join(cfg_lines) + "\n")

# 2. Build DAT (ASCII)
dt_us = int(1_000_000 / FS)
with open(f"{file_name}.dat", "w", newline="") as f:
    for i in range(npts):
        t_us = i * dt_us
        vals = ",".join([f"{sig[i]:.6f}" for sig in all_signals])
        f.write(f"{i+1},{t_us},{vals}\n")

print(f"✅ Files created: {file_name}.cfg and {file_name}.dat (IEEE C37.111-1999)")
```

::: {.output .stream .stdout}
    ✅ Files created: filtered_data.cfg and filtered_data.dat (IEEE C37.111-1999)
:::
::::

::: {.cell .code execution_count="11"}
``` python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def analyze_signals_full(data_dict, fs=5000, fault_idx=None, threshold=5.0):
    """
    Analyze the aperiodic component.
    Estimate DC% using the pre-fault history (before fault inception),
    where filtering should completely suppress DC.
    """
    window_size = int(fs * 0.02)  # 20 ms window
    half_w = window_size // 2
    results = {}
    
    names = list(data_dict.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 4 * len(names)))
    if len(names) == 1:
        axes = [axes]

    print(f"{'Channel':<12} | {'DC %':<10} | {'Ta (ms)':<10} | {'Status'}")
    print("-" * 50)

    for i, name in enumerate(names):
        signal_full = np.array(data_dict[name]).flatten()
        
        # Use the pre-fault history or the entire signal to estimate DC
        if fault_idx is not None and fault_idx > window_size:
            eval_sig = signal_full[:fault_idx]
        else:
            eval_sig = signal_full
        
        if len(eval_sig) > window_size:
            aper_eval = uniform_filter1d(eval_sig, size=window_size, mode='nearest')
            sig_eval = eval_sig[half_w:-half_w]
            aper_eval = aper_eval[half_w:-half_w]
        else:
            aper_eval = uniform_filter1d(eval_sig, size=len(eval_sig), mode='nearest')
            sig_eval = eval_sig
        
        ac_eval = sig_eval - aper_eval
        peak_ac = np.max(np.abs(ac_eval))
        max_dc = np.max(np.abs(aper_eval))
        ratio = (max_dc / peak_ac * 100) if peak_ac > 0 else 0
        
        abs_dc = np.abs(aper_eval)
        idx_peak = np.argmax(abs_dc)
        val_peak = abs_dc[idx_peak]
        target = val_peak * 0.368
        ta = 0
        fragment = abs_dc[idx_peak:]
        for j, val in enumerate(fragment):
            if val <= target:
                ta = (j / fs) * 1000
                break
        
        results[name] = {"ratio": ratio, "ta": ta}
        status = "⚠️ HIGH" if ratio > threshold else "✅ NORMAL"
        ta_display = f"{ta:.1f}" if ta > 0 else "> window"
        print(f"{name:<12} | {ratio:>7.2f}% | {ta_display:>9} | {status}")

        ax = axes[i]
        t_full = np.arange(len(signal_full)) / fs * 1000
        aper_full = uniform_filter1d(signal_full, size=window_size, mode='nearest')
        ax.plot(t_full, signal_full, color='gray', alpha=0.4, label='Original signal')
        ax.plot(t_full, aper_full, color='red', linewidth=2, label='DC (aperiodic)')
        ax.plot(t_full, signal_full - aper_full, color='blue', alpha=0.7, label='AC (fundamental)')
        if fault_idx is not None:
            ax.axvline(x=fault_idx/fs*1000, color='magenta', linestyle='--', alpha=0.5, label='Fault')
        ax.axvspan(0, half_w/fs*1000, color='yellow', alpha=0.1)
        ax.axvspan(t_full[-1] - half_w/fs*1000, t_full[-1], color='yellow', alpha=0.1)
        ax.set_title(f"Channel analysis: {name} (DC={ratio:.1f}%, Ta={ta_display} ms)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return results

data_to_test = {
    "IA_filt": np.array(I_prim_filt[0]).flatten(),
    "IB_filt": np.array(I_prim_filt[1]).flatten(),
    "IC_filt": np.array(I_prim_filt[2]).flatten(),
    "UA_filt": np.array(U_prim_filt[0]).flatten(),
    "UB_filt": np.array(U_prim_filt[1]).flatten(),
    "UC_filt": np.array(U_prim_filt[2]).flatten()
}
```
:::

::: {.cell .markdown}
## Section 3: Fault Inception Detection
:::

::::: {.cell .code execution_count="12"}
``` python
def detect_fault_inception(ia, ib, ic, ua, ub, uc, fs, f_net, Inom, eta_I=0.5, eta_U=0.85):
    """Detect fault inception using a combined sliding-window RMS criterion.

    The algorithm compares current and voltage RMS in two adjacent windows
    of one-quarter period each (k = fs/(4*f_net)).
    A fault is detected when the current rises sharply in at least one phase
    (I_post / I_pre > 1 + eta_I) and the voltage drops
    (U_post / U_pre < eta_U).
    """
    k = int(fs / (4 * f_net))
    n = len(ia)

    def window_rms(x, start, end):
        return np.sqrt(np.mean(x[start:end]**2))

    I_ratio = np.ones(n)
    U_ratio = np.ones(n)

    for i in range(k, n - k):
        I_pre = max(window_rms(ia, i - k, i),
                    window_rms(ib, i - k, i),
                    window_rms(ic, i - k, i))
        I_post = max(window_rms(ia, i, i + k),
                     window_rms(ib, i, i + k),
                     window_rms(ic, i, i + k))
        U_pre = min(window_rms(ua, i - k, i),
                    window_rms(ub, i - k, i),
                    window_rms(uc, i - k, i))
        U_post = min(window_rms(ua, i, i + k),
                     window_rms(ub, i, i + k),
                     window_rms(uc, i, i + k))

        I_ratio[i] = I_post / (I_pre + 1e-9) if I_pre > 1e-6 else 1.0
        U_ratio[i] = U_post / (U_pre + 1e-9) if U_pre > 1e-6 else 1.0

    # Skip the first 2 windows to prevent false triggering
    # at the start of the recording (energization, switching surge)
    skip = 2 * k
    candidates = np.where((I_ratio > (1.0 + eta_I)) & (U_ratio < eta_U))[0]
    candidates = candidates[candidates >= skip]

    if len(candidates) == 0:
        return None, I_ratio, U_ratio

    fault_idx = candidates[0]
    return fault_idx, I_ratio, U_ratio

ETA_I = 0.5   # 50% current increase
ETA_U = 0.85  # voltage drops below 85%

fault_idx_detected, I_ratio, U_ratio = detect_fault_inception(
    IA_filtered, IB_filtered, IC_filtered,
    UA_filtered, UB_filtered, UC_filtered,
    data['fs'], data['frequency'], data['Inom'], ETA_I, ETA_U
)

fault_time_detected = fault_idx_detected / data['fs'] if fault_idx_detected is not None else None

print("✓ Fault detection complete")
print(f"  Current threshold: I_post / I_pre > {1.0 + ETA_I:.2f}")
print(f"  Voltage threshold: U_post / U_pre < {ETA_U:.2f}")
if fault_time_detected is not None:
    print(f"  Detected: sample {fault_idx_detected}, time {fault_time_detected*1000:.2f} ms")
    if data['fault_time'] is not None:
        print(f"  True time: {data['fault_time']*1000:.2f} ms")
        print(f"  Error: {abs(fault_time_detected - data['fault_time'])*1000:.2f} ms")
else:
    print("  ⚠️ No fault detected")

# Assess filtering quality using the pre-fault history
results = analyze_signals_full(data_to_test, fs=data['fs'], fault_idx=fault_idx_detected)
```

::: {.output .stream .stdout}
    ✓ Fault detection complete
      Current threshold: I_post / I_pre > 1.50
      Voltage threshold: U_post / U_pre < 0.85
      Detected: sample 5041, time 1008.20 ms
    Channel      | DC %       | Ta (ms)    | Status
    --------------------------------------------------
    IA_filt      |    3.63% | > window | ✅ NORMAL
    IB_filt      |    0.07% | > window | ✅ NORMAL
    IC_filt      |    0.07% | > window | ✅ NORMAL
    UA_filt      |    2.34% | > window | ✅ NORMAL
    UB_filt      |    0.68% | > window | ✅ NORMAL
    UC_filt      |    0.68% | > window | ✅ NORMAL
:::

::: {.output .display_data}
![](a752d64e9ad543bc556da718647e4edbbac61b93.png)
:::
:::::

::::: {.cell .code execution_count="13"}
``` python
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
t_ms = np.arange(len(IA_filtered)) / data['fs'] * 1000

axes[0].plot(t_ms, IA_filtered, 'r-', label='IA')
axes[0].plot(t_ms, IB_filtered, 'g-', label='IB')
axes[0].plot(t_ms, IC_filtered, 'b-', label='IC')
if data['fault_time'] is not None:
    axes[0].axvline(x=data['fault_time']*1000, color='cyan', linestyle='-', linewidth=2, label='True fault inception')
if fault_time_detected is not None:
    axes[0].axvline(x=fault_time_detected*1000, color='magenta', linestyle='--', linewidth=2, label='Detected')
axes[0].set_ylabel('Current, kA')
axes[0].set_title('Fault inception detection')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t_ms, I_ratio, 'k-', label='$I_{post} / I_{pre}$')
axes[1].plot(t_ms, U_ratio, 'b--', label='$U_{post} / U_{pre}$')
axes[1].axhline(y=1.0 + ETA_I, color='red', linestyle='--', label='Current threshold')
axes[1].axhline(y=ETA_U, color='green', linestyle='--', label='Voltage threshold')
if data['fault_time'] is not None:
    axes[1].axvline(x=data['fault_time']*1000, color='cyan', linestyle='-', linewidth=2)
if fault_time_detected is not None:
    axes[1].axvline(x=fault_time_detected*1000, color='magenta', linestyle='--', linewidth=2)
    axes[1].scatter([fault_time_detected*1000], [I_ratio[fault_idx_detected]], color='magenta', s=100)
axes[1].set_ylabel('RMS ratio')
axes[1].set_xlabel('Time, ms')
axes[1].set_title('Current and voltage changes (sliding window = T/4)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

if fault_idx_detected is not None:
    print(f"Maximum current ratio: {np.max(I_ratio):.3f}")
    print(f"Minimum voltage ratio: {np.min(U_ratio):.3f}")
```

::: {.output .display_data}
![](7344852421219c04efd47774323bb79c43bf596b.png)
:::

::: {.output .stream .stdout}
    Maximum current ratio: 2.503
    Minimum voltage ratio: 0.000
:::
:::::

::: {.cell .markdown}
## Section 4: Signal Windowing
:::

:::: {.cell .code execution_count="14"}
``` python
T_PRE = 0.050
T_POST = 0.150

N_PRE = int(T_PRE * data['fs'])
N_POST = int(T_POST * data['fs'])
L_WINDOW = N_PRE + N_POST

if fault_idx_detected is None:
    # If no fault is detected, fall back to the center of the recording
    fault_idx_detected = len(IA_filtered) // 2
    print("⚠️ No fault detected; the window is centered on the middle of the recording")

def extract_window(signal, center, n_pre, n_post):
    """Extract a fixed-length n_pre + n_post window, padding with zeros outside the bounds."""
    start = center - n_pre
    end = center + n_post
    pad_left = max(0, -start)
    pad_right = max(0, end - len(signal))
    valid_start = max(0, start)
    valid_end = min(len(signal), end)
    window = np.concatenate([
        np.zeros(pad_left),
        signal[valid_start:valid_end],
        np.zeros(pad_right)
    ])
    return window

IA_window = extract_window(IA_filtered, fault_idx_detected, N_PRE, N_POST)
IB_window = extract_window(IB_filtered, fault_idx_detected, N_PRE, N_POST)
IC_window = extract_window(IC_filtered, fault_idx_detected, N_PRE, N_POST)
UA_window = extract_window(UA_filtered, fault_idx_detected, N_PRE, N_POST)
UB_window = extract_window(UB_filtered, fault_idx_detected, N_PRE, N_POST)
UC_window = extract_window(UC_filtered, fault_idx_detected, N_PRE, N_POST)

t_window = np.arange(-N_PRE, N_POST) / data['fs'] * 1000

print("✓ Windowing complete")
print(f"  Pre-fault history: {T_PRE*1000:.0f} ms = {N_PRE} samples")
print(f"  Post-fault history: {T_POST*1000:.0f} ms = {N_POST} samples")
print(f"  Total length: {L_WINDOW} samples")
```

::: {.output .stream .stdout}
    ✓ Windowing complete
      Pre-fault history: 50 ms = 250 samples
      Post-fault history: 150 ms = 750 samples
      Total length: 1000 samples
:::
::::

::::: {.cell .code execution_count="15"}
``` python
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(t_window, IA_window, 'r-', label='IA')
axes[0].plot(t_window, IB_window, 'g-', label='IB')
axes[0].plot(t_window, IC_window, 'b-', label='IC')
axes[0].axvline(x=0, color='magenta', linestyle='--', linewidth=2, label='Fault')
axes[0].axvspan(-T_PRE*1000, 0, alpha=0.1, color='green', label='Pre-fault history')
axes[0].axvspan(0, T_POST*1000, alpha=0.1, color='red', label='Post-fault history')
axes[0].set_ylabel('Current, kA')
axes[0].set_title('Currents in the analysis window')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t_window, UA_window, 'r-', label='UA')
axes[1].plot(t_window, UB_window, 'g-', label='UB')
axes[1].plot(t_window, UC_window, 'b-', label='UC')
axes[1].axvline(x=0, color='magenta', linestyle='--', linewidth=2)
axes[1].axvspan(-T_PRE*1000, 0, alpha=0.1, color='green')
axes[1].axvspan(0, T_POST*1000, alpha=0.1, color='red')
axes[1].set_ylabel('Voltage, kV')
axes[1].set_xlabel('Time from fault inception, ms')
axes[1].set_title('Voltages in the analysis window')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

print(f"IA amplitude before the fault: {np.max(np.abs(IA_window[:N_PRE])):.2f} kA")
print(f"IA amplitude after the fault: {np.max(np.abs(IA_window[N_PRE:])):.2f} kA")
```

::: {.output .display_data}
![](89c3d2f954e7eb157bbb80ec76f50fbff32bd165.png)
:::

::: {.output .stream .stdout}
    IA amplitude before the fault: 0.80 kA
    IA amplitude after the fault: 2.84 kA
:::
:::::

::: {.cell .markdown}
## Section 5: Symmetrical Components (Fortescue)
:::

:::: {.cell .code execution_count="16"}
``` python
def fortescue_instantaneous(ia, ib, ic, fs, f_net):
    """Calculate instantaneous symmetrical components using the Fortescue transform."""
    L = len(ia)
    i0 = np.zeros(L)
    i1 = np.zeros(L)
    i2 = np.zeros(L)
    
    k = int(fs / (3 * f_net))
    
    for n in range(L):
        i0[n] = (ia[n] + ib[n] + ic[n]) / 3
        
        idx_b1 = n - k
        idx_c1 = n - 2*k
        if idx_b1 >= 0 and idx_c1 >= 0:
            i1[n] = (ia[n] + ib[idx_b1] + ic[idx_c1]) / 3
        else:
            i1[n] = i0[n]
        
        idx_b2 = n - 2*k
        idx_c2 = n - k
        if idx_b2 >= 0 and idx_c2 >= 0:
            i2[n] = (ia[n] + ib[idx_b2] + ic[idx_c2]) / 3
        else:
            i2[n] = 0
    
    return i0, i1, i2

I0, I1, I2 = fortescue_instantaneous(IA_window, IB_window, IC_window, data['fs'], data['frequency'])
U0, U1, U2 = fortescue_instantaneous(UA_window, UB_window, UC_window, data['fs'], data['frequency'])

print("✓ Symmetrical components calculated")
print(f"  Shift k = {int(data['fs']/(3*data['frequency']))} samples = 120°")
```

::: {.output .stream .stdout}
    ✓ Symmetrical components calculated
      Shift k = 33 samples = 120°
:::
::::

::::: {.cell .code execution_count="17"}
``` python
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

axes[0, 0].plot(t_window, I0, 'purple', label='I₀')
axes[0, 0].plot(t_window, I1, 'blue', label='I₁')
axes[0, 0].plot(t_window, I2, 'orange', label='I₂')
axes[0, 0].axvline(x=0, color='magenta', linestyle='--')
axes[0, 0].set_ylabel('Current, kA')
axes[0, 0].set_title('Symmetrical current components')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(t_window, U0, 'purple', label='U₀')
axes[0, 1].plot(t_window, U1, 'blue', label='U₁')
axes[0, 1].plot(t_window, U2, 'orange', label='U₂')
axes[0, 1].axvline(x=0, color='magenta', linestyle='--')
axes[0, 1].set_ylabel('Voltage, kV')
axes[0, 1].set_title('Symmetrical voltage components')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(t_window, IA_window, 'r-', alpha=0.5, label='IA')
axes[1, 0].plot(t_window, IB_window, 'g-', alpha=0.5, label='IB')
axes[1, 0].plot(t_window, IC_window, 'b-', alpha=0.5, label='IC')
axes[1, 0].plot(t_window, I1, 'k-', linewidth=2, label='I₁')
axes[1, 0].axvline(x=0, color='magenta', linestyle='--')
axes[1, 0].set_ylabel('Current, kA')
axes[1, 0].set_title('Phase currents vs I₁')
axes[1, 0].legend()
axes[1, 0].grid(True)

sum_abc = (IA_window + IB_window + IC_window) / 3
axes[1, 1].plot(t_window, sum_abc, 'gray', alpha=0.7, label='(IA+IB+IC)/3')
axes[1, 1].plot(t_window, I0, 'purple', linewidth=2, label='I₀')
axes[1, 1].axvline(x=0, color='magenta', linestyle='--')
axes[1, 1].set_ylabel('Current, kA')
axes[1, 1].set_title('Check: I₀ = mean')
axes[1, 1].legend()
axes[1, 1].grid(True)

post_fault_idx = N_PRE
I1_rms = np.sqrt(np.mean(I1[post_fault_idx:]**2))
I2_rms = np.sqrt(np.mean(I2[post_fault_idx:]**2))
I0_rms = np.sqrt(np.mean(I0[post_fault_idx:]**2))

ratio_21 = I2_rms / (I1_rms + 1e-10)
ratio_01 = I0_rms / (I1_rms + 1e-10)

axes[2, 0].bar(['I₂/I₁', 'I₀/I₁'], [ratio_21, ratio_01], color=['orange', 'purple'])
axes[2, 0].axhline(y=1.0, color='red', linestyle='--')
axes[2, 0].set_ylabel('Ratio')
axes[2, 0].set_title('Ratios (fault type features)')
axes[2, 0].grid(True, axis='y')

axes[2, 1].axis('off')

plt.tight_layout()
plt.show()

print(f"\nI₁ (RMS): {I1_rms:.3f} kA")
print(f"I₂ (RMS): {I2_rms:.3f} kA")
print(f"I₀ (RMS): {I0_rms:.3f} kA")
print(f"I₂/I₁ = {ratio_21:.3f}")
print(f"I₀/I₁ = {ratio_01:.3f}")
```

::: {.output .display_data}
![](aa79aee8d7e006014707d8399bc19de170f2ae6e.png)
:::

::: {.output .stream .stdout}

    I₁ (RMS): 0.482 kA
    I₂ (RMS): 0.712 kA
    I₀ (RMS): 0.508 kA
    I₂/I₁ = 1.477
    I₀/I₁ = 1.054
:::
:::::

::: {.cell .markdown}
## Section 6: Fault Type Classification
:::

:::: {.cell .code execution_count="18"}
``` python
def classify_fault_type(I1_rms, I2_rms, I0_rms):
    """Fault type classification"""
    ratio_21 = I2_rms / (I1_rms + 1e-10)
    ratio_01 = I0_rms / (I1_rms + 1e-10)
    
    if ratio_21 < 0.15 and ratio_01 < 0.15:
        fault_type = '3ph'
        one_hot = np.array([1, 0, 0, 0])
    elif ratio_21 > 0.7 and ratio_01 < 0.25:
        fault_type = '2ph'
        one_hot = np.array([0, 1, 0, 0])
    elif ratio_21 > 0.7 and ratio_01 > 0.7:
        fault_type = '1ph-G'
        one_hot = np.array([0, 0, 0, 1])
    elif 0.25 < ratio_21 < 0.7 and 0.25 < ratio_01 < 0.7:
        fault_type = '2ph-G'
        one_hot = np.array([0, 0, 1, 0])
    else:
        fault_type = 'undefined'
        one_hot = np.array([0.25, 0.25, 0.25, 0.25])
    
    return fault_type, one_hot, (ratio_21, ratio_01)

fault_type, one_hot, ratios = classify_fault_type(I1_rms, I2_rms, I0_rms)
ratio_21, ratio_01 = ratios

print(f"✓ Classification: {fault_type}")
```

::: {.output .stream .stdout}
    ✓ Classification: 1ph-G
:::
::::

::::: {.cell .code execution_count="19"}
``` python
# Visualize the classification
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].set_xlim([0, 1.5])
axes[0].set_ylim([0, 1.5])

from matplotlib.patches import Rectangle
axes[0].add_patch(Rectangle((0, 0), 0.15, 0.15, facecolor='green', alpha=0.3))
axes[0].add_patch(Rectangle((0.7, 0), 0.8, 0.25, facecolor='blue', alpha=0.3))
axes[0].add_patch(Rectangle((0.7, 0.7), 0.8, 0.8, facecolor='red', alpha=0.3))
axes[0].add_patch(Rectangle((0.25, 0.25), 0.45, 0.45, facecolor='orange', alpha=0.3))

axes[0].scatter([ratio_21], [ratio_01], s=300, c='magenta', marker='*', edgecolors='black', linewidths=2)
axes[0].set_xlabel('I₂/I₁')
axes[0].set_ylabel('I₀/I₁')
axes[0].set_title('Decision regions')
axes[0].grid(True)

labels = ['3ph', '2ph', '2ph-G', '1ph-G']
colors = ['green', 'blue', 'orange', 'red']
bars = axes[1].bar(labels, one_hot, color=colors, edgecolor='black', linewidth=2)

for bar, val in zip(bars, one_hot):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

axes[1].set_ylabel('Probability')
axes[1].set_title(f'Result: {fault_type}')
axes[1].set_ylim([0, 1.2])
axes[1].grid(True, axis='y')

plt.tight_layout()
plt.show()

print(f"One-hot: {one_hot}")
```

::: {.output .display_data}
![](29e3e16fe440aaf124f7b05a3a06b5aeeb25c784.png)
:::

::: {.output .stream .stdout}
    One-hot: [0 0 0 1]
:::
:::::

::: {.cell .markdown}
## Section 7: Per-Unit Normalization (p.u.) {#section-7-per-unit-normalization-pu}
:::

:::: {.cell .code execution_count="20"}
``` python
# =============================================================================
# Per-unit normalization (p.u.)
# =============================================================================
# Set the required values manually:
S_BASE_MVA = 100.0          # Base power, MVA
FAULT_DISTANCE_KM = 10.0    # Fault distance, km (set manually)

LINE_PARAMS = {
    'Unom_kv': 110.0,
    'L_km': 50.0,
    'r1_ohm_km': 0.1,
    'x1_ohm_km': 0.4,
}

U_base_kV = LINE_PARAMS['Unom_kv'] / np.sqrt(3)
I_base_kA = S_BASE_MVA / (np.sqrt(3) * LINE_PARAMS['Unom_kv'])

print("=== BASE QUANTITIES ===")
print(f"U_base = {U_base_kV:.2f} kV (phase voltage)")
print(f"I_base = {I_base_kA:.3f} kA")
print(f"Z₁ = {np.sqrt((LINE_PARAMS['r1_ohm_km']*LINE_PARAMS['L_km'])**2 + (LINE_PARAMS['x1_ohm_km']*LINE_PARAMS['L_km'])**2):.2f} ohm")

IA_pu = IA_window / I_base_kA
IB_pu = IB_window / I_base_kA
IC_pu = IC_window / I_base_kA
UA_pu = UA_window / U_base_kV
UB_pu = UB_window / U_base_kV
UC_pu = UC_window / U_base_kV

I0_pu = I0 / I_base_kA
I1_pu = I1 / I_base_kA
I2_pu = I2 / I_base_kA
U0_pu = U0 / U_base_kV
U1_pu = U1 / U_base_kV
U2_pu = U2 / U_base_kV

if FAULT_DISTANCE_KM is not None and LINE_PARAMS['L_km'] > 0:
    d_target_pu = FAULT_DISTANCE_KM / LINE_PARAMS['L_km']
    print(f"\nTarget: {FAULT_DISTANCE_KM:.1f} km = {d_target_pu:.3f} p.u.")
else:
    d_target_pu = None
    print("\nTarget: not set (FAULT_DISTANCE_KM = None)")
```

::: {.output .stream .stdout}
    === BASE QUANTITIES ===
    U_base = 63.51 kV (phase voltage)
    I_base = 0.525 kA
    Z₁ = 20.62 ohm

    Target: 10.0 km = 0.200 p.u.
:::
::::

::::: {.cell .code execution_count="21"}
``` python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(t_window, I1_pu, 'b-', label='I₁')
axes[0, 0].plot(t_window, I2_pu, 'orange', label='I₂')
axes[0, 0].plot(t_window, I0_pu, 'purple', label='I₀')
axes[0, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].axvline(x=0, color='magenta', linestyle='--')
axes[0, 0].set_ylabel('Current, p.u.')
axes[0, 0].set_title('Currents in p.u.')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(t_window, U1_pu, 'b-', label='U₁')
axes[0, 1].plot(t_window, U2_pu, 'orange', label='U₂')
axes[0, 1].plot(t_window, U0_pu, 'purple', label='U₀')
axes[0, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[0, 1].axvline(x=0, color='magenta', linestyle='--')
axes[0, 1].set_ylabel('Voltage, p.u.')
axes[0, 1].set_title('Voltages in p.u.')
axes[0, 1].legend()
axes[0, 1].grid(True)

categories = ['I₁', 'I₂', 'I₀', 'U₁', 'U₂', 'U₀']
after = [np.max(np.abs(I1_pu)), np.max(np.abs(I2_pu)), np.max(np.abs(I0_pu)),
         np.max(np.abs(U1_pu)), np.max(np.abs(U2_pu)), np.max(np.abs(U0_pu))]

x = np.arange(len(categories))
axes[1, 0].bar(x, after, color=['blue', 'orange', 'purple', 'blue', 'orange', 'purple'])
axes[1, 0].set_ylabel('Amplitude, p.u.')
axes[1, 0].set_title('Maximum values in p.u.')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(categories)
axes[1, 0].grid(True, axis='y')

if d_target_pu is not None:
    axes[1, 1].barh(['Distance'], [d_target_pu], color='green', edgecolor='black', height=0.3)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_xlabel('p.u.')
    axes[1, 1].set_title(f'Target: {d_target_pu:.3f} p.u.')
    axes[1, 1].text(d_target_pu/2, 0, f'{FAULT_DISTANCE_KM:.1f} km',
                    ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    axes[1, 1].set_yticks([])
else:
    axes[1, 1].text(0.5, 0.5, 'Target not set', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
    axes[1, 1].set_title('Target: not set')

plt.tight_layout()
plt.show()

print(f"Max I₁ after the fault: {np.max(np.abs(I1_pu[N_PRE:])):.2f} p.u.")
print(f"Min U₁ after the fault: {np.min(np.abs(U1_pu[N_PRE:])):.2f} p.u.")
```

::: {.output .display_data}
![](a8967d378cbe617f80fcb7ce736fe7185ae5ab2c.png)
:::

::: {.output .stream .stdout}
    Max I₁ after the fault: 1.63 p.u.
    Min U₁ after the fault: 0.00 p.u.
:::
:::::

::: {.cell .markdown}
## Section 8: Build the Final Tensor

Tensor `X` contains 12 channels:

- Currents: `IA, IB, IC` (phase) + `I₁, I₂, I₀` (symmetrical components)
- Voltages: `UA, UB, UC` (phase) + `U₁, U₂, U₀` (symmetrical components)
- Target variable: `y` --- fault distance in p.u.

The fault type is **not included** in the tensor. Classification is performed
separately, and its result is used to select the appropriate model from
the ensemble.
:::

:::: {.cell .code execution_count="22"}
``` python
X = np.array([
    IA_pu, IB_pu, IC_pu,
    I1_pu, I2_pu, I0_pu,
    UA_pu, UB_pu, UC_pu,
    U1_pu, U2_pu, U0_pu
])

y = d_target_pu if d_target_pu is not None else np.nan

print("✓ Final tensor created")
print(f"  X shape: {X.shape} (channels × time)")
print("  Channels: IA, IB, IC, I₁, I₂, I₀, UA, UB, UC, U₁, U₂, U₀")
if d_target_pu is not None:
    print(f"  Target variable y: {y:.4f} p.u.")
else:
    print("  Target variable y: not set (set FAULT_DISTANCE_KM)")
```

::: {.output .stream .stdout}
    ✓ Final tensor created
      X shape: (10, 1000) (channels × time)
      Target variable y: 0.2000 p.u.
:::
::::

::::: {.cell .code execution_count="23"}
``` python
# Visualize the final tensor
fig, axes = plt.subplots(6, 2, figsize=(14, 18))
axes = axes.flatten()

channel_names = [
    'IA (phase A current)', 'IB (phase B current)', 'IC (phase C current)',
    'I₁ (positive-sequence current)', 'I₂ (negative-sequence current)', 'I₀ (zero-sequence current)',
    'UA (phase A voltage)', 'UB (phase B voltage)', 'UC (phase C voltage)',
    'U₁ (positive-sequence voltage)', 'U₂ (negative-sequence voltage)', 'U₀ (zero-sequence voltage)'
]

colors = ['#d62728', '#2ca02c', '#1f77b4',
          'blue', 'orange', 'purple',
          '#d62728', '#2ca02c', '#1f77b4',
          'blue', 'orange', 'purple']

for i in range(12):
    axes[i].plot(t_window, X[i], color=colors[i], linewidth=1.5)
    axes[i].axvline(x=0, color='magenta', linestyle='--', alpha=0.7)
    axes[i].set_ylabel(f'Channel {i}')
    axes[i].set_title(channel_names[i])
    axes[i].grid(True)

fig.suptitle(f'Final tensor X ∈ R^(12×{L_WINDOW})\nTarget: {y:.3f} p.u.',
             fontsize=14, fontweight='bold')

for i in [10, 11]:
    axes[i].set_xlabel('Time from fault inception, ms')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print(f"\n{'Channel':<8} {'Name':<25} {'Range':<20}")
print("-" * 55)
for i, name in enumerate(channel_names):
    min_val, max_val = np.min(X[i]), np.max(X[i])
    print(f"{i:<8} {name:<25} [{min_val:6.2f}, {max_val:6.2f}]")

print(f"\nTarget variable: y = {y:.4f} p.u. = {y*100:.1f}% of the line length")
```

::: {.output .display_data}
![](d8515f579fe4ab7046b9ef5335f2d3686be9813c.png)
:::

::: {.output .stream .stdout}

    Channel  Name                      Range
    -------------------------------------------------------
    0        I₁ (positive-sequence current) [ -1.60,   1.63]
    1        I₂ (negative-sequence current) [ -2.34,   2.36]
    2        I₀ (zero-sequence current) [ -1.72,   1.72]
    3        U₁ (positive-sequence voltage) [ -0.97,   1.66]
    4        U₂ (negative-sequence voltage) [ -2.93,   2.93]
    5        U₀ (zero-sequence voltage) [ -1.54,   1.74]
    6        one-hot: 3ph              [  0.00,   0.00]
    7        one-hot: 2ph              [  0.00,   0.00]
    8        one-hot: 2ph-G            [  0.00,   0.00]
    9        one-hot: 1ph-G            [  1.00,   1.00]

    Target variable: y = 0.2000 p.u. = 20.0% of the line length
:::
:::::

:::: {.cell .code execution_count="24"}
``` python
output_data = {
    'X': X,
    'y': y,
    'metadata': {
        'fault_type': fault_type,
        'fault_distance_km': FAULT_DISTANCE_KM,
        'line_length_km': LINE_PARAMS['L_km'],
        'sample_rate': data['fs'],
        'window_samples': L_WINDOW,
        'channel_names': channel_names,
        'tensor_description': 'X ∈ R^(12 × L_window): [IA, IB, IC, I1, I2, I0, UA, UB, UC, U1, U2, U0]'
    }
}

np.savez('fault_sample_processed.npz', **output_data)
print("✓ Saved to 'fault_sample_processed.npz'")
```

::: {.output .stream .stdout}
    ✓ Saved to 'fault_sample_processed.npz'
:::
::::
