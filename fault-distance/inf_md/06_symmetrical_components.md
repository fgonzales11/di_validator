# Stage 5: Symmetrical components (sliding_window_symseq)

## Purpose
Compute time-varying symmetrical-component magnitudes (positive, negative, and zero sequence) for currents and voltages.

## Mathematics

### 5.1. Sliding DFT window

For each time index `t` (from 0 to T-1):

```
window_len = period · window_cycles   // period = fs/f_net, typically window_cycles=1
half = window_len // 2

start = max(0, t - half)
end   = min(T, start + window_len)

// Adjust the boundaries
if end - start < window_len and start > 0:
    start = end - window_len

w = sig[:, start:end]   // (6, W) — 6 channels, W samples in the window
```

### 5.2. Windowed FFT

Apply a Hann window:

```
hann[n] = 0.5 · (1 - cos(2πn / (W-1))),   n = 0, ..., W-1

w_windowed = w · hann   // elementwise multiplication
```

DFT:
```
X[k] = Σ_{n=0}^{W-1} w_windowed[n] · e^(-j·2π·k·n/W)
```

Fundamental frequency bin:
```
k = round(f0 · W / fs)
```

### 5.3. Phasor normalization

```
phasor = X[k] · (2 / Σ hann)
```

where:
- `X[k]` — complex DFT coefficient at the fundamental frequency
- `Σ hann` — sum of the window coefficients (amplitude normalization)
- The factor of 2 converts the two-sided spectrum to peak amplitude

### 5.4. Fortescue transform

For a three-phase system of phasors `Va, Vb, Vc`:

```
         [1   1     1   ] [Va]
[V0]   1 [1   a     a²  ] [Vb]
[V1] = - [1   a²    a   ] [Vc]
[V2]   3

where a = e^(j·2π/3) = -0.5 + j·√3/2
```

### 5.5. Symmetrical-component magnitudes

For currents (channels 0-2):
```
i0, i1, i2 = abc_to_seq(phasor_IA, phasor_IB, phasor_IC)
|I1|[t] = |i1|   // positive sequence
|I2|[t] = |i2|   // negative sequence
|I0|[t] = |i0|   // zero sequence
```

For voltages (channels 3-5):
```
u0, u1, u2 = abc_to_seq(phasor_UA, phasor_UB, phasor_UC)
|U1|[t] = |u1|
|U2|[t] = |u2|
|U0|[t] = |u0|
```

## Result
```
symseq = [|I1|, |I2|, |I0|, |U1|, |U2|, |U0|]   // shape (6, T)
```

## Final 12-channel tensor
```
tensor = [IA, IB, IC, |I1|, |I2|, |I0|, UA, UB, UC, |U1|, |U2|, |U0|]
         // shape (12, T)
```

## Parameters
| Parameter | Value | Description |
|---|---|---|
| `fs` | from CSV | Sampling frequency [Hz] |
| `f0` | 50.0 | Fundamental frequency [Hz] |
| `window_cycles` | 1 | Number of cycles in the DFT window |

## Example
For `fs = 2000` Hz, `f0 = 50` Hz:
- `period = round(2000/50) = 40` samples
- `window_len = 40` samples = 20 ms = 1 cycle
- `k = round(50 · 40 / 2000) = 1` — fundamental frequency bin

## Properties
- **Time-varying**: each sample has its own symmetrical-component magnitudes
- **Hann window**: reduces spectral leakage
- **Fortescue transform**: a standard method for analyzing unbalance
- **Phase shift invariance**: magnitudes do not depend on absolute phase
