# Stage 1: Centering using the pre-fault history (center_by_prehistory)

## Purpose
Remove the signal's constant offset using the first 20 ms of the recording (the pre-fault history).

## Mathematics

For a single-channel signal `x[n]` of length `N`:

```
pre_window = round(fs · 0.020)   // number of samples in 20 ms

μ = (1/pre_window) · Σ_{n=0}^{pre_window-1} x[n]   // pre-fault mean

x̃[n] = x[n] - μ,   n = 0, 1, ..., N-1
```

## For a multichannel signal
If `X` is a matrix of shape `(C, N)` (C channels, N samples):

```
for each channel c = 0, ..., C-1:
    μ_c = (1/pre_window) · Σ_{n=0}^{pre_window-1} X[c, n]
    X̃[c, n] = X[c, n] - μ_c
```

## Parameters
| Parameter | Value | Description |
|---|---|---|
| `pre_ms` | 20.0 | Pre-fault history length [ms] |
| `fs` | from CSV | Sampling frequency [Hz] |

## Example
For `fs = 5000` Hz:
- `pre_window = round(5000 · 0.020) = 100` samples
- Compute the mean of the first 100 samples
- Subtract it from the entire signal

## Properties
- **Linear operation**: preserves the signal shape
- **No phase distortion**: does not use filtering
- **Removes DC offset**: centers the signal around zero
- **Does not affect amplitude**: applies only a vertical shift
