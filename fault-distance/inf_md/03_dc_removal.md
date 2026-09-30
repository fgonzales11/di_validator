# Stage 2: Removing the aperiodic (DC) component (remove_dc_period)

## Purpose
Remove the slowly decaying aperiodic component (DC offset) caused by a short circuit by subtracting a moving average over one mains cycle.

## Mathematics

### Mains cycle in samples
```
period = round(fs / f_net)
```

where:
- `fs` — sampling frequency [Hz]
- `f_net` — mains frequency (50 or 60 Hz)

### Moving average (rolling mean)
For each sample `n`, compute the mean over a window of length `period` centered on `n`:

```
aper[n] = (1/period) · Σ_{k=n-half}^{n+half} x[k]
```

where `half = period // 2`.

At the boundaries (the first and last `half` samples), extrapolate the mean from the nearest full window.

### Result
```
y[n] = x[n] - aper[n]
```

## Implementation (pandas)
```python
s = pd.Series(x)
aper = s.rolling(window=period, center=True, min_periods=1).mean()
# Boundary conditions
aper.iloc[:half] = aper.iloc[half:period].mean()
aper.iloc[-half:] = aper.iloc[-period:-half].mean()
y = x - aper.values
```

## Parameters
| Parameter | Value | Description |
|---|---|---|
| `fs` | from CSV | Sampling frequency [Hz] |
| `f_net` | 50.0 | Mains frequency [Hz] |

## Example
For `fs = 5000` Hz, `f_net = 50` Hz:
- `period = round(5000 / 50) = 100` samples
- Window: 100 samples = 20 ms = 1 cycle at 50 Hz
- Subtract the moving average from the original signal

## Properties
- **No phase distortion**: symmetric window (center=True)
- **Preserves the fundamental component**: the window spans one mains cycle
- **Removes the aperiodic component**: a one-cycle moving average suppresses the DC component
- **Equivalent to a high-pass filter** with a cutoff frequency of ~`f_net`

## Comparison with Butterworth
| | `remove_dc_period` | Butterworth high-pass |
|---|---|---|
| Phase distortion | None | Present (nonlinear phase response) |
| Amplitude distortion | Minimal | Depends on order and fc |
| Computational complexity | O(N·period) | O(N·log N) or O(N·order) |
| Parameters | fs, f_net | fs, fc, order, type |
