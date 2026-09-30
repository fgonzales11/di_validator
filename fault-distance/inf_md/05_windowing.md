# Stage 4: Cropping a window around t₀ (crop_around_t0)

## Purpose
Extract a relevant, fixed-length waveform segment around the fault inception time.

## Mathematics

### Window sizes in samples
```
pre_samp  = round(pre_fault_ms  · 10⁻³ · fs)
post_samp = round(post_fault_ms · 10⁻³ · fs)
```

### Window boundaries
```
start = max(0, t₀ - pre_samp)
end   = min(N, t₀ + post_samp)
```

### Window extraction
```
window = x[start:end]   // shape (T_crop, C)
```

### Resampling to the target length
If `target_length` is specified and `T_crop ≠ target_length`:
```
window_resampled = resample(window, target_length, axis=0)
```

where `resample` is linear interpolation via scipy.signal.resample.

### Position of t₀ in the new window
```
total = pre_samp + post_samp
frac = pre_samp / total
t₀_local = round(frac · (target_length - 1))
```

## Parameters
| Parameter | Value | Description |
|---|---|---|
| `pre_fault_ms` | 50.0 | Pre-fault history [ms] |
| `post_fault_ms` | 150.0 | Post-fault history [ms] |
| `target_length` | 400 | Target length [samples] |
| `fs` | from CSV | Sampling frequency [Hz] |

## Example
For `fs = 2000` Hz:
- `pre_samp = round(50 · 0.001 · 2000) = 100` samples
- `post_samp = round(150 · 0.001 · 2000) = 300` samples
- `T_crop = 100 + 300 = 400` samples = 200 ms
- This matches `SEQ_LENGTH = 400` → no resampling needed

For `fs = 5000` Hz:
- `pre_samp = 250`, `post_samp = 750`
- `T_crop = 1000` samples
- Resample to 400 samples

## Pad / Trim (fallback)
If t₀ is not detected:
```
if T < SEQ_LENGTH:
    pad with zeros to SEQ_LENGTH
if T > SEQ_LENGTH:
    keep only the first SEQ_LENGTH samples
```

## Properties
- **Fixed output length**: always `SEQ_LENGTH` samples
- **Preserves proportions**: t₀ occupies a fixed position (~25% from the start)
- **Resampling**: adapts to different sampling frequencies
