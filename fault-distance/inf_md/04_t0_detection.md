# Stage 3: Fault inception detection (detect_t0_rms)

## Purpose
Automatically determine the fault inception time (t₀) from changes in the RMS currents and voltages.

## Mathematics

### Analysis window
```
k = fs / (4 · f_net)   // window length = one quarter of a mains cycle [samples]
```

### Window RMS
```
RMS(x, start, end) = sqrt( (1/(end-start)) · Σ_{n=start}^{end-1} x[n]² )
```

### Ratios at each position i
For each sample `i` (from `k` to `N-k`), compute:

**Currents** (maximum across the three phases):
```
I_pre(i)  = max( RMS(ia, i-k, i),   RMS(ib, i-k, i),   RMS(ic, i-k, i) )
I_post(i) = max( RMS(ia, i, i+k),   RMS(ib, i, i+k),   RMS(ic, i, i+k) )

I_ratio[i] = I_post(i) / (I_pre(i) + ε)   if I_pre > 10⁻⁶
           = 1.0                           otherwise
```

**Voltages** (minimum across the three phases):
```
U_pre(i)  = min( RMS(ua, i-k, i),   RMS(ub, i-k, i),   RMS(uc, i-k, i) )
U_post(i) = min( RMS(ua, i, i+k),   RMS(ub, i, i+k),   RMS(uc, i, i+k) )

U_ratio[i] = U_post(i) / (U_pre(i) + ε)   if U_pre > 10⁻⁶
           = 1.0                           otherwise
```

### Fault detection condition
```
A fault is detected at position i if:
    I_ratio[i] > (1 + η_I)    AND    U_ratio[i] < η_U
```

where:
- `η_I = 0.5` — current rise threshold (current increased by more than 50%)
- `η_U = 0.85` — voltage drop threshold (voltage decreased by more than 15%)

### Final t₀
```
skip = 2 · k   // skip the first 2 windows to avoid false detections

candidates = { i | I_ratio[i] > 1.5  AND  U_ratio[i] < 0.85  AND  i ≥ skip }

t₀ = min(candidates)   // first detection
```

## Parameters
| Parameter | Value | Description |
|---|---|---|
| `fs` | from CSV | Sampling frequency [Hz] |
| `f_net` | 50.0 | Mains frequency [Hz] |
| `η_I` | 0.5 | Current rise threshold |
| `η_U` | 0.85 | Voltage drop threshold |

## Example
For `fs = 5000` Hz, `f_net = 50` Hz:
- `k = 5000 / (4 · 50) = 25` samples = 5 ms
- `skip = 50` samples = 10 ms
- Condition: current increased by >50% AND voltage decreased by >15%

## Properties
- **Two-criterion check**: current + voltage (reduces false detections)
- **Adaptability**: relative thresholds (ratios rather than absolute values)
- **Robustness**: skipping the first 2 windows protects against startup transients
