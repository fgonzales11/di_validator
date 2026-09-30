# Stage 6: Normalization

## Two normalization modes

### Mode 1: Standard (statistical)

#### Signals (per-channel StandardScaler)
For each channel `c`:

```
μ_c = (1/(N·T)) · Σ_{i=0}^{N-1} Σ_{t=0}^{T-1} X[i, c, t]

σ_c = sqrt( (1/(N·T)) · Σ_{i=0}^{N-1} Σ_{t=0}^{T-1} (X[i, c, t] - μ_c)² )

X_norm[i, c, t] = (X[i, c, t] - μ_c) / (σ_c + 10⁻⁸)
```

where:
- `N` — number of examples in the dataset
- `T` — sequence length (SEQ_LENGTH)
- `c` — channel index

#### Distance (MinMaxScaler)
```
d_min = min(d_i)
d_max = max(d_i)

d_norm[i] = (d_i - d_min) / (d_max - d_min + 10⁻⁸)
```

#### Inverse transform
```
d_km = d_norm · (d_max - d_min) + d_min
```

---

### Mode 2: Per-Unit (physical)

#### Base quantities
```
Ubase = Unom · 1000 / √3          // phase voltage [V]

Ibase = S_base · 10⁶ / (√3 · Unom · 1000)   // base current [A]
```

where:
- `Unom` — nominal line-to-line voltage [kV]
- `S_base` — base power [MVA]

#### Signal normalization
```
// Phase currents (channels 0-2)
I_pu = I_A / Ibase

// For a 6-channel tensor:
U_pu = U_kV / Unom

// For a 12-channel tensor:
// Phase voltages (channels 6-8)
U_pu = U_kV / Unom

// Sequence-component current magnitudes (channels 3-5)
|I1|_pu = |I1| / Ibase
|I2|_pu = |I2| / Ibase
|I0|_pu = |I0| / Ibase

// Sequence-component voltage magnitudes (channels 9-11)
|U1|_pu = |U1| / Unom
|U2|_pu = |U2| / Unom
|U0|_pu = |U0| / Unom
```

#### Distance normalization
```
d_pu = d_km / L_km
```

where `L_km` is the line length [km].

#### Inverse transform
```
d_km = d_pu · L_km
```

## Parameters
| Parameter | Value | Description |
|---|---|---|
| `NORMALIZATION_MODE` | 'standard' | 'standard' or 'pu' |
| `Unom` | 110.0 kV | Nominal voltage |
| `L_km` | 50.0 km | Line length |
| `S_base` | 100.0 MVA | Base power |

## Mode comparison
| | Standard | Per-Unit |
|---|---|---|
| Requires fitted scalers | Yes | No |
| Depends on the dataset | Yes | No |
| Physical interpretation | No | Yes |
| Applicability to new lines | Limited | Universal |
| Preserves amplitude ratios | No | Yes |
