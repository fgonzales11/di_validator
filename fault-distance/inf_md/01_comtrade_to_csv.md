# Stage 0: COMTRADE → CSV conversion

## Source format
COMTRADE (IEEE Std C37.111) consists of two files:
- `.cfg` — configuration (channel names, units, sampling frequency)
- `.dat` — binary or ASCII sample data

## Output CSV format
After conversion by `tools/data_comtrade_to_csv.py`, each CSV file contains:

| Column | Description | Units |
|---|---|---|
| `time` | Time since the start of the recording | s |
| `CT1IA` | Phase A current | A |
| `CT1IB` | Phase B current | A |
| `CT1IC` | Phase C current | A |
| `S1) BUS1UA` | Phase A voltage | kV |
| `S1) BUS1UB` | Phase B voltage | kV |
| `S1) BUS1UC` | Phase C voltage | kV |
| `distance_km` | Distance to the fault | km (constant within the file) |
| `fs_hz` | Sampling frequency | Hz (constant within the file) |

## Mathematical model of a CSV signal
Each channel is a signal sampled in time:

```
x[n] = x(t=n·Ts),  n = 0, 1, ..., N-1
```

where:
- `Ts = 1/fs` — sampling interval [s]
- `N` — total number of samples in the file
- `fs` — sampling frequency [Hz]
<!--
## Example -->
For `fs = 5000` Hz and `N = 2000` samples:
- `Ts = 0.0002` s = 0.2 ms
- Recording duration: `T = N·Ts = 2000 · 0.0002 = 0.4` s = 400 ms

## Notes
- `fs_hz` is written to every CSV row as a constant
- This allows `dataset.py` to read `fs` directly from the file
- Older CSV files without `fs_hz` use the fallback `cfg.SAMPLING_FREQ_HZ`
