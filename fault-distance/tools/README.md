# Tools — Project utilities

Each script here is a standalone CLI utility. Run it directly:
`python tools/<script>.py [args]`

## Categories

### Data processing (`data_*`)

| Script | Description | Dependencies |
|---|---|---|
| `data_comtrade_to_csv.py` | Converts COMTRADE (.cfg + .dat) to the project's CSV format. Reads settings from `data_comtrade_config.ini`. | `comtrade` (optional) |

**Configuration:** `data_comtrade_config.ini` — line parameters (`line_length_km`, `Unom`, `R1_ohm/km`, `X1_ohm/km`).

### Debugging / visualization (`debug_*`)

| Script | Description | Dependencies |
|---|---|---|
| `debug_fortescue.py` | Minimal check of the Fortescue matrix using synthetic data (no CSV or FFT). | numpy |
| `debug_inspect_symseq.py` | Visual inspection of symmetrical components using an actual CSV file. Plots sliding window → FFT → Fortescue. | numpy, pandas, matplotlib |

### Result export (`export_*`)

| Script | Description | Dependencies |
|---|---|---|
| `export_symseq_to_comtrade.py` | Exports symmetrical components (sliding window) back to COMTRADE (.cfg + .dat ASCII). | numpy, pandas |
