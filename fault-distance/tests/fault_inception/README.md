# Testing fault inception time detection (t0)

## Directory contents

```
tests/fault_inception/
├── README.md          ← this file
├── check_t0.py        ← manual verification script
└── oscillograms/      ← place your .cfg + .dat files here
```

## How to run

```bash
# From the repository root:
python tests/fault_inception/check_t0.py
```

## What the script does

1. Scans `tests/fault_inception/oscillograms/` for `.cfg` files.
2. Parses each COMTRADE file and extracts the phase A, B, C currents.
3. Runs `detect_t0_multi_phase` from `data/fault_inception.py`.
4. Prints a table to the console:

```
File                     | t0 (sample) | t0 (ms)   | fs (Hz)
-------------------------|-------------|-----------|--------
cutAB_5km_t10ms.cfg      |       500   |  10.000   | 50000
cutAC0_10km_t25ms.cfg    |      1250   |  25.000   | 50000
...
```

5. Prints the total success/error counts after the table.

## Input file format

Standard COMTRADE ASCII: `.cfg` + `.dat` in the same directory.\
The sampling frequency is read from the `.cfg` header.

## Tuning the algorithm

If the detector misses the fault inception time, edit the parameters at the top of `check_t0.py`:

```python
PARAMS_OVERRIDE = dict(
    coarse_top_k     = 5,      # number of D4 peaks to consider
    coarse_window_ms = 200.0,  # search half-width around the coarse estimate, ms
    pre_fault_ms     = 20.0,   # affects cropping only, not detection
    post_fault_ms    = 60.0,
    threshold_mult   = 1.0,    # smaller values trigger earlier
)
```
