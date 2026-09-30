# COMTRADE fault detection and distance

The imported `fault-distance` repository is integrated into **Event Lab** and **Notebook**. The original repository, notebook and recordings are preserved. No trained checkpoint, Hugging Face package, model download or external service is used.

## Browser workflow

1. Open **Event Lab → COMTRADE recordings**. The catalog discovers the matching `.cfg`/`.dat` pairs in `fault-distance/data/data_test`. Select individual recordings or all 99, then import. An unchanged source pair reuses its existing dataset version.
2. Choose **Open in Event Lab**, or select an imported recording in the Recording menu. The six synchronized traces show native samples. The **Fault inception + distance** adapter is selected automatically for COMTRADE data.

   Event Lab defaults to **1A_val1**, the bundled notebook's example, when no recording was explicitly selected. **Use notebook example** selects that recording and resets the detector settings to the notebook defaults. Its apparent distance is **0.8909 km**. **1A_val99 is a different recording**: the same notebook calculation returns 87.1208 km with the example impedances. Explicit recording selections and bookmarked dataset URLs are respected. Each distance result names its source recording.
3. Verify the phase mapping, current direction and line parameters. Defaults reproduce the notebook examples: 110 kV line voltage, 100 MVA base, 50 km length, R1 = 0.1 Ω/km and X1 = 0.4 Ω/km. These values are **unverified examples**. Supply both R0 and X0 to compensate ground-loop current, or leave both blank for an explicitly uncompensated estimate.
4. Run the detector. The result lists fault candidates, onset, heuristic fault type, selected loop, apparent distance, individual measured cycles and tensor padding. Selecting a candidate navigates to its waveform. **Export fault results and tensors** downloads configuration, provenance, annotations, event CSV, filtered phase CSV, per-cycle distance CSV, NPZ tensors and an HTML report.
5. For detection evaluation, annotate reviewed intervals and independently established **Fault inception** points. Unreviewed detections are excluded. A manual onset can be supplied for distance calculation, but it is not counted as a detector prediction and detection metrics are disabled for that run.

The model configuration can be saved as an Event Lab preset. Additional ASCII recordings can be placed as matching pairs in the imported repository's `data/data_test` directory; the catalog refreshes automatically. This catalog is specifically an RTDS simulation source. General CSV/Parquet recording imports remain available separately.

## Notebook

In **Notebook → Open notebook**, select **fault_distance.ipynb**. Set `RECORD_NAME` in the first code cell, verify `LINE_PARAMS` in setup, then run all cells. `await recordings()` lists the available names. Import the selected source through Event Lab first so notebook reads use its frozen source version.

The notebook banner identifies the default `RECORD_NAME = "1A_val1"` and links directly to that recording in Event Lab. The data source selector prepares a code snippet; it does not replace `RECORD_NAME` in the notebook.

This is the complete imported 42-cell notebook, adapted for the local browser workspace: Google Drive and installation cells have been replaced with a local data loader; file paths use browser storage; the original filtering, plots, sequence features, tensor export, distance calculation and line diagram are retained. The original notebook at `fault-distance/fault_distance.ipynb` is unchanged. The adaptation records its source checksum in notebook metadata.

The pure-Python COMTRADE 0.1.2 reader and its MIT license are bundled alongside the notebook. NumPy, pandas, SciPy and Matplotlib come from the locally hosted Pyodide distribution. Outputs appear in `fault_outputs/<record name>` in JupyterLite's file browser. Download notebooks and output artifacts to retain them independently of browser storage. The notebook requires contiguous recordings; Event Lab handles separate contiguous segments around gaps.

## Data and timing

- The 99 supplied RTDS simulations each contain 400 samples at 2 kHz, with three current and three voltage waveforms at a declared 50 Hz grid frequency. These are simulations, not real-data validation or independently labeled fault locations.
- CFG calibration `a × raw + b` is applied in float64. Source kA/kV are stored as A/V. The pipeline converts to kA/kV before using the notebook's equations. The original units, gains, offsets, phase tokens, instrument ratios and channel skew are retained in dataset metadata.
- DAT timestamps are read directly; they are not regenerated from the sampling rate. Native Parquet uses integer nanoseconds, and gaps remain gaps. Originals are retained both in the imported repository and as frozen copies alongside the dataset.
- RTDS 1991 dates use month/day/year. The supplied two-digit years are interpreted as 2000 + YY in an in-memory CFG copy. Original CFG bytes are unchanged. The reported start and trigger clocks remain in metadata; neither is a verified fault label.
- The recording timezone is unknown. UTC epoch timestamps in storage are **elapsed-time reference coordinates**, not the actual recording date. `clock_verified=false` and the clock explanation accompany notebook reads and experiment exports. Do not align these coordinates with AMI or another recording's absolute clock.
- Each source has a stable recording-local terminal identity; physical identity across separate recordings is unverified. Do not infer asset independence for machine-learning splits from these IDs.
- This import preset accepts ASCII CFG/DAT with a single positive declared sample rate, explicit ordered timestamps and current/voltage analog channels in A/kA/V/kV. Binary/multi-rate inputs and other analog quantities are rejected explicitly. Digital states, if present, are retained in native storage and the original pair; these detectors use the six selected analog channels. Missing/misaligned samples, unsupported units and nonzero channel skew are never silently repaired.

## Method and limits

The reviewed numerical functions in `di_validator/faults/numerics.py` come from the imported notebook:

1. Center phase waveforms using the first electrical cycle, then subtract a centered one-cycle moving mean.
2. Compare RMS in adjacent quarter-cycle windows. Default inception requires the maximum phase-current RMS to rise above 1.5 times its preceding value while minimum phase-voltage RMS falls below 0.85 times its preceding value. The first qualifying candidate after the startup exclusion is selected per contiguous segment.
3. Calculate centered one-cycle fundamental sequence peak magnitudes using the Fortescue transform. Sequence ratios classify a heuristic fault type. AUTO selects the largest-current ground phase, the two largest-current phases, or positive sequence; uncertain types require an explicit loop.
4. Fit RMS phasors by least squares to measured post-fault cycles after the configured settling interval. Apparent distance is `imag(U_loop / I_loop) / X1_per_km`. Ground loops use `I_phase + k0 × (IA + IB + IC)`, with `k0 = (Z0 − Z1) / (3 × Z1)` when zero-sequence impedance is supplied. Report the median across usable cycles without clipping to line length.
5. Export the notebook's 12-channel per-unit tensor in `[IA, IB, IC, UA, UB, UC, I1, I2, I0, U1, U2, U0]` order. Padding is identified in metadata and excluded from distance cycles. Sequence magnitudes at segment edges use endpoint extension, as in the notebook.

This is **offline** analysis: centered filtering and post-fault cycles need future samples. Candidate onset and the time the complete segment became available are kept separately; real-time latency is not claimed. Processing resets at missing observations and time gaps. Select an analysis window of at most 200,000 native samples; temporary sequence windows are bounded in memory. Results are independent of Parquet read chunk size.

Distance can be unavailable when there is no inception, insufficient post-fault data, low loop current or an uncertain AUTO loop. A source filename is never interpreted as distance ground truth. Cycle-to-cycle spread is not a confidence interval. Line impedance error, instrument/reference-direction error, fault resistance, load and remote infeed can affect an apparent single-ended estimate. The notebook links the [SEL single-ended fault-location reference](https://selinc.com/api/download/4912/?lang=en); COMTRADE parsing follows the pinned [python-comtrade implementation](https://github.com/dparrini/python-comtrade).

## Python and API

```python
from di_validator.events import run

result = run({
    "dataset_id": "<imported recording ID>",
    "algorithm": "fault-distance",
    "parameters": {
        "fault_loop": "AG",
        "line_length_km": 50,
        "r1_ohm_km": 0.1,
        "x1_ohm_km": 0.4,
        "line_parameters_verified": False,
    },
})
```

Use `POST /api/v1/events/run` for a background job with frozen annotation, dataset and pipeline identities. COMTRADE discovery/import endpoints are `GET /api/v1/faults/sources` and `POST /api/v1/faults/imports` with `source_ids`. Original pairs are downloadable through `/api/v1/faults/sources/{id}/cfg` and `/dat`; the notebook explicitly requests the normalized CFG view. API schemas are in `/docs`.

Verification tests are in `tests/test_faults.py` and `frontend/e2e/faults.spec.ts`. They cover all 99 source pairs, native timing/calibration, original-notebook numerical parity, all seven fault loops, gaps, chunk boundaries, manual onset, review coverage, provenance exports and complete local notebook execution.
