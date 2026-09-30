# DI Validator

A local browser workbench for transformer and meter measurements, reproducible EV/PV classification, time-series forecasting, browser notebooks, and native-resolution event detection. Source files stay intact. SQLite stores metadata and frozen experiment definitions; compressed Parquet stores interval readings and recording chunks.

## Run on Windows

Prerequisites: **Python 3.13**, **Node.js 24**, and **uv**. Dependency versions are pinned in `uv.lock` and `frontend/package-lock.json`.

Double-click **Start-DI-Validator.cmd**, or run:

```powershell
.\scripts\start.ps1
```

Open **http://127.0.0.1:8765**. The launcher installs pinned dependencies, builds the UI, and starts hidden API and worker processes. To use an existing build, pass `-NoBuild`; to avoid opening a browser, pass `-NoBrowser`. Stop with `scripts\stop.ps1`. Typed API documentation is at `/docs`.

On the Datasets page choose **Register available files**. The seven supplied CSVs are registered as six versioned measurement datasets and one PV registry. Imports run one at a time and can be monitored, cancelled, or rerun in **Activity**. Failed or cancelled imports never enter the dataset catalog. Unfinished jobs are marked interrupted after a worker restart. Work is stored under `runtime/` (override with `DI_WORKSPACE` in both API and worker environments).

For development:

```powershell
uv sync --frozen
uv run python -m di_validator.worker
# In another terminal:
uv run uvicorn di_validator.api:app --host 127.0.0.1 --port 8765
# In another terminal:
cd frontend
npm ci
npm run dev
```

The services bind to loopback. This release has no shared-server authentication or live device acquisition.

## Workflows

**Meter Lab.** Replay validated scenarios through PVDetectionAgent or FaultLocationAgent in a separate HW 4.2 ARM emulator. Inspect the live data flow, synchronized inputs and ARM diagnostics, actual DataServer outcomes, saved runs, comparisons, and evidence exports. Requires local Windows and Ubuntu-22.04 WSL; hosted deployments cannot control it. See [Meter Lab setup and validation](meter_emulator/METER_LAB.md).

**Forecasting.** Train and compare seasonal naive, Random Forest, XGBoost, LightGBM, Prophet, GluonTS DeepAR, NeuralProphet, AutoGluon tree models, and equal-weight ensembles. Inspect chronological validation and a separate final holdout, export fitted models, and reuse saved models for future inference. No Hugging Face-dependent models are included. The supplied anomaly CSV is available directly as an unverified source table. See [FORECASTING.md](FORECASTING.md) for runtimes, evaluation rules, and the JupyterLite forecasting example.

**Datasets → Explorer.** Import CSV or Parquet, declare asset level and clock/measurement conventions, inspect the quality audit, and compare up to eight profiles. Views include time series, hourly shape, seasonal shape, calendar heatmaps, and distributions. The plot toolbar exports images; browser data is bounded to 10,000 displayed points per channel. The source view retains negative net energy and missing observations.

**Explorer → Review labels.** Record confirmed presence, absence, or a withdrawal of evidence with effective dates and a source note. Every edit is a new revision. Bulk label CSV columns are `asset_id,target,value,evidence,valid_from,valid_to,verification`; blank values mean unknown and verification is `confirmed` or `provisional`. Asset IDs have the form `transformer:CKT_245_16925:1544655E`.

**Experiments.** Select hourly datasets, dates, evidence policy, models, and parameters. All runs include a majority baseline, an asset-group holdout, and training-only cross-validation. Selection ranks training CV average precision, then balanced accuracy. Optional threshold selection uses training out-of-fold predictions. Unknown labels do not enter training or reported classification metrics. Fewer than five labeled assets in either class, or insufficient independent groups, produces an actionable failed-job message instead of a misleading metric. Grouped holdout size approximates the requested fraction when group sizes differ.

Each run freezes datasets, labels, relationships, feature settings, registry version, parameters and random seed. Exported ZIP bundles contain feature and quality tables, label evidence, split/fold assignments, fold scores, predictions, fitted pipelines, an environment manifest, and an HTML report. Related assets stay in one partition, and identical profiles are excluded across the cohort. Scores from class-weighted models are uncalibrated; SVM scores are ranking margins. The selected model is frozen before holdout evaluation.

**Event Lab.** Generate a synthetic fixture or import a recording. Inspect synchronized traces, zoom from a min/max overview into native samples, replay at selectable speed, add events and reviewed intervals, and run causal detectors. Load steps use available power (kW), derived voltage/current power (kW), or current RMS (A). Voltage events use voltage RMS; raw voltage is converted with a causal nearest-whole-sample cycle window. These are engineering baselines, not standards-certified power-quality measurements. Multi-phase waveform calculations require separate compatible channel pairs; the baseline uses the declared channel types and phase match. A gap or nonfinite required sample resets detector state.

One-to-one matching requires the same asset/class and onset within the chosen tolerance (100 ms by default). Unreviewed areas and gaps are excluded from the denominator. Duplicate detections are false positives. Point-event duration overlap is undefined; interval-event overlap is recorded in the match details. Onset and emission time are separate. Entire assets/recordings belong to a development or evaluation partition; annotation edits cannot mix partitions. Synthetic evidence is always identified separately.

## Notebook workspace

Open the **Notebook** tab to use a locally hosted JupyterLite Python workspace. Starter notebooks cover hourly profiles and native recording analysis. Select a dataset and choose **Copy data code** to prepare a DataFrame query, or open **Data guide** for instructions. The workspace stays mounted when switching application tabs, preserving its running kernel. **Open in new tab** provides the full JupyterLite interface.

Unmapped CSV files in `data/` also appear under **Source files** in the Notebook selector. `from di_sources import read_csv` followed by `await read_csv("filename.csv")` reads all original columns without assigning a timezone, units, asset level, or label verification. These source reads are limited to 20 MiB; larger files require an import and paged reads. **03 - Power anomaly exploration.ipynb** demonstrates the supplied anomaly table: 1,300 one-second samples dated October 10, 2025, with 900 normal and 400 anomalous source labels. The filename's “2022” is retained. Event Lab registration awaits the source clock and measurement metadata. **Register available files** applies the known AMI and PV presets only; other layouts require explicit mapping through **Import data**.

Python runs in the browser using Pyodide, independently of the workstation's Python worker. The `di_data.DIClient` helper reads the current dataset catalog, native readings, quality audits, manual label revisions, PV registry versions, annotations, experiments, and predictions from the local API. Native reading queries are capped at 100,000 rows per page and use stable cursors; `di.iter_frames()` traverses larger selections. Interval queries require explicit assets and accept ISO timestamps; recording queries accept seconds from origin and optional channel selection. Missing data is preserved and no chart downsampling is applied. Units, clock conventions, provenance, and synthetic status are stored in `DataFrame.attrs`.

Notebooks are saved in **browser storage for the current hostname and port**, separately from `runtime/catalog.sqlite`. Download `.ipynb` files from JupyterLite's File menu or file browser for backups and sharing. The notebook file browser can also upload additional local files. It does not mount Windows paths or the complete AMI archive. Bundled NumPy, pandas, SciPy, scikit-learn, and Matplotlib are served locally; installing additional compatible packages may require internet access.

The launcher builds JupyterLite automatically, downloading its matching Pyodide distribution (about 350 MB) on the first build. Later launches reuse it. To build or refresh the notebook workspace manually:

```powershell
uv run python -m scripts.build_notebooks
```

The generated site is in `runtime/jupyterlite`, with downloaded assets cached under `runtime/jupyterlite-cache`. Builds do not replace notebooks already saved in browser storage. Example source files are under `notebooks/files`. This follows JupyterLite's documented [local Pyodide hosting](https://jupyterlite.readthedocs.io/en/stable/howto/pyodide/pyodide.html) and [kernel file access](https://jupyterlite.readthedocs.io/en/stable/howto/content/python.html) model.

## Data conventions

- Existing AMI files contain **net kWh per hourly interval at transformer level**, with Pacific local clock labels. Positive net consumption is import, negative is export.
- Naive timestamps are localized to the declared timezone. Invalid, spring-gap, ambiguous fall-back, and duplicate timestamps are audited. Unresolved rows are excluded from analysis, not shifted or averaged. Source rows remain in the original file.
- The existing files do not document interval-start versus interval-end timestamps. Imports retain `unknown` until declared; aggregation blocks unknown or mismatched conventions. Reading coverage includes missing UTC intervals.
- Each wide AMI CSV uses a timestamp column and one column per asset. Quotes around asset IDs are normalized. Recording tables use a timestamp column plus explicitly declared channel columns, or offsets/sample indices with a recording origin.
- Source channel names `t_ns` and `offset` are reserved. Absolute sample instants are stored as int64 UTC nanoseconds; browser plots use seconds relative to the recording origin.
- Registry joins use **circuit + STRUCT_NUM**, only at transformer level. PV evidence requires qualifying technology/status, PTO by feature-window start, and no contract end within the window. The registry update date is a conservative cutoff assumption, not proof of complete coverage.
- Nonmatches remain unknown in standard mode. **Exploratory provisional negatives** are opt-in. Circuits missing entirely from the registry remain unknown even in exploratory mode. EV heuristic experiments report agreement with rules, not verified EV ownership.
- The historical reference scripts and binary models are preserved under `reference/`. Legacy artifacts remain unverified and disabled until their training environment, feature schema, granularity, and channel requirements are established.

## Meter aggregation

Use **Datasets → Asset relationships** to upload an effective-dated inventory CSV:

```csv
meter_id,transformer_id,valid_from,valid_to
meter:C1:M1,transformer:C1:T1,2025-01-01,2025-12-31T23:59:59
meter:C1:M2,transformer:C1:T1,2025-01-01,2025-12-31T23:59:59
```

Include the complete expected inventory, including meters with missing readings. Time intervals for one meter must not overlap. Selected sources must share units, channels, duration, timezone, and explicit interval start/end convention. Aggregation stores expected/contributing meters and coverage with every total; the default requires all expected meters. A lower coverage threshold is an explicit partial-total policy, not imputation.

## Python extension interface

Use `di_validator.ingest.import_dataset`, `di_validator.classification.freeze/run`, `di_validator.events.Detector/run`, and `di_validator.query` directly from notebooks. API schemas are in `di_validator.schemas`.

For a new classifier, create a trusted local Python module:

```python
from sklearn.ensemble import ExtraTreesClassifier
from di_validator.adapters import register_classifier

def build(seed):
    return ExtraTreesClassifier(random_state=seed, n_jobs=1)

register_classifier("extra-trees", "Extra trees", build, parameters={"n_estimators"})
```

Set `DI_PLUGINS` to the importable module name in both API and worker environments before startup. The example module is `examples.custom_algorithms`. The engine wraps the estimator in training-only imputation/scaling and applies the same grouped evaluation. Estimators must support scikit-learn cloning, binary prediction, and either `predict_proba` or `decision_function`. Plugin IDs and their supported parameters appear in `/api/v1/algorithms` and are accepted in experiment configurations.

Event extensions use `register_event_detector(identifier, name, factory, required_channel_kinds)`. A factory receives dataset metadata and parameters and returns an object implementing `EventProcessor`: `process_chunk(frame)` plus an `events` list. Frames contain `t_ns`, `offset`, and native channel columns. Each event supplies `id,asset_id,kind,start,end,emitted_at`; time values are seconds from recording start. Preserve causal state across chunks, reset at gaps, and expose an event when confirmed. Update its end while an interval event continues. Set the adapter ID through the `algorithm` field of `/api/v1/events/run`. Worker adapters remain trusted local Python modules. Custom code written in the Notebook tab executes in the separate browser runtime.

## Verification

See [CONTRIBUTING.md](CONTRIBUTING.md) for the code map, extension boundaries, and formatting commands.

```powershell
uv run pytest
uv run ruff check di_validator tests
cd frontend
npm run build
npx playwright install chromium
npm run test:e2e
```

Browser tests expect the application running on port 8765. They exercise desktop/tablet/mobile navigation and key workflows. Run the independent 24-hour, four-channel, 1 kHz benchmark with:

```powershell
uv run python -m scripts.benchmark
# Include fresh recording generation in the measured run:
uv run python -m scripts.benchmark --fresh
```

It writes `runtime/benchmark/benchmark.json`, measures process RSS and cold/warm chart queries, checks the display-point limit, and runs native-sample detectors. Targets are under 8 GiB peak RSS and under two seconds for indexed chart windows. The generated recording is retained in the isolated benchmark workspace for repeatable checks. Imports and synthetic generation build a chunk index and consolidated min/max overview; `uv run python -m scripts.index_recordings` rebuilds these derived indexes without changing native samples.

See [VERIFICATION.md](VERIFICATION.md) for workstation measurements and the supplied-data reconciliation. Completed examples in the workspace include a PV exploratory comparison and a synthetic event evaluation. Their evidence policies are visible in the result and export; neither establishes accuracy against independently confirmed field labels.

Keep `data/` and `reference/` as source material. Back up `runtime/catalog.sqlite` together with `runtime/datasets` and `runtime/experiments` while the services are stopped. Metadata schema version 1 uses additive initialization; future breaking migrations must be explicit.

## COMTRADE fault-distance integration

Event Lab includes the imported RTDS COMTRADE catalog and a **Fault inception + distance** adapter. Import individual recordings or all 99, inspect the six native waveforms, configure the line impedance, run fault analysis and export its evidence and tensors. In Notebook, choose **fault_distance.ipynb** from **Open notebook** to run the full imported pipeline locally. See [FAULT_DISTANCE.md](FAULT_DISTANCE.md) for the workflow, timing conventions, method and limitations. Default line values are unverified examples; ground-loop estimates without R0/X0 are explicitly uncompensated.

## Hosted deployment

The hosted configuration uses a private OpenAI Sites frontend with a Google Cloud Run API and worker, PostgreSQL metadata, and a private Cloud Storage workspace. See [DEPLOYMENT.md](DEPLOYMENT.md) for resources, deployment commands, credential handling, migration, and recovery. Local and hosted workspaces are independent after the initial copy.
