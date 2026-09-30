# Release verification

Verified on this Windows workstation on September 19, 2026: Python 3.13, 14 logical CPU cores, 31.5 GiB RAM. Dependencies are frozen in the Python and frontend lockfiles.

## Supplied data

All six original measurement files and the PV registry have been imported into the local workspace. SHA-256 source checksums were compared after import; the originals are unchanged.

| File | Transformer profiles | Source timestamp rows | Rows with resolved timestamps | Audited clock exclusions |
| --- | ---: | ---: | ---: | ---: |
| AMI_CKT_218_14401.csv | 368 | 15,312 | 15,308 | 4 |
| AMI_CKT_245_16925.csv | 688 | 8,760 | 8,758 | 2 |
| AMI_CKT_3250_02485.csv | 673 | 15,312 | 15,308 | 4 |
| AMI_CKT_3818_09625.csv | 1,303 | 15,312 | 15,308 | 4 |
| AMI_CKT_3921_04109.csv | 751 | 15,312 | 15,308 | 4 |
| AMI_CKT_4835_16761.csv | 576 | 15,312 | 15,308 | 4 |
| **Total** | **4,359** | | | **22** |

There are 62,237,232 source reading positions and 62,221,172 positions with resolved timestamps. These counts include missing measurement values; they are not counts of nonmissing readings. Quality reports retain those distinctions. The 22 excluded timestamp rows remain documented in source audits. Interval-start versus interval-end remains unknown for these sources.

Machine-readable reconciliation: `runtime/data-verification.json`.

## Automated and workflow checks

- **27 backend/reference tests passed.** Coverage includes DST gaps and repeated hours, normalized identifier collisions, missing and negative readings, kWh conversion, incomplete meter aggregation, multi-circuit PV evidence, grouped splits, frozen labels, training-only selection, export contents, event chunk invariance, waveform derivation, gaps, duplicate detections, reviewed intervals, annotation withdrawal, cancellation, and restart recovery.
- Python lint and the production TypeScript/Vite build passed.
- **12 browser checks passed** across 1440 × 1000 desktop, 820 × 1180 tablet, and 390 × 844 phone layouts. They cover navigation, import controls, real-data explorer views, label review, native event zoom, replay, annotation controls, classifier configuration, prediction review, and provenance. The mobile event panel was corrected so its controls remain visible and clickable.
- A classifier was configured and submitted through the browser, completed in the separate worker, and opened in the result inspector. It used 664 labeled transformer profiles from CKT 245, with 531 training and 133 holdout assets. Two independently configured runs retained the same split and evidence identities.
- A review interval was saved from the phone layout and a new detector run completed from the same interface. The downloaded classifier ZIP was inspected for fitted pipelines, PNG/SVG plots, evidence, predictions, split membership, configuration, environment metadata, and its HTML report.
- The supplied-data runs use the explicit **exploratory registry evidence versus provisional negatives** policy. Their scores do not establish accuracy against independently confirmed positive and negative field labels.
- Services were stopped and restarted with the Windows scripts; imported data and completed results persisted. An isolated worker test verifies that a previously running job becomes interrupted and retains its saved configuration.

## Full-day recording benchmark

The retained fixture contains four 1 kHz channels over 24 hours. A deliberate one-second gap leaves **86,399,000 native sample instants**, or 345,596,000 channel values. Raw data is processed in bounded chunks.

| Measurement | Result |
| --- | ---: |
| Fresh recording generation | 272.84 s |
| Peak process RSS during fresh generation, queries, and detection | 0.308 GiB |
| Detection using the retained native recording | 25.70 s |
| Indexed chart query function, across four windows and three repetitions | 0.09–0.38 s |
| API route, response validation, and JSON serialization | 0.22–1.31 s |
| Largest displayed trace in the API checks | 6,668 points/channel |

The API measurements use FastAPI's TestClient and exclude network transit and browser rendering. Detector timing excludes worker startup and dependency imports. RSS measures the computational process, not the entire workstation or browser. First query calls and repeated calls met the two-second target in these runs; all plotted traces stayed below 10,000 points per channel.

Benchmark artifacts are in `runtime/benchmark/`: `fresh-generation-benchmark.json`, `benchmark.json`, and `api-benchmark.json`. The cached index can be rebuilt independently of the immutable sample files.

The five deliberately injected events were matched without duplicates or missed events. This checks the synthetic fixture and implementation, not field detection performance. Real recordings and independently reviewed event labels are still needed for that assessment.

## Current boundaries

- Historical pickle/H5 artifacts are inventoried but remain disabled because their environments and input contracts are unverified.
- Event baselines require one channel per measurement kind and compatible waveform phase pairs. Multi-phase algorithms can be added through a Python adapter. RMS derivation is an engineering baseline, not a certified power-quality instrument.
- No live device acquisition or advanced spectral analysis is included. Browser Python editing and hosted deployment are covered by the additions below.
- Tests currently emit non-failing dependency deprecation warnings for the test client and pandas/NumPy timedelta interoperability.

## Notebook addition

The Notebook tab serves JupyterLite 0.8.3 with the 0.8.6 Pyodide kernel and its matching Pyodide 314.0.6 runtime from localhost. The `comm` wheel needed for kernel startup is also bundled. Standard data analysis requires no external requests after the local build.

Eight targeted backend tests passed (three new data-access tests and five existing API/lifecycle tests). They cover stable pagination across assets with identical timestamps, preservation of negative/missing values, recording channel selection, exact nanosecond timestamps and gaps, all-missing numeric columns, request bounds, cursor validation, registered-source downloads, and existing API behavior. Python lint and the production frontend build passed.

Both starter notebooks were executed end to end against the imported AMI data and synthetic 1 kHz recording. This includes plotting, quality audits, registry access, paginated processing, custom event scoring, and annotations. **Three browser checks passed** at desktop (1440 × 1000), tablet (820 × 1180), and phone (390 × 844) widths, with external requests blocked. They verify native AMI and recording reads, Matplotlib output, persistence after reload, retained kernel identity across application tabs, and automatic file-browser collapse for narrow notebook panels.

Notebook documents are stored in browser storage, not in the application's SQLite backup. Download `.ipynb` files for portable backups. The browser Python runtime is separate from the workstation's worker environment.

## Power anomaly source addition

`data/power_anomaly_dataset_2022.csv` contains 1,300 rows and nine columns. Reported timestamps span 2025-10-10 16:38:02.049675 through 16:59:41.049675 at exactly one-second spacing. There are no missing cells or duplicate timestamps. The supplied labels are Normal (900), Cyber_Spike (200), Gradual_Drift (100), and Voltage_Sag (100). Timezone, provenance, asset level, and voltage/current measurement type remain unverified, so the file is available as a source table in Notebook rather than a normalized Event Lab recording.

Source SHA-256 remains `3e764e301a36a93acdb23242bb97cd50077a4c4ba6e8e6ab066d9a99bb76d0c6`. The source-file API returns the original bytes; the browser helper records this checksum and preserves all columns without localizing timestamps. The anomaly starter notebook includes timing and label audits, plots, and a custom score with a chronological development baseline. It does not promote source labels to confirmed annotations.

Five targeted data-access tests passed, including source-file path confinement, the 20 MiB read limit, byte preservation, and a regression check that automatic registration does not apply the AMI preset to the anomaly table. Python lint, the JupyterLite build, and the production frontend build passed.

Three source-notebook browser checks passed on desktop, tablet, and phone with external requests blocked. All five code cells executed, including CSV loading, checksum capture, timing and label inspection, Matplotlib plots, and custom scoring. The source selector and data guide remained usable without horizontal page overflow.

## Forecasting addition

The full Python suite passed: **37 tests**. Five forecasting tests cover holdout isolation, reproducible forecasts and saved-model reuse, source snapshots, missing-data policy, negative values, unsupported metrics, artifact checksums, export contents, label-target exclusion, and rejection of excluded model IDs. Forecasting Python lint, the production TypeScript/Vite build, and the local JupyterLite build passed. A broad script lint scan also found an existing semicolon-style warning in `scripts/benchmark.py`; forecasting files are clean.

Actual fit and reload checks passed for **eight providers**: seasonal naive, Random Forest, XGBoost, LightGBM, Prophet, GluonTS DeepAR, AutoGluon tree ensembles, and NeuralProphet. Each active interpreter was checked and contains neither `huggingface_hub` nor `transformers`. GluonTS sampling is seeded independently of fitting so saved-model replay is repeatable. These small fixtures check implementation and persistence, not predictive quality. Evidence is saved at `runtime/verification/forecasting/16a47567851948b6929fcdc894e15a5d/results.json`.

A worker benchmark also completed on the existing hourly AMI dataset `AMI_CKT_4835_16761`, asset `1552405E`, with 300 source observations and a 24-step holdout. Its source checksum, net kWh units, timezone, and unknown interval-position convention are retained. The anomaly source file remains unchanged and available for source-clock forecasting without promoting its labels to verified evidence.

**Four browser checks passed**: the complete benchmark/export/saved-inference workflow at desktop, tablet, and phone widths, plus execution of the forecasting notebook with all external requests blocked. The notebook check runs once on desktop; its duplicate tablet/phone cases are intentionally skipped. All three notebook code cells completed, including local worker submission, metrics retrieval, and Matplotlib output. Checks also verify source-label exclusion, the model allowlist, and absence of horizontal page overflow. Browser testing caught and resolved a stale-result race after submitting a new job and a missing explicit plotting import in Pyodide.

Chronos, TimesFM, and TabPFN adapters are excluded. AutoGluon uses only its Tabular tree allowlist. Deletion of two abandoned setup environments (`runtime/forecast-runtimes/torch` and `runtime/forecast-runtimes/autogluon`) was blocked by automatic approval review. They remain unused; the active groups are `gluonts`, `autogluon_trees`, and `neuralprophet` plus the main application environment.

## Optional forecasting interpolation

**Nine forecasting tests passed**, including linear interpolation at hourly and 1 kHz resolution, negative values, omitted timestamps, gap-length limits, coverage requirements, refusal to extrapolate at context/origin boundaries, and rejection before fitting when any required history is unsuitable. Changing unseen holdout values does not change validation selection or holdout predictions. Evaluation targets remain missing; exports preserve source values and include per-window preprocessing summaries plus every estimated training value.

**Three browser workflows passed** on desktop, tablet, and phone. They import a small generated fixture with missing readings, enable/disable the checkbox, preserve the limit across reload and preset selection, complete a worker forecast, inspect filled-value counts and observed-target metric counts, and check for page overflow and JavaScript errors. Screenshots are in `runtime/verification/interpolation-*.png`. The production frontend build and forecasting Python lint also passed. Interpolation is opt-in, defaults to gaps of at most three steps, and retains the original observed-coverage requirement.

## COMTRADE and fault-distance integration

**53 Python tests passed:** 41 application tests (`python -m pytest tests -q`) and 12 preserved reference tests (`python -m pytest reference/test_ev_pv_detection.py -q`). The 12 new fault tests cover every supplied CFG/DAT pair, calibrated native samples and exact timestamps, source preservation, original-notebook numerical/tensor agreement, AG/BG/CG/AB/BC/CA/POS distance calculations, missing observations, DAT gaps and truncation, arbitrary read chunk boundaries, unavailable estimates, manual onset, frozen provenance, reviewed-interval evaluation and export contents.

**Six complete browser workflows passed:** COMTRADE import/detection/distance/export and full notebook execution, each on desktop, tablet and phone. The notebook executes all 26 code cells in the 42-cell adaptation, produces the 12 × 400 tensor and distance result, and saves its output artifacts. External HTTP requests were blocked; the successful notebook runs requested no external services. Three additional Event Lab checks passed after refining the offline caption, completed-job status and unverified-clock display in the dataset catalog. The production frontend build and fault-module lint passed.

The live import job completed for **99 recordings, 39,600 sample instants and 237,600 analog readings**. `python -m scripts.verify_faults` verified all current original pairs against their frozen checksums, exact 500,000 ns sample spacing, and the unchanged original notebook checksum. All 99 recordings produced apparent-distance estimates with the notebook's example parameters. The audit is saved in `runtime/verification/comtrade-inventory.json`; browser screenshots are `runtime/verification/fault-events-*.png` and `fault-notebook-*.png`.

For `1A_val1`, the notebook and Event Lab identify onset at 0.0995 s and an AG uncompensated apparent distance of approximately **0.890890 km**. Agreement with the original saved result is within 1e-6 km; differences arise from retaining float64 COMTRADE calibration instead of the original reader's float32 default. This demonstrates pipeline agreement, not fault-location accuracy: line values are unverified examples, recording clocks lack a verified timezone, and no independent fault-distance ground truth was supplied. The initial library-loading problem in the browser notebook was fixed with explicit local Pyodide package loading before the complete browser checks passed.

After the final frontend build and API restart, three desktop regression workflows passed: the complete COMTRADE workflow, the original AMI/recording notebook data-access and session-persistence workflow, and switching from COMTRADE back to baseline detectors with native zoom, replay and annotation controls. A live indexed window request returned all six channels and 400 samples per channel in 0.047 seconds on this workstation; this is a small-recording response measurement, not a large-recording benchmark.

## Fault-distance recording selection correction

The reported distance mismatch was a recording-selection mismatch. The latest user run selected `1A_val99` (87.120830 km with the example impedances), while the saved notebook selects `1A_val1` (0.890890 km). Executing the original notebook's phasor/distance functions on the application's filtered samples gave identical results for `1A_val1`, `1A_val50`, and `1A_val99`; the diagnostic values are in `runtime/verification/fault-distance-discrepancy.json`.

Event Lab now defaults to the notebook's `1A_val1` recording, respects explicit dataset URLs, and provides **Use notebook example** to restore that recording and the notebook parameter defaults. Results display their source recording. The Notebook tab identifies its example record and links directly to the matching Event Lab dataset. Numerical algorithms and original recordings were not changed. The production frontend build and all **12 fault-analysis Python tests** passed. A live browser inspection confirmed `Recording: 1A_val1` and **0.8909 km**, with no browser errors or horizontal overflow; its screenshot is `runtime/verification/fault-distance-corrected.png`.

All **nine browser tests passed**: import/detection/export, default and explicit recording selection with back/forward navigation, and complete local notebook execution, each at desktop, tablet and phone widths. All three notebook executions reproduced 0.8909 km with external HTTP requests blocked.

## Hosted deployment — September 21, 2026

The private Sites frontend connects through API Gateway to Google Cloud Run in `thtank/us-west2`. The migrated workspace reports **112 datasets, 4,465 assets, and 62,936,212 readings**, including all 99 COMTRADE recordings. Sources and result artifacts reside in the private Cloud Storage bucket; metadata and jobs use the dedicated PostgreSQL database. The workstation workspace remains independent. Resource names and update instructions are in `DEPLOYMENT.md`.

The full Python/reference suite passed **56 tests with one PostgreSQL integration test skipped** after local Docker became unavailable. Before that interruption, all three targeted cloud tests passed, including real PostgreSQL transactions, rollback, job claiming, and exclusive worker ownership. The production frontend build and gateway routing/authentication checks passed. The remote container build succeeded with the optional CPU forecasting environments installed and checked for the absence of `huggingface_hub` and `transformers`.

Actual Cloud Run jobs completed training, validation, and holdout evaluation for all nine enabled forecasting options: seasonal naive, Random Forest, XGBoost, LightGBM, Prophet, GluonTS DeepAR, AutoGluon tree ensemble, NeuralProphet, and the equal-weight ensemble. Saved Random Forest inference and forecast ZIP export also passed. These small deployment checks establish operational behavior, not comparative model quality. A fault-analysis job reproduced **0.8908903568462712 km** for `1A_val1`; its ZIP manifest and report were inspected.

After deploying a new Cloud Run revision, the migrated data, completed jobs, and saved results remained available, and a new queued job completed successfully. The service reported ready with all traffic on `di-validator-api-00002-fsj`. Backend results and persistence evidence are retained in `runtime/cloud-deploy/backend-verification.json`.

Access checks confirmed that anonymous Sites API requests return 401, direct unauthenticated Cloud Run requests return 403, and API Gateway rejects calls missing either the restricted API key or the application token. The site remains owner-only. API Gateway uses managed Google identity to invoke Cloud Run; no service-account private keys or public IAM bindings were created.

Three hosted browser workflows passed against the private Sites origin: API job submission/prediction review/saved inference/exports; navigation, native charts, and responsive layouts at 1440, 820, and 390 pixels; and JupyterLite Python execution with native cloud data access. The notebook imported `di_data`, constructed a pandas frame from 17 AMI readings, fetched all 400 native samples from `1A_val1`, and displayed `DI_CLOUD_NOTEBOOK_OK 17 400` without Python errors. Browser isolation headers fixed an initial notebook filesystem issue. Deep links preserve their application routes. The rendered Event Lab shows the correct recording and **0.8909 km** with populated waveforms.

The verified frontend publication is **Sites version 4**, built from source commit `75460fab46cb4186056e0e16cc9bdd8240a111d9`. Deployment identifiers are in `runtime/cloud-deploy/deployment.json`; browser logs and screenshots are under the same directory. The hosted gateway has a **32 MB request/response limit**; larger source uploads and artifacts use the private bucket as described in `DEPLOYMENT.md`. Notebook documents persist in the browser and should be downloaded for backup.
