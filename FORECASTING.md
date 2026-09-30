# Forecasting

Open **Forecasting** at http://127.0.0.1:8765/forecasting. Choose an indexed dataset, asset, and channel, or a small numeric source CSV from `data/`. Configure a horizon in samples, training context, lag features, season length, chronological validation windows, and model parameters. Save reusable presets, run a benchmark, inspect the holdout and future chart, and export the fitted models and provenance.

## Available models

| Model | Implementation | Environment |
|---|---|---|
| Seasonal naive | Repeat the latest season | Main application |
| Random Forest | Recursive regression on past lags and calendar features | Main application |
| XGBoost | Recursive lag regression | Main application |
| LightGBM | Recursive lag regression | Main application |
| Prophet | Additive/multiplicative seasonality with trend | Main application |
| GluonTS | DeepAR trained locally on the selected series | Isolated Python 3.13 |
| AutoGluon | Tabular GBM, RF, XT, XGB ensemble on lag features | Isolated Python 3.13 |
| NeuralProphet | Local autoregression, trend, seasonality | Isolated Python 3.11 |
| Ensemble | Equal-weight mean of successful selected models | Main application |

**No models requiring Hugging Face are included.** Chronos, TimesFM, and TabPFN are excluded. AutoGluon uses Tabular with an explicit tree-model allowlist; its TimeSeries foundation models are not installed. All supported models fit locally without pretrained checkpoint downloads. `randomforecast` and `randomforest` are accepted as aliases for Random Forest.

Main dependencies are pinned in `uv.lock`. Optional environments are installed using **Install runtime** in the model library, or `uv run python -m scripts.forecast_runtime gluonts` (substitute `autogluon` or `neuralprophet`). Installation requires internet access for Python/packages. Optional environments live in `runtime/forecast-runtimes/`, with resolved package freezes and import checks; each fitted model also records its runtime versions. NeuralProphet is isolated because it uses an older Python/NumPy/PyTorch combination. Execution is CPU-only in this release.

## Evaluation and integrity

This release forecasts **one asset and one numeric channel per run**. It accepts both imported interval data and regularly sampled recordings, retaining their native resolution. It does not perform automatic resampling, multi-series/global fitting, external covariates, or automated parameter search.

The last horizon is the holdout. Two to eight nonoverlapping validation windows precede it. Each fit uses only the configured context before that window. Validation MAE selects the winner; the winner is frozen before evaluating the holdout. The ensemble's equal weights are fixed. A failed validation model is listed with its error; a holdout fitting failure fails the job instead of changing the selected winner. AutoGluon's internal tuning uses the latest portion of the training context, never the external holdout.

MAE and RMSE use source units. sMAPE/WAPE are percentages; zero-denominator WAPE and constant-history MASE are unavailable. Missing evaluation targets remain missing and are omitted from metric counts. Bands use absolute validation residuals at a nominal 90% level; they are empirical error bands, not calibrated probability intervals or a coverage guarantee. MASE uses training-only seasonal differences; validation uses the first training window's scale and holdout uses the holdout training window's scale.

After evaluation, each successful model is refitted through the final source observation to produce future forecasts and a reusable artifact. **Use saved fitted model** runs inference with that artifact and compatible newer history; it reports no new evaluation metrics. Reuse checks source asset/channel/unit/resolution/timezone, training cutoff, adapter version, artifact checksums, and exact recorded environment. The requested horizon cannot exceed the training horizon. Ensemble reuse is not supported; individual constituent models can be reused.

Default gap handling rejects missing intervals. Check **Interpolate missing data** to enable linear interpolation for short interior gaps in each training history. The default maximum is three consecutive missing samples (three hours for hourly data, three milliseconds for 1 kHz data); it is configurable from 1 to 100. A gap must be fully bounded by observed values inside that training context, before its forecast origin. Leading, trailing, and over-limit gaps are rejected without extrapolation or partial filling. Every training window must meet the observed-coverage requirement before filling, as must evaluation windows. All required histories are checked before model fitting begins. Interpolants are estimates, not reconstructed ground truth; suitability for a particular physical signal still needs engineering judgment.

The checkbox is off by default and its policy/limit persist in presets, queued jobs, and exported configuration. Results show the filled-value count, gap count, and original observed coverage for each window. Exports include `preprocessing.json` and `training_imputations.csv` with every estimated training value and timestamp. Missing validation/holdout targets remain missing and never become scored observations. Original files and `source_window.csv` preserve their missing readings. The separate bounded forward-fill option remains available when interpolation is unchecked.

Duplicate/nonmonotonic timestamps and irregular sampling are rejected. Negative net energy remains valid. Dates without timezone information in the supplied anomaly CSV retain their reported clock and unverified provenance. Label columns cannot be forecast targets. Unmapped CSV reads are limited to 20 MiB; import larger files first. History is bounded to 200,000 samples, and each forecast to 512 steps; this is a bounded-window forecasting workbench, not a bulk 1 kHz foundation-model pipeline. API and notebook configurations opt in with `missing_policy="interpolate"` and `max_gap_steps=3`.

Runs freeze source checksums, configuration, adapter version, seed, split boundaries, and model environments. Export bundles contain predictions, metrics in `result.json`, `source_window.csv`, configuration, split manifest, fitted model artifacts, checksums, and `report.html`. Plot images can be exported through the chart toolbar. The comparison table within each run uses the same evaluation population. Inspect `population_id` and split manifests before comparing separate runs. Torch/AutoML reproducibility also depends on the recorded package versions and workstation; time-limited AutoML can train a different subset of models under different machine load.

## Notebook and Python use

**Notebook example** opens `04 - Forecasting.ipynb` in the local JupyterLite workspace. Its `di_forecast` helper submits and polls jobs, reads model/run catalogs, cancels jobs, and returns prediction DataFrames. Training executes in the workstation worker; browser Python stays responsive. Interrupting a notebook cell does not cancel its server-side job.

Native Python can use `di_validator.forecasting.engine.freeze/run` and `ForecastConfig` directly. HTTP endpoints under `/api/v1/forecasting` expose models, runtime setup, runs, predictions, and ZIP exports; typed contracts appear in `/docs`. All API training/setup requests return persistent job IDs and share the application's single worker and cancellation/recovery behavior.

## Verification

Run `uv run pytest` for regression tests. `uv run python -m scripts.verify_forecasting` performs actual fitting and saved-model reload checks for all installed providers, verifies that their environments lack `huggingface_hub` and `transformers`, and writes evidence under `runtime/verification/forecasting/`. Use `--models prophet gluonts` to select providers. Optional runtimes must be installed first. Browser checks are in `frontend/e2e/forecasting.spec.ts`.

Adapter references: [GluonTS DeepAR](https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.deepar.html), [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html), [Prophet](https://facebook.github.io/prophet/docs/quick_start.html), [NeuralProphet](https://neuralprophet.com/code/forecaster.html).
