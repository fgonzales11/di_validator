# DI Validator notebooks

Open **01 - Getting started.ipynb**, select a code cell, and press **Shift+Enter**.
Python runs in your browser. NumPy, pandas, SciPy, scikit-learn, and Matplotlib
are available from the locally served Pyodide distribution.

`di_data.py` provides async access to the current application workspace:

```python
from di_data import DIClient
di = DIClient()
datasets = await di.datasets()
assets = await di.assets(datasets[0]["id"])
page = await di.readings(datasets[0]["id"], asset_ids=[assets[0]["id"]])
frame = di.to_frame(page)
```

The API returns native measurements, never chart envelopes. Missing readings
remain missing; omitted invalid clock rows remain documented in `di.quality()`.
Timestamps retain nanosecond precision. Units, interval conventions, provenance,
and synthetic status are carried in `frame.attrs`. Recording bounds use seconds
from origin; interval-data bounds use ISO timestamps. Each page contains at most
100,000 rows. `di.iter_frames()` traverses large selections without loading an
entire recording into browser memory.

Labels, registry evidence, annotations, experiments, and predictions are also
available through the helper. Registered original files can be downloaded using
the data panel in the surrounding Notebook page.

For small source tables whose measurement mapping is not yet known, use:

```python
from di_sources import read_csv
frame = await read_csv("power_anomaly_dataset_2022.csv")
```

This reader preserves every supplied column, leaves timestamps as strings, and
records the source checksum in `frame.attrs`. Its limit is 20 MiB; larger files
should be imported for paged access. Open **03 - Power anomaly exploration.ipynb**
to inspect the 1 Hz source table, its labels, and an exploratory custom score.
Source labels remain unverified; no timezone or measurement type is inferred.

Notebooks and files created here are saved in this browser's storage for this
localhost address. Download `.ipynb` files from the File menu for a portable copy.
Keep the same hostname and port to return to the same browser workspace.
Clearing browser storage removes these notebook copies. It does not change
original data or completed DI Validator experiments.

Custom code uses the browser's Python environment, separate from the workstation
worker. Additional compatible packages can be installed with `%pip install`;
packages outside the bundled distribution may require internet access. Desktop
native extensions, subprocesses, and direct Windows paths are unavailable in
JupyterLite. Use the data helper or upload additional files into its file browser.

## Fault-distance notebook

Open `fault_distance.ipynb` for the full imported COMTRADE pipeline. Import the recording in Event Lab first; then choose `RECORD_NAME` in the first cell and verify the example line parameters. `di_faults.fetch_record` copies the frozen pair into browser storage without changing the source. The supplied data is simulated and its timezone is unverified. The bundled COMTRADE reader is version 0.1.2, licensed under MIT (COMTRADE-LICENSE.md). Outputs are written under `fault_outputs/<record name>`; download artifacts to retain them outside browser storage.
