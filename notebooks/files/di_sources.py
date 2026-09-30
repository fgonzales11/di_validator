"""Read small source CSVs before their measurement and clock mapping is known."""
from io import BytesIO
from urllib.parse import quote


async def read_csv(filename, **options):
    """Load a CSV from data/ (up to 20 MiB), retaining all supplied columns.

    Timestamps remain strings unless parse_dates is explicitly requested.
    No units, timezone, asset level, or verification of labels are inferred.
    For large imported recordings use di_data.DIClient.iter_frames instead.
    """
    import pandas as pd
    from js import location
    from pyodide.http import pyfetch

    url = str(location.origin) + "/api/v1/notebooks/files/" + quote(filename, safe="")
    response = await pyfetch(url)
    if not response.ok:
        error = await response.json()
        raise ValueError(error.get("detail", f"Source read failed: {response.status}"))
    frame = pd.read_csv(BytesIO(await response.bytes()), **options)
    frame.attrs.update(source_file=filename, source_checksum=response.headers.get("x-source-sha256"),
                       mapping="unverified", label_evidence="source-supplied, unverified")
    return frame
