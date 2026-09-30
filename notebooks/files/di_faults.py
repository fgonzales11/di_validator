"""Fetch immutable COMTRADE pairs from the local DI Validator catalog."""
from pathlib import Path


async def recordings():
    from js import location
    from pyodide.http import pyfetch
    response = await pyfetch(str(location.origin)+"/api/v1/faults/sources")
    if not response.ok:
        raise ValueError("Could not read the local COMTRADE catalog")
    return await response.json()


async def fetch_record(name="1A_val1"):
    """Copy a catalog pair into browser storage; server originals stay unchanged.

    CFG dates receive the documented RTDS 1991 two-digit-year normalization.
    DAT bytes are unmodified. The checksums describe the original source pair.
    Import the recording through Event Lab first to freeze its source version.
    """
    from js import location
    from pyodide.http import pyfetch
    choices = await recordings()
    selected = next((r for r in choices if r["name"] == name), None)
    if selected is None:
        raise ValueError(f"Unknown recording {name!r}; use await di_faults.recordings()")
    if not selected.get("dataset_id"):
        raise ValueError("Import this recording in Event Lab → COMTRADE recordings before opening it here")
    destination = Path("comtrade_data")/selected["id"]
    destination.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for extension in ["cfg", "dat"]:
        url = str(location.origin)+f"/api/v1/faults/sources/{selected['id']}/{extension}"
        if extension == "cfg":
            url += "?normalized=true"
        response = await pyfetch(url)
        if not response.ok:
            raise ValueError(f"Could not fetch {name}.{extension}")
        checksums[extension] = response.headers.get("x-source-sha256")
        (destination/f"{name}.{extension}").write_bytes(await response.bytes())
    response = await pyfetch(str(location.origin)+"/api/v1/datasets/"+selected["dataset_id"])
    metadata = await response.json()
    metadata["original_checksums"] = checksums
    print(f"Loaded {name}: {metadata['sample_rate']:g} Hz, {metadata['valid_rows']} native samples")
    print(metadata["comtrade"]["timing_note"])
    print(metadata["comtrade"]["instrument_note"])
    return destination, metadata
