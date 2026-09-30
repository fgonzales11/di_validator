"""Audit the live COMTRADE catalog and run every supplied recording in memory."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from di_validator import store
from di_validator.faults.comtrade_io import sources
from di_validator.faults.pipeline import analyze_segment, validate_dataset


def main():
    rows = []
    for item in sources():
        if not item["dataset_id"]:
            raise ValueError(f"Not imported: {item['name']}")
        ds = store.get("dataset", item["dataset_id"])
        folder = Path(ds["folder"])
        for extension in ["cfg", "dat"]:
            expected = ds["comtrade"]["source_checksums"][extension]
            assert store.checksum(item[extension+"_path"]) == expected
            assert store.checksum(folder/f"original.{extension}") == expected
        frame = pd.read_parquet(folder/"samples")
        assert len(frame) == ds["valid_rows"] == 400
        np.testing.assert_array_equal(frame.t_ns, np.arange(400)*500000)
        cfg = validate_dataset(ds, {})
        analysis, _ = analyze_segment(frame, ds, cfg)
        rows.append(dict(name=item["name"], dataset_id=ds["id"], samples=len(frame), readings=ds["readings"],
                         source_checksum=ds["source_checksum"], onset=analysis.get("onset"),
                         status=analysis["status"], distance=analysis.get("distance")))
    notebook = json.loads((store.ROOT/"notebooks/files/fault_distance.ipynb").read_text(encoding="utf-8"))
    assert store.checksum(store.ROOT/"fault-distance/fault_distance.ipynb") == notebook["metadata"]["di_validator"]["source_sha256"]
    report = dict(recordings=len(rows), samples=sum(r["samples"] for r in rows), readings=sum(r["readings"] for r in rows),
                  source_files_unchanged=True, original_notebook_unchanged=True, timezone_verified=False,
                  parameters="Notebook examples; unverified", results=rows)
    target = store.workspace()/"verification/comtrade-inventory.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(store.encode(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    print("Estimated:", sum(r.get("distance", {}).get("status") == "estimated" for r in rows))
    print("Audit:", target)


if __name__ == "__main__":
    main()
