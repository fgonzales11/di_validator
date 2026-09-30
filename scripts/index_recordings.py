"""Rebuild optional derived indexes for existing recording datasets."""
import os
from di_validator import store
from di_validator.recording_index import build

for root in [store.ROOT/"runtime",store.ROOT/"runtime"/"benchmark"]:
    os.environ["DI_WORKSPACE"] = str(root)
    store.initialize()
    for dataset in store.listing("dataset"):
        if dataset["format"] == "recording":
            build(dataset["folder"])
            print(f"Indexed {dataset['name']}",flush=True)
