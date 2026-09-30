from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import Field

from .. import store
from ..schemas import Model
from .comtrade_io import normalized_header, source, sources
from .wavewin_io import source as wavewin_source, sources as wavewin_sources

router = APIRouter(prefix="/api/v1/faults", tags=["COMTRADE and fault distance"])


class ImportSelection(Model):
    source_ids: list[str] = Field(min_length=1, max_length=500)


@router.get("/sources")
def catalog():
    return sources()


@router.post("/imports")
def queue_import(config: ImportSelection):
    chosen = []
    for identifier in dict.fromkeys(config.source_ids):
        item = source(identifier)
        item["checksums"] = dict(cfg=store.checksum(item["cfg_path"]), dat=store.checksum(item["dat_path"]))
        chosen.append(item)
    return store.enqueue("comtrade_import", {"sources": chosen})


@router.get("/sources/{source_id}/{extension}")
def source_file(source_id: str, extension: str, normalized: bool = False):
    if extension not in {"cfg", "dat"}:
        raise HTTPException(404, "Choose cfg or dat")
    item = source(source_id)
    # Prefer the registered, immutable version for notebook reproducibility.
    if item.get("dataset_id"):
        dataset = store.get("dataset", item["dataset_id"])
        path = Path(dataset["folder"]) / f"original.{extension}"
    else:
        path = Path(item[extension + "_path"])
    if normalized and extension == "cfg":
        header, _ = normalized_header(path)
        return PlainTextResponse(
            header,
            headers={
                "X-Source-SHA256": store.checksum(path),
                "X-Date-Policy": "1991 two-digit years: 2000+YY",
            },
        )
    return FileResponse(
        path, filename=f"{item['name']}.{extension}", headers={"X-Source-SHA256": store.checksum(path)}
    )


@router.get("/wavewin/sources")
def wavewin_catalog():
    return wavewin_sources()


@router.post("/wavewin/imports")
def queue_wavewin_import(config: ImportSelection):
    catalog = {s["id"]: s for s in wavewin_sources()}
    chosen = []
    for identifier in dict.fromkeys(config.source_ids):
        item = catalog.get(identifier)
        if not item:
            raise ValueError("Choose a Wavewin event from the source catalog")
        item["checksum"] = store.checksum(item["cev_path"])
        chosen.append(item)
    return store.enqueue("wavewin_import", {"sources": chosen})


@router.get("/wavewin/sources/{source_id}/cev")
def wavewin_source_file(source_id: str):
    item = wavewin_source(source_id)
    if item.get("dataset_id"):
        dataset = store.get("dataset", item["dataset_id"])
        path = Path(dataset["folder"]) / "original.cev"
    else:
        path = Path(item["cev_path"])
    return FileResponse(
        path, filename=f"{item['name']}.CEV", headers={"X-Source-SHA256": store.checksum(path)}
    )
