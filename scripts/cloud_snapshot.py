"""Export a consistent local catalog for first cloud deployment; never modify originals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from di_validator import store


def replace_paths(value, root):
    if isinstance(value, dict):
        return {k: replace_paths(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_paths(v, root) for v in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        base = str(root).replace("\\", "/")
        if normalized.startswith(base + "/runtime/"):
            return "/data/runtime/" + normalized[len(base + "/runtime/"):]
        if normalized.startswith(base + "/data/"):
            return "/data/sources/" + normalized[len(base + "/data/"):]
        if normalized.startswith(base + "/"):
            return "/app/" + normalized[len(base + "/"):]
    return value


def export(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # sqlite backup takes a consistent snapshot, including committed WAL data.
    with sqlite3.connect(store.workspace()/"catalog.sqlite") as original:
        with sqlite3.connect(folder/"catalog.sqlite") as snapshot:
            original.backup(snapshot)
    result = {}
    with sqlite3.connect(folder/"catalog.sqlite") as db:
        db.row_factory = sqlite3.Row
        for table in ["objects", "jobs", "logs"]:
            rows = []
            for raw in db.execute(f"SELECT * FROM {table}"):
                row = dict(raw)
                for field in ["body", "config", "result"]:
                    if row.get(field):
                        row[field] = store.encode(replace_paths(json.loads(row[field]), store.ROOT))
                if table == "jobs" and row["status"] in {"running", "queued"}:
                    row.update(status="interrupted", message="Copied to cloud; rerun the saved configuration")
                rows.append(row)
            result[table] = rows
    (folder/"catalog.json").write_text(store.encode(result), encoding="utf-8")
    summary = {"source_root": str(store.ROOT), "tables": {k: len(v) for k, v in result.items()}}
    (folder/"manifest.json").write_text(store.encode(summary), encoding="utf-8")
    print(store.encode(summary))


def restore(path):
    import os
    if not os.environ.get("DI_DATABASE_URL"):
        raise RuntimeError("Restore is only supported for the dedicated cloud database")
    store.initialize()
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    with store.connection() as db:
        # One transaction. An existing deployment must never be overwritten.
        for table in ["objects", "jobs", "logs"]:
            if db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]:
                raise ValueError("Cloud catalog is not empty; restore refused")
        for table, rows in content.items():
            if table not in {"objects", "jobs", "logs"}:
                raise ValueError("Invalid snapshot table")
            for row in rows:
                columns = list(row)
                allowed = {"kind", "id", "body", "created_at", "status", "config", "progress", "message",
                           "result", "error", "updated_at", "cancel_requested", "job_id", "time"}
                if not set(columns) <= allowed:
                    raise ValueError("Invalid snapshot columns")
                db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                           tuple(row.values()))
    print(store.encode({"restored": {k: len(v) for k, v in content.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["export", "restore"])
    parser.add_argument("path")
    args = parser.parse_args()
    (export if args.action == "export" else restore)(args.path)
