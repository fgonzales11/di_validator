import json
import pickle
import sqlite3
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["__".join(str(part) for part in col) for col in df.columns.values]
    return df


def _sanitize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_cols = [col for col in df.columns if df[col].dtype == "object"]
    if not object_cols:
        return df

    def _normalize_value(value):
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            return value
        try:
            return json.dumps(value, default=_json_default)
        except TypeError:
            return str(value)

    for col in object_cols:
        df[col] = df[col].map(_normalize_value)
    return df


def _dtype_to_sqlite(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    return "TEXT"


def _ensure_columns(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    existing = {row[1] for row in cur.fetchall()}
    missing = [col for col in df.columns if col not in existing]
    for col in missing:
        col_type = _dtype_to_sqlite(df[col].dtype)
        cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" {col_type}')


def _reset_index_with_names(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        if df.index.name is None:
            df = df.copy()
            df.index.name = "index"
        return df.reset_index()

    idx_names = list(df.index.names)
    if all(name is None or name == "" for name in idx_names):
        idx_names = [f"level_{i}" for i in range(len(idx_names))]
    else:
        idx_names = [name if name not in (None, "") else f"level_{i}" for i, name in enumerate(idx_names)]

    cols = set(df.columns)
    deduped = []
    for i, name in enumerate(idx_names):
        candidate = name
        if candidate in cols or candidate in deduped:
            candidate = f"{candidate}_idx"
        if candidate in cols or candidate in deduped:
            candidate = f"idx_{i}"
        deduped.append(candidate)
    df = df.copy()
    df.index.set_names(deduped, inplace=True)
    return df.reset_index()


def _iter_dataframe_tables(net: Dict) -> Iterable[Tuple[str, pd.DataFrame]]:
    for key, value in net.items():
        if isinstance(value, pd.DataFrame):
            yield key, value


def _init_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    conn.executescript(schema_path.read_text(encoding="utf-8"))


def _upsert_network(conn: sqlite3.Connection, network_id: str, meta: Dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO networks
        (network_id, name, description, sn_mva, f_hz, rho_ohmm, version, format_version, user_pf_options)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            network_id,
            meta.get("name"),
            meta.get("description"),
            meta.get("sn_mva"),
            meta.get("f_hz"),
            meta.get("rho_ohmm"),
            meta.get("version"),
            meta.get("format_version"),
            json.dumps(meta.get("user_pf_options", {}), default=_json_default),
        ),
    )


def export_network(pkl_path: Path, db_path: Path, network_id: str | None = None) -> None:
    repo_root = pkl_path.resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    with pkl_path.open("rb") as f:
        net = pickle.load(f)
    if not isinstance(net, Mapping):
        raise TypeError(f"Unsupported network payload type: {type(net).__name__}")

    schema_path = db_path.parent / "multiconductor_schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found at {schema_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    network_id = network_id or pkl_path.stem

    conn = sqlite3.connect(db_path)
    try:
        _init_schema(conn, schema_path)
        meta_keys = [
            "name",
            "description",
            "sn_mva",
            "f_hz",
            "rho_ohmm",
            "version",
            "format_version",
            "converged",
            "OPF_converged",
            "user_pf_options",
            "std_types",
        ]
        meta = {key: net.get(key) for key in meta_keys if key in net}
        _upsert_network(conn, network_id, meta)

        conn.execute("DELETE FROM network_meta WHERE network_id = ?", (network_id,))
        meta_rows = []
        for key, value in meta.items():
            meta_rows.append({"network_id": network_id, "key": key, "value": json.dumps(value, default=_json_default)})
        pd.DataFrame(meta_rows).to_sql("network_meta", conn, if_exists="append", index=False)

        for table_name, df in _iter_dataframe_tables(net):
            df = _flatten_columns(df)
            df = _reset_index_with_names(df)
            df = _sanitize_object_columns(df)
            df.insert(0, "network_id", network_id)
            _ensure_columns(conn, table_name, df)
            conn.execute(f'DELETE FROM "{table_name}" WHERE network_id = ?', (network_id,))
            try:
                df.to_sql(table_name, conn, if_exists="append", index=False)
            except Exception as exc:
                raise RuntimeError(f"Failed writing table '{table_name}'") from exc
        conn.commit()
    finally:
        conn.close()


def export_networks_directory(networks_dir: Path, db_path: Path) -> tuple[int, int]:
    pkl_files = sorted(networks_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {networks_dir}")

    exported = 0
    failed = 0
    for pkl_path in pkl_files:
        try:
            export_network(pkl_path, db_path)
            exported += 1
            print(f"Exported {pkl_path.name} (network_id={pkl_path.stem})")
        except Exception as exc:
            failed += 1
            print(f"Skipped {pkl_path.name}: {exc}")

    return exported, failed


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    networks_dir = repo_root / "networks"
    db_path = here / "multiconductor.db"
    exported, failed = export_networks_directory(networks_dir, db_path)
    print(
        f"Export complete: {exported} exported, {failed} skipped from {networks_dir} into {db_path}"
    )
