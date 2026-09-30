#!/usr/bin/env python3
"""
COMTRADE → CSV Converter
=========================
Output format:
- Each .cfg file → a separate .csv
- CSV name: <name without _val#>_km.csv  (example: 1A_0.5km.csv)
- Each row = one time point (sample)
- Columns: distance_km, fs_hz, <currents>, <voltages>, <others>
  * distance_km — constant throughout the file (first column)
  * fs_hz       — sampling frequency in Hz, constant (second column)
- Time is NOT included
"""

import re
import sys
import logging
import configparser
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import comtrade
except ImportError:
    print("[ERROR] Install: pip install comtrade")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comtrade_to_csv.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Channel detection ─────────────────────────────────────────────────────
# Unicode escapes preserve Cyrillic channel names in regular expressions.
_PHASE = r"[aAbBcCnN0-3\u0410\u0430\u0411\u0431\u0412\u0432\u0421\u0441\u041d\u043d]"
_CURRENT_RE = re.compile(
    r"(?:^|[\s_\-\.])([Ii]" + _PHASE + r"{0,2})(?:[\s_\-\.]|$)"
    r"|^[Ii]" + _PHASE + r"{0,2}$"
)
_VOLTAGE_RE = re.compile(
    r"(?:^|[\s_\-\.])([UuVv]" + _PHASE + r"{0,2})(?:[\s_\-\.]|$)"
    r"|^[UuVv]" + _PHASE + r"{0,2}$"
)
_CURRENT_KEYWORD_RE = re.compile(r"[Ii][AaBbCc\u0410\u0430\u0411\u0431\u0412\u0432]", re.UNICODE)
_VOLTAGE_KEYWORD_RE = re.compile(r"[UuVv][AaBbCc\u0410\u0430\u0411\u0431\u0412\u0432]", re.UNICODE)


def _is_current(name: str) -> bool:
    n = name.strip()
    if re.fullmatch(r"[Ii]" + _PHASE + r"{0,2}", n):
        return True
    if bool(_CURRENT_RE.search(n)):
        return True
    if _CURRENT_KEYWORD_RE.search(n) and not _VOLTAGE_KEYWORD_RE.search(n):
        return True
    return False


def _is_voltage(name: str) -> bool:
    n = name.strip()
    if re.fullmatch(r"[UuVv]" + _PHASE + r"{0,2}", n):
        return True
    if bool(_VOLTAGE_RE.search(n)):
        return True
    if _VOLTAGE_KEYWORD_RE.search(n):
        return True
    return False


def _sort_channels(names: list, arrays: list) -> list:
    """Order: currents → voltages → others."""
    currents = [(n, d) for n, d in zip(names, arrays) if _is_current(n)]
    voltages = [(n, d) for n, d in zip(names, arrays) if _is_voltage(n)]
    others   = [(n, d) for n, d in zip(names, arrays)
                if not _is_current(n) and not _is_voltage(n)]
    return currents + voltages + others


# ── Extract the sampling frequency from COMTRADE ──────────────────────────────
def _extract_fs(rec) -> float:
    """
    Extract the sampling frequency from a COMTRADE record.

    COMTRADE (.cfg) stores a list of (samp_rate, last_sample_num) pairs.
    Use the first pair, which describes the main signal.
    If sample_rates are unavailable or zero, derive the rate from the time vector.

    Returns
    -------
    float
        Sampling frequency in Hz.
    """
    try:
        rates = rec.cfg.sample_rates   # list [(samp_rate, end_sample), ...]
        if rates:
            fs = float(rates[0][0])
            if fs > 0:
                return fs
    except AttributeError:
        pass

    # Fallback: derive from the time vector
    time = np.array(rec.time, dtype=float)
    if len(time) >= 2:
        dt = np.median(np.diff(time))   # the median is robust to outliers
        if dt > 0:
            return round(1.0 / dt, 2)

    raise ValueError("Could not determine the sampling frequency from the COMTRADE file")


# ── VAL regex ──────────────────────────────────────────────────────────────────
_VAL_RE = re.compile(r"_val(\d+)(?:\.[^.]+)?$", re.IGNORECASE)


def _extract_val(stem: str):
    m = _VAL_RE.search(stem)
    return int(m.group(1)) if m else None


def _build_output_name(stem: str, km: float) -> str:
    clean = _VAL_RE.sub("", stem)
    km_str = f"{km:.1f}" if km != int(km) else f"{int(km)}"
    return f"{clean}_{km_str}km.csv"


# ── Config ─────────────────────────────────────────────────────────────────────
def _load_config(ini_path: str = None) -> dict:
    if ini_path is None:
        ini_path = os.path.join(os.path.dirname(__file__), "data_comtrade_config.ini")
    cfg = configparser.ConfigParser()
    cfg.read(ini_path, encoding="utf-8")
    s = cfg["settings"]
    return {
        "line_length_km": float(s.get("line_length_km", 100.0)),
        "input_folder":   s.get("input_folder",  "./comtrade_files"),
        "output_folder":  s.get("output_folder", "./csv_output"),
        "test":           s.getboolean("test",      fallback=False),
        "recursive":      s.getboolean("recursive", fallback=False),
    }


def _find_cfg_files(folder_or_file: str, recursive: bool) -> list:
    p = Path(folder_or_file)

    if p.is_file() and p.suffix.lower() == ".cfg":
        return [p]

    if not p.is_dir():
        log.error(f"Path does not exist or is not a directory/file: {p}")
        return []

    pattern = "**/*.cfg" if recursive else "*.cfg"
    return sorted(p.glob(pattern))


# ── One file → one CSV ───────────────────────────────────────────────────────
def _process_file(cfg_path: Path, line_length_km: float,
                  output_folder: Path) -> bool:
    stem = cfg_path.stem
    val  = _extract_val(stem)

    if val is None:
        log.warning(f"SKIP {cfg_path.name} — missing _val# pattern")
        return False

    km          = round((val / 100.0) * line_length_km, 6)
    output_name = _build_output_name(stem, km)
    output_path = output_folder / output_name

    try:
        rec = comtrade.load(str(cfg_path))
    except Exception as e:
        log.error(f"ERROR {cfg_path.name} — {e}")
        return False

    # Sampling frequency
    try:
        fs = _extract_fs(rec)
    except ValueError as e:
        log.error(f"ERROR {cfg_path.name} — {e}")
        return False

    names  = list(rec.analog_channel_ids)
    arrays = list(rec.analog)

    if not arrays:
        log.warning(f"SKIP {cfg_path.name} — no analog channels")
        return False

    ordered = _sort_channels(names, arrays)
    n_cur   = sum(1 for n, _ in ordered if _is_current(n))
    n_volt  = sum(1 for n, _ in ordered if _is_voltage(n))
    n_rows  = len(ordered[0][1]) if ordered else 0

    # distance_km and fs_hz are the first two columns, both constant
    data = {
        "distance_km": km,
        "fs_hz":       fs,
    }
    for ch_name, ch_data in ordered:
        data[ch_name] = np.array(ch_data, dtype=np.float64)

    df = pd.DataFrame(data)

    output_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6g")

    log.info(
        f"OK {cfg_path.name} → {output_name}"
        f" | km={km}"
        f" | fs={fs:.1f} Hz"
        f" | I×{n_cur} U×{n_volt}"
        f" | {n_rows} rows"
    )
    return True


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    config        = _load_config("config.ini")
    input_folder  = Path(config["input_folder"])
    output_folder = Path(config["output_folder"])

    log.info("=" * 65)
    log.info(f"Input directory  : {input_folder}")
    log.info(f"Output directory : {output_folder}")
    log.info(f"Line length      : {config['line_length_km']} km")
    log.info(f"Test={config['test']}  Recursive={config['recursive']}")
    log.info("=" * 65)

    cfg_files = _find_cfg_files(str(input_folder), config["recursive"])
    if not cfg_files:
        log.error("No .cfg files found in the specified directory")
        return

    if config["test"]:
        cfg_files = cfg_files[:1]
        log.info(f"TEST MODE → only: {cfg_files[0].name}")

    total   = len(cfg_files)
    success = sum(
        _process_file(f, config["line_length_km"], output_folder)
        for f in cfg_files
    )

    log.info("=" * 65)
    log.info(f"Done: {success}/{total} files converted")


if __name__ == "__main__":
    main()
