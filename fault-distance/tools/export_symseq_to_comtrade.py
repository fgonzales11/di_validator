#!/usr/bin/env python3
"""
symseq_to_comtrade.py
=====================
Export symmetrical components (sliding window) to COMTRADE format
(.cfg + .dat) for viewing in any oscillography software
(OMICRON Transview, PowerDB, DIgSILENT, RTDS, etc.).

Module architecture
------------------
CSV (signals)
    └─► sliding_symseq()        — sliding-window calculation of I0/I1/I2, U0/U1/U2
            └─► SymSeqResult    — dataclass result container
                    └─► ComtradeExporter.export()  — write .cfg / .dat

Extensibility
-------------
* Add a new channel     → add a field to SymSeqResult + an entry in
                           ComtradeExporter._CURRENT_CHANNELS / _VOLTAGE_CHANNELS
* Change the .dat format → override ComtradeExporter._write_dat()
* Export a file batch   → export_batch() at the bottom of the module

Dependencies: numpy and pandas only (already in the project's requirements.txt).

Usage:
    python tools/symseq_to_comtrade.py
    python tools/symseq_to_comtrade.py --csv data/data_training/2AB_10km.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── path to project packages ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils.column_detector import (
    detect_signal_columns,
    detect_distance_column,
)
from symseq.power_systems import symseq_from_waveforms

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_CSV         = "data/data_training/2AB_10km.csv"
DEFAULT_F0          = 50.0     # Hz
DEFAULT_FS_FALLBACK = 1000.0   # Hz — if the CSV has no fs_hz
OUTPUT_DIR          = Path(__file__).parent / "output_tools"


# ═══════════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SymSeqResult:
    """
    Result of the sliding-window calculation of symmetrical components.

    All arrays have length n_steps = N - win + 1.
    The t_ms time axis is aligned with the center of each window.
    """
    source_file : str
    distance_km : float
    fs          : float          # Hz
    f0          : float          # Hz
    win         : int            # samples per period
    n_steps     : int

    t_ms   : np.ndarray          # (n_steps,) — window centers, ms

    # Amplitudes, A
    I0_mag : np.ndarray
    I1_mag : np.ndarray
    I2_mag : np.ndarray

    # Angles, rad (converted to degrees when written)
    I0_ang : np.ndarray = field(repr=False)
    I1_ang : np.ndarray = field(repr=False)
    I2_ang : np.ndarray = field(repr=False)

    # Voltage amplitudes, V
    U0_mag : np.ndarray = field(repr=False)
    U1_mag : np.ndarray = field(repr=False)
    U2_mag : np.ndarray = field(repr=False)

    # Angles, rad (converted to degrees when written)
    U0_ang : np.ndarray = field(repr=False)
    U1_ang : np.ndarray = field(repr=False)
    U2_ang : np.ndarray = field(repr=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Sliding-window calculation
# ═══════════════════════════════════════════════════════════════════════════════

def sliding_symseq(
    csv_path    : str | Path,
    f0          : float = DEFAULT_F0,
    fs_fallback : float = DEFAULT_FS_FALLBACK,
) -> SymSeqResult:
    """
    Load a CSV and compute symmetrical components using a sliding window.

    Window = exactly 1 period (win = round(fs / f0)) → a rectangular window
    is valid, with no spectral leakage.

    Parameters
    ----------
    csv_path    : path to a project CSV file
    f0          : grid frequency, Hz
    fs_fallback : fs if the CSV has no fs_hz column

    Returns
    -------
    SymSeqResult
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # ── sampling frequency ─────────────────────────────────────────────────
    if "fs_hz" in df.columns:
        fs = float(df["fs_hz"].iloc[0])
    else:
        fs = fs_fallback

    win = int(round(fs / f0))

    # ── columns ───────────────────────────────────────────────────────────────
    dist_col    = detect_distance_column(list(df.columns))
    col_map     = detect_signal_columns(list(df.columns), distance_col=dist_col)
    distance_km = float(df[dist_col].iloc[0])

    sig = df[
        [col_map["Ia"], col_map["Ib"], col_map["Ic"],
         col_map["Ua"], col_map["Ub"], col_map["Uc"]]
    ].values.astype(float)                              # (N, 6)

    N = len(sig)
    if N < win:
        raise ValueError(
            f"{csv_path.name}: rows {N} < window size {win}. "
            "Check fs and f0."
        )

    n_steps = N - win + 1

    # ── output buffers ───────────────────────────────────────────────────────
    buf = {k: np.zeros(n_steps) for k in
           ("I0m","I1m","I2m","I0a","I1a","I2a",
            "U0m","U1m","U2m","U0a","U1a","U2a")}

    # ── sliding-window loop ───────────────────────────────────────────────────────
    for i in range(n_steps):
        sl = slice(i, i + win)

        ri = symseq_from_waveforms(
            sig[sl, 0], sig[sl, 1], sig[sl, 2],
            fs=fs, f0=f0, window=False,
        )
        ru = symseq_from_waveforms(
            sig[sl, 3], sig[sl, 4], sig[sl, 5],
            fs=fs, f0=f0, window=False,
        )

        buf["I0m"][i] = ri["X0_mag"]; buf["I0a"][i] = ri["X0_ang"]
        buf["I1m"][i] = ri["X1_mag"]; buf["I1a"][i] = ri["X1_ang"]
        buf["I2m"][i] = ri["X2_mag"]; buf["I2a"][i] = ri["X2_ang"]

        buf["U0m"][i] = ru["X0_mag"]; buf["U0a"][i] = ru["X0_ang"]
        buf["U1m"][i] = ru["X1_mag"]; buf["U1a"][i] = ru["X1_ang"]
        buf["U2m"][i] = ru["X2_mag"]; buf["U2a"][i] = ru["X2_ang"]

    t_ms = (np.arange(n_steps) + win / 2) / fs * 1000.0

    return SymSeqResult(
        source_file = csv_path.name,
        distance_km = distance_km,
        fs          = fs,
        f0          = f0,
        win         = win,
        n_steps     = n_steps,
        t_ms        = t_ms,
        I0_mag = buf["I0m"], I1_mag = buf["I1m"], I2_mag = buf["I2m"],
        I0_ang = buf["I0a"], I1_ang = buf["I1a"], I2_ang = buf["I2a"],
        U0_mag = buf["U0m"], U1_mag = buf["U1m"], U2_mag = buf["U2m"],
        U0_ang = buf["U0a"], U1_ang = buf["U1a"], U2_ang = buf["U2a"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COMTRADE exporter
# ═══════════════════════════════════════════════════════════════════════════════

class ComtradeExporter:
    """
    Write SymSeqResult to a pair of COMTRADE 1999 files:
        <stem>.cfg  — configuration file (ASCII)
        <stem>.dat  — data in ASCII format

    ASCII format is chosen for maximum compatibility with all
    oscillography software (Waves, PowerDB, DIgSILENT, etc.).

    Channel line in .cfg (IEEE Std C37.111-1999, section 5.3.5):
        n, ch_id, ph, ccbm, uu, a, b, skew, min, max, primary, secondary, PS

        n       — channel number (starting at 1)
        ch_id   — channel name (I1_mag, etc.)
        ph      — phase: empty string for calculated channels
        ccbm    — component identifier: empty string for calculated channels
        uu      — unit of measurement (A, deg, V)
        a       — scale factor: physical = a * raw + b
        b       — offset (0 for all our channels)
        skew    — channel time offset, microseconds (0)
        min/max — raw value range (±32767 for int16; in ASCII,
                  this field is informational; actual data are written as floats)
        primary/secondary — transformation ratio (1/1 for calculated channels)
        PS      — primary/secondary flag (S)

    Default channels (extend via _CURRENT_CHANNELS/_VOLTAGE_CHANNELS):
        I1_mag, I2_mag, I0_mag  — current amplitudes, A
        I1_ang, I2_ang, I0_ang  — current angles, degrees
        U1_mag, U2_mag, U0_mag  — voltage amplitudes, V
        U1_ang, U2_ang, U0_ang  — voltage angles, degrees
    """

    # Channel definition: (attr_name, ch_id, uu)
    # ph and ccbm are empty strings for calculated (not measured) channels
    _CURRENT_CHANNELS = [
        ("I1_mag", "I1_mag", "A"),
        ("I2_mag", "I2_mag", "A"),
        ("I0_mag", "I0_mag", "A"),
        ("I1_ang", "I1_ang", "deg"),
        ("I2_ang", "I2_ang", "deg"),
        ("I0_ang", "I0_ang", "deg"),
    ]
    _VOLTAGE_CHANNELS = [
        ("U1_mag", "U1_mag", "V"),
        ("U2_mag", "U2_mag", "V"),
        ("U0_mag", "U0_mag", "V"),
        ("U1_ang", "U1_ang", "deg"),
        ("U2_ang", "U2_ang", "deg"),
        ("U0_ang", "U0_ang", "deg"),
    ]

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── public interface ───────────────────────────────────────────────────

    def export(
        self,
        result : SymSeqResult,
        stem   : Optional[str] = None,
    ) -> tuple[Path, Path]:
        """
        Write .cfg and .dat.

        Parameters
        ----------
        result : SymSeqResult
        stem   : base filename without an extension.
                 Default: <source_stem>_symseq

        Returns
        -------
        (cfg_path, dat_path)
        """
        if stem is None:
            stem = Path(result.source_file).stem + "_symseq"

        cfg_path = self.output_dir / f"{stem}.cfg"
        dat_path = self.output_dir / f"{stem}.dat"

        channels, phys_matrix = self._build_channels(result)
        self._write_cfg(cfg_path, result, channels)
        self._write_dat(dat_path, result, phys_matrix)

        return cfg_path, dat_path

    # ── build channels and the physical value matrix ─────────────────────

    def _build_channels(self, r: SymSeqResult):
        """
        Build a channel list and a matrix of physical values (float64).

        ASCII format requires no scaling: write physical
        values directly. Set a=1, b=0 in .cfg.

        Returns
        -------
        channels    : list of dicts with fields for .cfg
        phys_matrix : np.ndarray (n_steps, n_channels), dtype float64
        """
        all_defs = self._CURRENT_CHANNELS + self._VOLTAGE_CHANNELS
        channels  = []
        data_cols = []

        for idx, (attr, ch_id, uu) in enumerate(all_defs, start=1):
            phys = getattr(r, attr).copy()

            # radians → degrees for angle channels
            if uu == "deg":
                phys = np.degrees(phys)

            channels.append({
                "n"         : idx,
                "ch_id"     : ch_id,
                "ph"        : "",      # empty — calculated channel
                "ccbm"      : "",      # empty — calculated channel
                "uu"        : uu,
                "a"         : 1.0,     # physical = 1.0 * raw + 0  (ASCII: raw=physical)
                "b"         : 0.0,
                "skew"      : 0.0,
                "min"       : float(phys.min()),
                "max"       : float(phys.max()),
                "primary"   : 1.0,
                "secondary" : 1.0,
                "PS"        : "S",
            })
            data_cols.append(phys)

        phys_matrix = np.column_stack(data_cols)   # (n_steps, n_ch), float64
        return channels, phys_matrix

    # ── write .cfg ───────────────────────────────────────────────────────────

    def _write_cfg(
        self,
        path     : Path,
        r        : SymSeqResult,
        channels : list,
    ) -> None:
        """
        COMTRADE 1999 (.cfg) format, ASCII.
        Specification: IEEE Std C37.111-1999, section 5.
        """
        n_analog  = len(channels)
        n_digital = 0
        n_total   = n_analog + n_digital

        fs_out = r.fs
        stamp  = datetime.now().strftime("%d/%m/%Y,%H:%M:%S.%f")[:26]

        lines = []

        # line 1: station_name, rec_dev_id, rev_year
        lines.append(f"SymSeq_{r.source_file},{r.distance_km:.2f}km,1999")

        # line 2: TT,nA,nD
        lines.append(f"{n_total},{n_analog}A,{n_digital}D")

        # analog channel lines
        # format: n,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,PS
        for ch in channels:
            lines.append(
                f"{ch['n']},{ch['ch_id']},{ch['ph']},{ch['ccbm']},{ch['uu']},"
                f"{ch['a']:.9e},{ch['b']:.9e},{ch['skew']:.6f},"
                f"{ch['min']:.6f},{ch['max']:.6f},"
                f"{ch['primary']:.6f},{ch['secondary']:.6f},{ch['PS']}"
            )

        # grid frequency
        lines.append(f"{r.f0:.3f}")

        # sampling frequency: nrates / samp,endsamp
        lines.append("1")
        lines.append(f"{fs_out:.6f},{r.n_steps}")

        # timestamps of the first and last samples
        lines.append(stamp)
        lines.append(stamp)

        # data file format — ASCII
        lines.append("ASCII")

        # time step multiplier (timemult)
        lines.append("1")

        path.write_text("\n".join(lines) + "\n", encoding="ascii")

    # ── write .dat (ASCII) ───────────────────────────────────────────────────

    def _write_dat(
        self,
        path        : Path,
        r           : SymSeqResult,
        phys_matrix : np.ndarray,
    ) -> None:
        """
        COMTRADE ASCII .dat:
            each line = sample_number,timestamp,ch1,ch2,...,chN
            timestamp — microseconds since the first sample (integer)
            channel values — physical float values, comma-separated

        Example line:
            1,0,125.34,88.12,0.03,45.00,178.21,-2.10,...
        """
        dt_us = int(round(1_000_000.0 / r.fs))   # time step, microseconds

        lines = []
        for i in range(r.n_steps):
            sample_num = i + 1
            timestamp  = i * dt_us
            values     = ",".join(f"{v:.6f}" for v in phys_matrix[i])
            lines.append(f"{sample_num},{timestamp},{values}")

        path.write_text("\n".join(lines) + "\n", encoding="ascii")


# ═══════════════════════════════════════════════════════════════════════════════
# Batch export of multiple CSV files
# ═══════════════════════════════════════════════════════════════════════════════

def export_batch(
    csv_paths   : list[str | Path],
    output_dir  : Path = OUTPUT_DIR,
    f0          : float = DEFAULT_F0,
    fs_fallback : float = DEFAULT_FS_FALLBACK,
) -> list[tuple[Path, Path]]:
    """
    Process a list of CSV files and export each to COMTRADE.

    Parameters
    ----------
    csv_paths   : list of CSV paths
    output_dir  : where to write .cfg / .dat
    f0          : grid frequency
    fs_fallback : fs if not provided in the CSV

    Returns
    -------
    List of (cfg_path, dat_path) pairs for each file.
    """
    exporter = ComtradeExporter(output_dir=output_dir)
    results  = []

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        print(f"  ► {csv_path.name} ", end="", flush=True)
        try:
            result = sliding_symseq(csv_path, f0=f0, fs_fallback=fs_fallback)
            cfg_p, dat_p = exporter.export(result)
            print(f"→ {cfg_p.name}  ✓")
            results.append((cfg_p, dat_p))
        except Exception as e:
            print(f"  ERROR: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Export symmetrical components to COMTRADE (.cfg + .dat ASCII)"
    )
    p.add_argument(
        "--csv", default=DEFAULT_CSV,
        help=f"Path to the CSV file (default: {DEFAULT_CSV})"
    )
    p.add_argument(
        "--f0", type=float, default=DEFAULT_F0,
        help=f"Mains frequency in Hz (default: {DEFAULT_F0})"
    )
    p.add_argument(
        "--fs", type=float, default=DEFAULT_FS_FALLBACK,
        help=f"Fallback fs in Hz if missing from the CSV (default: {DEFAULT_FS_FALLBACK})"
    )
    p.add_argument(
        "--out", default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    p.add_argument(
        "--batch", nargs="+", metavar="CSV",
        help="Batch mode: multiple CSV files separated by spaces"
    )
    return p.parse_args()


def main():
    args = _parse_args()
    out  = Path(args.out)

    if args.batch:
        print(f"Batch mode: {len(args.batch)} files → {out}")
        export_batch(args.batch, output_dir=out, f0=args.f0, fs_fallback=args.fs)
        return

    print(f"Loading  : {args.csv}")
    result = sliding_symseq(args.csv, f0=args.f0, fs_fallback=args.fs)

    print(f"fs       : {result.fs:.1f} Hz")
    print(f"Window   : {result.win} samples ({1000/result.f0:.0f} ms)")
    print(f"Steps    : {result.n_steps}")
    print(f"Distance : {result.distance_km:.1f} km")
    print()
    print("Peak values:")
    print(f"  I1 = {result.I1_mag.max():.4f} A   I2 = {result.I2_mag.max():.4f} A   I0 = {result.I0_mag.max():.4f} A")
    print(f"  U1 = {result.U1_mag.max():.4f} V   U2 = {result.U2_mag.max():.4f} V   U0 = {result.U0_mag.max():.4f} V")
    print()

    exporter = ComtradeExporter(output_dir=out)
    cfg_p, dat_p = exporter.export(result)

    print("Written  :")
    print(f"  {cfg_p}")
    print(f"  {dat_p}")


if __name__ == "__main__":
    main()
