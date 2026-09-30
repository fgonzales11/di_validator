"""Writers for multiconductor time-series study outputs."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResultSpec:
    """Select one result column from one result table."""

    table: str
    column: str
    alias: str | None = None


def _jsonable_index(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable_index(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _phase_from_index(index: object) -> object | None:
    if isinstance(index, tuple) and index:
        return _jsonable_index(index[-1])
    return None


class OutputWriter:
    """Generic writer for canonical time-series results and status rows."""

    RESULT_COLUMNS = [
        "CIRCUIT_KEY",
        "TIME_STEP",
        "TABLE",
        "INDEX",
        "PHASE",
        "COLUMN",
        "VALUE",
    ]
    STATUS_COLUMNS = [
        "CIRCUIT_KEY",
        "TIME_STEP",
        "STATUS",
        "MESSAGE",
    ]

    def __init__(
        self,
        net,
        time_steps,
        *,
        result_specs: list[ResultSpec] | None = None,
        output_path: str = "ts_results.csv",
        status_path: str | None = None,
        chunk_size: int | None = None,
        write_time: float | None = None,
        append: bool = False,
    ) -> None:
        self._circuit_key = getattr(net, "CIRCUIT_KEY", "")
        self._time_steps = list(time_steps)
        self._output_path = output_path
        self._status_path = status_path or self._derive_status_path(output_path)
        self._chunk_size = chunk_size
        self._write_time = write_time
        self._append = append
        self._result_specs = (
            [
                ResultSpec("res_bus", "vm_pu"),
                ResultSpec("res_bus", "va_degree"),
            ]
            if result_specs is None
            else list(result_specs)
        )
        self._result_rows: list[dict[str, object]] = []
        self._status_rows: list[dict[str, object]] = []
        self._result_history: list[dict[str, object]] = []
        self._status_history: list[dict[str, object]] = []
        self._last_flush_time = time.time()
        self._header_written = append and os.path.exists(self._output_path)
        self._status_header_written = append and os.path.exists(self._status_path)

    @staticmethod
    def _derive_status_path(output_path: str) -> str:
        root, ext = os.path.splitext(output_path)
        suffix = ext or ".csv"
        return f"{root}.status{suffix}"

    @property
    def result_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._result_history, columns=self.RESULT_COLUMNS)

    @property
    def status_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._status_history, columns=self.STATUS_COLUMNS)

    def record_time_step(
        self,
        net,
        time_step,
        converged: bool,
        *,
        status: str = "converged",
        message: str | None = None,
    ) -> None:
        if converged:
            for spec in self._result_specs:
                table = getattr(net, spec.table, None)
                if not isinstance(table, pd.DataFrame) or spec.column not in table.columns:
                    continue
                label = spec.alias or spec.column
                for index, value in table[spec.column].items():
                    self._result_rows.append(
                        {
                            "CIRCUIT_KEY": self._circuit_key,
                            "TIME_STEP": time_step,
                            "TABLE": spec.table,
                            "INDEX": json.dumps(_jsonable_index(index)),
                            "PHASE": _phase_from_index(index),
                            "COLUMN": label,
                            "VALUE": value,
                        }
                    )
                    self._result_history.append(self._result_rows[-1].copy())
        self.record_status(time_step, status=status, message=message)
        self._maybe_flush()

    def record_status(
        self,
        time_step,
        *,
        status: str,
        message: str | None = None,
    ) -> None:
        row = {
            "CIRCUIT_KEY": self._circuit_key,
            "TIME_STEP": time_step,
            "STATUS": status,
            "MESSAGE": message,
        }
        self._status_rows.append(row)
        self._status_history.append(row.copy())

    def _maybe_flush(self) -> None:
        if self._chunk_size is not None and len(self._status_rows) >= self._chunk_size:
            self.flush()
            return
        if self._write_time is None:
            return
        elapsed_minutes = (time.time() - self._last_flush_time) / 60.0
        if elapsed_minutes >= self._write_time:
            self.flush()

    def flush(self) -> None:
        self._flush_rows(
            self._output_path,
            self.RESULT_COLUMNS,
            self._result_rows,
            header_written_attr="_header_written",
        )
        self._flush_rows(
            self._status_path,
            self.STATUS_COLUMNS,
            self._status_rows,
            header_written_attr="_status_header_written",
        )
        self._last_flush_time = time.time()

    def finalize(self) -> None:
        self.flush()

    def _flush_rows(
        self,
        path: str,
        columns: list[str],
        rows: list[dict[str, object]],
        *,
        header_written_attr: str,
    ) -> None:
        if not rows:
            return
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        frame = pd.DataFrame(rows, columns=columns)
        header_written = getattr(self, header_written_attr)
        try:
            if header_written:
                frame.to_csv(path, mode="a", header=False, index=False)
            else:
                frame.to_csv(path, index=False)
                setattr(self, header_written_attr, True)
        except Exception as exc:  # pragma: no cover - surfaced in callers
            raise IOError(f"Failed to write results to {path}: {exc}") from exc
        rows.clear()


class MC_OutputWriter:
    """Legacy wide bus-voltage writer used by notebook-era workflows."""

    OUTPUT_COLUMNS = [
        "CIRCUIT_KEY",
        "TIME_STEP",
        "BUS",
        "Vm_A_pu",
        "Va_A_deg",
        "Vm_B_pu",
        "Va_B_deg",
        "Vm_C_pu",
        "Va_C_deg",
    ]

    _VM_A = 0
    _VA_A = 1
    _VM_B = 2
    _VA_B = 3
    _VM_C = 4
    _VA_C = 5

    def __init__(
        self,
        net,
        time_steps,
        output_path: str = "ts_results.csv",
        write_time=None,
    ):
        bus_index = net.bus.index.get_level_values(0).unique()
        self._bus_indices = np.array(sorted(bus_index), dtype=int)
        self._n_buses = len(self._bus_indices)
        self._circuit_key = net.CIRCUIT_KEY
        self._time_steps = list(time_steps)
        self._n_time_steps = len(self._time_steps)
        self._output_path = output_path
        total_rows = self._n_time_steps * self._n_buses
        self._voltage_buffer = np.full((total_rows, 6), np.nan, dtype=np.float64)
        self._time_step_buffer = np.empty(total_rows, dtype=object)
        self._bus_buffer = np.empty(total_rows, dtype=int)

        for i, ts in enumerate(self._time_steps):
            start = i * self._n_buses
            end = start + self._n_buses
            self._time_step_buffer[start:end] = ts
            self._bus_buffer[start:end] = self._bus_indices

        self._write_ptr = 0
        self._flush_ptr = 0
        self._write_time = write_time
        self._last_flush_time = time.time()
        self._header_written = False

    def record_time_step(self, net, time_step, converged):
        start = self._write_ptr
        end = start + self._n_buses

        if converged:
            res_bus = net.res_bus
            for i, bus_idx in enumerate(self._bus_indices):
                row = start + i
                for phase, vm_idx, va_idx in (
                    (1, self._VM_A, self._VA_A),
                    (2, self._VM_B, self._VA_B),
                    (3, self._VM_C, self._VA_C),
                ):
                    try:
                        self._voltage_buffer[row, vm_idx] = res_bus.loc[
                            (bus_idx, phase), "vm_pu"
                        ]
                        self._voltage_buffer[row, va_idx] = res_bus.loc[
                            (bus_idx, phase), "va_degree"
                        ]
                    except KeyError:
                        continue

        self._write_ptr = end

        if self._write_time is not None:
            elapsed_minutes = (time.time() - self._last_flush_time) / 60.0
            if elapsed_minutes >= self._write_time:
                self._periodic_flush()

    def finalize(self):
        filled = self._write_ptr
        df = self._build_dataframe(self._flush_ptr, filled)
        parent = os.path.dirname(self._output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            if self._header_written:
                df.to_csv(self._output_path, mode="a", header=False, index=False)
            else:
                df.to_csv(self._output_path, index=False)
        except Exception as exc:  # pragma: no cover - surfaced in callers
            raise IOError(
                f"Failed to write results to {self._output_path}: {exc}"
            ) from exc

    def _build_dataframe(self, start_row, end_row):
        n = end_row - start_row
        if n <= 0:
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        data = {
            "CIRCUIT_KEY": [self._circuit_key] * n,
            "TIME_STEP": self._time_step_buffer[start_row:end_row],
            "BUS": self._bus_buffer[start_row:end_row],
            "Vm_A_pu": self._voltage_buffer[start_row:end_row, self._VM_A],
            "Va_A_deg": self._voltage_buffer[start_row:end_row, self._VA_A],
            "Vm_B_pu": self._voltage_buffer[start_row:end_row, self._VM_B],
            "Va_B_deg": self._voltage_buffer[start_row:end_row, self._VA_B],
            "Vm_C_pu": self._voltage_buffer[start_row:end_row, self._VM_C],
            "Va_C_deg": self._voltage_buffer[start_row:end_row, self._VA_C],
        }
        return pd.DataFrame(data, columns=self.OUTPUT_COLUMNS)

    def _periodic_flush(self):
        filled = self._write_ptr
        if filled <= self._flush_ptr:
            return

        df = self._build_dataframe(self._flush_ptr, filled)
        try:
            parent = os.path.dirname(self._output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if not self._header_written:
                df.to_csv(self._output_path, index=False)
                self._header_written = True
            else:
                df.to_csv(self._output_path, mode="a", header=False, index=False)
            self._flush_ptr = filled
        except Exception:
            logger.error(
                "Periodic flush to %s failed; data remains in buffer.",
                self._output_path,
                exc_info=True,
            )
        self._last_flush_time = time.time()
