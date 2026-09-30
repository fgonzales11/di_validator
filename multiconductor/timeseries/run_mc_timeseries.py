"""Canonical and compatibility runners for multiconductor time series."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import pickle
import sys
from collections.abc import Iterable

from multiconductor.pycci.cci_powerflow import LoadflowNotConverged, run_pf
from multiconductor.studies import InvalidStudyInput
from multiconductor.timeseries.data_source import (
    MC_CSVDataSource,
    ProfileBinding,
    bindings_identity_hash,
)
from multiconductor.timeseries.output_writer import MC_OutputWriter, OutputWriter

logger = logging.getLogger(__name__)


def _build_name_map(table):
    """Build a name → {phase_int: MultiIndex_label} lookup from a network table."""

    name_map: dict[str, dict[int, tuple]] = {}
    for idx_label, row in table.iterrows():
        name = row["name"]
        phase = int(row["from_phase"])
        name_map.setdefault(name, {})[phase] = idx_label
    return name_map


def _apply_time_step_data(net, data, load_name_map, sgen_name_map):
    """Apply legacy CSV time-step values onto asymmetric load/sgen tables."""

    phase_map = {"a": 1, "b": 2, "c": 3}

    for (name, type_), values in data.items():
        if type_ == "GROSS":
            name_map = load_name_map
            table_name = "asymmetric_load"
        elif type_ == "GEN":
            name_map = sgen_name_map
            table_name = "asymmetric_sgen"
        else:
            logger.warning(
                "Unrecognized TYPE '%s' for element '%s'; skipping.", type_, name
            )
            continue

        if name not in name_map:
            logger.warning(
                "NAME '%s' (TYPE='%s') not found in %s; skipping.",
                name,
                type_,
                table_name,
            )
            continue

        phase_labels = name_map[name]
        table = net[table_name]

        for suffix, phase_int in phase_map.items():
            if phase_int not in phase_labels:
                continue
            idx_label = phase_labels[phase_int]
            table.at[idx_label, "p_mw"] = values[f"p_mw_{suffix}"]
            table.at[idx_label, "q_mvar"] = values[f"q_mvar_{suffix}"]


def _time_steps_list(time_steps: Iterable[object]) -> list[object]:
    return list(time_steps)


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _network_hash(net) -> str:
    try:
        return _hash_bytes(pickle.dumps(net, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        tables = {}
        for name in (
            "bus",
            "line",
            "asymmetric_load",
            "asymmetric_sgen",
            "controller",
        ):
            table = getattr(net, name, None)
            if table is None:
                continue
            try:
                tables[name] = table.to_json(orient="split", default_handler=str)
            except Exception:
                tables[name] = repr(table)
        payload = json.dumps(
            {
                "CIRCUIT_KEY": getattr(net, "CIRCUIT_KEY", ""),
                "tables": tables,
            },
            sort_keys=True,
            default=str,
        )
        return _hash_bytes(payload.encode("utf-8"))


def _time_steps_hash(time_steps: list[object]) -> str:
    payload = json.dumps(time_steps, default=str)
    return _hash_bytes(payload.encode("utf-8"))


def _capture_table_snapshot(table):
    if table is None:
        return None
    if hasattr(table, "copy"):
        return table.copy(deep=True)
    return copy.deepcopy(table)


def _capture_baselines(net, table_names: set[str]) -> dict[str, object]:
    snapshot = {}
    for table_name in table_names:
        snapshot[table_name] = _capture_table_snapshot(getattr(net, table_name, None))
    return snapshot


def _restore_baselines(net, snapshot: dict[str, object]) -> None:
    for table_name, table in snapshot.items():
        if table is None:
            continue
        setattr(net, table_name, _capture_table_snapshot(table))


def _invalidate_solver_cache(net) -> None:
    for attr in ("model", "_ppc", "_ppc0", "_topology_hash", "_topology_cache"):
        if hasattr(net, attr):
            try:
                delattr(net, attr)
            except AttributeError:
                continue


def _binding_phase_series(table, binding: ProfileBinding):
    if binding.phase is None:
        return table.index == table.index

    if "from_phase" in table.columns:
        return table["from_phase"].astype("Int64").eq(binding.phase)
    if "phase" in table.columns:
        return table["phase"].astype("Int64").eq(binding.phase)
    if getattr(table.index, "nlevels", 1) > 1:
        return table.index.get_level_values(-1) == binding.phase
    return table.index == table.index


def _binding_circuit_series(table, binding: ProfileBinding):
    if binding.circuit is None:
        return table.index == table.index
    if "circuit" in table.columns:
        return table["circuit"].astype("Int64").eq(binding.circuit)
    if getattr(table.index, "nlevels", 1) > 1:
        return table.index.get_level_values(-1) == binding.circuit
    return table.index == table.index


def _resolve_binding_rows(net, binding: ProfileBinding) -> list[object]:
    table = getattr(net, binding.table)
    if binding.variable not in table.columns:
        raise KeyError(f"{binding.table} is missing column {binding.variable!r}")

    mask = _binding_phase_series(table, binding) & _binding_circuit_series(table, binding)
    if binding.element is None:
        rows = list(table.index[mask])
    elif "name" in table.columns and isinstance(binding.element, str):
        rows = list(table.index[mask & table["name"].eq(binding.element)])
    elif getattr(table.index, "nlevels", 1) > 1 and not isinstance(binding.element, tuple):
        rows = list(
            table.index[
                mask & (table.index.get_level_values(0) == binding.element)
            ]
        )
    elif binding.element in table.index:
        rows = [binding.element] if bool(mask[table.index.get_loc(binding.element)]) else []
    else:
        rows = []

    if not rows:
        raise InvalidStudyInput(
            f"ProfileBinding for {binding.table}.{binding.variable} did not match any rows"
        )
    return rows


def _binding_value(step_values, binding: ProfileBinding, row_label, table) -> float:
    raw = step_values[binding.profile]
    if isinstance(raw, dict):
        if binding.phase is not None and binding.phase in raw:
            raw = raw[binding.phase]
        elif isinstance(row_label, tuple) and row_label[-1] in raw:
            raw = raw[row_label[-1]]
        else:
            raise KeyError(
                f"Profile {binding.profile!r} provided phase-specific values without a matching phase"
            )
    return float(raw) * float(binding.scale_factor)


def _apply_profile_bindings(net, step_values, binding_rows: dict[ProfileBinding, list[object]]):
    for binding, rows in binding_rows.items():
        if binding.profile not in step_values:
            raise KeyError(f"Missing profile {binding.profile!r} for time step")
        table = getattr(net, binding.table)
        for row_label in rows:
            value = _binding_value(step_values, binding, row_label, table)
            current = float(table.at[row_label, binding.variable])
            if binding.mode == "absolute":
                table.at[row_label, binding.variable] = value
            elif binding.mode == "scale":
                table.at[row_label, binding.variable] = current * value
            else:
                table.at[row_label, binding.variable] = current + value


def _write_checkpoint(
    checkpoint_path: str,
    *,
    net_hash: str,
    data_source_hash: str,
    bindings_hash: str,
    time_steps_hash: str,
    next_index: int,
) -> None:
    parent = checkpoint_path.rsplit("/", 1)[0] if "/" in checkpoint_path else ""
    if parent:
        import os

        os.makedirs(parent, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "net_hash": net_hash,
                "data_source_hash": data_source_hash,
                "bindings_hash": bindings_hash,
                "time_steps_hash": time_steps_hash,
                "next_index": next_index,
            },
            handle,
            indent=2,
            sort_keys=True,
        )


def _load_resume_index(
    checkpoint_path: str,
    *,
    net_hash: str,
    data_source_hash: str,
    bindings_hash: str,
    time_steps_hash: str,
) -> int:
    with open(checkpoint_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "net_hash": net_hash,
        "data_source_hash": data_source_hash,
        "bindings_hash": bindings_hash,
        "time_steps_hash": time_steps_hash,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise InvalidStudyInput(f"Checkpoint mismatch for {key}")
    return int(payload.get("next_index", 0))


def run_timeseries(
    net,
    data_source=None,
    time_steps=None,
    bindings: list[ProfileBinding] | None = None,
    output_writer=None,
    output_path="ts_results.csv",
    *,
    result_specs=None,
    tol_vmag_pu=1e-5,
    tol_vang_rad=1e-5,
    max_iter=100,
    run_control=False,
    failure_policy="raise",
    auto_fix=True,
    verbose=True,
    write_time=None,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = None,
    resume: bool = False,
    chunk_size: int | None = None,
    restore_baseline: bool = True,
    invalidate_topology_cache: bool = False,
):
    """Run the canonical multiconductor time-series workflow."""

    if failure_policy not in {"raise", "stop", "record_and_continue"}:
        raise InvalidStudyInput(
            "failure_policy must be 'raise', 'stop', or 'record_and_continue'"
        )

    if data_source is None:
        data_source = MC_CSVDataSource("ts.csv")

    if time_steps is None:
        time_steps = data_source.get_time_steps()
    time_steps = _time_steps_list(time_steps)

    legacy_mode = (
        bindings is None
        and hasattr(data_source, "get_time_step_values")
        and hasattr(net, "asymmetric_load")
        and hasattr(net, "asymmetric_sgen")
    )
    if output_writer is None:
        if legacy_mode:
            output_writer = MC_OutputWriter(net, time_steps, output_path, write_time)
        else:
            output_writer = OutputWriter(
                net,
                time_steps,
                result_specs=result_specs,
                output_path=output_path,
                write_time=write_time,
                chunk_size=chunk_size,
                append=resume,
            )

    load_name_map = {}
    sgen_name_map = {}
    binding_rows: dict[ProfileBinding, list[object]] = {}
    touched_tables = {"controller"}
    step_tables: set[str] = set()

    if legacy_mode:
        load_name_map = _build_name_map(net.asymmetric_load)
        sgen_name_map = _build_name_map(net.asymmetric_sgen)
        touched_tables.update({"asymmetric_load", "asymmetric_sgen"})
        step_tables.update({"asymmetric_load", "asymmetric_sgen"})
    else:
        if bindings is None:
            raise InvalidStudyInput(
                "Canonical run_timeseries requires bindings or a legacy MC_CSVDataSource"
            )
        for binding in bindings:
            touched_tables.add(binding.table)
            step_tables.add(binding.table)
            binding_rows[binding] = _resolve_binding_rows(net, binding)

    n_steps = len(time_steps)
    if n_steps > 8760:
        n_buses = len(net.bus.index.get_level_values(0).unique())
        est_bytes = n_steps * n_buses * 6 * 8
        est_mb = est_bytes / (1024 * 1024)
        logger.info(
            "Time series has %d time steps (%d buses). Estimated result buffer: %.1f MB.",
            n_steps,
            n_buses,
            est_mb,
        )

    if auto_fix:
        try:
            import sce.wrapper as sce_wrapper

            sce_wrapper.run_pf(net, auto_fix=True)
        except ImportError:
            logger.warning("sce.wrapper is not available; skipping auto-fix step.")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    iterator = time_steps
    if verbose and tqdm is not None:
        iterator = tqdm(time_steps, desc="MC Timeseries", file=sys.stderr)

    net_hash = _network_hash(net)
    data_source_hash = data_source.identity_hash()
    bindings_hash = bindings_identity_hash(bindings)
    time_steps_digest = _time_steps_hash(time_steps)
    start_index = 0
    if resume:
        if checkpoint_path is None:
            raise InvalidStudyInput("resume=True requires checkpoint_path")
        start_index = _load_resume_index(
            checkpoint_path,
            net_hash=net_hash,
            data_source_hash=data_source_hash,
            bindings_hash=bindings_hash,
            time_steps_hash=time_steps_digest,
        )

    baseline_snapshot = _capture_baselines(net, touched_tables)
    step_snapshot = _capture_baselines(net, step_tables)
    completed = 0

    try:
        for index, time_step in enumerate(iterator):
            if index < start_index:
                continue

            _restore_baselines(net, step_snapshot)
            data = data_source.get_time_step_values(time_step)
            if legacy_mode:
                _apply_time_step_data(net, data, load_name_map, sgen_name_map)
            else:
                _apply_profile_bindings(net, data, binding_rows)

            if invalidate_topology_cache:
                _invalidate_solver_cache(net)

            converged = True
            status = "converged"
            message = None
            try:
                solver_kwargs = {
                    "tol_vmag_pu": tol_vmag_pu,
                    "tol_vang_rad": tol_vang_rad,
                    "MaxIter": max_iter,
                }
                if run_control:
                    solver_kwargs["run_control"] = True
                run_pf(net, **solver_kwargs)
            except LoadflowNotConverged as exc:
                converged = False
                status = "nonconverged"
                message = str(exc)
                logger.error("Power flow did not converge at time step %s.", time_step)
                if failure_policy == "raise":
                    raise
            except Exception as exc:
                converged = False
                status = "error"
                message = f"{type(exc).__name__}: {exc}"
                if failure_policy == "raise":
                    raise

            if isinstance(output_writer, OutputWriter):
                output_writer.record_time_step(
                    net,
                    time_step,
                    converged,
                    status=status,
                    message=message,
                )
            else:
                output_writer.record_time_step(net, time_step, converged)

            completed = index + 1
            if checkpoint_path is not None and checkpoint_every is not None:
                if completed % checkpoint_every == 0:
                    _write_checkpoint(
                        checkpoint_path,
                        net_hash=net_hash,
                        data_source_hash=data_source_hash,
                        bindings_hash=bindings_hash,
                        time_steps_hash=time_steps_digest,
                        next_index=completed,
                    )

            if not converged and failure_policy == "stop":
                break

        output_writer.finalize()
        if checkpoint_path is not None:
            _write_checkpoint(
                checkpoint_path,
                net_hash=net_hash,
                data_source_hash=data_source_hash,
                bindings_hash=bindings_hash,
                time_steps_hash=time_steps_digest,
                next_index=completed,
            )
        return output_writer
    finally:
        if restore_baseline:
            _restore_baselines(net, baseline_snapshot)


def run_mc_timeseries(
    net,
    data_source=None,
    time_steps=None,
    output_writer=None,
    output_path="ts_results.csv",
    tol_vmag_pu=1e-5,
    tol_vang_rad=1e-5,
    max_iter=100,
    continue_on_divergence=False,
    auto_fix=True,
    verbose=True,
    write_time=None,
):
    """Compatibility wrapper around :func:`run_timeseries`."""

    failure_policy = "record_and_continue" if continue_on_divergence else "raise"
    return run_timeseries(
        net,
        data_source=data_source,
        time_steps=time_steps,
        output_writer=output_writer,
        output_path=output_path,
        tol_vmag_pu=tol_vmag_pu,
        tol_vang_rad=tol_vang_rad,
        max_iter=max_iter,
        run_control=False,
        failure_policy=failure_policy,
        auto_fix=auto_fix,
        verbose=verbose,
        write_time=write_time,
    )
