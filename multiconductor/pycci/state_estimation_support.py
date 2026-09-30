"""Frontend-facing multiconductor state-estimation orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

import multiconductor as mc
import numpy as np
import pandas as pd

from multiconductor.pycci.state_estimator import (
    StateEstimationNotConverged,
    UnobservableNetworkError,
    estimate,
    measurements_from_dataframe,
    measurements_from_power_flow,
)
from multiconductor.studies import InvalidStudyInput

from .network_converters import mc_result_tables_to_ui


SUPPORTED_MEASUREMENT_TYPES = {"p", "q", "i", "im", "ia", "v", "vm", "va", "vr", "vi"}
SUPPORTED_MEASUREMENT_ELEMENTS = {
    "bus",
    "node",
    "zero_injection",
    "line",
    "transformer",
    "trafo1ph",
    "source",
    "ext_grid",
    "ext_grid_sequence",
    "switch",
}


def compatible_state_measurement_count(net: Any) -> int:
    """Count rows that can be targeted by the phase-aware estimator adapter."""
    frame = getattr(net, "measurement", None)
    if frame is None or frame.empty:
        return 0

    count = 0
    for _, row in frame.iterrows():
        kind = str(row.get("measurement_type", "")).strip().lower()
        element_type = str(row.get("element_type", "")).strip().lower()
        try:
            value = float(row.get("value"))
            std_dev = float(row.get("std_dev"))
        except (TypeError, ValueError):
            continue
        if kind not in SUPPORTED_MEASUREMENT_TYPES:
            continue
        if element_type not in SUPPORTED_MEASUREMENT_ELEMENTS:
            continue
        if not math.isfinite(value) or not math.isfinite(std_dev) or std_dev <= 0:
            continue
        status = str(row.get("status", "")).strip().lower()
        if status in {"disabled", "inactive", "out_of_service", "rejected"}:
            continue
        terminal = row.get("terminal", np.nan)
        has_terminal = not pd.isna(terminal) and bool(str(terminal).strip())
        phase = row.get("phase", row.get("circuit", np.nan))
        if pd.isna(phase):
            phase = row.get("circuit", np.nan)
        if element_type in {"bus", "node", "zero_injection"} and pd.isna(phase) and not has_terminal:
            continue
        if element_type in {
            "transformer",
            "trafo1ph",
            "source",
            "ext_grid",
            "ext_grid_sequence",
            "switch",
        } and pd.isna(phase) and not has_terminal:
            continue
        count += 1
    return count


def analysis_capabilities(net: Any, engine: str) -> dict[str, Any]:
    frame = getattr(net, "measurement", None)
    has_measurements = bool(frame is not None and not frame.empty)
    compatible_count = compatible_state_measurement_count(net) if engine == "mc" else 0
    return {
        "hasMeasurements": has_measurements,
        "stateEstimation": {
            "supported": engine == "mc",
            "compatibleMeasurementCount": compatible_count,
        },
    }


def _json_source_index(value: Any) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, (str, int, float)):
            return scalar
    return str(value)


def run_mc_state_estimation(
    net: Any,
    network_id: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run state estimation and return the shared browser response contract."""
    options = options or {}
    compatible_count = compatible_state_measurement_count(net)
    source = str(
        options.get(
            "measurementSource",
            "network" if compatible_count > 0 else "synthetic",
        )
    ).strip().lower()
    if source not in {"network", "synthetic"}:
        raise InvalidStudyInput("measurementSource must be 'network' or 'synthetic'.")

    if source == "network":
        measurements = measurements_from_dataframe(net)
    else:
        synthetic = options.get("synthetic")
        synthetic = synthetic if isinstance(synthetic, dict) else {}
        mc.run_pf(
            net,
            tol_vmag_pu=1e-7,
            tol_vang_rad=1e-7,
            MaxIter=50,
            run_control=False,
        )
        measurements = measurements_from_power_flow(
            net,
            sigma_p=float(synthetic.get("sigmaP", 0.001)),
            sigma_q=float(synthetic.get("sigmaQ", 0.001)),
            sigma_vm=float(synthetic.get("sigmaVm", 0.0005)),
            sigma_vpmu=float(synthetic.get("sigmaVpmu", 0.0005)),
            include_line_flows=bool(synthetic.get("includeLineFlows", False)),
            noise=bool(synthetic.get("noise", False)),
            seed=int(synthetic.get("seed", 0)),
        )

    ordered = measurements.ordered_measurements()
    bad_data_threshold_value = options.get("badDataThreshold", 4.0)
    bad_data_threshold = (
        None
        if bad_data_threshold_value is None
        else float(bad_data_threshold_value)
    )
    result = estimate(
        net,
        measurements,
        tol=float(options.get("tol", 1e-8)),
        max_iter=int(options.get("maxIter", 30)),
        objective_tol=float(options.get("objectiveTol", 1e-10)),
        bad_data_threshold=bad_data_threshold,
        max_bad_data=int(options.get("maxBadData", 1)),
    )
    bus_results, line_results = mc_result_tables_to_ui(
        net,
        res_bus=result.res_bus,
        res_line=result.res_line,
        res_trafo=result.res_trafo,
    )
    rejected = [
        {
            "position": int(position),
            "sourceIndex": _json_source_index(source_index),
        }
        for position, source_index in zip(
            result.bad_measurement_positions,
            result.bad_measurement_indices,
        )
    ]
    p_value = float(result.p_value) if math.isfinite(result.p_value) else None
    return {
        "success": bool(result.converged),
        "networkId": network_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "measurementSource": source,
        "observable": result.observability.structural_rank >= result.observability.state_count,
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "maxStateError": float(result.error),
        "objective": float(result.objective),
        "chiSquare": float(result.chi_square),
        "degreesOfFreedom": int(result.degrees_of_freedom),
        "pValue": p_value,
        "stateCount": int(result.state_count),
        "observabilityRank": int(result.observability.structural_rank),
        "numericObservabilityRank": (
            None if result.observability.numeric_rank is None else int(result.observability.numeric_rank)
        ),
        "zeroObservabilityColumns": list(result.observability.zero_columns),
        "measurementCount": len(ordered),
        "acceptedMeasurementCount": len(ordered) - len(rejected),
        "rejectedMeasurements": rejected,
        "measurementResults": net.res_measurement_est.to_dict(orient="records"),
        "busResults": bus_results,
        "lineResults": line_results,
    }


__all__ = [
    "StateEstimationNotConverged",
    "UnobservableNetworkError",
    "analysis_capabilities",
    "compatible_state_measurement_count",
    "run_mc_state_estimation",
]
