from __future__ import annotations

from typing import Optional

import numpy as np


def effective_three_phase_turns_ratio(
        vn_hv_kv: float,
        vn_lv_kv: float,
        tap_side: str = "lv",
        tap_neutral: int = 0,
        tap_pos: int = 0,
        tap_step_percent: float = 0.0) -> float:
    tap_multiplier = 1.0 + (tap_pos - tap_neutral) * tap_step_percent / 100.0
    if abs(tap_multiplier) < 1e-12:
        raise ValueError("Three-phase tap settings produce a zero effective winding voltage.")

    if tap_side == "lv":
        hv_eff_kv = vn_hv_kv
        lv_eff_kv = vn_lv_kv * tap_multiplier
    elif tap_side == "hv":
        hv_eff_kv = vn_hv_kv * tap_multiplier
        lv_eff_kv = vn_lv_kv
    else:
        raise ValueError(f'Unsupported tap_side "{tap_side}" for three-phase transformer.')

    if hv_eff_kv <= 0 or lv_eff_kv <= 0:
        raise ValueError("Three-phase transformer effective winding voltages must be positive.")
    return hv_eff_kv / lv_eff_kv


def sequence_impedance_to_phase_impedance(z0: complex, z1: complex, z2: Optional[complex] = None) -> np.ndarray:
    if z2 is None:
        z2 = z1

    phase_shift = np.exp(1j * 2.0 * np.pi / 3.0)
    sequence_to_phase = np.array([
        [1.0, 1.0, 1.0],
        [1.0, phase_shift ** 2, phase_shift],
        [1.0, phase_shift, phase_shift ** 2],
    ], dtype=np.complex128)

    return sequence_to_phase @ np.diag([z0, z1, z2]) @ np.linalg.inv(sequence_to_phase)


def three_phase_port_admittance(nt: float, z0: complex, z1: complex, z2: Optional[complex] = None) -> np.ndarray:
    if abs(nt) < 1e-12:
        raise ValueError("Three-phase transformer turns ratio must be non-zero.")

    z_phase = sequence_impedance_to_phase_impedance(z0, z1, z2)
    y_phase = np.linalg.inv(z_phase)

    y_port = np.zeros((6, 6), dtype=np.complex128)
    y_port[0:3, 0:3] = y_phase / (nt * nt)
    y_port[0:3, 3:6] = -y_phase / nt
    y_port[3:6, 0:3] = -y_phase / nt
    y_port[3:6, 3:6] = y_phase
    return y_port


def winding_voltage_incidence(connection: str) -> np.ndarray:
    normalized = connection.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"y", "yn", "wye", "grounded_wye", "grounded_wye_neutral"}:
        return np.array([
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, -1.0],
        ], dtype=np.complex128)

    if normalized in {"d", "delta"}:
        return np.array([
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
        ], dtype=np.complex128)

    raise ValueError(f'Unsupported three-phase transformer winding connection "{connection}".')


def three_phase_primitive_admittance(
        nt: float,
        z0: complex,
        z1: complex,
        z2: Optional[complex] = None,
        hv_connection: str = "wye",
        lv_connection: str = "wye") -> np.ndarray:
    y_port = three_phase_port_admittance(nt, z0, z1, z2)

    hv_incidence = winding_voltage_incidence(hv_connection)
    lv_incidence = winding_voltage_incidence(lv_connection)

    voltage_incidence = np.zeros((6, 8), dtype=np.complex128)
    voltage_incidence[0:3, 0:4] = hv_incidence
    voltage_incidence[3:6, 4:8] = lv_incidence

    return voltage_incidence.T @ y_port @ voltage_incidence


def primitive_admittance_to_per_unit(y_primitive_actual: np.ndarray, vbase_volts: np.ndarray, sbase_va: float) -> np.ndarray:
    row_scale = np.diag(vbase_volts / sbase_va)
    col_scale = np.diag(vbase_volts)
    return row_scale @ y_primitive_actual @ col_scale