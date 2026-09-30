from __future__ import annotations

import numpy as np


def effective_center_tap_turns_ratio(
        vn_hv_kv: float,
        vn_lv_kv: float,
        tap_side: str = "lv",
        tap_neutral: int = 0,
        tap_pos: int = 0,
        tap_step_percent: float = 0.0) -> float:
    tap_multiplier = 1.0 + (tap_pos - tap_neutral) * tap_step_percent / 100.0
    if abs(tap_multiplier) < 1e-12:
        raise ValueError("Center-tap tap settings produce a zero effective winding voltage.")

    if tap_side == "lv":
        hv_eff_kv = vn_hv_kv
        lv_eff_kv = vn_lv_kv * tap_multiplier
    elif tap_side == "hv":
        hv_eff_kv = vn_hv_kv * tap_multiplier
        lv_eff_kv = vn_lv_kv
    else:
        raise ValueError(f'Unsupported tap_side "{tap_side}" for center-tap transformer.')

    if hv_eff_kv <= 0 or lv_eff_kv <= 0:
        raise ValueError("Center-tap transformer effective winding voltages must be positive.")
    return hv_eff_kv / lv_eff_kv


def center_tap_port_admittance(nt: float, z0: complex, z1: complex, z2: complex) -> np.ndarray:
    if abs(nt) < 1e-12:
        raise ValueError("Center-tap transformer turns ratio must be non-zero.")

    d_matrix = np.array([[1.0, -1.0]], dtype=np.complex128) / (2.0 * nt)
    e_matrix = np.array([[1.0], [1.0]], dtype=np.complex128) / (2.0 * nt)

    z0_referred = z0 / (4.0 * nt * nt)
    f_matrix = np.array([
        [z1 + z0_referred, -z0_referred],
        [z0_referred, -(z2 + z0_referred)],
    ], dtype=np.complex128)

    f_inv = np.linalg.inv(f_matrix)

    y_port = np.zeros((3, 3), dtype=np.complex128)
    y_port[1:, 0:1] = f_inv @ e_matrix
    y_port[1:, 1:] = -f_inv
    y_port[0, 0] = (d_matrix @ f_inv @ e_matrix)[0, 0]
    y_port[0, 1:] = -(d_matrix @ f_inv)[0]
    return y_port


def center_tap_primitive_admittance(nt: float, z0: complex, z1: complex, z2: complex) -> np.ndarray:
    y_port = center_tap_port_admittance(nt, z0, z1, z2)

    # Voltage rows follow the figure:
    #   Vs  = V(hv_from) - V(hv_to)
    #   V1  = V(hot_1) - V(neutral)
    #   V2  = V(neutral) - V(hot_2)
    voltage_incidence = np.array([
        [1.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, -1.0],
    ], dtype=np.complex128)

    # Current rows map port-current sign conventions to nodal injections.
    current_incidence = np.array([
        [1.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, 1.0],
    ], dtype=np.complex128)

    return current_incidence.T @ y_port @ voltage_incidence


def primitive_admittance_to_per_unit(y_primitive_actual: np.ndarray, vbase_volts: np.ndarray, sbase_va: float) -> np.ndarray:
    row_scale = np.diag(vbase_volts / sbase_va)
    col_scale = np.diag(vbase_volts)
    return row_scale @ y_primitive_actual @ col_scale
