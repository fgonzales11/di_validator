"""Deterministic synthetic fault waveforms with a known uniform-line distance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import PHASES


def fault_frame(
    fs: float,
    f0: float,
    loop: str,
    distance_km: float,
    z1_ohm_km: complex,
    z0_ohm_km: complex | None = None,
    onset_seconds: float = 0.1,
    fault_seconds: float | None = None,
    total_seconds: float = 0.3,
    nominal_kv: float = 13.2,
    load_ka: float = 0.1,
    fault_ka: float = 3.0,
) -> pd.DataFrame:
    """Offset, IA..IC in A and UA..UC in V for a bolted fault `distance_km` down a uniform line.

    The faulted loop satisfies U_loop = I_loop * z1 * d exactly, with ground loops
    using I_phase + k0*(IA+IB+IC) when z0 is given, so the reactance method returns
    `distance_km`. `loop="none"` keeps balanced load throughout. The fault clears
    after `fault_seconds` (None: persists to the end).
    """
    a = np.exp(-2j * np.pi / 3)
    rotation = np.array([1, a, a**2])
    healthy_i = load_ka * np.exp(-0.3j) * rotation
    healthy_u = nominal_kv / np.sqrt(3) * rotation
    fault_i, fault_u = healthy_i.copy(), healthy_u.copy()
    z = z1_ohm_km * distance_km
    current = fault_ka * np.exp(-1.2j)
    if loop == "POS":
        fault_i = current * rotation
        fault_u = fault_i * z
    elif loop.endswith("G"):
        phase = "ABC".index(loop[0])
        fault_i[phase] = current
        k0 = 0j if z0_ohm_km is None else (z0_ohm_km - z1_ohm_km) / (3 * z1_ohm_km)
        fault_u[phase] = (current + k0 * fault_i.sum()) * z
    elif loop != "none":
        left, right = "ABC".index(loop[0]), "ABC".index(loop[1])
        fault_i[left], fault_i[right] = current, -current
        fault_u[left], fault_u[right] = current * z, -current * z
    t = np.arange(int(round(total_seconds * fs))) / fs
    end = np.inf if fault_seconds is None else onset_seconds + fault_seconds
    faulted = (t >= onset_seconds) & (t < end)
    phasors = np.where(
        faulted[:, None], np.r_[fault_i, fault_u][None, :], np.r_[healthy_i, healthy_u][None, :]
    )
    waveform = np.sqrt(2) * np.real(phasors * np.exp(2j * np.pi * f0 * t)[:, None]) * 1000
    frame = pd.DataFrame(waveform, columns=list(PHASES))
    frame.insert(0, "offset", t)
    return frame
