"""Reviewed numerical functions from fault-distance/fault_distance.ipynb.

Preserves the notebook DC filter, inception criterion, sequence features and
single-ended reactance method. Offline: centered filtering uses future samples.
"""

import numpy as np
import pandas as pd


def remove_dc_period(values: np.ndarray, fs: float, f_net: float) -> np.ndarray:
    """Subtract a centered moving average over one grid period."""
    period = int(round(fs / f_net))
    filtered = np.zeros_like(values)
    for index, waveform in enumerate(values):
        aper = pd.Series(waveform).rolling(window=period, center=True, min_periods=1).mean()
        half = period // 2
        aper.iloc[:half] = aper.iloc[half:period].mean()
        aper.iloc[-half:] = aper.iloc[-period:-half].mean()
        filtered[index] = waveform - aper.to_numpy()
    return filtered


def detect_fault_inception(ia, ib, ic, ua, ub, uc, fs, f_net, Inom, eta_I=0.5, eta_U=0.85):
    """Detect fault inception using a combined sliding-window RMS criterion.

    The algorithm compares current and voltage RMS in two adjacent windows
    of one-quarter period each (k = fs/(4*f_net)).
    A fault is detected when the current rises sharply in at least one phase
    (I_post / I_pre > 1 + eta_I) and the voltage drops
    (U_post / U_pre < eta_U).
    """
    k = int(fs / (4 * f_net))
    n = len(ia)

    def window_rms(x, start, end):
        return np.sqrt(np.mean(x[start:end] ** 2))

    I_ratio = np.ones(n)
    U_ratio = np.ones(n)

    for i in range(k, n - k):
        I_pre = max(window_rms(ia, i - k, i), window_rms(ib, i - k, i), window_rms(ic, i - k, i))
        I_post = max(window_rms(ia, i, i + k), window_rms(ib, i, i + k), window_rms(ic, i, i + k))
        U_pre = min(window_rms(ua, i - k, i), window_rms(ub, i - k, i), window_rms(uc, i - k, i))
        U_post = min(window_rms(ua, i, i + k), window_rms(ub, i, i + k), window_rms(uc, i, i + k))

        I_ratio[i] = I_post / (I_pre + 1e-9) if I_pre > 1e-6 else 1.0
        U_ratio[i] = U_post / (U_pre + 1e-9) if U_pre > 1e-6 else 1.0

    # Skip the first 2 windows to prevent false triggering
    # at the start of the recording (energization, switching surge)
    skip = 2 * k
    candidates = np.where((I_ratio > (1.0 + eta_I)) & (U_ratio < eta_U))[0]
    candidates = candidates[candidates >= skip]

    if len(candidates) == 0:
        return None, I_ratio, U_ratio

    fault_idx = candidates[0]
    return fault_idx, I_ratio, U_ratio


def extract_window(signal, center, n_pre, n_post):
    """Extract a fixed-length n_pre + n_post window, padding with zeros outside the bounds."""
    start = center - n_pre
    end = center + n_post
    pad_left = max(0, -start)
    pad_right = max(0, end - len(signal))
    valid_start = max(0, start)
    valid_end = min(len(signal), end)
    window = np.concatenate([np.zeros(pad_left), signal[valid_start:valid_end], np.zeros(pad_right)])
    return window


def sliding_fortescue_magnitudes(abc: np.ndarray, fs: float, f0: float) -> np.ndarray:
    """Return one-cycle fundamental peak magnitudes in zero/positive/negative order.

    Args:
        abc: Finite signals with shape (3, N), in A/B/C order.
        fs: Sampling frequency in Hz.
        f0: Fundamental frequency in Hz.

    Returns:
        Array with shape (3, N): |X0|, |X1|, |X2|. Divide by sqrt(2) for RMS.

    Each centered DFT window spans one electrical period. Linear interpolation
    handles noninteger fs/f0. Outside the record, samples repeat the nearest
    endpoint; results within half a period of either edge are approximate.
    """
    abc = np.asarray(abc, dtype=float)
    if abc.ndim != 2 or abc.shape[0] != 3 or abc.shape[1] == 0:
        raise ValueError("abc must have shape (3, N) with N > 0.")
    if not np.isfinite(abc).all():
        raise ValueError("abc must contain only finite samples.")
    if not np.isfinite(fs) or not np.isfinite(f0) or f0 <= 0 or fs <= 2 * f0:
        raise ValueError("Require finite fs > 2*f0 > 0.")

    n_samples = abc.shape[1]
    n_cycle = max(3, int(round(fs / f0)))
    phase_steps = np.arange(n_cycle) - n_cycle // 2
    offsets = phase_steps * fs / (f0 * n_cycle)
    reference = np.exp(-2j * np.pi * phase_steps / n_cycle)
    # Preserve the notebook calculation while bounding the temporary DFT matrix.
    phasors = np.empty((3, n_samples), dtype=complex)
    sample_index = np.arange(n_samples)
    for start in range(0, n_samples, 256):
        stop = min(n_samples, start + 256)
        positions = np.arange(start, stop)[:, None] + offsets
        for channel, phase in enumerate(abc):
            phasors[channel, start:stop] = (2.0 / n_cycle) * (
                np.interp(positions, sample_index, phase) @ reference
            )
    a = np.exp(2j * np.pi / 3)
    fortescue = (
        np.array(
            [
                [1, 1, 1],
                [1, a, a**2],
                [1, a**2, a],
            ]
        )
        / 3
    )
    return np.abs(fortescue @ phasors)


def classify_fault_type(I1_rms, I2_rms, I0_rms):
    """Fault type classification"""
    ratio_21 = I2_rms / (I1_rms + 1e-10)
    ratio_01 = I0_rms / (I1_rms + 1e-10)

    if ratio_21 < 0.15 and ratio_01 < 0.15:
        fault_type = "3ph"
        one_hot = np.array([1, 0, 0, 0])
    elif ratio_21 > 0.7 and ratio_01 < 0.25:
        fault_type = "2ph"
        one_hot = np.array([0, 1, 0, 0])
    elif ratio_21 > 0.7 and ratio_01 > 0.7:
        fault_type = "1ph-G"
        one_hot = np.array([0, 0, 0, 1])
    elif 0.25 < ratio_21 < 0.7 and 0.25 < ratio_01 < 0.7:
        fault_type = "2ph-G"
        one_hot = np.array([0, 0, 1, 0])
    else:
        fault_type = "undefined"
        one_hot = np.array([0.25, 0.25, 0.25, 0.25])

    return fault_type, one_hot, (ratio_21, ratio_01)


def fit_fundamental_phasors(samples: np.ndarray, fs: float, f0: float) -> np.ndarray:
    """Fit fundamental RMS phasors to (channels, time) signals with a DC offset.

    A waveform sqrt(2)*abs(P)*cos(2*pi*f0*t + angle(P)) returns phasor P.
    All channels must share the same sample times and reference direction.
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3 or not np.isfinite(values).all():
        raise ValueError("Phasor inputs must be finite (channels, time) arrays.")
    if not np.isfinite([fs, f0]).all() or not 0 < f0 < fs / 2:
        raise ValueError("Require finite sampling/grid frequencies with 0 < f0 < fs/2.")
    angle = 2 * np.pi * f0 * np.arange(values.shape[1]) / fs
    basis = np.column_stack([np.cos(angle), np.sin(angle), np.ones_like(angle)])
    coefficients, _, rank, _ = np.linalg.lstsq(basis, values.T, rcond=None)
    if rank != 3:
        raise ValueError("The selected samples cannot resolve a fundamental phasor.")
    return (coefficients[0] - 1j * coefficients[1]) / np.sqrt(2)


LOW_RATE_SAMPLES_PER_CYCLE = 4
PEAK_CURRENT_FRACTION = 0.9
# Apparent distances at or below this are not positive: resistive V/I rounds either side of zero.
MIN_DISTANCE_KM = 1e-6


def loop_quantities(i_phase, u_phase, loop, k0=0j):
    """Loop current and voltage from (3, ...) phase phasors; ground loops add k0*(IA+IB+IC)."""
    if loop.endswith("G"):
        phase = "ABC".index(loop[0])
        return i_phase[phase] + k0 * i_phase.sum(axis=0), u_phase[phase]
    if loop == "POS":
        a = np.exp(2j * np.pi / 3)
        positive_sequence = np.array([1, a, a**2]) / 3
        return positive_sequence @ i_phase, positive_sequence @ u_phase
    left, right = "ABC".index(loop[0]), "ABC".index(loop[1])
    return i_phase[left] - i_phase[right], u_phase[left] - u_phase[right]


def peak_sample_pair_loops(currents_ka, voltages_kv, loop, min_current_ka, k0=0j):
    """Loop phasors at the fault-current peak of a 4 samples/cycle relay report.

    Relay filtered event reports are fundamental-filtered, so consecutive samples
    lie 90 degrees apart and x[k] + j*x[k-1] is the phasor at k. Samples within
    PEAK_CURRENT_FRACTION of the largest loop current are returned. This resolves
    faults cleared within a cycle or two, which whole-cycle fits smear into the
    current decay and breaker opening, and needs no inception sample.
    """
    i = currents_ka[:, 1:] + 1j * currents_ka[:, :-1]
    u = voltages_kv[:, 1:] + 1j * voltages_kv[:, :-1]
    i_loop, u_loop = loop_quantities(i, u, loop, k0)
    magnitude = np.abs(i_loop)
    if not len(magnitude) or magnitude.max() < min_current_ka:
        raise ValueError("No sample carries the minimum loop current")
    peak = np.flatnonzero(magnitude >= max(PEAK_CURRENT_FRACTION * magnitude.max(), min_current_ka))
    return [dict(sample=int(k) + 1, current=i_loop[k], voltage=u_loop[k]) for k in peak]


def _line(line_params, fault_loop):
    r1 = float(line_params["r1_ohm_km"])
    x1 = float(line_params["x1_ohm_km"])
    line_length = float(line_params["L_km"])
    if not np.isfinite([r1, x1, line_length]).all() or r1 < 0 or x1 <= 0 or line_length <= 0:
        raise ValueError("Require finite r1 >= 0, x1 > 0, and line length > 0.")
    z1 = complex(r1, x1)
    loop = fault_loop.upper()
    if loop not in {"AG", "BG", "CG", "AB", "BC", "CA", "POS"}:
        raise ValueError("Choose AG, BG, CG, AB, BC, CA, or POS.")
    ground_loop = loop.endswith("G")
    r0, x0 = line_params.get("r0_ohm_km"), line_params.get("x0_ohm_km")
    if ground_loop and (r0 is None) != (x0 is None):
        raise ValueError("Ground compensation requires both r0_ohm_km and x0_ohm_km.")
    compensated = ground_loop and r0 is not None and x0 is not None
    k0 = 0j
    if compensated:
        r0, x0 = float(r0), float(x0)
        if not np.isfinite([r0, x0]).all() or r0 < 0 or x0 <= 0:
            raise ValueError("Require finite zero-sequence parameters r0 >= 0 and x0 > 0.")
        k0 = (complex(r0, x0) - z1) / (3 * z1)
    return loop, x1, line_length, ground_loop, compensated, k0


def _row(number, start, stop, center_idx, fs, i_loop, u_loop, x1):
    z_apparent = u_loop / i_loop  # kV / kA = ohm; no base-value conversion.
    distance_km = float(z_apparent.imag / x1)
    if not np.isfinite(distance_km):
        raise ValueError("The apparent impedance did not produce a finite distance.")
    return {
        "cycle": number,
        "start_sample": int(start),
        "stop_sample": int(stop),
        "start_after_fault_ms": float((start - center_idx) / fs * 1000),
        "stop_after_fault_ms": float((stop - center_idx) / fs * 1000),
        "loop_current_rms_ka": float(abs(i_loop)),
        "loop_voltage_rms_kv": float(abs(u_loop)),
        "r_apparent_ohm": float(z_apparent.real),
        "x_apparent_ohm": float(z_apparent.imag),
        "distance_km": distance_km,
    }


def _summarize(
    rows, measurement, loop, line_length, ground_loop, compensated, k0, skipped, decayed, max_spread
):
    """Median estimate plus plausibility gate; implausible results keep R/X but carry no estimate."""
    distances = np.array([row["distance_km"] for row in rows])
    estimate = float(np.median(distances))
    low, high = float(distances.min()), float(distances.max())
    status, reason = "estimated", None
    if estimate <= MIN_DISTANCE_KM:
        status = "behind_relay"
        reason = "Measured loop reactance is not positive; the fault may be behind the relay."
    elif estimate > line_length:
        status = "out_of_range"
        reason = "Apparent distance exceeds the configured line length."
    elif max_spread is not None and high - low > max_spread * estimate:
        status = "inconsistent"
        reason = "Post-fault cycle estimates disagree by more than max_cycle_spread of the median."
    result = {"status": status}
    if reason:
        result["reason"] = reason
    result.update(
        {
            "method": "single-ended simple reactance",
            "measurement": measurement,
            "fault_loop": loop,
            ("estimated_distance_km" if status == "estimated" else "apparent_distance_km"): estimate,
            "distance_percent_of_line": float(100 * estimate / line_length),
            "within_line": bool(0 <= estimate <= line_length),
            "ground_compensated": bool(compensated),
            "uncompensated_ground_estimate": bool(ground_loop and not compensated),
            "k0_real": float(k0.real),
            "k0_imag": float(k0.imag),
            "r_apparent_ohm": float(np.median([row["r_apparent_ohm"] for row in rows])),
            "x_apparent_ohm": float(np.median([row["x_apparent_ohm"] for row in rows])),
            "cycle_min_km": low,
            "cycle_max_km": high,
            "cycles_used": len(rows),
            "cycles_skipped_low_current": skipped,
            "cycles_skipped_decayed": decayed,
            "cycles": rows,
        }
    )
    return result


def estimate_peak_distance(
    currents_ka: np.ndarray,
    voltages_kv: np.ndarray,
    fs: float,
    center_idx: int,
    line_params: dict,
    fault_loop: str,
    min_current_ka: float = 1e-3,
) -> dict:
    """Apparent distance from the fault-current peak of a 4 samples/cycle relay report.

    Uses the whole segment, so a late or unreliable inception sample cannot move
    the measurement past breaker opening. Rows are peak samples, not cycles.
    """
    currents = np.asarray(currents_ka, dtype=float)
    voltages = np.asarray(voltages_kv, dtype=float)
    if currents.shape != voltages.shape or currents.ndim != 2 or currents.shape[0] != 3:
        raise ValueError("Current and voltage arrays must both have shape (3, N).")
    if not np.isfinite(currents).all() or not np.isfinite(voltages).all():
        raise ValueError("Waveforms must contain only finite values.")
    if not np.isfinite(min_current_ka) or min_current_ka <= 0:
        raise ValueError("Minimum loop current must be positive.")
    loop, x1, line_length, ground_loop, compensated, k0 = _line(line_params, fault_loop)
    rows = [
        _row(number + 1, p["sample"], p["sample"] + 1, center_idx, fs, p["current"], p["voltage"], x1)
        for number, p in enumerate(peak_sample_pair_loops(currents, voltages, loop, min_current_ka, k0))
    ]
    return _summarize(rows, "peak-sample-pair", loop, line_length, ground_loop, compensated, k0, 0, 0, None)


def estimate_impedance_distance(
    currents_ka: np.ndarray,
    voltages_kv: np.ndarray,
    fs: float,
    f0: float,
    center_idx: int,
    measured_stop: int,
    line_params: dict,
    fault_loop: str,
    settle_cycles: float = 1.0,
    average_cycles: int = 3,
    min_current_ka: float = 1e-3,
    fault_current_fraction: float = 0.8,
    max_cycle_spread: float = 0.5,
) -> dict:
    """Estimate apparent distance using real post-fault cycles already in memory.

    Uses d_km = imag(U_loop / I_loop) / x1_ohm_km. For ground loops,
    I_loop = I_phase + k0*(IA+IB+IC), k0=(Z0-Z1)/(3*Z1), if Z0 is supplied.
    Missing Z0 yields an explicitly uncompensated apparent distance.
    The caller supplies the exclusive end of measured data to exclude padding.
    Cycles whose loop current fell below fault_current_fraction of the largest
    whole-cycle loop current after inception are skipped: the breaker opened.
    """
    currents = np.asarray(currents_ka, dtype=float)
    voltages = np.asarray(voltages_kv, dtype=float)
    if currents.shape != voltages.shape or currents.ndim != 2 or currents.shape[0] != 3:
        raise ValueError("Current and voltage arrays must both have shape (3, N).")
    if not np.isfinite(currents).all() or not np.isfinite(voltages).all():
        raise ValueError("Waveforms must contain only finite values.")
    if not np.isfinite([fs, f0]).all() or not 0 < f0 < fs / 2:
        raise ValueError("Require finite sampling/grid frequencies with 0 < f0 < fs/2.")
    if not isinstance(center_idx, (int, np.integer)) or not isinstance(measured_stop, (int, np.integer)):
        raise ValueError("Window bounds must be integer sample indices.")
    if not 0 <= center_idx < measured_stop <= currents.shape[1]:
        raise ValueError("Measured post-fault bounds must lie inside the phase window.")
    if not np.isfinite(settle_cycles) or settle_cycles < 0:
        raise ValueError("settle_cycles must be finite and nonnegative.")
    if isinstance(average_cycles, bool) or not isinstance(average_cycles, (int, np.integer)):
        raise ValueError("average_cycles must be a positive integer.")
    if average_cycles < 1 or not np.isfinite(min_current_ka) or min_current_ka <= 0:
        raise ValueError("Cycle count and minimum loop current must be positive.")
    if not 0 <= fault_current_fraction < 1 or not max_cycle_spread > 0:
        raise ValueError("Require 0 <= fault_current_fraction < 1 and max_cycle_spread > 0.")
    loop, x1, line_length, ground_loop, compensated, k0 = _line(line_params, fault_loop)

    cycle_samples = int(np.ceil(fs / f0))
    first = center_idx + int(np.ceil(settle_cycles * fs / f0))
    cycle_count = min(average_cycles, max(0, (measured_stop - first) // cycle_samples))
    if cycle_count == 0:
        raise ValueError("Less than one measured post-fault cycle remains after settling.")

    def cycle_loop(start):
        phasors = fit_fundamental_phasors(
            np.vstack(
                [currents[:, start : start + cycle_samples], voltages[:, start : start + cycle_samples]]
            ),
            fs,
            f0,
        )
        return loop_quantities(phasors[:3], phasors[3:], loop, k0)

    selected = [cycle_loop(first + cycle * cycle_samples) for cycle in range(cycle_count)]
    peak = max(abs(i_loop) for i_loop, _ in selected)
    for start in range(center_idx, measured_stop - cycle_samples + 1, cycle_samples):
        peak = max(peak, abs(cycle_loop(start)[0]))
    rows = []
    skipped = decayed = 0
    for cycle, (i_loop, u_loop) in enumerate(selected):
        if abs(i_loop) < min_current_ka:
            skipped += 1
            continue
        if abs(i_loop) < fault_current_fraction * peak:
            decayed += 1
            continue
        start = first + cycle * cycle_samples
        rows.append(_row(cycle + 1, start, start + cycle_samples, center_idx, fs, i_loop, u_loop, x1))
    if not rows:
        if decayed:
            raise ValueError(
                "Fault current decayed below fault_current_fraction of its peak in every selected cycle "
                "(breaker opened); reduce settle_cycles."
            )
        raise ValueError("Every selected cycle has insufficient loop current for division.")
    return _summarize(
        rows, "cycle-fit", loop, line_length, ground_loop, compensated, k0, skipped, decayed, max_cycle_spread
    )
