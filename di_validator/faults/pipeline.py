"""Offline reproduction of the imported fault-distance notebook pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from .. import store
from ..schemas import Model
from .numerics import (
    LOW_RATE_SAMPLES_PER_CYCLE,
    classify_fault_type,
    detect_fault_inception,
    estimate_impedance_distance,
    estimate_peak_distance,
    extract_window,
    remove_dc_period,
    sliding_fortescue_magnitudes,
)

VERSION = "notebook-reactance-2"
PHASES = ("IA", "IB", "IC", "UA", "UB", "UC")
TENSOR_CHANNELS = [*PHASES, "I1", "I2", "I0", "U1", "U2", "U0"]


class FaultConfig(Model):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    channel_map: dict[str, str] = Field(default_factory=dict)
    start: float = Field(default=0, ge=0)
    end: float | None = Field(default=None, gt=0)
    manual_onset: float | None = Field(default=None, ge=0)
    eta_i: float = Field(default=0.5, gt=0)
    eta_u: float = Field(default=0.85, gt=0, lt=1)
    pre_seconds: float = Field(default=0.05, gt=0, le=2)
    post_seconds: float = Field(default=0.15, gt=0, le=2)
    fault_loop: Literal["AUTO", "AG", "BG", "CG", "AB", "BC", "CA", "POS"] = "AUTO"
    line_length_km: float = Field(default=50, gt=0)
    nominal_voltage_kv: float = Field(default=110, gt=0)
    base_power_mva: float = Field(default=100, gt=0)
    r1_ohm_km: float = Field(default=0.1, ge=0)
    x1_ohm_km: float = Field(default=0.4, gt=0)
    r0_ohm_km: float | None = Field(default=None, ge=0)
    x0_ohm_km: float | None = Field(default=None, gt=0)
    line_parameters_verified: bool = False
    settle_cycles: float = Field(default=1, ge=0, le=20)
    average_cycles: int = Field(default=3, ge=1, le=50)
    min_current_ka: float = Field(default=0.001, gt=0)
    fault_current_fraction: float = Field(default=0.8, ge=0, lt=1)
    max_cycle_spread: float = Field(default=0.5, gt=0)
    known_distance_km: float | None = Field(default=None, ge=0)
    dataset_checksum: str | None = None
    pipeline_checksum: str | None = None

    @model_validator(mode="after")
    def compatible(self):
        if (self.r0_ohm_km is None) != (self.x0_ohm_km is None):
            raise ValueError("Ground compensation requires both r0 and x0, or neither")
        if self.end is not None and self.end <= self.start:
            raise ValueError("Analysis end must follow start")
        if self.known_distance_km is not None and self.known_distance_km > self.line_length_km:
            raise ValueError("Known distance must lie within the configured line")
        return self

    def line_params(self):
        return dict(
            L_km=self.line_length_km,
            Unom_kv=self.nominal_voltage_kv,
            r1_ohm_km=self.r1_ohm_km,
            x1_ohm_km=self.x1_ohm_km,
            r0_ohm_km=self.r0_ohm_km,
            x0_ohm_km=self.x0_ohm_km,
        )


def code_checksum():
    return store.digest({p.name: store.checksum(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def validate_dataset(dataset, parameters):
    cfg = FaultConfig.model_validate(parameters)
    fs, f0 = dataset.get("sample_rate", 0), dataset.get("config", {}).get("nominal_frequency", 0)
    if dataset["format"] != "recording" or not 0 < f0 <= fs / 4:
        raise ValueError(
            "Fault analysis requires waveforms and a declared grid frequency with at least four samples per cycle"
        )
    end = min(cfg.end if cfg.end is not None else dataset["duration_seconds"], dataset["duration_seconds"])
    if end <= cfg.start or (end - cfg.start) * fs > 200000:
        raise ValueError("Select a nonempty analysis window of at most 200,000 native samples")
    if (cfg.pre_seconds + cfg.post_seconds) * fs > 200000 or fs / f0 > 10000:
        raise ValueError("Requested feature window or electrical period is too large")
    available = {c["name"]: c for c in dataset["channels"]}
    mapping = dict(cfg.channel_map)
    if not mapping:
        for phase in PHASES:
            kind = "current" if phase[0] == "I" else "voltage"
            matches = [c["name"] for c in available.values() if c["kind"] == kind and c["phase"] == phase[1]]
            if len(matches) != 1:
                raise ValueError(f"Select exactly one {phase} waveform channel")
            mapping[phase] = matches[0]
    if set(mapping) != set(PHASES) or len(set(mapping.values())) != 6:
        raise ValueError("Map six distinct waveform channels in IA, IB, IC, UA, UB, UC order")
    for phase, name in mapping.items():
        channel = available.get(name, {})
        expected = ("current", "A") if phase[0] == "I" else ("voltage", "V")
        if (channel.get("kind"), channel.get("unit")) != expected:
            raise ValueError(f"{phase} requires a {expected[0]} waveform in {expected[1]}")
    for channel in dataset.get("comtrade", {}).get("original_channels", []):
        if channel["name"].strip() in mapping.values() and float(channel.get("skew", 0)) != 0:
            raise ValueError(
                "Fault distance requires synchronized waveforms; nonzero COMTRADE channel skew is unsupported"
            )
    if cfg.manual_onset is not None and not cfg.start <= cfg.manual_onset <= end:
        raise ValueError("Manual onset must lie inside the selected analysis window")
    checksum = code_checksum()
    if cfg.dataset_checksum and cfg.dataset_checksum != dataset["source_checksum"]:
        raise ValueError("Dataset changed after this configuration was frozen")
    if cfg.pipeline_checksum and cfg.pipeline_checksum != checksum:
        raise ValueError("Fault pipeline changed after queueing; create a new run")
    return cfg.model_copy(
        update=dict(
            channel_map=mapping,
            end=end,
            dataset_checksum=dataset["source_checksum"],
            pipeline_checksum=checksum,
        )
    )


def analyze_segment(frame, dataset, cfg, segment_number=0):
    """Process one finite, regularly spaced segment; no sample interpolation."""
    fs = dataset["sample_rate"]
    f0 = dataset["config"]["nominal_frequency"]
    period = int(round(fs / f0))
    info = dict(
        segment=segment_number,
        start=float(frame.offset.iloc[0]),
        end=float(frame.offset.iloc[-1]),
        measured_samples=len(frame),
        status="no_fault",
        reason="No fault inception met the combined RMS criterion",
    )
    if len(frame) < 2 * period:
        return {
            **info,
            "status": "insufficient_data",
            "reason": "Need at least two contiguous electrical cycles",
        }, None
    values = frame[[cfg.channel_map[p] for p in PHASES]].to_numpy(dtype=float).T / 1000
    centered = values - values[:, :period].mean(axis=1, keepdims=True)
    filtered = remove_dc_period(centered, fs, f0)
    detected, i_ratio, u_ratio = detect_fault_inception(
        *filtered, fs, f0, cfg.base_power_mva / (np.sqrt(3) * cfg.nominal_voltage_kv), cfg.eta_i, cfg.eta_u
    )
    manual = cfg.manual_onset is not None and info["start"] <= cfg.manual_onset <= info["end"]
    if cfg.manual_onset is not None and not manual:
        return {
            **info,
            "status": "outside_manual_selection",
            "reason": "Manual onset belongs to another segment",
        }, None
    if detected is None and not manual:
        if dataset.get("wavewin"):
            from .network_model import network_location

            info["network_location"] = network_location(
                dataset, values[:3], values[3:], fs, f0, 0, len(frame), cfg.fault_loop, cfg
            )
        return info, None
    center = int(np.argmin(np.abs(frame.offset.to_numpy() - cfg.manual_onset))) if manual else int(detected)
    npre, npost = round(cfg.pre_seconds * fs), round(cfg.post_seconds * fs)
    if min(npre, npost) < 1:
        raise ValueError("Pre/post windows must each contain at least one sample")
    windows = np.stack([extract_window(row, center, npre, npost) for row in filtered])
    sequences = np.vstack(
        [
            sliding_fortescue_magnitudes(filtered[:3], fs, f0),
            sliding_fortescue_magnitudes(filtered[3:], fs, f0),
        ]
    )
    seq_windows = np.stack([extract_window(row, center, npre, npost) for row in sequences])
    stop = min(npre + npost, npre + len(frame) - center)
    i0, i1, i2 = np.sqrt(np.mean(seq_windows[:3, npre:stop] ** 2, axis=1) / 2)
    kind, _, ratios = classify_fault_type(i1, i2, i0)
    loop = cfg.fault_loop
    phase_rms = np.sqrt(np.mean(windows[:3, npre:stop] ** 2, axis=1))
    if loop == "AUTO":
        if kind == "1ph-G":
            loop = "ABC"[int(np.argmax(phase_rms))] + "G"
        elif kind in {"2ph", "2ph-G"}:
            pair = frozenset(np.argsort(phase_rms)[-2:].tolist())
            loop = {frozenset([0, 1]): "AB", frozenset([1, 2]): "BC", frozenset([0, 2]): "CA"}[pair]
        elif kind == "3ph":
            loop = "POS"
    try:
        if loop == "AUTO":
            raise ValueError("Fault type is uncertain; select the fault loop explicitly")
        if period == LOW_RATE_SAMPLES_PER_CYCLE:
            # Relay reports at 4 samples/cycle: onset detection lags the fault, so
            # measure at the fault-current peak of the whole raw segment instead.
            distance = estimate_peak_distance(
                values[:3], values[3:], fs, center, cfg.line_params(), loop, cfg.min_current_ka
            )
        else:
            distance = estimate_impedance_distance(
                windows[:3],
                windows[3:],
                fs,
                f0,
                npre,
                stop,
                cfg.line_params(),
                loop,
                cfg.settle_cycles,
                cfg.average_cycles,
                cfg.min_current_ka,
                cfg.fault_current_fraction,
                cfg.max_cycle_spread,
            )
        if distance["status"] == "estimated" and cfg.known_distance_km is not None:
            distance["absolute_error_km"] = abs(distance["estimated_distance_km"] - cfg.known_distance_km)
    except ValueError as error:
        distance = dict(status="unavailable", reason=str(error))
    network = {}
    if dataset.get("wavewin"):
        from .network_model import network_location

        network = dict(
            network_location=network_location(dataset, values[:3], values[3:], fs, f0, center, len(frame), loop, cfg)
        )
    base_i = cfg.base_power_mva / (np.sqrt(3) * cfg.nominal_voltage_kv)
    base_u = cfg.nominal_voltage_kv / np.sqrt(3)
    tensor = np.vstack(
        [
            windows[:3] / base_i,
            windows[3:] / base_u,
            seq_windows[[1, 2, 0]] / base_i,
            seq_windows[[4, 5, 3]] / base_u,
        ]
    ).astype(np.float32)
    info.update(
        status="analyzed",
        reason=None,
        onset=float(frame.offset.iloc[center]),
        inception_source="manual" if manual else "detected",
        detected_sample=int(detected) if detected is not None else None,
        center_sample=center,
        fault_type_heuristic=kind,
        sequence_ratios=dict(negative_positive=float(ratios[0]), zero_positive=float(ratios[1])),
        loop_selection="heuristic" if cfg.fault_loop == "AUTO" else "manual",
        distance=distance,
        **network,
        padding_left_samples=max(0, npre - center),
        padding_right_samples=max(0, center + npost - len(frame)),
        pre_window_samples=npre,
        post_window_samples=npost,
        tensor_channels=TENSOR_CHANNELS,
        emitted_at=info["end"] + 1 / fs,
        decision_mode="offline; full segment available",
        tensor_shape=list(tensor.shape),
    )
    return info, dict(
        X=tensor,
        y=np.nan if cfg.known_distance_km is None else cfg.known_distance_km / cfg.line_length_km,
        time_seconds=frame.offset.to_numpy(),
        filtered_ka_kv=filtered,
        sequence_peak_ka_kv=sequences,
        current_ratio=i_ratio,
        voltage_ratio=u_ratio,
        window_time_ms=np.arange(-npre, npost) / fs * 1000,
    )
