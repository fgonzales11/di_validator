"""Locate Wavewin faults on the recorded feeder's multiconductor network model.

A single-ended reactance measurement on a branched feeder with mixed conductors
does not map to one distance. Each primary bus instead carries the cumulative
loop impedance from the substation source along its path, summed from the
model's per-phase impedance matrices, and every path crossing the measured loop
reactance is a candidate location.
"""

from __future__ import annotations

import io
import math
import pickle
import re
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import store
from .numerics import LOW_RATE_SAMPLES_PER_CYCLE, fit_fundamental_phasors, loop_quantities, peak_sample_pair_loops

FEEDER_ROOT = store.ROOT / "teco" / "pkls" / "teco"
PHASES = {1: "A", 2: "B", 3: "C"}
LOOPS = ("AG", "BG", "CG", "AB", "BC", "CA", "POS")
MAX_CANDIDATES = 25
KM_PER_MILE = 1.609344


class _Net(dict):
    """Stands in for pandapower.auxiliary.pandapowerNet; importing pandapower takes minutes here."""

    def __setstate__(self, state):
        for part in state if isinstance(state, tuple) else (state,):
            if isinstance(part, dict):
                self.update(part)


class _ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("pandapower.auxiliary", "pandapowerNet"):
            return _Net
        if module.split(".")[0] in {"numpy", "pandas"} or (module, name) == ("builtins", "slice"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Network model references disallowed type {module}.{name}")


def feeder_id(dataset):
    fields = Path(dataset.get("source", "")).name.split(",")
    match = re.search(r"(\d+)\s*\(", fields[4]) if len(fields) > 4 else None
    return match[1] if match else None


def model_path(feeder):
    matches = sorted(FEEDER_ROOT.glob(f"*/TECO_{feeder}.pkl"))
    if len(matches) > 1:
        raise ValueError(f"Feeder {feeder} matches several network models")
    return matches[0] if matches else None


def _load(path):
    return _ModelUnpickler(io.BytesIO(Path(path).read_bytes())).load()


def _loop_impedances(z, phases):
    index = {p: i for i, p in enumerate(phases)}
    loops = {}
    for p in phases:
        loops[PHASES[p] + "G"] = z[index[p], index[p]]
    for a, b in [(1, 2), (2, 3), (3, 1)]:
        if a in index and b in index:
            i, j = index[a], index[b]
            loops[PHASES[a] + PHASES[b]] = (z[i, i] + z[j, j] - z[i, j] - z[j, i]) / 2
    if len(index) == 3:
        loops["POS"] = np.trace(z) / 3 - (z.sum() - np.trace(z)) / 6
    return loops


def _profile(path):
    net = _load(path)
    matrices = net["std_types"]["matrix"]
    lines, switches = net["line"], net["switch"]
    open_conductors = {
        (int(e), int(p))
        for e, p, et, closed in zip(switches["element"], switches["phase"], switches["et"], switches["closed"])
        if et == "l" and not closed
    }
    edges = defaultdict(list)
    for line_id, rows in lines.groupby(level=0, sort=False):
        first = rows.iloc[0]
        if not bool(first["in_service"]):
            continue
        std = matrices[first["std_type"]]
        size = len(std["r_1_ohm_per_km"])
        phases = [int(p) for p in rows["from_phase"]]
        if len(phases) != size:
            raise ValueError(f"Line {line_id}: conductor count does not match {first['std_type']}")
        z = np.array(
            [[complex(r, x) for r, x in zip(std[f"r_{k}_ohm_per_km"], std[f"x_{k}_ohm_per_km"])] for k in range(1, size + 1)]
        ) * float(first["length_km"])
        keep = [i for i, p in enumerate(phases) if (int(line_id), p) not in open_conductors]
        if not keep:
            continue
        kept = [phases[i] for i in keep]
        edge = dict(
            line=int(line_id),
            section=str(first.get("section_id", "")),
            length_km=float(first["length_km"]),
            phases=kept,
            loops=_loop_impedances(z[np.ix_(keep, keep)], kept),
        )
        a, b = int(first["from_bus"]), int(first["to_bus"])
        edges[a].append((b, edge))
        edges[b].append((a, edge))
    source = int(net["ext_grid_sequence"]["bus"].iloc[0])
    geo = net["bus_geodata"].groupby(level=0).first()
    buses = {source: dict(parent=None, edge=None, distance_km=0.0, loops={k: 0j for k in LOOPS})}
    queue = deque([source])
    # Breadth-first order keeps the fewest-segment path if the model contains a closed loop.
    while queue:
        u = queue.popleft()
        for v, edge in edges[u]:
            if v in buses:
                continue
            parent = buses[u]["loops"]
            buses[v] = dict(
                parent=u,
                edge=edge,
                distance_km=buses[u]["distance_km"] + edge["length_km"],
                loops={k: parent[k] + edge["loops"][k] if k in edge["loops"] and parent[k] is not None else None for k in LOOPS},
            )
            queue.append(v)
    for bus, record in buses.items():
        record["lon"], record["lat"] = (
            (float(geo.at[bus, "longitude"]), float(geo.at[bus, "latitude"])) if bus in geo.index else (None, None)
        )
    source_kv = float(net["bus"].loc[source]["vn_kv"].max())
    return dict(
        feeder=str(net.get("CIRCUIT_ID", Path(path).stem)),
        substation=str(net.get("SUB_ID", Path(path).parent.name)),
        nominal_kv=source_kv,
        buses=buses,
        modeled_km=float(sum(e["length_km"] for bus in buses.values() if (e := bus["edge"]))),
        estimated_impedance_share=float(lines["impedance_estimated"].mean()) if "impedance_estimated" in lines else None,
    )


@lru_cache(maxsize=8)
def _cached_profile(path, size, mtime_ns):
    return _profile(path)


def profile(path):
    stat = Path(path).stat()
    return _cached_profile(str(path), stat.st_size, stat.st_mtime_ns)


def measured_loop_impedance(currents_ka, voltages_kv, fs, f0, center, stop, loop, cfg):
    """Uncompensated loop impedance at the fault-current peak, matching the model's per-phase sums.

    Relay filtered event reports at 4 samples/cycle are fundamental-filtered, so
    consecutive samples lie 90° apart and x[k] + j·x[k-1] is the phasor at k.
    That resolves faults cleared within a cycle or two, which whole-cycle fits
    smear into the current decay and breaker opening.
    """
    if round(fs / f0) != LOW_RATE_SAMPLES_PER_CYCLE:
        cycle = int(np.ceil(fs / f0))
        first = center + int(np.ceil(cfg.settle_cycles * fs / f0))
        count = min(cfg.average_cycles, max(0, (stop - first) // cycle))
        rows = []
        for number in range(count):
            start = first + number * cycle
            phasors = fit_fundamental_phasors(
                np.vstack([currents_ka[:, start : start + cycle], voltages_kv[:, start : start + cycle]]), fs, f0
            )
            i_loop, u_loop = loop_quantities(phasors[:3], phasors[3:], loop)
            if abs(i_loop) >= cfg.min_current_ka:
                rows.append(dict(sample=start, impedance=u_loop / i_loop, current_ka=float(abs(i_loop))))
    else:
        # The fault-current peak is used directly: onset detection is unreliable at 4 samples/cycle.
        rows = [
            dict(sample=p["sample"], impedance=p["voltage"] / p["current"], current_ka=float(abs(p["current"])))
            for p in peak_sample_pair_loops(currents_ka[:, :stop], voltages_kv[:, :stop], loop, cfg.min_current_ka)
        ]
    if not rows:
        raise ValueError("No post-fault cycle carries the minimum loop current")
    z = np.array([r["impedance"] for r in rows])
    return complex(np.median(z.real), np.median(z.imag)), rows


def equivalent_line(model):
    """Uniform-line settings equivalent to the longest three-phase path of a feeder profile.

    Reads the model only. z1 is the path's positive-sequence loop impedance per km;
    the ground-loop (self) impedance Zs = (Z0 + 2*Z1)/3 per km gives Z0 = 3*Zs - 2*Z1,
    averaged over the three phases, so ground loops can be k0-compensated.
    """
    three_phase = [b for b in model["buses"].values() if b["edge"] is not None and b["loops"]["POS"] is not None]
    if not three_phase:
        raise ValueError("The network model has no three-phase path from the source")
    far = max(three_phase, key=lambda b: b["distance_km"])
    length = far["distance_km"]
    z1 = far["loops"]["POS"] / length
    z0 = sum(far["loops"][k] for k in ("AG", "BG", "CG")) / length - 2 * z1  # 3 * mean(Zs) - 2 * Z1
    return dict(
        line_length_km=length,
        nominal_voltage_kv=model["nominal_kv"],
        r1_ohm_km=z1.real,
        x1_ohm_km=z1.imag,
        r0_ohm_km=z0.real,
        x0_ohm_km=z0.imag,
    )


def locate(model, loop, impedance):
    reactance, resistance = impedance.imag, impedance.real
    if reactance <= 0:
        raise ValueError("Measured loop reactance is not positive; the fault may be behind the relay")
    buses = model["buses"]
    candidates = []
    for bus, record in buses.items():
        edge, parent = record["edge"], record["parent"]
        if edge is None:
            continue
        z_to, z_from = record["loops"][loop], buses[parent]["loops"][loop]
        if z_to is None or z_from is None or not z_from.imag <= reactance < z_to.imag:
            continue
        f = (reactance - z_from.imag) / (z_to.imag - z_from.imag)
        z_at = z_from + f * (z_to - z_from)
        start, end = buses[parent], record
        point = (
            (start["lon"] + f * (end["lon"] - start["lon"]), start["lat"] + f * (end["lat"] - start["lat"]))
            if None not in (start["lon"], end["lon"])
            else (None, None)
        )
        distance_km = start["distance_km"] + f * edge["length_km"]
        candidates.append(
            dict(
                distance_km=distance_km,
                distance_mi=distance_km / KM_PER_MILE,
                section_id=edge["section"],
                from_bus=parent,
                to_bus=bus,
                segment_fraction=f,
                phases="".join(PHASES[p] for p in edge["phases"]),
                model_resistance_ohm=z_at.real,
                fault_resistance_ohm=resistance - z_at.real,
                longitude=point[0],
                latitude=point[1],
            )
        )
    reach = max((r["loops"][loop].imag for r in buses.values() if r["loops"][loop] is not None), default=0.0)
    if not candidates:
        raise ValueError(
            f"Measured reactance {reactance:.3f} Ω exceeds every modeled {loop} path (maximum {reach:.3f} Ω)"
        )
    # Fault resistance is physically nonnegative; the least-resistance crossing is listed first.
    candidates.sort(key=lambda c: (c["fault_resistance_ohm"] < -0.05, abs(c["fault_resistance_ohm"])))
    return candidates, reach


def relay_fault_loop(dataset):
    """Loop from the relay's own fault classification (e.g. "BG T", "BCG", "ABC")."""
    phases = str(dataset.get("wavewin", {}).get("event_type") or "").split(" ")[0].upper()
    ground = phases.endswith("G")
    phases = phases.rstrip("G")
    if not phases or set(phases) - set("ABC") or len(set(phases)) != len(phases):
        return None
    if len(phases) == 3:
        return "POS"
    if len(phases) == 1:
        return phases + "G" if ground else None
    return {"AB": "AB", "BA": "AB", "BC": "BC", "CB": "BC", "CA": "CA", "AC": "CA"}[phases]


def feeder_configuration(dataset):
    """FaultConfig defaults for a Wavewin recording, plus their provenance.

    The fault loop comes from the relay's own event type, since sequence-based
    classification is unreliable at 4 samples/cycle. Line settings are the
    feeder model's equivalent line; without a model the generic defaults remain.
    """
    settings = {}
    loop = relay_fault_loop(dataset)
    if loop:
        settings["fault_loop"] = loop
    provenance = dict(fault_loop_source="relay event type" if loop else "agent heuristic")
    feeder = feeder_id(dataset)
    path = model_path(feeder) if feeder else None
    if path is None:
        provenance.update(
            line_source="generic",
            feeder=feeder,
            warning="No network model for this feeder: generic line settings, distance is not feeder-referenced",
        )
        return settings, provenance
    line = equivalent_line(profile(path))
    if not (line["r0_ohm_km"] >= 0 and line["x0_ohm_km"] > 0):
        line.pop("r0_ohm_km")
        line.pop("x0_ohm_km")
    settings.update(line)
    provenance.update(
        line_source=f"{path.name} equivalent line (longest three-phase path)",
        feeder=feeder,
        model=str(path.relative_to(store.ROOT)),
    )
    return settings, provenance


def network_location(dataset, currents_ka, voltages_kv, fs, f0, center, stop, loop, cfg):
    feeder = feeder_id(dataset)
    if not feeder:
        return dict(status="unavailable", reason="Feeder ID is not present in the Wavewin file name")
    path = model_path(feeder)
    if path is None:
        return dict(status="unavailable", feeder=feeder, reason=f"No network model found for feeder {feeder}")
    relay_loop = relay_fault_loop(dataset)
    if cfg.fault_loop != "AUTO":
        loop, loop_source = cfg.fault_loop, "manual"
    elif relay_loop:
        loop, loop_source = relay_loop, "relay event type"
    else:
        loop_source = "heuristic"
    try:
        if loop not in LOOPS:
            raise ValueError("Select a fault loop to locate the fault on the network model")
        impedance, cycles = measured_loop_impedance(currents_ka, voltages_kv, fs, f0, center, stop, loop, cfg)
        model = profile(path)
        candidates, reach = locate(model, loop, impedance)
    except ValueError as error:
        return dict(
            status="unavailable",
            feeder=feeder,
            model=str(path.relative_to(store.ROOT)),
            reason=str(error),
            **_relay_location(dataset),
        )
    return dict(
        status="located",
        feeder=model["feeder"],
        substation=model["substation"],
        model=str(path.relative_to(store.ROOT)),
        nominal_kv=model["nominal_kv"],
        loop=loop,
        loop_source=loop_source,
        measured_resistance_ohm=impedance.real,
        measured_reactance_ohm=impedance.imag,
        measurement="peak sample-pair phasors" if round(fs / f0) == LOW_RATE_SAMPLES_PER_CYCLE else "post-fault cycle fits",
        samples=[
            dict(
                sample=c["sample"],
                resistance_ohm=c["impedance"].real,
                reactance_ohm=c["impedance"].imag,
                current_ka=c["current_ka"],
            )
            for c in cycles
        ],
        candidate_count=len(candidates),
        candidates=candidates[:MAX_CANDIDATES],
        modeled_reach_ohm=reach,
        modeled_km=model["modeled_km"],
        modeled_mi=model["modeled_km"] / KM_PER_MILE,
        estimated_impedance_share=model["estimated_impedance_share"],
        **_relay_location(dataset),
        assumptions=[
            "Relay-reported location is taken as miles (relay LL setting units).",
            "Model phases 1/2/3 are taken as relay phases A/B/C.",
            "Loop impedance is uncompensated (e.g. Ua/Ia) and pre-fault load current is not removed.",
            "Sample-pair phasors assume the relay report is fundamental-filtered at 4 samples/cycle.",
            "Candidates are points where cumulative modeled reactance equals the measured reactance.",
        ],
    )


def _relay_location(dataset):
    try:
        miles = float(dataset.get("wavewin", {}).get("location"))
    except (TypeError, ValueError):
        return {}
    return dict(relay_location_mi=miles, relay_location_km=miles * KM_PER_MILE) if math.isfinite(miles) else {}
