"""Independent NumPy oracle for the experimental meter-only PV algorithm.

Input is run-length encoding of explicitly constant one-second samples. Hourly
AMI is never expanded into synthetic one-second observations by this module.
"""

from __future__ import annotations

import datetime as dt
import math
import numpy as np

DEFAULTS = dict(
    latitude=999.0,
    longitude=999.0,
    polarity=1.0,
    power_scale=1.0,
    reactive_scale=1.0,
    min_coverage=0.9,
    export_threshold_kw=0.05,
    export_fraction=0.05,
    valley_threshold=0.55,
    valley_fraction=0.35,
    valley_median=0.8,
    min_r2=0.5,
    max_nrmse=0.25,
    q_margin=0.1,
    storage=0,
    other_generation=0,
)
DAY, INTERVAL = 86400, 900


def solar(utc, latitude, longitude):
    t = dt.datetime.fromtimestamp(utc, dt.timezone.utc)
    leap = t.year % 4 == 0 and (t.year % 100 != 0 or t.year % 400 == 0)
    hour = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * math.pi / (366 if leap else 365) * (t.timetuple().tm_yday - 1 + (hour - 12) / 24)
    eq = 229.18 * (
        0.000075
        + 0.001868 * math.cos(g)
        - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g)
        - 0.040849 * math.sin(2 * g)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(g)
        + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g)
        + 0.00148 * math.sin(3 * g)
    )
    sh = (hour + (eq + 4 * longitude) / 60 + 48) % 24
    lat, ha = math.radians(latitude), math.radians(sh * 15 - 180)
    elevation = math.degrees(
        math.asin(
            np.clip(math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha), -1, 1)
        )
    )
    return elevation, sh


def empty_model():
    return dict(
        valid=False,
        reason="model_rank",
        b0=0.0,
        b1=0.0,
        q_min=0.0,
        q_max=0.0,
        residual95=0.0,
        r2=0.0,
        nrmse=0.0,
    )


def robust(rows):
    x = np.asarray([[1.0, r["reactive_kvar"]] for r in rows])
    y = np.asarray([r["net_kw"] for r in rows])
    if len(rows) < 20:
        return None
    w = np.ones(len(y))
    for _ in range(11):
        mq = np.average(x[:, 1], weights=w)
        norm = np.linalg.norm(np.sqrt(w) * (x[:, 1] - mq))
        if norm <= 1e-10 * max(1.0, abs(mq)) * np.sqrt(w.sum()):
            return None
        beta = np.linalg.lstsq(x * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        residual = y - x @ beta
        scale = max(0.01, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        w = np.minimum(1.0, 1.345 * scale / np.maximum(1e-30, np.abs(residual)))
    return beta


def fit(train, test, all_rows, cfg):
    m = empty_model()
    beta = robust(train)
    if beta is None or len(test) < 20:
        return m
    y = np.asarray([r["net_kw"] for r in test])
    q = np.asarray([r["reactive_kvar"] for r in test])
    residual = y - np.maximum(0, beta[0] + beta[1] * q)
    sse, sst, squares = sum(residual**2), sum((y - y.mean()) ** 2), sum(y**2)
    m.update(
        reason="model_validation",
        r2=float(1 - sse / sst) if sst > 1e-12 else 0.0,
        nrmse=float(np.sqrt(sse / squares)) if squares > 1e-12 else 1.0,
        residual95=float(np.quantile(np.abs(residual), 0.95)),
    )
    if below(m["r2"], cfg["min_r2"]) or below(cfg["max_nrmse"], m["nrmse"]):
        return m
    beta = robust(all_rows)
    if beta is None:
        m["reason"] = "model_rank"
        return m
    m.update(
        valid=True,
        reason="available",
        b0=float(beta[0]),
        b1=float(beta[1]),
        q_min=min(r["reactive_kvar"] for r in all_rows),
        q_max=max(r["reactive_kvar"] for r in all_rows),
    )
    return m


def detection():
    return dict(
        state="learning",
        candidate="learning",
        streak=0,
        valid_days=0,
        export_days=0,
        profile_days=0,
        export_fraction=0.0,
        valley_fraction=0.0,
        valley_median=0.0,
        night_export=False,
        model=empty_model(),
    )


def below(value, threshold):
    return value < threshold - 1e-12 * max(1.0, abs(value), abs(threshold))


def evaluate(history, previous, cfg):
    d = detection()
    d.update({k: previous[k] for k in ["state", "candidate", "streak"]})
    if cfg["latitude"] == 999:
        return detection()
    days = {}
    for r in history:
        days.setdefault(r["start"] // DAY, []).append(r)
    selected = []
    for rows in days.values():
        if sum(r["p_seconds"] for r in rows) < DAY * cfg["min_coverage"]:
            continue
        eligible = [r for r in rows if r["p_seconds"] >= INTERVAL * cfg["min_coverage"]]
        elevations = [solar(r["start"] + 450, cfg["latitude"], cfg["longitude"])[0] for r in eligible]
        if sum(s > 10 for s in elevations) >= 4 and sum(s < -6 for s in elevations) >= 4:
            selected.append(eligible)
    selected = selected[-30:]
    d["valid_days"] = len(selected)
    if len(selected) < 30:
        d.update(state="learning", candidate="learning", streak=0)
        return d
    shoulder_min = max(0.1 * np.median([abs(r["net_kw"]) for rows in selected for r in rows]), 1e-6)
    valleys, train, test, all_rows = [], [], [], []
    light = exports = night_days = 0
    for index, rows in enumerate(selected):
        morning, midday, evening = [], [], []
        day_export = night_export = False
        for r in rows:
            elevation, hour = solar(r["start"] + 450, cfg["latitude"], cfg["longitude"])
            if elevation > 10:
                light += 1
                if below(r["net_kw"], -cfg["export_threshold_kw"]):
                    exports += 1
                    day_export = True
            if elevation < -6:
                night_export |= below(r["net_kw"], -cfg["export_threshold_kw"])
                if r["q_seconds"] >= INTERVAL * cfg["min_coverage"]:
                    all_rows.append(r)
                    (train if index < 25 else test).append(r)
            if 6 <= hour < 10:
                morning.append(r["net_kw"])
            if 10 <= hour < 16:
                midday.append(r["net_kw"])
            if 17 <= hour < 21:
                evening.append(r["net_kw"])
        d["export_days"] += int(day_export)
        night_days += int(night_export)
        if len(morning) >= 12 and len(midday) >= 20 and len(evening) >= 12:
            shoulder = min(np.mean(morning), np.mean(evening))
            if shoulder > shoulder_min:
                valleys.append(float(np.mean(midday) / shoulder))
    d.update(
        night_export=night_days >= 3,
        export_fraction=exports / light if light else 0.0,
        profile_days=len(valleys),
        valley_median=float(np.median(valleys)) if valleys else 0.0,
        valley_fraction=sum(below(v, cfg["valley_threshold"]) for v in valleys) / len(valleys)
        if valleys
        else 0.0,
        model=fit(train, test, all_rows, cfg),
    )
    candidate = "no_evidence"
    if cfg["storage"] or cfg["other_generation"] or d["night_export"]:
        candidate = "ambiguous"
    elif (d["export_fraction"] >= cfg["export_fraction"] and d["export_days"] >= 5) or (
        len(valleys) >= 15
        and d["valley_fraction"] >= cfg["valley_fraction"]
        and below(d["valley_median"], cfg["valley_median"])
    ):
        candidate = "likely_pv"
    d["streak"] = min(3, d["streak"] + 1) if candidate == d["candidate"] else 1
    d["candidate"] = candidate
    if d["streak"] >= 3:
        d["state"] = candidate
    return d


def estimate(row, d, cfg):
    out = {k: v for k, v in row.items() if k != "raw_flags"}
    out.update(
        state=d["state"],
        reason="warmup",
        flags=16 | 32 | 512 | row["raw_flags"],
        available=False,
        generation_kw=None,
        low_kw=None,
        high_kw=None,
        generation_kwh=None,
        estimated_seconds=0,
        state_changed=False,
    )
    if row["p_seconds"] < 900:
        out["flags"] |= 4
    if row["q_seconds"] < 900 * cfg["min_coverage"]:
        out["flags"] |= 8
    reason = None
    if cfg["storage"] or cfg["other_generation"]:
        reason = "other_der"
        out["flags"] |= 64
    elif cfg["latitude"] == 999:
        reason = "coordinates"
        out["flags"] |= 128
    elif d["night_export"]:
        reason = "night_export"
        out["flags"] |= 64
    elif d["valid_days"] < 30 or d["state"] == "learning":
        reason = "warmup"
        out["flags"] |= 256
    elif d["state"] != "likely_pv":
        reason = "no_pv_evidence"
    elif row["p_seconds"] < 900 * cfg["min_coverage"]:
        reason = "coverage"
    elif row["q_seconds"] < 900 * cfg["min_coverage"]:
        reason = "reactive_missing"
    elif not d["model"]["valid"]:
        reason = d["model"]["reason"]
    if reason:
        out["reason"] = reason
        return out
    elevation, _ = solar(row["start"] + 450, cfg["latitude"], cfg["longitude"])
    if -6 <= elevation <= 10:
        out["reason"] = "twilight"
        return out
    gen = low = high = 0.0
    if elevation > 10:
        m = d["model"]
        margin = (m["q_max"] - m["q_min"]) * cfg["q_margin"]
        q = row["reactive_kvar"]
        if below(q, m["q_min"] - margin) or below(m["q_max"] + margin, q):
            out["reason"] = "q_range"
            return out
        raw = max(0.0, m["b0"] + m["b1"] * q) - row["net_kw"]
        floor = row["export_kwh"] * 3600 / row["p_seconds"]
        gen = max(floor, 0.0, raw)
        low = max(floor, 0.0, raw - m["residual95"])
        high = max(gen, raw + m["residual95"])
    out.update(
        available=True,
        reason="available",
        flags=out["flags"] & ~512,
        generation_kw=gen,
        low_kw=low,
        high_kw=high,
        generation_kwh=gen * row["p_seconds"] / 3600,
        estimated_seconds=row["p_seconds"],
    )
    return out


def analyze(runs, cfg):
    buckets = {}
    last = -1
    unassigned_import = unassigned_export = 0.0
    for start, seconds, watts, vars_, pv, qv, tv in runs:
        p, q = watts * cfg["polarity"] * cfg["power_scale"] / 1000, vars_ * cfg["reactive_scale"] / 1000
        if not tv or start < 946684800 or start > 2145916799:
            if pv:
                unassigned_import += max(0.0, p) * seconds / 3600
                unassigned_export += max(0.0, -p) * seconds / 3600
            continue
        if start <= last and buckets:
            buckets[last // 900 * 900]["raw_flags"] |= 2
        end = start + seconds
        start = max(start, last + 1)
        while start < end:
            b = start // 900 * 900
            r = buckets.setdefault(
                b,
                dict(
                    start=b,
                    p_seconds=0,
                    q_seconds=0,
                    ps=0.0,
                    qs=0.0,
                    import_kwh=0.0,
                    export_kwh=0.0,
                    raw_flags=0,
                ),
            )
            n = min(end, b + 900) - start
            if last >= 0 and start > last + 1:
                r["raw_flags"] |= 1
            if pv:
                r["p_seconds"] += n
                r["ps"] += p * n
                r["import_kwh"] += max(0.0, p) * n / 3600
                r["export_kwh"] += max(0.0, -p) * n / 3600
            if qv:
                r["q_seconds"] += n
                r["qs"] += q * n
            start += n
            last = start - 1
    d = detection()
    results, history = [], []
    if buckets:
        for b in range(min(buckets), (last + 1) // 900 * 900, 900):
            r = buckets.get(
                b,
                dict(
                    start=b,
                    p_seconds=0,
                    q_seconds=0,
                    ps=0.0,
                    qs=0.0,
                    import_kwh=0.0,
                    export_kwh=0.0,
                    raw_flags=0,
                ),
            ).copy()
            r["net_kw"] = r.pop("ps") / r["p_seconds"] if r["p_seconds"] else r.pop("ps")
            r["reactive_kvar"] = r.pop("qs") / r["q_seconds"] if r["q_seconds"] else r.pop("qs")
            result = estimate(r, d, cfg)
            history.append(r)
            history = [h for h in history if h["start"] >= b + 900 - 60 * DAY]
            if (b + 900) % DAY == 0:
                previous = d["state"]
                d = evaluate(history, d, cfg)
                result["state_changed"] = previous != d["state"]
            results.append(result)
    return dict(
        intervals=results,
        detection=d,
        unassigned_import_kwh=unassigned_import,
        unassigned_export_kwh=unassigned_export,
    )
