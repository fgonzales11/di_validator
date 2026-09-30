"""Versioned PV DI outcome decoder. Flag 16 = PV-only/unity-inverter-PF assumption;
flag 32 = uncertainty is not calibrated for daytime generation accuracy.
"""

import base64
import json
import math
import struct

STATES = ["learning", "likely_pv", "no_evidence", "ambiguous"]
REASONS = [
    "available",
    "warmup",
    "coordinates",
    "coverage",
    "no_pv_evidence",
    "other_der",
    "night_export",
    "model_rank",
    "model_validation",
    "q_range",
    "twilight",
    "reactive_missing",
    "invalid_time",
    "model_invalid",
]


def decode(payload):
    if isinstance(payload, bytes):
        payload = payload.decode("ascii")
    if payload.startswith("PVD1:"):
        return {
            "version": 1,
            "kind": "daily",
            "assumptions": "No material battery/generator contribution; negligible inverter reactive contribution",
            "uncertainty": "Uncalibrated for daytime PV accuracy",
            **json.loads(payload[5:]),
        }
    if payload.startswith("98#PV#1#"):
        _, _, version, utc, state, config, model, note = payload.split("#")
        return dict(
            version=int(version),
            kind="transition",
            utc=int(utc),
            state=state,
            config=config,
            model=model,
            note=note,
            id=f"pv1:{config}:{utc}:{state}",
        )
    if not payload.startswith("PVH1:"):
        raise ValueError("Unsupported PV payload version")
    raw = base64.b64decode(payload[5:], validate=True)
    start, config, model, count = struct.unpack_from("<qQQB", raw)
    row_format = "<B7fHHIBB"
    size = struct.calcsize(row_format)
    if not 1 <= count <= 4 or len(raw) != 25 + size * count:
        raise ValueError("Invalid PV batch length")
    records = []
    for i in range(count):
        offset, *values = struct.unpack_from(row_format, raw, 25 + i * size)
        net, im, ex, pv, low, high, energy, observed, estimated, flags, state, reason = values
        if offset > 3 or observed > 900 or estimated > 900:
            raise ValueError("Invalid PV interval")

        def optional(x):
            return x if math.isfinite(x) else None

        records.append(
            dict(
                id=f"pv1:{config:016x}:{start + 900 * offset}",
                start=start + 900 * offset,
                net_kw=optional(net),
                import_kwh=im,
                export_kwh=ex,
                generation_kw=optional(pv),
                low_kw=optional(low),
                high_kw=optional(high),
                generation_kwh=optional(energy),
                measured_seconds=observed,
                estimated_seconds=estimated,
                flags=flags,
                state=STATES[state],
                reason=REASONS[reason],
            )
        )
    return dict(
        version=1,
        kind="hourly",
        config=f"{config:016x}",
        model=f"{model:016x}",
        assumptions="No material battery/generator contribution; negligible inverter reactive contribution",
        uncertainty="Uncalibrated for daytime PV accuracy",
        records=records,
    )
