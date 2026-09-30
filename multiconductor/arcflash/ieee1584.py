"""IEEE 1584-2018 arc-flash calculation equations.

The coefficient tables and equation layout in this module are adapted from
Li-aung Yip's MIT-licensed ``arcflash`` project (Copyright 2022 Li-aung Yip,
https://github.com/LiaungYip/arcflash).  That implementation is independently
tested against the IEEE 1584-2018 Annex D examples and the public IEEE
calculator.  The equations here use explicit engineering units (kV, kA, mm,
seconds, J/cm²) instead of adding a units-package dependency.

IEEE 1584 is an empirical model.  Inputs outside its published model range are
rejected instead of extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, log10, sqrt
from typing import Iterable


CAL_PER_JOULE = 1.0 / 4.184
ARC_FLASH_THRESHOLD_J_CM2 = 5.0208  # 1.2 cal/cm²


class ArcFlashInputError(ValueError):
    """Raised when data is outside the IEEE 1584-2018 model range."""


class ElectrodeConfiguration(str, Enum):
    """IEEE 1584-2018 electrode configurations."""

    VCB = "VCB"
    VCBB = "VCBB"
    HCB = "HCB"
    VOA = "VOA"
    HOA = "HOA"

    @classmethod
    def normalize(cls, value: "ElectrodeConfiguration | str") -> "ElectrodeConfiguration":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            raise ArcFlashInputError(
                "electrode_configuration must be VCB, VCBB, HCB, VOA, or HOA"
            ) from exc


@dataclass(frozen=True, slots=True)
class IEEE1584Input:
    """Physical inputs for one IEEE 1584 calculation.

    ``arcing_current_case`` is ``"full"`` for the average arcing current or
    ``"reduced"`` for the variation-adjusted lower arcing current.  The
    latter can produce more incident energy when a protective device clears
    more slowly at the lower current.
    """

    voltage_kv: float
    bolted_fault_current_ka: float
    gap_mm: float
    working_distance_mm: float
    electrode_configuration: ElectrodeConfiguration | str
    arc_duration_s: float
    enclosure_height_mm: float | None = None
    enclosure_width_mm: float | None = None
    enclosure_depth_mm: float | None = None
    arcing_current_case: str = "full"


@dataclass(frozen=True, slots=True)
class IEEE1584Result:
    """Calculated arcing current, energy, boundary, and audit intermediates."""

    voltage_kv: float
    bolted_fault_current_ka: float
    arcing_current_ka: float
    full_arcing_current_ka: float
    reduced_arcing_current_ka: float
    arcing_current_case: str
    arc_duration_s: float
    incident_energy_j_cm2: float
    incident_energy_cal_cm2: float
    arc_flash_boundary_mm: float
    variation_correction_factor: float
    enclosure_correction_factor: float
    equivalent_enclosure_size_in: float | None
    electrode_configuration: str
    gap_mm: float
    working_distance_mm: float
    energy_600_j_cm2: float | None = None
    energy_2700_j_cm2: float | None = None
    energy_14300_j_cm2: float | None = None


# IEEE 1584-2018 Tables 1 through 5 and Table 7.  Keys in Tables 3-5
# correspond to k1 ... k13 in the published equations.
_IARC_COEFFICIENTS = {
    ("VCB", 0.6): (-0.04287, 1.035, -0.083, 0, 0, -4.783e-09, 1.962e-06, -0.000229, 0.003141, 1.092),
    ("VCB", 2.7): (0.0065, 1.001, -0.024, -1.557e-12, 4.556e-10, -4.186e-08, 8.346e-07, 5.482e-05, -0.003191, 0.9729),
    ("VCB", 14.3): (0.005795, 1.015, -0.011, -1.557e-12, 4.556e-10, -4.186e-08, 8.346e-07, 5.482e-05, -0.003191, 0.9729),
    ("VCBB", 0.6): (-0.017432, 0.98, -0.05, 0, 0, -5.767e-09, 2.524e-06, -0.00034, 0.01187, 1.013),
    ("VCBB", 2.7): (0.002823, 0.995, -0.0125, 0, -9.204e-11, 2.901e-08, -3.262e-06, 0.0001569, -0.004003, 0.9825),
    ("VCBB", 14.3): (0.014827, 1.01, -0.01, 0, -9.204e-11, 2.901e-08, -3.262e-06, 0.0001569, -0.004003, 0.9825),
    ("HCB", 0.6): (0.054922, 0.988, -0.11, 0, 0, -5.382e-09, 2.316e-06, -0.000302, 0.0091, 0.9725),
    ("HCB", 2.7): (0.001011, 1.003, -0.0249, 0, 0, 4.859e-10, -1.814e-07, -9.128e-06, -0.0007, 0.9881),
    ("HCB", 14.3): (0.008693, 0.999, -0.02, 0, -5.043e-11, 2.233e-08, -3.046e-06, 0.000116, -0.001145, 0.9839),
    ("VOA", 0.6): (0.043785, 1.04, -0.18, 0, 0, -4.783e-09, 1.962e-06, -0.000229, 0.003141, 1.092),
    ("VOA", 2.7): (-0.02395, 1.006, -0.0188, -1.557e-12, 4.556e-10, -4.186e-08, 8.346e-07, 5.482e-05, -0.003191, 0.9729),
    ("VOA", 14.3): (0.005371, 1.0102, -0.029, -1.557e-12, 4.556e-10, -4.186e-08, 8.346e-07, 5.482e-05, -0.003191, 0.9729),
    ("HOA", 0.6): (0.111147, 1.008, -0.24, 0, 0, -3.895e-09, 1.641e-06, -0.000197, 0.002615, 1.1),
    ("HOA", 2.7): (0.000435, 1.006, -0.038, 0, 0, 7.859e-10, -1.914e-07, -9.128e-06, -0.0007, 0.9981),
    ("HOA", 14.3): (0.000904, 0.999, -0.02, 0, 0, 7.859e-10, -1.914e-07, -9.128e-06, -0.0007, 0.9981),
}

_VARIATION_COEFFICIENTS = {
    "VCB": (0, -1.4269e-06, 8.3137e-05, -0.0019382, 0.022366, -0.12645, 0.30226),
    "VCBB": (1.138e-06, -6.0287e-05, 0.0012758, -0.013778, 0.080217, -0.24066, 0.33524),
    "HCB": (0, -3.097e-06, 0.00016405, -0.0033609, 0.033308, -0.16182, 0.34627),
    "VOA": (9.5606e-07, -5.1543e-05, 0.0011161, -0.01242, 0.075125, -0.23584, 0.33696),
    "HOA": (0, -3.1555e-06, 0.0001682, -0.0034607, 0.034124, -0.1599, 0.34629),
}

_ENERGY_COEFFICIENTS = {
    0.6: {
        "VCB": (0.753364, 0.566, 1.752636, 0, 0, -4.783e-09, 1.962e-06, -0.000229, 0.003141, 1.092, 0, -1.598, 0.957),
        "VCBB": (3.068459, 0.26, -0.098107, 0, 0, -5.767e-09, 2.524e-06, -0.00034, 0.01187, 1.013, -0.06, -1.809, 1.19),
        "HCB": (4.073745, 0.344, -0.370259, 0, 0, -5.382e-09, 2.316e-06, -0.000302, 0.0091, 0.9725, 0, -2.03, 1.036),
        "VOA": (0.679294, 0.746, 1.222636, 0, 0, -4.783e-09, 1.962e-06, -0.000229, 0.003141, 1.092, 0, -1.598, 0.997),
        "HOA": (3.470417, 0.465, -0.261863, 0, 0, -3.895e-09, 1.641e-06, -0.000197, 0.002615, 1.1, 0, -1.99, 1.04),
    },
    2.7: {
        "VCB": (2.40021, 0.165, 0.354202, -1.557e-12, 4.556e-10, -4.186e-08, 8.346e-07, 5.482e-05, -0.003191, 0.9729, 0, -1.569, 0.9778),
        "VCBB": (3.870592, 0.185, -0.736618, 0, -9.204e-11, 2.901e-08, -3.262e-06, 0.0001569, -0.004003, 0.9825, 0, -1.742, 1.09),
        "HCB": (3.486391, 0.177, -0.193101, 0, 0, 4.859e-10, -1.814e-07, -9.128e-06, -0.0007, 0.9881, 0.027, -1.723, 1.055),
        "VOA": (3.880724, 0.105, -1.906033, -1.557e-12, 4.556e-10, -4.186e-08, 8.346e-07, 5.482e-05, -0.003191, 0.9729, 0, -1.515, 1.115),
        "HOA": (3.616266, 0.149, -0.761561, 0, 0, 7.859e-10, -1.914e-07, -9.128e-06, -0.0007, 0.9981, 0, -1.639, 1.078),
    },
    14.3: {
        "VCB": (3.825917, 0.11, -0.999749, -1.557e-12, 4.556e-10, -4.186e-08, 8.346e-07, 5.482e-05, -0.003191, 0.9729, 0, -1.568, 0.99),
        "VCBB": (3.644309, 0.215, -0.585522, 0, -9.204e-11, 2.901e-08, -3.262e-06, 0.0001569, -0.004003, 0.9825, 0, -1.677, 1.06),
        "HCB": (3.044516, 0.125, 0.245106, 0, -5.043e-11, 2.233e-08, -3.046e-06, 0.000116, -0.001145, 0.9839, 0, -1.655, 1.084),
        "VOA": (3.405454, 0.12, -0.93245, -1.557e-12, 4.556e-10, -4.186e-08, 8.346e-07, 5.482e-05, -0.003191, 0.9729, 0, -1.534, 0.979),
        "HOA": (2.04049, 0.177, 1.005092, 0, 0, 7.859e-10, -1.914e-07, -9.128e-06, -0.0007, 0.9981, -0.05, -1.633, 1.151),
    },
}

_ENCLOSURE_COEFFICIENTS = {
    ("Typical", "VCB"): (-0.000302, 0.03441, 0.4325),
    ("Typical", "VCBB"): (-0.0002976, 0.032, 0.479),
    ("Typical", "HCB"): (-0.0001923, 0.01935, 0.6899),
    ("Shallow", "VCB"): (0.002222, -0.02556, 0.6222),
    ("Shallow", "VCBB"): (-0.002778, 0.1194, -0.2778),
    ("Shallow", "HCB"): (-0.0005556, 0.03722, 0.4778),
}


@dataclass(frozen=True, slots=True)
class _Geometry:
    enclosure_factor: float
    equivalent_size_in: float | None
    enclosure_type: str | None


def _finite_positive(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ArcFlashInputError(f"{name} must be a finite positive value") from exc
    if not isfinite(value) or value <= 0:
        raise ArcFlashInputError(f"{name} must be a finite positive value")
    return value


def _validate(data: IEEE1584Input) -> tuple[ElectrodeConfiguration, str]:
    voltage = _finite_positive(data.voltage_kv, "voltage_kv")
    current = _finite_positive(data.bolted_fault_current_ka, "bolted_fault_current_ka")
    gap = _finite_positive(data.gap_mm, "gap_mm")
    distance = _finite_positive(data.working_distance_mm, "working_distance_mm")
    duration = _finite_positive(data.arc_duration_s, "arc_duration_s")
    del duration
    ec = ElectrodeConfiguration.normalize(data.electrode_configuration)
    case = str(data.arcing_current_case).strip().lower()
    if case not in {"full", "reduced"}:
        raise ArcFlashInputError('arcing_current_case must be "full" or "reduced"')
    if not 0.208 <= voltage <= 15.0:
        raise ArcFlashInputError("voltage_kv must be between 0.208 and 15.0 kV")
    if voltage <= 0.6:
        if not 0.5 <= current <= 106.0:
            raise ArcFlashInputError("LV bolted fault current must be between 0.5 and 106 kA")
        if not 6.35 <= gap <= 76.2:
            raise ArcFlashInputError("LV conductor gap must be between 6.35 and 76.2 mm")
    else:
        if not 0.2 <= current <= 65.0:
            raise ArcFlashInputError("HV bolted fault current must be between 0.2 and 65 kA")
        if not 19.05 <= gap <= 254.0:
            raise ArcFlashInputError("HV conductor gap must be between 19.05 and 254 mm")
    if distance < 305.0:
        raise ArcFlashInputError("working_distance_mm must be at least 305 mm")
    if ec not in {ElectrodeConfiguration.VOA, ElectrodeConfiguration.HOA}:
        height = _finite_positive(data.enclosure_height_mm, "enclosure_height_mm")
        width = _finite_positive(data.enclosure_width_mm, "enclosure_width_mm")
        _finite_positive(data.enclosure_depth_mm, "enclosure_depth_mm")
        if width < 4.0 * gap:
            raise ArcFlashInputError("enclosure width must be at least four times the conductor gap")
        if height <= 0:  # keeps type checkers and error messages straightforward
            raise ArcFlashInputError("enclosure_height_mm must be positive")
    return ec, case


def _poly(coefficients: tuple[float, ...], value: float) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


def _variation_factor(ec: str, voltage_kv: float) -> float:
    return _poly(_VARIATION_COEFFICIENTS[ec], voltage_kv)


def _adjust_dimension(dim_mm: float, voltage_kv: float, ec: str) -> float:
    constants = {"VCB": (4.0, 20.0), "VCBB": (10.0, 24.0), "HCB": (10.0, 22.0)}
    a, b = constants[ec]
    return (660.4 + (dim_mm - 660.4) * ((voltage_kv + a) / b)) / 25.4


def _geometry(data: IEEE1584Input, ec: str) -> _Geometry:
    if ec in {"VOA", "HOA"}:
        return _Geometry(1.0, None, None)
    height = float(data.enclosure_height_mm)
    width = float(data.enclosure_width_mm)
    depth = float(data.enclosure_depth_mm)
    shallow = data.voltage_kv < 0.6 and height < 508 and width < 508 and depth <= 203.2
    enclosure_type = "Shallow" if shallow else "Typical"
    mm_to_in = 0.03937  # intentionally matches the printed standard

    def width_1() -> float:
        if width < 508:
            return width * mm_to_in if shallow else 20.0
        if width <= 660.4:
            return width * mm_to_in
        return _adjust_dimension(min(width, 1244.6), data.voltage_kv, ec)

    def height_1() -> float:
        if height < 508:
            return height * mm_to_in if shallow else 20.0
        if height <= 660.4:
            return height * mm_to_in
        if ec == "VCB":
            return height * mm_to_in if height <= 1244.6 else 49.0
        return _adjust_dimension(min(height, 1244.6), data.voltage_kv, ec)

    ees = (height_1() + width_1()) / 2.0
    b1, b2, b3 = _ENCLOSURE_COEFFICIENTS[(enclosure_type, ec)]
    polynomial = b1 * ees**2 + b2 * ees + b3
    factor = polynomial if enclosure_type == "Typical" else 1.0 / polynomial
    if not 0.0 < factor <= 3.0:
        raise ArcFlashInputError("calculated enclosure correction factor is outside 0..3")
    return _Geometry(factor, ees, enclosure_type)


def _intermediate_arcing_current(ec: str, voltage: float, ibf: float, gap: float) -> float:
    k = _IARC_COEFFICIENTS[(ec, voltage)]
    x1 = k[0] + k[1] * log10(ibf) + k[2] * log10(gap)
    x2 = (
        k[3] * ibf**6
        + k[4] * ibf**5
        + k[5] * ibf**4
        + k[6] * ibf**3
        + k[7] * ibf**2
        + k[8] * ibf
        + k[9]
    )
    return 10**x1 * x2


def _interpolate(voltage_kv: float, x600: float, x2700: float, x14300: float) -> float:
    x1 = ((x2700 - x600) / 2.1) * (voltage_kv - 2.7) + x2700
    x2 = ((x14300 - x2700) / 11.6) * (voltage_kv - 14.3) + x14300
    x3 = x1 * (2.7 - voltage_kv) / 2.1 + x2 * (voltage_kv - 0.6) / 2.1
    return x3 if voltage_kv <= 2.7 else x2


def _lv_final_arcing_current(voltage_kv: float, iarc600: float, ibf: float) -> float:
    x1 = (0.6 / voltage_kv) ** 2
    x2 = 1.0 / iarc600**2
    x3 = (0.6**2 - voltage_kv**2) / (0.6**2 * ibf**2)
    return 1.0 / sqrt(x1 * (x2 - x3))


def _intermediate_energy(
    *,
    voltage: float,
    ec: str,
    iarc: float,
    ibf: float,
    duration_s: float,
    gap: float,
    enclosure_factor: float,
    distance: float,
    iarc600_for_lv: float | None = None,
) -> float:
    k = _ENERGY_COEFFICIENTS[0.6 if voltage <= 0.6 else voltage][ec]
    duration_ms = duration_s * 1000.0
    x1 = 12.552 / 50.0 * duration_ms
    x2 = k[0] + k[1] * log10(gap)
    numerator_current = iarc if iarc600_for_lv is None else iarc600_for_lv
    numerator = k[2] * numerator_current
    denominator = (
        k[3] * ibf**7
        + k[4] * ibf**6
        + k[5] * ibf**5
        + k[6] * ibf**4
        + k[7] * ibf**3
        + k[8] * ibf**2
        + k[9] * ibf
    )
    x3 = numerator / denominator
    x4 = k[10] * log10(ibf) + k[12] * log10(iarc) + log10(1.0 / enclosure_factor)
    x5 = k[11] * log10(distance)
    return x1 * 10 ** (x2 + x3 + x4 + x5)


def _boundary_from_energy(voltage: float, ec: str, energy_j_cm2: float, distance_mm: float) -> float:
    k = _ENERGY_COEFFICIENTS[0.6 if voltage <= 0.6 else voltage][ec]
    f_value = energy_j_cm2 / distance_mm ** k[11]
    return (ARC_FLASH_THRESHOLD_J_CM2 / f_value) ** (1.0 / k[11])


def calculate_ieee1584(data: IEEE1584Input) -> IEEE1584Result:
    """Calculate arcing current, incident energy, and arc-flash boundary."""

    ec_value, case = _validate(data)
    ec = ec_value.value
    geometry = _geometry(data, ec)
    voltage = float(data.voltage_kv)
    ibf = float(data.bolted_fault_current_ka)
    variation = _variation_factor(ec, voltage)
    reduction = 1.0 - 0.5 * variation

    full_intermediates = {
        level: _intermediate_arcing_current(ec, level, ibf, float(data.gap_mm))
        for level in (0.6, 2.7, 14.3)
    }
    if voltage <= 0.6:
        full_iarc = _lv_final_arcing_current(voltage, full_intermediates[0.6], ibf)
        reduced_iarc = full_iarc * reduction
        selected_iarc = full_iarc if case == "full" else reduced_iarc
        energy = _intermediate_energy(
            voltage=voltage,
            ec=ec,
            iarc=selected_iarc,
            ibf=ibf,
            duration_s=float(data.arc_duration_s),
            gap=float(data.gap_mm),
            enclosure_factor=geometry.enclosure_factor,
            distance=float(data.working_distance_mm),
            iarc600_for_lv=full_intermediates[0.6],
        )
        boundary = _boundary_from_energy(voltage, ec, energy, float(data.working_distance_mm))
        energies = (None, None, None)
    else:
        reduced_intermediates = {key: value * reduction for key, value in full_intermediates.items()}
        selected = full_intermediates if case == "full" else reduced_intermediates
        full_iarc = _interpolate(voltage, *full_intermediates.values())
        reduced_iarc = _interpolate(voltage, *reduced_intermediates.values())
        selected_iarc = full_iarc if case == "full" else reduced_iarc
        energy_values = {
            level: _intermediate_energy(
                voltage=level,
                ec=ec,
                iarc=selected[level],
                ibf=ibf,
                duration_s=float(data.arc_duration_s),
                gap=float(data.gap_mm),
                enclosure_factor=geometry.enclosure_factor,
                distance=float(data.working_distance_mm),
            )
            for level in (0.6, 2.7, 14.3)
        }
        boundary_values = {
            level: _boundary_from_energy(level, ec, energy_values[level], float(data.working_distance_mm))
            for level in (0.6, 2.7, 14.3)
        }
        energy = _interpolate(voltage, *energy_values.values())
        boundary = _interpolate(voltage, *boundary_values.values())
        energies = tuple(energy_values.values())

    return IEEE1584Result(
        voltage_kv=voltage,
        bolted_fault_current_ka=ibf,
        arcing_current_ka=selected_iarc,
        full_arcing_current_ka=full_iarc,
        reduced_arcing_current_ka=reduced_iarc,
        arcing_current_case=case,
        arc_duration_s=float(data.arc_duration_s),
        incident_energy_j_cm2=energy,
        incident_energy_cal_cm2=energy * CAL_PER_JOULE,
        arc_flash_boundary_mm=boundary,
        variation_correction_factor=variation,
        enclosure_correction_factor=geometry.enclosure_factor,
        equivalent_enclosure_size_in=geometry.equivalent_size_in,
        electrode_configuration=ec,
        gap_mm=float(data.gap_mm),
        working_distance_mm=float(data.working_distance_mm),
        energy_600_j_cm2=energies[0],
        energy_2700_j_cm2=energies[1],
        energy_14300_j_cm2=energies[2],
    )


def combine_ieee1584_results(results: Iterable[IEEE1584Result]) -> tuple[float, float]:
    """Combine multi-segment energy and return ``(cal/cm², boundary_mm)``.

    IEEE 1584 high-voltage boundary interpolation must occur after summing the
    three intermediate energies, not by scaling the final incident energy.
    """

    values = tuple(results)
    if not values:
        return 0.0, 0.0
    first = values[0]
    for value in values[1:]:
        if (
            value.voltage_kv != first.voltage_kv
            or value.electrode_configuration != first.electrode_configuration
            or value.working_distance_mm != first.working_distance_mm
        ):
            raise ArcFlashInputError("multi-segment results must describe the same equipment geometry")
    total_j = sum(value.incident_energy_j_cm2 for value in values)
    if first.voltage_kv <= 0.6:
        boundary = _boundary_from_energy(
            first.voltage_kv,
            first.electrode_configuration,
            total_j,
            first.working_distance_mm,
        )
    else:
        intermediate = []
        for level, field in (
            (0.6, "energy_600_j_cm2"),
            (2.7, "energy_2700_j_cm2"),
            (14.3, "energy_14300_j_cm2"),
        ):
            energy = sum(float(getattr(value, field)) for value in values)
            intermediate.append(
                _boundary_from_energy(
                    level,
                    first.electrode_configuration,
                    energy,
                    first.working_distance_mm,
                )
            )
        boundary = _interpolate(first.voltage_kv, *intermediate)
    return total_j * CAL_PER_JOULE, boundary
