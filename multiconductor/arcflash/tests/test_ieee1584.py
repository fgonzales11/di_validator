from dataclasses import replace

import pytest

from multiconductor.arcflash import (
    ArcFlashInputError,
    IEEE1584Input,
    calculate_ieee1584,
)
from multiconductor.arcflash.ieee1584 import combine_ieee1584_results


@pytest.mark.parametrize(
    "data, expected_current, expected_energy_j, expected_boundary",
    [
        (
            IEEE1584Input(
                4.16, 15.0, 104.0, 914.4, "VCB", 0.197, 1143.0, 762.0, 508.0
            ),
            12.979,
            12.152,
            1606.0,
        ),
        (
            IEEE1584Input(
                4.16,
                15.0,
                104.0,
                914.4,
                "VCB",
                0.223,
                1143.0,
                762.0,
                508.0,
                "reduced",
            ),
            12.675,
            13.343,
            1704.0,
        ),
        (
            IEEE1584Input(
                0.48, 45.0, 32.0, 609.6, "VCB", 0.0613, 610.0, 610.0, 254.0
            ),
            28.793,
            11.585,
            1029.0,
        ),
        (
            IEEE1584Input(
                0.48,
                45.0,
                32.0,
                609.6,
                "VCB",
                0.319,
                610.0,
                610.0,
                254.0,
                "reduced",
            ),
            25.244,
            53.156,
            2669.0,
        ),
    ],
)
def test_ieee_1584_annex_d_examples(
    data, expected_current, expected_energy_j, expected_boundary
):
    result = calculate_ieee1584(data)
    assert result.arcing_current_ka == pytest.approx(expected_current, abs=0.001)
    assert result.incident_energy_j_cm2 == pytest.approx(expected_energy_j, abs=0.001)
    assert result.arc_flash_boundary_mm == pytest.approx(expected_boundary, abs=1.0)


@pytest.mark.parametrize("configuration", ["VCB", "VCBB", "HCB", "VOA", "HOA"])
def test_every_electrode_configuration(configuration):
    enclosed = configuration not in {"VOA", "HOA"}
    result = calculate_ieee1584(
        IEEE1584Input(
            voltage_kv=0.48,
            bolted_fault_current_ka=20.0,
            gap_mm=32.0,
            working_distance_mm=610.0,
            electrode_configuration=configuration,
            arc_duration_s=0.2,
            enclosure_height_mm=610.0 if enclosed else None,
            enclosure_width_mm=610.0 if enclosed else None,
            enclosure_depth_mm=254.0 if enclosed else None,
        )
    )
    assert result.arcing_current_ka > 0.0
    assert result.incident_energy_cal_cm2 > 0.0
    assert result.arc_flash_boundary_mm > 0.0


def test_model_range_rejections():
    valid = IEEE1584Input(0.48, 20.0, 32.0, 610.0, "VCB", 0.2, 610, 610, 254)
    for changed in (
        replace(valid, voltage_kv=0.12),
        replace(valid, bolted_fault_current_ka=0.1),
        replace(valid, gap_mm=2.0),
        replace(valid, working_distance_mm=100.0),
        replace(valid, enclosure_width_mm=100.0),
    ):
        with pytest.raises(ArcFlashInputError):
            calculate_ieee1584(changed)


def test_segment_energy_and_boundary_accumulate():
    base = IEEE1584Input(0.48, 20.0, 32.0, 610.0, "VCB", 0.4, 610, 610, 254)
    whole = calculate_ieee1584(base)
    parts = [
        calculate_ieee1584(replace(base, arc_duration_s=0.1)),
        calculate_ieee1584(replace(base, arc_duration_s=0.3)),
    ]
    energy, boundary = combine_ieee1584_results(parts)
    assert energy == pytest.approx(whole.incident_energy_cal_cm2)
    assert boundary == pytest.approx(whole.arc_flash_boundary_mm)

