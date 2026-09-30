from __future__ import annotations

import re

import pandas as pd
import pytest

import multiconductor as mc
from multiconductor.arcflash import (
    ArcFlashStudyConfig,
    ArcFlashStudyResult,
    create_arc_flash_location,
    create_arc_flash_scenario,
    export_arc_flash_artifacts,
    register_arc_flash_source,
)
from multiconductor.arcflash.models import (
    RES_ARC_FLASH_COLUMNS,
    RES_ARC_FLASH_OPERATION_COLUMNS,
    RES_ARC_FLASH_WORST_COLUMNS,
)


def test_input_tables_and_motor_default_profile():
    net = mc.create_empty_network()
    bus = mc.create_bus(net, 0.48)
    location = create_arc_flash_location(
        net,
        bus,
        "MCC/1",
        "Front doors",
        conductor_gap_mm=32,
        working_distance_mm=610,
        electrode_configuration="VCB",
        enclosure_height_mm=610,
        enclosure_width_mm=610,
        enclosure_depth_mm=254,
        site_ppe="PPE procedure 14",
        limited_approach_boundary_mm=1070,
        restricted_approach_boundary_mm=305,
        glove_class="Class 00",
        label_count=2,
    )
    create_arc_flash_scenario(net, "maximum", "max")
    create_arc_flash_scenario(net, "minimum", "min")
    net.asymmetric_sgen.loc[(0, 1), :] = None
    net.asymmetric_sgen.loc[(0, 1), "in_service"] = True
    source = register_arc_flash_source(
        net,
        "asymmetric_sgen",
        (0, 1),
        "induction_motor",
        frequency_hz=60,
    )
    assert location in net.arc_flash_location.index
    assert source in net.arc_flash_source.index
    assert net.arc_flash_source.at[source, "decrement_profile"] == [
        [0.0, 1.0],
        [pytest.approx(5 / 60), 0.0],
    ]


def _sample_result() -> ArcFlashStudyResult:
    all_row = {column: None for column in RES_ARC_FLASH_COLUMNS}
    all_row.update(
        {
            "location_id": 0,
            "scenario_id": 1,
            "equipment_id": "SWGR/1",
            "access_area": "Front: Main",
            "status": "valid",
            "duration_capped": True,
            "nominal_voltage_kv": 0.48,
            "working_distance_mm": 610.0,
            "incident_energy_cal_cm2": 12.34,
            "arc_flash_boundary_mm": 2669.0,
            "warnings": ["capped"],
        }
    )
    worst_row = {column: all_row.get(column) for column in RES_ARC_FLASH_WORST_COLUMNS}
    worst_row.update(
        {
            "site_ppe": "Site PPE procedure E-12",
            "limited_approach_boundary_mm": 1070,
            "restricted_approach_boundary_mm": 305,
            "glove_class": "Class 00",
            "report_number": "RPT-42",
            "revision": "A",
            "issue_date": "2026-08-25",
            "label_count": 2,
        }
    )
    return ArcFlashStudyResult(
        all_cases=pd.DataFrame([all_row], columns=RES_ARC_FLASH_COLUMNS),
        worst_cases=pd.DataFrame([worst_row], columns=RES_ARC_FLASH_WORST_COLUMNS),
        operations=pd.DataFrame(columns=RES_ARC_FLASH_OPERATION_COLUMNS),
        study_config=ArcFlashStudyConfig(2, "AF-1", "RPT-42", "A", "2026-08-25"),
    )


def test_csv_svg_and_repeated_label_artifacts(tmp_path):
    artifacts = export_arc_flash_artifacts(_sample_result(), tmp_path, include_pdf=False)
    assert artifacts.all_cases_csv.name == "arc_flash_all_cases.csv"
    assert artifacts.worst_cases_csv.exists()
    assert artifacts.operations_csv.exists()
    assert len(artifacts.svg_files) == 2
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+\.svg", path.name) for path in artifacts.svg_files)
    svg = artifacts.svg_files[0].read_text(encoding="utf-8")
    assert 'width="6in" height="4in"' in svg
    assert "WARNING" in svg
    assert "12.34 cal/cm²" in svg
    assert "duration cap" in svg
    assert "Site PPE procedure E-12" in svg


def test_invalid_case_does_not_get_label(tmp_path):
    study = _sample_result()
    study.worst_cases.loc[:, "status"] = "invalid"
    artifacts = export_arc_flash_artifacts(study, tmp_path, include_pdf=False)
    assert not artifacts.svg_files

