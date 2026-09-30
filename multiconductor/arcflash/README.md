# Arc-flash studies

This package implements the IEEE 1584-2018 three-phase AC empirical model for
VCB, VCBB, HCB, VOA, and HOA electrodes from 208 V through 15 kV. It also runs
explicit network scenarios, models an arc as an equivalent per-phase
resistance, accumulates protection operating state across source/topology
changes, and exports audit CSV files plus 6 × 4 inch labels.

The equation implementation is adapted from Li-aung Yip's MIT-licensed
[`arcflash`](https://github.com/LiaungYip/arcflash) implementation and is
checked against the IEEE 1584-2018 Annex D examples. IEEE 1584 remains the
controlling standard; inputs outside its empirical range are rejected.

## Minimal workflow

```python
from multiconductor import (
    ArcFlashStudyConfig,
    create_arc_flash_location,
    create_arc_flash_scenario,
    export_arc_flash_artifacts,
    run_arc_flash,
)

create_arc_flash_location(
    net,
    bus=7,
    equipment_id="SWGR-1",
    access_area="Front",
    conductor_gap_mm=32,
    working_distance_mm=610,
    electrode_configuration="VCB",
    enclosure_height_mm=610,
    enclosure_width_mm=610,
    enclosure_depth_mm=254,
    protection_device_ids=[3],
    site_ppe="Site PPE procedure E-12",
    limited_approach_boundary_mm=1070,
    restricted_approach_boundary_mm=305,
    glove_class="Class 00",
    label_count=2,
)
create_arc_flash_scenario(net, "Normal maximum", "max")
create_arc_flash_scenario(net, "Normal minimum", "min")

study = run_arc_flash(
    net,
    ArcFlashStudyConfig(
        max_arc_duration_s=2.0,
        study_id="AF-2026-01",
        report_number="RPT-42",
        revision="0",
        issue_date="2026-08-25",
    ),
)
artifacts = export_arc_flash_artifacts(study, "arc-flash-output")
```

Scenario actions are mappings with `table`, `index`, `column`, and `value`
keys. They are applied only to deep study copies. Register generators and
motors with `register_arc_flash_source`; induction motors default to removal
after five cycles, while all other decrement profiles must be supplied from
field-verified data.

Relay devices used by an arc-flash location require an explicit
`opening_time_s`. Fuse curves must set
`curve_kind="manufacturer_total_clearing"`; generic curves are intentionally
rejected for arc-flash studies.

PPE, shock boundaries, glove class, protection scope, report metadata, and the
maximum arc duration are approved project inputs. They are never inferred.
Every result and generated label requires review by qualified engineering
personnel. Labels do not claim material or printer certification.
