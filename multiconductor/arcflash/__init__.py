"""IEEE 1584 arc-flash analysis and hazard-label generation."""

from multiconductor.arcflash.ieee1584 import (
    ArcFlashInputError,
    ElectrodeConfiguration,
    IEEE1584Input,
    IEEE1584Result,
    calculate_ieee1584,
)
from multiconductor.arcflash.models import (
    ArcFlashStudyConfig,
    ArcFlashStudyResult,
    create_arc_flash_location,
    create_arc_flash_scenario,
    register_arc_flash_source,
)
from multiconductor.arcflash.study import ArcFlashStudyError, run_arc_flash
from multiconductor.arcflash.labels import ArcFlashArtifacts, export_arc_flash_artifacts

__all__ = [
    "ArcFlashInputError",
    "ArcFlashArtifacts",
    "ArcFlashStudyConfig",
    "ArcFlashStudyError",
    "ArcFlashStudyResult",
    "ElectrodeConfiguration",
    "IEEE1584Input",
    "IEEE1584Result",
    "calculate_ieee1584",
    "create_arc_flash_location",
    "create_arc_flash_scenario",
    "export_arc_flash_artifacts",
    "register_arc_flash_source",
    "run_arc_flash",
]
