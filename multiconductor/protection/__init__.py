"""Phase-aware protection devices for multiconductor studies."""

import sys

from multiconductor.protection.base import ProtectionDevice, switch_current_ka
from multiconductor.protection.devices import (
    DTOC,
    IDMT,
    IDTOC,
    Fuse,
    OCRelay,
    create_dtoc,
    create_fuse,
    create_idmt,
    create_idtoc,
    create_oc_relay,
    create_overcurrent_relay,
)
from multiconductor.protection.run_protection import (
    calculate_protection_times,
    protection_coordination_report,
    reset_protection_devices,
    run_protection,
)


DTOCRelay = DTOC
IDMTRelay = IDMT
IDTOCRelay = IDTOC

__all__ = [
    "DTOC",
    "DTOCRelay",
    "Fuse",
    "IDMT",
    "IDMTRelay",
    "IDTOC",
    "IDTOCRelay",
    "OCRelay",
    "ProtectionDevice",
    "calculate_protection_times",
    "create_dtoc",
    "create_fuse",
    "create_idmt",
    "create_idtoc",
    "create_oc_relay",
    "create_overcurrent_relay",
    "protection_coordination_report",
    "reset_protection_devices",
    "run_protection",
    "switch_current_ka",
]


_PARENT = sys.modules.get("multiconductor")
if _PARENT is not None:
    for _name in __all__:
        setattr(_PARENT, _name, globals()[_name])
