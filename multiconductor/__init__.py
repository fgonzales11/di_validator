# ruff: noqa: E402,F401,F403

import os
from importlib import import_module, util

mc_dir = os.path.dirname(os.path.realpath(__file__))

from multiconductor._version import __version__, __format_version__
from multiconductor.file_io import create_empty_network, from_excel, from_pickle
from multiconductor.create import *
from multiconductor.pycci import *
from multiconductor.load_allocation import *

_LAZY_EXPORTS = {
    "ArcFlashArtifacts": ("multiconductor.arcflash", "ArcFlashArtifacts"),
    "ArcFlashStudyConfig": ("multiconductor.arcflash", "ArcFlashStudyConfig"),
    "ArcFlashStudyResult": ("multiconductor.arcflash", "ArcFlashStudyResult"),
    "IEEE1584Input": ("multiconductor.arcflash", "IEEE1584Input"),
    "IEEE1584Result": ("multiconductor.arcflash", "IEEE1584Result"),
    "ContingencyAction": ("multiconductor.contingency", "ContingencyAction"),
    "ContingencyCase": ("multiconductor.contingency", "ContingencyCase"),
    "ContingencyCaseResult": ("multiconductor.contingency", "ContingencyCaseResult"),
    "ContingencyResult": ("multiconductor.contingency", "ContingencyResult"),
    "CSVData": ("multiconductor.timeseries", "CSVData"),
    "DFData": ("multiconductor.timeseries", "DFData"),
    "DataSource": ("multiconductor.timeseries", "DataSource"),
    "DiagnosticExecutionError": (
        "multiconductor.tools.diagnostics",
        "DiagnosticExecutionError",
    ),
    "EquivalentResult": ("multiconductor.grid_equivalents", "EquivalentResult"),
    "EquivalentSolveResult": ("multiconductor.grid_equivalents", "EquivalentSolveResult"),
    "ElementRef": ("multiconductor.studies", "ElementRef"),
    "FaultSpec": ("multiconductor.shortcircuit", "FaultSpec"),
    "GridEquivalentModel": ("multiconductor.grid_equivalents", "GridEquivalentModel"),
    "InvalidStudyInput": ("multiconductor.studies", "InvalidStudyInput"),
    "MulticonductorStudyError": ("multiconductor.studies", "MulticonductorStudyError"),
    "OPFNotConverged": ("multiconductor.opf", "OPFNotConverged"),
    "OPFResult": ("multiconductor.opf", "OPFResult"),
    "OutputWriter": ("multiconductor.timeseries", "OutputWriter"),
    "ProfileBinding": ("multiconductor.timeseries", "ProfileBinding"),
    "ResultSpec": ("multiconductor.timeseries", "ResultSpec"),
    "StudyNotConverged": ("multiconductor.studies", "StudyNotConverged"),
    "StudySnapshot": ("multiconductor.studies", "StudySnapshot"),
    "StudyStatus": ("multiconductor.studies", "StudyStatus"),
    "StateEstimationError": (
        "multiconductor.pycci.state_estimator",
        "StateEstimationError",
    ),
    "StateEstimationNotConverged": (
        "multiconductor.pycci.state_estimator",
        "StateEstimationNotConverged",
    ),
    "ShortCircuitCalculationError": (
        "multiconductor.shortcircuit",
        "ShortCircuitCalculationError",
    ),
    "ShortCircuitSourceError": (
        "multiconductor.shortcircuit",
        "ShortCircuitSourceError",
    ),
    "TerminalRef": ("multiconductor.studies", "TerminalRef"),
    "UnobservableNetworkError": (
        "multiconductor.pycci.state_estimator",
        "UnobservableNetworkError",
    ),
    "ValidationMetrics": ("multiconductor.grid_equivalents", "ValidationMetrics"),
    "calc_sc": ("multiconductor.shortcircuit", "calc_sc"),
    "calc_sc_native": ("multiconductor.shortcircuit", "calc_sc_native"),
    "calculate_ieee1584": ("multiconductor.arcflash", "calculate_ieee1584"),
    "create_arc_flash_location": (
        "multiconductor.arcflash",
        "create_arc_flash_location",
    ),
    "create_arc_flash_scenario": (
        "multiconductor.arcflash",
        "create_arc_flash_scenario",
    ),
    "create_state_measurement": (
        "multiconductor.pycci.state_estimator",
        "create_state_measurement",
    ),
    "create_poly_cost": ("multiconductor.opf", "create_poly_cost"),
    "create_pwl_cost": ("multiconductor.opf", "create_pwl_cost"),
    "estimate": ("multiconductor.pycci.state_estimator", "estimate"),
    "export_arc_flash_artifacts": (
        "multiconductor.arcflash",
        "export_arc_flash_artifacts",
    ),
    "generate_nminus1_cases": ("multiconductor.contingency", "generate_nminus1_cases"),
    "get_equivalent": ("multiconductor.grid_equivalents", "get_equivalent"),
    "preserved_network": ("multiconductor.studies", "preserved_network"),
    "run_contingency": ("multiconductor.contingency", "run_contingency"),
    "register_arc_flash_source": (
        "multiconductor.arcflash",
        "register_arc_flash_source",
    ),
    "run_arc_flash": ("multiconductor.arcflash", "run_arc_flash"),
    "run_diagnostics": ("multiconductor.tools.diagnostics", "run_diagnostics"),
    "run_opf": ("multiconductor.opf", "run_opf"),
    "run_mc_timeseries": ("multiconductor.timeseries", "run_mc_timeseries"),
    "run_timeseries": ("multiconductor.timeseries", "run_timeseries"),
    "solve_equivalent": ("multiconductor.grid_equivalents", "solve_equivalent"),
}

if util.find_spec("multiconductor.protection") is not None:
    _LAZY_EXPORTS.update(
        {
            "DTOC": ("multiconductor.protection", "DTOC"),
            "Fuse": ("multiconductor.protection", "Fuse"),
            "IDMT": ("multiconductor.protection", "IDMT"),
            "IDTOC": ("multiconductor.protection", "IDTOC"),
            "OCRelay": ("multiconductor.protection", "OCRelay"),
            "ProtectionDevice": ("multiconductor.protection", "ProtectionDevice"),
            "calculate_protection_times": (
                "multiconductor.protection",
                "calculate_protection_times",
            ),
            "create_dtoc": ("multiconductor.protection", "create_dtoc"),
            "create_fuse": ("multiconductor.protection", "create_fuse"),
            "create_idmt": ("multiconductor.protection", "create_idmt"),
            "create_idtoc": ("multiconductor.protection", "create_idtoc"),
            "create_oc_relay": ("multiconductor.protection", "create_oc_relay"),
            "protection_coordination_report": (
                "multiconductor.protection",
                "protection_coordination_report",
            ),
            "reset_protection_devices": (
                "multiconductor.protection",
                "reset_protection_devices",
            ),
            "run_protection": ("multiconductor.protection", "run_protection"),
        }
    )

for _lazy_name in tuple(_LAZY_EXPORTS):
    globals().pop(_lazy_name, None)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = sorted(
    {
        *(
            name
            for name in globals()
            if not name.startswith("_")
            and name not in {"import_module", "mc_dir", "os", "util"}
        ),
        *_LAZY_EXPORTS,
    }
)
