"""Multiconductor Network Diagnostics Suite."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from multiconductor.studies import MulticonductorStudyError
from multiconductor.tools.network_validators import ValidationResult

from ._common import build_network_graph
from ._diag_controls import check_controls
from ._diag_duplicates import check_duplicates
from ._diag_grounding import check_grounding
from ._diag_impedance import check_impedance
from ._diag_load_model import check_load_model
from ._diag_open_conductor import check_open_conductor
from ._diag_phase import check_phase_connectivity
from ._diag_topology import check_topology
from ._diag_transformer import check_transformers
from ._diag_voltage_base import check_voltage_base

_ALL_CATEGORIES: dict[str, Any] = {
    "voltage_base": check_voltage_base,
    "transformer": check_transformers,
    "grounding": check_grounding,
    "phase": check_phase_connectivity,
    "impedance": check_impedance,
    "open_conductor": check_open_conductor,
    "load_model": check_load_model,
    "controls": check_controls,
    "duplicates": check_duplicates,
    "topology": check_topology,
}

_FAST_CATEGORIES = {"voltage_base", "duplicates", "impedance", "load_model"}
_SEVERITY_ORDER = {"critical": 1, "high": 2, "medium": 3, "low": 4, "info": 5}
_NON_EMITTING_RULE_CODES = {"imp_05"}
_ISSUE_COLUMNS = [
    "code",
    "category",
    "severity",
    "check",
    "element_type",
    "element_index",
    "phase",
    "field",
    "evidence",
    "message",
    "recommendation",
    "suggestion",
]


class DiagnosticExecutionError(MulticonductorStudyError):
    """Raised when a diagnostic checker fails under strict execution."""


@dataclass(frozen=True, slots=True)
class DiagnosticsResult:
    issues: pd.DataFrame
    recommendations: pd.DataFrame
    summary: pd.DataFrame
    execution_errors: list[dict[str, Any]] = field(default_factory=list)


def _iter_rule_functions(category: str, check_fn: Any) -> list[tuple[str, Any]]:
    module = inspect.getmodule(check_fn)
    if module is None:
        return []
    rules: list[tuple[str, Any]] = []
    for name, value in vars(module).items():
        if not callable(value):
            continue
        match = re.match(r"^_([a-z]+_\d{2})_", name)
        if not match:
            continue
        code = match.group(1)
        if code in _NON_EMITTING_RULE_CODES:
            continue
        rules.append((code, value))
    return sorted(rules, key=lambda item: item[0])


def _collect_rule_codes() -> tuple[str, ...]:
    codes: set[str] = set()
    for category, check_fn in _ALL_CATEGORIES.items():
        codes.update(code for code, _ in _iter_rule_functions(category, check_fn))
    return tuple(sorted(codes))


RULE_CODES = _collect_rule_codes()


def _normalize_issue(
    raw_issue: dict[str, Any],
    *,
    code: str,
    category: str,
) -> dict[str, Any]:
    field = raw_issue.get("field")
    check = raw_issue.get("check", category)
    evidence = raw_issue.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {"field": field}
    recommendation = raw_issue.get("recommendation", raw_issue.get("suggestion"))
    normalized = dict(raw_issue)
    normalized.update(
        {
            "code": code,
            "category": check,
            "check": check,
            "phase": raw_issue.get("phase"),
            "evidence": evidence,
            "recommendation": recommendation,
            "suggestion": raw_issue.get("suggestion", recommendation),
        }
    )
    return normalized


def run_diagnostics(
    net: Any,
    *,
    categories: str | Sequence[str] = "all",
    severity_threshold: str = "info",
    fast_only: bool = False,
    strict: bool = True,
) -> DiagnosticsResult:
    if categories == "all":
        selected = set(_ALL_CATEGORIES)
    else:
        selected = set(categories)

    if fast_only:
        selected &= _FAST_CATEGORIES

    all_issues: list[dict[str, Any]] = []
    execution_errors: list[dict[str, Any]] = []
    for name, check_fn in _ALL_CATEGORIES.items():
        if name not in selected:
            continue
        kwargs: dict[str, Any] = {}
        if name == "voltage_base" and fast_only:
            kwargs["include_bfs"] = False
        elif name == "impedance" and fast_only:
            kwargs["include_matrix_checks"] = False
        elif name == "load_model" and fast_only:
            kwargs["include_capacity_checks"] = False
        rules = _iter_rule_functions(name, check_fn)
        if not rules:
            rules = [(name, lambda local_net, fn=check_fn, local_kwargs=kwargs: fn(local_net, **local_kwargs))]
        for code, rule_fn in rules:
            if fast_only and name == "voltage_base" and code == "vb_05":
                continue
            try:
                signature = inspect.signature(rule_fn)
                if "graph" in signature.parameters:
                    raw_result = rule_fn(net, build_network_graph(net))
                else:
                    raw_result = rule_fn(net)
                for raw_issue in raw_result:
                    all_issues.append(_normalize_issue(raw_issue, code=code, category=name))
            except Exception as exc:
                error = {
                    "category": name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                if strict:
                    raise DiagnosticExecutionError(
                        f"Diagnostic category '{name}' raised {type(exc).__name__}: {exc}"
                    ) from exc
                execution_errors.append(error)

    issues_df = pd.DataFrame(all_issues, columns=_ISSUE_COLUMNS)
    threshold_rank = _SEVERITY_ORDER.get(severity_threshold, 5)
    if not issues_df.empty:
        if "category" in issues_df.columns and "check" in issues_df.columns:
            issues_df["category"] = issues_df["category"].fillna(issues_df["check"])
        if "recommendation" in issues_df.columns and "suggestion" in issues_df.columns:
            issues_df["recommendation"] = issues_df["recommendation"].fillna(
                issues_df["suggestion"]
            )
        issues_df["_rank"] = issues_df["severity"].map(_SEVERITY_ORDER).fillna(99)
        issues_df = issues_df[issues_df["_rank"] <= threshold_rank]
        issues_df = issues_df.sort_values(
            by=["_rank", "code", "check", "element_type", "element_index", "field"],
            kind="stable",
        ).drop(columns=["_rank"]).reset_index(drop=True)

    if issues_df.empty:
        return DiagnosticsResult(
            issues=issues_df,
            recommendations=pd.DataFrame(columns=["priority", "issue_type", "recommendation"]),
            summary=pd.DataFrame([{"severity": "info", "count": 0}]),
            execution_errors=execution_errors,
        )

    rec_df = _build_recommendations(issues_df)
    summary_df = issues_df.groupby("severity", dropna=False).size().reset_index(name="count")
    summary_df["_rank"] = summary_df["severity"].map(_SEVERITY_ORDER).fillna(99)
    summary_df = summary_df.sort_values("_rank").drop(columns=["_rank"]).reset_index(drop=True)
    return DiagnosticsResult(
        issues=issues_df,
        recommendations=rec_df,
        summary=summary_df,
        execution_errors=execution_errors,
    )


_ACTION_MAP = {
    "voltage_base": "Verify and correct bus nominal voltages (vn_kv) and source voltage settings.",
    "transformer_model": "Fix transformer impedance, turns ratio, or tap settings per nameplate data.",
    "grounding": "Ensure neutral grounding paths are continuous and substation ground is defined.",
    "phase_connectivity": "Correct phase assignments and verify per-phase reachability from sources.",
    "impedance_data": "Verify conductor impedance data against standard libraries and check units.",
    "open_conductor": "Investigate broken conductor conditions and restore phase continuity.",
    "load_model": "Correct load power factor, voltage level, and balance across phases.",
    "control_error": "Fix control element references, setpoints, and switching thresholds.",
    "duplicate_equipment": "Remove or merge duplicate elements and resolve contradictory data.",
    "topology": "Repair network connectivity, remove unintended loops, and verify source paths.",
}


def _build_recommendations(issues_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        issues_df.groupby(["check", "severity"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(
            by=["severity", "count"],
            key=lambda s: s.map(_SEVERITY_ORDER).fillna(99) if s.name == "severity" else s,
            ascending=[True, False],
        )
    )
    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        issue_type = str(row["check"])
        rows.append(
            {
                "priority": str(row["severity"]),
                "issue_type": issue_type,
                "recommendation": _ACTION_MAP.get(
                    issue_type, "Investigate and resolve reported issues."
                ),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "DiagnosticExecutionError",
    "DiagnosticsResult",
    "RULE_CODES",
    "ValidationResult",
    "run_diagnostics",
]
