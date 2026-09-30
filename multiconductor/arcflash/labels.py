"""CSV, editable SVG, and optional ReportLab PDF arc-flash artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from multiconductor.arcflash.models import ArcFlashStudyResult


@dataclass(frozen=True, slots=True)
class ArcFlashArtifacts:
    """Paths created by :func:`export_arc_flash_artifacts`."""

    all_cases_csv: Path
    worst_cases_csv: Path
    operations_csv: Path
    svg_files: tuple[Path, ...]
    pdf_file: Path | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype != object:
            continue
        result[column] = result[column].map(
            lambda value: json.dumps(value, sort_keys=True, default=_json_default)
            if isinstance(value, (dict, list, tuple, np.ndarray))
            else value
        )
    return result


def _safe_filename(value: str, fallback: str = "arc-flash-label") -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized).strip("-._")
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned[:100] or fallback


def _number(row: pd.Series, name: str, decimals: int = 2) -> str:
    value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
    return "N/A" if not np.isfinite(value) else f"{float(value):.{decimals}f}"


def _label_fields(row: pd.Series) -> dict[str, str]:
    capped = bool(row.get("duration_capped", False))
    return {
        "equipment": str(row.get("equipment_id", "")),
        "access": str(row.get("access_area", "")),
        "voltage": _number(row, "nominal_voltage_kv", 3),
        "energy": _number(row, "incident_energy_cal_cm2", 2),
        "distance": _number(row, "working_distance_mm", 0),
        "boundary": _number(row, "arc_flash_boundary_mm", 0),
        "ppe": str(row.get("site_ppe", "")),
        "limited": _number(row, "limited_approach_boundary_mm", 0),
        "restricted": _number(row, "restricted_approach_boundary_mm", 0),
        "glove": str(row.get("glove_class", "")),
        "report": str(row.get("report_number", "")),
        "revision": str(row.get("revision", "")),
        "issue_date": str(row.get("issue_date", "")),
        "cap_notice": (
            "NOTICE: Incident energy is limited by the approved study duration cap."
            if capped
            else ""
        ),
    }


def _svg_label(row: pd.Series) -> str:
    values = {key: escape(value) for key, value in _label_fields(row).items()}
    cap = ""
    if values["cap_notice"]:
        cap = (
            '<rect x="35" y="1035" width="1730" height="70" fill="#fff3cd" '
            'stroke="#000" stroke-width="3"/>'
            f'<text x="900" y="1080" text-anchor="middle" class="cap">{values["cap_notice"]}</text>'
        )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="6in" height="4in" viewBox="0 0 1800 1200">
  <title>Arc-flash hazard label for {values["equipment"]} {values["access"]}</title>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #000; }}
    .warning {{ font-size: 112px; font-weight: 700; letter-spacing: 5px; }}
    .headline {{ font-size: 49px; font-weight: 700; }}
    .id {{ font-size: 43px; font-weight: 700; }}
    .label {{ font-size: 33px; font-weight: 700; }}
    .value {{ font-size: 33px; }}
    .energy {{ font-size: 55px; font-weight: 700; }}
    .small {{ font-size: 25px; }}
    .cap {{ font-size: 25px; font-weight: 700; }}
  </style>
  <rect width="1800" height="1200" fill="#fff" stroke="#000" stroke-width="14"/>
  <rect x="7" y="7" width="1786" height="190" fill="#f58220"/>
  <text x="900" y="137" text-anchor="middle" class="warning">WARNING</text>
  <text x="900" y="260" text-anchor="middle" class="headline">ARC FLASH AND SHOCK HAZARD</text>
  <line x1="35" y1="290" x2="1765" y2="290" stroke="#000" stroke-width="5"/>
  <text x="55" y="348" class="id">Equipment: {values["equipment"]}</text>
  <text x="55" y="405" class="id">Access: {values["access"]}</text>
  <line x1="35" y1="435" x2="1765" y2="435" stroke="#000" stroke-width="3"/>
  <text x="55" y="500" class="label">Nominal voltage:</text><text x="480" y="500" class="value">{values["voltage"]} kV AC</text>
  <text x="55" y="570" class="label">Incident energy:</text><text x="480" y="570" class="energy">{values["energy"]} cal/cm²</text>
  <text x="55" y="635" class="label">Working distance:</text><text x="480" y="635" class="value">{values["distance"]} mm</text>
  <text x="55" y="705" class="label">Arc-flash boundary:</text><text x="480" y="705" class="value">{values["boundary"]} mm (1.2 cal/cm²)</text>
  <line x1="900" y1="450" x2="900" y2="955" stroke="#000" stroke-width="3"/>
  <text x="940" y="500" class="label">Site-approved PPE:</text>
  <text x="940" y="550" class="value">{values["ppe"]}</text>
  <text x="940" y="635" class="label">Limited approach:</text><text x="1335" y="635" class="value">{values["limited"]} mm</text>
  <text x="940" y="705" class="label">Restricted approach:</text><text x="1370" y="705" class="value">{values["restricted"]} mm</text>
  <text x="940" y="775" class="label">Voltage-rated glove:</text><text x="1370" y="775" class="value">{values["glove"]}</text>
  <text x="940" y="865" class="small">Report {values["report"]} · Rev {values["revision"]} · {values["issue_date"]}</text>
  <text x="940" y="915" class="small">IEEE 1584 calculation — qualified engineering review required</text>
  {cap}
</svg>
'''


def _draw_pdf_label(canvas, row: pd.Series, page_width: float, page_height: float) -> None:
    from reportlab.lib.colors import HexColor, black, white

    fields = _label_fields(row)
    canvas.setStrokeColor(black)
    canvas.setLineWidth(3)
    canvas.rect(3, 3, page_width - 6, page_height - 6)
    canvas.setFillColor(HexColor("#f58220"))
    canvas.rect(3, page_height - 70, page_width - 6, 67, stroke=0, fill=1)
    canvas.setFillColor(black)
    canvas.setFont("Helvetica-Bold", 38)
    canvas.drawCentredString(page_width / 2, page_height - 52, "WARNING")
    canvas.setFont("Helvetica-Bold", 17)
    canvas.drawCentredString(page_width / 2, page_height - 91, "ARC FLASH AND SHOCK HAZARD")
    canvas.line(12, page_height - 101, page_width - 12, page_height - 101)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(16, page_height - 120, f"Equipment: {fields['equipment']}")
    canvas.drawString(16, page_height - 137, f"Access: {fields['access']}")
    canvas.line(12, page_height - 145, page_width - 12, page_height - 145)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(16, page_height - 165, "Nominal voltage:")
    canvas.drawString(16, page_height - 187, "Incident energy:")
    canvas.drawString(16, page_height - 209, "Working distance:")
    canvas.drawString(16, page_height - 231, "Arc-flash boundary:")
    canvas.setFont("Helvetica", 10)
    canvas.drawString(122, page_height - 165, f"{fields['voltage']} kV AC")
    canvas.setFont("Helvetica-Bold", 15)
    canvas.drawString(122, page_height - 188, f"{fields['energy']} cal/cm²")
    canvas.setFont("Helvetica", 10)
    canvas.drawString(122, page_height - 209, f"{fields['distance']} mm")
    canvas.drawString(122, page_height - 231, f"{fields['boundary']} mm (1.2 cal/cm²)")
    canvas.line(page_width / 2, page_height - 151, page_width / 2, 42)
    right = page_width / 2 + 12
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(right, page_height - 165, "Site-approved PPE:")
    canvas.setFont("Helvetica", 9)
    canvas.drawString(right, page_height - 181, fields["ppe"][:46])
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(right, page_height - 202, f"Limited approach: {fields['limited']} mm")
    canvas.drawString(right, page_height - 219, f"Restricted approach: {fields['restricted']} mm")
    canvas.drawString(right, page_height - 236, f"Voltage-rated glove: {fields['glove']}")
    canvas.setFont("Helvetica", 7)
    canvas.drawString(right, page_height - 255, f"Report {fields['report']} · Rev {fields['revision']} · {fields['issue_date']}")
    canvas.drawString(right, page_height - 267, "IEEE 1584 calculation — qualified engineering review required")
    if fields["cap_notice"]:
        canvas.setFillColor(HexColor("#fff3cd"))
        canvas.rect(12, 15, page_width - 24, 22, stroke=1, fill=1)
        canvas.setFillColor(black)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.drawCentredString(page_width / 2, 24, fields["cap_notice"])


def _write_pdf(rows: list[pd.Series], path: Path) -> None:
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    page_size = (6 * inch, 4 * inch)
    document = canvas.Canvas(str(path), pagesize=page_size, pageCompression=1)
    document.setTitle("Arc-Flash Hazard Labels")
    for row in rows:
        _draw_pdf_label(document, row, *page_size)
        document.showPage()
    document.save()


def export_arc_flash_artifacts(
    result_or_net,
    output_directory: str | Path,
    *,
    include_pdf: bool = True,
) -> ArcFlashArtifacts:
    """Export audit CSVs and valid worst-case labels.

    Invalid cases remain in ``arc_flash_all_cases.csv`` but cannot produce a
    sticker.  If ReportLab is unavailable, CSV and SVG files are still written
    and the returned artifact manifest contains an installation warning.
    """

    if isinstance(result_or_net, ArcFlashStudyResult):
        all_cases = result_or_net.all_cases
        worst = result_or_net.worst_cases
        operations = result_or_net.operations
    else:
        all_cases = result_or_net.get("res_arc_flash")
        worst = result_or_net.get("res_arc_flash_worst")
        operations = result_or_net.get("res_arc_flash_operation")
    if not all(isinstance(value, pd.DataFrame) for value in (all_cases, worst, operations)):
        raise TypeError("result_or_net does not contain completed arc-flash result tables")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    all_path = output / "arc_flash_all_cases.csv"
    worst_path = output / "arc_flash_worst_cases.csv"
    operation_path = output / "arc_flash_operations.csv"
    _csv_frame(all_cases).to_csv(all_path, index=False)
    _csv_frame(worst).to_csv(worst_path, index=False)
    _csv_frame(operations).to_csv(operation_path, index=False)

    svg_paths: list[Path] = []
    pdf_rows: list[pd.Series] = []
    used_names: set[str] = set()
    for ordinal, (_, row) in enumerate(worst.iterrows(), start=1):
        if str(row.get("status", "")) != "valid":
            continue
        copies = max(1, int(row.get("label_count", 1)))
        stem = _safe_filename(f"{row.get('equipment_id', '')}-{row.get('access_area', '')}")
        for copy_number in range(1, copies + 1):
            candidate = stem if copies == 1 else f"{stem}-copy-{copy_number}"
            if candidate.lower() in used_names:
                candidate = f"{candidate}-{ordinal}"
            used_names.add(candidate.lower())
            path = output / f"{candidate}.svg"
            path.write_text(_svg_label(row), encoding="utf-8", newline="\n")
            svg_paths.append(path)
            pdf_rows.append(row)

    warnings: list[str] = []
    pdf_path: Path | None = None
    if include_pdf and pdf_rows:
        candidate = output / "arc_flash_labels.pdf"
        try:
            _write_pdf(pdf_rows, candidate)
            pdf_path = candidate
        except ImportError:
            warnings.append(
                "ReportLab is not installed; install the 'arcflash' optional dependency to create PDF labels"
            )
    return ArcFlashArtifacts(
        all_cases_csv=all_path,
        worst_cases_csv=worst_path,
        operations_csv=operation_path,
        svg_files=tuple(svg_paths),
        pdf_file=pdf_path,
        warnings=tuple(warnings),
    )

