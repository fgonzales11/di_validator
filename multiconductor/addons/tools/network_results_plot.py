"""Interactive MC/OpenDSS result plots shared by the notebook and Streamlit."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go


SOURCE_COLORS = {"MC": "#1f77b4", "DSS": "#ff7f0e"}
SOURCE_SYMBOLS = {"MC": "circle", "DSS": "diamond"}


def _circuit_key(value: object) -> str:
    """Use the filename stem so notebook and app circuit labels agree."""
    return Path(str(value)).stem


def build_plot_dataframes(
    comparison_df: pd.DataFrame,
    line_comparison_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert Streamlit benchmark frames to the notebook plotting schema.

    The returned frames have the same ``circuit_key``/identifier columns and
    ``<FIELD>_MC``/``<FIELD>_DSS`` pairs created by ``mesh_benchmark.ipynb``.
    """
    bus = pd.DataFrame(index=comparison_df.index)
    if not comparison_df.empty:
        bus["circuit_key"] = comparison_df["network"].map(_circuit_key)
        if "dss_bus" in comparison_df:
            bus["NODE_ID"] = comparison_df["dss_bus"]
        elif "bus_idx" in comparison_df:
            bus["NODE_ID"] = comparison_df["bus_idx"].astype(str)

        if "vn_kv" in comparison_df:
            bus["BASE_VOLTAGE_MC"] = pd.to_numeric(
                comparison_df["vn_kv"], errors="coerce"
            )
        if "dss_kv_base_ln" in comparison_df:
            # The notebook reports the OpenDSS L-N kVBase as L-L kV.
            bus["BASE_VOLTAGE_DSS"] = (
                pd.to_numeric(comparison_df["dss_kv_base_ln"], errors="coerce")
                * np.sqrt(3.0)
            )

        for phase_upper, phase_lower in (("A", "a"), ("B", "b"), ("C", "c")):
            mappings = {
                f"V{phase_upper}_MC": f"vm_{phase_lower}_pu",
                f"V{phase_upper}_DSS": f"dss_vm_{phase_lower}_pu",
                f"V{phase_upper}_ANGLE_MC": f"va_{phase_lower}_degree",
                f"V{phase_upper}_ANGLE_DSS": f"dss_va_{phase_lower}_degree",
            }
            for output_column, source_column in mappings.items():
                if source_column in comparison_df:
                    bus[output_column] = pd.to_numeric(
                        comparison_df[source_column], errors="coerce"
                    )

    line = pd.DataFrame(index=line_comparison_df.index)
    if not line_comparison_df.empty:
        line["circuit_key"] = line_comparison_df["network"].map(_circuit_key)
        line["LINE_ID"] = line_comparison_df.get(
            "line_name", line_comparison_df.get("line_idx")
        )
        line["PHASE"] = line_comparison_df["phase"].astype(str).str.upper()
        line_fields = {
            "P_FROM": "p_from_mw",
            "Q_FROM": "q_from_mvar",
            "P_TO": "p_to_mw",
            "Q_TO": "q_to_mvar",
            "I_FROM": "i_from_ka",
            "I_FROM_ANGLE": "i_from_angle_degree",
            "I_TO": "i_to_ka",
            "I_TO_ANGLE": "i_to_angle_degree",
        }
        for plot_field, result_field in line_fields.items():
            for source, prefix in (("MC", "mc"), ("DSS", "dss")):
                source_column = f"{prefix}_{result_field}"
                if source_column in line_comparison_df:
                    line[f"{plot_field}_{source}"] = pd.to_numeric(
                        line_comparison_df[source_column], errors="coerce"
                    )

    return bus.reset_index(drop=True), line.reset_index(drop=True)


def _fields(frame: pd.DataFrame) -> list[str]:
    return [
        column[:-3]
        for column in frame.columns
        if column.endswith("_MC") and f"{column[:-3]}_DSS" in frame.columns
    ]


def plot_options(
    df_combined: pd.DataFrame,
    df_power_combined: pd.DataFrame,
) -> dict[str, dict[str, list[str]]]:
    """Return available circuits and result fields for each plot category."""
    candidates = {
        "Bus": (df_combined, ["NODE_ID"]),
        "Line": (df_power_combined, ["LINE_ID", "PHASE"]),
    }
    options: dict[str, dict[str, list[str]]] = {}
    for category, (frame, id_columns) in candidates.items():
        required = {"circuit_key", *id_columns}
        fields = _fields(frame)
        if frame.empty or not required.issubset(frame.columns) or not fields:
            continue
        circuits = [str(value) for value in frame["circuit_key"].dropna().unique()]
        if circuits:
            options[category] = {"circuits": circuits, "fields": fields}
    return options


def plot_results(
    df_combined: pd.DataFrame,
    df_power_combined: pd.DataFrame,
    *,
    category: str | None = None,
    circuit_key: str | None = None,
    field: str | None = None,
    sources: Iterable[str] = ("MC", "DSS"),
) -> go.Figure:
    """Build a Plotly comparison figure and return it without displaying it."""
    datasets = {
        "Bus": {"frame": df_combined, "id_columns": ["NODE_ID"]},
        "Line": {"frame": df_power_combined, "id_columns": ["LINE_ID", "PHASE"]},
    }
    options = plot_options(df_combined, df_power_combined)
    if not options:
        raise ValueError("No paired MC/OpenDSS bus or line result fields are available.")

    category = category if category in options else next(iter(options))
    category_options = options[category]
    if circuit_key not in category_options["circuits"]:
        circuit_key = category_options["circuits"][0]
    if field not in category_options["fields"]:
        preferred = "VA" if category == "Bus" and "VA" in category_options["fields"] else None
        field = preferred or category_options["fields"][0]

    selected_sources = [source.upper() for source in sources]
    selected_sources = [source for source in ("MC", "DSS") if source in selected_sources]
    if not selected_sources:
        selected_sources = ["MC", "DSS"]

    dataset = datasets[category]
    frame = dataset["frame"]
    subset = frame.loc[frame["circuit_key"].astype(str) == circuit_key].reset_index(drop=True)

    if len(dataset["id_columns"]) == 1:
        labels = subset[dataset["id_columns"][0]].astype(str)
    else:
        labels = subset[dataset["id_columns"]].astype(str).agg(" · ".join, axis=1)

    figure = go.Figure()
    for source in selected_sources:
        column = f"{field}_{source}"
        values = pd.to_numeric(subset[column], errors="coerce")
        valid = values.notna()
        figure.add_trace(
            go.Scatter(
                x=np.flatnonzero(valid.to_numpy()),
                y=values.loc[valid].to_numpy(),
                text=labels.loc[valid].to_numpy(),
                mode="markers",
                name=source,
                marker={
                    "color": SOURCE_COLORS[source],
                    "symbol": SOURCE_SYMBOLS[source],
                    "size": 7,
                    "opacity": 0.75,
                },
                hovertemplate=f"%{{text}}<br>{field}: %{{y}}<extra></extra>",
            )
        )

    figure.update_layout(
        template="plotly_dark",
        title=f"{category} · {circuit_key} — {field}",
        xaxis_title="Element index",
        yaxis_title=field,
        height=560,
        legend_title="Source",
        margin={"t": 70, "r": 30, "b": 60, "l": 70},
    )
    return figure
