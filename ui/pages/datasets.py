"""Dataset Explorer — inspect the loaded data and its quality."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driftguard import data

from .. import charts, components as ui, theme


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop internal/list columns that do not belong in a data table."""
    drop = [c for c in ("text", "alternative_matches") if c in frame.columns]
    display = frame.drop(columns=drop)
    for column in display.columns:
        if display[column].dtype == object and display[column].map(
            lambda v: isinstance(v, (list, tuple))
        ).any():
            display[column] = display[column].map(
                lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v
            )
    return display


def render(result) -> None:
    baseline = st.session_state.baseline
    tickets = st.session_state.tickets
    labelled = st.session_state.labelled
    unit = result.settings.size_unit.replace("_", " ")

    ui.page_header(
        "Dataset explorer",
        "The three inputs DriftGuard works from, their schema, completeness "
        "and contents.",
    )

    datasets: dict[str, pd.DataFrame] = {
        "Baseline requirements": baseline,
        "Project tickets": tickets,
    }
    if labelled is not None:
        datasets["Labelled test set"] = labelled

    ui.metric_row(
        [
            {
                "label": "Baseline requirements",
                "value": theme.fmt_number(len(baseline)),
                "footnote": f"{theme.fmt_number(baseline['size'].sum())} {unit} agreed",
            },
            {
                "label": "Project tickets",
                "value": theme.fmt_number(len(tickets)),
                "footnote": f"{theme.fmt_number(tickets['size'].sum())} {unit} raised",
            },
            {
                "label": "Labelled for evaluation",
                "value": theme.fmt_number(len(labelled)) if labelled is not None else "—",
                "footnote": "manual ground truth" if labelled is not None else "none supplied",
            },
            {
                "label": "Source",
                "value": "Sample" if st.session_state.source == "sample" else "Uploaded",
                "footnote": st.session_state.get("project_name", ""),
            },
        ]
    )

    warnings = st.session_state.get("load_warnings", []) + list(result.warnings)
    if warnings:
        ui.section("Data quality warnings", "Raised while loading or analysing.")
        for warning in warnings:
            st.warning(warning, icon="ℹ️")
    else:
        ui.section("Data quality warnings")
        st.success("No warnings were raised while loading this project.", icon="✅")

    # ---- Reference vs current -------------------------------------------
    ui.section(
        "Reference versus current",
        "The agreed baseline is the reference; the tickets raised against it "
        "are the current state. Sizes are compared on the same scale.",
    )
    distribution = charts.size_distribution(baseline["size"], tickets["size"], unit)
    comparison = pd.DataFrame(
        {
            "Dataset": ["Baseline requirements", "Project tickets"],
            "Items": [len(baseline), len(tickets)],
            f"Total {unit}": [baseline["size"].sum(), tickets["size"].sum()],
            "Median": [baseline["size"].median(), tickets["size"].median()],
            "Min": [baseline["size"].min(), tickets["size"].min()],
            "Max": [baseline["size"].max(), tickets["size"].max()],
        }
    ).round(1)
    ui.chart(distribution, table=comparison, key="ds_sizes")
    st.caption(
        "Tickets are typically smaller than requirements because a requirement "
        "is usually delivered by several tickets. A large gap in totals is the "
        "scope drift itself, quantified on the Scope Drift Analysis page."
    )

    # ---- Per-dataset detail ----------------------------------------------
    ui.section("Dataset detail", "Schema, completeness and full contents.")
    chosen = st.selectbox("Dataset", list(datasets))
    frame = datasets[chosen]

    ui.quality_panel(frame, chosen)

    st.markdown("##### Contents")
    display = _display_frame(frame)
    search = st.text_input(
        "Filter rows", placeholder="Search any column…", key=f"ds_search_{chosen}"
    )
    if search:
        mask = display.astype(str).apply(
            lambda column: column.str.contains(search, case=False, na=False)
        ).any(axis=1)
        display = display[mask]
        st.caption(f"{len(display):,} of {len(frame):,} rows match.")

    if display.empty:
        ui.empty_state("No matching rows", "Nothing in this dataset matches that filter.", icon="🔍")
    else:
        ui.data_table(display, height=460)

    st.download_button(
        f"Download {chosen.lower()} (CSV)",
        data.frame_to_csv_bytes(display),
        file_name=f"driftguard_{chosen.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
