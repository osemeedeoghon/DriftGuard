"""Reusable presentation components for the DriftGuard dashboard.

Every component here is pure presentation. None of them computes an analysis
figure; they render values handed to them. Where a number rests on an
assumption, the component is built so the assumption travels with it.
"""

from __future__ import annotations

import html
from contextlib import contextmanager
from typing import Iterable, Iterator

import pandas as pd
import streamlit as st

from . import theme


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------
# Page furniture
# --------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "") -> None:
    """Title block at the top of a page."""
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(
            f"<div class='dg-section-sub' style='margin:-0.5rem 0 1.2rem 0;"
            f"max-width:60rem'>{_esc(subtitle)}</div>",
            unsafe_allow_html=True,
        )


def section(title: str, subtitle: str = "") -> None:
    """Rule-and-title separator between blocks on a page."""
    sub = (
        f"<div class='dg-section-sub'>{_esc(subtitle)}</div>" if subtitle else ""
    )
    st.markdown(
        f"<div class='dg-section'><div class='dg-section-title'>{_esc(title)}"
        f"</div>{sub}</div>",
        unsafe_allow_html=True,
    )


def severity_pill(label: str, colour: str, icon: str) -> str:
    """Inline severity badge. Icon and label always accompany the colour."""
    return (
        f"<span class='dg-pill' style='background:{colour}1a;"
        f"border-color:{colour}59;color:{colour}'>"
        f"<span aria-hidden='true'>{_esc(icon)}</span>{_esc(label)}</span>"
    )


# --------------------------------------------------------------------------
# Metric cards
# --------------------------------------------------------------------------
def metric_card(
    label: str,
    value: str,
    *,
    delta: str | None = None,
    delta_good: bool | None = None,
    footnote: str | None = None,
    help_text: str | None = None,
) -> None:
    """A single stat tile.

    ``delta_good`` controls the delta's ink: True renders it as a positive
    outcome, False as a negative one, None as neutral. It is passed explicitly
    rather than inferred from sign, because in this domain a *rising* number is
    usually bad news.
    """
    hint = ""
    if help_text:
        hint = (
            f"<span title='{_esc(help_text)}' style='cursor:help;"
            f"color:{theme.INK_MUTED};font-weight:400'>&#9432;</span>"
        )

    delta_html = ""
    if delta:
        if delta_good is True:
            colour = theme.TEXT_GOOD
        elif delta_good is False:
            colour = theme.TEXT_BAD
        else:
            colour = theme.INK_SECONDARY
        delta_html = (
            f"<div class='dg-metric-delta' style='color:{colour}'>{_esc(delta)}</div>"
        )

    foot_html = (
        f"<div class='dg-metric-foot'>{_esc(footnote)}</div>" if footnote else ""
    )

    st.markdown(
        f"<div class='dg-card'>"
        f"<div class='dg-metric-label'>{_esc(label)}{hint}</div>"
        f"<div class='dg-metric-value'>{_esc(value)}</div>"
        f"{delta_html}{foot_html}</div>",
        unsafe_allow_html=True,
    )


def metric_row(cards: list[dict]) -> None:
    """A row of stat tiles. Each dict is forwarded to :func:`metric_card`."""
    if not cards:
        return
    for column, card in zip(st.columns(len(cards)), cards):
        with column:
            metric_card(**card)


def hero_figure(
    value: str,
    label: str,
    *,
    severity: tuple[str, str, str] | None = None,
    caption: str = "",
) -> None:
    """The one number a page leads with."""
    pill = ""
    if severity:
        pill = "<div style='margin-top:0.55rem'>" + severity_pill(*severity) + "</div>"
    cap = (
        f"<div class='dg-metric-foot' style='margin-top:0.6rem;max-width:34rem'>"
        f"{_esc(caption)}</div>"
        if caption
        else ""
    )
    st.markdown(
        f"<div class='dg-hero'>"
        f"<div class='dg-metric-label'>{_esc(label)}</div>"
        f"<div class='dg-hero-value'>{_esc(value)}</div>"
        f"{pill}{cap}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Notes, empty states, errors
# --------------------------------------------------------------------------
def assumption(text: str) -> None:
    """Flag a number that rests on a user-supplied or default assumption."""
    st.markdown(
        f"<div class='dg-note'><strong>Assumption.</strong> {_esc(text)}</div>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str, *, icon: str = "📂") -> None:
    """Shown when a page has nothing to display, explaining what to do next."""
    st.markdown(
        f"<div class='dg-empty'>"
        f"<div style='font-size:2rem;line-height:1' aria-hidden='true'>{icon}</div>"
        f"<div class='dg-empty-title' style='margin-top:0.6rem'>{_esc(title)}</div>"
        f"<div class='dg-empty-body'>{_esc(body)}</div></div>",
        unsafe_allow_html=True,
    )


def no_evidence(message: str) -> None:
    """State plainly that a claim cannot be supported, rather than guessing."""
    st.info(message, icon="🔍")


@contextmanager
def working(message: str = "Computing…") -> Iterator[None]:
    """Loading indicator for a block of work."""
    with st.spinner(message):
        yield


# --------------------------------------------------------------------------
# Chart + table pairing
# --------------------------------------------------------------------------
def chart(figure, *, table: pd.DataFrame | None = None, table_label: str = "View as table", key: str | None = None) -> None:
    """Render a Plotly figure, with the table view the relief rule requires.

    Two of the four class colours sit below 3:1 contrast against the chart
    surface. That is only permissible when the chart carries direct labels or
    an accompanying table view, so this helper makes the table view the default
    way charts are rendered.
    """
    st.plotly_chart(figure, use_container_width=True, key=key)
    if table is not None and not table.empty:
        with st.expander(table_label):
            st.dataframe(table, use_container_width=True, hide_index=True)


def percent_column(label: str | None = None, decimals: int = 1):
    """A table column rendered as a percentage.

    Streamlit applies the printf format to the *raw* cell value, so a column
    holding 0.956 and formatted "%.1f%%" renders as "1.0%", not "95.6%". Pair
    this with :func:`scale_to_percent` so the stored value is already 0-100.
    """
    return st.column_config.NumberColumn(label, format=f"%.{decimals}f%%")


def scale_to_percent(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Multiply 0-1 fraction columns by 100 for percentage display."""
    scaled = frame.copy()
    for column in columns:
        if column in scaled.columns:
            scaled[column] = pd.to_numeric(scaled[column], errors="coerce") * 100.0
    return scaled


def data_table(
    frame: pd.DataFrame,
    *,
    height: int | None = None,
    column_config: dict | None = None,
    hide_index: bool = True,
) -> None:
    """Standard table rendering."""
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=hide_index,
        height=height,
        column_config=column_config or {},
    )


def download_row(items: Iterable[tuple[str, bytes, str, str]]) -> None:
    """A row of download buttons: (label, payload, filename, mime)."""
    items = list(items)
    if not items:
        return
    for column, (label, payload, filename, mime) in zip(st.columns(len(items)), items):
        column.download_button(
            label,
            payload,
            file_name=filename,
            mime=mime,
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------
def quality_summary(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per-column completeness and type summary for an uploaded dataset."""
    if frame.empty:
        return pd.DataFrame(
            columns=["Column", "Type", "Non-null", "Missing", "Missing %", "Distinct"]
        )

    rows = []
    total = len(frame)
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        if series.dtype == object:
            blank = int(
                series.fillna("").astype(str).str.strip().eq("").sum()
            )
            missing = max(missing, blank)
        try:
            distinct = int(series.nunique(dropna=True))
        except TypeError:  # unhashable cell contents, e.g. a list column
            distinct = -1
        rows.append(
            {
                "Column": str(column),
                "Type": str(series.dtype),
                "Non-null": total - missing,
                "Missing": missing,
                # Stored 0-100, matching the "%.1f%%" column format below.
                "Missing %": (missing / total * 100.0) if total else 0.0,
                "Distinct": distinct if distinct >= 0 else None,
            }
        )
    summary = pd.DataFrame(rows)
    summary.attrs["dataset"] = name
    summary.attrs["rows"] = total
    return summary


def quality_panel(frame: pd.DataFrame, name: str) -> None:
    """Row count, completeness and per-column detail for one dataset."""
    summary = quality_summary(frame, name)
    if summary.empty:
        empty_state("No rows", f"{name} contains no rows to summarise.")
        return

    total_cells = len(frame) * len(frame.columns)
    missing_cells = int(summary["Missing"].sum())
    completeness = 1.0 - (missing_cells / total_cells if total_cells else 0.0)
    incomplete = summary.loc[summary["Missing"] > 0, "Column"].tolist()

    metric_row(
        [
            {"label": "Rows", "value": theme.fmt_number(len(frame))},
            {"label": "Columns", "value": theme.fmt_number(len(frame.columns))},
            {
                "label": "Completeness",
                "value": theme.fmt_percent(completeness),
                "footnote": f"{missing_cells:,} empty cells of {total_cells:,}",
            },
            {
                "label": "Columns with gaps",
                "value": theme.fmt_number(len(incomplete)),
                "footnote": ", ".join(incomplete[:3]) + ("…" if len(incomplete) > 3 else "")
                if incomplete
                else "none",
            },
        ]
    )

    data_table(
        summary,
        column_config={
            # quality_summary already stores this 0-100.
            "Missing %": percent_column("Missing %"),
        },
    )
