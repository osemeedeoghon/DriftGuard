"""Scope Drift Analysis — the index, its timeline, and its cost."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driftguard import projects

from .. import charts, components as ui, state, theme


def render(result) -> None:
    drift = result.drift
    estimates = result.estimates
    settings = result.settings
    unit = settings.size_unit.replace("_", " ")
    severity = theme.severity_for(drift.sdi)

    ui.page_header(
        "Scope drift analysis",
        "How far the project has moved from the agreed baseline, when it "
        "moved, and what the movement costs.",
    )

    st.markdown(
        f"**Definition.** `SDI = (size added − size removed) ÷ baseline size`. "
        f"Here that is "
        f"`({theme.fmt_number(drift.size_added, 1)} − {theme.fmt_number(drift.size_removed, 1)}) "
        f"÷ {theme.fmt_number(drift.baseline_size)}` = "
        f"**{theme.fmt_percent(drift.sdi, signed=True)}**."
    )

    ui.metric_row(
        [
            {
                "label": "Scope Drift Index",
                "value": theme.fmt_percent(drift.sdi, signed=True),
                "footnote": severity[0],
            },
            {
                "label": "Size added",
                "value": theme.fmt_number(drift.size_added, 1),
                "footnote": f"{unit}, gross",
            },
            {
                "label": "Size removed",
                "value": theme.fmt_number(drift.size_removed, 1),
                "footnote": f"{unit}, gross",
            },
            {
                "label": "Net change",
                "value": theme.fmt_signed(drift.net_change, 1),
                "delta_good": drift.net_change <= 0,
                "footnote": "drives the estimate",
            },
        ]
    )

    ui.assumption(
        f"Refinement tickets contribute {settings.refinement_weight:.2f} of their "
        "size as added scope. At 0 refinement is free; at 1 it is treated as "
        "entirely new work. Adjust it in the sidebar and every figure on this "
        "page moves."
    )

    # ---- Timeline ---------------------------------------------------------
    ui.section(
        "Drift over time",
        "Cumulative index as tickets arrive, ordered by creation date.",
    )
    timeline = result.timeline
    if timeline.empty:
        ui.empty_state(
            "No dated tickets",
            "The drift timeline needs a created_date column in the tickets "
            "file. The overall index above is unaffected.",
            icon="📅",
        )
    else:
        date_column = timeline.columns[0]
        table = timeline.rename(
            columns={
                date_column: "Date",
                "cumulative_added": "Cumulative added",
                "cumulative_removed": "Cumulative removed",
                "cumulative_sdi": "Cumulative SDI",
                "projected_size": "Projected size",
            }
        ).round(3)
        ui.chart(
            charts.drift_timeline(timeline, date_column),
            table=table,
            key="dr_timeline",
        )
        excluded = timeline.attrs.get("excluded_tickets", 0)
        if excluded:
            st.caption(
                f"{excluded} ticket(s) have no usable date and are absent from "
                "this chart, though they still count towards the overall index."
            )

    # ---- Composition ------------------------------------------------------
    ui.section("Composition of the change")
    left, right = st.columns(2)
    with left:
        ui.chart(
            charts.scope_waterfall(
                drift.baseline_size, drift.size_added, drift.size_removed,
                drift.revised_size, unit,
            ),
            key="dr_waterfall",
        )
    with right:
        ui.chart(
            charts.class_bar(drift.per_class),
            table=drift.per_class.round(1),
            key="dr_class",
        )

    # ---- Estimate impact --------------------------------------------------
    ui.section(
        "Effort and schedule impact",
        "COCOMO II Post-Architecture, published COCOMO II.2000 calibration. "
        "No local calibration is performed.",
    )

    revl_applied = 0.0
    if estimates.revised.size_ksloc:
        revl_applied = (
            estimates.revised.effective_ksloc / estimates.revised.size_ksloc - 1
        ) * 100

    comparison = pd.DataFrame(
        {
            "Quantity": [
                f"Size ({unit})",
                "Size (KSLOC)",
                "REVL applied (%)",
                "Effective KSLOC after REVL",
                "Effort (person-months)",
                "Duration (calendar months)",
                "Average staffing (FTE)",
            ],
            "Baseline": [
                drift.baseline_size,
                estimates.baseline.size_ksloc,
                settings.drivers.revl_percent,
                estimates.baseline.effective_ksloc,
                estimates.baseline.person_months,
                estimates.baseline.duration_months,
                estimates.baseline.average_staff,
            ],
            "Revised": [
                drift.revised_size,
                estimates.revised.size_ksloc,
                revl_applied,
                estimates.revised.effective_ksloc,
                estimates.revised.person_months,
                estimates.revised.duration_months,
                estimates.revised.average_staff,
            ],
        }
    )
    comparison["Change"] = comparison["Revised"] - comparison["Baseline"]

    left, right = st.columns([3, 2])
    with left:
        ui.data_table(comparison.round(2))
    with right:
        ui.metric_row(
            [
                {
                    "label": "Size change",
                    "value": theme.fmt_signed(estimates.size_change_percent, 1, "%"),
                },
            ]
        )
        ui.metric_row(
            [
                {
                    "label": "Effort change",
                    "value": theme.fmt_signed(estimates.effort_change_percent, 1, "%"),
                    "delta_good": estimates.effort_change_percent <= 0,
                },
            ]
        )
        ui.metric_row(
            [
                {
                    "label": "Duration change",
                    "value": theme.fmt_signed(estimates.duration_change_percent, 1, "%"),
                    "delta_good": estimates.duration_change_percent <= 0,
                },
            ]
        )

    ui.chart(
        charts.estimate_comparison(
            estimates.baseline.person_months,
            estimates.revised.person_months,
            estimates.baseline.duration_months,
            estimates.revised.duration_months,
        ),
        key="dr_estimates",
    )

    if estimates.effort_change_percent > estimates.size_change_percent > 0:
        st.info(
            f"Effort grows faster than size "
            f"({theme.fmt_signed(estimates.effort_change_percent, 1, '%')} against "
            f"{theme.fmt_signed(estimates.size_change_percent, 1, '%')}) for two "
            f"reasons: the COCOMO II exponent E = {estimates.revised.exponent_e:.4f} "
            "exceeds 1, which is the diseconomy of scale, and churn is charged "
            f"as {revl_applied:.1f}% REVL rework on top.",
            icon="📈",
        )

    _render_history(result)

    with st.expander("Why REVL excludes new scope"):
        st.markdown(
            """
REVL inflates size *inside* the effort equation. The revised size already
contains the newly added scope, so feeding the whole Scope Drift Index into
REVL would charge for that work twice — an earlier version of this tool did
exactly that and reported +246% effort for +76% size growth.

What REVL models is **rework**. DriftGuard therefore computes:

```
REVL% = 100 × (refinement-driven additions + removals) ÷ baseline size
```

* **Refinements** are requirements that moved after being agreed — the
  textbook definition of volatility.
* **Removals** are work that was planned, possibly partly built, then dropped:
  the size leaves the revised total but the effort spent does not come back.
* **New scope** is excluded, because the larger revised size already carries it.

Removals are counted at their full size, which assumes the work was built
before being dropped. Where removals were cancelled before starting, this
overstates rework — switch the REVL link off in the sidebar to see the pure
size effect.
"""
        )


def _render_history(result) -> None:
    """In-session scope history: record snapshots and compare them."""
    project_store = state.store()
    history = project_store.history_frame()

    ui.section(
        "Scope history",
        "Record a snapshot of the current reading, then compare it with an "
        "earlier one to see what moved and why.",
    )

    left, right = st.columns([2, 1])
    with left:
        label = st.text_input(
            "Snapshot label",
            value=f"Reading {len(history) + 1}",
            key="hist_label",
        )
        note = st.text_input(
            "Note (optional)",
            placeholder="e.g. after reviewing the mobile app request",
            key="hist_note",
        )
    with right:
        st.write("")
        st.write("")
        if st.button("Record snapshot", type="primary", use_container_width=True, key="hist_record"):
            if project_store.active is None:
                st.warning(
                    "No project is registered in this session, so there is "
                    "nothing to attach a snapshot to.",
                    icon="⚠️",
                )
            else:
                project_store.record(
                    projects.snapshot_from_result(result, label=label, note=note)
                )
                st.rerun()
        if not history.empty and st.button(
            "Clear history", use_container_width=True, key="hist_clear"
        ):
            project_store.clear_history()
            st.rerun()

    st.caption(
        "History is held in memory for this session only. Closing the tab "
        "discards it; durable history would need a datastore, which DriftGuard "
        "deliberately does not add."
    )

    if history.empty:
        ui.empty_state(
            "No snapshots recorded",
            "Record one now, change a classification or an assumption, then "
            "record another. The comparison below will show which of the two "
            "caused the index to move.",
            icon="🕑",
        )
        return

    ui.chart(
        charts.snapshot_history(history),
        table=history.round(3),
        key="dr_history",
    )

    snapshots = project_store.history()
    if len(snapshots) < 2:
        st.caption("Record a second snapshot to enable comparison.")
        return

    ui.section("Compare two snapshots")
    labels = [s.label for s in snapshots]
    columns = st.columns(2)
    first_label = columns[0].selectbox(
        "Earlier", labels, index=0, key="hist_first"
    )
    second_label = columns[1].selectbox(
        "Later", labels, index=len(labels) - 1, key="hist_second"
    )

    first = next(s for s in snapshots if s.label == first_label)
    second = next(s for s in snapshots if s.label == second_label)

    if first is second:
        st.caption("Pick two different snapshots to compare.")
        return

    changed = projects.assumptions_changed(first, second)
    if changed:
        st.warning(
            "**Assumptions changed between these snapshots:** "
            + "; ".join(changed)
            + ". Part of any movement in the index or the estimate is the "
            "result of that change, not of the project itself.",
            icon="⚠️",
        )
    else:
        st.success(
            "Assumptions are identical across these snapshots, so any "
            "movement reflects reclassification rather than retuning.",
            icon="✅",
        )

    comparison = projects.compare_snapshots(first, second)
    ui.data_table(comparison.round(3))

    moved = second.sdi - first.sdi
    if abs(moved) > 1e-9:
        st.markdown(
            f"The Scope Drift Index moved **{moved * 100:+.1f} percentage "
            f"points** between *{first.label}* and *{second.label}*."
        )
    if second.contributors:
        st.markdown("##### Top contributors at the later snapshot")
        ui.data_table(
            pd.DataFrame(
                second.contributors, columns=["Ticket", "Net scope change"]
            ).round(1)
        )
