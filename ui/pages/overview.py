"""Executive Overview — the headline scope and cost position."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driftguard import features

from .. import charts, components as ui, theme


def render(result) -> None:
    drift = result.drift
    estimates = result.estimates
    unit = result.settings.size_unit.replace("_", " ")
    severity = theme.severity_for(drift.sdi)

    ui.page_header(
        "Executive overview",
        "Where the project stands against the agreed baseline, and what the "
        "difference costs in effort and schedule.",
    )

    left, right = st.columns([1, 2])
    with left:
        ui.hero_figure(
            theme.fmt_percent(drift.sdi, signed=True),
            "Scope Drift Index",
            severity=severity,
            caption=(
                f"Net change of {theme.fmt_signed(drift.net_change, 0)} {unit} "
                f"against a {theme.fmt_number(drift.baseline_size)} {unit} baseline. "
                "Severity bands are a reading convention for this dashboard, "
                "not an industry standard."
            ),
        )
    with right:
        ui.metric_row(
            [
                {
                    "label": "Baseline scope",
                    "value": f"{theme.fmt_number(drift.baseline_size)}",
                    "footnote": f"{unit} agreed",
                },
                {
                    "label": "Revised scope",
                    "value": f"{theme.fmt_number(drift.revised_size)}",
                    "delta": theme.fmt_signed(drift.net_change, 0, f" {unit}"),
                    "delta_good": drift.net_change <= 0,
                },
                {
                    "label": "Tickets analysed",
                    "value": theme.fmt_number(len(result.classified)),
                    "footnote": f"{int(result.classified['reviewed'].sum())} reviewed by a human",
                },
            ]
        )
        st.write("")
        ui.metric_row(
            [
                {
                    "label": "Effort",
                    "value": theme.fmt_number(estimates.revised.person_months, 1),
                    "delta": theme.fmt_signed(estimates.delta_person_months, 1, " PM"),
                    "delta_good": estimates.delta_person_months <= 0,
                    "footnote": "person-months of work",
                    "help_text": "Effort, not duration. These are different quantities.",
                },
                {
                    "label": "Duration",
                    "value": theme.fmt_number(estimates.revised.duration_months, 1),
                    "delta": theme.fmt_signed(estimates.delta_duration_months, 1, " mo"),
                    "delta_good": estimates.delta_duration_months <= 0,
                    "footnote": "calendar months elapsed",
                    "help_text": "Calendar duration, not effort. Adding staff does not convert one into the other.",
                },
                {
                    "label": "Average staffing",
                    "value": theme.fmt_number(estimates.revised.average_staff, 1),
                    "delta": theme.fmt_signed(
                        estimates.revised.average_staff - estimates.baseline.average_staff, 1
                    ),
                    "delta_good": None,
                    "footnote": "full-time equivalent",
                },
            ]
        )

    # ---- Headline read -------------------------------------------------
    label, colour, icon = severity
    if drift.sdi > 0.25:
        st.error(
            f"**{label}.** Net scope growth of {theme.fmt_percent(drift.sdi)} "
            "against the agreed baseline. The tickets responsible are listed below.",
            icon="🚨",
        )
    elif drift.sdi > 0.10:
        st.warning(
            f"**{label}.** Net scope growth of {theme.fmt_percent(drift.sdi)}.",
            icon="⚠️",
        )
    elif drift.sdi < -0.10:
        st.info(
            f"**{label}.** Net de-scoping of {theme.fmt_percent(abs(drift.sdi))}.",
            icon="📉",
        )
    else:
        st.success(f"**{label}.** Scope is close to the agreed baseline.", icon="✅")

    rate = result.settings.monthly_labour_rate
    if rate:
        st.caption(
            f"At the supplied rate of {theme.fmt_number(rate)} per person-month, the "
            f"revised effort is an estimated {theme.fmt_number(estimates.revised.labour_cost(rate))}. "
            "This is a derived labour-cost estimate, not an actual or committed "
            "cost, and excludes tooling, licences, overhead and contingency."
        )

    # ---- Composition ---------------------------------------------------
    ui.section(
        "Where the drift comes from",
        "New scope adds its full size; refinement adds a fraction set by the "
        "reviewer; removals subtract.",
    )
    left, right = st.columns(2)
    with left:
        ui.chart(
            charts.class_bar(drift.per_class),
            table=drift.per_class.rename(
                columns={
                    "classification": "Class",
                    "tickets": "Tickets",
                    "size": "Size",
                    "added": "Added",
                    "removed": "Removed",
                }
            ).round(1),
            key="ov_class",
        )
    with right:
        waterfall_table = pd.DataFrame(
            {
                "Step": ["Baseline", "Added", "Removed", "Revised"],
                unit.capitalize(): [
                    drift.baseline_size,
                    drift.size_added,
                    -drift.size_removed,
                    drift.revised_size,
                ],
            }
        )
        ui.chart(
            charts.scope_waterfall(
                drift.baseline_size, drift.size_added, drift.size_removed,
                drift.revised_size, unit,
            ),
            table=waterfall_table.round(1),
            key="ov_waterfall",
        )

    # ---- Attribution ---------------------------------------------------
    ui.section("Tickets driving the most drift", "Ranked by absolute net effect on scope.")
    top = drift.top_contributors(10)
    if top.empty:
        ui.empty_state("No drift to attribute", "No ticket changed the agreed scope.")
    else:
        ui.chart(
            charts.contributor_bars(top, unit, title=None),
            table=top[
                [c for c in ("ticket_id", "title", "final_class", "final_size", "net_contribution") if c in top.columns]
            ].rename(
                columns={
                    "ticket_id": "Ticket",
                    "title": "Title",
                    "final_class": "Class",
                    "final_size": "Size",
                    "net_contribution": "Net change",
                }
            ).round(1),
            key="ov_contrib",
        )

    # ---- Scope composition counts ---------------------------------------
    ui.section(
        "Scope position in detail",
        "Every figure below is counted from the classified tickets, using the "
        "reviewer's final classification where one was recorded.",
    )
    per_class = drift.per_class.set_index("classification")

    def tickets_in(name: str) -> int:
        return int(per_class.loc[name, "tickets"]) if name in per_class.index else 0

    ui.metric_row(
        [
            {
                "label": "Baseline requirements",
                "value": theme.fmt_number(len(st.session_state.baseline)),
                "footnote": f"{theme.fmt_number(drift.baseline_size)} {unit} agreed",
            },
            {
                "label": "In-scope tickets",
                "value": theme.fmt_number(tickets_in("In-scope")),
                "footnote": "deliver agreed work; add no scope",
            },
            {
                "label": "Refinement tickets",
                "value": theme.fmt_number(tickets_in("Refinement")),
                "footnote": f"{theme.fmt_number(per_class.loc['Refinement', 'added'] if 'Refinement' in per_class.index else 0, 1)} {unit} added",
            },
            {
                "label": "New-scope tickets",
                "value": theme.fmt_number(tickets_in("New scope")),
                "footnote": f"{theme.fmt_number(per_class.loc['New scope', 'added'] if 'New scope' in per_class.index else 0, 0)} {unit} added",
                "delta_good": False if tickets_in("New scope") else None,
            },
            {
                "label": "Removals",
                "value": theme.fmt_number(tickets_in("Removal")),
                "footnote": f"{theme.fmt_number(drift.size_removed, 0)} {unit} withdrawn",
            },
        ]
    )

    pending = int((~result.classified["reviewed"].astype(bool)).sum())
    ui.metric_row(
        [
            {
                "label": "Scope growth",
                "value": theme.fmt_percent(drift.sdi, signed=True),
                "footnote": "net, against the agreed baseline",
            },
            {
                "label": "Pending human review",
                "value": theme.fmt_number(pending),
                "footnote": f"of {len(result.classified)} tickets",
                "delta_good": None,
            },
            {
                "label": "Estimated effort",
                "value": f"{theme.fmt_number(estimates.revised.person_months, 1)} PM",
                "footnote": "person-months of work",
            },
            {
                "label": "Estimated schedule",
                "value": f"{theme.fmt_number(estimates.revised.duration_months, 1)} mo",
                "footnote": "calendar months elapsed",
            },
            {
                "label": "Estimated cost",
                "value": theme.fmt_number(estimates.revised.labour_cost(rate), 0)
                if rate
                else "no rate supplied",
                "footnote": "derived estimate, labour only" if rate else "DriftGuard assumes no rate",
            },
        ]
    )

    # ---- Risk indicators --------------------------------------------------
    ui.section(
        "Project risk indicators",
        "Measured values with a severity reading. The bands are a convention "
        "for this dashboard, not thresholds from any standard.",
    )
    indicators = features.risk_indicators(result)
    worst = indicators.loc[indicators["Level"] == "Critical"]
    if not worst.empty:
        st.error(
            "**Critical:** " + "; ".join(worst["Indicator"]) + ".",
            icon="🚨",
        )
    ui.data_table(indicators)

    # ---- Recommendations ---------------------------------------------------
    ui.section(
        "Key findings and recommended actions",
        "Each follows from a measured figure. None is a prediction.",
    )
    for item in features.recommendations(result):
        colour = {
            "High": theme.STATUS_CRITICAL,
            "Medium": theme.STATUS_SERIOUS,
            "Low": theme.INK_MUTED,
        }[item["priority"]]
        st.markdown(
            f"<div class='dg-card' style='margin-bottom:0.6rem;border-left:3px solid {colour}'>"
            f"<div class='dg-metric-label' style='color:{colour}'>{item['priority']} priority</div>"
            f"<div style='font-weight:600;margin:0.15rem 0 0.3rem 0'>{item['action']}</div>"
            f"<div class='dg-metric-foot'>{item['because']}</div></div>",
            unsafe_allow_html=True,
        )

    # ---- Model health ---------------------------------------------------
    health = features.model_health(result.classified)
    ui.section(
        "Model health",
        "How much of this classification a human should be looking at. These "
        "are counts and shares, not a quality score.",
    )
    ui.metric_row(
        [
            {
                "label": "Mean confidence",
                "value": theme.fmt_percent(health["mean_confidence"], 0),
                "footnote": "across all tickets",
            },
            {
                "label": "Low confidence",
                "value": theme.fmt_number(health["low_confidence"]),
                "footnote": f"{theme.fmt_percent(health['low_confidence_share'], 0)} flagged for review",
                "delta_good": None,
            },
            {
                "label": "Human reviewed",
                "value": theme.fmt_number(health["reviewed"]),
                "footnote": f"{health['overturned']} classification(s) overturned",
            },
            {
                "label": "Median similarity",
                "value": f"{health['median_similarity']:.2f}",
                "footnote": "ticket to matched requirement",
            },
        ]
    )
    st.caption(
        "Nothing here says the model is accurate. Accuracy requires a labelled "
        "test set and is reported on the Model Performance page."
    )
