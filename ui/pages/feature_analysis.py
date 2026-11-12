"""Feature-Level Analysis.

"Feature" means what it means to this model: the TF-IDF vocabulary and the
similarity score derived from it. This page shows how those features behave
across classes and inside a single prediction.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driftguard import features
from driftguard.matching import RequirementMatcher

from .. import charts, components as ui, theme


def render(result) -> None:
    baseline = st.session_state.baseline
    tickets = st.session_state.tickets
    classified = result.classified
    unit = result.settings.size_unit.replace("_", " ")

    ui.page_header(
        "Feature-level analysis",
        "The model's features are the TF-IDF vocabulary and the cosine "
        "similarity built from it. This page shows how they separate the "
        "classes, and which terms decide an individual match.",
    )

    st.info(
        "This application classifies text against a requirements baseline. It "
        "has no tabular feature matrix, so there are no population-stability "
        "or Kolmogorov–Smirnov feature-drift statistics to report — those would "
        "be fabricated here. What follows are the real features.",
        icon="ℹ️",
    )

    # ---- Similarity by class ---------------------------------------------
    ui.section(
        "Similarity distribution by class",
        "How far each class sits from its matched baseline requirement.",
    )
    summary = features.class_similarity_summary(classified)
    ui.chart(
        charts.similarity_distribution(classified, "final_class"),
        table=summary.round(3).rename(
            columns={
                "class": "Class",
                "tickets": "Tickets",
                "min": "Min",
                "median": "Median",
                "mean": "Mean",
                "max": "Max",
            }
        ),
        key="fa_simdist",
    )

    thresholds = result.settings.thresholds
    st.caption(
        f"The weak-match threshold is {thresholds.weak_match:.2f} and the "
        f"strong-match threshold {thresholds.strong_match:.2f}. Similarity "
        "separates *related to the baseline* from *unrelated*; it cannot "
        "separate implementing a requirement from rewriting one, which is what "
        "the keyword cues decide."
    )

    # ---- Threshold sensitivity -------------------------------------------
    ui.section(
        "Where the thresholds bite",
        "Tickets near a boundary are the ones a reviewer should check.",
    )
    near_weak = classified.loc[
        (classified["similarity"] - thresholds.weak_match).abs() < 0.05
    ]
    ui.metric_row(
        [
            {
                "label": "Below weak threshold",
                "value": theme.fmt_number((classified["similarity"] < thresholds.weak_match).sum()),
                "footnote": "treated as unrelated to the baseline",
            },
            {
                "label": "In the uncertain band",
                "value": theme.fmt_number(
                    (
                        (classified["similarity"] >= thresholds.weak_match)
                        & (classified["similarity"] < thresholds.strong_match)
                    ).sum()
                ),
                "footnote": "related, but not a confident match",
            },
            {
                "label": "Confident matches",
                "value": theme.fmt_number((classified["similarity"] >= thresholds.strong_match).sum()),
                "footnote": "at or above strong threshold",
            },
            {
                "label": "Within 0.05 of the weak line",
                "value": theme.fmt_number(len(near_weak)),
                "footnote": "would flip if it moved",
                "delta_good": None,
            },
        ]
    )
    if not near_weak.empty:
        ui.data_table(
            near_weak[
                [c for c in ("ticket_id", "title", "similarity", "predicted_class", "final_class") if c in near_weak.columns]
            ].sort_values("similarity").rename(
                columns={
                    "ticket_id": "Ticket",
                    "title": "Title",
                    "similarity": "Similarity",
                    "predicted_class": "Model said",
                    "final_class": "Final",
                }
            ),
            column_config={"Similarity": st.column_config.NumberColumn(format="%.3f")},
        )

    # ---- Baseline coverage ------------------------------------------------
    ui.section(
        "Baseline coverage",
        "Tickets attached to each agreed requirement. Only In-scope and "
        "Refinement tickets count as covering one.",
    )
    coverage = features.requirement_coverage(classified, baseline)
    ui.chart(
        charts.coverage_bars(coverage, unit),
        table=coverage[["requirement_id", "title", "size", "tickets"]].rename(
            columns={
                "requirement_id": "Requirement",
                "title": "Title",
                "size": "Size",
                "tickets": "Tickets",
            }
        ),
        key="fa_coverage",
    )
    uncovered = coverage.loc[coverage["tickets"] == 0]
    if uncovered.empty:
        st.success("Every baseline requirement has at least one ticket matched to it.", icon="✅")
    else:
        st.warning(
            f"{len(uncovered)} requirement(s) have no ticket matched to them. "
            "That may mean unstarted work, or a ticket the matcher failed to "
            "connect. Both are worth checking.",
            icon="🔍",
        )
        ui.data_table(uncovered[["requirement_id", "title", "size"]])

    # ---- Single prediction ------------------------------------------------
    ui.section(
        "Inside one prediction",
        "The terms shared by a ticket and its matched requirement, ranked by "
        "the product of their TF-IDF weights — exactly what the cosine "
        "similarity sums over.",
    )

    ticket_ids = list(classified["ticket_id"].astype(str))
    selected = st.selectbox("Ticket", ticket_ids, key="fa_ticket")
    row = classified.loc[classified["ticket_id"].astype(str) == selected].iloc[0]

    left, right = st.columns([2, 3])
    with left:
        ui.metric_row(
            [
                {
                    "label": "Similarity",
                    "value": f"{row['similarity']:.3f}",
                    "footnote": f"to {row['matched_requirement_id']}",
                },
            ]
        )
        ui.metric_row(
            [
                {
                    "label": "Margin over runner-up",
                    "value": f"{row.get('match_margin', 0):.3f}",
                    "footnote": f"runner-up {row.get('runner_up_id', '—')}",
                    "help_text": "A small margin means the match was nearly a tie.",
                },
            ]
        )
        st.markdown(f"**{row.get('title', '')}**")
        st.caption(str(row.get("description", "")))

    with right:
        with ui.working("Recomputing term weights…"):
            matcher = RequirementMatcher().fit(baseline, tickets)
            terms = matcher.shared_term_weights(
                str(row.get("text", "")), str(row["matched_requirement_id"])
            )
        if terms.empty:
            ui.no_evidence(
                "This ticket shares no distinctive vocabulary with its nearest "
                "requirement, which is why the similarity is near zero."
            )
        else:
            ui.chart(
                charts.term_weights(
                    terms, f"Terms driving {selected} → {row['matched_requirement_id']}"
                ),
                table=terms.round(4),
                key="fa_terms",
            )

    alternatives = row.get("alternative_matches")
    if isinstance(alternatives, list) and len(alternatives) > 1:
        st.markdown("##### Other candidate requirements")
        ui.data_table(
            pd.DataFrame(alternatives, columns=["Requirement", "Similarity"]).round(3),
            column_config={"Similarity": st.column_config.NumberColumn(format="%.3f")},
        )

    st.markdown("##### Classification rationale")
    st.success(row["rationale"])
