"""Sidebar control panel: data sources, model settings and assumptions."""

from __future__ import annotations

import streamlit as st

from driftguard import __version__, data, synthetic
from driftguard.cocomo import (
    DRIVER_NAMES,
    EFFORT_MULTIPLIERS,
    RATING_LABELS,
    SCALE_FACTORS,
    CocomoDrivers,
    SizeConversion,
)
from driftguard.config import MatchingThresholds
from driftguard.data import DataValidationError
from driftguard.pipeline import AnalysisSettings

from . import state

SIZE_UNITS = {
    "story_points": "Story points",
    "function_points": "Function points",
    "sloc": "Source lines of code",
}


def _load_uploads() -> list[str]:
    """Handle the three file uploaders. Returns accumulated warnings."""
    warnings: list[str] = []

    baseline_file = st.file_uploader(
        "Baseline requirements",
        type=["csv", "txt", "md"],
        help="CSV with requirement_id and description, or a TXT file with one requirement per line.",
    )
    tickets_file = st.file_uploader(
        "Project tickets",
        type=["csv"],
        help="CSV with ticket_id and description. Add size and created_date for full analysis.",
    )
    labelled_file = st.file_uploader(
        "Labelled test set (optional)",
        type=["csv"],
        help="Tickets with a true_class column. Used only to measure classifier quality.",
    )

    if baseline_file is not None:
        try:
            frame, report = data.load_baseline(baseline_file, filename=baseline_file.name)
            st.session_state.baseline = frame
            st.session_state.source = "upload"
            warnings += report.warnings
        except DataValidationError as exc:
            st.error(f"**Baseline rejected.** {exc}", icon="⚠️")

    if tickets_file is not None:
        try:
            frame, report = data.load_tickets(tickets_file)
            st.session_state.tickets = frame
            st.session_state.source = "upload"
            warnings += report.warnings
        except DataValidationError as exc:
            st.error(f"**Tickets rejected.** {exc}", icon="⚠️")

    if labelled_file is not None:
        try:
            frame, report = data.load_tickets(labelled_file)
            if "true_class" not in frame.columns:
                st.error(
                    "**Labelled set rejected.** It needs a `true_class` column "
                    "holding your manual labels.",
                    icon="⚠️",
                )
            else:
                st.session_state.labelled = frame
                warnings += report.warnings
        except DataValidationError as exc:
            st.error(f"**Labelled set rejected.** {exc}", icon="⚠️")

    return warnings


def render() -> AnalysisSettings:
    """Draw the sidebar and return the settings it describes."""
    st.markdown(
        "<div style='font-size:1.35rem;font-weight:650;letter-spacing:-0.02em;"
        "color:#fff'>DriftGuard</div>"
        "<div style='font-size:0.76rem;color:#94a3b8;margin-top:0.15rem'>"
        f"v{__version__} · Scope drift analytics</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.7rem;color:#64748b;margin:0.55rem 0 0.9rem 0'>"
        "Runs fully offline · no LLM API, cloud database or paid service</div>",
        unsafe_allow_html=True,
    )

    # ---- Project switching -------------------------------------------
    project_store = state.store()
    if len(project_store) > 1:
        with st.expander("Projects", expanded=True):
            names = project_store.names()
            ids = list(names)
            current = project_store.active_id
            chosen = st.selectbox(
                "Active project",
                ids,
                index=ids.index(current) if current in ids else 0,
                format_func=lambda pid: names[pid],
                key="sidebar_project_select",
            )
            if chosen != current:
                state.switch_to(chosen)
                st.rerun()
            st.caption(
                f"{len(project_store)} projects in this session. Projects and "
                "their history are held in memory only and are lost when the "
                "session ends."
            )

    # ---- Data --------------------------------------------------------
    with st.expander("Project data", expanded=not state.has_project()):
        if st.button(
            "Load sample project",
            use_container_width=True,
            type="primary",
            key="sidebar_load_sample",
        ):
            state.load_sample_project()
            st.rerun()

        warnings = _load_uploads()
        if warnings:
            st.session_state.load_warnings = warnings

        if state.has_project():
            st.caption(
                f"Loaded: {len(st.session_state.baseline)} requirements, "
                f"{len(st.session_state.tickets)} tickets"
                + (", labelled set present" if state.has_labels() else "")
            )
            if st.button(
                "Clear project", use_container_width=True, key="sidebar_clear_project"
            ):
                state.reset_project()
                st.rerun()

    # ---- Synthetic test data -------------------------------------------
    with st.expander("Generate test data", expanded=False):
        st.caption(
            "Controlled synthetic projects for exercising the application. "
            "Everything generated is clearly marked as synthetic and is not "
            "evidence about any real project."
        )
        scenario = st.selectbox(
            "Scenario",
            list(synthetic.SCENARIOS),
            format_func=lambda key: key.replace("_", " ").capitalize(),
            key="syn_scenario",
        )
        st.caption(synthetic.SCENARIOS[scenario])
        columns = st.columns(2)
        requirements = columns[0].number_input(
            "Requirements", min_value=3, max_value=60, value=15, key="syn_reqs"
        )
        tickets = columns[1].number_input(
            "Tickets", min_value=4, max_value=200, value=40, key="syn_tickets"
        )
        seed = st.number_input("Seed", min_value=0, value=7, key="syn_seed")
        if st.button("Generate project", use_container_width=True, key="syn_generate"):
            state.load_synthetic_project(
                scenario,
                seed=int(seed),
                requirements=int(requirements),
                tickets=int(tickets),
            )
            st.rerun()

    # ---- Classifier --------------------------------------------------
    with st.expander("Classifier", expanded=False):
        choice = st.radio(
            "Model",
            ["Rule-based (explainable)", "Logistic regression (learned)"],
            help=(
                "The rule-based model explains every decision. The learned "
                "model needs labelled training data and, on a single project, "
                "tends to overfit."
            ),
        )
        classifier = "ml" if choice.startswith("Logistic") else "rule"

        defaults = MatchingThresholds()
        weak = st.slider(
            "Weak-match threshold",
            0.0,
            1.0,
            defaults.weak_match,
            0.01,
            help="Below this similarity a ticket is treated as unrelated to the baseline.",
        )
        strong = st.slider(
            "Strong-match threshold",
            0.0,
            1.0,
            defaults.strong_match,
            0.01,
            help="At or above this, a match is confident rather than flagged for review.",
        )
        if weak > strong:
            st.warning(
                "The weak threshold cannot exceed the strong one; defaults restored.",
                icon="⚠️",
            )
            weak, strong = defaults.weak_match, defaults.strong_match

    # ---- Assumptions -------------------------------------------------
    with st.expander("Drift assumptions", expanded=False):
        refinement_weight = st.slider(
            "Refinement weight",
            0.0,
            1.0,
            0.5,
            0.05,
            help=(
                "Fraction of a Refinement ticket's size counted as added scope. "
                "0 treats refinement as free, 1 as entirely new work."
            ),
        )
        link_revl = st.checkbox(
            "Feed measured churn into REVL",
            value=True,
            help=(
                "Adds refinement churn and removals to the revised estimate as "
                "requirements volatility rework. New scope is excluded because "
                "the revised size already carries it."
            ),
        )

    # ---- Size conversion ---------------------------------------------
    with st.expander("Size conversion", expanded=False):
        size_unit = st.selectbox(
            "Ticket size unit",
            list(SIZE_UNITS),
            format_func=lambda key: SIZE_UNITS[key],
        )
        if size_unit == "story_points":
            factor = st.number_input(
                "SLOC per story point", min_value=1.0, value=60.0, step=5.0
            )
            conversion = SizeConversion(sloc_per_story_point=factor)
        elif size_unit == "function_points":
            factor = st.number_input(
                "SLOC per function point", min_value=1.0, value=53.0, step=1.0
            )
            conversion = SizeConversion(sloc_per_function_point=factor)
        else:
            conversion = SizeConversion()
        st.caption(
            "This factor is an organisational assumption, not a measurement. "
            "Every effort figure scales with it."
        )

    # ---- COCOMO drivers ----------------------------------------------
    with st.expander("COCOMO II drivers", expanded=False):
        st.caption("All drivers default to Nominal, which is an assumption.")
        ratings: dict[str, str] = {}
        for code, table in {**SCALE_FACTORS, **EFFORT_MULTIPLIERS}.items():
            options = list(table)
            ratings[code] = st.selectbox(
                f"{code} · {DRIVER_NAMES[code]}",
                options,
                index=options.index("N"),
                format_func=lambda rating: RATING_LABELS[rating],
                key=f"driver_{code}",
            )
        drivers = CocomoDrivers(ratings=ratings)

    # ---- Labour rate -------------------------------------------------
    with st.expander("Labour rate (optional)", expanded=False):
        rate = st.number_input(
            "Cost per person-month",
            min_value=0.0,
            value=0.0,
            step=500.0,
            help="Leave at zero to report effort and duration only.",
        )
        st.caption("DriftGuard will not assume a rate. No rate, no cost figure.")

    return AnalysisSettings(
        thresholds=MatchingThresholds(strong_match=strong, weak_match=weak),
        refinement_weight=refinement_weight,
        conversion=conversion,
        size_unit=size_unit,
        drivers=drivers,
        link_drift_to_revl=link_revl,
        classifier=classifier,
        monthly_labour_rate=rate if rate > 0 else None,
    )
