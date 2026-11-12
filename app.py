"""DriftGuard — Streamlit dashboard entry point.

Run with:   ./run.sh      (or: streamlit run app.py)

This module is a router. All presentation lives in the ``ui`` package and all
analysis in ``driftguard``, so every number the dashboard shows can be
reproduced headlessly and is covered by the test suite.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="DriftGuard · Scope drift analytics",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

from driftguard.pipeline import prepare_training_data, run_analysis  # noqa: E402

from ui import components as ui, sidebar, state, theme, welcome  # noqa: E402
from ui.pages import (  # noqa: E402
    datasets,
    drift_analysis,
    feature_analysis,
    methodology,
    overview,
    performance,
    reports,
    review,
    scenarios,
)

#: Section order. The eight established sections, plus "Scenarios" for the
#: sensitivity and what-if analysis. "Ticket review" carries the
#: human-in-the-loop correction workflow.
PAGES = {
    "Executive overview": overview.render,
    "Dataset explorer": datasets.render,
    "Scope drift analysis": drift_analysis.render,
    "Scenarios": scenarios.render,
    "Model performance": performance.render,
    "Feature-level analysis": feature_analysis.render,
    "Ticket review": review.render,
    "Reports and export": reports.render,
    "Methodology and about": methodology.render,
}


def main() -> None:
    state.init()
    theme.inject_css()

    with st.sidebar:
        settings = sidebar.render()

    if not state.has_project():
        welcome.render()
        return

    baseline = st.session_state.baseline
    tickets = st.session_state.tickets

    training_data = None
    if settings.classifier == "ml":
        if not state.has_labels():
            st.error(
                "**The learned classifier needs a labelled test set.** Upload a "
                "tickets CSV with a `true_class` column, or switch back to the "
                "rule-based classifier in the sidebar.",
                icon="⚠️",
            )
            return
        with ui.working("Training the learned classifier…"):
            training_data = prepare_training_data(baseline, st.session_state.labelled)

    try:
        with ui.working("Matching tickets and computing drift…"):
            result = run_analysis(
                baseline,
                tickets,
                settings,
                overrides=st.session_state.overrides,
                size_overrides=st.session_state.size_overrides,
                training_data=training_data,
            )
    except ValueError as exc:
        st.error(f"**Analysis failed.** {exc}", icon="⚠️")
        return

    # ---- Navigation ------------------------------------------------------
    choice = st.radio(
        "section-nav",
        list(PAGES),
        horizontal=True,
        label_visibility="collapsed",
        key="nav",
    )

    review_count = state.review_count()
    project_name = st.session_state.get("project_name") or "Untitled project"
    source_label = {
        "sample": "bundled sample",
        "synthetic": "synthetic test data",
        "upload": "uploaded data",
    }.get(st.session_state.source, "unknown source")

    st.markdown(
        f"<div style='display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap;"
        f"font-size:0.78rem;color:{theme.INK_MUTED};margin:-0.3rem 0 0.2rem 0'>"
        f"<span style='color:{theme.INK_SECONDARY};font-weight:600'>{project_name}</span>"
        f"<span>{source_label}</span>"
        f"<span>{len(result.classified):,} tickets</span>"
        f"<span>{len(baseline):,} requirements</span>"
        f"<span>{review_count} reviewed</span>"
        f"<span>{'labelled set loaded' if state.has_labels() else 'no labels'}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # A synthetic project must announce itself on every page, not just one.
    if state.is_synthetic():
        st.warning(
            "**Synthetic test data.** This project was produced by DriftGuard's "
            "generator to exercise the application. Every figure below "
            "demonstrates behaviour only and is not evidence about any real "
            "project, organisation or industry benchmark.",
            icon="🧪",
        )

    st.divider()

    PAGES[choice](result)


if __name__ == "__main__":
    main()
