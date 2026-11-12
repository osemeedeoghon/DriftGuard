"""Landing screen shown before a project is loaded."""

from __future__ import annotations

import streamlit as st

from driftguard import data

from . import components as ui, state, theme


def render() -> None:
    st.markdown("# DriftGuard")
    st.markdown(
        "<div class='dg-section-sub' style='margin:-0.5rem 0 1.4rem 0;"
        "max-width:60rem;font-size:0.98rem'>"
        "Scope creep detection and COCOMO II impact estimation for software "
        "projects. Match project tickets against an agreed requirements "
        "baseline, quantify the drift, and cost it in effort and schedule."
        "</div>",
        unsafe_allow_html=True,
    )

    ui.empty_state(
        "No project loaded",
        "Load the bundled sample project to see a complete worked example "
        "immediately, or upload your own baseline and tickets using the panel "
        "on the left.",
        icon="📐",
    )

    st.write("")
    left, right = st.columns([1, 2])
    with left:
        if st.button(
            "Load sample project",
            type="primary",
            use_container_width=True,
            key="welcome_load_sample",
        ):
            state.load_sample_project()
            st.rerun()
        st.caption(
            "18 baseline requirements, 45 tickets, manually labelled for "
            "evaluation."
        )
    with right:
        st.markdown(
            "**What you get.** Eight sections covering the executive position, "
            "the datasets and their quality, scope drift over time, classifier "
            "performance against ground truth, feature-level evidence, a "
            "human review workflow, exportable reports, and the full "
            "methodology with its assumptions and limits stated."
        )

    ui.section("Expected input format")
    left, right = st.columns(2)
    with left:
        st.markdown("**Baseline requirements**")
        st.code(
            "requirement_id,title,description,size\n"
            "R-01,Student login,Students can sign in with a password,8\n"
            "R-02,Course search,Students can search the catalogue,13",
            language="text",
        )
        st.caption(
            "`requirement_id` and `description` are required; `size` is "
            "strongly recommended. A plain TXT file with one requirement per "
            "line also works, with sizes defaulting to a uniform value."
        )
    with right:
        st.markdown("**Project tickets**")
        st.code(
            "ticket_id,title,description,size,created_date\n"
            "T-101,Build sign-in page,Implement the login screen,5,2025-01-08\n"
            "T-301,Mobile app,Introduce a native iOS app,21,2025-03-24",
            language="text",
        )
        st.caption(
            "`ticket_id` and `description` are required. Add `created_date` "
            "for the drift timeline and an optional `true_class` column to "
            "measure classifier accuracy."
        )

    ui.section("Sample files")
    ui.download_row(
        [
            (
                "Baseline (CSV)",
                data.read_sample_bytes("baseline_requirements.csv"),
                "baseline_requirements.csv",
                "text/csv",
            ),
            (
                "Baseline (TXT)",
                data.read_sample_bytes("baseline_requirements.txt"),
                "baseline_requirements.txt",
                "text/plain",
            ),
            (
                "Tickets (CSV)",
                data.read_sample_bytes("tickets.csv"),
                "tickets.csv",
                "text/csv",
            ),
            (
                "Labelled test set (CSV)",
                data.read_sample_bytes("labelled_test_set.csv"),
                "labelled_test_set.csv",
                "text/csv",
            ),
        ]
    )

    st.caption(
        "DriftGuard runs fully offline. No LLM API, API key, cloud database or "
        "paid service is involved in producing any figure it shows."
    )
