"""Reports and Export."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from driftguard import evaluation, features, pdf_report, reporting
from driftguard.classification import RuleClassifier
from driftguard.data import frame_to_csv_bytes
from driftguard.pipeline import prepare_training_data

from .. import components as ui, state, theme


def render(result) -> None:
    ui.page_header(
        "Reports and export",
        "A self-contained write-up of this analysis, plus the underlying data.",
    )

    project_name = st.text_input(
        "Report title",
        value=st.session_state.get("project_name", "DriftGuard analysis"),
    )
    st.session_state["project_name"] = project_name

    include_evaluation = False
    if state.has_labels():
        include_evaluation = st.checkbox(
            "Include classifier evaluation",
            value=True,
            help="Scores the rule classifier against the labelled test set.",
        )
    else:
        st.caption(
            "No labelled test set is loaded, so the report will state plainly "
            "that classifier accuracy was not measured."
        )

    evaluation_result = None
    cross_validated = None
    if include_evaluation:
        with ui.working("Scoring the classifier for the report…"):
            matched = prepare_training_data(
                st.session_state.baseline, st.session_state.labelled
            )
            predictions = RuleClassifier(result.settings.thresholds).classify(matched)
            evaluation_result = evaluation.evaluate(predictions)
            try:
                cross_validated = evaluation.cross_validated_scores(matched)
            except ValueError:
                cross_validated = None

    generated_at = datetime.now()
    with ui.working("Assembling report…"):
        markdown = reporting.build_markdown_report(
            result,
            project_name=project_name,
            evaluation=evaluation_result,
            cross_validated=cross_validated,
            generated_at=generated_at,
        )

    with ui.working("Rendering PDF…"):
        try:
            pdf_bytes = pdf_report.build_pdf_report(
                result,
                project_name=project_name,
                evaluation=evaluation_result,
                cross_validated=cross_validated,
                risk_indicators=features.risk_indicators(result),
                recommendations=features.recommendations(result),
                synthetic=state.is_synthetic(),
                generated_at=generated_at,
            )
            pdf_error = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the user below
            pdf_bytes, pdf_error = None, str(exc)

    ui.section("Downloads")
    stamp = generated_at.strftime("%Y%m%d")

    if pdf_bytes:
        ui.download_row(
            [
                (
                    "📄 Analysis report (PDF)",
                    pdf_bytes,
                    f"driftguard_report_{stamp}.pdf",
                    "application/pdf",
                ),
            ]
        )
        st.caption(
            f"{len(pdf_bytes) / 1024:,.0f} KB · cover page, executive summary, "
            "tables, charts, methodology, assumptions, limitations and "
            "conclusion. Every figure is read from this same analysis."
        )
    else:
        st.error(
            f"**PDF generation failed.** {pdf_error}. The Markdown and CSV "
            "exports below are unaffected.",
            icon="⚠️",
        )

    ui.download_row(
        [
            (
                "Analysis report (Markdown)",
                reporting.report_bytes(markdown),
                f"driftguard_report_{stamp}.md",
                "text/markdown",
            ),
            (
                "Classified tickets (CSV)",
                frame_to_csv_bytes(reporting.classified_export(result.classified)),
                f"driftguard_tickets_{stamp}.csv",
                "text/csv",
            ),
            (
                "Drift timeline (CSV)",
                frame_to_csv_bytes(result.timeline),
                f"driftguard_timeline_{stamp}.csv",
                "text/csv",
            ),
            (
                "Class totals (CSV)",
                frame_to_csv_bytes(result.drift.per_class),
                f"driftguard_class_totals_{stamp}.csv",
                "text/csv",
            ),
        ]
    )

    if result.timeline.empty:
        st.caption(
            "The timeline export will be empty: no ticket in this project "
            "carries a usable date."
        )

    ui.section("Report preview")
    st.markdown(
        f"<div class='dg-card' style='max-height:620px;overflow-y:auto'>",
        unsafe_allow_html=True,
    )
    st.markdown(markdown)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Raw Markdown"):
        st.code(markdown, language="markdown")
