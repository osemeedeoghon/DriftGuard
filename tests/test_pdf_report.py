"""Tests for PDF report generation."""

from __future__ import annotations

import pytest

from driftguard import data, evaluation, features, pdf_report, pipeline
from driftguard.classification import RuleClassifier
from driftguard.pipeline import AnalysisSettings

pymupdf = pytest.importorskip("pymupdf", reason="PDF text extraction needs pymupdf")


@pytest.fixture(scope="module")
def analysis():
    return pipeline.run_analysis(data.sample_baseline(), data.sample_tickets())


@pytest.fixture(scope="module")
def scored():
    matched = pipeline.prepare_training_data(
        data.sample_baseline(), data.sample_labelled_tickets()
    )
    return evaluation.evaluate(RuleClassifier().classify(matched))


def extract(pdf: bytes) -> str:
    """Flatten a PDF to searchable text.

    Whitespace is collapsed because the renderer wraps paragraphs at arbitrary
    points, so a phrase present in the document can otherwise straddle a line
    break and defeat a substring check.
    """
    import re

    document = pymupdf.open(stream=pdf, filetype="pdf")
    raw = "\n".join(page.get_text() for page in document)
    return re.sub(r"\s+", " ", raw)


def test_pdf_is_a_valid_document(analysis):
    pdf = pdf_report.build_pdf_report(analysis)
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 10_000
    assert pymupdf.open(stream=pdf, filetype="pdf").page_count >= 3


def test_pdf_contains_every_required_section(analysis, scored):
    pdf = pdf_report.build_pdf_report(
        analysis,
        project_name="Test Project",
        evaluation=scored,
        risk_indicators=features.risk_indicators(analysis),
        recommendations=features.recommendations(analysis),
    )
    text = extract(pdf)
    for section in (
        "Scope Drift Analysis",
        "Executive summary",
        "Scope position",
        "Ticket classification",
        "Effort and schedule estimates",
        "Tickets driving the drift",
        "Human review activity",
        "Risk indicators",
        "Recommended actions",
        "Model evaluation",
        "Methodology",
        "Assumptions",
        "Limitations",
        "Conclusion",
    ):
        assert section in text, section


def test_pdf_figures_match_the_analysis(analysis):
    pdf = pdf_report.build_pdf_report(analysis, project_name="Test Project")
    text = extract(pdf)
    assert f"{analysis.drift.sdi:+.1%}" in text
    assert f"{analysis.drift.baseline_size:,.0f}" in text
    assert f"{analysis.estimates.revised.person_months:,.1f}" in text


def test_pdf_states_when_accuracy_was_not_measured(analysis):
    text = extract(pdf_report.build_pdf_report(analysis))
    assert "not measured" in text
    assert "not been scored against ground truth" in text


def test_pdf_carries_the_in_sample_caveat(analysis, scored):
    text = extract(pdf_report.build_pdf_report(analysis, evaluation=scored))
    assert "in-sample" in text
    assert "overstate performance" in text


def test_pdf_reports_no_cost_without_a_rate(analysis):
    text = extract(pdf_report.build_pdf_report(analysis))
    assert "not supplied" in text
    assert "Labour cost estimate" not in text


def test_pdf_labels_cost_as_an_estimate_when_a_rate_is_given():
    result = pipeline.run_analysis(
        data.sample_baseline(),
        data.sample_tickets(),
        AnalysisSettings(monthly_labour_rate=12000.0),
    )
    text = extract(pdf_report.build_pdf_report(result))
    assert "Labour cost estimate" in text
    assert "not an actual or" in text or "derived labour-cost estimate" in text


def test_pdf_distinguishes_effort_from_duration(analysis):
    text = extract(pdf_report.build_pdf_report(analysis))
    assert "Effort (person-months)" in text
    assert "Duration (calendar months)" in text
    assert "not interchangeable by" in text


def test_pdf_explains_the_revl_decision(analysis):
    text = extract(pdf_report.build_pdf_report(analysis))
    assert "REVL" in text
    assert "charge for the same work twice" in text


def test_synthetic_pdf_carries_a_prominent_notice(analysis):
    text = extract(pdf_report.build_pdf_report(analysis, synthetic=True))
    assert "SYNTHETIC DATA" in text
    assert "not evidence about any real project" in text


def test_non_synthetic_pdf_has_no_notice(analysis):
    text = extract(pdf_report.build_pdf_report(analysis, synthetic=False))
    assert "SYNTHETIC DATA" not in text


def test_pdf_handles_a_project_without_dates():
    tickets = data.sample_tickets().drop(columns=["created_date"])
    result = pipeline.run_analysis(data.sample_baseline(), tickets)
    pdf = pdf_report.build_pdf_report(result)
    assert pdf.startswith(b"%PDF-")
    assert "Scope change over time" not in extract(pdf)


def test_pdf_includes_cross_validated_figures(analysis, scored):
    matched = pipeline.prepare_training_data(
        data.sample_baseline(), data.sample_labelled_tickets()
    )
    folds = evaluation.cross_validated_scores(matched)
    text = extract(
        pdf_report.build_pdf_report(analysis, evaluation=scored, cross_validated=folds)
    )
    assert "out-of-sample" in text
    assert "stratified folds" in text
