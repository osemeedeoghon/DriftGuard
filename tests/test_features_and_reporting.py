"""Tests for the feature-level views, reporting, and presentation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from driftguard import data, features, pipeline, reporting
from driftguard.classification import RuleClassifier
from driftguard.config import IN_SCOPE, NEW_SCOPE, REFINEMENT, REMOVAL
from driftguard.evaluation import cross_validated_scores, evaluate
from driftguard.matching import RequirementMatcher


@pytest.fixture(scope="module")
def project():
    return data.sample_baseline(), data.sample_tickets()


@pytest.fixture(scope="module")
def analysis(project):
    baseline, tickets = project
    return pipeline.run_analysis(baseline, tickets)


# --------------------------------------------------------------------------
# Requirement coverage
# --------------------------------------------------------------------------
def test_coverage_counts_only_implementing_classes(project, analysis):
    """New scope tickets are matched for evidence, but do not cover anything."""
    baseline, _ = project
    coverage = features.requirement_coverage(analysis.classified, baseline)

    assert len(coverage) == len(baseline)
    counted = coverage["tickets"].sum()
    implementing = analysis.classified["final_class"].isin([IN_SCOPE, REFINEMENT]).sum()
    assert counted == implementing
    # New scope is plentiful in the sample, so this is a real exclusion.
    assert (analysis.classified["final_class"] == NEW_SCOPE).sum() > 0


def test_coverage_marks_unmatched_requirements_as_zero():
    baseline = pd.DataFrame(
        {
            "requirement_id": ["R-01", "R-02"],
            "title": ["Login", "Reporting"],
            "description": ["sign in", "reports"],
            "size": [5.0, 8.0],
        }
    )
    classified = pd.DataFrame(
        {
            "ticket_id": ["T-1"],
            "matched_requirement_id": ["R-01"],
            "final_class": [IN_SCOPE],
        }
    )
    coverage = features.requirement_coverage(classified, baseline).set_index("requirement_id")
    assert coverage.loc["R-01", "tickets"] == 1
    assert coverage.loc["R-02", "tickets"] == 0


def test_coverage_of_empty_baseline_is_empty():
    assert features.requirement_coverage(pd.DataFrame(), pd.DataFrame()).empty


# --------------------------------------------------------------------------
# Similarity summary
# --------------------------------------------------------------------------
def test_similarity_summary_covers_every_class(analysis):
    summary = features.class_similarity_summary(analysis.classified)
    assert list(summary["class"]) == [IN_SCOPE, REFINEMENT, NEW_SCOPE, REMOVAL]
    assert summary["tickets"].sum() == len(analysis.classified)


def test_in_scope_similarity_exceeds_new_scope(analysis):
    """The separation the whole classifier rests on."""
    summary = features.class_similarity_summary(analysis.classified).set_index("class")
    assert summary.loc[IN_SCOPE, "median"] > summary.loc[NEW_SCOPE, "median"]


def test_similarity_summary_handles_empty_input():
    assert features.class_similarity_summary(pd.DataFrame()).empty


# --------------------------------------------------------------------------
# Model health
# --------------------------------------------------------------------------
def test_model_health_counts_are_consistent(analysis):
    health = features.model_health(analysis.classified)
    assert health["tickets"] == len(analysis.classified)
    assert 0 <= health["low_confidence"] <= health["tickets"]
    assert health["low_confidence_share"] == pytest.approx(
        health["low_confidence"] / health["tickets"]
    )
    assert 0.0 <= health["mean_confidence"] <= 1.0


def test_model_health_reports_reviewer_activity(project):
    baseline, tickets = project
    reviewed = pipeline.run_analysis(baseline, tickets, overrides={"T-301": IN_SCOPE})
    health = features.model_health(reviewed.classified)
    assert health["reviewed"] == 1
    assert health["overturned"] == 1


def test_model_health_on_empty_input_is_all_zero():
    health = features.model_health(pd.DataFrame())
    assert health["tickets"] == 0
    assert health["low_confidence_share"] == 0.0


def test_review_queue_is_ordered_by_confidence(analysis):
    queue = features.review_queue(analysis.classified)
    if len(queue) > 1:
        assert list(queue["confidence"]) == sorted(queue["confidence"])
    assert (queue["confidence"] < 0.5).all()


# --------------------------------------------------------------------------
# Per-prediction term weights
# --------------------------------------------------------------------------
def test_shared_term_weights_rank_by_product(project):
    baseline, tickets = project
    matcher = RequirementMatcher().fit(baseline, tickets)
    text = tickets.loc[tickets["ticket_id"] == "T-105", "text"].iloc[0]
    terms = matcher.shared_term_weights(text, "R-03")

    assert not terms.empty
    assert list(terms["weight"]) == sorted(terms["weight"], reverse=True)
    assert terms["weight"].iloc[0] == pytest.approx(
        terms["ticket_weight"].iloc[0] * terms["requirement_weight"].iloc[0]
    )
    assert "enrolment" in " ".join(terms["term"]).lower() or "seat" in " ".join(terms["term"]).lower()


def test_shared_term_weights_empty_when_nothing_overlaps(project):
    baseline, tickets = project
    matcher = RequirementMatcher().fit(baseline, tickets)
    assert matcher.shared_term_weights("zzzz qqqq xxxx", "R-01").empty


def test_shared_term_weights_rejects_unknown_requirement(project):
    baseline, tickets = project
    matcher = RequirementMatcher().fit(baseline, tickets)
    with pytest.raises(ValueError, match="Unknown requirement id"):
        matcher.shared_term_weights("anything", "R-999")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def test_report_contains_the_headline_figures(analysis):
    report = reporting.build_markdown_report(analysis, project_name="Test project")
    assert "# Test project" in report
    assert f"{analysis.drift.sdi * 100:,.1f}%" in report
    assert "COCOMO II.2000" in report
    assert "Scope Drift Index" in report


def test_report_states_plainly_when_accuracy_was_not_measured(analysis):
    """The constraint against unsupported quality claims."""
    report = reporting.build_markdown_report(analysis)
    assert "not measured" in report
    assert "has not been scored against ground truth" in report


def test_report_includes_evaluation_with_in_sample_caveat(project):
    baseline, tickets = project
    result = pipeline.run_analysis(baseline, tickets)
    matched = pipeline.prepare_training_data(baseline, data.sample_labelled_tickets())
    scored = evaluate(RuleClassifier().classify(matched))
    report = reporting.build_markdown_report(result, evaluation=scored)

    assert "Macro-F1" in report
    assert "in-sample" in report
    assert "overstate performance" in report


def test_report_includes_cross_validated_section(project):
    baseline, tickets = project
    result = pipeline.run_analysis(baseline, tickets)
    matched = pipeline.prepare_training_data(baseline, data.sample_labelled_tickets())
    scored = evaluate(RuleClassifier().classify(matched))
    folds = cross_validated_scores(matched)
    report = reporting.build_markdown_report(
        result, evaluation=scored, cross_validated=folds
    )
    assert "out-of-sample" in report.lower()
    assert "stratified folds" in report


def test_report_reports_no_cost_without_a_rate(analysis):
    report = reporting.build_markdown_report(analysis)
    assert "No labour rate was supplied" in report


def test_report_labels_cost_as_an_estimate_when_a_rate_is_given(project):
    baseline, tickets = project
    settings = pipeline.AnalysisSettings(monthly_labour_rate=12000.0)
    result = pipeline.run_analysis(baseline, tickets, settings)
    report = reporting.build_markdown_report(result)
    assert "estimated" in report
    assert "not an \n" not in report  # no broken sentence
    assert "actual or committed cost" in report


def test_report_distinguishes_effort_from_duration(analysis):
    report = reporting.build_markdown_report(analysis)
    assert "Effort (person-months)" in report
    assert "Duration (calendar months)" in report
    assert "do not convert into one" in report


def test_report_lists_assumptions(analysis):
    report = reporting.build_markdown_report(analysis)
    assert "SLOC per story point" in report
    assert "Refinement weight" in report
    assert "organisational assumption, not a" in report


def test_report_encodes_to_bytes(analysis):
    payload = reporting.report_bytes(reporting.build_markdown_report(analysis))
    assert isinstance(payload, bytes)
    assert payload.startswith(b"#")


def test_classified_export_flattens_list_columns(analysis):
    export = reporting.classified_export(analysis.classified)
    assert "alternative_matches" not in export.columns
    if "shared_terms" in export.columns:
        assert export["shared_terms"].map(lambda v: isinstance(v, str)).all()
    # Must survive a CSV round trip.
    assert data.frame_to_csv_bytes(export)


# --------------------------------------------------------------------------
# Presentation helpers (pure functions, no Streamlit runtime needed)
# --------------------------------------------------------------------------
def test_rgba_converts_hex_to_plotly_safe_string():
    from ui import theme

    assert theme.rgba("#1baf7a", 0.5) == "rgba(27,175,122,0.5)"
    assert theme.rgba("1baf7a", 0.1) == "rgba(27,175,122,0.1)"


def test_rgba_rejects_short_hex():
    from ui import theme

    with pytest.raises(ValueError, match="6-digit hex"):
        theme.rgba("#fff", 0.5)


def test_severity_bands_are_ordered_and_always_carry_an_icon():
    from ui import theme

    for sdi, expected in ((0.40, "Critical"), (0.15, "Serious"), (0.07, "Watch"), (0.0, "Stable")):
        label, colour, icon = theme.severity_for(sdi)
        assert label == expected
        assert colour.startswith("#")
        assert icon  # never colour alone


def test_de_scoping_has_its_own_severity_label():
    from ui import theme

    label, _, icon = theme.severity_for(-0.3)
    assert label == "De-scoped"
    assert icon


def test_percent_formatting_helpers():
    from ui import theme

    assert theme.fmt_percent(0.758, 1, signed=True) == "+75.8%"
    assert theme.fmt_number(1234.5, 1) == "1,234.5"
    assert theme.fmt_signed(-3.25, 1, " PM") == "-3.2 PM"


def test_quality_summary_stores_percentages_on_a_0_to_100_scale():
    """Guards the formatting bug where a fraction rendered as 1.0% not 95.6%."""
    from ui import components

    frame = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    summary = components.quality_summary(frame, "test").set_index("Column")
    assert summary.loc["a", "Missing"] == 2
    assert summary.loc["a", "Missing %"] == pytest.approx(50.0)
    assert summary.loc["b", "Missing %"] == pytest.approx(0.0)


def test_scale_to_percent_only_touches_named_columns():
    from ui import components

    frame = pd.DataFrame({"Accuracy": [0.956], "Macro-F1": [0.96]})
    scaled = components.scale_to_percent(frame, ["Accuracy"])
    assert scaled["Accuracy"].iloc[0] == pytest.approx(95.6)
    assert scaled["Macro-F1"].iloc[0] == pytest.approx(0.96)


def test_class_colours_are_six_digit_hex():
    """Plotly fill properties reject 8-digit hex, so the source must be clean."""
    from driftguard.config import CLASS_COLOURS

    for name, colour in CLASS_COLOURS.items():
        assert colour.startswith("#") and len(colour) == 7, name


# --------------------------------------------------------------------------
# "No evidence, no claim" guard
# --------------------------------------------------------------------------
def test_has_labels_is_false_without_a_true_class_column():
    """The Model Performance page shows an empty state on this branch rather
    than reporting an accuracy it cannot support."""
    import streamlit as st

    from ui import state

    st.session_state["labelled"] = None
    assert state.has_labels() is False

    st.session_state["labelled"] = pd.DataFrame({"ticket_id": ["T-1"]})
    assert state.has_labels() is False

    st.session_state["labelled"] = pd.DataFrame(
        {"ticket_id": ["T-1"], "true_class": [IN_SCOPE]}
    )
    assert state.has_labels() is True


def test_has_project_requires_both_inputs():
    import streamlit as st

    from ui import state

    st.session_state["baseline"] = None
    st.session_state["tickets"] = None
    assert state.has_project() is False

    st.session_state["baseline"] = pd.DataFrame({"a": [1]})
    assert state.has_project() is False

    st.session_state["tickets"] = pd.DataFrame({"a": [1]})
    assert state.has_project() is True


# --------------------------------------------------------------------------
# Scope-impact columns and the high-impact threshold
# --------------------------------------------------------------------------
def test_classified_carries_the_scope_impact_columns(analysis):
    """The bug this guards: `classified` lacked `net_contribution`, so the
    risk indicator silently reported zero and the review page would crash."""
    for column in ("added", "removed", "net_contribution"):
        assert column in analysis.classified.columns, column
    assert analysis.classified["net_contribution"].abs().sum() > 0


def test_classified_and_per_ticket_agree(analysis):
    assert len(analysis.classified) == len(analysis.drift.per_ticket)
    assert analysis.classified["net_contribution"].sum() == pytest.approx(
        analysis.drift.net_change
    )


def test_high_impact_threshold_is_a_share_of_baseline():
    assert features.high_impact_threshold(157.0) == pytest.approx(7.85)
    assert features.high_impact_threshold(0.0) == 0.0


def test_reviewing_a_high_impact_ticket_never_raises_the_unreviewed_count():
    """A quantile cutoff moved when a classification changed, so reviewing a
    large ticket could increase the count of unreviewed high-impact tickets.
    An absolute share cannot do that."""
    baseline, tickets = data.sample_baseline(), data.sample_tickets()

    def unreviewed(overrides):
        result = pipeline.run_analysis(baseline, tickets, overrides=overrides)
        row = features.risk_indicators(result).set_index("Indicator").loc[
            "High-impact tickets not yet reviewed", "Value"
        ]
        return int(str(row).split(" of ")[0])

    none = unreviewed({})
    one = unreviewed({"T-304": IN_SCOPE})
    three = unreviewed({"T-304": IN_SCOPE, "T-301": IN_SCOPE, "T-310": IN_SCOPE})
    assert none > one > three


def test_high_impact_mask_excludes_zero_impact_tickets(analysis):
    mask = features.high_impact_mask(analysis.classified, analysis.baseline_size)
    assert mask.sum() > 0
    impacts = analysis.classified.loc[mask, "net_contribution"].abs()
    assert (impacts > 0).all()


def test_high_impact_mask_handles_empty_input():
    assert features.high_impact_mask(pd.DataFrame(), 100.0).empty


# --------------------------------------------------------------------------
# Risk indicators and recommendations
# --------------------------------------------------------------------------
def test_risk_indicators_state_their_basis(analysis):
    indicators = features.risk_indicators(analysis)
    assert len(indicators) >= 6
    assert indicators["Basis"].str.len().min() > 10
    assert set(indicators["Level"]) <= {"Critical", "Serious", "Watch", "Stable"}


def test_risk_indicators_flag_heavy_growth(analysis):
    indicators = features.risk_indicators(analysis).set_index("Indicator")
    assert indicators.loc["Net scope growth", "Level"] == "Critical"


def test_recommendations_cite_measured_figures(analysis):
    items = features.recommendations(analysis)
    assert items
    for item in items:
        assert item["priority"] in {"High", "Medium", "Low"}
        assert len(item["action"]) > 10
        assert len(item["because"]) > 20


def test_recommendation_about_the_sloc_factor_is_always_present(analysis):
    actions = " ".join(item["action"] for item in features.recommendations(analysis))
    assert "SLOC-per-story-point" in actions


def test_no_cost_recommendation_once_a_rate_is_supplied():
    from driftguard.pipeline import AnalysisSettings

    result = pipeline.run_analysis(
        data.sample_baseline(),
        data.sample_tickets(),
        AnalysisSettings(monthly_labour_rate=10000.0),
    )
    actions = " ".join(item["action"] for item in features.recommendations(result))
    assert "Supply a labour rate" not in actions


# --------------------------------------------------------------------------
# Cue evidence and per-ticket explanation
# --------------------------------------------------------------------------
def test_cue_evidence_reports_the_cues_that_fired(analysis):
    row = analysis.classified.loc[analysis.classified["ticket_id"] == "T-401"].iloc[0]
    cues = features.cue_evidence(str(row["text"]))
    assert not cues.empty
    assert (cues["Signals"] == REMOVAL).any()


def test_cue_evidence_is_empty_for_plain_implementation_text(analysis):
    row = analysis.classified.loc[analysis.classified["ticket_id"] == "T-105"].iloc[0]
    assert features.cue_evidence(str(row["text"])).empty


def test_explain_ticket_reports_the_similarity_band(analysis):
    from driftguard.config import MatchingThresholds

    thresholds = MatchingThresholds()
    high = analysis.classified.loc[analysis.classified["ticket_id"] == "T-105"].iloc[0]
    low = analysis.classified.loc[analysis.classified["ticket_id"] == "T-301"].iloc[0]

    assert "Confident" in features.explain_ticket(high, thresholds)["similarity_band"]
    assert "Below weak" in features.explain_ticket(low, thresholds)["similarity_band"]


def test_explain_ticket_marks_an_override():
    from driftguard.config import MatchingThresholds

    result = pipeline.run_analysis(
        data.sample_baseline(), data.sample_tickets(), overrides={"T-301": IN_SCOPE}
    )
    row = result.classified.loc[result.classified["ticket_id"] == "T-301"].iloc[0]
    detail = features.explain_ticket(row, MatchingThresholds())
    assert detail["overridden"] is True
    assert detail["predicted_class"] == NEW_SCOPE
    assert detail["final_class"] == IN_SCOPE
