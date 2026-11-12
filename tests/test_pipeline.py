"""End-to-end pipeline tests, including the REVL double-counting guard."""

from __future__ import annotations

import pandas as pd
import pytest

from driftguard import data, drift
from driftguard.cocomo import CocomoDrivers, SizeConversion
from driftguard.config import IN_SCOPE, NEW_SCOPE, REFINEMENT, REMOVAL
from driftguard.pipeline import AnalysisSettings, prepare_training_data, run_analysis


@pytest.fixture(scope="module")
def project():
    return data.sample_baseline(), data.sample_tickets()


# --------------------------------------------------------------------------
# REVL: the double-counting trap
# --------------------------------------------------------------------------
def make_result(rows: list[tuple[str, str, float]]) -> drift.DriftResult:
    frame = pd.DataFrame(rows, columns=["ticket_id", "final_class", "size"])
    return drift.calculate_drift(frame, baseline_size=100.0, refinement_weight=0.5)


def test_revl_excludes_new_scope_because_revised_size_already_carries_it():
    """The regression this function exists to prevent.

    New scope shows up in the revised size. If it also inflated REVL, the
    effort equation would charge for the same work twice.
    """
    result = make_result([("T-1", NEW_SCOPE, 50.0)])
    assert result.sdi == pytest.approx(0.50)
    assert drift.revl_from_drift(result) == pytest.approx(0.0)
    # The naive mapping is what would have double-counted.
    assert drift.revl_from_sdi(result.sdi) == pytest.approx(50.0)


def test_revl_counts_refinement_churn():
    result = make_result([("T-1", REFINEMENT, 20.0)])  # 0.5 * 20 = 10 added
    assert drift.revl_from_drift(result) == pytest.approx(10.0)


def test_revl_counts_removed_work_as_rework():
    """Removed size leaves the revised total, but the effort spent does not."""
    result = make_result([("T-1", REMOVAL, 15.0)])
    assert result.sdi == pytest.approx(-0.15)
    assert drift.revl_from_drift(result) == pytest.approx(15.0)


def test_revl_is_never_negative():
    result = make_result([("T-1", REMOVAL, 500.0)])
    assert drift.revl_from_drift(result) >= 0.0


def test_revl_is_zero_for_a_perfectly_stable_project():
    result = make_result([("T-1", IN_SCOPE, 40.0), ("T-2", IN_SCOPE, 10.0)])
    assert drift.revl_from_drift(result) == pytest.approx(0.0)


def test_cocomo_receives_net_size_not_gross_added_size(project):
    """Effort must be driven by net scope change, exactly as the SDI is.

    Gross additions (148 story points on the sample project) are bookkeeping
    for attribution. The size handed to the effort equation is the *net*
    revised total (276), the same quantity the Scope Drift Index is computed
    from. Using the gross figure would silently inflate every estimate.
    """
    baseline, tickets = project
    result = run_analysis(baseline, tickets)
    d, conversion = result.drift, result.settings.conversion

    assert d.size_added > d.net_change  # gross and net genuinely differ here
    assert result.revised_ksloc == pytest.approx(
        conversion.to_ksloc(d.revised_size, "story_points")
    )
    gross_ksloc = conversion.to_ksloc(d.baseline_size + d.size_added, "story_points")
    assert result.revised_ksloc < gross_ksloc
    # And the net size is consistent with the index itself.
    assert d.revised_size == pytest.approx(d.baseline_size * (1 + d.sdi))


def test_effort_ratio_decomposes_into_size_and_revl_exactly(project):
    """Pin down where the effort multiple comes from.

    With all drivers nominal the effort ratio is exactly

        (size_ratio ** E) * (revl_factor ** E)

    so any future surprise in the headline multiple is attributable to one of
    those two terms and nothing else.
    """
    baseline, tickets = project
    result = run_analysis(baseline, tickets)
    e = result.estimates
    exponent = e.revised.exponent_e

    size_ratio = e.revised.size_ksloc / e.baseline.size_ksloc
    revl_factor = e.revised.effective_ksloc / e.revised.size_ksloc
    actual = e.revised.person_months / e.baseline.person_months

    assert actual == pytest.approx(size_ratio**exponent * revl_factor**exponent)
    # The size term alone is the pure diseconomy-of-scale effect.
    assert size_ratio**exponent == pytest.approx(1.86, abs=0.01)


def test_without_revl_effort_ratio_is_purely_size_to_the_exponent(project):
    """Turning the REVL link off must leave nothing but the size effect."""
    baseline, tickets = project
    result = run_analysis(baseline, tickets, AnalysisSettings(link_drift_to_revl=False))
    e = result.estimates

    assert e.revised.effective_ksloc == pytest.approx(e.revised.size_ksloc)
    size_ratio = e.revised.size_ksloc / e.baseline.size_ksloc
    actual = e.revised.person_months / e.baseline.person_months
    assert actual == pytest.approx(size_ratio ** e.revised.exponent_e)


def test_revl_percentage_counts_only_churn_on_sample_data(project):
    """Guards the exact REVL figure against silent drift."""
    baseline, tickets = project
    result = run_analysis(baseline, tickets)
    d = result.drift

    refinement_added = d.per_ticket.loc[
        d.per_ticket[d.classification_column] == REFINEMENT, "added"
    ].sum()
    new_scope_added = d.per_ticket.loc[
        d.per_ticket[d.classification_column] == NEW_SCOPE, "added"
    ].sum()
    expected = (refinement_added + d.size_removed) / d.baseline_size * 100

    applied = (
        result.estimates.revised.effective_ksloc / result.estimates.revised.size_ksloc
        - 1
    ) * 100
    assert applied == pytest.approx(expected)
    # New scope is large here, and must be contributing none of it.
    assert new_scope_added > 0
    assert applied < new_scope_added / d.baseline_size * 100


def test_effort_growth_stays_proportionate_to_size_growth(project):
    """Guard against a re-introduced double count.

    With COCOMO's exponent near 1.1, effort should grow roughly as size^1.1
    plus a modest churn allowance -- not as the square of the size change.
    """
    baseline, tickets = project
    result = run_analysis(baseline, tickets)
    size_ratio = result.revised_ksloc / result.baseline_ksloc
    effort_ratio = (
        result.estimates.revised.person_months
        / result.estimates.baseline.person_months
    )
    assert effort_ratio > size_ratio  # diseconomy of scale is real
    assert effort_ratio < size_ratio**2  # but it is not quadratic


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def test_pipeline_runs_on_sample_data(project):
    baseline, tickets = project
    result = run_analysis(baseline, tickets)

    assert len(result.classified) == 45
    assert result.baseline_size == pytest.approx(157.0)
    assert result.drift.sdi > 0
    assert len(result.timeline) > 0
    assert result.estimates.revised.person_months > result.estimates.baseline.person_months
    assert result.estimates.revised.duration_months > result.estimates.baseline.duration_months


def test_effort_and_duration_remain_distinct(project):
    baseline, tickets = project
    result = run_analysis(baseline, tickets)
    assert result.estimates.revised.person_months != result.estimates.revised.duration_months
    # Effort grows faster than the calendar: adding scope costs more work than
    # it costs time, which is the whole point of reporting both.
    assert (
        result.estimates.effort_change_percent
        > result.estimates.duration_change_percent
    )


def test_overrides_change_the_reported_drift(project):
    baseline, tickets = project
    before = run_analysis(baseline, tickets)
    after = run_analysis(
        baseline, tickets, overrides={"T-301": IN_SCOPE}  # 21 points of new scope
    )
    assert after.drift.sdi < before.drift.sdi
    assert after.drift.sdi == pytest.approx(before.drift.sdi - 21.0 / 157.0)


def test_size_override_changes_the_reported_drift(project):
    baseline, tickets = project
    before = run_analysis(baseline, tickets)
    after = run_analysis(baseline, tickets, size_overrides={"T-301": 1.0})
    assert after.drift.sdi == pytest.approx(before.drift.sdi - 20.0 / 157.0)


def test_predictions_are_preserved_alongside_reviewer_decisions(project):
    baseline, tickets = project
    result = run_analysis(baseline, tickets, overrides={"T-301": IN_SCOPE})
    row = result.classified.loc[result.classified["ticket_id"] == "T-301"].iloc[0]
    assert row["final_class"] == IN_SCOPE
    assert row["predicted_class"] == NEW_SCOPE


def test_refinement_weight_moves_the_index(project):
    baseline, tickets = project
    free = run_analysis(baseline, tickets, AnalysisSettings(refinement_weight=0.0))
    full = run_analysis(baseline, tickets, AnalysisSettings(refinement_weight=1.0))
    assert full.drift.sdi > free.drift.sdi


def test_size_conversion_factor_scales_ksloc(project):
    baseline, tickets = project
    small = run_analysis(
        baseline, tickets, AnalysisSettings(conversion=SizeConversion(sloc_per_story_point=30.0))
    )
    large = run_analysis(
        baseline, tickets, AnalysisSettings(conversion=SizeConversion(sloc_per_story_point=60.0))
    )
    assert large.baseline_ksloc == pytest.approx(small.baseline_ksloc * 2)
    # Drift is dimensionless, so the conversion factor must not move it.
    assert large.drift.sdi == pytest.approx(small.drift.sdi)


def test_disabling_revl_link_lowers_the_revised_effort(project):
    baseline, tickets = project
    linked = run_analysis(baseline, tickets, AnalysisSettings(link_drift_to_revl=True))
    unlinked = run_analysis(baseline, tickets, AnalysisSettings(link_drift_to_revl=False))
    assert linked.estimates.revised.person_months > unlinked.estimates.revised.person_months
    # Both share the same nominal revised size; only REVL differs.
    assert linked.revised_ksloc == pytest.approx(unlinked.revised_ksloc)


def test_driver_ratings_flow_into_the_estimate(project):
    baseline, tickets = project
    nominal = run_analysis(baseline, tickets)
    complex_project = run_analysis(
        baseline, tickets, AnalysisSettings(drivers=CocomoDrivers(ratings={"CPLX": "VH"}))
    )
    assert (
        complex_project.estimates.revised.person_months
        > nominal.estimates.revised.person_months
    )


def test_ml_classifier_requires_training_data(project):
    baseline, tickets = project
    with pytest.raises(ValueError, match="labelled training tickets"):
        run_analysis(baseline, tickets, AnalysisSettings(classifier="ml"))


def test_ml_pipeline_runs_and_warns_about_in_sample_scoring(project):
    baseline, tickets = project
    training = prepare_training_data(baseline, data.sample_labelled_tickets())
    result = run_analysis(
        baseline, tickets, AnalysisSettings(classifier="ml"), training_data=training
    )
    assert result.drift.sdi > 0
    assert any("in-sample" in w for w in result.warnings)


def test_empty_baseline_rejected():
    with pytest.raises(ValueError, match="empty baseline"):
        run_analysis(pd.DataFrame(), data.sample_tickets())


def test_zero_size_baseline_rejected(project):
    baseline, tickets = project
    zeroed = baseline.copy()
    zeroed["size"] = 0.0
    with pytest.raises(ValueError, match="undefined"):
        run_analysis(zeroed, tickets)


def test_missing_dates_warn_instead_of_failing(project):
    baseline, tickets = project
    undated = tickets.drop(columns=["created_date"])
    result = run_analysis(baseline, undated)
    assert result.timeline.empty
    assert any("created_date" in w for w in result.warnings)


def test_reported_numbers_are_reproducible(project):
    """The dashboard must not be the only place a number can be derived."""
    baseline, tickets = project
    first = run_analysis(baseline, tickets)
    second = run_analysis(baseline, tickets)
    assert first.drift.sdi == pytest.approx(second.drift.sdi)
    assert first.estimates.revised.person_months == pytest.approx(
        second.estimates.revised.person_months
    )
