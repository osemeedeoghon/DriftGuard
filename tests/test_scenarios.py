"""Tests for scenario planning and sensitivity analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from driftguard import data, pipeline, scenarios
from driftguard.cocomo import CocomoDrivers, SizeConversion
from driftguard.config import IN_SCOPE
from driftguard.pipeline import AnalysisSettings


@pytest.fixture(scope="module")
def analysis():
    return pipeline.run_analysis(data.sample_baseline(), data.sample_tickets())


@pytest.fixture(scope="module")
def settings(analysis):
    return analysis.settings


# --------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------
def test_standard_scenarios_separate_records_from_hypotheticals(analysis):
    built = scenarios.build_scenarios(analysis)
    recorded = [s for s in built if not s.hypothetical]
    hypothetical = [s for s in built if s.hypothetical]

    assert [s.name for s in recorded] == ["Agreed baseline", "Current classification"]
    assert len(hypothetical) == 3
    assert all(s.description for s in built)


def test_recorded_scenarios_match_the_analysis(analysis):
    built = {s.name: s for s in scenarios.build_scenarios(analysis)}
    assert built["Agreed baseline"].size == pytest.approx(analysis.drift.baseline_size)
    assert built["Current classification"].size == pytest.approx(
        analysis.drift.revised_size
    )


def test_rejecting_new_scope_lands_between_baseline_and_current(analysis):
    built = {s.name: s for s in scenarios.build_scenarios(analysis)}
    rejected = built["Reject all new scope"].size
    assert rejected < built["Current classification"].size
    # Removals still apply, so it can fall below the baseline.
    assert rejected == pytest.approx(
        analysis.drift.baseline_size
        + analysis.drift.per_ticket.loc[
            analysis.drift.per_ticket[analysis.drift.classification_column]
            == "Refinement",
            "added",
        ].sum()
        - analysis.drift.size_removed
    )


def test_pessimistic_scenario_is_the_largest(analysis):
    built = scenarios.build_scenarios(analysis)
    sizes = {s.name: s.size for s in built}
    assert sizes["Accept all, remove nothing"] == max(sizes.values())


def test_scenarios_can_omit_the_hypotheticals(analysis):
    built = scenarios.build_scenarios(analysis, include_pending=False)
    assert len(built) == 2
    assert not any(s.hypothetical for s in built)


# --------------------------------------------------------------------------
# Costing
# --------------------------------------------------------------------------
def test_comparison_is_relative_to_the_reference(analysis, settings):
    built = scenarios.build_scenarios(analysis)
    comparison = scenarios.compare_scenarios(
        built,
        conversion=settings.conversion,
        size_unit=settings.size_unit,
        drivers=settings.drivers,
        reference="Agreed baseline",
    )
    reference_row = comparison.table.loc[
        comparison.table["Scenario"] == "Agreed baseline"
    ].iloc[0]
    assert reference_row["Δ Effort (PM)"] == pytest.approx(0.0)
    assert reference_row["Δ Duration (months)"] == pytest.approx(0.0)


def test_current_classification_matches_the_pipeline_estimate(analysis, settings):
    """The scenario engine must not produce a different number for the same
    scope than the main pipeline does."""
    built = scenarios.build_scenarios(analysis)
    comparison = scenarios.compare_scenarios(
        built,
        conversion=settings.conversion,
        size_unit=settings.size_unit,
        drivers=settings.drivers,
    )
    row = comparison.table.loc[
        comparison.table["Scenario"] == "Current classification"
    ].iloc[0]
    assert row["Effort (PM)"] == pytest.approx(
        analysis.estimates.revised.person_months, rel=1e-9
    )
    assert row["Duration (months)"] == pytest.approx(
        analysis.estimates.revised.duration_months, rel=1e-9
    )


def test_baseline_scenario_matches_the_pipeline_baseline(analysis, settings):
    comparison = scenarios.compare_scenarios(
        scenarios.build_scenarios(analysis),
        conversion=settings.conversion,
        size_unit=settings.size_unit,
        drivers=settings.drivers,
    )
    row = comparison.table.loc[comparison.table["Scenario"] == "Agreed baseline"].iloc[0]
    assert row["Effort (PM)"] == pytest.approx(
        analysis.estimates.baseline.person_months, rel=1e-9
    )


def test_cost_columns_appear_only_with_a_rate(analysis, settings):
    built = scenarios.build_scenarios(analysis)
    without = scenarios.compare_scenarios(
        built,
        conversion=settings.conversion,
        size_unit=settings.size_unit,
        drivers=settings.drivers,
    )
    assert "Cost estimate" not in without.table.columns
    assert any("No labour rate" in note for note in without.notes)

    with_rate = scenarios.compare_scenarios(
        built,
        conversion=settings.conversion,
        size_unit=settings.size_unit,
        drivers=settings.drivers,
        rate=10000.0,
    )
    assert "Cost estimate" in with_rate.table.columns
    assert any("not an actual or committed cost" in note for note in with_rate.notes)


def test_unknown_reference_is_rejected(analysis, settings):
    with pytest.raises(ValueError, match="not in the comparison"):
        scenarios.compare_scenarios(
            scenarios.build_scenarios(analysis),
            conversion=settings.conversion,
            size_unit=settings.size_unit,
            drivers=settings.drivers,
            reference="Nonexistent",
        )


def test_empty_scenario_list_is_rejected(settings):
    with pytest.raises(ValueError, match="No scenarios"):
        scenarios.compare_scenarios(
            [],
            conversion=settings.conversion,
            size_unit=settings.size_unit,
            drivers=settings.drivers,
        )


def test_hypothetical_names_are_exposed(analysis, settings):
    comparison = scenarios.compare_scenarios(
        scenarios.build_scenarios(analysis),
        conversion=settings.conversion,
        size_unit=settings.size_unit,
        drivers=settings.drivers,
    )
    assert "Accept all, remove nothing" in comparison.hypothetical_names
    assert "Agreed baseline" not in comparison.hypothetical_names


# --------------------------------------------------------------------------
# Sensitivity sweeps
# --------------------------------------------------------------------------
def test_sloc_sweep_is_monotonic_in_effort(settings):
    sweep = scenarios.sweep_sloc_per_point(
        276.0, size_unit="story_points", drivers=settings.drivers
    )
    assert sweep["Effort (PM)"].is_monotonic_increasing
    assert sweep["KSLOC"].is_monotonic_increasing


def test_effort_elasticity_equals_the_cocomo_exponent(settings):
    """The strongest available correctness check on the sweep.

    Effort is A x Size^E, so in logs the elasticity of effort with respect to
    any multiplicative size input is exactly E.
    """
    sweep = scenarios.sweep_sloc_per_point(
        276.0, size_unit="story_points", drivers=settings.drivers
    )
    measured = scenarios.elasticity(sweep, "SLOC per story point", "Effort (PM)")
    assert measured == pytest.approx(settings.drivers.exponent_e(), abs=1e-9)


def test_duration_elasticity_equals_e_times_f(settings):
    """TDEV is C x PM_NS^F and PM_NS is A x Size^E, so the elasticity is E x F."""
    sweep = scenarios.sweep_sloc_per_point(
        276.0, size_unit="story_points", drivers=settings.drivers
    )
    measured = scenarios.elasticity(sweep, "SLOC per story point", "Duration (months)")
    expected = settings.drivers.exponent_e() * settings.drivers.exponent_f()
    assert measured == pytest.approx(expected, abs=1e-9)


def test_elasticity_uses_logs_not_an_arc_ratio(settings):
    """Guards the bug this function was written to avoid.

    The arc ratio over a 20-to-120 sweep reads about 1.24; the true elasticity
    is the exponent 1.0997. Reporting the arc ratio would imply an exponent
    the model does not have.
    """
    sweep = scenarios.sweep_sloc_per_point(
        276.0, size_unit="story_points", drivers=settings.drivers
    )
    first, last = sweep.iloc[0], sweep.iloc[-1]
    arc = ((last["Effort (PM)"] / first["Effort (PM)"]) - 1) / (
        (last["SLOC per story point"] / first["SLOC per story point"]) - 1
    )
    measured = scenarios.elasticity(sweep, "SLOC per story point", "Effort (PM)")
    assert abs(arc - measured) > 0.1
    assert measured == pytest.approx(1.0997, abs=1e-4)


def test_elasticity_handles_degenerate_input():
    assert scenarios.elasticity(pd.DataFrame({"x": [1], "y": [1]}), "x", "y") == 0.0
    assert (
        scenarios.elasticity(pd.DataFrame({"x": [0, 1], "y": [1, 2]}), "x", "y") == 0.0
    )


def test_revl_sweep_raises_effort_without_changing_nominal_size(settings):
    sweep = scenarios.sweep_revl(
        276.0,
        conversion=settings.conversion,
        size_unit=settings.size_unit,
        drivers=settings.drivers,
    )
    assert sweep["Effort (PM)"].is_monotonic_increasing
    assert sweep["Effective KSLOC"].is_monotonic_increasing
    assert sweep["REVL %"].iloc[0] == pytest.approx(0.0)


def test_cost_appears_in_sweeps_only_with_a_rate(settings):
    without = scenarios.sweep_sloc_per_point(
        276.0, size_unit="story_points", drivers=settings.drivers
    )
    assert "Cost estimate" not in without.columns
    with_rate = scenarios.sweep_sloc_per_point(
        276.0, size_unit="story_points", drivers=settings.drivers, rate=9000.0
    )
    assert "Cost estimate" in with_rate.columns


# --------------------------------------------------------------------------
# Driver sweeps
# --------------------------------------------------------------------------
def test_driver_sweep_covers_every_rating(settings):
    sweep = scenarios.sweep_driver(
        "CPLX",
        276.0,
        conversion=settings.conversion,
        size_unit=settings.size_unit,
        drivers=settings.drivers,
    )
    assert len(sweep) == 6  # VL..XH
    assert sweep["Effort (PM)"].is_monotonic_increasing


def test_sced_sweep_shows_effort_and_duration_moving_oppositely(settings):
    """Compressing a schedule costs more effort and less calendar time.

    This exercises the two distinct SCED tables, whose conflation was a real
    bug earlier in the project's history.
    """
    sweep = scenarios.sweep_driver(
        "SCED",
        276.0,
        conversion=settings.conversion,
        size_unit=settings.size_unit,
        drivers=settings.drivers,
    ).set_index("Code")

    assert sweep.loc["VL", "Effort (PM)"] > sweep.loc["N", "Effort (PM)"]
    assert sweep.loc["VL", "Duration (months)"] < sweep.loc["N", "Duration (months)"]
    # Stretching costs no extra effort but extends the calendar.
    assert sweep.loc["VH", "Effort (PM)"] == pytest.approx(sweep.loc["N", "Effort (PM)"])
    assert sweep.loc["VH", "Duration (months)"] > sweep.loc["N", "Duration (months)"]
    assert sweep.attrs["is_sced"] is True


def test_unknown_driver_is_rejected(settings):
    with pytest.raises(ValueError, match="Unknown COCOMO driver"):
        scenarios.sweep_driver(
            "NOPE",
            276.0,
            conversion=settings.conversion,
            size_unit=settings.size_unit,
            drivers=settings.drivers,
        )


# --------------------------------------------------------------------------
# Tornado
# --------------------------------------------------------------------------
def test_tornado_ranks_drivers_by_effort_swing(settings):
    chart = scenarios.tornado(
        276.0,
        conversion=settings.conversion,
        size_unit=settings.size_unit,
        drivers=settings.drivers,
        top_n=5,
    )
    assert len(chart) == 5
    assert chart["Swing (PM)"].is_monotonic_decreasing
    assert (chart["Highest effort (PM)"] >= chart["Lowest effort (PM)"]).all()
    # Product complexity has the widest published range, so it should lead.
    assert chart.iloc[0]["Driver"] == "CPLX"


def test_tornado_swings_are_positive(settings):
    chart = scenarios.tornado(
        276.0,
        conversion=settings.conversion,
        size_unit=settings.size_unit,
        drivers=settings.drivers,
    )
    assert (chart["Swing (PM)"] > 0).all()


# --------------------------------------------------------------------------
# Scenarios respond to reviewer overrides
# --------------------------------------------------------------------------
def test_scenarios_reflect_human_overrides():
    baseline, tickets = data.sample_baseline(), data.sample_tickets()
    plain = pipeline.run_analysis(baseline, tickets)
    overridden = pipeline.run_analysis(
        baseline, tickets, overrides={"T-301": IN_SCOPE}
    )
    plain_current = {s.name: s.size for s in scenarios.build_scenarios(plain)}
    over_current = {s.name: s.size for s in scenarios.build_scenarios(overridden)}
    assert over_current["Current classification"] < plain_current["Current classification"]
