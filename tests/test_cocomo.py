"""Tests for the COCOMO II Post-Architecture implementation.

Expected values are computed from the published equations by hand inside the
tests themselves, so a regression in the module cannot quietly rewrite the
yardstick it is measured against.
"""

from __future__ import annotations

import math

import pytest

from driftguard import cocomo
from driftguard.cocomo import (
    A_CONSTANT,
    B_CONSTANT,
    C_CONSTANT,
    D_CONSTANT,
    CocomoDrivers,
    SizeConversion,
)


# --------------------------------------------------------------------------
# Exponents
# --------------------------------------------------------------------------
def test_nominal_scale_factor_sum():
    # PREC 3.72 + FLEX 3.04 + RESL 4.24 + TEAM 3.29 + PMAT 4.68
    assert CocomoDrivers().scale_factor_sum() == pytest.approx(18.97)


def test_nominal_exponent_e():
    # E = B + 0.01 * 18.97
    assert CocomoDrivers().exponent_e() == pytest.approx(0.91 + 0.1897)


def test_nominal_exponent_f():
    # F = D + 0.2 * (E - B)
    drivers = CocomoDrivers()
    assert drivers.exponent_f() == pytest.approx(0.28 + 0.2 * 0.1897)


def test_exponent_e_rises_as_scale_factors_worsen():
    bad = CocomoDrivers(ratings={k: "VL" for k in cocomo.SCALE_FACTORS})
    good = CocomoDrivers(ratings={k: "XH" for k in cocomo.SCALE_FACTORS})
    assert bad.exponent_e() > CocomoDrivers().exponent_e() > good.exponent_e()
    # All scale factors XH -> sum 0 -> E collapses to B exactly.
    assert good.exponent_e() == pytest.approx(B_CONSTANT)


# --------------------------------------------------------------------------
# Effort
# --------------------------------------------------------------------------
def test_nominal_effort_matches_published_equation():
    size = 100.0
    e = 0.91 + 0.1897
    expected = A_CONSTANT * size**e
    result = cocomo.estimate(size)
    assert result.person_months == pytest.approx(expected)
    # Sanity anchor: the literature's nominal 100 KSLOC project is ~465 PM.
    assert 460 < result.person_months < 470


def test_nominal_duration_matches_published_equation():
    size = 100.0
    e = B_CONSTANT + 0.1897
    f = D_CONSTANT + 0.2 * 0.1897
    pm_ns = A_CONSTANT * size**e
    expected = C_CONSTANT * pm_ns**f
    result = cocomo.estimate(size)
    assert result.duration_months == pytest.approx(expected)
    assert 25 < result.duration_months < 27


def test_effort_multipliers_multiply_into_effort():
    nominal = cocomo.estimate(50.0)
    # CPLX High = 1.17, ACAP High = 0.85 -> product 0.9945
    drivers = CocomoDrivers(ratings={"CPLX": "H", "ACAP": "H"})
    adjusted = cocomo.estimate(50.0, drivers)
    assert adjusted.em_product == pytest.approx(1.17 * 0.85)
    assert adjusted.person_months == pytest.approx(
        nominal.person_months * 1.17 * 0.85
    )


def test_effort_is_superlinear_in_size():
    """E > 1 means doubling size more than doubles effort (diseconomy of scale)."""
    single = cocomo.estimate(50.0).person_months
    double = cocomo.estimate(100.0).person_months
    assert double > 2 * single


def test_zero_size_yields_zero_effort_and_duration():
    result = cocomo.estimate(0.0)
    assert result.person_months == 0.0
    assert result.duration_months == 0.0
    assert result.average_staff == 0.0


def test_negative_size_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        cocomo.estimate(-1.0)


# --------------------------------------------------------------------------
# SCED: affects effort, and is excluded from the PM used for duration
# --------------------------------------------------------------------------
def test_sced_excluded_from_nominal_schedule_effort():
    drivers = CocomoDrivers(ratings={"SCED": "VL"})
    result = cocomo.estimate(100.0, drivers)
    assert result.em_product == pytest.approx(1.43)
    assert result.em_product_ns == pytest.approx(1.0)
    assert result.person_months == pytest.approx(
        result.person_months_nominal_schedule * 1.43
    )


def test_sced_compresses_duration_and_raises_effort():
    nominal = cocomo.estimate(100.0)
    crashed = cocomo.estimate(100.0, CocomoDrivers(ratings={"SCED": "VL"}))
    # Accelerating the schedule costs more effort but buys a shorter calendar.
    # The effort penalty (EM 1.43) and the calendar compression (75% of
    # nominal) come from two different published tables.
    assert crashed.person_months == pytest.approx(nominal.person_months * 1.43)
    assert crashed.duration_months < nominal.duration_months
    assert crashed.duration_months == pytest.approx(nominal.duration_months * 0.75)


def test_stretched_schedule_lengthens_calendar_without_effort_penalty():
    nominal = cocomo.estimate(100.0)
    stretched = cocomo.estimate(100.0, CocomoDrivers(ratings={"SCED": "VH"}))
    assert stretched.person_months == pytest.approx(nominal.person_months)
    assert stretched.duration_months == pytest.approx(nominal.duration_months * 1.60)


def test_sced_tables_cover_the_same_ratings():
    assert set(cocomo.SCED_SCHEDULE_PERCENT) == set(cocomo.EFFORT_MULTIPLIERS["SCED"])


# --------------------------------------------------------------------------
# REVL
# --------------------------------------------------------------------------
def test_revl_inflates_size_before_the_effort_equation():
    drivers = CocomoDrivers(revl_percent=25.0)
    result = cocomo.estimate(80.0, drivers)
    assert result.size_ksloc == pytest.approx(80.0)
    assert result.effective_ksloc == pytest.approx(100.0)
    assert result.person_months == pytest.approx(cocomo.estimate(100.0).person_months)


def test_negative_revl_rejected():
    with pytest.raises(ValueError, match="REVL"):
        CocomoDrivers(revl_percent=-5.0)


# --------------------------------------------------------------------------
# Effort vs duration must stay distinct
# --------------------------------------------------------------------------
def test_effort_and_duration_are_distinct_quantities():
    result = cocomo.estimate(100.0)
    assert not math.isclose(result.person_months, result.duration_months)
    assert result.average_staff == pytest.approx(
        result.person_months / result.duration_months
    )


def test_person_hours_uses_declared_convention():
    result = cocomo.estimate(10.0)
    assert result.person_hours == pytest.approx(result.person_months * 152)


def test_labour_cost_requires_a_supplied_rate():
    result = cocomo.estimate(10.0)
    assert result.labour_cost(10_000) == pytest.approx(result.person_months * 10_000)
    with pytest.raises(ValueError):
        result.labour_cost(-1)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_unknown_driver_rejected():
    with pytest.raises(ValueError, match="Unknown COCOMO driver"):
        CocomoDrivers(ratings={"NOPE": "N"})


def test_rating_not_defined_for_driver_rejected():
    # DATA has no Very Low rating in the published table.
    with pytest.raises(ValueError, match="not defined for driver"):
        CocomoDrivers(ratings={"DATA": "VL"})


def test_every_driver_is_one_at_nominal():
    for code, table in cocomo.EFFORT_MULTIPLIERS.items():
        assert table["N"] == 1.00, f"{code} nominal must be 1.00"


def test_every_driver_has_a_display_name():
    codes = set(cocomo.SCALE_FACTORS) | set(cocomo.EFFORT_MULTIPLIERS)
    assert codes == set(cocomo.DRIVER_NAMES)


# --------------------------------------------------------------------------
# Size conversion
# --------------------------------------------------------------------------
def test_story_point_conversion_is_explicit():
    conv = SizeConversion(sloc_per_story_point=50.0)
    assert conv.to_ksloc(100, "story_points") == pytest.approx(5.0)


def test_function_point_conversion_uses_gearing_ratio():
    conv = SizeConversion(sloc_per_function_point=40.0)
    assert conv.to_ksloc(250, "function_points") == pytest.approx(10.0)


def test_sloc_and_ksloc_passthrough():
    conv = SizeConversion()
    assert conv.to_ksloc(2500, "sloc") == pytest.approx(2.5)
    assert conv.to_ksloc(2.5, "ksloc") == pytest.approx(2.5)


def test_unknown_unit_rejected_with_guidance():
    with pytest.raises(ValueError, match="Unknown size unit"):
        SizeConversion().to_ksloc(10, "bananas")


def test_negative_size_conversion_rejected():
    with pytest.raises(ValueError):
        SizeConversion().to_ksloc(-3, "story_points")


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def test_comparison_reports_deltas_and_percentages():
    comparison = cocomo.compare(100.0, 120.0)
    assert comparison.delta_ksloc == pytest.approx(20.0)
    assert comparison.size_change_percent == pytest.approx(20.0)
    assert comparison.delta_person_months > 0
    assert comparison.delta_duration_months > 0
    # Diseconomy of scale: 20% more code costs more than 20% more effort.
    assert comparison.effort_change_percent > 20.0


def test_comparison_handles_de_scoping():
    comparison = cocomo.compare(100.0, 90.0)
    assert comparison.delta_person_months < 0
    assert comparison.effort_change_percent < 0


def test_revised_run_can_carry_its_own_revl():
    comparison = cocomo.compare(
        100.0,
        100.0,
        drivers=CocomoDrivers(),
        revised_drivers=CocomoDrivers(revl_percent=30.0),
    )
    assert comparison.delta_ksloc == pytest.approx(0.0)
    # Same nominal size, but volatility rework pushes effort up.
    assert comparison.delta_person_months > 0
