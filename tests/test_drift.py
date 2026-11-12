"""Tests for the Scope Drift Index and drift attribution."""

from __future__ import annotations

import pandas as pd
import pytest

from driftguard import drift
from driftguard.config import IN_SCOPE, NEW_SCOPE, REFINEMENT, REMOVAL


def make_tickets(rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["ticket_id", "final_class", "size", "created_date"]
    )


# --------------------------------------------------------------------------
# Per-ticket contribution
# --------------------------------------------------------------------------
def test_in_scope_contributes_nothing():
    assert drift.contribution(IN_SCOPE, 13, 0.5) == (0.0, 0.0)


def test_new_scope_adds_full_size():
    assert drift.contribution(NEW_SCOPE, 8, 0.5) == (8.0, 0.0)


def test_removal_removes_full_size():
    assert drift.contribution(REMOVAL, 5, 0.5) == (0.0, 5.0)


def test_refinement_adds_weighted_size():
    assert drift.contribution(REFINEMENT, 10, 0.5) == (5.0, 0.0)
    assert drift.contribution(REFINEMENT, 10, 0.0) == (0.0, 0.0)
    assert drift.contribution(REFINEMENT, 10, 1.0) == (10.0, 0.0)


def test_unknown_classification_rejected_not_ignored():
    with pytest.raises(ValueError, match="Unknown classification"):
        drift.contribution("Maybe", 5, 0.5)


def test_negative_ticket_size_rejected_with_guidance():
    with pytest.raises(ValueError, match="Removal"):
        drift.contribution(NEW_SCOPE, -4, 0.5)


# --------------------------------------------------------------------------
# SDI
# --------------------------------------------------------------------------
def test_sdi_matches_the_definition():
    tickets = make_tickets(
        [
            ("T-1", IN_SCOPE, 20, "2025-01-05"),
            ("T-2", NEW_SCOPE, 15, "2025-01-10"),
            ("T-3", REFINEMENT, 10, "2025-01-12"),   # 0.5 * 10 = 5 added
            ("T-4", REMOVAL, 8, "2025-01-15"),
        ]
    )
    result = drift.calculate_drift(tickets, baseline_size=100.0, refinement_weight=0.5)
    assert result.size_added == pytest.approx(20.0)   # 15 + 5
    assert result.size_removed == pytest.approx(8.0)
    assert result.sdi == pytest.approx((20.0 - 8.0) / 100.0)
    assert result.sdi == pytest.approx(0.12)
    assert result.drift_percent == pytest.approx(12.0)


def test_revised_size_is_baseline_times_one_plus_sdi():
    tickets = make_tickets([("T-1", NEW_SCOPE, 30, "2025-02-01")])
    result = drift.calculate_drift(tickets, baseline_size=120.0)
    assert result.revised_size == pytest.approx(150.0)
    assert result.revised_size == pytest.approx(120.0 * (1 + result.sdi))


def test_net_de_scoping_yields_negative_sdi():
    tickets = make_tickets(
        [("T-1", REMOVAL, 25, "2025-03-01"), ("T-2", NEW_SCOPE, 5, "2025-03-02")]
    )
    result = drift.calculate_drift(tickets, baseline_size=100.0)
    assert result.sdi == pytest.approx(-0.20)
    assert result.revised_size == pytest.approx(80.0)


def test_all_in_scope_tickets_produce_zero_drift():
    tickets = make_tickets(
        [("T-1", IN_SCOPE, 40, "2025-01-01"), ("T-2", IN_SCOPE, 60, "2025-01-02")]
    )
    result = drift.calculate_drift(tickets, baseline_size=100.0)
    assert result.sdi == 0.0
    assert result.revised_size == pytest.approx(100.0)


def test_empty_ticket_set_produces_zero_drift():
    tickets = pd.DataFrame(columns=["ticket_id", "final_class", "size", "created_date"])
    result = drift.calculate_drift(tickets, baseline_size=50.0)
    assert result.sdi == 0.0
    assert result.revised_size == pytest.approx(50.0)
    assert len(result.per_class) == 4


def test_zero_baseline_rejected_because_sdi_is_undefined():
    tickets = make_tickets([("T-1", NEW_SCOPE, 5, "2025-01-01")])
    with pytest.raises(ValueError, match="undefined"):
        drift.calculate_drift(tickets, baseline_size=0.0)


def test_refinement_weight_must_be_a_fraction():
    tickets = make_tickets([("T-1", REFINEMENT, 5, "2025-01-01")])
    with pytest.raises(ValueError, match="between 0 and 1"):
        drift.calculate_drift(tickets, baseline_size=10.0, refinement_weight=1.5)


def test_missing_column_names_itself():
    tickets = pd.DataFrame({"ticket_id": ["T-1"], "size": [3]})
    with pytest.raises(ValueError, match="final_class"):
        drift.calculate_drift(tickets, baseline_size=10.0)


def test_non_numeric_size_names_the_offending_ticket():
    tickets = make_tickets([("T-7", NEW_SCOPE, "large", "2025-01-01")])
    with pytest.raises(ValueError, match="T-7"):
        drift.calculate_drift(tickets, baseline_size=10.0)


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------
def test_per_class_totals_cover_every_class():
    tickets = make_tickets([("T-1", NEW_SCOPE, 5, "2025-01-01")])
    result = drift.calculate_drift(tickets, baseline_size=10.0)
    assert set(result.per_class["classification"]) == {
        IN_SCOPE,
        REFINEMENT,
        NEW_SCOPE,
        REMOVAL,
    }
    new_row = result.per_class.set_index("classification").loc[NEW_SCOPE]
    assert new_row["added"] == pytest.approx(5.0)
    assert new_row["tickets"] == 1


def test_top_contributors_ranks_by_absolute_net_effect():
    tickets = make_tickets(
        [
            ("T-1", NEW_SCOPE, 3, "2025-01-01"),
            ("T-2", REMOVAL, 21, "2025-01-02"),
            ("T-3", NEW_SCOPE, 13, "2025-01-03"),
            ("T-4", IN_SCOPE, 40, "2025-01-04"),
        ]
    )
    result = drift.calculate_drift(tickets, baseline_size=100.0)
    top = result.top_contributors(2)
    assert list(top["ticket_id"]) == ["T-2", "T-3"]


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------
def test_cumulative_sdi_accumulates_in_date_order():
    tickets = make_tickets(
        [
            ("T-2", NEW_SCOPE, 10, "2025-01-20"),
            ("T-1", NEW_SCOPE, 10, "2025-01-10"),
            ("T-3", REMOVAL, 5, "2025-01-30"),
        ]
    )
    timeline = drift.drift_over_time(tickets, baseline_size=100.0)
    assert list(timeline["cumulative_sdi"].round(4)) == [0.10, 0.20, 0.15]
    assert list(timeline["projected_size"].round(2)) == [110.0, 120.0, 115.0]


def test_same_day_tickets_collapse_into_one_timeline_point():
    tickets = make_tickets(
        [
            ("T-1", NEW_SCOPE, 10, "2025-01-10"),
            ("T-2", NEW_SCOPE, 10, "2025-01-10"),
        ]
    )
    timeline = drift.drift_over_time(tickets, baseline_size=100.0)
    assert len(timeline) == 1
    assert timeline["cumulative_sdi"].iloc[0] == pytest.approx(0.20)


def test_undated_tickets_are_excluded_from_timeline_and_reported():
    tickets = make_tickets(
        [
            ("T-1", NEW_SCOPE, 10, "2025-01-10"),
            ("T-2", NEW_SCOPE, 10, "not a date"),
        ]
    )
    timeline = drift.drift_over_time(tickets, baseline_size=100.0)
    assert timeline.attrs["excluded_tickets"] == 1
    assert timeline["cumulative_sdi"].iloc[-1] == pytest.approx(0.10)
    # The overall SDI still counts the undated ticket.
    overall = drift.calculate_drift(tickets, baseline_size=100.0)
    assert overall.sdi == pytest.approx(0.20)


def test_timeline_requires_a_date_column():
    tickets = pd.DataFrame(
        {"ticket_id": ["T-1"], "final_class": [NEW_SCOPE], "size": [5]}
    )
    with pytest.raises(ValueError, match="created_date"):
        drift.drift_over_time(tickets, baseline_size=10.0)


# --------------------------------------------------------------------------
# REVL bridge
# --------------------------------------------------------------------------
def test_revl_from_positive_drift():
    assert drift.revl_from_sdi(0.18) == pytest.approx(18.0)


def test_revl_clamped_at_zero_for_net_de_scoping():
    assert drift.revl_from_sdi(-0.25) == 0.0
