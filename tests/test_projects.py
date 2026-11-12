"""Tests for the in-session project store and scope history."""

from __future__ import annotations

import pandas as pd
import pytest

from driftguard import data, pipeline, projects
from driftguard.config import IN_SCOPE
from driftguard.pipeline import AnalysisSettings


@pytest.fixture
def store():
    return projects.ProjectStore()


@pytest.fixture(scope="module")
def result():
    return pipeline.run_analysis(data.sample_baseline(), data.sample_tickets())


def make_project(name: str = "Alpha") -> projects.ProjectRecord:
    return projects.ProjectRecord(
        name=name,
        description="test project",
        baseline=data.sample_baseline(),
        tickets=data.sample_tickets(),
        labelled=data.sample_labelled_tickets(),
        source="sample",
    )


# --------------------------------------------------------------------------
# Project records
# --------------------------------------------------------------------------
def test_project_reports_readiness_and_labels():
    project = make_project()
    assert project.is_ready is True
    assert project.has_labels is True

    empty = projects.ProjectRecord(name="Empty")
    assert empty.is_ready is False
    assert empty.has_labels is False


def test_project_ids_are_unique():
    assert make_project("A").project_id != make_project("B").project_id


def test_summary_describes_the_datasets():
    summary = make_project("Alpha").summary()
    assert summary["Project"] == "Alpha"
    assert summary["Requirements"] == 18
    assert summary["Tickets"] == 45
    assert summary["Labelled"] is True


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
def test_adding_a_project_activates_it(store):
    pid = store.add(make_project("Alpha"))
    assert store.active_id == pid
    assert store.active.name == "Alpha"
    assert len(store) == 1


def test_second_project_can_be_added_without_activating(store):
    store.add(make_project("Alpha"))
    second = store.add(make_project("Beta"), activate=False)
    assert store.active.name == "Alpha"
    store.activate(second)
    assert store.active.name == "Beta"


def test_duplicate_names_are_rejected(store):
    store.add(make_project("Alpha"))
    with pytest.raises(ValueError, match="already exists"):
        store.add(make_project("alpha"))


def test_unnamed_project_is_rejected(store):
    with pytest.raises(ValueError, match="needs a name"):
        store.add(projects.ProjectRecord(name="   "))


def test_removing_the_active_project_activates_another(store):
    first = store.add(make_project("Alpha"))
    store.add(make_project("Beta"), activate=False)
    store.remove(first)
    assert store.active is not None
    assert store.active.name == "Beta"


def test_removing_the_last_project_leaves_none(store):
    pid = store.add(make_project("Alpha"))
    store.remove(pid)
    assert store.active is None
    assert len(store) == 0


def test_activating_an_unknown_project_raises(store):
    with pytest.raises(KeyError):
        store.activate("nope")


def test_catalogue_marks_the_active_project(store):
    store.add(make_project("Alpha"))
    store.add(make_project("Beta"), activate=False)
    catalogue = store.catalogue()
    assert len(catalogue) == 2
    assert catalogue.loc[catalogue["Project"] == "Alpha", "Active"].iloc[0]
    assert not catalogue.loc[catalogue["Project"] == "Beta", "Active"].iloc[0]


def test_empty_catalogue_has_the_right_columns(store):
    assert store.catalogue().empty
    assert "Project" in store.catalogue().columns


# --------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------
def test_snapshot_captures_the_analysis_figures(result):
    snapshot = projects.snapshot_from_result(result, label="Initial")
    assert snapshot.label == "Initial"
    assert snapshot.sdi == pytest.approx(result.drift.sdi)
    assert snapshot.revised_size == pytest.approx(result.drift.revised_size)
    assert snapshot.person_months == pytest.approx(
        result.estimates.revised.person_months
    )
    assert snapshot.sloc_per_point == pytest.approx(
        result.settings.conversion.sloc_per_story_point
    )


def test_snapshot_records_top_contributors(result):
    snapshot = projects.snapshot_from_result(result, label="Initial")
    assert len(snapshot.contributors) == 5
    ticket_id, contribution = snapshot.contributors[0]
    assert ticket_id.startswith("T-")
    assert abs(contribution) > 0


def test_recording_requires_an_active_project(store, result):
    snapshot = projects.snapshot_from_result(result, label="Initial")
    with pytest.raises(ValueError, match="No active project"):
        store.record(snapshot)


def test_history_is_per_project(store, result):
    first = store.add(make_project("Alpha"))
    second = store.add(make_project("Beta"), activate=False)

    store.record(projects.snapshot_from_result(result, label="A1"), project_id=first)
    store.record(projects.snapshot_from_result(result, label="A2"), project_id=first)
    store.record(projects.snapshot_from_result(result, label="B1"), project_id=second)

    assert len(store.history(first)) == 2
    assert len(store.history(second)) == 1
    assert [s.label for s in store.history(first)] == ["A1", "A2"]


def test_history_frame_is_tabular(store, result):
    store.add(make_project("Alpha"))
    store.record(projects.snapshot_from_result(result, label="Initial", note="first run"))
    frame = store.history_frame()
    assert len(frame) == 1
    assert frame.iloc[0]["Snapshot"] == "Initial"
    assert frame.iloc[0]["Note"] == "first run"


def test_empty_history_frame_has_the_right_columns(store):
    store.add(make_project("Alpha"))
    frame = store.history_frame()
    assert frame.empty
    assert "SDI" in frame.columns


def test_clearing_history_keeps_the_project(store, result):
    store.add(make_project("Alpha"))
    store.record(projects.snapshot_from_result(result, label="Initial"))
    store.clear_history()
    assert store.history() == []
    assert len(store) == 1


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def test_comparison_shows_the_change_between_snapshots():
    baseline, tickets = data.sample_baseline(), data.sample_tickets()
    before = pipeline.run_analysis(baseline, tickets)
    after = pipeline.run_analysis(baseline, tickets, overrides={"T-301": IN_SCOPE})

    first = projects.snapshot_from_result(before, label="Before review")
    second = projects.snapshot_from_result(after, label="After review")
    comparison = projects.compare_snapshots(first, second)

    sdi_row = comparison.loc[comparison["Measure"] == "Scope Drift Index"].iloc[0]
    assert sdi_row["Change"] < 0  # the override reduced drift
    overturned = comparison.loc[
        comparison["Measure"] == "Classifications overturned"
    ].iloc[0]
    assert overturned["Change"] == 1


def test_comparison_separates_outcomes_from_assumptions():
    result = pipeline.run_analysis(data.sample_baseline(), data.sample_tickets())
    snapshot = projects.snapshot_from_result(result, label="Only")
    comparison = projects.compare_snapshots(snapshot, snapshot)
    assert set(comparison["Kind"]) >= {"ratio", "size", "assumption"}
    assert (comparison["Change"] == 0).all()


def test_assumption_changes_are_detected():
    baseline, tickets = data.sample_baseline(), data.sample_tickets()
    first = projects.snapshot_from_result(
        pipeline.run_analysis(baseline, tickets), label="Default"
    )
    second = projects.snapshot_from_result(
        pipeline.run_analysis(
            baseline, tickets, AnalysisSettings(refinement_weight=0.9)
        ),
        label="Heavier refinement",
    )
    changes = projects.assumptions_changed(first, second)
    assert any("refinement weight" in c for c in changes)


def test_no_assumption_change_is_reported_when_settings_match(result):
    snapshot = projects.snapshot_from_result(result, label="Only")
    assert projects.assumptions_changed(snapshot, snapshot) == []


def test_sloc_factor_change_is_detected():
    from driftguard.cocomo import SizeConversion

    baseline, tickets = data.sample_baseline(), data.sample_tickets()
    first = projects.snapshot_from_result(
        pipeline.run_analysis(baseline, tickets), label="60"
    )
    second = projects.snapshot_from_result(
        pipeline.run_analysis(
            baseline,
            tickets,
            AnalysisSettings(conversion=SizeConversion(sloc_per_story_point=30.0)),
        ),
        label="30",
    )
    changes = projects.assumptions_changed(first, second)
    assert any("SLOC per story point" in c for c in changes)
