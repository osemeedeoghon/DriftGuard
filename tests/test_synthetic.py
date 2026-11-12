"""Tests for the synthetic project generator."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from driftguard import data, pipeline, synthetic
from driftguard.config import CLASSES, NEW_SCOPE, REFINEMENT, REMOVAL


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def test_every_scenario_generates_a_usable_project():
    for scenario in synthetic.SCENARIOS:
        project = synthetic.generate_project(
            scenario, requirements=10, tickets=24, seed=1
        )
        assert len(project.baseline) == 10
        assert len(project.tickets) >= 24  # the dirty scenario appends a duplicate
        assert set(project.labelled["true_class"]) <= set(CLASSES)
        assert project.scenario == scenario


def test_generation_is_reproducible_from_the_seed():
    first = synthetic.generate_project("normal", seed=42)
    second = synthetic.generate_project("normal", seed=42)
    pd.testing.assert_frame_equal(first.tickets, second.tickets)


def test_different_seeds_give_different_projects():
    first = synthetic.generate_project("normal", seed=1)
    second = synthetic.generate_project("normal", seed=2)
    assert not first.labelled["true_class"].equals(second.labelled["true_class"])


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="Unknown scenario"):
        synthetic.generate_project("chaos")


def test_tiny_requests_are_rejected():
    with pytest.raises(ValueError, match="at least 3 requirements"):
        synthetic.generate_project("normal", requirements=2)
    with pytest.raises(ValueError, match="at least 4 tickets"):
        synthetic.generate_project("normal", tickets=3)


# --------------------------------------------------------------------------
# Scenarios actually produce the mix they advertise
# --------------------------------------------------------------------------
def test_new_scope_surge_is_dominated_by_new_scope():
    project = synthetic.generate_project("new_scope_surge", tickets=60, seed=9)
    share = (project.labelled["true_class"] == NEW_SCOPE).mean()
    assert share > 0.35


def test_refinement_heavy_is_dominated_by_refinement():
    project = synthetic.generate_project("refinement_heavy", tickets=60, seed=9)
    share = (project.labelled["true_class"] == REFINEMENT).mean()
    assert share > 0.35


def test_removals_scenario_produces_substantial_removals():
    project = synthetic.generate_project("removals", tickets=60, seed=9)
    share = (project.labelled["true_class"] == REMOVAL).mean()
    assert share > 0.20


def test_normal_scenario_is_mostly_in_scope():
    project = synthetic.generate_project("normal", tickets=60, seed=9)
    share = (project.labelled["true_class"] == "In-scope").mean()
    assert share > 0.55


# --------------------------------------------------------------------------
# Synthetic marking — the safeguard against mistaking this for evidence
# --------------------------------------------------------------------------
def test_generated_frames_are_marked_synthetic():
    project = synthetic.generate_project("normal", seed=4)
    for frame in (project.baseline, project.tickets, project.labelled):
        assert frame["synthetic"].all()
        assert frame.attrs["synthetic"] is True
        assert "SYNTHETIC" in frame.attrs["notice"]


def test_ids_are_prefixed_so_they_cannot_collide_with_real_data():
    project = synthetic.generate_project("normal", seed=4)
    assert project.baseline["requirement_id"].str.startswith("SYN-R").all()
    assert project.tickets["ticket_id"].str.startswith("SYN-T").all()


def test_csv_export_survives_a_round_trip_through_the_loaders():
    """The bug this guards: a leading '#' notice line made the export
    unreadable, because read_csv treats it as the header row."""
    project = synthetic.generate_project("normal", requirements=8, tickets=16, seed=6)

    baseline, baseline_report = data.load_baseline_csv(
        io.BytesIO(synthetic.to_csv_bytes(project.baseline))
    )
    tickets, ticket_report = data.load_tickets(
        io.BytesIO(synthetic.to_csv_bytes(project.tickets))
    )

    assert len(baseline) == 8
    assert len(tickets) == 16
    assert any("SYNTHETIC" in w for w in baseline_report.warnings)
    assert any("SYNTHETIC" in w for w in ticket_report.warnings)


def test_loaders_warn_even_when_the_file_is_re_uploaded_later():
    """The warning must come from the data, not from generator state."""
    project = synthetic.generate_project("normal", seed=6)
    csv = project.tickets.to_csv(index=False).encode("utf-8")
    _, report = data.load_tickets(io.BytesIO(csv))
    assert any("not evidence about any real project" in w for w in report.warnings)


def test_real_sample_data_is_not_flagged_as_synthetic():
    """The flag must not fire on the bundled worked example."""
    _, report = data.load_tickets(data.SAMPLE_DIR / "tickets.csv")
    assert not any("SYNTHETIC" in w for w in report.warnings)


# --------------------------------------------------------------------------
# The dirty scenario exercises the validation path
# --------------------------------------------------------------------------
def test_dirty_scenario_triggers_every_data_quality_warning():
    project = synthetic.generate_project("dirty", requirements=10, tickets=30, seed=3)
    _, report = data.load_tickets(
        io.BytesIO(project.tickets.to_csv(index=False).encode("utf-8"))
    )
    joined = " ".join(report.warnings)
    assert "duplicate" in joined
    assert "empty description" in joined
    assert "missing or non-numeric size" in joined
    assert "unrecognised date" in joined


def test_dirty_scenario_still_completes_an_analysis():
    """Dirty input must degrade gracefully, not crash the pipeline."""
    project = synthetic.generate_project("dirty", requirements=10, tickets=30, seed=3)
    baseline, _ = data.load_baseline_csv(
        io.BytesIO(project.baseline.to_csv(index=False).encode("utf-8"))
    )
    tickets, _ = data.load_tickets(
        io.BytesIO(project.tickets.to_csv(index=False).encode("utf-8"))
    )
    result = pipeline.run_analysis(baseline, tickets)
    assert len(result.classified) > 0
    assert result.estimates.revised.person_months > 0


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------
def test_generated_project_runs_through_the_full_pipeline():
    project = synthetic.generate_project("new_scope_surge", seed=11)
    baseline, _ = data.load_baseline_csv(
        io.BytesIO(project.baseline.to_csv(index=False).encode("utf-8"))
    )
    tickets, _ = data.load_tickets(
        io.BytesIO(project.tickets.to_csv(index=False).encode("utf-8"))
    )
    result = pipeline.run_analysis(baseline, tickets)
    assert result.drift.sdi > 0
    assert not result.timeline.empty


def test_scenario_catalogue_describes_every_scenario():
    catalogue = synthetic.describe_scenarios()
    assert set(catalogue["Scenario"]) == set(synthetic.SCENARIOS)
    assert catalogue["What it exercises"].str.len().min() > 20


def test_summary_counts_every_class():
    project = synthetic.generate_project("normal", seed=4)
    summary = project.summary()
    assert list(summary["Class"]) == list(CLASSES)
    assert summary["Tickets"].sum() == len(project.labelled)
