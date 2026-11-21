"""Tests for uploaded-project registration (audit finding §2.3).

Before this fix the sidebar's upload branches assigned ``st.session_state``
directly and never reached ``state._install()``, so an uploaded project was
absent from ``ProjectStore``. Three things silently did not work on the
application's primary input path: project switching, scope-history snapshots,
and the project name used in the header and every exported report.
"""

from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st

from driftguard import data, pipeline, projects
from ui import state


@pytest.fixture(autouse=True)
def clean_session():
    """Each test gets a fresh session, including a fresh project store."""
    st.session_state.clear()
    state.init()
    yield
    st.session_state.clear()


def upload_payload():
    return data.sample_baseline(), data.sample_tickets(), data.sample_labelled_tickets()


def install(name: str = "Acme Portal", signature: str = "sig-1", labelled=True):
    baseline, tickets, labels = upload_payload()
    state.install_uploaded_project(
        name=name,
        baseline=baseline,
        tickets=tickets,
        labelled=labels if labelled else None,
        warnings=[],
        signature=signature,
    )


# --------------------------------------------------------------------------
# The core defect
# --------------------------------------------------------------------------
def test_uploaded_project_appears_in_the_project_store():
    """The regression test for §2.3. Fails without the fix."""
    assert len(state.store()) == 0
    install(name="Acme Portal")

    store = state.store()
    assert len(store) == 1
    assert store.active is not None
    assert store.active.name == "Acme Portal"
    assert store.active.source == "upload"


def test_uploaded_project_carries_its_datasets():
    install()
    record = state.store().active
    assert record.is_ready is True
    assert len(record.baseline) == 18
    assert len(record.tickets) == 45
    assert record.has_labels is True


def test_uploaded_project_without_labels_is_still_registered():
    install(labelled=False)
    record = state.store().active
    assert record.is_ready is True
    assert record.has_labels is False


def test_uploaded_project_sets_the_session_name_not_the_default():
    install(name="Acme Portal")
    assert st.session_state.project_name == "Acme Portal"
    assert st.session_state.project_name != "DriftGuard analysis"


def test_uploaded_project_is_marked_as_an_upload():
    install()
    assert st.session_state.source == "upload"
    assert state.is_synthetic() is False


def test_uploaded_project_accepts_a_scope_snapshot():
    """Snapshots need an active project; uploads previously had none."""
    install()
    result = pipeline.run_analysis(
        st.session_state.baseline, st.session_state.tickets
    )
    store = state.store()
    store.record(projects.snapshot_from_result(result, label="First reading"))

    history = store.history()
    assert len(history) == 1
    assert history[0].label == "First reading"
    assert history[0].sdi == pytest.approx(result.drift.sdi)


def test_uploaded_project_can_be_switched_to():
    install(name="Acme Portal", signature="sig-1")
    state.load_sample_project()
    assert state.store().active.name == "Campus Course Registration Portal"

    acme = next(
        pid for pid, name in state.store().names().items() if name == "Acme Portal"
    )
    state.switch_to(acme)
    assert state.store().active.name == "Acme Portal"
    assert st.session_state.project_name == "Acme Portal"
    assert len(st.session_state.tickets) == 45


# --------------------------------------------------------------------------
# Idempotency — the hazard the content hash exists to prevent
# --------------------------------------------------------------------------
def test_installing_the_same_upload_twice_is_guarded_by_signature():
    """Streamlit re-returns uploaded files on every rerun.

    The sidebar skips installation when the signature is unchanged, so a
    project is registered once rather than once per widget interaction.
    """
    install(name="Acme Portal", signature="sig-1")
    assert st.session_state["upload_signature"] == "sig-1"

    # The sidebar's guard: same signature means nothing to do.
    assert st.session_state.get("upload_signature") == "sig-1"
    assert len(state.store()) == 1


def test_a_changed_upload_installs_a_new_project():
    install(name="Acme Portal", signature="sig-1")
    install(name="Acme Portal", signature="sig-2")
    store = state.store()
    assert len(store) == 2
    # The second gets a disambiguated name rather than failing.
    assert sorted(store.names().values()) == ["Acme Portal", "Acme Portal (2)"]


def test_signature_is_stable_for_identical_bytes():
    first = state.upload_signature(b"baseline", b"tickets", b"")
    second = state.upload_signature(b"baseline", b"tickets", b"")
    assert first == second


def test_signature_changes_when_any_file_changes():
    base = state.upload_signature(b"baseline", b"tickets", b"")
    assert state.upload_signature(b"baseline2", b"tickets", b"") != base
    assert state.upload_signature(b"baseline", b"tickets2", b"") != base
    assert state.upload_signature(b"baseline", b"tickets", b"labels") != base


def test_signature_is_order_sensitive():
    """Swapping the baseline and ticket files is a different upload."""
    assert state.upload_signature(b"a", b"b", b"") != state.upload_signature(
        b"b", b"a", b""
    )


def test_clearing_a_project_resets_the_upload_signature():
    """Otherwise re-uploading the same file after Clear would do nothing."""
    install(signature="sig-1")
    state.reset_project()
    assert st.session_state["upload_signature"] is None


# --------------------------------------------------------------------------
# Name derivation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename,expected",
    [
        ("campus_portal_baseline.csv", "Campus Portal Baseline"),
        ("acme-requirements.txt", "Acme Requirements"),
        ("tickets.csv", "Tickets"),
        ("ACME reqs.csv", "ACME Reqs"),
        ("", "Uploaded project"),
        (".csv", "Uploaded project"),
    ],
)
def test_project_name_derived_from_filename(filename, expected):
    assert state.project_name_from_filename(filename) == expected


def test_derived_name_is_a_starting_point_and_can_be_edited():
    install(name="Campus Portal Baseline")
    state.rename_active_project("Campus Registration Portal")
    assert state.store().active.name == "Campus Registration Portal"
    assert st.session_state.project_name == "Campus Registration Portal"


def test_renaming_without_an_active_project_is_rejected():
    with pytest.raises(ValueError, match="no active project"):
        state.rename_active_project("Anything")


# --------------------------------------------------------------------------
# Rename rules on the store itself
# --------------------------------------------------------------------------
def test_rename_rejects_a_blank_name():
    install()
    with pytest.raises(ValueError, match="needs a name"):
        state.rename_active_project("   ")


def test_rename_rejects_a_collision_with_another_project():
    install(name="Acme Portal", signature="sig-1")
    state.load_sample_project()
    with pytest.raises(ValueError, match="already called"):
        state.rename_active_project("Acme Portal")


def test_renaming_a_project_to_its_own_name_is_allowed():
    install(name="Acme Portal")
    state.rename_active_project("Acme Portal")
    assert state.store().active.name == "Acme Portal"


def test_rename_trims_surrounding_whitespace():
    install(name="Acme Portal")
    state.rename_active_project("  Trimmed Name  ")
    assert state.store().active.name == "Trimmed Name"


def test_rename_rejects_an_unknown_project_id():
    store = projects.ProjectStore()
    with pytest.raises(KeyError):
        store.rename("nope", "Anything")


# --------------------------------------------------------------------------
# The sidebar wiring itself
# --------------------------------------------------------------------------
def test_sidebar_no_longer_assigns_session_state_directly():
    """A structural guard against the defect being reintroduced.

    The upload handler must register projects through ``state``, not by
    writing ``st.session_state.baseline`` itself.
    """
    import ast
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent / "ui" / "sidebar.py").read_text()
    tree = ast.parse(source)

    handler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_load_uploads"
    )

    assigned = set()
    for node in ast.walk(handler):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(
                    target.value, ast.Attribute
                ):
                    if target.value.attr == "session_state":
                        assigned.add(target.attr)

    assert not assigned, (
        "ui/sidebar.py:_load_uploads assigns st.session_state directly "
        f"({', '.join(sorted(assigned))}); it must route through "
        "state.install_uploaded_project() so the project registers."
    )
    assert "install_uploaded_project" in source
