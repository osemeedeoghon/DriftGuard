"""Session state, project switching and data loading for the dashboard."""

from __future__ import annotations

import hashlib
import pathlib
import re

import pandas as pd
import streamlit as st

from driftguard import data
from driftguard.projects import ProjectRecord, ProjectStore

DEFAULTS: dict[str, object] = {
    "baseline": None,
    "tickets": None,
    "labelled": None,
    "load_warnings": [],
    "overrides": {},
    "size_overrides": {},
    "source": "none",
    "project_name": "DriftGuard analysis",
    "selected_ticket": None,
    "upload_signature": None,
}


def init() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    if "projects" not in st.session_state:
        st.session_state["projects"] = ProjectStore()


def store() -> ProjectStore:
    """The session's project store."""
    init()
    return st.session_state["projects"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _install(
    *,
    name: str,
    baseline: pd.DataFrame,
    tickets: pd.DataFrame,
    labelled: pd.DataFrame | None,
    source: str,
    warnings: list[str] | None = None,
    description: str = "",
) -> None:
    """Make a dataset the active project and reset per-project review state."""
    st.session_state.baseline = baseline
    st.session_state.tickets = tickets
    st.session_state.labelled = labelled
    st.session_state.overrides = {}
    st.session_state.size_overrides = {}
    st.session_state.load_warnings = warnings or []
    st.session_state.source = source
    st.session_state.project_name = name
    st.session_state.selected_ticket = None

    project_store = store()
    unique = project_store.unique_name(name)
    project_store.add(
        ProjectRecord(
            name=unique,
            description=description,
            baseline=baseline,
            tickets=tickets,
            labelled=labelled,
            source=source,
        )
    )
    st.session_state.project_name = unique


def project_name_from_filename(filename: str) -> str:
    """Derive a readable project name from an uploaded file name.

    ``campus_portal_baseline.csv`` becomes ``Campus Portal Baseline``. The
    result is a starting point the reviewer can edit, not a decision — a file
    name is weak evidence of what a project is called.
    """
    raw = (filename or "").strip()
    # A dotfile such as ".csv" has no real name: Path(".csv").stem is ".csv",
    # which would yield the project name "Csv".
    if raw.startswith("."):
        return "Uploaded project"
    stem = pathlib.Path(raw).stem.strip()
    if not stem:
        return "Uploaded project"
    words = [w for w in re.split(r"[_\-.\s]+", stem) if w]
    if not words:
        return "Uploaded project"
    return " ".join(word if word.isupper() else word.capitalize() for word in words)


def upload_signature(*payloads: bytes) -> str:
    """Content hash of an upload set.

    Streamlit hands back the same ``UploadedFile`` objects on every rerun, so
    installing on their mere presence would register a fresh project on each
    interaction. Keying installation on content means an upload is processed
    once, and re-processed only when the bytes actually change.
    """
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def install_uploaded_project(
    *,
    name: str,
    baseline: pd.DataFrame,
    tickets: pd.DataFrame,
    labelled: pd.DataFrame | None,
    warnings: list[str],
    signature: str,
) -> None:
    """Register an uploaded project exactly as the sample path does.

    Before this existed, the upload branches assigned ``st.session_state``
    directly and never reached :func:`_install`, so uploaded projects were
    absent from the project store: no switching, no scope history, and a
    default name on every exported report (audit finding §2.3).
    """
    _install(
        name=name,
        baseline=baseline,
        tickets=tickets,
        labelled=labelled,
        source="upload",
        warnings=warnings,
        description=f"Uploaded {len(baseline)} requirements and {len(tickets)} tickets.",
    )
    st.session_state["upload_signature"] = signature


def rename_active_project(new_name: str) -> None:
    """Rename the active project and keep the session label in step."""
    project_store = store()
    active_id = project_store.active_id
    if active_id is None:
        raise ValueError("There is no active project to rename.")
    project_store.rename(active_id, new_name)
    st.session_state.project_name = project_store.get(active_id).name


def load_sample_project() -> None:
    """Load the bundled worked example as a new project."""
    _install(
        name="Campus Course Registration Portal",
        baseline=data.sample_baseline(),
        tickets=data.sample_tickets(),
        labelled=data.sample_labelled_tickets(),
        source="sample",
        description=(
            "Bundled worked example: 18 agreed requirements, 45 tickets, "
            "manually labelled for evaluation."
        ),
    )


def load_synthetic_project(scenario: str, *, seed: int, requirements: int, tickets: int) -> None:
    """Generate and activate a synthetic project."""
    import io

    from driftguard import synthetic

    generated = synthetic.generate_project(
        scenario, requirements=requirements, tickets=tickets, seed=seed
    )
    warnings: list[str] = []
    baseline, baseline_report = data.load_baseline_csv(
        io.BytesIO(synthetic.to_csv_bytes(generated.baseline))
    )
    loaded_tickets, ticket_report = data.load_tickets(
        io.BytesIO(synthetic.to_csv_bytes(generated.tickets))
    )
    labelled, labelled_report = data.load_tickets(
        io.BytesIO(synthetic.to_csv_bytes(generated.labelled))
    )
    warnings += baseline_report.warnings + ticket_report.warnings

    _install(
        name=f"Synthetic · {scenario} (seed {seed})",
        baseline=baseline,
        tickets=loaded_tickets,
        labelled=labelled,
        source="synthetic",
        warnings=warnings,
        description=(
            f"Generated by DriftGuard's test-data generator, scenario "
            f"'{scenario}', seed {seed}. Not real project data."
        ),
    )


def switch_to(project_id: str) -> None:
    """Make a stored project active and restore its datasets."""
    project_store = store()
    project_store.activate(project_id)
    record = project_store.get(project_id)
    st.session_state.baseline = record.baseline
    st.session_state.tickets = record.tickets
    st.session_state.labelled = record.labelled
    st.session_state.overrides = {}
    st.session_state.size_overrides = {}
    st.session_state.source = record.source
    st.session_state.project_name = record.name
    st.session_state.selected_ticket = None
    st.session_state.load_warnings = []


def reset_project() -> None:
    """Clear the active dataset without discarding the project catalogue."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def has_project() -> bool:
    return (
        st.session_state.get("baseline") is not None
        and st.session_state.get("tickets") is not None
    )


def has_labels() -> bool:
    labelled = st.session_state.get("labelled")
    return (
        labelled is not None
        and isinstance(labelled, pd.DataFrame)
        and "true_class" in labelled.columns
    )


def is_synthetic() -> bool:
    return st.session_state.get("source") == "synthetic"


def review_count() -> int:
    return len(
        set(st.session_state.get("overrides", {}))
        | set(st.session_state.get("size_overrides", {}))
    )
