"""Multi-project organisation and in-session scope history.

Scope and persistence
---------------------
This store lives in memory for the duration of a session. It deliberately does
**not** introduce a database.

That is a real limitation, stated here rather than discovered later: close the
browser tab and the projects and their history are gone. Durable history needs
a datastore, and adding one would change DriftGuard's offline-first,
zero-infrastructure deployment story for a feature that is useful but not
central. The trade-off is recorded in the README as a future improvement.

What this *does* give you is a worked comparison within a working session:
load a baseline, analyse it, record a snapshot, change classifications or
assumptions, record another, and see how the Scope Drift Index moved and which
tickets moved it.

The module is pure data: no Streamlit import, so it is testable headlessly.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class ProjectRecord:
    """One software project and its datasets."""

    name: str
    description: str = ""
    baseline: pd.DataFrame | None = None
    tickets: pd.DataFrame | None = None
    labelled: pd.DataFrame | None = None
    project_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: datetime = field(default_factory=datetime.now)
    source: str = "upload"

    @property
    def is_ready(self) -> bool:
        return self.baseline is not None and self.tickets is not None

    @property
    def has_labels(self) -> bool:
        return (
            self.labelled is not None
            and isinstance(self.labelled, pd.DataFrame)
            and "true_class" in self.labelled.columns
        )

    def summary(self) -> dict[str, object]:
        return {
            "Project": self.name,
            "Requirements": len(self.baseline) if self.baseline is not None else 0,
            "Tickets": len(self.tickets) if self.tickets is not None else 0,
            "Labelled": self.has_labels,
            "Source": self.source,
            "Created": self.created_at.strftime("%d %b %Y %H:%M"),
        }


@dataclass(frozen=True)
class ScopeSnapshot:
    """A point-in-time record of one analysis.

    Stores derived figures rather than the whole result object, so a long
    session does not accumulate frames. ``contributors`` keeps the handful of
    tickets that drove the drift, which is what a reviewer wants when asking
    "what changed between these two readings?".
    """

    label: str
    recorded_at: datetime
    sdi: float
    baseline_size: float
    revised_size: float
    size_added: float
    size_removed: float
    person_months: float
    duration_months: float
    class_counts: dict[str, int]
    reviewed: int
    overturned: int
    refinement_weight: float
    sloc_per_point: float
    revl_percent: float
    contributors: tuple[tuple[str, float], ...] = ()
    note: str = ""

    def as_row(self) -> dict[str, object]:
        return {
            "Snapshot": self.label,
            "Recorded": self.recorded_at.strftime("%d %b %Y %H:%M"),
            "SDI": self.sdi,
            "Baseline size": self.baseline_size,
            "Revised size": self.revised_size,
            "Effort (PM)": self.person_months,
            "Duration (months)": self.duration_months,
            "Reviewed": self.reviewed,
            "Overturned": self.overturned,
            "Refinement weight": self.refinement_weight,
            "SLOC/point": self.sloc_per_point,
            "REVL %": self.revl_percent,
            "Note": self.note,
        }


def snapshot_from_result(result, *, label: str, note: str = "") -> ScopeSnapshot:
    """Capture the figures from a completed analysis."""
    drift = result.drift
    estimates = result.estimates
    settings = result.settings

    counts = (
        drift.per_class.set_index("classification")["tickets"].astype(int).to_dict()
        if not drift.per_class.empty
        else {}
    )
    reviewed = (
        int(result.classified["reviewed"].sum())
        if "reviewed" in result.classified.columns
        else 0
    )
    overturned = (
        int(result.classified["class_changed_by_reviewer"].sum())
        if "class_changed_by_reviewer" in result.classified.columns
        else 0
    )
    revl = (
        (estimates.revised.effective_ksloc / estimates.revised.size_ksloc - 1) * 100
        if estimates.revised.size_ksloc
        else 0.0
    )

    top = drift.top_contributors(5)
    contributors = tuple(
        (str(row.get("ticket_id", "?")), float(row["net_contribution"]))
        for _, row in top.iterrows()
    )

    return ScopeSnapshot(
        label=label,
        recorded_at=datetime.now(),
        sdi=drift.sdi,
        baseline_size=drift.baseline_size,
        revised_size=drift.revised_size,
        size_added=drift.size_added,
        size_removed=drift.size_removed,
        person_months=estimates.revised.person_months,
        duration_months=estimates.revised.duration_months,
        class_counts=counts,
        reviewed=reviewed,
        overturned=overturned,
        refinement_weight=settings.refinement_weight,
        sloc_per_point=settings.conversion.sloc_per_story_point,
        revl_percent=revl,
        contributors=contributors,
        note=note,
    )


class ProjectStore:
    """In-memory collection of projects and their analysis history."""

    def __init__(self) -> None:
        self._projects: dict[str, ProjectRecord] = {}
        self._history: dict[str, list[ScopeSnapshot]] = {}
        self._active: str | None = None

    # -- projects --------------------------------------------------------
    def add(self, project: ProjectRecord, *, activate: bool = True) -> str:
        if not project.name.strip():
            raise ValueError("A project needs a name.")
        if any(
            existing.name.strip().lower() == project.name.strip().lower()
            for existing in self._projects.values()
        ):
            raise ValueError(
                f"A project called {project.name!r} already exists in this "
                "session. Use a different name, or switch to it."
            )
        self._projects[project.project_id] = project
        self._history.setdefault(project.project_id, [])
        if activate or self._active is None:
            self._active = project.project_id
        return project.project_id

    def remove(self, project_id: str) -> None:
        self._projects.pop(project_id, None)
        self._history.pop(project_id, None)
        if self._active == project_id:
            self._active = next(iter(self._projects), None)

    def get(self, project_id: str) -> ProjectRecord:
        if project_id not in self._projects:
            raise KeyError(f"No project with id {project_id!r}.")
        return self._projects[project_id]

    def activate(self, project_id: str) -> None:
        if project_id not in self._projects:
            raise KeyError(f"No project with id {project_id!r}.")
        self._active = project_id

    @property
    def active(self) -> ProjectRecord | None:
        if self._active is None:
            return None
        return self._projects.get(self._active)

    @property
    def active_id(self) -> str | None:
        return self._active

    def __len__(self) -> int:
        return len(self._projects)

    def __contains__(self, project_id: object) -> bool:
        return project_id in self._projects

    def names(self) -> dict[str, str]:
        """Mapping of project id to display name, creation order preserved."""
        return {pid: record.name for pid, record in self._projects.items()}

    def has_name(self, name: str) -> bool:
        """Whether a project with this name already exists, ignoring case."""
        target = name.strip().lower()
        return any(
            record.name.strip().lower() == target for record in self._projects.values()
        )

    def unique_name(self, name: str) -> str:
        """``name``, suffixed if necessary so it does not collide."""
        if not self.has_name(name):
            return name
        suffix = 2
        while self.has_name(f"{name} ({suffix})"):
            suffix += 1
        return f"{name} ({suffix})"

    def catalogue(self) -> pd.DataFrame:
        """One row per project, for display."""
        if not self._projects:
            return pd.DataFrame(
                columns=["Project", "Requirements", "Tickets", "Labelled", "Source", "Created", "Snapshots"]
            )
        rows = []
        for pid, record in self._projects.items():
            row = record.summary()
            row["Snapshots"] = len(self._history.get(pid, []))
            row["Active"] = pid == self._active
            rows.append(row)
        return pd.DataFrame(rows)

    # -- history ---------------------------------------------------------
    def record(self, snapshot: ScopeSnapshot, *, project_id: str | None = None) -> None:
        target = project_id or self._active
        if target is None:
            raise ValueError("No active project to record a snapshot against.")
        if target not in self._projects:
            raise KeyError(f"No project with id {target!r}.")
        self._history[target].append(snapshot)

    def history(self, project_id: str | None = None) -> list[ScopeSnapshot]:
        target = project_id or self._active
        if target is None:
            return []
        return list(self._history.get(target, []))

    def clear_history(self, project_id: str | None = None) -> None:
        target = project_id or self._active
        if target is not None:
            self._history[target] = []

    def history_frame(self, project_id: str | None = None) -> pd.DataFrame:
        """Snapshot history as a table, oldest first."""
        snapshots = self.history(project_id)
        if not snapshots:
            return pd.DataFrame(
                columns=[
                    "Snapshot", "Recorded", "SDI", "Baseline size", "Revised size",
                    "Effort (PM)", "Duration (months)", "Reviewed", "Overturned",
                    "Refinement weight", "SLOC/point", "REVL %", "Note",
                ]
            )
        return pd.DataFrame([snapshot.as_row() for snapshot in snapshots])


# --------------------------------------------------------------------------
# Comparison between snapshots
# --------------------------------------------------------------------------
def compare_snapshots(first: ScopeSnapshot, second: ScopeSnapshot) -> pd.DataFrame:
    """Side-by-side difference between two readings of the same project.

    Includes the assumptions as well as the outcomes, because a change in SDI
    between two snapshots may be the result of reclassification *or* of someone
    moving the refinement weight -- and those are very different stories.
    """
    rows = [
        ("Scope Drift Index", first.sdi, second.sdi, "ratio"),
        ("Baseline size", first.baseline_size, second.baseline_size, "size"),
        ("Revised size", first.revised_size, second.revised_size, "size"),
        ("Size added", first.size_added, second.size_added, "size"),
        ("Size removed", first.size_removed, second.size_removed, "size"),
        ("Effort (person-months)", first.person_months, second.person_months, "number"),
        ("Duration (months)", first.duration_months, second.duration_months, "number"),
        ("Tickets reviewed", first.reviewed, second.reviewed, "count"),
        ("Classifications overturned", first.overturned, second.overturned, "count"),
        ("Refinement weight", first.refinement_weight, second.refinement_weight, "assumption"),
        ("SLOC per story point", first.sloc_per_point, second.sloc_per_point, "assumption"),
        ("REVL applied (%)", first.revl_percent, second.revl_percent, "assumption"),
    ]
    frame = pd.DataFrame(
        [
            {
                "Measure": name,
                "Kind": kind,
                first.label: a,
                second.label: b,
                "Change": b - a,
            }
            for name, a, b, kind in rows
        ]
    )
    return frame


def assumptions_changed(first: ScopeSnapshot, second: ScopeSnapshot) -> list[str]:
    """Which assumptions differ between two snapshots.

    Used to warn that a movement in the index may not be a movement in the
    project.
    """
    changes = []
    if abs(first.refinement_weight - second.refinement_weight) > 1e-9:
        changes.append(
            f"refinement weight {first.refinement_weight:.2f} → "
            f"{second.refinement_weight:.2f}"
        )
    if abs(first.sloc_per_point - second.sloc_per_point) > 1e-9:
        changes.append(
            f"SLOC per story point {first.sloc_per_point:g} → {second.sloc_per_point:g}"
        )
    if abs(first.revl_percent - second.revl_percent) > 1e-6:
        changes.append(
            f"REVL {first.revl_percent:.1f}% → {second.revl_percent:.1f}%"
        )
    return changes
