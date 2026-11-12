"""Synthetic project generator for testing and demonstration.

Everything this module produces is **fabricated**. It exists so that
DriftGuard's behaviour can be exercised under controlled conditions -- heavy
new scope, heavy refinement, mass removals, dirty data -- without waiting for a
real project to misbehave.

Two safeguards against synthetic output being mistaken for evidence:

* every generated frame carries ``synthetic=True`` as a column, and
  ``frame.attrs["synthetic"]`` as metadata;
* every requirement and ticket id is prefixed (``SYN-R``/``SYN-T``), and the
  description of each record names the scenario that produced it.

Nothing here is drawn from, or claims to represent, any real project,
organisation or industry benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CLASSES, IN_SCOPE, NEW_SCOPE, REFINEMENT, REMOVAL

SYNTHETIC_NOTICE = (
    "SYNTHETIC DATA — generated for testing. Not real project data and not "
    "representative of any organisation or industry benchmark."
)

#: Named scenarios the generator supports.
SCENARIOS: dict[str, str] = {
    "normal": "Normal project activity — mostly in-scope delivery work.",
    "new_scope_surge": "Increased new-scope requests arriving mid-project.",
    "refinement_heavy": "Frequent changes to requirements already agreed.",
    "removals": "Significant de-scoping of agreed requirements.",
    "dirty": "Missing sizes, blank descriptions, duplicate and undated rows.",
}

#: Vocabulary pools. Deliberately generic so the generated text does not
#: accidentally resemble the bundled campus-portal sample.
_DOMAINS = [
    ("inventory", ["stock level", "warehouse", "reorder point", "supplier", "shipment"]),
    ("billing", ["invoice", "tax rate", "payment terms", "credit note", "ledger"]),
    ("scheduling", ["shift", "roster", "availability", "booking", "calendar slot"]),
    ("reporting", ["dashboard", "export", "summary table", "audit trail", "metric"]),
    ("access", ["role", "permission", "session", "credential", "sign-in"]),
]

_REQUIREMENT_TEMPLATES = [
    "The system records {noun} details and makes them available to authorised users.",
    "Users can search and filter {noun} records for a selected period.",
    "The system validates {noun} entries against the configured business rules.",
    "Authorised staff can create, amend and withdraw a {noun} record.",
    "The system produces a {noun} summary for the operations team.",
]

_IN_SCOPE_TEMPLATES = [
    "Implement the {noun} screen as specified in the baseline.",
    "Build the {noun} service and wire it to the existing data layer.",
    "Add validation for {noun} entries per the agreed business rules.",
    "Create the {noun} listing with the agreed filters.",
]

_REFINEMENT_TEMPLATES = [
    "Revise the {noun} rules so the threshold is configurable rather than fixed.",
    "Update the {noun} screen to also display the owning team.",
    "Amend {noun} validation to accept the alternate format as well.",
    "Extend the {noun} export to include the agreed columns plus the reference code.",
]

_NEW_SCOPE_TEMPLATES = [
    "Introduce a {noun} mobile view. This was not part of the agreed scope.",
    "Add support for bulk {noun} import, newly requested by the operations team.",
    "Net new integration pushing {noun} data to the external partner system.",
    "Stakeholder request to introduce {noun} forecasting. Was never specified.",
]

_REMOVAL_TEMPLATES = [
    "The {noun} summary is de-scoped from this project; the central team will provide it.",
    "Remove the {noun} export from scope; it is no longer required.",
    "The {noun} screen is withdrawn from the project scope.",
    "Cancel the {noun} requirement; it will not be implemented in this release.",
]

#: Class mix per scenario, as weights over (In-scope, Refinement, New scope, Removal).
_MIXES: dict[str, tuple[float, float, float, float]] = {
    "normal": (0.70, 0.15, 0.12, 0.03),
    "new_scope_surge": (0.35, 0.12, 0.48, 0.05),
    "refinement_heavy": (0.33, 0.50, 0.12, 0.05),
    "removals": (0.40, 0.12, 0.13, 0.35),
    "dirty": (0.55, 0.18, 0.20, 0.07),
}

_TEMPLATES_BY_CLASS = {
    IN_SCOPE: _IN_SCOPE_TEMPLATES,
    REFINEMENT: _REFINEMENT_TEMPLATES,
    NEW_SCOPE: _NEW_SCOPE_TEMPLATES,
    REMOVAL: _REMOVAL_TEMPLATES,
}


@dataclass(frozen=True)
class SyntheticProject:
    """A generated baseline, ticket set and labelled copy of the tickets."""

    baseline: pd.DataFrame
    tickets: pd.DataFrame
    labelled: pd.DataFrame
    scenario: str
    seed: int

    @property
    def notice(self) -> str:
        return SYNTHETIC_NOTICE

    def summary(self) -> pd.DataFrame:
        counts = self.labelled["true_class"].value_counts().reindex(CLASSES).fillna(0)
        return pd.DataFrame(
            {
                "Class": list(CLASSES),
                "Tickets": counts.astype(int).to_list(),
            }
        )


def _mark(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["synthetic"] = True
    frame.attrs["synthetic"] = True
    frame.attrs["notice"] = SYNTHETIC_NOTICE
    return frame


def generate_project(
    scenario: str = "normal",
    *,
    requirements: int = 15,
    tickets: int = 40,
    seed: int = 7,
    start_date: str = "2026-01-05",
) -> SyntheticProject:
    """Generate a controlled synthetic project.

    Parameters
    ----------
    scenario:
        One of :data:`SCENARIOS`.
    requirements, tickets:
        Sizes of the generated baseline and ticket set.
    seed:
        Makes generation reproducible, so a defect found on generated data can
        be reproduced exactly.

    Returns
    -------
    SyntheticProject with ``baseline``, ``tickets`` (no labels) and
    ``labelled`` (the same tickets plus a ``true_class`` column).
    """
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario!r}. Available: {', '.join(SCENARIOS)}."
        )
    if requirements < 3:
        raise ValueError("Generate at least 3 requirements.")
    if tickets < 4:
        raise ValueError("Generate at least 4 tickets.")

    rng = np.random.default_rng(seed)
    story_points = (1, 2, 3, 5, 8, 13, 21)

    # ---- Baseline -------------------------------------------------------
    rows = []
    for index in range(requirements):
        domain, nouns = _DOMAINS[index % len(_DOMAINS)]
        noun = nouns[int(rng.integers(len(nouns)))]
        template = _REQUIREMENT_TEMPLATES[index % len(_REQUIREMENT_TEMPLATES)]
        rows.append(
            {
                "requirement_id": f"SYN-R{index + 1:02d}",
                "title": f"{domain.capitalize()} — {noun}",
                "description": template.format(noun=noun),
                "size": int(rng.choice(story_points[2:6])),
                "category": domain.capitalize(),
            }
        )
    baseline = pd.DataFrame(rows)

    # ---- Tickets --------------------------------------------------------
    weights = np.array(_MIXES[scenario], dtype=float)
    weights = weights / weights.sum()
    classes = rng.choice(list(CLASSES), size=tickets, p=weights)

    dates = pd.date_range(start=start_date, periods=tickets, freq="3D")
    ticket_rows = []
    for index, (label, date) in enumerate(zip(classes, dates)):
        requirement = baseline.iloc[index % len(baseline)]
        noun = str(requirement["title"]).split("—")[-1].strip()
        template = _TEMPLATES_BY_CLASS[label][
            int(rng.integers(len(_TEMPLATES_BY_CLASS[label])))
        ]
        # New scope deliberately uses vocabulary unrelated to the baseline, so
        # the matcher has a genuine chance of scoring it low.
        if label == NEW_SCOPE:
            other_domain, other_nouns = _DOMAINS[int(rng.integers(len(_DOMAINS)))]
            noun = other_nouns[int(rng.integers(len(other_nouns)))]

        ticket_rows.append(
            {
                "ticket_id": f"SYN-T{index + 1:03d}",
                "title": template.format(noun=noun).rstrip(".")[:70],
                "description": template.format(noun=noun),
                "size": int(rng.choice(story_points[:5])),
                "created_date": date.strftime("%Y-%m-%d"),
                "status": str(rng.choice(["To do", "In progress", "Done"])),
                "reporter": "synthetic.generator",
                "true_class": label,
            }
        )
    frame = pd.DataFrame(ticket_rows)

    if scenario == "dirty":
        frame = _dirty_up(frame, rng)

    labelled = _mark(frame)
    unlabelled = _mark(frame.drop(columns=["true_class"]))

    return SyntheticProject(
        baseline=_mark(baseline),
        tickets=unlabelled,
        labelled=labelled,
        scenario=scenario,
        seed=seed,
    )


def _dirty_up(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Introduce the data-quality problems the loaders are meant to catch.

    Deliberately includes every failure mode ``driftguard.data`` warns about,
    so the validation path can be demonstrated rather than described.
    """
    frame = frame.copy()
    count = len(frame)

    # Missing sizes
    for index in rng.choice(count, size=max(1, count // 10), replace=False):
        frame.loc[index, "size"] = np.nan

    # Blank descriptions
    for index in rng.choice(count, size=max(1, count // 12), replace=False):
        frame.loc[index, "description"] = "   "

    # Unparseable dates
    for index in rng.choice(count, size=max(1, count // 12), replace=False):
        frame.loc[index, "created_date"] = "not a date"

    # A duplicate id, appended so the first occurrence is kept
    duplicate = frame.iloc[0].copy()
    duplicate["title"] = duplicate["title"] + " (duplicate row)"
    frame = pd.concat([frame, duplicate.to_frame().T], ignore_index=True)

    return frame


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """CSV export that still announces itself as synthetic after a round trip.

    The marker is the ``synthetic`` column, not a leading ``#`` comment line.
    A comment line was the first implementation and was wrong: ``read_csv``
    treats it as the header row, so a generated file could not be re-uploaded.
    A real column survives the round trip, and ``driftguard.data`` warns when
    it sees one.
    """
    frame = frame.copy()
    if "synthetic" not in frame.columns:
        frame["synthetic"] = True
    return frame.to_csv(index=False).encode("utf-8")


def describe_scenarios() -> pd.DataFrame:
    """Scenario catalogue with the class mix each one targets."""
    rows = []
    for name, description in SCENARIOS.items():
        mix = _MIXES[name]
        rows.append(
            {
                "Scenario": name,
                "What it exercises": description,
                "In-scope": f"{mix[0]:.0%}",
                "Refinement": f"{mix[1]:.0%}",
                "New scope": f"{mix[2]:.0%}",
                "Removal": f"{mix[3]:.0%}",
            }
        )
    return pd.DataFrame(rows)
