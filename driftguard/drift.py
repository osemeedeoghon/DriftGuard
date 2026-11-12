"""Scope Drift Index (SDI) calculation and drift attribution.

Definition
----------
    SDI = (cumulative size added - cumulative size removed) / baseline size

Interpretation: SDI = 0.20 means the agreed baseline has grown by a net 20%.
A negative SDI means net de-scoping. SDI is dimensionless, so sizes may be
expressed in story points, function points or SLOC provided the baseline and
the tickets use the *same* unit.

How each classification contributes
-----------------------------------
    In-scope    added += 0                  (already inside the baseline)
    Refinement  added += w * size           (w = refinement weight, an assumption)
    New scope   added += size
    Removal     removed += size

The revised project size follows directly:

    revised_size = baseline_size + added - removed
                 = baseline_size * (1 + SDI)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import (
    CLASSES,
    DEFAULT_REFINEMENT_WEIGHT,
    IN_SCOPE,
    NEW_SCOPE,
    REFINEMENT,
    REMOVAL,
)


@dataclass(frozen=True)
class DriftResult:
    """Outcome of a scope drift calculation."""

    baseline_size: float
    size_added: float
    size_removed: float
    sdi: float
    revised_size: float
    per_ticket: pd.DataFrame      # one row per ticket, with its contribution
    per_class: pd.DataFrame       # totals by classification
    classification_column: str = "final_class"

    @property
    def net_change(self) -> float:
        return self.size_added - self.size_removed

    @property
    def drift_percent(self) -> float:
        return self.sdi * 100.0

    def top_contributors(self, n: int = 10) -> pd.DataFrame:
        """Tickets responsible for the most drift, largest absolute net first."""
        if self.per_ticket.empty:
            return self.per_ticket
        ordered = self.per_ticket.assign(
            abs_net=self.per_ticket["net_contribution"].abs()
        ).sort_values("abs_net", ascending=False)
        return ordered.drop(columns="abs_net").head(n)


def contribution(classification: str, size: float, refinement_weight: float) -> tuple[float, float]:
    """Return (added, removed) for one ticket.

    Raises on an unrecognised classification rather than silently scoring zero,
    because a typo in a reviewer override must not quietly suppress drift.
    """
    if size < 0:
        raise ValueError(
            "Ticket size cannot be negative. Record withdrawn work as a "
            f"'{REMOVAL}' ticket with a positive size."
        )
    if classification == IN_SCOPE:
        return 0.0, 0.0
    if classification == REFINEMENT:
        return refinement_weight * size, 0.0
    if classification == NEW_SCOPE:
        return size, 0.0
    if classification == REMOVAL:
        return 0.0, size
    raise ValueError(
        f"Unknown classification {classification!r}. Expected one of: "
        + ", ".join(CLASSES)
    )


def calculate_drift(
    tickets: pd.DataFrame,
    baseline_size: float,
    *,
    classification_column: str = "final_class",
    size_column: str = "size",
    refinement_weight: float = DEFAULT_REFINEMENT_WEIGHT,
) -> DriftResult:
    """Compute the Scope Drift Index over a set of classified tickets.

    Parameters
    ----------
    tickets:
        One row per ticket. Must contain the classification and size columns.
    baseline_size:
        Total agreed size of the baseline, in the same unit as ticket sizes.
        Must be strictly positive -- SDI is undefined against a zero baseline.
    refinement_weight:
        Fraction of a Refinement ticket's size counted as added scope.
    """
    if baseline_size <= 0:
        raise ValueError(
            "Baseline size must be greater than zero; the Scope Drift Index "
            "is undefined when there is no agreed baseline to drift from."
        )
    if not 0.0 <= refinement_weight <= 1.0:
        raise ValueError("Refinement weight must lie between 0 and 1 inclusive.")

    for column in (classification_column, size_column):
        if column not in tickets.columns:
            raise ValueError(
                f"Tickets are missing the required column {column!r}. "
                f"Columns present: {', '.join(map(str, tickets.columns))}."
            )

    work = tickets.copy()
    if work.empty:
        empty_per_class = pd.DataFrame(
            {
                "classification": list(CLASSES),
                "tickets": 0,
                "size": 0.0,
                "added": 0.0,
                "removed": 0.0,
            }
        )
        return DriftResult(
            baseline_size=baseline_size,
            size_added=0.0,
            size_removed=0.0,
            sdi=0.0,
            revised_size=baseline_size,
            per_ticket=work.assign(added=[], removed=[], net_contribution=[]),
            per_class=empty_per_class,
            classification_column=classification_column,
        )

    sizes = pd.to_numeric(work[size_column], errors="coerce")
    if sizes.isna().any():
        bad = work.loc[sizes.isna()]
        label = bad.iloc[0].get("ticket_id", "<unknown>")
        raise ValueError(
            f"Ticket {label!r} has a non-numeric size "
            f"({bad.iloc[0][size_column]!r}). Sizes must be numeric."
        )
    work[size_column] = sizes

    pairs = [
        contribution(str(cls), float(size), refinement_weight)
        for cls, size in zip(work[classification_column], work[size_column])
    ]
    work["added"] = [p[0] for p in pairs]
    work["removed"] = [p[1] for p in pairs]
    work["net_contribution"] = work["added"] - work["removed"]

    size_added = float(work["added"].sum())
    size_removed = float(work["removed"].sum())
    sdi = (size_added - size_removed) / baseline_size
    revised_size = baseline_size + size_added - size_removed

    per_class = (
        work.groupby(classification_column)
        .agg(
            tickets=(classification_column, "size"),
            size=(size_column, "sum"),
            added=("added", "sum"),
            removed=("removed", "sum"),
        )
        .reindex(list(CLASSES))
        .fillna(0.0)
        .reset_index()
        .rename(columns={classification_column: "classification"})
    )
    per_class["tickets"] = per_class["tickets"].astype(int)

    return DriftResult(
        baseline_size=baseline_size,
        size_added=size_added,
        size_removed=size_removed,
        sdi=sdi,
        revised_size=revised_size,
        per_ticket=work,
        per_class=per_class,
        classification_column=classification_column,
    )


def drift_over_time(
    tickets: pd.DataFrame,
    baseline_size: float,
    *,
    date_column: str = "created_date",
    classification_column: str = "final_class",
    size_column: str = "size",
    refinement_weight: float = DEFAULT_REFINEMENT_WEIGHT,
) -> pd.DataFrame:
    """Cumulative SDI as tickets arrive, ordered by date.

    Returns a frame with one row per ticket date carrying the running totals.
    Tickets with an unparseable or missing date are excluded from the timeline
    (they still count in the overall SDI); the caller is told how many via the
    ``excluded_tickets`` attribute on the returned frame's ``attrs``.
    """
    result = calculate_drift(
        tickets,
        baseline_size,
        classification_column=classification_column,
        size_column=size_column,
        refinement_weight=refinement_weight,
    )
    work = result.per_ticket

    if date_column not in work.columns:
        raise ValueError(
            f"Cannot build a drift timeline: tickets have no {date_column!r} "
            "column."
        )

    work = work.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    excluded = int(work[date_column].isna().sum())
    work = work.dropna(subset=[date_column]).sort_values(date_column)

    if work.empty:
        timeline = pd.DataFrame(
            columns=[
                date_column,
                "cumulative_added",
                "cumulative_removed",
                "cumulative_sdi",
                "projected_size",
            ]
        )
        timeline.attrs["excluded_tickets"] = excluded
        return timeline

    daily = (
        work.groupby(work[date_column].dt.normalize())[["added", "removed"]]
        .sum()
        .sort_index()
    )
    daily["cumulative_added"] = daily["added"].cumsum()
    daily["cumulative_removed"] = daily["removed"].cumsum()
    daily["cumulative_sdi"] = (
        daily["cumulative_added"] - daily["cumulative_removed"]
    ) / baseline_size
    daily["projected_size"] = baseline_size * (1.0 + daily["cumulative_sdi"])

    timeline = daily.reset_index().rename(columns={date_column: date_column})
    timeline.attrs["excluded_tickets"] = excluded
    return timeline


def revl_from_drift(result: DriftResult) -> float:
    """Translate measured *churn* into a COCOMO II REVL percentage.

    Careful here, because the obvious implementation is wrong. REVL inflates
    size inside the effort equation, so feeding it the whole Scope Drift Index
    would double-count: the revised size *already* contains the newly added
    scope, and multiplying it up again by the same percentage charges for that
    work twice.

    What REVL actually models is rework -- effort spent on requirements that
    were specified, possibly started, and then changed or discarded. The
    honest proxy for that in DriftGuard's taxonomy is:

        REVL% = 100 * (refinement-driven additions + removals) / baseline size

    * **Refinements** are requirements that moved after being agreed, which is
      the textbook definition of volatility.
    * **Removals** are work that was planned, and may have been partly built,
      before being dropped. The size leaves the revised total but the effort
      does not come back.
    * **New scope** is deliberately excluded: it is already carried by the
      larger revised size, so counting it here too would be the double-count.

    Returns a percentage, never negative.
    """
    if result.per_ticket.empty:
        refinement_added = 0.0
    else:
        is_refinement = result.per_ticket[result.classification_column] == REFINEMENT
        refinement_added = float(result.per_ticket.loc[is_refinement, "added"].sum())

    churn = refinement_added + result.size_removed
    if result.baseline_size <= 0:
        return 0.0
    return max(0.0, churn / result.baseline_size * 100.0)


def revl_from_sdi(sdi: float) -> float:
    """Naive mapping of net drift straight onto REVL.

    Kept because it is the obvious reading of the COCOMO II definition, and
    because the comparison is instructive -- but it double-counts newly added
    scope whenever it is applied to an already-revised size. Prefer
    :func:`revl_from_drift`.
    """
    return max(0.0, sdi * 100.0)
