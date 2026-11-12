"""Sensitivity analysis and scenario planning.

Every figure here comes from the existing :mod:`driftguard.cocomo`
implementation. No alternative formula is introduced, no confidence interval is
invented, and no precision is implied that the inputs do not support.

Two distinct things live in this module:

**Sensitivity analysis** sweeps one documented assumption across a range and
reports how the estimate responds. It answers "how much does this assumption
matter?" — which, for an unvalidated placeholder like SLOC-per-story-point, is
the question that decides whether the estimate is usable at all.

**Scenario planning** compares whole named alternatives: the agreed baseline,
the project as currently classified, and hypothetical what-ifs such as
accepting or rejecting a set of pending requests. Hypotheticals are labelled as
such throughout, because a scenario is an argument, not a record.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .cocomo import (
    EFFORT_MULTIPLIERS,
    SCED_SCHEDULE_PERCENT,
    CocomoDrivers,
    CocomoEstimate,
    SizeConversion,
    estimate,
)
from .config import NEW_SCOPE, REFINEMENT, REMOVAL
from .drift import revl_from_drift


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """One costed scenario.

    Attributes
    ----------
    name:
        Short label shown in comparisons.
    size:
        Project size in the analysis's own unit (story points by default).
    revl_percent:
        Requirements volatility rework applied inside the effort equation.
    hypothetical:
        True for what-if scenarios, False for scenarios that describe a real
        recorded state (the agreed baseline, or the current classification).
        The UI and reports rely on this to keep the two apart.
    """

    name: str
    size: float
    revl_percent: float = 0.0
    hypothetical: bool = True
    description: str = ""

    def costed(
        self, conversion: SizeConversion, size_unit: str, drivers: CocomoDrivers
    ) -> CocomoEstimate:
        """Run COCOMO II for this scenario."""
        ksloc = conversion.to_ksloc(max(0.0, self.size), size_unit)
        scenario_drivers = CocomoDrivers(
            ratings=dict(drivers.ratings), revl_percent=self.revl_percent
        )
        return estimate(ksloc, scenario_drivers)


@dataclass(frozen=True)
class ScenarioComparison:
    """A set of scenarios costed against a common reference."""

    table: pd.DataFrame
    reference: str
    size_unit: str
    rate: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def hypothetical_names(self) -> list[str]:
        return list(self.table.loc[self.table["Hypothetical"], "Scenario"])


def build_scenarios(result, *, include_pending: bool = True) -> list[Scenario]:
    """Derive the standard scenario set from a completed analysis.

    Produces, in order:

    1. **Agreed baseline** — recorded. The scope as signed off, no drift.
    2. **Current classification** — recorded. What the tickets say today,
       including any reviewer corrections.
    3. **Reject all new scope** — hypothetical. Keeps refinements and removals,
       drops every New scope ticket.
    4. **Reject new scope and refinements** — hypothetical. Baseline plus
       removals only.
    5. **Accept everything, nothing removed** — hypothetical. The pessimistic
       bound: additions land, withdrawals do not.
    """
    drift = result.drift
    per_ticket = drift.per_ticket
    column = drift.classification_column

    if per_ticket.empty:
        added_new = added_refinement = removed = 0.0
    else:
        added_new = float(per_ticket.loc[per_ticket[column] == NEW_SCOPE, "added"].sum())
        added_refinement = float(
            per_ticket.loc[per_ticket[column] == REFINEMENT, "added"].sum()
        )
        removed = float(drift.size_removed)

    baseline_size = drift.baseline_size
    churn_revl = revl_from_drift(drift) if result.settings.link_drift_to_revl else 0.0

    def revl_for(refinement_added: float, removed_size: float) -> float:
        if not result.settings.link_drift_to_revl or baseline_size <= 0:
            return 0.0
        return max(0.0, (refinement_added + removed_size) / baseline_size * 100.0)

    scenarios = [
        Scenario(
            name="Agreed baseline",
            size=baseline_size,
            revl_percent=result.settings.drivers.revl_percent,
            hypothetical=False,
            description="Scope as originally signed off, before any drift.",
        ),
        Scenario(
            name="Current classification",
            size=drift.revised_size,
            revl_percent=churn_revl,
            hypothetical=False,
            description=(
                "Scope implied by the tickets as currently classified, "
                "including reviewer corrections."
            ),
        ),
    ]

    if include_pending:
        scenarios += [
            Scenario(
                name="Reject all new scope",
                size=baseline_size + added_refinement - removed,
                revl_percent=revl_for(added_refinement, removed),
                description=(
                    "Every New scope ticket is declined. Refinements and "
                    "removals still apply."
                ),
            ),
            Scenario(
                name="Reject new scope and refinements",
                size=baseline_size - removed,
                revl_percent=revl_for(0.0, removed),
                description=(
                    "Only withdrawals are accepted; all growth is declined."
                ),
            ),
            Scenario(
                name="Accept all, remove nothing",
                size=baseline_size + added_new + added_refinement,
                revl_percent=revl_for(added_refinement, 0.0),
                description=(
                    "Pessimistic bound: every addition lands and no withdrawal "
                    "is actioned."
                ),
            ),
        ]

    return scenarios


def compare_scenarios(
    scenarios: list[Scenario],
    *,
    conversion: SizeConversion,
    size_unit: str,
    drivers: CocomoDrivers,
    rate: float | None = None,
    reference: str | None = None,
) -> ScenarioComparison:
    """Cost every scenario and express each against a reference scenario."""
    if not scenarios:
        raise ValueError("No scenarios to compare.")

    reference = reference or scenarios[0].name
    names = [s.name for s in scenarios]
    if reference not in names:
        raise ValueError(
            f"Reference scenario {reference!r} is not in the comparison. "
            f"Available: {', '.join(names)}."
        )

    costed = {s.name: s.costed(conversion, size_unit, drivers) for s in scenarios}
    base = costed[reference]

    rows = []
    for scenario in scenarios:
        run = costed[scenario.name]
        row = {
            "Scenario": scenario.name,
            "Hypothetical": scenario.hypothetical,
            f"Size ({size_unit.replace('_', ' ')})": scenario.size,
            "KSLOC": run.size_ksloc,
            "REVL %": scenario.revl_percent,
            "Effort (PM)": run.person_months,
            "Duration (months)": run.duration_months,
            "Staffing (FTE)": run.average_staff,
            "Δ Effort (PM)": run.person_months - base.person_months,
            "Δ Duration (months)": run.duration_months - base.duration_months,
            "Description": scenario.description,
        }
        if rate:
            row["Cost estimate"] = run.labour_cost(rate)
            row["Δ Cost"] = (run.person_months - base.person_months) * rate
        rows.append(row)

    notes = [
        f"All figures are relative to the '{reference}' scenario.",
        "Scenarios marked hypothetical are what-if arguments, not records of "
        "the project.",
    ]
    if rate:
        notes.append(
            f"Cost uses the supplied rate of {rate:,.0f} per person-month. It "
            "is a derived labour-cost estimate, not an actual or committed "
            "cost, and excludes tooling, licences, overhead and contingency."
        )
    else:
        notes.append("No labour rate supplied, so no cost column is shown.")

    return ScenarioComparison(
        table=pd.DataFrame(rows),
        reference=reference,
        size_unit=size_unit,
        rate=rate,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Sensitivity analysis
# --------------------------------------------------------------------------
#: Assumptions a user can sweep, with the status of each.
SENSITIVITY_PARAMETERS: dict[str, dict[str, object]] = {
    "sloc_per_story_point": {
        "label": "SLOC per story point",
        "status": "Unvalidated placeholder — measure from your own history",
        "default": 60.0,
        "range": (20.0, 120.0),
        "step": 5.0,
        "unit": "SLOC",
    },
    "revl_percent": {
        "label": "Requirements volatility (REVL)",
        "status": "Derived from measured churn; override to test sensitivity",
        "default": 0.0,
        "range": (0.0, 60.0),
        "step": 5.0,
        "unit": "%",
    },
    "monthly_labour_rate": {
        "label": "Labour rate per person-month",
        "status": "User-supplied; no default is assumed",
        "default": 0.0,
        "range": (0.0, 30000.0),
        "step": 1000.0,
        "unit": "currency",
    },
}


def sweep_sloc_per_point(
    size: float,
    *,
    size_unit: str,
    drivers: CocomoDrivers,
    values: tuple[float, ...] | None = None,
    rate: float | None = None,
) -> pd.DataFrame:
    """Effort and duration against the SLOC-per-story-point assumption.

    This is the single most consequential unvalidated input in DriftGuard:
    effort scales with it, so halving it halves every estimate.
    """
    values = values or tuple(np.arange(20.0, 125.0, 10.0))
    rows = []
    for value in values:
        conversion = SizeConversion(sloc_per_story_point=float(value))
        run = estimate(conversion.to_ksloc(size, size_unit), drivers)
        row = {
            "SLOC per story point": float(value),
            "KSLOC": run.size_ksloc,
            "Effort (PM)": run.person_months,
            "Duration (months)": run.duration_months,
            "Staffing (FTE)": run.average_staff,
        }
        if rate:
            row["Cost estimate"] = run.labour_cost(rate)
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_revl(
    size: float,
    *,
    conversion: SizeConversion,
    size_unit: str,
    drivers: CocomoDrivers,
    values: tuple[float, ...] | None = None,
    rate: float | None = None,
) -> pd.DataFrame:
    """Effort and duration against requirements volatility."""
    values = values or tuple(np.arange(0.0, 65.0, 5.0))
    ksloc = conversion.to_ksloc(size, size_unit)
    rows = []
    for value in values:
        run = estimate(
            ksloc,
            CocomoDrivers(ratings=dict(drivers.ratings), revl_percent=float(value)),
        )
        row = {
            "REVL %": float(value),
            "Effective KSLOC": run.effective_ksloc,
            "Effort (PM)": run.person_months,
            "Duration (months)": run.duration_months,
        }
        if rate:
            row["Cost estimate"] = run.labour_cost(rate)
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_driver(
    code: str,
    size: float,
    *,
    conversion: SizeConversion,
    size_unit: str,
    drivers: CocomoDrivers,
) -> pd.DataFrame:
    """Effort and duration across every rating of one COCOMO driver.

    Works for any scale factor or effort multiplier, including SCED, where the
    schedule percentage table applies as well as the effort multiplier.
    """
    from .cocomo import EFFORT_MULTIPLIERS as EM, RATING_LABELS, SCALE_FACTORS as SF

    table = SF.get(code) or EM.get(code)
    if table is None:
        raise ValueError(
            f"Unknown COCOMO driver {code!r}. Expected one of: "
            + ", ".join(list(SF) + list(EM))
        )

    ksloc = conversion.to_ksloc(size, size_unit)
    rows = []
    for rating in table:
        ratings = dict(drivers.ratings)
        ratings[code] = rating
        run = estimate(
            ksloc, CocomoDrivers(ratings=ratings, revl_percent=drivers.revl_percent)
        )
        rows.append(
            {
                "Rating": RATING_LABELS[rating],
                "Code": rating,
                "Driver value": table[rating],
                "Effort (PM)": run.person_months,
                "Duration (months)": run.duration_months,
                "Staffing (FTE)": run.average_staff,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["driver"] = code
    frame.attrs["is_sced"] = code == "SCED"
    return frame


def tornado(
    size: float,
    *,
    conversion: SizeConversion,
    size_unit: str,
    drivers: CocomoDrivers,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank COCOMO drivers by how much effort moves across their rating range.

    A tornado view answers "which of these knobs actually matters?" — useful
    before spending time arguing about any individual rating.
    """
    ksloc = conversion.to_ksloc(size, size_unit)
    nominal = estimate(ksloc, drivers).person_months

    rows = []
    for code, table in EFFORT_MULTIPLIERS.items():
        efforts = []
        for rating in table:
            ratings = dict(drivers.ratings)
            ratings[code] = rating
            efforts.append(
                estimate(
                    ksloc,
                    CocomoDrivers(ratings=ratings, revl_percent=drivers.revl_percent),
                ).person_months
            )
        low, high = min(efforts), max(efforts)
        rows.append(
            {
                "Driver": code,
                "Lowest effort (PM)": low,
                "Highest effort (PM)": high,
                "Swing (PM)": high - low,
                "Swing vs nominal": (high - low) / nominal if nominal else 0.0,
            }
        )

    frame = pd.DataFrame(rows).sort_values("Swing (PM)", ascending=False)
    return frame.head(top_n).reset_index(drop=True)


def elasticity(sweep: pd.DataFrame, x_column: str, y_column: str) -> float:
    """Percentage change in ``y`` per one percent change in ``x``.

    Computed in logs — ``ln(y1/y0) / ln(x1/x0)`` — not as an arc ratio over the
    endpoints. COCOMO's effort equation is a power law, so log elasticity
    recovers the exponent exactly and is constant across the sweep, whereas an
    arc ratio drifts with the width of the range. Over a 20-to-120 sweep the
    arc ratio reads 1.24 where the true elasticity is the exponent 1.0997;
    reporting the former would invite a reviewer to conclude the model has an
    exponent it does not have.

    This is a deterministic slope, not a quantity estimated from samples, so no
    confidence interval accompanies it.
    """
    if len(sweep) < 2:
        return 0.0
    first, last = sweep.iloc[0], sweep.iloc[-1]
    x0, x1 = float(first[x_column]), float(last[x_column])
    y0, y1 = float(first[y_column]), float(last[y_column])
    if min(x0, x1, y0, y1) <= 0:
        return 0.0
    denominator = np.log(x1 / x0)
    if denominator == 0:
        return 0.0
    return float(np.log(y1 / y0) / denominator)
