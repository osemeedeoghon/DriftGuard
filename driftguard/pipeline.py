"""End-to-end analysis pipeline: match -> classify -> review -> drift -> estimate.

Keeping this out of the Streamlit layer means the whole analysis can be run
and tested headlessly, and the dashboard stays a presentation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .classification import MLClassifier, RuleClassifier, apply_overrides
from .cocomo import CocomoDrivers, EstimateComparison, SizeConversion, compare
from .config import DEFAULT_REFINEMENT_WEIGHT, MatchingThresholds
from .drift import DriftResult, calculate_drift, drift_over_time, revl_from_drift
from .matching import RequirementMatcher, attach_requirement_details


@dataclass
class AnalysisSettings:
    """Every tunable assumption in one place, so the UI can display them all."""

    thresholds: MatchingThresholds = field(default_factory=MatchingThresholds)
    refinement_weight: float = DEFAULT_REFINEMENT_WEIGHT
    conversion: SizeConversion = field(default_factory=SizeConversion)
    size_unit: str = "story_points"
    drivers: CocomoDrivers = field(default_factory=CocomoDrivers)
    #: Feed measured drift into the revised estimate's REVL. Turning this off
    #: makes the revised estimate reflect size growth only.
    link_drift_to_revl: bool = True
    classifier: str = "rule"  # "rule" or "ml"
    monthly_labour_rate: float | None = None


@dataclass
class AnalysisResult:
    """Everything the dashboard needs from one run."""

    classified: pd.DataFrame
    drift: DriftResult
    timeline: pd.DataFrame
    estimates: EstimateComparison
    baseline_size: float
    settings: AnalysisSettings
    warnings: list[str] = field(default_factory=list)

    @property
    def baseline_ksloc(self) -> float:
        return self.estimates.baseline.size_ksloc

    @property
    def revised_ksloc(self) -> float:
        return self.estimates.revised.size_ksloc


def run_analysis(
    baseline: pd.DataFrame,
    tickets: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    *,
    overrides: dict[str, str] | None = None,
    size_overrides: dict[str, float] | None = None,
    training_data: pd.DataFrame | None = None,
) -> AnalysisResult:
    """Run the full DriftGuard analysis.

    Parameters
    ----------
    baseline, tickets:
        Frames produced by :mod:`driftguard.data`.
    overrides, size_overrides:
        Reviewer corrections, keyed by ticket id.
    training_data:
        Labelled, already-matched tickets used to fit the learned classifier.
        Required only when ``settings.classifier == "ml"``.
    """
    settings = settings or AnalysisSettings()
    warnings: list[str] = []

    if baseline.empty:
        raise ValueError("Cannot analyse an empty baseline.")

    baseline_size = float(baseline["size"].sum())
    if baseline_size <= 0:
        raise ValueError(
            "The baseline has a total size of zero, so the Scope Drift Index "
            "is undefined. Provide requirement sizes before analysing."
        )

    # 1. Match ------------------------------------------------------------
    matcher = RequirementMatcher().fit(baseline, tickets)
    matched = attach_requirement_details(matcher.match_tickets(tickets), baseline)

    # 2. Classify ---------------------------------------------------------
    if settings.classifier == "ml":
        if training_data is None:
            raise ValueError(
                "The learned classifier needs labelled training tickets. "
                "Upload a labelled test set, or switch to the rule-based "
                "classifier."
            )
        model = MLClassifier().fit(training_data)
        classified = model.classify(matched)
        warnings.append(
            "The learned classifier was trained on the labelled set supplied "
            "in this session. If that is the same data you are scoring, the "
            "reported accuracy is in-sample and will overstate real "
            "performance -- see the cross-validated figures."
        )
    else:
        classified = RuleClassifier(settings.thresholds).classify(matched)

    # 3. Reviewer overrides -----------------------------------------------
    reviewed = apply_overrides(classified, overrides, size_overrides)

    # 4. Drift -------------------------------------------------------------
    drift_result = calculate_drift(
        reviewed,
        baseline_size,
        classification_column="final_class",
        size_column="final_size",
        refinement_weight=settings.refinement_weight,
    )

    if "created_date" in reviewed.columns:
        timeline = drift_over_time(
            reviewed,
            baseline_size,
            classification_column="final_class",
            size_column="final_size",
            refinement_weight=settings.refinement_weight,
        )
        excluded = timeline.attrs.get("excluded_tickets", 0)
        if excluded:
            warnings.append(
                f"{excluded} ticket(s) have no usable date and are missing "
                "from the drift timeline, though they still count towards the "
                "overall Scope Drift Index."
            )
    else:
        timeline = pd.DataFrame()
        warnings.append(
            "Tickets carry no created_date, so drift over time cannot be "
            "plotted."
        )

    # 5. Estimate ----------------------------------------------------------
    baseline_ksloc = settings.conversion.to_ksloc(baseline_size, settings.size_unit)
    revised_ksloc = settings.conversion.to_ksloc(
        max(0.0, drift_result.revised_size), settings.size_unit
    )
    if drift_result.revised_size < 0:
        warnings.append(
            "Removals exceed the agreed baseline, giving a negative revised "
            "size. The revised size was clamped to zero; check the removal "
            "classifications."
        )

    revised_drivers = settings.drivers
    if settings.link_drift_to_revl:
        # Churn, not net growth -- the added scope is already in revised_ksloc,
        # so charging REVL for it as well would count that work twice.
        revised_drivers = CocomoDrivers(
            ratings=dict(settings.drivers.ratings),
            revl_percent=revl_from_drift(drift_result),
        )

    estimates = compare(
        baseline_ksloc,
        revised_ksloc,
        drivers=settings.drivers,
        revised_drivers=revised_drivers,
    )

    # Expose the drift-annotated frame as `classified`: it is the reviewed
    # frame plus `added`, `removed` and `net_contribution`. Callers that want
    # to rank or filter tickets by scope impact need those columns, and having
    # two near-identical frames in circulation invited exactly that bug.
    return AnalysisResult(
        classified=drift_result.per_ticket,
        drift=drift_result,
        timeline=timeline,
        estimates=estimates,
        baseline_size=baseline_size,
        settings=settings,
        warnings=warnings,
    )


def prepare_training_data(
    baseline: pd.DataFrame, labelled_tickets: pd.DataFrame
) -> pd.DataFrame:
    """Match labelled tickets so they can be used to fit the learned model."""
    matcher = RequirementMatcher().fit(baseline, labelled_tickets)
    return attach_requirement_details(
        matcher.match_tickets(labelled_tickets), baseline
    )
