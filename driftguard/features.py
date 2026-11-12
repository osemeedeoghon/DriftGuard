"""Feature-level and model-health views over a completed analysis.

"Feature" here means what it means to the model: the TF-IDF vocabulary and the
similarity score derived from it. These functions summarise how those features
behave across the ticket set, and how much of the classifier's output a human
should be looking at.

Nothing here invents a metric. Every figure is counted or measured from the
analysis the pipeline already produced.
"""

from __future__ import annotations

import pandas as pd

from .config import CLASSES, IN_SCOPE, REFINEMENT, REMOVAL


def requirement_coverage(
    classified: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    class_column: str = "final_class",
) -> pd.DataFrame:
    """How many tickets attach to each agreed baseline requirement.

    Only In-scope and Refinement tickets count as covering a requirement: a
    New scope ticket is matched to its nearest requirement for evidence, but it
    does not implement it, and counting it would overstate coverage.

    An uncovered requirement is genuinely ambiguous -- it may be unstarted
    work, or a ticket the matcher failed to connect -- so the caller is given
    the count and left to interpret it.
    """
    if baseline.empty:
        return pd.DataFrame(columns=["requirement_id", "title", "size", "tickets"])

    covering = classified
    if class_column in classified.columns:
        covering = classified.loc[classified[class_column].isin([IN_SCOPE, REFINEMENT])]

    counts = (
        covering["matched_requirement_id"].value_counts()
        if "matched_requirement_id" in covering.columns
        else pd.Series(dtype=int)
    )

    frame = baseline.copy()
    frame["tickets"] = (
        frame["requirement_id"].map(counts).fillna(0).astype(int)
    )
    columns = [c for c in ("requirement_id", "title", "size", "tickets") if c in frame.columns]
    frame = frame[columns].sort_values("tickets", ascending=False).reset_index(drop=True)
    frame["label"] = (
        frame["requirement_id"].astype(str)
        + " · "
        + frame.get("title", pd.Series([""] * len(frame))).astype(str).str.slice(0, 38)
    )
    return frame


def class_similarity_summary(
    classified: pd.DataFrame, *, class_column: str = "final_class"
) -> pd.DataFrame:
    """Similarity-score spread per class.

    This is the distribution comparison that is actually meaningful for this
    model: how far each class sits from its matched requirement.
    """
    if classified.empty or "similarity" not in classified.columns:
        return pd.DataFrame(
            columns=["class", "tickets", "min", "median", "mean", "max"]
        )

    grouped = (
        classified.groupby(class_column)["similarity"]
        .agg(tickets="size", min="min", median="median", mean="mean", max="max")
        .reindex(list(CLASSES))
        .reset_index()
        .rename(columns={class_column: "class"})
    )
    grouped["tickets"] = grouped["tickets"].fillna(0).astype(int)
    return grouped


def model_health(
    classified: pd.DataFrame,
    *,
    confidence_threshold: float = 0.5,
    class_column: str = "final_class",
) -> dict[str, float | int]:
    """Indicators of how much of the output warrants human attention.

    These are counts and shares, not a quality score. A low-confidence share
    says the model is unsure, which is information; it does not say the model
    is wrong, and nothing here claims the model is performing well. Accuracy
    claims require a labelled set and live in :mod:`driftguard.evaluation`.
    """
    total = len(classified)
    if total == 0:
        return {
            "tickets": 0,
            "low_confidence": 0,
            "low_confidence_share": 0.0,
            "reviewed": 0,
            "reviewed_share": 0.0,
            "overturned": 0,
            "overturned_share": 0.0,
            "mean_confidence": 0.0,
            "median_similarity": 0.0,
            "unmatched": 0,
        }

    low = int((classified["confidence"] < confidence_threshold).sum())
    reviewed = (
        int(classified["reviewed"].sum()) if "reviewed" in classified.columns else 0
    )
    overturned = (
        int(classified["class_changed_by_reviewer"].sum())
        if "class_changed_by_reviewer" in classified.columns
        else 0
    )
    unmatched = (
        int((classified["similarity"] <= 0).sum())
        if "similarity" in classified.columns
        else 0
    )

    return {
        "tickets": total,
        "low_confidence": low,
        "low_confidence_share": low / total,
        "reviewed": reviewed,
        "reviewed_share": reviewed / total,
        "overturned": overturned,
        "overturned_share": overturned / total,
        "mean_confidence": float(classified["confidence"].mean()),
        "median_similarity": float(classified["similarity"].median()),
        "unmatched": unmatched,
    }


def review_queue(
    classified: pd.DataFrame,
    *,
    confidence_threshold: float = 0.5,
    limit: int = 25,
) -> pd.DataFrame:
    """Tickets the classifier is least sure about, lowest confidence first."""
    if classified.empty:
        return classified

    queue = classified.loc[classified["confidence"] < confidence_threshold]
    columns = [
        c
        for c in (
            "ticket_id",
            "title",
            "predicted_class",
            "final_class",
            "confidence",
            "similarity",
            "matched_requirement_id",
            "reviewed",
        )
        if c in queue.columns
    ]
    return queue.sort_values("confidence")[columns].head(limit).reset_index(drop=True)


#: A ticket counts as high-impact when it moves scope by at least this share
#: of the agreed baseline on its own.
#:
#: Deliberately an absolute share rather than a quantile of the observed
#: impacts. A quantile cutoff moves whenever a classification changes, so
#: reviewing a large ticket could *raise* the count of unreviewed high-impact
#: tickets -- which is both wrong-looking and genuinely unhelpful. This
#: threshold is a reading convention for the dashboard, not a standard.
HIGH_IMPACT_SHARE = 0.05


def high_impact_threshold(baseline_size: float, share: float = HIGH_IMPACT_SHARE) -> float:
    """Absolute scope change at which a single ticket counts as high-impact."""
    return max(0.0, baseline_size * share)


def high_impact_mask(classified: pd.DataFrame, baseline_size: float) -> pd.Series:
    """Boolean mask of tickets that individually move scope materially."""
    if classified.empty or "net_contribution" not in classified.columns:
        return pd.Series(dtype=bool, index=classified.index)
    threshold = high_impact_threshold(baseline_size)
    return classified["net_contribution"].abs() >= max(threshold, 1e-9)


def risk_indicators(result) -> pd.DataFrame:
    """Project risk indicators derived from the analysis.

    Each row states the measured value, a presentational band, and the reason
    it matters. The bands are reading conventions for this dashboard, not
    thresholds from any standard, and the ``Basis`` column says so where it
    applies. Nothing here is a prediction.
    """
    drift = result.drift
    estimates = result.estimates
    classified = result.classified
    health = model_health(classified)

    rows: list[dict[str, object]] = []

    def add(name: str, value: str, level: str, why: str, basis: str) -> None:
        rows.append(
            {
                "Indicator": name,
                "Value": value,
                "Level": level,
                "Why it matters": why,
                "Basis": basis,
            }
        )

    sdi = drift.sdi
    add(
        "Net scope growth",
        f"{sdi:+.1%}",
        "Critical" if sdi > 0.25 else "Serious" if sdi > 0.10 else "Watch" if sdi > 0.05 else "Stable",
        "Growth against an agreed baseline is the headline governance concern.",
        "Measured from classified tickets",
    )

    effort_change = estimates.effort_change_percent
    add(
        "Effort growth",
        f"{effort_change:+.1f}%",
        "Critical" if effort_change > 50 else "Serious" if effort_change > 20 else "Stable",
        "Effort rises faster than scope because the COCOMO exponent exceeds 1.",
        "COCOMO II, published calibration",
    )

    schedule_change = estimates.duration_change_percent
    add(
        "Schedule growth",
        f"{schedule_change:+.1f}%",
        "Serious" if schedule_change > 20 else "Watch" if schedule_change > 10 else "Stable",
        "Calendar time grows more slowly than effort, so staffing must absorb the rest.",
        "COCOMO II, published calibration",
    )

    staffing_change = (
        estimates.revised.average_staff - estimates.baseline.average_staff
    )
    add(
        "Implied staffing change",
        f"{staffing_change:+.1f} FTE",
        "Serious" if staffing_change > 2 else "Watch" if staffing_change > 0.5 else "Stable",
        "Effort divided by duration. Adding people is not free or instant.",
        "Derived from effort and duration",
    )

    threshold = high_impact_threshold(drift.baseline_size)
    unreviewed_high_impact = 0
    high_impact_total = 0
    if not classified.empty and "net_contribution" in classified.columns:
        mask = high_impact_mask(classified, drift.baseline_size)
        high_impact_total = int(mask.sum())
        reviewed_flags = (
            classified["reviewed"].astype(bool)
            if "reviewed" in classified.columns
            else pd.Series(False, index=classified.index)
        )
        unreviewed_high_impact = int((mask & ~reviewed_flags).sum())
    add(
        "High-impact tickets not yet reviewed",
        f"{unreviewed_high_impact} of {high_impact_total}",
        "Serious" if unreviewed_high_impact > 5 else "Watch" if unreviewed_high_impact else "Stable",
        "The largest scope changes carry the most risk of misclassification.",
        f"Individually move scope by at least {threshold:,.1f} "
        f"({HIGH_IMPACT_SHARE:.0%} of baseline)",
    )

    add(
        "Low-confidence classifications",
        f"{health['low_confidence']} ({health['low_confidence_share']:.0%})",
        "Serious" if health["low_confidence_share"] > 0.25 else "Watch" if health["low_confidence_share"] > 0.1 else "Stable",
        "Tickets the model is unsure about need human adjudication.",
        "Classifier confidence below 0.50",
    )

    removal_share = 0.0
    if not drift.per_class.empty and drift.baseline_size:
        removal_share = drift.size_removed / drift.baseline_size
    add(
        "Baseline withdrawn",
        f"{removal_share:.1%}",
        "Serious" if removal_share > 0.20 else "Watch" if removal_share > 0.10 else "Stable",
        "Heavy de-scoping can mean the original agreement no longer holds.",
        "Removal tickets as a share of baseline",
    )

    return pd.DataFrame(rows)


def recommendations(result) -> list[dict[str, str]]:
    """Actionable next steps implied by the analysis.

    These follow mechanically from measured figures. None is a prediction, and
    none claims an outcome -- each names what was observed and what a project
    manager could do about it.
    """
    drift = result.drift
    estimates = result.estimates
    classified = result.classified
    health = model_health(classified)
    unit = result.settings.size_unit.replace("_", " ")
    items: list[dict[str, str]] = []

    if drift.sdi > 0.10:
        top = drift.top_contributors(3)
        names = ", ".join(str(t) for t in top.get("ticket_id", [])[:3])
        items.append(
            {
                "priority": "High",
                "action": "Take the largest scope additions back to the sponsor",
                "because": (
                    f"Net scope has grown {drift.sdi:.1%}. The three largest "
                    f"contributors ({names}) account for "
                    f"{top['net_contribution'].abs().sum():,.0f} {unit} between them."
                ),
            }
        )

    if estimates.effort_change_percent > 20:
        items.append(
            {
                "priority": "High",
                "action": "Re-baseline the effort estimate or reduce scope",
                "because": (
                    f"Estimated effort has moved from "
                    f"{estimates.baseline.person_months:,.1f} to "
                    f"{estimates.revised.person_months:,.1f} person-months "
                    f"({estimates.effort_change_percent:+.1f}%). The schedule has "
                    f"moved only {estimates.duration_change_percent:+.1f}%, so the "
                    "difference has to be absorbed by staffing."
                ),
            }
        )

    if health["low_confidence"] > 0:
        items.append(
            {
                "priority": "Medium",
                "action": f"Review the {health['low_confidence']} low-confidence classifications",
                "because": (
                    "The classifier flagged these as uncertain. Until they are "
                    "adjudicated, the Scope Drift Index carries their error."
                ),
            }
        )

    if health["reviewed"] == 0:
        items.append(
            {
                "priority": "Medium",
                "action": "Have a human confirm the classifications before quoting the index",
                "because": (
                    "No ticket has been reviewed yet, so every figure here is "
                    "unvalidated model output."
                ),
            }
        )

    if result.settings.monthly_labour_rate is None:
        items.append(
            {
                "priority": "Low",
                "action": "Supply a labour rate if a financial figure is needed",
                "because": (
                    "DriftGuard reports effort and duration only, and will not "
                    "assume a rate."
                ),
            }
        )

    items.append(
        {
            "priority": "Low",
            "action": "Validate the SLOC-per-story-point factor against completed work",
            "because": (
                f"Every effort figure scales with it. It is currently "
                f"{result.settings.conversion.sloc_per_story_point:g}, which is a "
                "placeholder rather than a measurement."
            ),
        }
    )

    return items


def cue_evidence(text: str) -> pd.DataFrame:
    """Which keyword cues a ticket triggered, and what each one signals.

    The rule classifier's rationale already names the cues in prose. This
    returns them as data so the dashboard can show the evidence separately
    from the conclusion -- a reviewer should be able to see *that* a cue fired
    without having to parse a sentence about it.

    Returns columns ``Cue``, ``Signals`` and ``Fired``; the frame lists only
    cues that actually matched.
    """
    from .classification import detect_removal, find_cues
    from .config import NEW_SCOPE_CUES, REFINEMENT_CUES

    rows: list[dict[str, object]] = []
    for cue in detect_removal(text or ""):
        rows.append({"Cue": cue, "Signals": REMOVAL, "Fired": True})
    for cue in find_cues(text or "", REFINEMENT_CUES):
        rows.append({"Cue": cue, "Signals": REFINEMENT, "Fired": True})
    for cue in find_cues(text or "", NEW_SCOPE_CUES):
        rows.append({"Cue": cue, "Signals": "New scope", "Fired": True})

    if not rows:
        return pd.DataFrame(columns=["Cue", "Signals", "Fired"])
    return pd.DataFrame(rows)


def explain_ticket(row: pd.Series, thresholds) -> dict[str, object]:
    """Everything a reviewer needs to audit one classification.

    Assembles the identifiers, the match, the evidence, the decision and the
    override status into one structure, so the explanation panel is not
    reaching into a dozen columns at the point of rendering.
    """
    predicted = str(row.get("predicted_class", ""))
    final = str(row.get("final_class", predicted))
    similarity = float(row.get("similarity", 0.0))

    if similarity >= thresholds.strong_match:
        band = "Confident match"
    elif similarity >= thresholds.weak_match:
        band = "Uncertain band — worth review"
    else:
        band = "Below weak threshold — treated as unrelated"

    return {
        "ticket_id": str(row.get("ticket_id", "")),
        "title": str(row.get("title", "")),
        "description": str(row.get("description", "")),
        "predicted_class": predicted,
        "final_class": final,
        "overridden": final != predicted,
        "confidence": float(row.get("confidence", 0.0)),
        "matched_requirement_id": str(row.get("matched_requirement_id", "")),
        "matched_requirement_title": row.get("matched_requirement_title"),
        "matched_requirement_text": row.get("matched_requirement_text"),
        "similarity": similarity,
        "similarity_band": band,
        "match_margin": float(row.get("match_margin", 0.0)),
        "runner_up_id": row.get("runner_up_id"),
        "shared_terms": tuple(row.get("shared_terms") or ()),
        "match_evidence": str(row.get("match_evidence", "")),
        "rationale": str(row.get("rationale", "")),
        "classifier": str(row.get("classifier", "")),
        "size": float(row.get("size", 0.0)),
        "final_size": float(row.get("final_size", row.get("size", 0.0))),
        "reviewed": bool(row.get("reviewed", False)),
    }


def dataset_profile(frame: pd.DataFrame, name: str) -> dict[str, object]:
    """Row/column counts and completeness for one uploaded dataset."""
    rows, columns = frame.shape
    cells = rows * columns
    missing = int(frame.isna().sum().sum()) if cells else 0
    return {
        "dataset": name,
        "rows": rows,
        "columns": columns,
        "missing_cells": missing,
        "completeness": 1.0 - (missing / cells) if cells else 1.0,
    }
