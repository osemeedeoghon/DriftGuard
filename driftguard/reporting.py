"""Exportable analysis reports.

Builds a Markdown summary from a completed analysis. Every figure is read from
the ``AnalysisResult``; nothing is recomputed differently for the report, and
nothing is asserted that the analysis does not support.

Two rules this module enforces:

* **No unsupported quality claims.** Classifier accuracy appears only when a
  labelled test set was supplied, and is always labelled in-sample or
  cross-validated.
* **No invented costs.** A currency figure appears only when the caller
  supplied a labour rate, and is always called an estimate.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .config import CLASSES
from .evaluation import EvaluationResult
from .pipeline import AnalysisResult


def _fmt(value: float, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}"


def _pct(value: float, decimals: int = 1, signed: bool = False) -> str:
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:,.{decimals}f}%"


def build_markdown_report(
    result: AnalysisResult,
    *,
    project_name: str = "DriftGuard analysis",
    evaluation: EvaluationResult | None = None,
    cross_validated: pd.DataFrame | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Assemble a self-contained Markdown report for the current analysis."""
    drift = result.drift
    estimates = result.estimates
    settings = result.settings
    unit = settings.size_unit.replace("_", " ")
    stamp = (generated_at or datetime.now()).strftime("%d %b %Y, %H:%M")

    revl_percent = 0.0
    if estimates.revised.size_ksloc:
        revl_percent = (
            estimates.revised.effective_ksloc / estimates.revised.size_ksloc - 1
        ) * 100

    lines: list[str] = [
        f"# {project_name}",
        "",
        f"Generated {stamp} by DriftGuard. All analysis ran locally.",
        "",
        "## Scope position",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Scope Drift Index | **{_pct(drift.sdi, signed=True)}** |",
        f"| Baseline size | {_fmt(drift.baseline_size, 0)} {unit} |",
        f"| Size added | {_fmt(drift.size_added, 0)} {unit} |",
        f"| Size removed | {_fmt(drift.size_removed, 0)} {unit} |",
        f"| Revised size | {_fmt(drift.revised_size, 0)} {unit} |",
        f"| Tickets analysed | {len(result.classified):,} |",
        "",
        "The Scope Drift Index is "
        "`(size added − size removed) ÷ baseline size`, computed on net "
        "change. Refinement tickets contribute "
        f"{settings.refinement_weight:.2f} of their size, which is an "
        "assumption set by the reviewer.",
        "",
        "## Classification breakdown",
        "",
        "| Class | Tickets | Size | Added | Removed |",
        "|---|---|---|---|---|",
    ]

    per_class = drift.per_class.set_index("classification")
    for name in CLASSES:
        if name not in per_class.index:
            continue
        row = per_class.loc[name]
        lines.append(
            f"| {name} | {int(row['tickets']):,} | {_fmt(row['size'], 0)} | "
            f"{_fmt(row['added'], 1)} | {_fmt(row['removed'], 1)} |"
        )

    lines += [
        "",
        "## Effort and schedule impact",
        "",
        "COCOMO II Post-Architecture, published COCOMO II.2000 calibration "
        "(A = 2.94, B = 0.91, C = 3.67, D = 0.28). No local calibration was "
        "performed.",
        "",
        "| Quantity | Baseline | Revised | Change |",
        "|---|---|---|---|",
        f"| Size ({unit}) | {_fmt(drift.baseline_size, 0)} | "
        f"{_fmt(drift.revised_size, 0)} | {_pct(estimates.size_change_percent / 100, signed=True)} |",
        f"| Size (KSLOC) | {_fmt(estimates.baseline.size_ksloc, 2)} | "
        f"{_fmt(estimates.revised.size_ksloc, 2)} | — |",
        f"| REVL applied | {_fmt(settings.drivers.revl_percent, 1)}% | "
        f"{_fmt(revl_percent, 1)}% | — |",
        f"| Effort (person-months) | {_fmt(estimates.baseline.person_months)} | "
        f"{_fmt(estimates.revised.person_months)} | "
        f"{_pct(estimates.effort_change_percent / 100, signed=True)} |",
        f"| Duration (calendar months) | {_fmt(estimates.baseline.duration_months)} | "
        f"{_fmt(estimates.revised.duration_months)} | "
        f"{_pct(estimates.duration_change_percent / 100, signed=True)} |",
        f"| Average staffing (FTE) | {_fmt(estimates.baseline.average_staff)} | "
        f"{_fmt(estimates.revised.average_staff)} | — |",
        "",
        "Person-months measure **effort**; months measure **calendar "
        "duration**. They are different quantities and do not convert into one "
        "another by adding headcount.",
        "",
    ]

    if settings.monthly_labour_rate:
        rate = settings.monthly_labour_rate
        lines += [
            f"At the supplied rate of {_fmt(rate, 0)} per person-month, the "
            f"revised effort is an estimated {_fmt(estimates.revised.labour_cost(rate), 0)}, "
            f"against {_fmt(estimates.baseline.labour_cost(rate), 0)} for the "
            "baseline. This is a derived labour-cost **estimate**, not an "
            "actual or committed cost, and excludes tooling, licences, "
            "overhead and contingency.",
            "",
        ]
    else:
        lines += [
            "No labour rate was supplied, so no cost figure is reported. "
            "Effort and duration are given above.",
            "",
        ]

    lines += ["## Tickets driving the drift", "", "| Ticket | Class | Size | Net change |", "|---|---|---|---|"]
    top = drift.top_contributors(10)
    for _, row in top.iterrows():
        title = str(row.get("title", ""))[:48]
        label = f"{row.get('ticket_id', '?')} — {title}" if title else row.get("ticket_id", "?")
        lines.append(
            f"| {label} | {row.get('final_class', '—')} | "
            f"{_fmt(float(row.get('final_size', 0)), 0)} | "
            f"{float(row['net_contribution']):+,.1f} |"
        )

    lines += ["", "## Classifier quality", ""]
    if evaluation is None:
        lines += [
            "No labelled test set was supplied, so classifier accuracy was "
            "**not measured**. The classifications above are model output that "
            "has not been scored against ground truth.",
            "",
        ]
    else:
        lines += [
            f"Scored against {evaluation.n_samples:,} manually labelled tickets.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Accuracy | {_pct(evaluation.accuracy)} |",
            f"| Macro-F1 | {evaluation.macro_f1:.3f} |",
            f"| Weighted F1 | {evaluation.weighted_f1:.3f} |",
            f"| Majority-class baseline | {_pct(evaluation.baseline_majority_accuracy)} |",
            "",
            "Macro-F1 is the headline figure rather than accuracy, because the "
            "four classes are imbalanced and a model that never predicts the "
            "rare Removal class can still post a flattering accuracy.",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---|---|---|---|",
        ]
        for _, row in evaluation.per_class.iterrows():
            lines.append(
                f"| {row['class']} | {row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['f1']:.3f} | {int(row['support'])} |"
            )
        lines += [
            "",
            "**These figures are in-sample.** The rule classifier's thresholds "
            "and keyword lexicon were tuned on this data, so the scores above "
            "overstate performance on unseen tickets.",
            "",
        ]

        if cross_validated is not None and not cross_validated.empty:
            lines += [
                "### Cross-validated (out-of-sample)",
                "",
                f"Mean accuracy {_pct(cross_validated.attrs.get('mean_accuracy', 0.0))}, "
                f"mean macro-F1 {cross_validated.attrs.get('mean_macro_f1', 0.0):.3f} "
                f"over {cross_validated.attrs.get('folds_used', len(cross_validated))} "
                "stratified folds of the learned classifier. The gap between "
                "these and the in-sample figures is the cost of fitting to a "
                "single project's vocabulary.",
                "",
            ]

    lines += ["## Assumptions in force", "", "| Assumption | Value |", "|---|---|"]
    conversion = settings.conversion
    lines += [
        f"| Ticket size unit | {unit} |",
        f"| SLOC per story point | {_fmt(conversion.sloc_per_story_point, 0)} |",
        f"| SLOC per function point | {_fmt(conversion.sloc_per_function_point, 0)} |",
        f"| Refinement weight | {settings.refinement_weight:.2f} |",
        f"| Drift linked to REVL | {'yes' if settings.link_drift_to_revl else 'no'} |",
        f"| Similarity thresholds | weak {settings.thresholds.weak_match:.2f}, "
        f"strong {settings.thresholds.strong_match:.2f} |",
        f"| COCOMO drivers | "
        f"{'all nominal' if set(settings.drivers.ratings.values()) == {'N'} else 'customised'} |",
        "",
        "The SLOC conversion factor is an organisational assumption, not a "
        "measurement. Every effort and duration figure scales with it.",
        "",
    ]

    if result.warnings:
        lines += ["## Warnings raised during this analysis", ""]
        lines += [f"- {warning}" for warning in result.warnings]
        lines += [""]

    lines += [
        "## Limitations",
        "",
        "- TF-IDF similarity measures lexical overlap, not meaning. A ticket "
        "that paraphrases a requirement without reusing its vocabulary will be "
        "misread as new scope.",
        "- The keyword lexicon is domain-specific and will not transfer "
        "unchanged to another project.",
        "- Ticket sizes are human estimates, and COCOMO II's own published "
        "accuracy is roughly within 30% of actuals 70% of the time under "
        "favourable conditions.",
        "- Scope is bounded to a single project and English-language tickets.",
        "",
    ]

    return "\n".join(lines)


def report_bytes(markdown: str) -> bytes:
    """Encode a report for download."""
    return markdown.encode("utf-8")


def classified_export(classified: pd.DataFrame) -> pd.DataFrame:
    """Ticket-level export, with list columns flattened for CSV."""
    frame = classified.drop(columns=["alternative_matches"], errors="ignore").copy()
    if "shared_terms" in frame.columns:
        frame["shared_terms"] = frame["shared_terms"].apply(
            lambda terms: ", ".join(terms) if isinstance(terms, (list, tuple)) else terms
        )
    return frame
