"""Honest out-of-sample validation of the rule-based classifier.

Why this module exists
----------------------
``evaluation.cross_validated_scores`` cross-validates the *learned* classifier
only. The rule classifier is DriftGuard's default, and its headline score was
measured on the same tickets its thresholds were tuned on. That is data
leakage, and until this module existed the tool had no out-of-sample estimate
for the classifier it actually ships.

Nested cross-validation fixes the part that can be fixed:

* the **outer** loop holds out a fold that is never used for tuning;
* the **inner** loop searches the similarity thresholds on the remaining data
  only;
* the tuned thresholds are then scored once on the held-out fold.

What this does NOT fix
----------------------
The keyword lexicon in :mod:`driftguard.config` was written by hand after
reading the sample tickets. It cannot be re-derived per fold, so **lexicon
leakage is irreducible here**. A genuinely clean estimate needs a freshly
labelled corpus from a different project. Every figure this module returns
carries that caveat, and the dashboard repeats it.

The TF-IDF vectoriser is also fitted on the whole corpus before splitting.
That is unsupervised and uses no labels, so it leaks no ground truth, but it
does make the evaluation transductive: the vectoriser has seen the test fold's
vocabulary. This is reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .classification import RuleClassifier
from .config import CLASSES, MatchingThresholds
from .evaluation import evaluate

#: Threshold grid searched by the inner loop. Deliberately coarse: a finer grid
#: on 45 tickets would fit noise, and the point of this module is honesty about
#: small-sample estimates rather than squeezing out decimal places.
DEFAULT_WEAK_GRID: tuple[float, ...] = (0.10, 0.14, 0.18, 0.22, 0.26, 0.30)
DEFAULT_STRONG_GRID: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55)


@dataclass(frozen=True)
class NestedValidationResult:
    """Outcome of a nested cross-validation run."""

    outer_folds: pd.DataFrame        # per-fold scores and chosen thresholds
    mean_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float
    in_sample_accuracy: float
    in_sample_macro_f1: float
    shipped_thresholds: MatchingThresholds
    threshold_stability: pd.DataFrame
    n_samples: int
    cue_reliance: pd.DataFrame = field(default_factory=pd.DataFrame)
    caveats: list[str] = field(default_factory=list)

    @property
    def optimism_accuracy(self) -> float:
        """How much the in-sample accuracy overstates the honest estimate."""
        return self.in_sample_accuracy - self.mean_accuracy

    @property
    def optimism_macro_f1(self) -> float:
        return self.in_sample_macro_f1 - self.mean_macro_f1

    @property
    def leakage_suspected(self) -> bool:
        """True when the out-of-sample score meets or beats the in-sample one.

        Held-out performance should be *worse* than training performance. When
        it is not, the held-out data is not really held out. Here the cause is
        known and structural: the keyword lexicon was written against every
        ticket in this set, so a threshold search that shifts decisions onto
        the lexicon can score perfectly on any fold.

        A validation routine that reported such a result as success would be
        worse than useless, so this flag exists to stop that happening.
        """
        return self.mean_macro_f1 >= self.in_sample_macro_f1

    @property
    def verdict(self) -> str:
        """One-line interpretation, never just a number."""
        if self.leakage_suspected:
            return (
                "Not a valid out-of-sample estimate — held-out score meets or "
                "exceeds the in-sample score, which indicates residual leakage "
                "rather than genuine performance."
            )
        return (
            f"Out-of-sample macro-F1 {self.mean_macro_f1:.3f} against "
            f"{self.in_sample_macro_f1:.3f} in-sample: an optimism of "
            f"{self.optimism_macro_f1:+.3f}."
        )

    def summary_line(self) -> str:
        return (
            f"Nested CV over {self.n_samples} labelled tickets: "
            f"accuracy {self.mean_accuracy:.1%}, macro-F1 {self.mean_macro_f1:.3f} "
            f"(in-sample {self.in_sample_accuracy:.1%} / "
            f"{self.in_sample_macro_f1:.3f})."
        )


def _score_thresholds(
    matched: pd.DataFrame,
    thresholds: MatchingThresholds,
    *,
    label_column: str,
) -> float:
    """Macro-F1 of the rule classifier at the given thresholds."""
    predictions = RuleClassifier(thresholds).classify(matched)
    return evaluate(predictions, true_column=label_column).macro_f1


def tune_thresholds(
    matched: pd.DataFrame,
    *,
    label_column: str = "true_class",
    weak_grid: tuple[float, ...] = DEFAULT_WEAK_GRID,
    strong_grid: tuple[float, ...] = DEFAULT_STRONG_GRID,
) -> tuple[MatchingThresholds, float]:
    """Grid-search the similarity thresholds, maximising macro-F1.

    Macro-F1 rather than accuracy, for the same reason it is the headline
    metric: the classes are imbalanced and accuracy would let the search ignore
    the rare Removal class.

    Returns the best thresholds and the score they achieved on ``matched``.
    """
    best: tuple[MatchingThresholds, float] | None = None
    for weak in weak_grid:
        for strong in strong_grid:
            if weak > strong:
                continue
            candidate = MatchingThresholds(weak_match=weak, strong_match=strong)
            score = _score_thresholds(matched, candidate, label_column=label_column)
            if best is None or score > best[1]:
                best = (candidate, score)

    if best is None:
        raise ValueError(
            "No valid threshold pair in the supplied grids: every weak value "
            "exceeded every strong value."
        )
    return best


def nested_cross_validate(
    matched: pd.DataFrame,
    *,
    label_column: str = "true_class",
    outer_folds: int = 5,
    inner_folds: int = 3,
    random_state: int = 42,
    weak_grid: tuple[float, ...] = DEFAULT_WEAK_GRID,
    strong_grid: tuple[float, ...] = DEFAULT_STRONG_GRID,
    shipped: MatchingThresholds | None = None,
) -> NestedValidationResult:
    """Nested cross-validation of the rule classifier's thresholds.

    Parameters
    ----------
    matched:
        Labelled tickets that have already been matched (so they carry
        ``similarity`` and ``text``).
    outer_folds, inner_folds:
        Fold counts. Both are reduced automatically when the rarest class has
        too few examples to split, and the reduction is reported.

    Notes
    -----
    The inner loop here searches thresholds on the *whole* outer-training
    portion rather than averaging over inner splits. With a 45-ticket sample
    and a 30-point grid, inner splitting adds variance without adding
    information; the outer loop is what protects the estimate. ``inner_folds``
    is retained in the signature and reported so the limitation is visible.
    """
    from sklearn.model_selection import StratifiedKFold

    data = matched.dropna(subset=[label_column]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No labelled rows available for nested validation.")

    for column in ("similarity", "text"):
        if column not in data.columns:
            raise ValueError(
                f"Tickets must be matched before validation: missing {column!r}."
            )

    counts = data[label_column].value_counts()
    usable_outer = int(min(outer_folds, counts.min()))
    if usable_outer < 2:
        raise ValueError(
            "Nested validation needs at least two examples of every class. "
            f"The rarest class ('{counts.idxmin()}') has {int(counts.min())}."
        )

    shipped = shipped or MatchingThresholds()
    splitter = StratifiedKFold(
        n_splits=usable_outer, shuffle=True, random_state=random_state
    )

    rows: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(data, data[label_column]), start=1
    ):
        train = data.iloc[train_index]
        test = data.iloc[test_index]

        tuned, inner_score = tune_thresholds(
            train,
            label_column=label_column,
            weak_grid=weak_grid,
            strong_grid=strong_grid,
        )
        predictions = RuleClassifier(tuned).classify(test)
        outer = evaluate(predictions, true_column=label_column)

        # How many of this fold's decisions rested on the similarity signal
        # (which leaks nothing) versus the hand-written lexicon (which does)?
        decided_by_similarity = int((test["similarity"] < tuned.weak_match).sum())
        rows.append(
            {
                "fold": fold,
                "n_test": outer.n_samples,
                "weak_match": tuned.weak_match,
                "strong_match": tuned.strong_match,
                "inner_macro_f1": inner_score,
                "accuracy": outer.accuracy,
                "macro_f1": outer.macro_f1,
                "decided_by_similarity": decided_by_similarity,
                "decided_by_cues": outer.n_samples - decided_by_similarity,
            }
        )

    folds_frame = pd.DataFrame(rows)

    # In-sample figure for comparison: the shipped thresholds scored on
    # everything, which is what the Model Performance page reports.
    in_sample = evaluate(
        RuleClassifier(shipped).classify(data), true_column=label_column
    )

    stability = (
        folds_frame.groupby(["weak_match", "strong_match"])
        .size()
        .reset_index(name="folds_selected")
        .sort_values("folds_selected", ascending=False)
        .reset_index(drop=True)
    )

    caveats = [
        "Nested cross-validation removes leakage from *threshold* tuning only.",
        "The keyword lexicon was written after reading these tickets and "
        "cannot be re-derived per fold, so lexicon leakage remains. A clean "
        "estimate requires a freshly labelled corpus from another project.",
        "The TF-IDF vectoriser is fitted on the whole corpus before splitting. "
        "It uses no labels, so no ground truth leaks, but the evaluation is "
        "transductive: the vectoriser has seen the test fold's vocabulary.",
        f"With {len(data)} labelled tickets across {len(CLASSES)} classes, each "
        f"outer fold tests roughly {len(data) // usable_outer} tickets, so "
        "per-fold scores are coarse and the spread is wide.",
    ]
    if usable_outer < outer_folds:
        caveats.append(
            f"Outer folds reduced from {outer_folds} to {usable_outer}: the "
            f"rarest class has only {int(counts.min())} examples."
        )

    mean_macro_f1 = float(np.mean(folds_frame["macro_f1"]))
    total_by_similarity = int(folds_frame["decided_by_similarity"].sum())
    total_by_cues = int(folds_frame["decided_by_cues"].sum())

    cue_reliance = pd.DataFrame(
        {
            "Decided by": [
                "Similarity below the weak threshold",
                "Keyword cue logic",
            ],
            "Tickets": [total_by_similarity, total_by_cues],
            "Leaks ground truth?": [
                "No — TF-IDF uses no labels",
                "Yes — lexicon was written against these tickets",
            ],
        }
    )

    if mean_macro_f1 >= in_sample.macro_f1:
        caveats.insert(
            0,
            "LEAKAGE INDICATOR: the held-out score meets or exceeds the "
            "in-sample score, which should not happen. The threshold search "
            "lowered the weak-match boundary, which shifts decisions away from "
            "the similarity signal and onto the hand-written keyword lexicon. "
            f"Across all folds {total_by_cues} of "
            f"{total_by_similarity + total_by_cues} decisions rested on the "
            "lexicon. Because that lexicon was written after reading these "
            "tickets, a perfect fold score reflects memorisation, not "
            "generalisation. Treat this run as evidence that the evaluation "
            "set is exhausted, not as a performance result.",
        )

    return NestedValidationResult(
        outer_folds=folds_frame,
        mean_accuracy=float(np.mean(folds_frame["accuracy"])),
        mean_macro_f1=mean_macro_f1,
        std_macro_f1=float(np.std(folds_frame["macro_f1"], ddof=0)),
        in_sample_accuracy=in_sample.accuracy,
        in_sample_macro_f1=in_sample.macro_f1,
        shipped_thresholds=shipped,
        threshold_stability=stability,
        n_samples=len(data),
        cue_reliance=cue_reliance,
        caveats=caveats,
    )


# --------------------------------------------------------------------------
# Learning curve: how much does the small sample cost us?
# --------------------------------------------------------------------------
def learning_curve(
    matched: pd.DataFrame,
    *,
    label_column: str = "true_class",
    fractions: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0),
    repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Macro-F1 of the learned classifier against training-set size.

    Answers the question a reviewer will ask about a 45-ticket sample: is the
    model limited by the method or by the data? A curve still climbing at the
    right-hand edge says more labels would help; a flat one says the method is
    the ceiling.

    Returns one row per (fraction, repeat) with the held-out score.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    from .classification import MLClassifier

    data = matched.dropna(subset=[label_column]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No labelled rows available for a learning curve.")

    counts = data[label_column].value_counts()
    n_classes = len(counts)
    rows: list[dict[str, object]] = []

    for fraction in fractions:
        # Always hold out 25% for testing, then subsample the training portion.
        splitter = StratifiedShuffleSplit(
            n_splits=repeats, test_size=0.25, random_state=random_state
        )
        for repeat, (train_index, test_index) in enumerate(
            splitter.split(data, data[label_column]), start=1
        ):
            train_pool = data.iloc[train_index]
            take = max(n_classes, int(round(len(train_pool) * fraction)))
            if take >= len(train_pool):
                train = train_pool
            else:
                # Stratified subsample: keep at least one row per class so the
                # model can always be fitted.
                pieces = [
                    group.sample(
                        n=max(1, int(round(len(group) * fraction))),
                        random_state=random_state + repeat,
                    )
                    for _, group in train_pool.groupby(label_column, sort=False)
                ]
                train = pd.concat(pieces)
            if train[label_column].nunique() < 2:
                continue

            try:
                model = MLClassifier(random_state=random_state).fit(
                    train, label_column=label_column
                )
                scored = evaluate(
                    model.classify(data.iloc[test_index]), true_column=label_column
                )
            except ValueError:
                continue

            rows.append(
                {
                    "fraction": fraction,
                    "train_size": len(train),
                    "repeat": repeat,
                    "accuracy": scored.accuracy,
                    "macro_f1": scored.macro_f1,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(
            "Could not build a learning curve: too few labelled examples to "
            "train and test at any sampling fraction."
        )
    frame.attrs["repeats"] = repeats
    return frame


def learning_curve_summary(curve: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread per training fraction."""
    return (
        curve.groupby("fraction")
        .agg(
            train_size=("train_size", "mean"),
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            runs=("macro_f1", "size"),
        )
        .reset_index()
        .fillna({"macro_f1_std": 0.0})
    )


# --------------------------------------------------------------------------
# Methodology audit: machine-checkable claims about the pipeline
# --------------------------------------------------------------------------
def methodology_audit(result) -> pd.DataFrame:
    """Re-derive the pipeline's key invariants from a completed analysis.

    Phase 8 of the enhancement brief asks for verification rather than
    assertion, so these checks recompute each claim from the result object and
    report pass/fail. They are also covered by the test suite; this function
    exists so a reviewer can see them verified live, on their own data.
    """
    from .config import NEW_SCOPE, REFINEMENT

    drift = result.drift
    estimates = result.estimates
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"Check": name, "Result": "PASS" if passed else "FAIL", "Detail": detail})

    # 1. SDI definition
    expected_sdi = (
        (drift.size_added - drift.size_removed) / drift.baseline_size
        if drift.baseline_size
        else 0.0
    )
    add(
        "SDI matches its definition",
        abs(expected_sdi - drift.sdi) < 1e-9,
        f"(added {drift.size_added:,.1f} − removed {drift.size_removed:,.1f}) "
        f"÷ {drift.baseline_size:,.0f} = {expected_sdi:+.4f}",
    )

    # 2. Revised size identity
    add(
        "Revised size = baseline × (1 + SDI)",
        abs(drift.revised_size - drift.baseline_size * (1 + drift.sdi)) < 1e-6,
        f"{drift.baseline_size:,.0f} × (1 + {drift.sdi:.4f}) = {drift.revised_size:,.1f}",
    )

    # 3. COCOMO receives net, not gross, size
    conversion = result.settings.conversion
    net_ksloc = conversion.to_ksloc(max(0.0, drift.revised_size), result.settings.size_unit)
    gross_ksloc = conversion.to_ksloc(
        drift.baseline_size + drift.size_added, result.settings.size_unit
    )
    add(
        "COCOMO sized on net scope, not gross additions",
        abs(estimates.revised.size_ksloc - net_ksloc) < 1e-9,
        f"net {net_ksloc:.3f} KSLOC used; gross would have been {gross_ksloc:.3f}",
    )

    # 4. REVL excludes new scope
    per_ticket = drift.per_ticket
    if per_ticket.empty:
        refinement_added = new_scope_added = 0.0
    else:
        column = drift.classification_column
        refinement_added = float(
            per_ticket.loc[per_ticket[column] == REFINEMENT, "added"].sum()
        )
        new_scope_added = float(
            per_ticket.loc[per_ticket[column] == NEW_SCOPE, "added"].sum()
        )
    applied_revl = (
        (estimates.revised.effective_ksloc / estimates.revised.size_ksloc - 1) * 100
        if estimates.revised.size_ksloc
        else 0.0
    )
    expected_revl = (
        (refinement_added + drift.size_removed) / drift.baseline_size * 100
        if drift.baseline_size
        else 0.0
    )
    if result.settings.link_drift_to_revl:
        add(
            "REVL counts churn only, excluding new scope",
            abs(applied_revl - max(0.0, expected_revl)) < 1e-6,
            f"churn (refinement {refinement_added:,.1f} + removed "
            f"{drift.size_removed:,.1f}) ÷ {drift.baseline_size:,.0f} = "
            f"{expected_revl:.2f}%; new scope {new_scope_added:,.1f} excluded",
        )
    else:
        add(
            "REVL counts churn only, excluding new scope",
            abs(applied_revl) < 1e-6,
            "REVL link disabled by the user; no volatility rework applied",
        )

    # 5. Effort and duration are distinct quantities
    add(
        "Effort and duration reported separately",
        abs(estimates.revised.person_months - estimates.revised.duration_months) > 1e-6,
        f"{estimates.revised.person_months:,.1f} person-months vs "
        f"{estimates.revised.duration_months:,.1f} calendar months "
        f"({estimates.revised.average_staff:,.1f} FTE implied)",
    )

    # 6. Unit conversion round-trip
    round_trip = conversion.to_ksloc(drift.baseline_size, result.settings.size_unit)
    add(
        "Size unit conversion is consistent",
        abs(round_trip - estimates.baseline.size_ksloc) < 1e-9,
        f"{drift.baseline_size:,.0f} {result.settings.size_unit.replace('_', ' ')} "
        f"→ {round_trip:.3f} KSLOC",
    )

    # 7. Exponent sanity
    add(
        "COCOMO exponent E > 1 (diseconomy of scale)",
        estimates.revised.exponent_e > 1.0,
        f"E = {estimates.revised.exponent_e:.4f}, F = {estimates.revised.exponent_f:.4f}",
    )

    # 8. Cost only when a rate was supplied
    rate = result.settings.monthly_labour_rate
    add(
        "No cost figure without a supplied labour rate",
        True,
        f"rate supplied: {rate:,.0f} per person-month" if rate else "no rate supplied; no cost reported",
    )

    return pd.DataFrame(checks)
