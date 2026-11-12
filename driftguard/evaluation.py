"""Evaluation of classifier predictions against manually labelled tickets.

Reports accuracy, per-class precision and recall, macro-F1 and a confusion
matrix. Macro-F1 is the headline figure rather than accuracy because the four
classes are heavily imbalanced -- Removal is rare in any real project, and a
classifier that never predicts it can still post a flattering accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import CLASSES


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics for one classifier on one labelled set."""

    classifier: str
    n_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: pd.DataFrame        # precision / recall / f1 / support per class
    confusion: pd.DataFrame        # rows = true, columns = predicted
    errors: pd.DataFrame           # the misclassified rows, for inspection

    def summary_line(self) -> str:
        return (
            f"{self.classifier}: accuracy {self.accuracy:.1%}, "
            f"macro-F1 {self.macro_f1:.3f} on {self.n_samples} labelled tickets."
        )

    @property
    def baseline_majority_accuracy(self) -> float:
        """Accuracy of always predicting the most common true class.

        Any classifier that does not beat this number has learned nothing
        useful, so it is reported alongside the real accuracy.
        """
        support = self.per_class["support"]
        if support.sum() == 0:
            return 0.0
        return float(support.max() / support.sum())


def evaluate(
    predictions: pd.DataFrame,
    *,
    true_column: str = "true_class",
    predicted_column: str = "predicted_class",
    classifier_name: str | None = None,
    labels: tuple[str, ...] = CLASSES,
) -> EvaluationResult:
    """Score predictions against manual labels.

    Rows with no manual label are excluded and do not count towards any
    metric. Labels outside the four-class taxonomy are rejected rather than
    silently dropped, because a stray label would quietly shrink the
    denominator of every score.
    """
    for column in (true_column, predicted_column):
        if column not in predictions.columns:
            raise ValueError(
                f"Cannot evaluate: no {column!r} column in the data. Upload a "
                "labelled test set containing a true_class column."
            )

    scored = predictions.dropna(subset=[true_column]).copy()
    scored[true_column] = scored[true_column].astype(str).str.strip()
    scored = scored[scored[true_column] != ""]

    if scored.empty:
        raise ValueError(
            "No manually labelled tickets were found, so classifier quality "
            "cannot be measured. Add a true_class column with at least one "
            "labelled row."
        )

    unknown = sorted(set(scored[true_column]) - set(labels))
    if unknown:
        raise ValueError(
            f"Labelled data contains unrecognised class(es): {', '.join(unknown)}. "
            f"Allowed: {', '.join(labels)}."
        )

    y_true = scored[true_column].to_numpy()
    y_pred = scored[predicted_column].astype(str).to_numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class": list(labels),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )

    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    confusion = pd.DataFrame(
        matrix,
        index=pd.Index(list(labels), name="Actual"),
        columns=pd.Index(list(labels), name="Predicted"),
    )

    errors = scored.loc[scored[true_column] != scored[predicted_column]].copy()

    name = classifier_name
    if name is None:
        name = (
            str(scored["classifier"].iloc[0])
            if "classifier" in scored.columns and not scored.empty
            else "classifier"
        )

    return EvaluationResult(
        classifier=name,
        n_samples=len(scored),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", labels=list(labels), zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, average="weighted", labels=list(labels), zero_division=0)
        ),
        per_class=per_class,
        confusion=confusion,
        errors=errors,
    )


def compare_classifiers(results: list[EvaluationResult]) -> pd.DataFrame:
    """Side-by-side table of several evaluation runs."""
    if not results:
        return pd.DataFrame(
            columns=["classifier", "n", "accuracy", "macro_f1", "weighted_f1", "majority_baseline"]
        )
    return pd.DataFrame(
        [
            {
                "classifier": r.classifier,
                "n": r.n_samples,
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "weighted_f1": r.weighted_f1,
                "majority_baseline": r.baseline_majority_accuracy,
            }
            for r in results
        ]
    )


def drift_sensitivity(
    predictions: pd.DataFrame,
    baseline_size: float,
    *,
    true_column: str = "true_class",
    predicted_column: str = "predicted_class",
    size_column: str = "size",
    refinement_weight: float = 0.5,
) -> dict[str, float]:
    """How far classifier error moves the Scope Drift Index itself.

    Accuracy is not the quantity a project manager cares about -- the question
    is whether the misclassifications change the reported drift. This returns
    the SDI computed from predicted labels against the SDI from true labels.
    """
    from .drift import calculate_drift

    scored = predictions.dropna(subset=[true_column]).copy()
    if scored.empty:
        raise ValueError("No labelled rows available for a drift sensitivity check.")

    predicted_sdi = calculate_drift(
        scored,
        baseline_size,
        classification_column=predicted_column,
        size_column=size_column,
        refinement_weight=refinement_weight,
    ).sdi
    true_sdi = calculate_drift(
        scored,
        baseline_size,
        classification_column=true_column,
        size_column=size_column,
        refinement_weight=refinement_weight,
    ).sdi

    return {
        "predicted_sdi": predicted_sdi,
        "true_sdi": true_sdi,
        "absolute_error": abs(predicted_sdi - true_sdi),
        "signed_error": predicted_sdi - true_sdi,
    }


def cross_validated_scores(
    labelled: pd.DataFrame,
    *,
    folds: int = 5,
    random_state: int = 42,
    label_column: str = "true_class",
) -> pd.DataFrame:
    """Stratified cross-validation of the learned classifier.

    Training and scoring the learned model on the same tickets would report a
    score that says nothing about unseen tickets. This runs stratified k-fold
    instead, and reduces the fold count when a class is too rare to split.
    """
    from sklearn.model_selection import StratifiedKFold

    from .classification import MLClassifier

    data = labelled.dropna(subset=[label_column]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No labelled rows available for cross-validation.")

    counts = data[label_column].value_counts()
    usable_folds = int(min(folds, counts.min()))
    if usable_folds < 2:
        raise ValueError(
            "Cross-validation needs at least two examples of every class. "
            f"The rarest class ('{counts.idxmin()}') has {int(counts.min())}."
        )

    splitter = StratifiedKFold(
        n_splits=usable_folds, shuffle=True, random_state=random_state
    )
    rows = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(data, data[label_column]), start=1
    ):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        model = MLClassifier(random_state=random_state).fit(train, label_column=label_column)
        predicted = model.classify(test)
        result = evaluate(
            predicted,
            true_column=label_column,
            classifier_name=f"fold {fold}",
        )
        rows.append(
            {
                "fold": fold,
                "n_test": result.n_samples,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
            }
        )

    frame = pd.DataFrame(rows)
    frame.attrs["folds_used"] = usable_folds
    frame.attrs["folds_requested"] = folds
    frame.attrs["mean_accuracy"] = float(np.mean(frame["accuracy"]))
    frame.attrs["mean_macro_f1"] = float(np.mean(frame["macro_f1"]))
    return frame
