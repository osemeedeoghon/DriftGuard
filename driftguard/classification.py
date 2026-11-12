"""Classification of tickets into In-scope, Refinement, New scope and Removal.

Two classifiers are provided, both local and both free to run:

``RuleClassifier``
    Similarity thresholds plus a small, readable keyword lexicon. Every
    decision carries a rationale naming the threshold and cues that produced
    it. This is the default, because a scope-creep argument with a project
    sponsor is won with evidence, not with a probability.

``MLClassifier``
    A scikit-learn logistic regression over TF-IDF features plus the
    similarity score, trained on manually labelled tickets. It is the
    "learned baseline" the evaluation page measures the rule classifier
    against. It needs labelled data and, on a typical single-project ticket
    set, it has very little of it -- see the limitations in the README.

Neither classifier calls out to an external service.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    CLASSES,
    IN_SCOPE,
    MatchingThresholds,
    NEW_SCOPE,
    CONTEXTUAL_REMOVAL_CUES,
    NEW_SCOPE_CUES,
    REFINEMENT,
    REFINEMENT_CUES,
    REMOVAL,
    SCOPE_CONTEXT_NOUNS,
    STRONG_REMOVAL_CUES,
    TEXT_LANGUAGE,
)


@dataclass(frozen=True)
class Decision:
    """One classification, with the reasoning that produced it."""

    predicted_class: str
    confidence: float
    rationale: str


def find_cues(text: str, cues: Sequence[str]) -> list[str]:
    """Cue phrases present in ``text``, matched on word boundaries.

    Word-boundary matching keeps "update" from firing inside "updated" being
    fine, while stopping "add" from matching "address" -- a real failure mode
    for a naive substring search over ticket text.
    """
    lowered = text.lower()
    found = []
    for cue in cues:
        pattern = r"\b" + re.escape(cue.lower()) + r"\w{0,3}\b"
        if re.search(pattern, lowered):
            found.append(cue)
    return found


class Classifier(Protocol):  # pragma: no cover - structural type only
    name: str

    def classify(self, matched_tickets: pd.DataFrame) -> pd.DataFrame: ...


# --------------------------------------------------------------------------
# Rule-based, fully explainable
# --------------------------------------------------------------------------
def detect_removal(text: str) -> list[str]:
    """Scope-withdrawal evidence in ``text``, or an empty list.

    An unambiguous multi-word phrase ("de-scoped", "no longer required") is
    enough on its own. A generic verb ("remove", "withdraw", "cancel") counts
    only when a scope noun appears in the same ticket, because those verbs are
    also ordinary domain vocabulary -- an administrator withdrawing a course
    section is in-scope work, not de-scoping.
    """
    strong = find_cues(text, STRONG_REMOVAL_CUES)
    if strong:
        return strong
    generic = find_cues(text, CONTEXTUAL_REMOVAL_CUES)
    if not generic:
        return []
    context = find_cues(text, SCOPE_CONTEXT_NOUNS)
    if not context:
        return []
    return generic + context


class RuleClassifier:
    """Similarity thresholds plus a visible keyword lexicon.

    The division of labour matters, and it is deliberate:

    * **Similarity** answers "does this ticket concern the agreed baseline at
      all?" It is good at that, because an unrelated ticket shares almost no
      distinctive vocabulary with any requirement.
    * **Keyword cues** answer "does it implement, change, or withdraw that
      requirement?" Similarity cannot answer this -- a ticket implementing a
      requirement and a ticket rewriting it look nearly identical to a
      bag-of-words model.

    Decision order (first rule that fires wins):

    1. Scope-withdrawal evidence            -> Removal
    2. similarity < weak_match              -> New scope
    3. refinement cue present               -> Refinement
    4. new-scope cue present                -> New scope
    5. otherwise                            -> In-scope
       (flagged for review when similarity is below ``strong_match``)
    """

    name = "Rule-based (similarity + keyword cues)"

    def __init__(self, thresholds: MatchingThresholds | None = None) -> None:
        self.thresholds = thresholds or MatchingThresholds()

    def decide(self, text: str, similarity: float) -> Decision:
        strong = self.thresholds.strong_match
        weak = self.thresholds.weak_match

        removal_cues = detect_removal(text)
        refinement_cues = find_cues(text, REFINEMENT_CUES)
        new_cues = find_cues(text, NEW_SCOPE_CUES)

        if removal_cues:
            return Decision(
                REMOVAL,
                confidence=min(1.0, 0.6 + 0.1 * len(removal_cues)),
                rationale=(
                    "Classified as Removal: the ticket uses scope-withdrawal "
                    f"language ({', '.join(removal_cues)}). This overrides the "
                    "similarity rules, because a ticket that withdraws a "
                    "requirement necessarily reads much like the requirement "
                    f"it withdraws (similarity {similarity:.2f})."
                ),
            )

        if similarity < weak:
            return Decision(
                NEW_SCOPE,
                confidence=min(1.0, 0.6 + (weak - similarity)),
                rationale=(
                    f"Classified as New scope: the best baseline match scored "
                    f"only {similarity:.2f}, below the weak-match threshold of "
                    f"{weak:.2f}. No agreed requirement shares meaningful "
                    "vocabulary with this ticket, so the work appears to sit "
                    "outside the baseline."
                ),
            )

        if refinement_cues:
            return Decision(
                REFINEMENT,
                confidence=min(1.0, 0.55 + similarity / 3),
                rationale=(
                    f"Classified as Refinement: similarity {similarity:.2f} "
                    f"(at or above the weak-match threshold of {weak:.2f}) "
                    "ties this ticket to an agreed requirement, and "
                    f"modification language ({', '.join(refinement_cues)}) "
                    "shows it alters that requirement rather than simply "
                    "implementing it."
                ),
            )

        if new_cues:
            return Decision(
                NEW_SCOPE,
                confidence=0.6,
                rationale=(
                    f"Classified as New scope: although similarity "
                    f"{similarity:.2f} shows shared vocabulary with the "
                    "baseline, the ticket signals unplanned work "
                    f"({', '.join(new_cues)}), which outweighs the lexical "
                    "overlap."
                ),
            )

        if similarity >= strong:
            return Decision(
                IN_SCOPE,
                confidence=min(1.0, 0.5 + similarity / 2),
                rationale=(
                    f"Classified as In-scope: similarity {similarity:.2f} is "
                    f"at or above the strong-match threshold ({strong:.2f}), "
                    "and the ticket contains no modification, new-work or "
                    "de-scoping language. It reads as a direct implementation "
                    "of the matched requirement."
                ),
            )

        return Decision(
            IN_SCOPE,
            confidence=0.45,
            rationale=(
                f"Classified as In-scope, but with low confidence. Similarity "
                f"{similarity:.2f} sits in the uncertain band between "
                f"{weak:.2f} and {strong:.2f}: the ticket is related to the "
                "matched requirement but does not clearly restate it, and no "
                "cue words settle the question. This case warrants human "
                "review."
            ),
        )

    def classify(self, matched_tickets: pd.DataFrame) -> pd.DataFrame:
        _require_match_columns(matched_tickets)
        if matched_tickets.empty:
            return _empty_classified(matched_tickets)

        decisions = [
            self.decide(str(text), float(sim))
            for text, sim in zip(
                matched_tickets["text"].fillna(""), matched_tickets["similarity"]
            )
        ]
        result = matched_tickets.copy()
        result["predicted_class"] = [d.predicted_class for d in decisions]
        result["confidence"] = [d.confidence for d in decisions]
        result["rationale"] = [d.rationale for d in decisions]
        result["classifier"] = self.name
        return result


# --------------------------------------------------------------------------
# Learned baseline
# --------------------------------------------------------------------------
class MLClassifier:
    """Logistic regression over TF-IDF text features plus the similarity score.

    Deliberately simple: with a single project's worth of tickets there is not
    enough labelled data to justify anything larger, and a model that cannot
    be inspected would undercut the point of the tool. ``class_weight`` is
    balanced because Removal is always the minority class.
    """

    name = "Logistic regression (TF-IDF + similarity)"

    def __init__(self, *, random_state: int = 42, max_iter: int = 2000) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(
                        [
                            (
                                "text",
                                TfidfVectorizer(
                                    stop_words=TEXT_LANGUAGE,
                                    ngram_range=(1, 2),
                                    sublinear_tf=True,
                                    strip_accents="unicode",
                                ),
                                "text",
                            ),
                            ("similarity", StandardScaler(), ["similarity"]),
                        ]
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=max_iter,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self._fitted = False
        self._classes: list[str] = []

    def fit(self, labelled: pd.DataFrame, *, label_column: str = "true_class") -> "MLClassifier":
        _require_match_columns(labelled)
        if label_column not in labelled.columns:
            raise ValueError(
                f"Training data has no {label_column!r} column. The learned "
                "classifier needs manually labelled tickets."
            )
        training = labelled.dropna(subset=[label_column])
        if training.empty:
            raise ValueError("No labelled rows available to train on.")
        if training[label_column].nunique() < 2:
            raise ValueError(
                "Training data contains only one class. At least two distinct "
                "labels are needed to fit a classifier."
            )

        features = training[["text", "similarity"]].copy()
        features["text"] = features["text"].fillna("").astype(str)
        self.pipeline.fit(features, training[label_column].astype(str))
        self._classes = list(self.pipeline.named_steps["model"].classes_)
        self._fitted = True
        return self

    def classify(self, matched_tickets: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("MLClassifier.fit() must be called before classify().")
        _require_match_columns(matched_tickets)
        if matched_tickets.empty:
            return _empty_classified(matched_tickets)

        features = matched_tickets[["text", "similarity"]].copy()
        features["text"] = features["text"].fillna("").astype(str)
        probabilities = self.pipeline.predict_proba(features)
        predictions = [self._classes[i] for i in np.argmax(probabilities, axis=1)]
        confidences = probabilities.max(axis=1)

        rationales = []
        for row, prediction in zip(probabilities, predictions):
            ranked = sorted(
                zip(self._classes, row), key=lambda pair: pair[1], reverse=True
            )
            spread = ", ".join(f"{name} {prob:.0%}" for name, prob in ranked)
            rationales.append(
                f"Logistic regression over TF-IDF terms and the similarity "
                f"score assigned {prediction}. Class probabilities: {spread}. "
                "This model reports correlation in the training vocabulary, "
                "not a reasoned judgement about scope."
            )

        result = matched_tickets.copy()
        result["predicted_class"] = predictions
        result["confidence"] = confidences
        result["rationale"] = rationales
        result["classifier"] = self.name
        return result


# --------------------------------------------------------------------------
# Human override
# --------------------------------------------------------------------------
def apply_overrides(
    classified: pd.DataFrame,
    overrides: dict[str, str] | None = None,
    size_overrides: dict[str, float] | None = None,
    *,
    id_column: str = "ticket_id",
) -> pd.DataFrame:
    """Fold reviewer corrections into ``final_class`` and ``final_size``.

    The predicted values are never overwritten, so the dashboard can always
    show what the model said alongside what the reviewer decided -- which is
    also what makes the evaluation page meaningful.
    """
    overrides = overrides or {}
    size_overrides = size_overrides or {}

    result = classified.copy()
    if result.empty:
        result["final_class"] = []
        result["final_size"] = []
        result["reviewed"] = []
        return result

    unknown = sorted(set(overrides.values()) - set(CLASSES))
    if unknown:
        raise ValueError(
            f"Override uses unrecognised class label(s): {', '.join(unknown)}. "
            f"Allowed labels: {', '.join(CLASSES)}."
        )

    ids = result[id_column].astype(str)
    unknown_ids = sorted((set(overrides) | set(size_overrides)) - set(ids))
    if unknown_ids:
        raise ValueError(
            "Override refers to ticket id(s) not present in the data: "
            + ", ".join(unknown_ids[:5])
            + ("..." if len(unknown_ids) > 5 else "")
        )

    result["final_class"] = [
        overrides.get(ticket_id, predicted)
        for ticket_id, predicted in zip(ids, result["predicted_class"])
    ]
    base_size = result["size"] if "size" in result.columns else pd.Series(0.0, index=result.index)
    result["final_size"] = [
        float(size_overrides.get(ticket_id, size))
        for ticket_id, size in zip(ids, base_size)
    ]
    if (result["final_size"] < 0).any():
        raise ValueError("Confirmed sizes must be zero or positive.")

    result["reviewed"] = [
        ticket_id in overrides or ticket_id in size_overrides for ticket_id in ids
    ]
    result["class_changed_by_reviewer"] = result["final_class"] != result["predicted_class"]
    return result


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _require_match_columns(frame: pd.DataFrame) -> None:
    for column in ("text", "similarity"):
        if column not in frame.columns:
            raise ValueError(
                f"Tickets must be matched before classification: missing "
                f"column {column!r}. Call RequirementMatcher.match_tickets() "
                "first."
            )


def _empty_classified(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("predicted_class", "confidence", "rationale", "classifier"):
        result[column] = []
    return result


def classify_project(
    baseline: pd.DataFrame,
    tickets: pd.DataFrame,
    *,
    thresholds: MatchingThresholds | None = None,
) -> pd.DataFrame:
    """Convenience path: match, then classify with the rule classifier."""
    from .matching import RequirementMatcher, attach_requirement_details

    matcher = RequirementMatcher().fit(baseline, tickets)
    matched = matcher.match_tickets(tickets)
    matched = attach_requirement_details(matched, baseline)
    return RuleClassifier(thresholds).classify(matched)
