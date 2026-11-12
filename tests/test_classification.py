"""Tests for matching, classification and reviewer overrides."""

from __future__ import annotations

import pandas as pd
import pytest

from driftguard import data, matching
from driftguard.classification import (
    MLClassifier,
    RuleClassifier,
    apply_overrides,
    detect_removal,
    find_cues,
)
from driftguard.config import (
    IN_SCOPE,
    MatchingThresholds,
    NEW_SCOPE,
    REFINEMENT,
    REMOVAL,
)


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    return data.sample_baseline()


@pytest.fixture(scope="module")
def labelled() -> pd.DataFrame:
    return data.sample_labelled_tickets()


@pytest.fixture(scope="module")
def matched(baseline, labelled) -> pd.DataFrame:
    matcher = matching.RequirementMatcher().fit(baseline, labelled)
    return matching.attach_requirement_details(
        matcher.match_tickets(labelled), baseline
    )


# --------------------------------------------------------------------------
# Cue detection
# --------------------------------------------------------------------------
def test_cue_matching_respects_word_boundaries():
    # "add" must not fire inside "address"
    assert find_cues("Capture the student address", ["add"]) == []
    assert find_cues("Add a filter", ["add"]) == ["add"]


def test_cue_matching_allows_short_inflections():
    assert find_cues("The report was revised", ["revise"]) == ["revise"]
    assert find_cues("The module was removed", ["remove"]) == ["remove"]


def test_cue_matching_does_not_stem_across_a_changed_root():
    """"removal" is not reachable from "remove" by suffixing, which is why the
    lexicon lists both spellings explicitly."""
    assert find_cues("Removal of the module", ["remove"]) == []
    assert find_cues("Removal of the module", ["removal"]) == ["removal"]


def test_strong_removal_phrase_fires_alone():
    assert detect_removal("This requirement is no longer required.")


def test_generic_withdrawal_verb_needs_scope_context():
    """The regression that motivated the two-tier lexicon."""
    domain_usage = (
        "Administrators can create, edit, publish and withdraw course sections."
    )
    assert detect_removal(domain_usage) == []

    scope_usage = "The faculty roster view is withdrawn from the project scope."
    assert detect_removal(scope_usage)


def test_enrolment_withdrawal_vocabulary_is_not_descoping():
    text = "Email the student whenever an enrolment is confirmed or withdrawn."
    assert detect_removal(text) == []


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def test_matcher_rejects_empty_baseline():
    with pytest.raises(ValueError, match="empty baseline"):
        matching.RequirementMatcher().fit(pd.DataFrame({"text": [], "requirement_id": []}))


def test_matcher_must_be_fitted_before_use(baseline, labelled):
    with pytest.raises(RuntimeError, match="fit"):
        matching.RequirementMatcher().match_tickets(labelled)


def test_similarity_is_bounded_and_populated(matched):
    assert matched["similarity"].between(0.0, 1.0).all()
    assert matched["matched_requirement_id"].notna().all()


def test_match_evidence_names_shared_terms(matched):
    row = matched.loc[matched["ticket_id"] == "T-105"].iloc[0]
    assert row["matched_requirement_id"] == "R-03"
    assert "enrolment" in " ".join(row["shared_terms"]).lower()
    assert "similarity" in row["match_evidence"].lower()


def test_in_scope_tickets_outrank_new_scope_tickets_on_similarity(matched):
    in_scope = matched.loc[matched["true_class"] == IN_SCOPE, "similarity"]
    new_scope = matched.loc[matched["true_class"] == NEW_SCOPE, "similarity"]
    assert in_scope.median() > new_scope.median()


def test_alternatives_are_ranked_descending(matched):
    for alternatives in matched["alternative_matches"]:
        scores = [score for _, score in alternatives]
        assert scores == sorted(scores, reverse=True)


def test_match_margin_is_never_negative(matched):
    assert (matched["match_margin"] >= -1e-9).all()


# --------------------------------------------------------------------------
# Rule classifier
# --------------------------------------------------------------------------
def test_rule_order_removal_beats_high_similarity():
    decision = RuleClassifier().decide(
        "This requirement is de-scoped from the project.", similarity=0.95
    )
    assert decision.predicted_class == REMOVAL


def test_low_similarity_means_new_scope():
    decision = RuleClassifier().decide("Build a holiday booking engine", 0.02)
    assert decision.predicted_class == NEW_SCOPE
    assert "below the weak-match threshold" in decision.rationale


def test_refinement_cue_beats_plain_implementation():
    decision = RuleClassifier().decide("Revise the password policy", 0.60)
    assert decision.predicted_class == REFINEMENT


def test_high_similarity_without_cues_is_in_scope():
    decision = RuleClassifier().decide("Implement the enrolment seat allocation", 0.70)
    assert decision.predicted_class == IN_SCOPE
    assert decision.confidence > 0.5


def test_uncertain_band_is_in_scope_but_flagged_for_review():
    thresholds = MatchingThresholds(weak_match=0.2, strong_match=0.5)
    decision = RuleClassifier(thresholds).decide("Implement the roster screen", 0.30)
    assert decision.predicted_class == IN_SCOPE
    assert decision.confidence < 0.5
    assert "human review" in decision.rationale


def test_every_decision_carries_a_rationale(matched):
    classified = RuleClassifier().classify(matched)
    assert classified["rationale"].str.len().min() > 40
    assert classified["predicted_class"].notna().all()


def test_classification_requires_matching_first(labelled):
    with pytest.raises(ValueError, match="matched before classification"):
        RuleClassifier().classify(labelled)


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError, match="weak_match <= strong_match"):
        MatchingThresholds(weak_match=0.8, strong_match=0.3)


def test_rule_classifier_beats_the_majority_baseline(matched):
    from driftguard import evaluation

    result = evaluation.evaluate(RuleClassifier().classify(matched))
    assert result.accuracy > result.baseline_majority_accuracy
    # Guardrail against a silent regression in the lexicon or thresholds.
    assert result.macro_f1 > 0.85


# --------------------------------------------------------------------------
# Learned classifier
# --------------------------------------------------------------------------
def test_ml_classifier_must_be_fitted(matched):
    with pytest.raises(RuntimeError, match="fit"):
        MLClassifier().classify(matched)


def test_ml_classifier_needs_labels(matched):
    with pytest.raises(ValueError, match="true_class"):
        MLClassifier().fit(matched.drop(columns=["true_class"]))


def test_ml_classifier_rejects_single_class_training(matched):
    single = matched.copy()
    single["true_class"] = IN_SCOPE
    with pytest.raises(ValueError, match="only one class"):
        MLClassifier().fit(single)


def test_ml_classifier_predicts_known_classes(matched):
    model = MLClassifier().fit(matched)
    predicted = model.classify(matched)
    assert set(predicted["predicted_class"]) <= {IN_SCOPE, REFINEMENT, NEW_SCOPE, REMOVAL}
    assert predicted["confidence"].between(0.0, 1.0).all()


# --------------------------------------------------------------------------
# Reviewer overrides
# --------------------------------------------------------------------------
def test_override_changes_final_class_but_preserves_prediction(matched):
    classified = RuleClassifier().classify(matched)
    reviewed = apply_overrides(classified, {"T-301": IN_SCOPE})
    row = reviewed.loc[reviewed["ticket_id"] == "T-301"].iloc[0]
    assert row["final_class"] == IN_SCOPE
    assert row["predicted_class"] == NEW_SCOPE
    assert bool(row["class_changed_by_reviewer"]) is True


def test_unreviewed_tickets_keep_the_prediction(matched):
    classified = RuleClassifier().classify(matched)
    reviewed = apply_overrides(classified, {"T-301": IN_SCOPE})
    untouched = reviewed.loc[reviewed["ticket_id"] != "T-301"]
    assert (untouched["final_class"] == untouched["predicted_class"]).all()
    assert not untouched["reviewed"].any()


def test_size_override_is_applied(matched):
    classified = RuleClassifier().classify(matched)
    reviewed = apply_overrides(classified, size_overrides={"T-301": 34.0})
    row = reviewed.loc[reviewed["ticket_id"] == "T-301"].iloc[0]
    assert row["final_size"] == pytest.approx(34.0)
    assert bool(row["reviewed"]) is True


def test_override_with_unknown_class_rejected(matched):
    classified = RuleClassifier().classify(matched)
    with pytest.raises(ValueError, match="unrecognised class"):
        apply_overrides(classified, {"T-301": "Probably fine"})


def test_override_of_unknown_ticket_rejected(matched):
    classified = RuleClassifier().classify(matched)
    with pytest.raises(ValueError, match="not present"):
        apply_overrides(classified, {"T-999": IN_SCOPE})


def test_negative_confirmed_size_rejected(matched):
    classified = RuleClassifier().classify(matched)
    with pytest.raises(ValueError, match="zero or positive"):
        apply_overrides(classified, size_overrides={"T-301": -5.0})
