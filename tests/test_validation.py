"""Tests for nested validation, the leakage detector and the methodology audit."""

from __future__ import annotations

import pandas as pd
import pytest

from driftguard import data, pipeline, validation
from driftguard.config import MatchingThresholds
from driftguard.pipeline import AnalysisSettings


@pytest.fixture(scope="module")
def matched():
    return pipeline.prepare_training_data(
        data.sample_baseline(), data.sample_labelled_tickets()
    )


@pytest.fixture(scope="module")
def analysis():
    return pipeline.run_analysis(data.sample_baseline(), data.sample_tickets())


@pytest.fixture(scope="module")
def nested(matched):
    return validation.nested_cross_validate(matched)


# --------------------------------------------------------------------------
# Threshold tuning
# --------------------------------------------------------------------------
def test_tune_thresholds_returns_a_valid_pair(matched):
    thresholds, score = validation.tune_thresholds(matched)
    assert isinstance(thresholds, MatchingThresholds)
    assert thresholds.weak_match <= thresholds.strong_match
    assert 0.0 <= score <= 1.0


def test_tuning_never_scores_worse_than_the_shipped_defaults(matched):
    """A grid search that cannot match the shipped settings is broken."""
    _, tuned_score = validation.tune_thresholds(matched)
    shipped_score = validation._score_thresholds(
        matched, MatchingThresholds(), label_column="true_class"
    )
    assert tuned_score >= shipped_score


def test_tuning_rejects_an_impossible_grid(matched):
    with pytest.raises(ValueError, match="No valid threshold pair"):
        validation.tune_thresholds(
            matched, weak_grid=(0.9,), strong_grid=(0.1,)
        )


# --------------------------------------------------------------------------
# Nested cross-validation
# --------------------------------------------------------------------------
def test_nested_cv_runs_one_row_per_outer_fold(nested):
    assert len(nested.outer_folds) == nested.outer_folds["fold"].nunique()
    assert nested.outer_folds["n_test"].sum() == nested.n_samples
    assert set(
        ["fold", "weak_match", "strong_match", "accuracy", "macro_f1"]
    ).issubset(nested.outer_folds.columns)


def test_nested_cv_reduces_folds_for_the_rarest_class(matched):
    """Removal has only 4 examples, so 5 outer folds are impossible."""
    result = validation.nested_cross_validate(matched, outer_folds=5)
    assert len(result.outer_folds) == 4
    assert any("reduced from 5 to 4" in c for c in result.caveats)


def test_nested_cv_requires_matched_tickets():
    frame = pd.DataFrame({"true_class": ["In-scope"], "ticket_id": ["T-1"]})
    with pytest.raises(ValueError, match="must be matched"):
        validation.nested_cross_validate(frame)


def test_nested_cv_rejects_unlabelled_input(matched):
    unlabelled = matched.copy()
    unlabelled["true_class"] = None
    with pytest.raises(ValueError, match="No labelled rows"):
        validation.nested_cross_validate(unlabelled)


def test_nested_cv_refuses_when_a_class_is_too_rare(matched):
    tiny = matched.head(3).copy()
    tiny["true_class"] = ["In-scope", "In-scope", "Removal"]
    with pytest.raises(ValueError, match="at least two examples"):
        validation.nested_cross_validate(tiny)


# --------------------------------------------------------------------------
# The leakage detector — the point of this module
# --------------------------------------------------------------------------
def test_leakage_is_detected_on_the_sample_project(nested):
    """On this dataset the held-out score meets the in-sample score.

    That cannot happen with genuinely held-out data, so the result must be
    flagged rather than presented as a performance figure. If this assertion
    ever fails because the scores separate properly, the caveat wording in the
    dashboard needs revisiting too.
    """
    assert nested.leakage_suspected is True
    assert "Not a valid out-of-sample estimate" in nested.verdict
    assert nested.caveats[0].startswith("LEAKAGE INDICATOR")


def test_leakage_caveat_quantifies_cue_reliance(nested):
    assert not nested.cue_reliance.empty
    total = nested.cue_reliance["Tickets"].sum()
    assert total == nested.n_samples
    # The lexicon row must be marked as leaking.
    leaking = nested.cue_reliance.loc[
        nested.cue_reliance["Decided by"] == "Keyword cue logic",
        "Leaks ground truth?",
    ].iloc[0]
    assert "Yes" in leaking


def test_cue_reliance_rises_as_the_weak_threshold_falls(matched):
    """The mechanism behind the leakage: a lower threshold hands more
    decisions to the hand-written lexicon."""
    from driftguard.classification import RuleClassifier

    high = RuleClassifier(MatchingThresholds(weak_match=0.30, strong_match=0.45))
    low = RuleClassifier(MatchingThresholds(weak_match=0.10, strong_match=0.35))
    decided_by_similarity_high = (matched["similarity"] < 0.30).sum()
    decided_by_similarity_low = (matched["similarity"] < 0.10).sum()
    assert decided_by_similarity_low < decided_by_similarity_high
    # Both still classify everything.
    assert len(high.classify(matched)) == len(low.classify(matched)) == len(matched)


def test_verdict_is_never_a_bare_number(nested):
    assert len(nested.verdict) > 40
    assert nested.summary_line()


def test_caveats_always_mention_the_irreducible_lexicon_leakage(nested):
    joined = " ".join(nested.caveats)
    assert "lexicon" in joined
    assert "freshly labelled corpus" in joined or "another project" in joined
    assert "transductive" in joined


# --------------------------------------------------------------------------
# Learning curve
# --------------------------------------------------------------------------
def test_learning_curve_covers_every_requested_fraction(matched):
    curve = validation.learning_curve(matched, fractions=(0.5, 1.0), repeats=3)
    assert set(curve["fraction"].unique()) == {0.5, 1.0}
    assert curve["macro_f1"].between(0.0, 1.0).all()


def test_learning_curve_summary_aggregates_repeats(matched):
    curve = validation.learning_curve(matched, fractions=(0.5, 1.0), repeats=3)
    summary = validation.learning_curve_summary(curve)
    assert len(summary) == 2
    assert (summary["runs"] > 0).all()
    assert "macro_f1_std" in summary.columns


def test_learning_curve_needs_labels(matched):
    unlabelled = matched.copy()
    unlabelled["true_class"] = None
    with pytest.raises(ValueError, match="No labelled rows"):
        validation.learning_curve(unlabelled)


# --------------------------------------------------------------------------
# Methodology audit
# --------------------------------------------------------------------------
def test_audit_passes_every_check_on_a_clean_analysis(analysis):
    audit = validation.methodology_audit(analysis)
    failures = audit.loc[audit["Result"] != "PASS"]
    assert failures.empty, failures.to_string(index=False)


def test_audit_verifies_the_sdi_definition(analysis):
    audit = validation.methodology_audit(analysis).set_index("Check")
    detail = audit.loc["SDI matches its definition", "Detail"]
    assert "÷" in detail and "added" in detail


def test_audit_verifies_cocomo_uses_net_not_gross_size(analysis):
    audit = validation.methodology_audit(analysis).set_index("Check")
    row = audit.loc["COCOMO sized on net scope, not gross additions"]
    assert row["Result"] == "PASS"
    assert "gross would have been" in row["Detail"]


def test_audit_verifies_revl_excludes_new_scope(analysis):
    audit = validation.methodology_audit(analysis).set_index("Check")
    row = audit.loc["REVL counts churn only, excluding new scope"]
    assert row["Result"] == "PASS"
    assert "excluded" in row["Detail"]


def test_audit_handles_revl_link_disabled():
    result = pipeline.run_analysis(
        data.sample_baseline(),
        data.sample_tickets(),
        AnalysisSettings(link_drift_to_revl=False),
    )
    audit = validation.methodology_audit(result).set_index("Check")
    row = audit.loc["REVL counts churn only, excluding new scope"]
    assert row["Result"] == "PASS"
    assert "disabled" in row["Detail"]


def test_audit_reports_no_cost_without_a_rate(analysis):
    audit = validation.methodology_audit(analysis).set_index("Check")
    assert "no rate supplied" in audit.loc[
        "No cost figure without a supplied labour rate", "Detail"
    ]


def test_audit_reports_the_rate_when_one_is_supplied():
    result = pipeline.run_analysis(
        data.sample_baseline(),
        data.sample_tickets(),
        AnalysisSettings(monthly_labour_rate=11000.0),
    )
    audit = validation.methodology_audit(result).set_index("Check")
    assert "11,000" in audit.loc[
        "No cost figure without a supplied labour rate", "Detail"
    ]


def test_audit_would_catch_a_broken_sdi(analysis):
    """The audit must be capable of failing, or it proves nothing."""
    import copy
    from dataclasses import replace

    broken = copy.copy(analysis)
    broken.drift = replace(analysis.drift, sdi=analysis.drift.sdi + 0.5)
    audit = validation.methodology_audit(broken).set_index("Check")
    assert audit.loc["SDI matches its definition", "Result"] == "FAIL"
