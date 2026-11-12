"""Tests for input loading/validation and for the evaluation metrics."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from driftguard import data, evaluation
from driftguard.config import CLASSES, IN_SCOPE, NEW_SCOPE, REFINEMENT, REMOVAL
from driftguard.data import DataValidationError


def csv(text: str) -> io.StringIO:
    return io.StringIO(text.strip() + "\n")


# --------------------------------------------------------------------------
# Baseline loading
# --------------------------------------------------------------------------
def test_baseline_csv_loads_and_builds_match_text():
    frame, report = data.load_baseline_csv(
        csv(
            "requirement_id,title,description,size\n"
            "R-01,Login,Students can sign in with a password,8"
        )
    )
    assert report.rows == 1
    assert frame.loc[0, "text"].startswith("Login.")
    assert frame.loc[0, "size"] == 8.0


def test_baseline_accepts_common_column_aliases():
    frame, _ = data.load_baseline_csv(
        csv("Req ID,Requirement Text,Story Points\nR-01,Students can sign in,5")
    )
    assert list(frame["requirement_id"]) == ["R-01"]
    assert frame.loc[0, "size"] == 5.0


def test_baseline_missing_required_column_names_it():
    with pytest.raises(DataValidationError, match="description"):
        data.load_baseline_csv(csv("requirement_id,size\nR-01,5"))


def test_baseline_without_sizes_warns_about_the_assumption():
    _, report = data.load_baseline_csv(
        csv("requirement_id,description\nR-01,Students can sign in"),
        default_size=5.0,
    )
    assert any("default size" in w for w in report.warnings)


def test_baseline_duplicate_ids_are_dropped_with_a_warning():
    frame, report = data.load_baseline_csv(
        csv(
            "requirement_id,description\n"
            "R-01,Students can sign in\n"
            "R-01,Students can sign out"
        )
    )
    assert len(frame) == 1
    assert any("duplicate" in w for w in report.warnings)


def test_baseline_negative_size_rejected():
    with pytest.raises(DataValidationError, match="negative size"):
        data.load_baseline_csv(csv("requirement_id,description,size\nR-01,Sign in,-3"))


def test_empty_file_rejected_with_guidance():
    with pytest.raises(DataValidationError, match="empty"):
        data.load_baseline_csv(io.StringIO(""))


def test_header_only_file_rejected():
    with pytest.raises(DataValidationError, match="no data rows"):
        data.load_baseline_csv(io.StringIO("requirement_id,description\n"))


# --------------------------------------------------------------------------
# Baseline TXT
# --------------------------------------------------------------------------
def test_txt_baseline_one_requirement_per_line():
    frame, report = data.load_baseline_txt(
        io.StringIO("Students can sign in.\n\nStudents can enrol in a course.\n")
    )
    assert len(frame) == 2
    assert list(frame["requirement_id"]) == ["R-001", "R-002"]
    assert any("carries no sizes" in w for w in report.warnings)


def test_txt_baseline_reuses_explicit_identifiers():
    frame, _ = data.load_baseline_txt(
        io.StringIO("R-07: Prerequisites are validated\nR-08: Fees are calculated\n")
    )
    assert list(frame["requirement_id"]) == ["R-07", "R-08"]
    assert frame.loc[0, "description"] == "Prerequisites are validated"


def test_txt_baseline_strips_bullet_markers():
    frame, _ = data.load_baseline_txt(
        io.StringIO("- Students can sign in\n2. Students can enrol\n")
    )
    assert frame.loc[0, "description"] == "Students can sign in"
    assert frame.loc[1, "description"] == "Students can enrol"


def test_blank_txt_rejected():
    with pytest.raises(DataValidationError, match="no non-blank lines"):
        data.load_baseline_txt(io.StringIO("\n\n   \n"))


def test_load_baseline_dispatches_on_extension(tmp_path):
    path = tmp_path / "reqs.txt"
    path.write_text("Students can sign in\n", encoding="utf-8")
    frame, _ = data.load_baseline(path)
    assert len(frame) == 1


# --------------------------------------------------------------------------
# Ticket loading
# --------------------------------------------------------------------------
def test_tickets_load_with_dates_parsed():
    frame, _ = data.load_tickets(
        csv(
            "ticket_id,description,size,created_date\n"
            "T-1,Build the sign in page,3,2025-01-08"
        )
    )
    assert pd.api.types.is_datetime64_any_dtype(frame["created_date"])


def test_unparseable_dates_warn_rather_than_fail():
    _, report = data.load_tickets(
        csv(
            "ticket_id,description,size,created_date\n"
            "T-1,Build the sign in page,3,not a date"
        )
    )
    assert any("unrecognised date" in w for w in report.warnings)


def test_missing_date_column_warns_about_the_timeline():
    _, report = data.load_tickets(csv("ticket_id,description,size\nT-1,Sign in,3"))
    assert any("over time" in w for w in report.warnings)


def test_unknown_true_class_label_rejected():
    with pytest.raises(DataValidationError, match="does not recognise"):
        data.load_tickets(
            csv("ticket_id,description,true_class\nT-1,Sign in,Scope Creep")
        )


def test_all_blank_descriptions_rejected():
    with pytest.raises(DataValidationError, match="empty description"):
        data.load_tickets(csv('ticket_id,description\nT-1," "'))


# --------------------------------------------------------------------------
# Bundled sample data
# --------------------------------------------------------------------------
def test_sample_data_loads_and_is_internally_consistent():
    baseline = data.sample_baseline()
    tickets = data.sample_tickets()
    labelled = data.sample_labelled_tickets()

    assert len(baseline) == 18
    assert len(tickets) == 45
    assert len(labelled) == 45
    # The unlabelled feed and the labelled test set describe the same tickets.
    assert list(tickets["ticket_id"]) == list(labelled["ticket_id"])
    assert "true_class" not in tickets.columns
    assert set(labelled["true_class"]) == set(CLASSES)


def test_sample_baseline_sizes_are_positive():
    baseline = data.sample_baseline()
    assert (baseline["size"] > 0).all()
    assert baseline["size"].sum() == 157


# --------------------------------------------------------------------------
# Evaluation metrics
# --------------------------------------------------------------------------
def frame(true: list[str], predicted: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"true_class": true, "predicted_class": predicted})


def test_perfect_predictions_score_one():
    result = evaluation.evaluate(
        frame([IN_SCOPE, NEW_SCOPE, REMOVAL, REFINEMENT] * 2, [IN_SCOPE, NEW_SCOPE, REMOVAL, REFINEMENT] * 2)
    )
    assert result.accuracy == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)


def test_metrics_match_hand_calculation():
    # 3 of 4 correct; New scope predicted twice, right once -> precision 0.5
    result = evaluation.evaluate(
        frame(
            [IN_SCOPE, IN_SCOPE, NEW_SCOPE, REMOVAL],
            [IN_SCOPE, NEW_SCOPE, NEW_SCOPE, REMOVAL],
        )
    )
    assert result.accuracy == pytest.approx(0.75)
    by_class = result.per_class.set_index("class")
    assert by_class.loc[NEW_SCOPE, "precision"] == pytest.approx(0.5)
    assert by_class.loc[NEW_SCOPE, "recall"] == pytest.approx(1.0)
    assert by_class.loc[IN_SCOPE, "recall"] == pytest.approx(0.5)
    assert by_class.loc[IN_SCOPE, "precision"] == pytest.approx(1.0)


def test_confusion_matrix_orientation_is_true_by_predicted():
    result = evaluation.evaluate(frame([IN_SCOPE], [NEW_SCOPE]))
    assert result.confusion.loc[IN_SCOPE, NEW_SCOPE] == 1
    assert result.confusion.loc[NEW_SCOPE, IN_SCOPE] == 0
    assert result.confusion.index.name == "Actual"
    assert result.confusion.columns.name == "Predicted"


def test_confusion_matrix_covers_all_four_classes_even_when_unseen():
    result = evaluation.evaluate(frame([IN_SCOPE], [IN_SCOPE]))
    assert list(result.confusion.index) == list(CLASSES)


def test_macro_f1_punishes_ignoring_a_rare_class():
    """Why macro-F1 is the headline figure rather than accuracy."""
    true = [IN_SCOPE] * 9 + [REMOVAL]
    predicted = [IN_SCOPE] * 10
    result = evaluation.evaluate(frame(true, predicted))
    assert result.accuracy == pytest.approx(0.9)
    assert result.macro_f1 < 0.3
    assert result.baseline_majority_accuracy == pytest.approx(0.9)


def test_errors_frame_holds_only_mistakes():
    result = evaluation.evaluate(
        frame([IN_SCOPE, NEW_SCOPE], [IN_SCOPE, REMOVAL])
    )
    assert len(result.errors) == 1
    assert result.errors.iloc[0]["true_class"] == NEW_SCOPE


def test_unlabelled_rows_are_excluded_not_counted_wrong():
    frame_with_gap = pd.DataFrame(
        {
            "true_class": [IN_SCOPE, None],
            "predicted_class": [IN_SCOPE, NEW_SCOPE],
        }
    )
    result = evaluation.evaluate(frame_with_gap)
    assert result.n_samples == 1
    assert result.accuracy == pytest.approx(1.0)


def test_no_labels_at_all_raises_a_useful_message():
    with pytest.raises(ValueError, match="No manually labelled tickets"):
        evaluation.evaluate(
            pd.DataFrame({"true_class": [None], "predicted_class": [IN_SCOPE]})
        )


def test_stray_label_rejected_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="unrecognised class"):
        evaluation.evaluate(frame(["Creep"], [IN_SCOPE]))


def test_missing_column_message_points_at_the_fix():
    with pytest.raises(ValueError, match="labelled test set"):
        evaluation.evaluate(pd.DataFrame({"predicted_class": [IN_SCOPE]}))


# --------------------------------------------------------------------------
# Drift sensitivity: the metric that actually matters to a project manager
# --------------------------------------------------------------------------
def test_drift_sensitivity_is_zero_when_predictions_are_perfect():
    tickets = pd.DataFrame(
        {
            "ticket_id": ["T-1", "T-2"],
            "size": [10.0, 5.0],
            "true_class": [NEW_SCOPE, REMOVAL],
            "predicted_class": [NEW_SCOPE, REMOVAL],
        }
    )
    result = evaluation.drift_sensitivity(tickets, baseline_size=100.0)
    assert result["absolute_error"] == pytest.approx(0.0)
    assert result["predicted_sdi"] == pytest.approx(0.05)


def test_drift_sensitivity_reports_signed_over_estimation():
    tickets = pd.DataFrame(
        {
            "ticket_id": ["T-1"],
            "size": [20.0],
            "true_class": [IN_SCOPE],
            "predicted_class": [NEW_SCOPE],
        }
    )
    result = evaluation.drift_sensitivity(tickets, baseline_size=100.0)
    assert result["signed_error"] == pytest.approx(0.20)


def test_classifier_comparison_table():
    results = [
        evaluation.evaluate(frame([IN_SCOPE], [IN_SCOPE]), classifier_name="a"),
        evaluation.evaluate(frame([IN_SCOPE], [NEW_SCOPE]), classifier_name="b"),
    ]
    table = evaluation.compare_classifiers(results)
    assert list(table["classifier"]) == ["a", "b"]
    assert table.loc[0, "accuracy"] == pytest.approx(1.0)


def test_empty_comparison_returns_an_empty_table():
    assert evaluation.compare_classifiers([]).empty


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
def test_cross_validation_reports_out_of_sample_scores():
    baseline = data.sample_baseline()
    labelled = data.sample_labelled_tickets()
    from driftguard.matching import RequirementMatcher

    matched = RequirementMatcher().fit(baseline, labelled).match_tickets(labelled)
    scores = evaluation.cross_validated_scores(matched, folds=5)
    # Removal has only 4 examples, so 5 folds are not possible.
    assert scores.attrs["folds_used"] == 4
    assert len(scores) == 4
    assert 0.0 <= scores.attrs["mean_accuracy"] <= 1.0
    # Out-of-sample must be materially below the in-sample fit; if this ever
    # stops holding, the honesty claim in the README needs re-checking.
    assert scores.attrs["mean_accuracy"] < 0.95


def test_cross_validation_refuses_when_a_class_is_too_rare():
    tiny = pd.DataFrame(
        {
            "text": ["a b c", "d e f", "g h i"],
            "similarity": [0.1, 0.2, 0.3],
            "true_class": [IN_SCOPE, IN_SCOPE, REMOVAL],
        }
    )
    with pytest.raises(ValueError, match="at least two examples"):
        evaluation.cross_validated_scores(tiny)
