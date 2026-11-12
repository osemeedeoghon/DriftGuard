"""Model Performance — measured against a manually labelled test set only."""

from __future__ import annotations

import streamlit as st

from driftguard import evaluation, validation
from driftguard.classification import MLClassifier, RuleClassifier
from driftguard.pipeline import prepare_training_data

from .. import charts, components as ui, state, theme


def render(result) -> None:
    ui.page_header(
        "Model performance",
        "Classifier quality measured against manual labels. Where no labels "
        "exist, nothing is claimed.",
    )

    if not state.has_labels():
        ui.empty_state(
            "No labelled test set loaded",
            "Classifier accuracy cannot be measured without ground truth. "
            "Upload a tickets CSV containing a true_class column, or load the "
            "sample project, and this page will populate. Until then no "
            "accuracy figure is shown, because there is nothing to support one.",
            icon="📋",
        )
        return

    baseline = st.session_state.baseline
    labelled = st.session_state.labelled

    with ui.working("Scoring classifiers against the labelled set…"):
        matched = prepare_training_data(baseline, labelled)
        rule_predictions = RuleClassifier(result.settings.thresholds).classify(matched)
        results = [evaluation.evaluate(rule_predictions)]
        ml_error = None
        try:
            ml_predictions = MLClassifier().fit(matched).classify(matched)
            results.append(evaluation.evaluate(ml_predictions))
        except ValueError as exc:
            ml_error = str(exc)

    # ---- Headline --------------------------------------------------------
    primary = results[0]
    ui.metric_row(
        [
            {
                "label": "Macro-F1",
                "value": f"{primary.macro_f1:.3f}",
                "footnote": "headline metric",
                "help_text": "Unweighted mean F1 across the four classes, so a rare class counts as much as a common one.",
            },
            {
                "label": "Accuracy",
                "value": theme.fmt_percent(primary.accuracy),
                "footnote": f"vs {theme.fmt_percent(primary.baseline_majority_accuracy)} majority baseline",
                "delta": theme.fmt_signed(
                    (primary.accuracy - primary.baseline_majority_accuracy) * 100, 1, " pp"
                ),
                "delta_good": primary.accuracy > primary.baseline_majority_accuracy,
            },
            {
                "label": "Labelled tickets",
                "value": theme.fmt_number(primary.n_samples),
                "footnote": "manual ground truth",
            },
            {
                "label": "Misclassified",
                "value": theme.fmt_number(len(primary.errors)),
                "footnote": "listed below",
                "delta_good": None,
            },
        ]
    )

    st.warning(
        "**These figures are in-sample.** The rule classifier's thresholds and "
        "keyword lexicon were tuned on this data, and the learned model is "
        "scored on the same tickets it trained on. The cross-validated figures "
        "lower down are the honest estimate for unseen tickets.",
        icon="⚠️",
    )
    if ml_error:
        st.info(f"Learned classifier not scored: {ml_error}", icon="ℹ️")

    # ---- Comparison ------------------------------------------------------
    ui.section("Classifier comparison", "Same labelled set, same metrics.")
    table = evaluation.compare_classifiers(results).rename(
        columns={
            "classifier": "Classifier",
            "n": "Tickets",
            "accuracy": "Accuracy",
            "macro_f1": "Macro-F1",
            "weighted_f1": "Weighted F1",
            "majority_baseline": "Majority baseline",
        }
    )
    ui.data_table(
        ui.scale_to_percent(table, ["Accuracy", "Majority baseline"]),
        column_config={
            "Accuracy": ui.percent_column(),
            "Majority baseline": ui.percent_column(),
            "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
            "Weighted F1": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "Macro-F1 leads rather than accuracy because the four classes are "
        "imbalanced: Removal is rare, and a model that never predicts it can "
        "still post a flattering accuracy. Anything at or below the majority "
        "baseline has learned nothing."
    )

    # ---- Per classifier detail -------------------------------------------
    chosen_name = st.selectbox("Inspect classifier", [r.classifier for r in results])
    chosen = next(r for r in results if r.classifier == chosen_name)

    left, right = st.columns(2)
    with left:
        ui.chart(
            charts.per_class_metrics(chosen.per_class),
            table=chosen.per_class.round(3),
            key="pf_perclass",
        )
    with right:
        ui.chart(
            charts.confusion_heatmap(chosen.confusion),
            table=chosen.confusion.reset_index(),
            key="pf_confusion",
        )
    st.caption(
        "Rows are the manual label, columns the prediction. The diagonal is "
        "correct; everything off it is an error."
    )

    # ---- Errors ----------------------------------------------------------
    ui.section("Misclassified tickets", "Every error, with the reasoning that produced it.")
    if chosen.errors.empty:
        st.success("No misclassifications on the labelled set.", icon="✅")
    else:
        columns = [
            c
            for c in ("ticket_id", "title", "true_class", "predicted_class", "similarity", "rationale")
            if c in chosen.errors.columns
        ]
        ui.data_table(
            chosen.errors[columns].rename(
                columns={
                    "ticket_id": "Ticket",
                    "title": "Title",
                    "true_class": "Actual",
                    "predicted_class": "Predicted",
                    "similarity": "Similarity",
                    "rationale": "Why the model said that",
                }
            ),
            column_config={"Similarity": st.column_config.NumberColumn(format="%.3f")},
            height=260,
        )

    # ---- Drift sensitivity -----------------------------------------------
    ui.section(
        "Does classifier error change the reported drift?",
        "The metric that actually matters to a project sponsor.",
    )
    sensitivity = evaluation.drift_sensitivity(
        rule_predictions,
        result.baseline_size,
        refinement_weight=result.settings.refinement_weight,
    )
    ui.metric_row(
        [
            {
                "label": "SDI from predicted labels",
                "value": theme.fmt_percent(sensitivity["predicted_sdi"], signed=True),
            },
            {
                "label": "SDI from manual labels",
                "value": theme.fmt_percent(sensitivity["true_sdi"], signed=True),
            },
            {
                "label": "Error in the drift figure",
                "value": theme.fmt_signed(sensitivity["signed_error"] * 100, 1, " pp"),
                "delta_good": abs(sensitivity["signed_error"]) < 0.05,
                "footnote": "positive = overstates growth",
            },
        ]
    )
    st.caption(
        "Accuracy is a proxy. What counts is whether the misclassifications "
        "move the number the decision rests on."
    )

    # ---- Cross-validation -------------------------------------------------
    ui.section(
        "Cross-validated performance",
        "Stratified k-fold on the learned classifier — the honest out-of-sample estimate.",
    )
    try:
        with ui.working("Running stratified cross-validation…"):
            scores = evaluation.cross_validated_scores(matched)
        ui.metric_row(
            [
                {
                    "label": "Mean accuracy (out-of-sample)",
                    "value": theme.fmt_percent(scores.attrs["mean_accuracy"]),
                    "delta": theme.fmt_signed(
                        (scores.attrs["mean_accuracy"] - primary.accuracy) * 100, 1, " pp"
                    ),
                    "delta_good": False,
                    "footnote": "against the in-sample figure above",
                },
                {
                    "label": "Mean macro-F1 (out-of-sample)",
                    "value": f"{scores.attrs['mean_macro_f1']:.3f}",
                    "footnote": f"{scores.attrs['folds_used']} stratified folds",
                },
            ]
        )
        fold_table = scores.rename(
            columns={
                "fold": "Fold",
                "n_test": "Test tickets",
                "accuracy": "Accuracy",
                "macro_f1": "Macro-F1",
            }
        )
        ui.data_table(
            ui.scale_to_percent(fold_table, ["Accuracy"]),
            column_config={
                "Accuracy": ui.percent_column(),
                "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        if scores.attrs["folds_used"] < scores.attrs["folds_requested"]:
            st.caption(
                f"Reduced to {scores.attrs['folds_used']} folds: the rarest "
                "class has too few examples for more."
            )
        st.caption(
            "The gap between these numbers and the in-sample table above is the "
            "cost of fitting a model to a single project's vocabulary."
        )
    except ValueError as exc:
        ui.no_evidence(f"Cross-validation not possible: {exc}")

    _render_nested_validation(matched, result)


def _render_nested_validation(matched, result) -> None:
    """Nested cross-validation of the rule classifier, and what it reveals."""
    ui.section(
        "Nested validation of the rule classifier",
        "The cross-validation above covers the learned model only. The rule "
        "classifier is the default, and its thresholds were tuned on this same "
        "data — so it needs its own, stricter check.",
    )

    st.markdown(
        "Nested cross-validation tunes the similarity thresholds on an inner "
        "split and scores them once on a fold that tuning never saw. That "
        "removes leakage from **threshold tuning**. It cannot remove leakage "
        "from the **keyword lexicon**, which was hand-written after reading "
        "these tickets and cannot be re-derived per fold."
    )

    try:
        with ui.working("Running nested cross-validation…"):
            nested = validation.nested_cross_validate(
                matched, shipped=result.settings.thresholds
            )
    except ValueError as exc:
        ui.no_evidence(f"Nested validation not possible: {exc}")
        return

    ui.metric_row(
        [
            {
                "label": "Held-out macro-F1",
                "value": f"{nested.mean_macro_f1:.3f}",
                "footnote": f"mean of {len(nested.outer_folds)} outer folds (sd {nested.std_macro_f1:.3f})",
            },
            {
                "label": "In-sample macro-F1",
                "value": f"{nested.in_sample_macro_f1:.3f}",
                "footnote": "shipped thresholds, scored on everything",
            },
            {
                "label": "Apparent optimism",
                "value": f"{nested.optimism_macro_f1:+.3f}",
                "delta_good": None,
                "footnote": "in-sample minus held-out",
                "help_text": "Should be positive. A negative value means the held-out data is not really held out.",
            },
            {
                "label": "Verdict",
                "value": "Leakage" if nested.leakage_suspected else "Usable",
                "footnote": "see explanation below",
            },
        ]
    )

    if nested.leakage_suspected:
        st.error(
            "**This is not a valid out-of-sample estimate.** Held-out "
            "performance should be *worse* than training performance; when it "
            "is not, the held-out data is not genuinely held out. The cause "
            "here is structural and is explained in the caveats below.",
            icon="🚩",
        )
    else:
        st.success(nested.verdict, icon="✅")

    left, right = st.columns(2)
    with left:
        ui.chart(
            charts.nested_fold_scores(nested.outer_folds, nested.in_sample_macro_f1),
            table=nested.outer_folds.round(3),
            key="pf_nested",
        )
    with right:
        st.markdown("##### Which signal decided each ticket?")
        ui.data_table(nested.cue_reliance)
        st.caption(
            "The similarity score is computed without labels, so it leaks no "
            "ground truth. The keyword lexicon does, because it was written "
            "against these tickets. When threshold tuning shifts decisions "
            "from the first onto the second, fold scores improve for reasons "
            "that will not transfer to a new project."
        )

    st.markdown("##### Thresholds chosen by the inner search")
    ui.data_table(nested.threshold_stability)
    st.caption(
        f"The shipped settings are weak {nested.shipped_thresholds.weak_match:.2f} / "
        f"strong {nested.shipped_thresholds.strong_match:.2f}. Where the search "
        "prefers a lower weak threshold, it is choosing to lean on the lexicon; "
        "the shipped value is deliberately the more conservative of the two."
    )

    with st.expander("Caveats on this run", expanded=nested.leakage_suspected):
        for caveat in nested.caveats:
            st.markdown(f"- {caveat}")

    # ---- Learning curve ---------------------------------------------------
    ui.section(
        "Is the model limited by the method or the data?",
        "Held-out macro-F1 of the learned classifier against training-set size.",
    )
    try:
        with ui.working("Building the learning curve…"):
            curve = validation.learning_curve(matched)
            summary = validation.learning_curve_summary(curve)
    except ValueError as exc:
        ui.no_evidence(f"Learning curve not possible: {exc}")
        return

    ui.chart(
        charts.learning_curve(summary),
        table=summary.round(3),
        key="pf_curve",
    )

    first, last = summary.iloc[0], summary.iloc[-1]
    gain = last["macro_f1"] - first["macro_f1"]
    if abs(gain) < 0.05:
        st.info(
            f"Macro-F1 is effectively flat between {first['train_size']:.0f} and "
            f"{last['train_size']:.0f} training tickets "
            f"({first['macro_f1']:.3f} → {last['macro_f1']:.3f}). On this "
            "evidence the learned classifier is limited by the **method**, not "
            "the sample size: more labelled tickets of the same kind would not "
            "lift it much. A richer representation would be the place to look.",
            icon="📊",
        )
    else:
        st.info(
            f"Macro-F1 moves {gain:+.3f} between {first['train_size']:.0f} and "
            f"{last['train_size']:.0f} training tickets, so the curve has not "
            "flattened: more labelled data would likely still help.",
            icon="📊",
        )
