"""Chart builders for DriftGuard.

Each function returns a Plotly figure; pages render them through
``components.chart`` so the table view travels with the picture.

Form choices are deliberate:

* magnitude comparisons use **ranked horizontal bars in a single hue**, not
  categorical colour, so the reader compares lengths rather than decoding a
  legend;
* the four scope classes use the **categorical** palette, because there the
  classes genuinely are the subject;
* the confusion matrix uses a **sequential** one-hue ramp, because its cells
  encode counts;
* nothing here uses a second y-axis, a pie or a rainbow scale.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from driftguard.config import CLASSES, CLASS_COLOURS

from . import theme

#: Rounded data-ends, anchored at the baseline.
_BAR_RADIUS = 4
#: Surface-coloured gap between adjacent/stacked fills.
_BAR_GAP_LINE = {"color": theme.SURFACE, "width": 2}


def class_bar(per_class: pd.DataFrame) -> go.Figure:
    """Ticket counts by classification, with every bar directly labelled."""
    frame = per_class.set_index("classification").reindex(list(CLASSES)).reset_index()
    counts = frame["tickets"].fillna(0).astype(int)

    figure = go.Figure(
        go.Bar(
            x=frame["classification"],
            y=counts,
            marker={
                "color": [CLASS_COLOURS[c] for c in frame["classification"]],
                "line": _BAR_GAP_LINE,
                "cornerradius": _BAR_RADIUS,
            },
            text=[f"{c:,}" for c in counts],
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 12},
            hovertemplate="<b>%{x}</b><br>%{y:,} tickets<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_yaxes(title_text="Tickets", rangemode="tozero")
    figure.update_xaxes(title_text="")
    return theme.styled(figure, height=320, title="Tickets by classification")


def scope_waterfall(
    baseline: float, added: float, removed: float, revised: float, unit: str
) -> go.Figure:
    """Baseline to revised size, showing what was added and removed."""
    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Baseline", "Added", "Removed", "Revised"],
            y=[baseline, added, -removed, revised],
            text=[
                f"{baseline:,.0f}",
                f"+{added:,.0f}",
                f"−{removed:,.0f}",
                f"{revised:,.0f}",
            ],
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 12},
            increasing={"marker": {"color": CLASS_COLOURS["New scope"]}},
            decreasing={"marker": {"color": CLASS_COLOURS["Removal"]}},
            totals={"marker": {"color": theme.INK_SECONDARY}},
            connector={"line": {"color": theme.AXIS_LINE, "width": 1}},
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} " + unit + "<extra></extra>",
        )
    )
    figure.update_yaxes(title_text=unit, rangemode="tozero")
    return theme.styled(figure, height=320, title="Baseline to revised scope")


def drift_timeline(
    timeline: pd.DataFrame, date_column: str, *, show_bands: bool = True
) -> go.Figure:
    """Cumulative Scope Drift Index as tickets arrive.

    One series, so no legend: the title names it.
    """
    figure = go.Figure()
    figure.add_scatter(
        x=timeline[date_column],
        y=timeline["cumulative_sdi"],
        mode="lines",
        name="Cumulative SDI",
        line={"color": CLASS_COLOURS["New scope"], "width": 2, "shape": "hv"},
        fill="tozeroy",
        fillcolor=theme.rgba(CLASS_COLOURS["New scope"], 0.10),
        hovertemplate="%{x|%d %b %Y}<br>Drift %{y:.1%}<extra></extra>",
    )
    figure.add_hline(y=0, line_width=1, line_color=theme.AXIS_LINE)

    if show_bands:
        for level, label, colour in (
            (0.10, "Serious 10%", theme.STATUS_SERIOUS),
            (0.25, "Critical 25%", theme.STATUS_CRITICAL),
        ):
            figure.add_hline(
                y=level,
                line_dash="dash",
                line_width=1,
                line_color=colour,
                annotation_text=label,
                annotation_position="top left",
                annotation_font={"color": colour, "size": 11},
            )

    figure.update_yaxes(title_text="Cumulative drift", tickformat=".0%")
    figure.update_xaxes(title_text="")
    figure.update_layout(hovermode="x unified", showlegend=False)
    return theme.styled(figure, height=360, title="Scope drift accumulates over time")


def contributor_bars(
    contributors: pd.DataFrame,
    unit: str,
    *,
    label_column: str = "ticket_id",
    title: str | None = "Tickets driving the most drift",
) -> go.Figure:
    """Ranked horizontal bars for the tickets driving the most drift.

    Magnitude comparison, so a single hue with signed direction rather than
    four categorical colours.
    """
    frame = contributors.copy()
    if frame.empty:
        return theme.styled(go.Figure(), height=260, title="No drift contributors")

    frame = frame.iloc[::-1]  # Plotly draws the first row at the bottom
    values = frame["net_contribution"].astype(float)
    colours = [
        CLASS_COLOURS["New scope"] if v >= 0 else CLASS_COLOURS["Removal"]
        for v in values
    ]
    labels = [
        f"{row[label_column]} · {str(row.get('title', ''))[:42]}"
        for _, row in frame.iterrows()
    ]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={
                "color": colours,
                "line": _BAR_GAP_LINE,
                "cornerradius": _BAR_RADIUS,
            },
            text=[f"{v:+,.0f}" for v in values],
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 12},
            hovertemplate="<b>%{y}</b><br>%{x:+,.0f} " + unit + "<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_vline(x=0, line_width=1, line_color=theme.AXIS_LINE)
    figure.update_xaxes(title_text=f"Net scope change ({unit})")
    figure.update_yaxes(title_text="", automargin=True)
    return theme.styled(
        figure, height=max(260, 42 * len(frame) + 70), title=title
    )


def estimate_comparison(
    baseline_pm: float, revised_pm: float, baseline_tdev: float, revised_tdev: float
) -> go.Figure:
    """Baseline versus revised effort and duration, as small multiples.

    Effort and duration are different quantities on different scales, so they
    get one panel each rather than a second y-axis.
    """
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Effort (person-months)", "Duration (calendar months)"),
        horizontal_spacing=0.16,
    )

    for column, (base, revised) in enumerate(
        [(baseline_pm, revised_pm), (baseline_tdev, revised_tdev)], start=1
    ):
        figure.add_trace(
            go.Bar(
                x=["Baseline", "Revised"],
                y=[base, revised],
                marker={
                    "color": [theme.CONTEXT_GREY, CLASS_COLOURS["New scope"]],
                    "line": _BAR_GAP_LINE,
                    "cornerradius": _BAR_RADIUS,
                },
                text=[f"{base:,.1f}", f"{revised:,.1f}"],
                textposition="outside",
                textfont={"color": theme.INK_SECONDARY, "size": 12},
                hovertemplate="<b>%{x}</b><br>%{y:,.1f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_yaxes(rangemode="tozero", row=1, col=column)

    figure.update_annotations(font={"size": 12, "color": theme.INK_SECONDARY})
    return theme.styled(figure, height=330)


def confusion_heatmap(confusion: pd.DataFrame) -> go.Figure:
    """Confusion matrix. Counts are magnitude, so one hue light to dark."""
    values = confusion.to_numpy()
    peak = values.max() if values.size else 1

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=list(confusion.columns),
            y=list(confusion.index),
            colorscale=[[i / (len(theme.SEQUENTIAL_BLUE) - 1), c]
                        for i, c in enumerate(theme.SEQUENTIAL_BLUE)],
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate=(
                "Actual <b>%{y}</b><br>Predicted <b>%{x}</b>"
                "<br>%{z:,} tickets<extra></extra>"
            ),
        )
    )

    # Every cell is labelled, so the matrix never relies on colour alone.
    for row_index, row_label in enumerate(confusion.index):
        for col_index, col_label in enumerate(confusion.columns):
            count = values[row_index][col_index]
            figure.add_annotation(
                x=col_label,
                y=row_label,
                text=f"{count:,}",
                showarrow=False,
                font={
                    "color": "#ffffff" if count > peak * 0.55 else theme.INK_PRIMARY,
                    "size": 13,
                },
            )

    figure.update_xaxes(title_text="Predicted", side="bottom")
    figure.update_yaxes(title_text="Actual", autorange="reversed")
    return theme.styled(figure, height=360, title="Confusion matrix")


def per_class_metrics(per_class: pd.DataFrame) -> go.Figure:
    """Precision and recall per class: two series, so a legend is present."""
    frame = per_class.copy()
    figure = go.Figure()
    for name, colour in (("precision", theme.ACCENT), ("recall", theme.ACCENT_DARK)):
        figure.add_bar(
            name=name.capitalize(),
            x=frame["class"],
            y=frame[name],
            marker={
                "color": colour,
                "line": _BAR_GAP_LINE,
                "cornerradius": _BAR_RADIUS,
            },
            text=[f"{v:.2f}" for v in frame[name]],
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 11},
            hovertemplate="<b>%{x}</b><br>" + name + " %{y:.3f}<extra></extra>",
        )
    figure.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    figure.update_yaxes(title_text="Score", range=[0, 1.12], tickformat=".1f")
    figure.update_xaxes(title_text="")
    return theme.styled(figure, height=340, title="Precision and recall by class")


def similarity_distribution(classified: pd.DataFrame, class_column: str) -> go.Figure:
    """Similarity score spread per class, as a box plot with points.

    This is the honest version of a "feature distribution comparison" for this
    application: the similarity score is the model's decisive feature, and the
    classes are the groups being compared.
    """
    figure = go.Figure()
    for name in CLASSES:
        subset = classified.loc[classified[class_column] == name, "similarity"]
        if subset.empty:
            continue
        figure.add_box(
            y=subset,
            name=name,
            marker={"color": CLASS_COLOURS[name], "size": 6},
            line={"width": 2},
            fillcolor=theme.rgba(CLASS_COLOURS[name], 0.14),
            boxpoints="all",
            jitter=0.45,
            pointpos=0,
            hovertemplate="<b>" + name + "</b><br>similarity %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    figure.update_yaxes(title_text="Cosine similarity", rangemode="tozero")
    figure.update_xaxes(title_text="")
    return theme.styled(
        figure, height=360, title="Similarity to the matched requirement, by class"
    )


def size_distribution(
    baseline_sizes: pd.Series, ticket_sizes: pd.Series, unit: str
) -> go.Figure:
    """Reference (baseline) versus current (tickets) size distributions."""
    figure = go.Figure()
    for name, series, colour in (
        ("Baseline requirements", baseline_sizes, theme.CONTEXT_GREY),
        ("Project tickets", ticket_sizes, theme.ACCENT),
    ):
        figure.add_box(
            x=series,
            name=name,
            orientation="h",
            marker={"color": colour, "size": 6},
            line={"width": 2},
            fillcolor=theme.rgba(colour, 0.18),
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            hovertemplate="<b>" + name + "</b><br>%{x:,.0f} " + unit + "<extra></extra>",
        )
    figure.update_xaxes(title_text=unit, rangemode="tozero")
    figure.update_yaxes(title_text="", automargin=True)
    figure.update_layout(showlegend=True)
    return theme.styled(figure, height=320, title="Size distribution: reference vs current")


def term_weights(terms: pd.DataFrame, title: str) -> go.Figure:
    """Ranked horizontal bars for TF-IDF term importance. One hue."""
    if terms.empty:
        return theme.styled(go.Figure(), height=240, title=title)

    frame = terms.iloc[::-1]
    figure = go.Figure(
        go.Bar(
            x=frame["weight"],
            y=frame["term"],
            orientation="h",
            marker={
                "color": theme.ACCENT,
                "line": _BAR_GAP_LINE,
                "cornerradius": _BAR_RADIUS,
            },
            text=[f"{v:.3f}" for v in frame["weight"]],
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 11},
            hovertemplate="<b>%{y}</b><br>weight %{x:.4f}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_xaxes(title_text="TF-IDF weight", rangemode="tozero")
    figure.update_yaxes(title_text="", automargin=True)
    return theme.styled(figure, height=max(240, 30 * len(frame) + 90), title=title)


def coverage_bars(coverage: pd.DataFrame, unit: str) -> go.Figure:
    """Tickets matched per baseline requirement, ranked. Single hue."""
    frame = coverage.iloc[::-1]
    colours = [
        theme.CONTEXT_GREY if n == 0 else theme.ACCENT for n in frame["tickets"]
    ]
    figure = go.Figure(
        go.Bar(
            x=frame["tickets"],
            y=frame["label"],
            orientation="h",
            marker={
                "color": colours,
                "line": _BAR_GAP_LINE,
                "cornerradius": _BAR_RADIUS,
            },
            text=[f"{n:,}" for n in frame["tickets"]],
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 11},
            hovertemplate="<b>%{y}</b><br>%{x:,} tickets matched<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_xaxes(title_text="Tickets matched", rangemode="tozero", dtick=1)
    figure.update_yaxes(title_text="", automargin=True)
    return theme.styled(
        figure,
        height=max(300, 26 * len(frame) + 90),
        title="Baseline coverage (grey = no ticket matched)",
    )


def snapshot_history(history: pd.DataFrame) -> go.Figure:
    """Scope Drift Index across recorded snapshots, in the order taken."""
    if history.empty:
        return theme.styled(go.Figure(), height=280, title="Scope history")

    figure = go.Figure()
    figure.add_scatter(
        x=history["Snapshot"],
        y=history["SDI"],
        mode="lines+markers",
        line={"color": CLASS_COLOURS["New scope"], "width": 2},
        marker={"size": 10, "color": CLASS_COLOURS["New scope"]},
        text=[f"{v:+.1%}" for v in history["SDI"]],
        textposition="top center",
        hovertemplate="<b>%{x}</b><br>SDI %{y:+.1%}<extra></extra>",
        showlegend=False,
    )
    figure.add_hline(y=0, line_width=1, line_color=theme.AXIS_LINE)
    figure.update_yaxes(title_text="Scope Drift Index", tickformat=".0%")
    figure.update_xaxes(title_text="")
    return theme.styled(
        figure, height=320, title="Scope Drift Index across recorded snapshots"
    )


def scenario_bars(table: pd.DataFrame, metric: str, unit_label: str) -> go.Figure:
    """Ranked horizontal bars comparing scenarios on one metric.

    Recorded scenarios take the accent hue; hypotheticals are drawn in the
    de-emphasis grey with a hatched fill, so a what-if can never be mistaken
    for a record of the project.
    """
    if table.empty:
        return theme.styled(go.Figure(), height=260, title=metric)

    frame = table.iloc[::-1]
    colours = [
        theme.CONTEXT_GREY if hypo else theme.ACCENT
        for hypo in frame["Hypothetical"]
    ]
    patterns = ["/" if hypo else "" for hypo in frame["Hypothetical"]]

    figure = go.Figure(
        go.Bar(
            x=frame[metric],
            y=frame["Scenario"],
            orientation="h",
            marker={
                "color": colours,
                "line": _BAR_GAP_LINE,
                "cornerradius": _BAR_RADIUS,
                "pattern": {"shape": patterns, "solidity": 0.35},
            },
            text=[f"{v:,.1f}" for v in frame[metric]],
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 11},
            hovertemplate="<b>%{y}</b><br>%{x:,.1f} " + unit_label + "<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_xaxes(title_text=f"{metric}", rangemode="tozero")
    figure.update_yaxes(title_text="", automargin=True)
    return theme.styled(
        figure,
        height=max(260, 46 * len(frame) + 80),
        title=f"{metric} by scenario (hatched = hypothetical)",
    )


def sensitivity_curve(
    sweep: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    current_x: float | None = None,
    title: str | None = None,
) -> go.Figure:
    """One assumption swept against one outcome. Single series, no legend."""
    if sweep.empty:
        return theme.styled(go.Figure(), height=300, title=title or y_column)

    figure = go.Figure()
    figure.add_scatter(
        x=sweep[x_column],
        y=sweep[y_column],
        mode="lines+markers",
        line={"color": theme.ACCENT, "width": 2},
        marker={"size": 8, "color": theme.ACCENT},
        hovertemplate=f"{x_column} %{{x:,.1f}}<br>{y_column} %{{y:,.1f}}<extra></extra>",
        showlegend=False,
    )
    if current_x is not None:
        figure.add_vline(
            x=current_x,
            line_dash="dash",
            line_width=1,
            line_color=theme.STATUS_SERIOUS,
            annotation_text="current setting",
            annotation_position="top right",
            annotation_font={"color": theme.STATUS_SERIOUS, "size": 11},
        )
    figure.update_xaxes(title_text=x_column)
    figure.update_yaxes(title_text=y_column, rangemode="tozero")
    return theme.styled(figure, height=320, title=title or f"{y_column} vs {x_column}")


def tornado_chart(frame: pd.DataFrame, nominal: float) -> go.Figure:
    """Effort swing per COCOMO driver, widest first."""
    if frame.empty:
        return theme.styled(go.Figure(), height=280, title="Driver sensitivity")

    ordered = frame.iloc[::-1]
    figure = go.Figure()
    figure.add_bar(
        y=ordered["Driver"],
        x=ordered["Lowest effort (PM)"] - nominal,
        orientation="h",
        name="Best rating",
        marker={"color": theme.rgba(theme.ACCENT, 0.55), "line": _BAR_GAP_LINE},
        hovertemplate="<b>%{y}</b> best case<br>%{customdata:,.1f} PM<extra></extra>",
        customdata=ordered["Lowest effort (PM)"],
    )
    figure.add_bar(
        y=ordered["Driver"],
        x=ordered["Highest effort (PM)"] - nominal,
        orientation="h",
        name="Worst rating",
        marker={"color": theme.ACCENT_DARK, "line": _BAR_GAP_LINE},
        hovertemplate="<b>%{y}</b> worst case<br>%{customdata:,.1f} PM<extra></extra>",
        customdata=ordered["Highest effort (PM)"],
    )
    figure.add_vline(
        x=0,
        line_width=1,
        line_color=theme.AXIS_LINE,
        annotation_text=f"nominal {nominal:,.1f} PM",
        annotation_position="top right",
        annotation_font={"color": theme.INK_MUTED, "size": 11},
    )
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title_text="Person-months away from nominal")
    figure.update_yaxes(title_text="", automargin=True)
    return theme.styled(
        figure,
        height=max(300, 30 * len(frame) + 110),
        title="Which COCOMO drivers move the estimate most?",
    )


def learning_curve(summary: pd.DataFrame) -> go.Figure:
    """Macro-F1 against training-set size, with the spread across repeats.

    One series, so no legend. A curve still climbing at the right edge means
    more labels would help; a flat one means the method is the ceiling.
    """
    if summary.empty:
        return theme.styled(go.Figure(), height=300, title="Learning curve")

    upper = summary["macro_f1"] + summary["macro_f1_std"]
    lower = summary["macro_f1"] - summary["macro_f1_std"]

    figure = go.Figure()
    figure.add_scatter(
        x=list(summary["train_size"]) + list(summary["train_size"])[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself",
        fillcolor=theme.rgba(theme.ACCENT, 0.12),
        line={"width": 0},
        hoverinfo="skip",
        showlegend=False,
    )
    figure.add_scatter(
        x=summary["train_size"],
        y=summary["macro_f1"],
        mode="lines+markers",
        line={"color": theme.ACCENT, "width": 2},
        marker={"size": 9, "color": theme.ACCENT},
        text=[f"{v:.3f}" for v in summary["macro_f1"]],
        hovertemplate="%{x:.0f} training tickets<br>macro-F1 %{y:.3f}<extra></extra>",
        showlegend=False,
    )
    figure.update_xaxes(title_text="Training tickets")
    figure.update_yaxes(title_text="Macro-F1 (held out)", range=[0, 1])
    return theme.styled(
        figure, height=320, title="Learning curve — does more labelled data help?"
    )


def nested_fold_scores(folds: pd.DataFrame, in_sample: float) -> go.Figure:
    """Per-fold held-out macro-F1 against the in-sample reference line."""
    if folds.empty:
        return theme.styled(go.Figure(), height=280, title="Nested folds")

    figure = go.Figure(
        go.Bar(
            x=[f"Fold {int(f)}" for f in folds["fold"]],
            y=folds["macro_f1"],
            marker={
                "color": theme.ACCENT,
                "line": _BAR_GAP_LINE,
                "cornerradius": _BAR_RADIUS,
            },
            text=[f"{v:.3f}" for v in folds["macro_f1"]],
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 11},
            hovertemplate="<b>%{x}</b><br>held-out macro-F1 %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_hline(
        y=in_sample,
        line_dash="dash",
        line_width=1,
        line_color=theme.STATUS_CRITICAL,
        annotation_text=f"in-sample {in_sample:.3f}",
        annotation_position="top left",
        annotation_font={"color": theme.STATUS_CRITICAL, "size": 11},
    )
    figure.update_yaxes(title_text="Macro-F1", range=[0, 1.12])
    figure.update_xaxes(title_text="")
    return theme.styled(
        figure, height=300, title="Held-out score per outer fold"
    )


def confidence_histogram(classified: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    """Classifier confidence spread, with the review threshold marked."""
    figure = go.Figure(
        go.Histogram(
            x=classified["confidence"],
            nbinsx=20,
            marker={
                "color": theme.ACCENT,
                "line": _BAR_GAP_LINE,
            },
            hovertemplate="confidence %{x:.2f}<br>%{y:,} tickets<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_vline(
        x=threshold,
        line_dash="dash",
        line_width=1,
        line_color=theme.STATUS_SERIOUS,
        annotation_text=f"review below {threshold:.2f}",
        annotation_position="top right",
        annotation_font={"color": theme.STATUS_SERIOUS, "size": 11},
    )
    figure.update_xaxes(title_text="Confidence", range=[0, 1])
    figure.update_yaxes(title_text="Tickets", rangemode="tozero")
    return theme.styled(figure, height=300, title="Classifier confidence distribution")
