"""Professional PDF report generation.

Built with ReportLab for layout and Matplotlib for chart images. Both run
offline; neither contacts a service.

Why these two libraries: ReportLab is the standard pure-Python PDF writer, and
Matplotlib was already a dependency of the scientific stack in use, so chart
rendering costs no additional package. Rendering the dashboard's Plotly figures
to images instead would have required Kaleido and a headless Chromium.

Every figure in the PDF is read from the same ``AnalysisResult`` the dashboard
renders, so the document cannot disagree with the screen. The structure
mirrors ``reporting.build_markdown_report`` deliberately.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from .config import CLASS_COLOURS, CLASSES
from .evaluation import EvaluationResult
from .pipeline import AnalysisResult

# Matplotlib must be configured for a non-interactive backend before pyplot is
# imported, or it will try to find a display.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#0b0b0b")
INK_SECONDARY = colors.HexColor("#52514e")
INK_MUTED = colors.HexColor("#898781")
ACCENT = colors.HexColor("#2a78d6")
RULE = colors.HexColor("#e1e0d9")
SURFACE = colors.HexColor("#f6f6f4")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "dgTitle", parent=base["Title"], fontSize=28, leading=32,
            textColor=INK, alignment=TA_CENTER, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "dgSubtitle", parent=base["Normal"], fontSize=13, leading=18,
            textColor=INK_SECONDARY, alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "dgH1", parent=base["Heading1"], fontSize=15, leading=19,
            textColor=INK, spaceBefore=14, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "dgH2", parent=base["Heading2"], fontSize=11.5, leading=15,
            textColor=INK, spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "dgBody", parent=base["Normal"], fontSize=9.5, leading=13.5,
            textColor=INK_SECONDARY, spaceAfter=5,
        ),
        "caption": ParagraphStyle(
            "dgCaption", parent=base["Normal"], fontSize=8, leading=11,
            textColor=INK_MUTED, spaceAfter=4,
        ),
        "cover_meta": ParagraphStyle(
            "dgCoverMeta", parent=base["Normal"], fontSize=10, leading=15,
            textColor=INK_SECONDARY, alignment=TA_CENTER,
        ),
    }


def _table(data: list[list[str]], *, widths: list[float] | None = None,
           align_right_from: int = 1) -> Table:
    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK),
                ("TEXTCOLOR", (0, 1), (-1, -1), INK_SECONDARY),
                ("BACKGROUND", (0, 0), (-1, 0), SURFACE),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


# --------------------------------------------------------------------------
# Charts (Matplotlib, so no extra dependency for image export)
# --------------------------------------------------------------------------
def _figure_to_image(figure, width_mm: float) -> Image:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=190, bbox_inches="tight",
                   facecolor="white")
    plt.close(figure)
    buffer.seek(0)
    width = width_mm * mm
    image = Image(buffer)
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    return image


def _style_axes(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#c3c2b7")
    axis.spines["bottom"].set_color("#c3c2b7")
    axis.tick_params(colors="#898781", labelsize=8)
    axis.yaxis.grid(True, color="#e1e0d9", linewidth=0.6)
    axis.set_axisbelow(True)


def _class_chart(per_class: pd.DataFrame, width_mm: float) -> Image:
    frame = per_class.set_index("classification").reindex(list(CLASSES)).fillna(0)
    figure, axis = plt.subplots(figsize=(5.0, 2.5))
    bars = axis.bar(
        list(frame.index),
        frame["tickets"].astype(int),
        color=[CLASS_COLOURS[c] for c in frame.index],
        width=0.62,
    )
    for bar, value in zip(bars, frame["tickets"].astype(int)):
        axis.annotate(
            f"{value:,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=8, color="#52514e",
        )
    axis.set_ylabel("Tickets", fontsize=8, color="#52514e")
    _style_axes(axis)
    return _figure_to_image(figure, width_mm)


def _timeline_chart(timeline: pd.DataFrame, width_mm: float) -> Image | None:
    if timeline.empty:
        return None
    date_column = timeline.columns[0]
    figure, axis = plt.subplots(figsize=(5.6, 2.4))
    axis.step(
        timeline[date_column], timeline["cumulative_sdi"],
        where="post", color=CLASS_COLOURS["New scope"], linewidth=1.8,
    )
    axis.fill_between(
        timeline[date_column], 0, timeline["cumulative_sdi"],
        step="post", color=CLASS_COLOURS["New scope"], alpha=0.10,
    )
    axis.axhline(0, color="#c3c2b7", linewidth=0.8)
    axis.set_ylabel("Cumulative drift", fontsize=8, color="#52514e")
    axis.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    )
    figure.autofmt_xdate(rotation=30, ha="right")
    _style_axes(axis)
    return _figure_to_image(figure, width_mm)


def _estimate_chart(estimates, width_mm: float) -> Image:
    figure, axes = plt.subplots(1, 2, figsize=(5.4, 2.3))
    pairs = [
        ("Effort (person-months)",
         [estimates.baseline.person_months, estimates.revised.person_months]),
        ("Duration (months)",
         [estimates.baseline.duration_months, estimates.revised.duration_months]),
    ]
    for axis, (title, values) in zip(axes, pairs):
        bars = axis.bar(
            ["Baseline", "Revised"], values,
            color=["#c9c8c2", CLASS_COLOURS["New scope"]], width=0.55,
        )
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:,.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8, color="#52514e",
            )
        axis.set_title(title, fontsize=8.5, color="#52514e")
        _style_axes(axis)
    figure.tight_layout()
    return _figure_to_image(figure, width_mm)


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------
def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK_MUTED)
    canvas.drawString(
        20 * mm, 12 * mm,
        "DriftGuard — scope drift analysis. Figures are estimates, not commitments.",
    )
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {canvas.getPageNumber()}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 16 * mm, 190 * mm, 16 * mm)
    canvas.restoreState()


def build_pdf_report(
    result: AnalysisResult,
    *,
    project_name: str = "DriftGuard analysis",
    evaluation: EvaluationResult | None = None,
    cross_validated: pd.DataFrame | None = None,
    risk_indicators: pd.DataFrame | None = None,
    recommendations: list[dict[str, str]] | None = None,
    synthetic: bool = False,
    generated_at: datetime | None = None,
) -> bytes:
    """Render the analysis as a paginated PDF and return its bytes."""
    styles = _styles()
    drift = result.drift
    estimates = result.estimates
    settings = result.settings
    unit = settings.size_unit.replace("_", " ")
    stamp = generated_at or datetime.now()
    content_width = 170.0

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title=f"{project_name} — DriftGuard analysis",
        author="DriftGuard",
        subject="Software project scope drift analysis",
    )

    story: list = []

    # ---- Cover ----------------------------------------------------------
    story.append(Spacer(1, 42 * mm))
    story.append(Paragraph("Scope Drift Analysis", styles["title"]))
    story.append(Paragraph(project_name, styles["subtitle"]))
    story.append(Spacer(1, 10 * mm))
    story.append(HRFlowable(width="40%", color=RULE, hAlign="CENTER"))
    story.append(Spacer(1, 10 * mm))

    severity = (
        "Critical" if drift.sdi > 0.25
        else "Serious" if drift.sdi > 0.10
        else "Watch" if drift.sdi > 0.05
        else "Stable"
    )
    story.append(
        Paragraph(
            f"<b>Scope Drift Index {drift.sdi:+.1%}</b> &nbsp;·&nbsp; {severity}",
            styles["cover_meta"],
        )
    )
    story.append(
        Paragraph(
            f"{drift.baseline_size:,.0f} → {drift.revised_size:,.0f} {unit} &nbsp;·&nbsp; "
            f"{len(result.classified):,} tickets analysed",
            styles["cover_meta"],
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        Paragraph(f"Analysis date: {stamp.strftime('%d %B %Y, %H:%M')}", styles["cover_meta"])
    )
    story.append(
        Paragraph("Generated locally by DriftGuard. No external service was used.",
                  styles["cover_meta"])
    )

    if synthetic:
        story.append(Spacer(1, 10 * mm))
        notice = Table(
            [[Paragraph(
                "<b>SYNTHETIC DATA.</b> This report was produced from generated "
                "test data. It demonstrates application behaviour and is not "
                "evidence about any real project.",
                styles["body"],
            )]],
            colWidths=[content_width * mm],
        )
        notice.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d03b3b")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf1f1")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(notice)

    story.append(PageBreak())

    # ---- Executive summary -----------------------------------------------
    story.append(Paragraph("Executive summary", styles["h1"]))
    cost_sentence = ""
    if settings.monthly_labour_rate:
        cost_sentence = (
            f" At the supplied rate of {settings.monthly_labour_rate:,.0f} per "
            f"person-month this is an estimated "
            f"{estimates.revised.labour_cost(settings.monthly_labour_rate):,.0f}, "
            "which is a derived labour-cost estimate rather than an actual or "
            "committed cost."
        )
    story.append(
        Paragraph(
            f"The agreed baseline for {project_name} comprised "
            f"{drift.baseline_size:,.0f} {unit} across "
            f"{len(result.classified):,} analysed tickets. Classification of "
            f"those tickets implies a revised scope of "
            f"{drift.revised_size:,.0f} {unit}, a net change of "
            f"{drift.net_change:+,.0f} and a Scope Drift Index of "
            f"<b>{drift.sdi:+.1%}</b>.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            f"Applying COCOMO II Post-Architecture, estimated effort moves from "
            f"{estimates.baseline.person_months:,.1f} to "
            f"{estimates.revised.person_months:,.1f} person-months "
            f"({estimates.effort_change_percent:+.1f}%), while estimated calendar "
            f"duration moves from {estimates.baseline.duration_months:,.1f} to "
            f"{estimates.revised.duration_months:,.1f} months "
            f"({estimates.duration_change_percent:+.1f}%). Effort grows faster "
            f"than duration because the model's exponent "
            f"E = {estimates.revised.exponent_e:.4f} exceeds one; the difference "
            f"implies average staffing rising from "
            f"{estimates.baseline.average_staff:,.1f} to "
            f"{estimates.revised.average_staff:,.1f} full-time equivalents."
            + cost_sentence,
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Person-months measure effort and months measure elapsed calendar "
            "time. They are distinct quantities and are not interchangeable by "
            "adding staff.",
            styles["caption"],
        )
    )

    # ---- Scope position ---------------------------------------------------
    story.append(Paragraph("Scope position", styles["h1"]))
    story.append(
        _table(
            [
                ["Measure", "Value"],
                ["Scope Drift Index", f"{drift.sdi:+.1%}"],
                [f"Baseline size ({unit})", f"{drift.baseline_size:,.0f}"],
                [f"Size added ({unit})", f"{drift.size_added:,.1f}"],
                [f"Size removed ({unit})", f"{drift.size_removed:,.1f}"],
                [f"Revised size ({unit})", f"{drift.revised_size:,.0f}"],
                ["Tickets analysed", f"{len(result.classified):,}"],
            ],
            widths=[100 * mm, 70 * mm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "SDI = (size added − size removed) ÷ baseline size, computed on net "
            f"change. Refinement tickets contribute {settings.refinement_weight:.2f} "
            "of their size, which is an assumption set by the reviewer.",
            styles["caption"],
        )
    )

    # ---- Classification ---------------------------------------------------
    story.append(Paragraph("Ticket classification", styles["h1"]))
    rows = [["Class", "Tickets", f"Size ({unit})", "Added", "Removed"]]
    per_class = drift.per_class.set_index("classification")
    for name in CLASSES:
        if name not in per_class.index:
            continue
        row = per_class.loc[name]
        rows.append([
            name, f"{int(row['tickets']):,}", f"{row['size']:,.0f}",
            f"{row['added']:,.1f}", f"{row['removed']:,.1f}",
        ])
    story.append(_table(rows, widths=[46 * mm, 26 * mm, 34 * mm, 32 * mm, 32 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(_class_chart(drift.per_class, content_width))

    # ---- Timeline ---------------------------------------------------------
    timeline_image = _timeline_chart(result.timeline, content_width)
    if timeline_image is not None:
        story.append(Paragraph("Scope change over time", styles["h1"]))
        story.append(timeline_image)
        story.append(
            Paragraph(
                "Cumulative Scope Drift Index as tickets arrive, ordered by "
                "creation date.",
                styles["caption"],
            )
        )

    # ---- Estimates --------------------------------------------------------
    # No forced break here: ReportLab flows the remaining sections, and tables
    # repeat their header row when they split. A hard break left a third of a
    # page blank.
    story.append(Paragraph("Effort and schedule estimates", styles["h1"]))
    revl_applied = (
        (estimates.revised.effective_ksloc / estimates.revised.size_ksloc - 1) * 100
        if estimates.revised.size_ksloc else 0.0
    )
    estimate_rows = [
        ["Quantity", "Baseline", "Revised", "Change"],
        [f"Size ({unit})", f"{drift.baseline_size:,.0f}", f"{drift.revised_size:,.0f}",
         f"{estimates.size_change_percent:+.1f}%"],
        ["Size (KSLOC)", f"{estimates.baseline.size_ksloc:,.2f}",
         f"{estimates.revised.size_ksloc:,.2f}", "—"],
        ["REVL applied", f"{settings.drivers.revl_percent:,.1f}%",
         f"{revl_applied:,.1f}%", "—"],
        ["Effort (person-months)", f"{estimates.baseline.person_months:,.1f}",
         f"{estimates.revised.person_months:,.1f}",
         f"{estimates.effort_change_percent:+.1f}%"],
        ["Duration (calendar months)", f"{estimates.baseline.duration_months:,.1f}",
         f"{estimates.revised.duration_months:,.1f}",
         f"{estimates.duration_change_percent:+.1f}%"],
        ["Average staffing (FTE)", f"{estimates.baseline.average_staff:,.1f}",
         f"{estimates.revised.average_staff:,.1f}", "—"],
    ]
    if settings.monthly_labour_rate:
        rate = settings.monthly_labour_rate
        estimate_rows.append([
            "Labour cost estimate",
            f"{estimates.baseline.labour_cost(rate):,.0f}",
            f"{estimates.revised.labour_cost(rate):,.0f}",
            f"{estimates.effort_change_percent:+.1f}%",
        ])
    story.append(_table(estimate_rows, widths=[56 * mm, 38 * mm, 38 * mm, 38 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(_estimate_chart(estimates, content_width))

    # ---- Significant changes ----------------------------------------------
    story.append(Paragraph("Tickets driving the drift", styles["h1"]))
    top = drift.top_contributors(10)
    if top.empty:
        story.append(Paragraph("No ticket changed the agreed scope.", styles["body"]))
    else:
        rows = [["Ticket", "Title", "Class", f"Net change ({unit})"]]
        for _, row in top.iterrows():
            rows.append([
                str(row.get("ticket_id", "?")),
                str(row.get("title", ""))[:52],
                str(row.get("final_class", "—")),
                f"{float(row['net_contribution']):+,.1f}",
            ])
        story.append(_table(rows, widths=[24 * mm, 84 * mm, 30 * mm, 32 * mm],
                            align_right_from=3))

    # ---- Human review -------------------------------------------------------
    story.append(Paragraph("Human review activity", styles["h1"]))
    reviewed = int(result.classified["reviewed"].sum()) if "reviewed" in result.classified else 0
    overturned = (
        int(result.classified["class_changed_by_reviewer"].sum())
        if "class_changed_by_reviewer" in result.classified else 0
    )
    story.append(
        _table(
            [
                ["Measure", "Value"],
                ["Tickets reviewed by a human", f"{reviewed:,}"],
                ["Classifications overturned", f"{overturned:,}"],
                ["Tickets awaiting review", f"{len(result.classified) - reviewed:,}"],
            ],
            widths=[120 * mm, 50 * mm],
        )
    )
    if reviewed == 0:
        story.append(
            Paragraph(
                "No ticket has been reviewed. Every figure in this report is "
                "unvalidated model output.",
                styles["caption"],
            )
        )

    # ---- Risk indicators ----------------------------------------------------
    if risk_indicators is not None and not risk_indicators.empty:
        story.append(Paragraph("Risk indicators", styles["h1"]))
        rows = [["Indicator", "Value", "Level", "Basis"]]
        for _, row in risk_indicators.iterrows():
            rows.append([
                Paragraph(str(row["Indicator"]), _styles()["body"]),
                str(row["Value"]),
                str(row["Level"]),
                Paragraph(str(row["Basis"]), _styles()["caption"]),
            ])
        story.append(_table(rows, widths=[48 * mm, 26 * mm, 22 * mm, 74 * mm],
                            align_right_from=4))
        story.append(
            Paragraph(
                "Severity bands are a reading convention for this report, not "
                "thresholds from any published standard.",
                styles["caption"],
            )
        )

    # ---- Recommendations ----------------------------------------------------
    if recommendations:
        story.append(Paragraph("Recommended actions", styles["h1"]))
        rows = [["Priority", "Action", "Because"]]
        for item in recommendations:
            rows.append([
                item["priority"],
                Paragraph(item["action"], _styles()["body"]),
                Paragraph(item["because"], _styles()["caption"]),
            ])
        story.append(_table(rows, widths=[20 * mm, 60 * mm, 90 * mm],
                            align_right_from=3))

    story.append(PageBreak())

    # ---- Evaluation ----------------------------------------------------------
    story.append(Paragraph("Model evaluation", styles["h1"]))
    if evaluation is None:
        story.append(
            Paragraph(
                "No labelled test set was supplied, so classifier accuracy was "
                "<b>not measured</b>. The classifications in this report are "
                "model output that has not been scored against ground truth.",
                styles["body"],
            )
        )
    else:
        story.append(
            Paragraph(
                f"Scored against {evaluation.n_samples:,} manually labelled tickets.",
                styles["body"],
            )
        )
        story.append(
            _table(
                [
                    ["Metric", "Value"],
                    ["Accuracy", f"{evaluation.accuracy:.1%}"],
                    ["Macro-F1", f"{evaluation.macro_f1:.3f}"],
                    ["Weighted F1", f"{evaluation.weighted_f1:.3f}"],
                    ["Majority-class baseline", f"{evaluation.baseline_majority_accuracy:.1%}"],
                ],
                widths=[120 * mm, 50 * mm],
            )
        )
        story.append(Spacer(1, 3 * mm))
        rows = [["Class", "Precision", "Recall", "F1", "Support"]]
        for _, row in evaluation.per_class.iterrows():
            rows.append([
                str(row["class"]), f"{row['precision']:.3f}", f"{row['recall']:.3f}",
                f"{row['f1']:.3f}", f"{int(row['support'])}",
            ])
        story.append(_table(rows, widths=[46 * mm, 32 * mm, 32 * mm, 30 * mm, 30 * mm]))
        story.append(Spacer(1, 3 * mm))
        story.append(
            Paragraph(
                "<b>These figures are in-sample.</b> The rule classifier's "
                "thresholds and keyword lexicon were tuned on this data, so the "
                "scores above overstate performance on unseen tickets. Macro-F1 "
                "leads rather than accuracy because the four classes are "
                "imbalanced.",
                styles["body"],
            )
        )
        if cross_validated is not None and not cross_validated.empty:
            story.append(
                Paragraph(
                    f"Cross-validated (out-of-sample): mean accuracy "
                    f"{cross_validated.attrs.get('mean_accuracy', 0.0):.1%}, mean "
                    f"macro-F1 {cross_validated.attrs.get('mean_macro_f1', 0.0):.3f} "
                    f"over {cross_validated.attrs.get('folds_used', len(cross_validated))} "
                    "stratified folds of the learned classifier.",
                    styles["body"],
                )
            )

    # ---- Methodology ----------------------------------------------------------
    story.append(Paragraph("Methodology", styles["h1"]))
    story.append(Paragraph("Classification", styles["h2"]))
    story.append(
        Paragraph(
            "Each ticket is matched to its closest baseline requirement by "
            "TF-IDF cosine similarity, then classified as In-scope, Refinement, "
            "New scope or Removal. Similarity establishes whether a ticket "
            "concerns the baseline at all; a visible keyword lexicon decides "
            "whether it implements, changes or withdraws the matched "
            "requirement. TF-IDF measures vocabulary overlap, not meaning, and "
            "is not a language model.",
            styles["body"],
        )
    )
    story.append(Paragraph("Estimation", styles["h2"]))
    story.append(
        Paragraph(
            "COCOMO II Post-Architecture with the published COCOMO II.2000 "
            "calibration (A = 2.94, B = 0.91, C = 3.67, D = 0.28). No local "
            "calibration was performed and no calibration data was fabricated. "
            "Requirements volatility (REVL) is charged on churn only — "
            "refinement-driven additions plus removals — because new scope is "
            "already carried by the larger revised size, and counting it in "
            "both places would charge for the same work twice.",
            styles["body"],
        )
    )

    story.append(Paragraph("Assumptions", styles["h1"]))
    story.append(
        _table(
            [
                ["Assumption", "Value", "Status"],
                ["Ticket size unit", unit, "User choice"],
                ["SLOC per story point", f"{settings.conversion.sloc_per_story_point:g}",
                 "Unvalidated placeholder"],
                ["SLOC per function point", f"{settings.conversion.sloc_per_function_point:g}",
                 "Common 3GL gearing ratio"],
                ["Refinement weight", f"{settings.refinement_weight:.2f}",
                 "Deliberate midpoint"],
                ["Weak-match threshold", f"{settings.thresholds.weak_match:.2f}",
                 "Tuned on the sample project"],
                ["Strong-match threshold", f"{settings.thresholds.strong_match:.2f}",
                 "Tuned on the sample project"],
                ["Churn fed into REVL", "yes" if settings.link_drift_to_revl else "no",
                 "Modelling choice"],
                ["Labour rate",
                 f"{settings.monthly_labour_rate:,.0f}" if settings.monthly_labour_rate else "not supplied",
                 "No rate means no cost reported"],
            ],
            widths=[62 * mm, 40 * mm, 68 * mm],
            align_right_from=3,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "The SLOC conversion factor scales every effort and duration figure "
            "in this report. At half the factor, every estimate halves. It is an "
            "organisational constant that should be measured, not inherited.",
            styles["caption"],
        )
    )

    # ---- Limitations -----------------------------------------------------------
    story.append(Paragraph("Limitations", styles["h1"]))
    for limitation in (
        "TF-IDF similarity measures lexical overlap, not meaning. A ticket that "
        "paraphrases a requirement without reusing its vocabulary is misread as "
        "new scope.",
        "The keyword lexicon is domain-specific and will not transfer unchanged "
        "to another project.",
        "Thresholds were tuned on the same labelled data used to report "
        "accuracy, so in-sample scores are optimistic.",
        "Ticket sizes are human estimates, and COCOMO II's own published "
        "accuracy is roughly within 30% of actuals 70% of the time under "
        "favourable conditions.",
        "Severity bands and the high-impact threshold are reading conventions "
        "for this report, not published standards.",
        "Analysis is bounded to a single project and English-language tickets.",
    ):
        story.append(Paragraph(f"• {limitation}", styles["body"]))

    # ---- Conclusion --------------------------------------------------------------
    story.append(Paragraph("Conclusion", styles["h1"]))
    direction = "grown" if drift.sdi > 0 else "contracted"
    story.append(
        Paragraph(
            f"Measured against its agreed baseline, {project_name} has "
            f"{direction} by {abs(drift.sdi):.1%} in scope terms. On the "
            "assumptions listed above, that corresponds to "
            f"{estimates.delta_person_months:+,.1f} person-months of effort and "
            f"{estimates.delta_duration_months:+,.1f} months of calendar time. "
            "These are estimates produced by a parametric model from "
            "human-estimated inputs, and should inform a scope conversation "
            "rather than substitute for one.",
            styles["body"],
        )
    )

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buffer.seek(0)
    return buffer.getvalue()
