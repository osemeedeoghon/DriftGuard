"""Scenario planning and sensitivity analysis.

Every figure on this page comes from the same COCOMO II implementation the
rest of the dashboard uses. Nothing here introduces an alternative formula.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driftguard import scenarios as engine
from driftguard.cocomo import DRIVER_NAMES, EFFORT_MULTIPLIERS, SizeConversion

from .. import charts, components as ui, theme


def render(result) -> None:
    settings = result.settings
    drift = result.drift
    unit = settings.size_unit.replace("_", " ")
    rate = settings.monthly_labour_rate

    ui.page_header(
        "Scenario and sensitivity analysis",
        "What the estimate would be under different scope decisions, and how "
        "much it depends on assumptions you have not yet validated.",
    )

    st.info(
        "Scenarios are arguments about possible futures, not records of the "
        "project. Hypothetical scenarios are hatched in the charts and flagged "
        "in every table so they cannot be mistaken for measured results.",
        icon="ℹ️",
    )

    # ======================================================================
    # Scenario comparison
    # ======================================================================
    ui.section(
        "Scope scenarios",
        "The agreed baseline and current classification are recorded states. "
        "The rest are what-ifs built from the same tickets.",
    )

    built = engine.build_scenarios(result)
    reference = st.selectbox(
        "Compare against",
        [s.name for s in built],
        index=0,
        help="Every delta on this page is measured from the scenario you pick here.",
    )

    with ui.working("Costing scenarios…"):
        comparison = engine.compare_scenarios(
            built,
            conversion=settings.conversion,
            size_unit=settings.size_unit,
            drivers=settings.drivers,
            rate=rate,
            reference=reference,
        )

    table = comparison.table
    size_column = f"Size ({unit})"

    display_columns = [
        "Scenario",
        "Hypothetical",
        size_column,
        "REVL %",
        "Effort (PM)",
        "Duration (months)",
        "Staffing (FTE)",
        "Δ Effort (PM)",
        "Δ Duration (months)",
    ]
    if rate:
        display_columns += ["Cost estimate", "Δ Cost"]

    ui.data_table(
        table[display_columns].round(2),
        column_config={
            "Hypothetical": st.column_config.CheckboxColumn("What-if?"),
            "Cost estimate": st.column_config.NumberColumn(format="%.0f")
            if rate
            else None,
            "Δ Cost": st.column_config.NumberColumn(format="%.0f") if rate else None,
        },
    )
    for note in comparison.notes:
        st.caption(note)

    metric = st.radio(
        "Chart metric",
        ["Effort (PM)", "Duration (months)", size_column]
        + (["Cost estimate"] if rate else []),
        horizontal=True,
        key="scenario_metric",
    )
    unit_label = {
        "Effort (PM)": "person-months",
        "Duration (months)": "calendar months",
        size_column: unit,
        "Cost estimate": "currency",
    }[metric]
    ui.chart(
        charts.scenario_bars(table, metric, unit_label),
        table=table[["Scenario", "Hypothetical", metric, "Description"]].round(2),
        key="sc_bars",
    )

    with st.expander("What each scenario assumes"):
        ui.data_table(table[["Scenario", "Hypothetical", "Description"]])

    # ======================================================================
    # Custom scenario
    # ======================================================================
    ui.section(
        "Build your own scenario",
        "Add or remove scope by hand and cost it against the reference.",
    )

    left, right = st.columns(2)
    with left:
        added = st.number_input(
            f"Scope to add ({unit})", min_value=0.0, value=0.0, step=5.0, key="sc_add"
        )
        removed = st.number_input(
            f"Scope to remove ({unit})",
            min_value=0.0,
            value=0.0,
            step=5.0,
            key="sc_remove",
        )
    with right:
        custom_revl = st.slider(
            "Requirements volatility for this scenario (REVL %)",
            0.0,
            60.0,
            float(
                round(
                    (
                        result.estimates.revised.effective_ksloc
                        / result.estimates.revised.size_ksloc
                        - 1
                    )
                    * 100
                    if result.estimates.revised.size_ksloc
                    else 0.0,
                    1,
                )
            ),
            1.0,
            key="sc_revl",
            help=(
                "Rework charged inside the effort equation. The default is the "
                "churn measured on this project."
            ),
        )

    if added or removed:
        custom_size = max(0.0, drift.revised_size + added - removed)
        custom = engine.Scenario(
            name="Custom scenario",
            size=custom_size,
            revl_percent=custom_revl,
            hypothetical=True,
            description=(
                f"Current classification {added:+,.0f} added, {removed:,.0f} removed."
            ),
        )
        custom_comparison = engine.compare_scenarios(
            [s for s in built if not s.hypothetical] + [custom],
            conversion=settings.conversion,
            size_unit=settings.size_unit,
            drivers=settings.drivers,
            rate=rate,
            reference="Current classification",
        )
        custom_row = custom_comparison.table.loc[
            custom_comparison.table["Scenario"] == "Custom scenario"
        ].iloc[0]

        ui.metric_row(
            [
                {
                    "label": f"Scenario size ({unit})",
                    "value": theme.fmt_number(custom_size),
                    "delta": theme.fmt_signed(custom_size - drift.revised_size, 0),
                    "delta_good": custom_size <= drift.revised_size,
                },
                {
                    "label": "Effort",
                    "value": theme.fmt_number(custom_row["Effort (PM)"], 1),
                    "delta": theme.fmt_signed(custom_row["Δ Effort (PM)"], 1, " PM"),
                    "delta_good": custom_row["Δ Effort (PM)"] <= 0,
                    "footnote": "person-months",
                },
                {
                    "label": "Duration",
                    "value": theme.fmt_number(custom_row["Duration (months)"], 1),
                    "delta": theme.fmt_signed(
                        custom_row["Δ Duration (months)"], 1, " mo"
                    ),
                    "delta_good": custom_row["Δ Duration (months)"] <= 0,
                    "footnote": "calendar months",
                },
                {
                    "label": "Cost estimate",
                    "value": theme.fmt_number(custom_row["Cost estimate"], 0)
                    if rate
                    else "no rate supplied",
                    "delta": theme.fmt_signed(custom_row["Δ Cost"], 0) if rate else None,
                    "delta_good": (custom_row["Δ Cost"] <= 0) if rate else None,
                },
            ]
        )
        st.caption(
            "Measured against the current classification. This is a "
            "hypothetical scenario."
        )
    else:
        ui.empty_state(
            "No custom scenario yet",
            f"Enter scope to add or remove above and it will be costed against "
            f"the current classification of {drift.revised_size:,.0f} {unit}.",
            icon="🧮",
        )

    # ======================================================================
    # Sensitivity
    # ======================================================================
    ui.section(
        "Sensitivity to documented assumptions",
        "How far the estimate moves when an assumption changes. The current "
        "setting is marked on each curve.",
    )

    which = st.radio(
        "Assumption",
        ["SLOC per story point", "Requirements volatility (REVL)", "COCOMO driver"],
        horizontal=True,
        key="sens_which",
    )

    if which == "SLOC per story point":
        ui.assumption(
            "This factor is an unvalidated placeholder. Effort scales with it, "
            "so the curve below is the single largest source of uncertainty in "
            "every estimate DriftGuard produces."
        )
        sweep = engine.sweep_sloc_per_point(
            drift.revised_size,
            size_unit=settings.size_unit,
            drivers=settings.drivers,
            rate=rate,
        )
        x_column = "SLOC per story point"
        current = settings.conversion.sloc_per_story_point
        elasticity = engine.elasticity(sweep, x_column, "Effort (PM)")
        st.caption(
            f"Effort elasticity with respect to this factor is "
            f"**{elasticity:.4f}** — that is the COCOMO exponent E, recovered "
            "from the sweep. A 10% error in the factor produces roughly an 11% "
            "error in effort."
        )
    elif which == "Requirements volatility (REVL)":
        sweep = engine.sweep_revl(
            drift.revised_size,
            conversion=settings.conversion,
            size_unit=settings.size_unit,
            drivers=settings.drivers,
            rate=rate,
        )
        x_column = "REVL %"
        current = (
            (
                result.estimates.revised.effective_ksloc
                / result.estimates.revised.size_ksloc
                - 1
            )
            * 100
            if result.estimates.revised.size_ksloc
            else 0.0
        )
        st.caption(
            "REVL charges rework inside the effort equation. DriftGuard derives "
            "it from measured churn — refinement additions plus removals — and "
            "deliberately excludes new scope, which the revised size already "
            "carries."
        )
    else:
        code = st.selectbox(
            "Driver",
            list(EFFORT_MULTIPLIERS),
            format_func=lambda c: f"{c} · {DRIVER_NAMES[c]}",
            key="sens_driver",
        )
        sweep = engine.sweep_driver(
            code,
            drift.revised_size,
            conversion=settings.conversion,
            size_unit=settings.size_unit,
            drivers=settings.drivers,
        )
        ui.data_table(sweep.round(2))
        if code == "SCED":
            st.caption(
                "SCED is the one driver with two published tables: an effort "
                "multiplier and a separate schedule percentage. Compressing a "
                "schedule raises effort *and* shortens the calendar; stretching "
                "it extends the calendar at no effort penalty. Both tables are "
                "applied here."
            )
        ui.chart(
            charts.sensitivity_curve(
                sweep.assign(index=range(len(sweep))),
                "Driver value",
                "Effort (PM)",
                title=f"Effort across {code} ratings",
            ),
            table=sweep.round(2),
            key="sc_driver",
        )
        sweep = None  # handled above

    if sweep is not None:
        left, right = st.columns(2)
        with left:
            ui.chart(
                charts.sensitivity_curve(
                    sweep, x_column, "Effort (PM)", current_x=current
                ),
                table=sweep.round(2),
                key="sc_effort",
            )
        with right:
            ui.chart(
                charts.sensitivity_curve(
                    sweep, x_column, "Duration (months)", current_x=current
                ),
                key="sc_duration",
            )
        if rate and "Cost estimate" in sweep.columns:
            ui.chart(
                charts.sensitivity_curve(
                    sweep, x_column, "Cost estimate", current_x=current
                ),
                key="sc_cost",
            )

    # ======================================================================
    # Tornado
    # ======================================================================
    ui.section(
        "Which drivers matter most?",
        "Effort swing across each driver's full published rating range, widest "
        "first. Useful before arguing about any individual rating.",
    )
    with ui.working("Sweeping every effort multiplier…"):
        tornado = engine.tornado(
            drift.revised_size,
            conversion=settings.conversion,
            size_unit=settings.size_unit,
            drivers=settings.drivers,
        )
    ui.chart(
        charts.tornado_chart(tornado, result.estimates.revised.person_months),
        table=tornado.round(2),
        key="sc_tornado",
    )
    st.caption(
        "Swings are computed by moving one driver across its range while "
        "holding the others at their current ratings. They are not additive: "
        "setting two drivers to their worst ratings multiplies, it does not sum."
    )
