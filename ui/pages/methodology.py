"""Methodology and About."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driftguard import __version__, validation
from driftguard.cocomo import describe_drivers
from driftguard.config import CLASS_DEFINITIONS, CLASSES

from .. import components as ui, theme


def render(result) -> None:
    settings = result.settings
    estimates = result.estimates
    unit = settings.size_unit.replace("_", " ")

    ui.page_header(
        "Methodology and about",
        "What DriftGuard measures, what it assumes, and what it cannot tell you.",
    )

    st.markdown(
        """
DriftGuard compares an agreed **baseline** of requirements against the
**tickets** raised on a project. It matches each ticket to its closest
requirement using local text similarity, classifies the relationship,
quantifies the net scope change as a **Scope Drift Index**, and translates that
into effort and schedule using **COCOMO II Post-Architecture**.

Everything runs on this machine. There is no LLM API, no API key, no cloud
database and no paid service anywhere in the execution path.
"""
    )

    # ---- Pipeline ---------------------------------------------------------
    ui.section("Pipeline")
    stages = pd.DataFrame(
        {
            "Stage": ["Load", "Match", "Classify", "Review", "Measure", "Estimate"],
            "What happens": [
                "CSV/TXT parsed, columns normalised, sizes and dates validated",
                "TF-IDF vectors, cosine similarity of every ticket against every requirement",
                "Similarity thresholds plus a visible keyword lexicon, or a logistic regression",
                "A human confirms or overrules any classification and size",
                "Scope Drift Index from net added and removed size",
                "COCOMO II effort and schedule for baseline and revised sizes",
            ],
            "Module": [
                "driftguard.data",
                "driftguard.matching",
                "driftguard.classification",
                "driftguard.classification",
                "driftguard.drift",
                "driftguard.cocomo",
            ],
        }
    )
    ui.data_table(stages)

    # ---- Taxonomy ---------------------------------------------------------
    ui.section("Classification taxonomy")
    ui.data_table(
        pd.DataFrame(
            {
                "Class": list(CLASSES),
                "Definition": [CLASS_DEFINITIONS[c] for c in CLASSES],
            }
        )
    )

    st.markdown(
        """
The two signals do different jobs, deliberately:

* **Cosine similarity** answers *"does this ticket concern the agreed baseline
  at all?"* It is good at that, because an unrelated ticket shares almost no
  distinctive vocabulary with any requirement.
* **Keyword cues** answer *"does it implement, change, or withdraw that
  requirement?"* Similarity cannot answer this — a ticket implementing a
  requirement and a ticket rewriting it look nearly identical to a
  bag-of-words model.

Removal detection is two-tier. Unambiguous phrases ("de-scoped", "no longer
required") fire on their own, but generic verbs ("remove", "withdraw",
"cancel") count only when a scope noun appears alongside them. Without that
guard, ordinary domain language — *"administrators can withdraw a course
section"* — reads as de-scoping.
"""
    )

    # ---- Equations --------------------------------------------------------
    ui.section("Equations")
    st.markdown(
        """
**Scope Drift Index**

```
SDI = (cumulative size added − cumulative size removed) ÷ baseline size
revised size = baseline × (1 + SDI)
```

**COCOMO II Post-Architecture**

```
E    = B + 0.01 × Σ SF_j                 (5 scale factors)
PM   = A × Size^E × Π EM_i               (17 effort multipliers)
F    = D + 0.2 × (E − B)
TDEV = C × PM_NS^F × (SCED% ÷ 100)
```

with the published COCOMO II.2000 calibration `A = 2.94`, `B = 0.91`,
`C = 3.67`, `D = 0.28`, `Size` in KSLOC, and `PM_NS` the effort excluding the
SCED multiplier. **No local calibration is performed and no calibration data is
fabricated.**
"""
    )
    ui.metric_row(
        [
            {"label": "Exponent E", "value": f"{estimates.revised.exponent_e:.4f}",
             "footnote": "> 1 means diseconomy of scale"},
            {"label": "Exponent F", "value": f"{estimates.revised.exponent_f:.4f}"},
            {"label": "Effort multiplier product", "value": f"{estimates.revised.em_product:.4f}"},
            {"label": "Person-month", "value": "152 hours", "footnote": "COCOMO convention"},
        ]
    )

    st.markdown(
        """
**Effort is not duration.** `PM` is person-months of work; `TDEV` is elapsed
calendar months. They are different quantities, reported separately everywhere
in this dashboard, and they do not convert into one another by adding staff.

**REVL excludes new scope.** Feeding the whole Scope Drift Index into COCOMO's
requirements-volatility factor would double-count, because the revised size
already contains the added scope. DriftGuard charges REVL only for churn:
`(refinement-driven additions + removals) ÷ baseline size`.
"""
    )

    # ---- Assumptions ------------------------------------------------------
    ui.section("Assumptions currently in force", "Every one of these is adjustable in the sidebar.")
    assumptions = pd.DataFrame(
        {
            "Assumption": [
                "Ticket size unit",
                "SLOC per story point",
                "SLOC per function point",
                "Refinement weight",
                "Weak-match threshold",
                "Strong-match threshold",
                "Churn fed into REVL",
                "Labour rate",
            ],
            "Value": [
                unit,
                f"{settings.conversion.sloc_per_story_point:g}",
                f"{settings.conversion.sloc_per_function_point:g}",
                f"{settings.refinement_weight:.2f}",
                f"{settings.thresholds.weak_match:.2f}",
                f"{settings.thresholds.strong_match:.2f}",
                "yes" if settings.link_drift_to_revl else "no",
                f"{settings.monthly_labour_rate:,.0f} per person-month"
                if settings.monthly_labour_rate
                else "not supplied",
            ],
            "Status": [
                "User choice",
                "Unvalidated placeholder — replace with your own history",
                "Common 3GL gearing ratio; Python is nearer 20–30",
                "Deliberate midpoint, no empirical basis",
                "Tuned on the bundled sample project",
                "Tuned on the bundled sample project",
                "Modelling choice",
                "No rate supplied means no cost is reported",
            ],
        }
    )
    ui.data_table(assumptions)

    ui.assumption(
        "The SLOC conversion factor scales every effort and duration figure "
        "in this dashboard. At half the factor, every estimate halves. It is "
        "an organisational constant you should measure, not inherit."
    )

    with st.expander("COCOMO II driver ratings in force"):
        ui.data_table(pd.DataFrame(describe_drivers(settings.drivers)))

    # ---- Live audit -------------------------------------------------------
    ui.section(
        "Methodology audit",
        "These claims are re-derived from the current analysis rather than "
        "asserted, so a reviewer can see them verified on their own data.",
    )
    audit = validation.methodology_audit(result)
    passed = int((audit["Result"] == "PASS").sum())
    if passed == len(audit):
        st.success(
            f"All {len(audit)} checks pass on the loaded project.", icon="✅"
        )
    else:
        st.error(
            f"{len(audit) - passed} of {len(audit)} checks failed. The figures "
            "on other pages should not be trusted until this is resolved.",
            icon="🚩",
        )
    ui.data_table(audit)
    st.caption(
        "Each check recomputes a quantity from the result object and compares "
        "it with what the pipeline reported. The same invariants are asserted "
        "in the test suite; this panel runs them against live data."
    )

    # ---- Evaluation integrity ---------------------------------------------
    ui.section(
        "Evaluation integrity",
        "Where the measurement of this tool's accuracy is, and is not, sound.",
    )
    st.markdown(
        """
**Training, validation and testing are not the same thing.**

- *Training data* fits a model's parameters.
- *Validation data* selects between models or settings — thresholds, in our case.
- *Test data* is touched once, to estimate performance on work never seen.

DriftGuard's 45 labelled tickets have served all three roles, which is the
central methodological weakness of this project and should be stated first in
any defence of it.

| Component | Tuned on | Honest estimate available? |
|---|---|---|
| TF-IDF vectoriser | whole corpus, no labels | Leaks no ground truth, but the evaluation is *transductive* — the vectoriser has seen the test fold's vocabulary |
| Similarity thresholds | the 45 labelled tickets | Yes — nested cross-validation on the Model Performance page |
| Keyword lexicon | the 45 labelled tickets, by hand | **No** — it cannot be re-derived per fold |
| Logistic regression | per fold | Yes — stratified cross-validation |

**What would fix it.** A second, independently labelled ticket set from a
different project, held back entirely and scored once. Nothing short of that
gives the rule classifier a clean number, because the lexicon cannot be
un-learned from the data that produced it.

**On the 45-ticket sample.** Each outer validation fold tests roughly eleven
tickets, and the rarest class (Removal) has four examples in total. Per-class
precision and recall therefore move in steps of 10–25 percentage points, and a
single reclassified ticket visibly moves macro-F1. Treat the figures as
indicative of order of magnitude, not as precise measurements, and do not
quote them to three decimal places as though they were stable.
"""
    )

    # ---- Limitations ------------------------------------------------------
    ui.section("Limitations")
    st.markdown(
        """
**TF-IDF has no understanding of meaning.** It measures lexical overlap
weighted by how distinctive each term is. A ticket that paraphrases a
requirement without reusing its vocabulary scores low and is misread as new
scope.

**The keyword lexicon is domain-specific.** The cues that work for a university
enrolment system will mislabel tickets in a trading platform or a medical
device. They are small, visible and editable in `driftguard/config.py`
precisely so you can replace them — not because they generalise.

**The thresholds were tuned on the bundled sample.** Any accuracy figure
measured on that same data is in-sample and optimistic. The cross-validated
score on the Model Performance page is the honest estimate for unseen tickets.

**Accuracy is a proxy for the thing that matters.** What a sponsor cares about
is whether misclassification moves the drift figure. That sensitivity is
reported directly rather than left to inference.

**Class imbalance makes accuracy flattering.** Removal is rare in any real
project, so macro-F1 leads and a majority-class baseline is shown beside every
score.

**Estimates inherit every upstream error.** Ticket sizes are human estimates,
the SLOC factor is an organisational guess, and COCOMO II's own published
accuracy is roughly within 30% of actuals 70% of the time under favourable
conditions. Treat the output as a structured argument about magnitude, not a
prediction.

**Scope is bounded** to a single project and English-language tickets.

**This is not an ML data-drift monitor.** DriftGuard measures scope change in a
software project. It does not compute population-stability or
Kolmogorov–Smirnov statistics over a tabular feature matrix, because it has no
such matrix — and inventing those numbers here would be fabrication.
"""
    )

    ui.section("About")
    st.markdown(
        f"""
**DriftGuard v{__version__}** — built for Northeastern University Toronto
coursework.

Stack: Python, Streamlit, pandas, NumPy, scikit-learn, Plotly. Tests with
pytest. No paid service, API key or cloud dependency in the execution path.

All analysis lives in the `driftguard` package rather than the dashboard, so
every number shown here can be reproduced headlessly and is covered by the
test suite.
"""
    )
