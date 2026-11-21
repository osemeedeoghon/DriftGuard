# DriftGuard

[![CI](https://github.com/osemeedeoghon/DriftGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/osemeedeoghon/DriftGuard/actions/workflows/ci.yml)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/osemeedeoghon/DriftGuard/actions/workflows/ci.yml)

Detects software project **scope creep** by comparing an agreed baseline of
requirements against project tickets, classifies each change, measures a
**Scope Drift Index**, and estimates what the drift costs in effort and
schedule using **COCOMO II Post-Architecture**.

Built as a local, run-on-demand application for Northeastern University Toronto
coursework.

> **Presenting this?** See **[PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md)**
> for the problem statement, architecture, evaluation methodology, a
> step-by-step demonstration sequence, key findings and limitations.
>
> **Assessing the engineering?** See **[docs/](docs/README.md)** — a full
> project audit, a requirements specification and a prioritised roadmap.

---

## Maturity

Stated plainly, because the documentation elsewhere depends on this being
honest.

**What is solid.** The analytical core is well covered and well separated:
**345 tests**, a `driftguard` package that imports no UI code and can be
driven headlessly, COCOMO II with published constants and no fabricated
calibration, and an evaluation layer that detects and reports leakage in its
own results rather than publishing the flattering number. Measured throughput
is 686 ms for 2,500 tickets against 300 requirements.

**What is not production-ready.** A November 2026 audit
([`docs/PROJECT_AUDIT.md`](docs/PROJECT_AUDIT.md)) found no durable
persistence, no audit logging, no packaging and no automated coverage of the
UI layer. Those gaps are catalogued with evidence and scheduled in
[`docs/DEVELOPMENT_ROADMAP.md`](docs/DEVELOPMENT_ROADMAP.md).

**Fixed since the audit.** The confirmed HTML-injection path from uploaded
ticket data is closed: untrusted text is escaped at every render boundary, and
a static check over the whole codebase fails CI if a new unescaped site appears
([ADR-0005](docs/adr/0005-html-escaping-policy.md)). Uploaded projects now
register properly, so project switching, scope history and report titles work
on the upload path. The suite runs in CI on every push across Python 3.11,
3.12 and 3.13, and runtime dependencies are pinned exactly, so a fresh clone
reproduces the environment the tests were verified against
([ADR-0006](docs/adr/0006-continuous-integration.md)). One new finding is open
and unfixed: CSV exports do not neutralise formula-leading cells (audit §7.6).

**What it has been validated on.** One synthetic project. DriftGuard is a
demonstrated method with a worked example — **not** a product validated
against real project data. The keyword lexicon is domain-specific and would
need re-tuning, plus a freshly labelled sample, before its accuracy on your
project could be claimed at all.

---

## Cost constraint

DriftGuard runs entirely on your machine. There is **no LLM API, no API key,
no cloud database and no paid hosting** anywhere in its execution path. Text
matching and classification use scikit-learn's TF-IDF and logistic regression
locally. The only cost is CPU time.

---

## Setup

Requires Python 3.11 or newer (developed and tested on 3.13; nothing in the
code is version-specific).

```bash
cd DriftGuard
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # runtime + test tooling
```

Check that `python3 --version` reports 3.11 or newer first. If your default
`python3` is older — or if you would rather build the environment on an
interpreter you already have the packages for — point `venv` at that
interpreter explicitly, for example:

```bash
/opt/homebrew/anaconda3/bin/python -m venv .venv
```

### Run the dashboard

```bash
./run.sh
```

Then press **Load sample project** in the sidebar. A full worked example — 18
baseline requirements and 45 tickets — is bundled, so the dashboard is usable
immediately after install with no data preparation.

`run.sh` selects an interpreter that actually has the dependencies installed —
a project-local `.venv` if present, otherwise an Anaconda install, otherwise
`python3` — so it works from any shell without activating anything first. The
direct equivalent, if your environment is already active, is:

```bash
streamlit run app.py
```

> **`zsh: command not found: streamlit`?**
> Your shell is using a Python that does not have the dependencies installed —
> commonly a system Python that takes PATH priority over Anaconda. Either
> activate the environment that does have them:
>
> ```bash
> conda activate base && streamlit run app.py
> ```
>
> or call the interpreter's own `streamlit` by full path, which needs no
> activation:
>
> ```bash
> /opt/homebrew/anaconda3/bin/streamlit run app.py
> ```
>
> Check which Python you are on with `which python3`. Using `./run.sh` avoids
> the issue entirely. The same applies to the tests:
> `/opt/homebrew/anaconda3/bin/python -m pytest tests`.

### Run the tests

```bash
python -m pytest tests -v
```

---

## The dashboard

Nine sections, navigable from the bar at the top:

| Section | What it shows |
|---|---|
| **Executive overview** | Scope Drift Index, full scope composition, effort/schedule/cost, risk indicators, recommended actions |
| **Dataset explorer** | Schema, completeness, missing values, duplicates, reference-vs-current size distributions |
| **Scope drift analysis** | The index with its arithmetic shown, cumulative timeline, COCOMO comparison, in-session scope history |
| **Scenarios** | Scope what-ifs, custom add/remove, sensitivity sweeps, driver tornado |
| **Model performance** | Accuracy, macro-F1, per-class metrics, confusion matrix, cross-validation, **nested validation with leakage detection**, learning curve |
| **Feature-level analysis** | Similarity distributions by class, threshold sensitivity, baseline coverage, per-prediction TF-IDF term weights |
| **Ticket review** | Search, filter, sort by scope impact, review queue, full classification audit, human override |
| **Reports and export** | PDF and Markdown reports plus CSV exports |
| **Methodology and about** | Equations, live methodology audit, evaluation integrity, assumptions, limitations |

Where a figure rests on an assumption, the assumption is shown beside it. Where
a claim cannot be supported — classifier accuracy with no labelled data, cost
with no labour rate, feature-drift statistics with no feature matrix — the
application says so instead of producing a number.

## What it does

1. **Upload a baseline** of agreed requirements from CSV or plain TXT.
2. **Upload project tickets** from CSV.
3. **Match** each ticket to its closest baseline requirement using local TF-IDF
   cosine similarity.
4. **Classify** each ticket as *In-scope*, *Refinement*, *New scope* or
   *Removal*.
5. **Show the evidence**: the matched requirement, the similarity score, the
   exact shared terms that drove the match, the runner-up requirements, and a
   written rationale for the classification.
6. **Accept human corrections** to any classification and confirmed size.
   Model predictions are kept separately from reviewer decisions.
7. **Measure the Scope Drift Index**, overall and cumulatively over time.
8. **Estimate COCOMO II effort and schedule** for the baseline and revised
   sizes.
9. **Attribute the drift** to the specific tickets responsible.
10. **Evaluate the classifiers** against a manually labelled test set:
    accuracy, per-class precision and recall, macro-F1, confusion matrix, and
    cross-validated out-of-sample scores.
11. **Validate the methodology**: nested cross-validation of the rule
    classifier's thresholds, automatic detection of residual data leakage, a
    learning curve, and a live audit that re-derives the pipeline's invariants.
12. **Explore assumptions** with sensitivity sweeps and scope scenarios, using
    the same COCOMO II implementation.
13. **Generate synthetic test projects** across five controlled scenarios,
    clearly marked as synthetic throughout.
14. **Track scope history** within a session and compare any two readings.
15. **Export** a professional PDF report, a Markdown report, and CSVs.

---

## Input formats

### Baseline requirements (CSV)

| Column | Required | Notes |
|---|---|---|
| `requirement_id` | yes | Unique. Aliases accepted: `Req ID`, `ID`. |
| `description` | yes | The requirement text. Aliases: `Requirement Text`, `Text`. |
| `title` | no | Prepended to the description for matching. |
| `size` | no | Story points by default. Aliases: `Story Points`, `Estimate`. |
| `category`, `priority` | no | Carried through for display. |

If `size` is absent, every requirement is given a uniform default and the
dashboard says so — a warning, not a silent assumption.

### Baseline requirements (TXT)

One requirement per line. Blank lines ignored. Bullet markers (`-`, `*`,
`1.`) are stripped, and a leading identifier such as `R-07:` is reused as the
requirement id. A text file carries no sizes, so all requirements receive a
uniform default size; this is reported prominently.

### Project tickets (CSV)

| Column | Required | Notes |
|---|---|---|
| `ticket_id` | yes | Aliases: `Issue Key`, `Key`, `ID`. |
| `description` | yes | The ticket text. |
| `title` | no | Prepended to the description for matching. |
| `size` | no | Same unit as the baseline. |
| `created_date` | no | Needed for the drift-over-time chart. |
| `true_class` | no | Manual label; makes the file usable as a test set. |

Column names are normalised, so `Ticket ID`, `ticket_id` and `TicketId` all
work.

---

## How classification works

The two signals do different jobs, deliberately:

- **Cosine similarity** answers *"does this ticket concern the agreed baseline
  at all?"* It is good at that: an unrelated ticket shares almost no
  distinctive vocabulary with any requirement.
- **Keyword cues** answer *"does it implement, change, or withdraw that
  requirement?"* Similarity **cannot** answer this. A ticket implementing a
  requirement and a ticket rewriting it look nearly identical to a
  bag-of-words model.

Decision order (first rule that fires wins):

| # | Condition | Class |
|---|---|---|
| 1 | Scope-withdrawal language | Removal |
| 2 | similarity < weak threshold (0.18) | New scope |
| 3 | Modification language present | Refinement |
| 4 | New-work language present | New scope |
| 5 | Otherwise | In-scope (flagged for review below 0.45) |

Removal detection is two-tier on purpose. Unambiguous phrases ("de-scoped",
"no longer required") fire alone, but generic verbs ("remove", "withdraw",
"cancel") only count when a scope noun appears alongside them. Without that
guard, ordinary domain vocabulary — *"administrators can withdraw a course
section"*, *"email the student when an enrolment is withdrawn"* — reads as
de-scoping. That single fix accounted for a large share of the classifier's
accuracy.

A scikit-learn logistic regression over TF-IDF features plus the similarity
score is also provided as a learned baseline, and the evaluation page scores
both.

---

## The Scope Drift Index

```
SDI = (cumulative size added − cumulative size removed) / baseline size
```

| Class | Contribution |
|---|---|
| In-scope | nothing — already inside the baseline |
| Refinement | `w × size`, where `w` is the refinement weight (default 0.5) |
| New scope | full size added |
| Removal | full size removed |

The revised size follows directly: `revised = baseline × (1 + SDI)`.

SDI is dimensionless, so story points, function points or SLOC all work —
provided the baseline and the tickets use the **same** unit.

---

## COCOMO II Post-Architecture

```
E    = B + 0.01 × Σ SF_j                 (5 scale factors)
PM   = A × Size^E × Π EM_i               (17 effort multipliers)
F    = D + 0.2 × (E − B)
TDEV = C × PM_NS^F × (SCED% / 100)
```

with the published COCOMO II.2000 calibration `A = 2.94`, `B = 0.91`,
`C = 3.67`, `D = 0.28`; `Size` in KSLOC; `PM_NS` the effort excluding the SCED
multiplier.

**DriftGuard performs no local calibration and fabricates no calibration
data.** Constant `A` is the published nominal value. An organisation using
this for real planning must re-calibrate it against its own completed-project
history.

### Explicit assumptions

Every one of these is adjustable in the sidebar and labelled as an assumption:

| Assumption | Default | Status |
|---|---|---|
| SLOC per story point | 60 | **Placeholder.** Replace with your own historical figure. |
| SLOC per function point | 53 | Common 3GL gearing ratio; for Python, 20–30 is more realistic. |
| Refinement weight | 0.5 | Deliberate midpoint; no empirical basis. |
| All scale factors and effort multipliers | Nominal | Nominal is an assumption, not a measurement of your project. |
| Person-month | 152 hours | COCOMO convention. |
| Similarity thresholds | 0.18 / 0.45 | Tuned on the bundled sample; re-tune on your own data. |

### Effort is not duration

`PM` is **effort** — person-months of work. `TDEV` is **calendar duration** —
elapsed months. They are different quantities, reported separately throughout,
and they do not convert into one another by adding headcount. `PM / TDEV`
gives the average staffing the estimate implies, not a schedule you can
compress at will.

### Cost

DriftGuard reports effort and duration only. It shows a currency figure **only
if you supply an hourly or monthly labour rate**, and when it does, it labels
the result a derived labour-cost *estimate* — not an actual or committed cost,
and exclusive of tooling, licences, overhead and contingency.

### REVL, and a trap worth naming

COCOMO II's REVL factor inflates size to account for requirements evolution:
`Size_effective = Size × (1 + REVL/100)`.

The obvious implementation — feed the Scope Drift Index straight into REVL —
is **wrong**, and DriftGuard originally did it. The revised size *already*
contains the newly added scope, so inflating it again by the same percentage
charges for that work twice. On the sample project it reported +246% effort
for +76% size growth.

What REVL actually models is **rework**. DriftGuard therefore computes:

```
REVL% = 100 × (refinement-driven additions + removals) / baseline size
```

Refinements are requirements that moved after being agreed — the textbook
definition of volatility. Removals are work that was planned, possibly partly
built, then dropped: the size leaves the revised total but the effort spent
does not come back. New scope is deliberately excluded, because the larger
revised size already carries it.

A regression test (`test_revl_excludes_new_scope_because_revised_size_already_carries_it`)
exists specifically to stop that double-count returning.

---

## Limitations of similarity-based classification

These are real and worth stating plainly.

**TF-IDF has no understanding of meaning.** It measures lexical overlap
weighted by how distinctive each term is. A ticket that paraphrases a
requirement without reusing its vocabulary scores low and is misread as new
scope. Both residual errors on the sample project are exactly this: a login
ticket and a waitlist-notification ticket that sit just below the similarity
floor.

**Keyword cues are brittle and domain-specific.** The lexicon that works for a
university enrolment system will mislabel tickets in a trading platform or a
medical device. The cue lists are small, visible and editable in
`driftguard/config.py` precisely so you can replace them — not because they
generalise.

**The thresholds are tuned on the bundled sample.** The headline 95.6%
accuracy is therefore **in-sample and optimistic**. The honest figure for
unseen tickets is the cross-validated score the evaluation page reports for
the learned model: about **75.6% accuracy, macro-F1 0.563** on the same data.
That gap is the cost of fitting to a single project's vocabulary, and the
dashboard says so rather than hiding it.

**Accuracy is a proxy for the thing that matters.** What a project sponsor
cares about is whether misclassification moves the drift figure. The
evaluation page reports this directly: on the sample project the predicted SDI
is +75.8% against a true +72.9%, so the classifier **overstates** drift by 2.9
percentage points. A classifier can be mediocre on accuracy and still produce
a usable drift number, or accurate and still produce a misleading one.

**Class imbalance makes accuracy flattering.** Removal is rare in any real
project. A model that never predicts it can still post high accuracy, which is
why macro-F1 is the headline metric and why a majority-class baseline is shown
alongside every score.

**The size estimates inherit every upstream error.** Ticket sizes are human
estimates, the SLOC-per-story-point factor is an organisational guess, and
COCOMO II's own published accuracy is roughly within 30% of actuals 70% of the
time under good conditions. Treat the output as a structured argument about
magnitude, not a prediction.

**Nested cross-validation exposed leakage rather than resolving it.** A nested
run returned a *perfect* held-out macro-F1 of 1.000 against 0.960 in-sample.
Held-out performance cannot legitimately beat training performance, so the tool
flags this as a leakage indicator rather than reporting it as success. The
mechanism is measurable: lowering the weak-match threshold shifts decisions
from the un-leaked similarity signal onto the hand-written keyword lexicon
(33 tickets to 40, across folds). Because that lexicon was written after
reading these tickets, a perfect fold score reflects memorisation. The shipped
threshold of 0.18 is deliberately the more conservative choice over the 0.10
the search prefers. **Only an independently labelled corpus from another
project can give the rule classifier a clean estimate.**

**The learned classifier is limited by the method, not the sample size.** Its
learning curve is flat at macro-F1 ~0.56 between 13 and 33 training examples.
More labelled tickets of the same kind would not lift it.

**Projects and scope history are held in memory only.** Closing the tab
discards them. Durable persistence would need a datastore, which would change
the zero-infrastructure deployment story; it is deferred rather than
half-implemented.

**Severity bands and the high-impact threshold are reading conventions** for
this dashboard (the latter being 5% of baseline size), not published standards.

**Scope is deliberately bounded** to a single project at a time and
English-language tickets.

---

## Project layout

```
DriftGuard/
├── app.py                       Router only
├── run.sh                       Launcher that finds a working interpreter
├── requirements.txt
├── README.md
├── PRESENTATION_GUIDE.md        Demonstration guide + 10-minute script + Q&A
├── .streamlit/config.toml       Pinned light theme for a deterministic demo
├── driftguard/                  All analysis — independently testable
│   ├── config.py                Class taxonomy, cue lexicons, thresholds, palette
│   ├── data.py                  Loading and validation of CSV/TXT inputs
│   ├── matching.py              TF-IDF matcher, match evidence, term weights
│   ├── classification.py        Rule and learned classifiers, overrides
│   ├── drift.py                 Scope Drift Index, timeline, REVL bridge
│   ├── cocomo.py                COCOMO II Post-Architecture model
│   ├── features.py              Coverage, health, risk indicators, recommendations
│   ├── evaluation.py            Metrics, confusion matrix, cross-validation
│   ├── validation.py            Nested CV, leakage detection, methodology audit
│   ├── scenarios.py             Sensitivity sweeps and scenario costing
│   ├── synthetic.py             Controlled synthetic project generator
│   ├── projects.py              Multi-project store and in-session scope history
│   ├── reporting.py             Markdown report
│   ├── pdf_report.py            PDF report (ReportLab + Matplotlib)
│   ├── pipeline.py              End-to-end orchestration
│   └── sample_data/             Bundled worked example
├── ui/                          Presentation only — computes no figure
│   ├── theme.py                 Design tokens, Plotly template, formatting
│   ├── components.py            Metric cards, sections, empty states, quality
│   ├── charts.py                Chart builders, each with a table view
│   ├── sidebar.py               Control panel, project switching, generator
│   ├── state.py                 Session state and project management
│   ├── welcome.py               Landing screen
│   └── pages/                   One module per section
└── tests/                       300 tests
```

All analysis lives in the package, not the dashboard, so every number shown in
the UI can be reproduced headlessly:

```python
from driftguard import data, pipeline

baseline, tickets = data.sample_baseline(), data.sample_tickets()
result = pipeline.run_analysis(baseline, tickets)
print(f"SDI {result.drift.sdi:+.1%}")
print(f"Effort {result.estimates.revised.person_months:.1f} person-months")
```

---

## Sample project

A fictional *Campus Course Registration Portal*: 18 agreed requirements
totalling 157 story points, and 45 tickets spanning all four classes.

Results with default settings:

| Measure | Value |
|---|---|
| Scope Drift Index | **+75.8%** |
| Size | 157 → 276 story points |
| Effort | 34.6 → 84.1 person-months (+142.7%) |
| Duration | 11.3 → 15.0 calendar months (+32.6%) |
| Rule classifier | 95.6% accuracy, macro-F1 0.960 (in-sample) |
| Learned classifier, cross-validated | 75.6% accuracy, macro-F1 0.563 |
| Drift figure error | +2.9 percentage points |

The drift is deliberately dramatic so the dashboard has something to show. A
real project at +76% would be in serious trouble.

Note that effort grows faster than size (+142.7% against +75.8%). That is not
a bug: COCOMO II's exponent `E = 1.0997` exceeds 1, encoding the diseconomy of
scale, and the churn-based REVL adds rework on top.

---

## Testing

300 tests covering the drift calculations and COCOMO II formulas as required,
plus loading, validation, matching, classification, overrides, metrics,
nested validation, sensitivity analysis, scenario costing, the synthetic
generator, the project store, report generation and the presentation helpers.

```bash
python -m pytest tests -v
```

Expected COCOMO values are derived from the published equations inside the
tests themselves, so a regression in the module cannot quietly rewrite the
yardstick it is measured against. Bugs found this way during development:

- the SCED effort multiplier conflated with the SCED schedule percentage;
- the REVL double-count described above;
- percentage columns formatted from raw fractions, so 95.6% rendered as 1.0%;
- 8-digit hex passed to Plotly fill properties, which reject it;
- `AnalysisResult.classified` missing the `net_contribution` column, so a risk
  indicator silently reported zero;
- a high-impact threshold defined as a quantile, so reviewing a large ticket
  *raised* the count of unreviewed high-impact tickets;
- `elasticity` computed as an arc ratio (1.24) rather than in logs (1.0997),
  implying a COCOMO exponent the model does not have;
- the synthetic CSV export prefixed with a `#` notice line, which made the
  generated file unreadable by `read_csv` on re-upload.

Three tests exist specifically to stop a prior bug returning:
`test_revl_excludes_new_scope_because_revised_size_already_carries_it`,
`test_reviewing_a_high_impact_ticket_never_raises_the_unreviewed_count`, and
`test_elasticity_uses_logs_not_an_arc_ratio`.
