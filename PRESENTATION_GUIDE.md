# DriftGuard — Presentation Guide

Companion for the live academic demonstration: what the project is, how it
works, what to show, what to say, and what it cannot do.

---

## Contents

1. [Project overview](#1-project-overview)
2. [Problem statement](#2-problem-statement)
3. [System architecture](#3-system-architecture)
4. [Technology stack](#4-technology-stack)
5. [Dataset description and sources](#5-dataset-description-and-sources)
6. [Data-processing methodology](#6-data-processing-methodology)
7. [Classification methodology](#7-classification-methodology)
8. [SDI calculation](#8-sdi-calculation)
9. [REVL reasoning](#9-revl-reasoning)
10. [COCOMO II estimation](#10-cocomo-ii-estimation)
11. [Model evaluation](#11-model-evaluation)
12. [Results and findings](#12-results-and-findings)
13. [Limitations](#13-limitations)
14. [Future improvements](#14-future-improvements)
15. [Installation and execution](#15-installation-and-execution)
16. [Demonstration sequence](#16-demonstration-sequence)
17. [Ten-minute presentation script](#17-ten-minute-presentation-script)
18. [Anticipated questions](#18-anticipated-questions)
19. [Pre-demonstration checklist](#19-pre-demonstration-checklist)

---

## 1. Project overview

**DriftGuard** detects software project **scope creep** by comparing an agreed
baseline of requirements against the tickets actually raised. It classifies
each ticket's relationship to the baseline, explains every classification,
accepts human correction, quantifies net change as a **Scope Drift Index**,
and translates that change into effort and schedule using **COCOMO II
Post-Architecture**. It also supports sensitivity analysis, scenario planning,
in-session scope history, and professional report export.

It runs entirely on a local machine. There is no LLM API, no API key, no cloud
database and no paid hosting anywhere in its execution path.

| | |
|---|---|
| **Scale** | ~13,600 lines across 33 modules |
| **Tests** | **300**, pytest |
| **Sections** | 9 |
| **Launch** | `./run.sh` |
| **Scope** | One project at a time, English-language tickets |

---

## 2. Problem statement

Scope creep is usually diagnosed late and argued about anecdotally. A sponsor
says "this feels bigger than we agreed"; a delivery lead says "most of this was
always implied". Neither side has a number.

The evidence is **unstructured text**. The agreement lives in a requirements
document; the reality lives in a ticket tracker. Nobody maintains a mapping
between them, so four questions go unanswered:

1. **Which tickets fall outside the agreed baseline?** Reading hundreds of
   tickets against dozens of requirements by hand does not scale.
2. **How much scope has changed?** Without a normalised measure, "a lot" is
   the best anyone manages.
3. **What does that change cost?** Scope growth is only actionable when
   expressed as effort and schedule a sponsor can trade against.
4. **Why should anyone believe the answer?** A number without an audit trail
   loses the argument in the room.

DriftGuard answers all four, and states clearly where its answers are
assumptions rather than measurements.

---

## 3. System architecture

```
DriftGuard/
├── app.py                      Router only (~130 lines)
├── run.sh                      Launcher that finds a working interpreter
├── driftguard/                 All analysis — independently testable
│   ├── config.py               Taxonomy, cue lexicons, thresholds, palette
│   ├── data.py                 CSV/TXT loading and validation
│   ├── matching.py             TF-IDF matcher, evidence, per-term weights
│   ├── classification.py       Rule and learned classifiers, overrides
│   ├── drift.py                Scope Drift Index, timeline, REVL bridge
│   ├── cocomo.py               COCOMO II Post-Architecture
│   ├── features.py             Coverage, health, risk, recommendations
│   ├── evaluation.py           Metrics, confusion matrix, cross-validation
│   ├── validation.py           Nested CV, leakage detection, audit
│   ├── scenarios.py            Sensitivity sweeps and scenario costing
│   ├── synthetic.py            Controlled test-data generator
│   ├── projects.py             Multi-project store and scope history
│   ├── reporting.py            Markdown report
│   ├── pdf_report.py           PDF report
│   └── pipeline.py             End-to-end orchestration
├── ui/                         Presentation only — computes no figure
│   ├── theme.py                Design tokens, Plotly template, formatting
│   ├── components.py           Cards, sections, empty states, quality panels
│   ├── charts.py               Chart builders, each with a table view
│   ├── sidebar.py / state.py / welcome.py
│   └── pages/                  One module per section
└── tests/                      300 tests
```

**The separation is load-bearing.** Every number the dashboard displays can be
reproduced headlessly, which is what makes it testable:

```python
from driftguard import data, pipeline
result = pipeline.run_analysis(data.sample_baseline(), data.sample_tickets())
print(f"SDI {result.drift.sdi:+.1%}")
```

**Nine sections:** Executive overview · Dataset explorer · Scope drift analysis
· Scenarios · Model performance · Feature-level analysis · Ticket review ·
Reports and export · Methodology and about.

---

## 4. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ (tested 3.13) | Required; nothing version-specific |
| UI | Streamlit | Pure-Python dashboards, no JS build step |
| Data | pandas, NumPy | Required |
| ML | scikit-learn | TF-IDF, logistic regression, stratified CV |
| Charts (screen) | Plotly | Interactive, works inside Streamlit |
| Charts (print) | Matplotlib | Already present; avoids adding Kaleido + headless Chromium |
| PDF | ReportLab | Standard pure-Python PDF writer, offline |
| Tests | pytest | Required |

**No paid service, API key or cloud dependency appears anywhere.** The only
runtime cost is CPU time. `pymupdf` is a test-only dependency used to read the
generated PDF back; the application never imports it.

---

## 5. Dataset description and sources

### Bundled worked example

A **synthetic** project written for this coursework: *Campus Course
Registration Portal*, a fictional student enrolment system. It is **not** real
institutional data and makes no claim to represent any actual project.

| | |
|---|---|
| Baseline | 18 requirements, 157 story points |
| Tickets | 45, spanning all four classes |
| Labels | All 45 manually labelled |
| Location | `driftguard/sample_data/` |

Class distribution: 20 In-scope, 10 Refinement, 11 New scope, 4 Removal. The
imbalance is deliberate — Removal is rare in real projects, and that imbalance
is exactly what makes accuracy a misleading headline metric.

### Synthetic generator

`driftguard/synthetic.py` produces controlled projects across five scenarios:
**normal**, **new_scope_surge**, **refinement_heavy**, **removals** and
**dirty** (missing sizes, blank descriptions, unparseable dates, duplicate
rows). Generation is reproducible from a seed.

Everything generated is marked synthetic in three places: a `synthetic` column
that survives a CSV round trip, `SYN-` prefixed identifiers, and a banner on
every dashboard page. The loaders warn when they see the marker, so a file
re-uploaded months later still announces itself.

### Your own data

A baseline CSV (or plain TXT, one requirement per line) and a tickets CSV.
Column names are normalised, so `Ticket ID`, `ticket_id` and `Issue Key` all
work. Schema is documented in the README.

---

## 6. Data-processing methodology

| Stage | What happens |
|---|---|
| **Load** | CSV/TXT parsed; column names normalised against an alias table; sizes coerced and validated; dates parsed |
| **Validate** | Duplicate ids dropped with a warning; blank descriptions flagged; non-numeric sizes defaulted and reported; negative sizes rejected outright; unknown class labels rejected |
| **Match** | TF-IDF vectors (unigrams); cosine similarity of every ticket against every requirement; top-3 candidates retained with shared terms |
| **Classify** | Similarity thresholds plus a visible keyword lexicon, or logistic regression |
| **Review** | A human confirms or overrules any classification and size; predictions preserved separately |
| **Measure** | Scope Drift Index from net added and removed size |
| **Estimate** | COCOMO II effort and schedule for baseline and revised sizes |

Failures **degrade rather than crash**: a dirty file produces warnings and a
completed analysis, not a traceback. This is covered by
`test_dirty_scenario_still_completes_an_analysis`.

---

## 7. Classification methodology

Four classes: **In-scope**, **Refinement**, **New scope**, **Removal**.

The two signals do different jobs, deliberately:

- **Cosine similarity** answers *"does this ticket concern the baseline at
  all?"* It is good at that, because an unrelated ticket shares almost no
  distinctive vocabulary with any requirement.
- **Keyword cues** answer *"does it implement, change, or withdraw that
  requirement?"* Similarity **cannot** answer this: a ticket implementing a
  requirement and a ticket rewriting it look nearly identical to a
  bag-of-words model.

Decision order (first rule wins):

| # | Condition | Class |
|---|---|---|
| 1 | Scope-withdrawal language | Removal |
| 2 | similarity < 0.18 | New scope |
| 3 | Modification language present | Refinement |
| 4 | New-work language present | New scope |
| 5 | Otherwise | In-scope (flagged for review below 0.45) |

**Removal detection is two-tier**, and this matters. Unambiguous phrases
("de-scoped", "no longer required") fire alone, but generic verbs ("remove",
"withdraw", "cancel") count only when a scope noun appears alongside them.
Without that guard, ordinary domain language — *"administrators can withdraw a
course section"* — reads as de-scoping. Fixing this single issue took the
classifier from 60% to 95.6%.

**TF-IDF is not a language model.** It weights vocabulary by distinctiveness
and understands nothing. The dashboard says so wherever similarity appears.

### Explainability

For any ticket, the Ticket Review page shows: identifier, title, description,
original model prediction, final classification, matched requirement,
similarity with its band, margin over the runner-up, the shared TF-IDF terms,
which keyword cues fired and what each signals, the written rationale, and
whether a human overrode the decision.

A useful demo case is **T-203**: refinement cues *did* fire, but similarity
(0.172) fell below the weak threshold, so rule 2 won and it was classified New
scope. The panel shows the cues *and* notes they did not decide the ticket.

---

## 8. SDI calculation

```
SDI = (cumulative size added − cumulative size removed) ÷ baseline size
revised size = baseline × (1 + SDI)
```

| Class | Contribution |
|---|---|
| In-scope | nothing — already inside the baseline |
| Refinement | `w × size`, default `w = 0.5` (an assumption) |
| New scope | full size added |
| Removal | full size removed |

SDI is dimensionless, so story points, function points or SLOC all work,
provided baseline and tickets use the **same** unit. The Scope Drift Analysis
page shows the arithmetic with the actual numbers substituted, so the figure is
derived in front of the audience rather than asserted.

---

## 9. REVL reasoning

COCOMO's REVL factor inflates size to account for requirements volatility:
`Size_effective = Size × (1 + REVL/100)`.

The obvious implementation — feed the Scope Drift Index straight into REVL —
is **wrong**, and an earlier version of this tool did exactly that, reporting
**+246% effort for +76% size growth**. The revised size *already contains* the
newly added scope, so inflating it again by the same percentage charges for
that work twice.

What REVL models is **rework**. DriftGuard therefore computes:

```
REVL% = 100 × (refinement-driven additions + removals) ÷ baseline size
```

- **Refinements** are requirements that moved after being agreed — the textbook
  definition of volatility.
- **Removals** are work that was planned, possibly partly built, then dropped:
  the size leaves the revised total but the effort spent does not come back.
- **New scope** is excluded, because the larger revised size already carries it.

On the sample project this gives **27.4%**, against 75.8% under the naive
reading. A named regression test
(`test_revl_excludes_new_scope_because_revised_size_already_carries_it`) exists
specifically to stop the double-count returning, and the Methodology page
re-derives the figure live.

**Known weakness, state it before you are asked:** removals are charged at
their full size, which assumes the work was built before being dropped. Where
removals were cancelled before starting, this overstates rework. The REVL link
can be switched off in the sidebar to see the pure size effect.

---

## 10. COCOMO II estimation

```
E    = B + 0.01 × Σ SF_j                 (5 scale factors)
PM   = A × Size^E × Π EM_i               (17 effort multipliers)
F    = D + 0.2 × (E − B)
TDEV = C × PM_NS^F × (SCED% ÷ 100)
```

Published COCOMO II.2000 calibration: `A = 2.94`, `B = 0.91`, `C = 3.67`,
`D = 0.28`. Size in KSLOC; `PM_NS` excludes the SCED multiplier. **No local
calibration is performed and no calibration data is fabricated.**

Three distinctions the demo should make explicitly:

- **Effort ≠ duration.** `PM` is person-months of work; `TDEV` is elapsed
  calendar months. They do not convert into one another by adding staff.
- **No cost without a rate.** A currency figure appears only when the user
  supplies a labour rate, and is always labelled a derived estimate.
- **SCED has two tables.** An effort multiplier *and* a schedule percentage.
  Compressing a schedule raises effort and shortens the calendar; stretching it
  extends the calendar at no effort penalty. Conflating the two was a real bug
  found by the tests.

### Sensitivity and scenarios

The Scenarios page sweeps the documented assumptions through the same COCOMO
implementation — no alternative formula is introduced. It reports **elasticity
in logs**, which recovers the exponent exactly: effort elasticity with respect
to SLOC-per-point is **1.0997 = E**, and duration elasticity is
**0.3496 = E × F**. (An arc ratio over the same sweep reads 1.24 and would
imply an exponent the model does not have.)

It also offers five scope scenarios — two recorded, three hypothetical, always
visually distinguished — a custom add/remove scenario, and a tornado chart
ranking which COCOMO drivers move the estimate most (CPLX leads, with a swing
of about one nominal project's worth of effort).

---

## 11. Model evaluation

Evaluation is against the manually labelled test set only. **Where no labels
exist, the application claims nothing** — Model Performance shows an empty
state rather than a number.

| Metric | Why it is reported |
|---|---|
| **Macro-F1** | Headline. Unweighted across classes, so rare Removal counts as much as common In-scope |
| Accuracy | Reported, never alone |
| Majority baseline | What "always guess the commonest class" scores — anything at or below it has learned nothing |
| Per-class precision/recall | Where the model fails, not just how often |
| Confusion matrix | Which classes get confused for which |
| **Drift sensitivity** | Whether the errors actually move the SDI |
| Cross-validated (learned model) | Stratified k-fold |
| **Nested CV (rule classifier)** | Thresholds tuned on inner folds, scored on a held-out fold |
| Learning curve | Whether the ceiling is the method or the data |

### Training, validation and testing

- *Training data* fits a model's parameters.
- *Validation data* selects between models or settings — thresholds, here.
- *Test data* is touched once, to estimate performance on unseen work.

DriftGuard's 45 labelled tickets have served **all three roles**. That is the
central methodological weakness of the project and should be stated first in
any defence of it.

| Component | Tuned on | Honest estimate available? |
|---|---|---|
| TF-IDF vectoriser | whole corpus, no labels | Leaks no ground truth, but evaluation is *transductive* |
| Similarity thresholds | the 45 labelled tickets | Yes — nested cross-validation |
| Keyword lexicon | the 45 tickets, by hand | **No** — cannot be re-derived per fold |
| Logistic regression | per fold | Yes — stratified cross-validation |

---

## 12. Results and findings

### Headline figures (sample project, default settings)

| Measure | Value |
|---|---|
| Scope Drift Index | **+75.8%** (157 → 276 story points) |
| Size added / removed | 148 / 29 story points |
| Effort | 34.6 → 84.1 person-months (**+142.7%**) |
| Duration | 11.3 → 15.0 calendar months (**+32.6%**) |
| Implied staffing | 3.1 → 5.6 FTE |
| REVL (churn only) | 27.4% |
| Rule classifier, in-sample | 95.6% accuracy, macro-F1 0.960 |
| Majority-class baseline | 44.4% |
| Learned classifier, cross-validated | 75.6% accuracy, macro-F1 0.563 |
| Drift figure error | **+2.9 percentage points** (overstates growth) |
| Methodology audit | 8 of 8 checks pass |

### Findings

1. **The dominant error source was domain vocabulary, not the model.** Generic
   withdrawal verbs collided with enrolment-domain language. Guarding them with
   a scope-noun requirement moved accuracy from 60% to 95.6% — a lexicon fix,
   not a modelling one.

2. **Unigrams beat unigrams-plus-bigrams.** Tickets paraphrase requirements
   rather than quoting them, so bigram overlap is sparse and the extra
   zero-weight features dilute the signal. Median in-scope similarity rose from
   0.39 to 0.58.

3. **Similarity and keywords answer different questions.** Similarity separates
   related from unrelated (In-scope median 0.60 vs New scope 0.12) but cannot
   separate implementing from rewriting. Treating them as one signal was the
   original design error.

4. **REVL double-counting is an easy, invisible mistake.** Feeding the drift
   index into a size that already contains the added scope inflated effort to
   +246%. The corrected churn-only figure is +143%.

5. **Effort grows faster than scope.** +75.8% scope produced +142.7% effort but
   only +32.6% duration — exponent `E = 1.0997` plus 27.4% churn rework.
   Schedule is a poor proxy for cost.

6. **Classifier error barely moved the decision.** Despite 2 misclassifications
   in 45, the reported SDI was off by only 2.9 points, in the conservative
   direction. Model accuracy and decision quality are different things.

7. **Nested cross-validation exposed residual leakage rather than resolving
   it** — the most interesting finding in the project. Nested CV returned a
   *perfect* held-out macro-F1 of 1.000 against 0.960 in-sample. Held-out
   performance cannot legitimately exceed training performance, so the tool
   flags it as a leakage indicator rather than reporting it as success.
   The mechanism is measurable: the threshold search lowers the weak-match
   boundary, which shifts decisions from the un-leaked similarity signal
   (12 → 5 tickets) onto the hand-written lexicon (33 → 40). Because that
   lexicon was written after reading these very tickets, a perfect fold score
   reflects memorisation. **The shipped threshold of 0.18 is deliberately the
   more conservative choice over the 0.10 the search prefers.**

8. **The learned classifier is limited by the method, not the sample.** The
   learning curve is flat at macro-F1 ≈ 0.56 from 13 to 33 training examples.
   More labelled tickets of the same kind would not lift it; a richer
   representation would be the place to look.

---

## 13. Limitations

- **TF-IDF has no understanding of meaning.** A ticket that paraphrases a
  requirement without reusing its vocabulary is misread as new scope. Both
  residual errors on the sample are exactly this.
- **The keyword lexicon is domain-specific** and will not transfer unchanged.
- **Thresholds were tuned on the evaluation data**, so in-sample scores are
  optimistic — and nested CV cannot fully repair this, because the lexicon
  leakage is irreducible without a freshly labelled corpus.
- **45 labelled tickets is a small sample.** Each outer fold tests about
  eleven tickets and the rarest class has four examples in total, so per-class
  metrics move in steps of 10–25 points. Do not quote them to three decimals as
  though they were stable.
- **Class imbalance makes accuracy flattering** — hence macro-F1 as headline.
- **Every estimate inherits the SLOC conversion factor**, an organisational
  guess. At half the factor, every estimate halves.
- **COCOMO II's own accuracy** is roughly within 30% of actuals 70% of the time
  under favourable conditions.
- **Ticket sizes are human estimates**, with all the bias that implies.
- **Removals are charged as full rework** in REVL, overstating churn where work
  was cancelled before starting.
- **Severity bands and the 5%-of-baseline high-impact threshold are reading
  conventions** for this dashboard, not published standards.
- **Projects and scope history are held in memory only.** Closing the tab
  discards them.
- **Single project at a time, English only.**
- **This is not an ML data-drift monitor.** It measures scope change in a
  software project. It computes no population-stability or
  Kolmogorov–Smirnov statistics, because there is no tabular feature matrix —
  and inventing those numbers would be fabrication.

---

## 14. Future improvements

| Improvement | Why |
|---|---|
| **Independently labelled corpus from a second project** | The only thing that gives the rule classifier a clean out-of-sample estimate. Highest-value next step by a distance. |
| Local sentence embeddings (small offline sentence-transformer) | Would catch paraphrase that TF-IDF misses, while staying offline and free |
| Calibrate SLOC-per-story-point from completed projects | Turns the single largest assumption into a measurement |
| Durable persistence (SQLite) for projects and history | Currently in-session only; deliberately deferred to avoid changing the zero-infrastructure deployment story |
| Removal rework fraction as a parameter | Removals are currently all-or-nothing in REVL |
| Re-calibrate COCOMO constant `A` on organisational history | The published value is nominal |
| Active learning from the review queue | Reviewer corrections inform the drift figure but do not retrain the model |
| Jira / Azure DevOps ingestion | Removes manual CSV export from the workflow |

---

## 15. Installation and execution

```bash
cd DriftGuard
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./run.sh
```

`run.sh` selects an interpreter that actually has the dependencies installed —
a project-local `.venv` if present, otherwise an Anaconda install, otherwise
`python3` — so it works from any shell without activating anything first.

Tests:

```bash
python -m pytest tests
```

---

## 16. Demonstration sequence

**Setup:** run `./run.sh`, click **Load sample project**, leave it on Executive
Overview. Allow about 12 minutes including questions.

| # | Section | Say this | Show this |
|---|---|---|---|
| 1 | **Executive overview** | "157 story points agreed. Now tracking 276 — a 76% overrun." | Hero SDI +75.8%, Critical pill |
| 2 | | "Here is the full position: in-scope, refinement, new scope, removals, pending reviews." | Detail metric rows |
| 3 | | "And the risks, each with the basis it was measured from." | Risk indicator table |
| 4 | | "With recommended actions that follow from measured figures, not guesses." | Recommendation cards |
| 5 | **Dataset explorer** | "Two inputs, their completeness, and how their sizes compare." | Reference-vs-current box plots, quality panel |
| 6 | **Scope drift analysis** | "The index is derived, not asserted — here is the arithmetic." | SDI formula with real numbers |
| 7 | | "Stable through January and February, then four large unplanned requests in March." | Stepped drift timeline |
| 8 | | "And I can snapshot a reading, change something, and see what moved." | Record snapshot → compare |
| 9 | **Scenarios** | "What if we decline all new scope? Hypotheticals are hatched so they can't be mistaken for records." | Scenario comparison |
| 10 | | "The SLOC factor is a guess, so here's how much it matters. Elasticity 1.0997 — that's the COCOMO exponent, recovered from the sweep." | Sensitivity curve |
| 11 | **Model performance** | "95.6% — but in-sample, and I want to be explicit about that." | Warning banner |
| 12 | | **"The interesting result is here. Nested CV returned a *perfect* held-out score. That's not good news — held-out can't beat in-sample. The tool flags it as leakage and shows the mechanism."** | Leakage verdict + cue-reliance table |
| 13 | | "And the learning curve is flat, so more labels wouldn't help — the method is the ceiling." | Learning curve |
| 14 | **Feature-level analysis** | "The features are TF-IDF terms. In-scope sits at 0.60 similarity, new scope at 0.12." | Similarity box plots |
| 15 | **Ticket review** | "Nothing is a black box — here is every input to one decision, including which cue words fired." | Classification audit for T-203 |
| 16 | | "A reviewer overrules it, and the model's original prediction is kept so evaluation still measures the model." | Override → back to Overview |
| 17 | **Reports** | "The whole analysis exports as a PDF with assumptions and limitations attached." | Generate PDF |
| 18 | **Methodology** | "And every claim the tool makes about itself is re-derived live, on your data." | Audit — 8 of 8 pass |

### Live override demo (step 16, worth rehearsing)

1. Ticket review → select **T-301** (Native mobile application, 21 points, New
   scope).
2. Change to **In-scope**, press **Save**.
3. Return to **Executive overview**: SDI drops 75.8% → 62.4%, revised size
   276 → 255.
4. Point out that **Model said: New scope** is still recorded.

---

## 17. Ten-minute presentation script

Timings are cumulative. Rehearse to 10:00; questions follow.

### Introduction — 0:00–0:45

> "I'm Oseme Edeoghon. This is DriftGuard, a tool for detecting and costing
> scope creep in software projects. It compares what a project agreed to build
> against what its tickets say it's actually building, and turns the difference
> into an effort and schedule number a sponsor can act on. Everything runs
> locally — there's no API key and no cloud service anywhere in it."

### Problem statement — 0:45–2:00

> "Scope creep gets diagnosed late and argued about anecdotally. The sponsor
> says it feels bigger than we agreed; the delivery lead says most of it was
> always implied. Neither has a number.
>
> The reason is that the evidence is unstructured text. The agreement lives in
> a requirements document; the reality lives in a ticket tracker; nobody
> maintains a mapping between them. So three questions go unanswered: which
> tickets are outside the baseline, how much scope has changed, and what that
> change costs."

### Proposed solution — 2:00–3:00

> "DriftGuard matches each ticket to its closest agreed requirement, classifies
> the relationship as in-scope, refinement, new scope or removal, and measures
> a Scope Drift Index — net size added minus removed, over baseline size. It
> then runs COCOMO II to express that as effort and schedule.
>
> Two design commitments shape everything: every classification is explainable
> down to the individual word, and a human can overrule any of them."

### System architecture — 3:00–4:00

> "The analysis lives in a `driftguard` package, the interface in a separate
> `ui` package. The dashboard computes nothing — it renders what the package
> returns. That's why all 300 tests can run headlessly, and why a figure on
> screen can always be reproduced in three lines of Python.
>
> The pipeline is load, match, classify, review, measure, estimate."

### Live demonstration — 4:00–7:00

*Follow steps 1, 6, 7, 12, 15, 16 of the sequence above. Keep moving.*

> "157 points agreed, 276 now implied — a 76% overrun, and here's the
> arithmetic rather than an assertion. The timeline shows it was stable for two
> months and then four large requests landed in March.
>
> Here's one ticket's full audit trail: the matched requirement, the similarity
> score, the shared terms, which cue words fired. Note this one — the
> refinement cues *did* fire, but similarity fell below the threshold, so the
> rule order sent it to new scope. That's a genuine misclassification, and the
> tool shows you why rather than hiding it.
>
> I overrule it, and the index updates everywhere — while the model's original
> prediction stays on record, so evaluation still measures the model."

### Technical methodology — 7:00–8:15

> "Two signals do different jobs. Cosine similarity answers 'does this ticket
> concern the baseline at all' — it's good at that. Keyword cues answer 'does
> it implement, change or withdraw it' — similarity can't answer that, because
> implementing a requirement and rewriting it look identical to a bag-of-words
> model. TF-IDF is not a language model and I don't present it as one.
>
> On estimation, one decision is worth naming. COCOMO's volatility factor,
> REVL, inflates size inside the effort equation. Feeding it the drift index
> double-counts, because the revised size already contains the added scope — my
> first version did exactly that and reported +246% effort for +76% growth.
> REVL now charges churn only: refinement plus removals, excluding new scope."

### Results — 8:15–9:00

> "On the sample project: 76% scope growth, effort from 35 to 84 person-months,
> duration from 11 to 15 months. Effort grows faster than schedule because the
> COCOMO exponent exceeds one.
>
> The classifier scores 95.6% accuracy, macro-F1 0.96 — but that's in-sample,
> and the honest number matters more than the flattering one."

### Limitations — 9:00–9:40

> "Which brings me to the most interesting result. I built nested
> cross-validation to get an honest figure for the rule classifier. It returned
> a *perfect* held-out score — better than in-sample. That can't be right, and
> the tool flags it as a leakage indicator rather than reporting it as success.
>
> The mechanism is measurable: the threshold search shifts decisions off the
> similarity signal and onto my hand-written keyword lexicon, which was written
> after reading these exact tickets. So it's memorisation. Nested CV fixed the
> threshold leakage and exposed the lexicon leakage, which is irreducible
> without a second labelled corpus.
>
> Alongside that: TF-IDF misses paraphrase, the lexicon is domain-specific,
> 45 tickets is a small sample, and every effort figure scales with a
> SLOC-per-story-point factor that is a placeholder, not a measurement."

### Future improvements and conclusion — 9:40–10:00

> "The highest-value next step is an independently labelled ticket set from a
> different project — it's the only thing that gives the classifier a clean
> number. After that, local sentence embeddings for paraphrase, and calibrating
> the size factor against real history.
>
> What DriftGuard does today is turn an anecdotal argument into an auditable
> one, and it's explicit about the difference between what it measured and what
> it assumed. Thank you — happy to take questions."

---

## 18. Anticipated questions

**1. Is the 95.6% accuracy real?**
It is real but in-sample, and therefore optimistic. The thresholds and keyword
lexicon were tuned on the same 45 tickets used to score it. The learned
classifier's cross-validated figure is 75.6% / macro-F1 0.563. The rule
classifier has no clean figure at all, and the tool says so rather than
implying otherwise.

**2. You built nested cross-validation — why doesn't it fix the leakage?**
It fixes the part caused by threshold tuning. It cannot fix the lexicon,
because the lexicon is hand-written and cannot be re-derived per fold. The
nested run actually returned a perfect held-out score, which is diagnostic of
leakage rather than reassuring, and the tool flags it as such. The measurable
mechanism is that lowering the weak threshold moves decisions from the
un-leaked similarity signal onto the leaked lexicon — 33 tickets to 40, across
folds.

**3. Isn't reporting a perfect score and calling it leakage just excusing a bad
result?** The opposite. A validation routine that reported 1.000 as success
would be worse than useless. The flag fires on a structural rule — held-out
cannot legitimately beat in-sample — and the test suite asserts that it fires
on this dataset.

**4. Why not use an LLM for classification?**
A hard constraint of the brief: no paid API in the execution path. Beyond that,
TF-IDF decisions are inspectable term by term, which is what makes the audit
panel possible. An LLM's decision would not be.

**5. Where does 60 SLOC per story point come from?**
It is an unvalidated placeholder. Every effort figure scales with it, which is
why the Scenarios page sweeps it and reports elasticity 1.0997 — a 10% error in
the factor produces about an 11% error in effort. It should be measured from
your own completed projects.

**6. How do you know the COCOMO implementation is correct?**
Three ways. Expected values are derived from the published equations inside the
tests, so a regression cannot rewrite its own yardstick. The sensitivity sweep
recovers the exponent exactly — effort elasticity equals `E = 1.0997` and
duration elasticity equals `E × F = 0.3496`. And the Methodology page re-derives
eight invariants live on whatever data is loaded. Testing found two real bugs
this way: the SCED multiplier conflated with the SCED schedule percentage, and
the REVL double-count.

**7. Why is macro-F1 the headline rather than accuracy?**
The four classes are imbalanced — Removal has four examples in 45. A model that
never predicts Removal can still post high accuracy. Macro-F1 weights each
class equally, and a majority-class baseline (44.4%) is shown beside every
score so the reader can see what "learned nothing" would look like.

**8. Could this work on a real project?**
The method yes, the lexicon no. The keyword cues are domain-specific and would
need re-tuning plus a freshly labelled sample. The honest answer is that
DriftGuard is a demonstrated method with one worked example, not a validated
product.

**9. Why count removals at full size in REVL?**
It assumes the work was built before being dropped. Where removals were
cancelled before starting, that overstates rework, and the REVL link can be
switched off to see the pure size effect. Parameterising the removal rework
fraction is listed as a future improvement rather than hidden.

**10. The sample shows +76% drift — isn't that unrealistically dramatic?**
Yes, deliberately. It is synthetic data chosen so the dashboard has something
to show. A real project at +76% would be in serious trouble. All generated and
sample data is labelled as synthetic throughout.

**11. What happens with messy real-world data?**
The generator has a `dirty` scenario for exactly this: missing sizes, blank
descriptions, unparseable dates, duplicate rows. The loaders warn on each and
the analysis still completes — that path is covered by tests rather than
asserted.

**12. Why is the drift sensitivity metric more important than accuracy?**
Because accuracy is a proxy. A sponsor cares whether misclassification moves
the number the decision rests on. Here it moves it by 2.9 percentage points, in
the conservative direction — so the classifier is good enough for the decision
even though it is not good enough to publish.

**13. Why is history in memory only rather than in a database?**
Adding a datastore would change the zero-infrastructure deployment story for a
feature that is useful but not central. The limitation is stated in the UI and
the README, and durable persistence is listed as a future improvement rather
than half-implemented.

**14. Isn't "scope drift" just data drift?**
No, and conflating them would be a category error. Data drift is statistical
distribution shift between a reference and a current dataset, measured with
tests like PSI or Kolmogorov–Smirnov. DriftGuard measures change to an agreed
set of requirements. It has no tabular feature matrix, so it computes no such
statistics — and the Feature-Level page says exactly that rather than
fabricating them.

---

## 19. Pre-demonstration checklist

- [ ] `./run.sh` starts the app without activating anything
- [ ] **Load sample project** populates all nine sections
- [ ] All nine nav sections render with no error traces
- [ ] Charts render (overview, timeline, scenarios, confusion matrix, box plots, learning curve)
- [ ] Nested validation shows the leakage flag
- [ ] Snapshot → compare works
- [ ] Override round-trip moves the SDI and is reversible
- [ ] Synthetic generator produces a banner on every page
- [ ] PDF generates and downloads
- [ ] Methodology audit reports 8 of 8
- [ ] `python -m pytest tests` → **300 passed**
- [ ] No API keys, credentials or personal data in the repo
- [ ] Browser zoom 100%, window maximised, sidebar expanded
