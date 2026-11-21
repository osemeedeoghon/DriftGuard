# DriftGuard — Project Audit

**Audit date:** 20 November 2026
**Commit audited:** `fb8c56b`
**Auditor scope:** read-only inspection. No application code was modified.

> **Remediation status (20 Nov 2026).** Findings **S-1** (HTML injection) and
> **§2.3** (uploaded projects not registered) have since been fixed under
> roadmap tasks P0.1 and P0.2. The findings below are preserved as written —
> an audit is a point-in-time record — with a status line appended to each.
> A new finding, **S-4**, was discovered during that work and is recorded in
> §7.6.

---

## How to read this document

Every claim below is one of three kinds, and is labelled:

| Mark | Meaning |
|---|---|
| **[M]** | **Measured** — produced by running a command against this repository. The command is shown or described. |
| **[I]** | **Inspected** — read directly from source, with `file:line` given. |
| **[A]** | **Assumption or judgement** — reasoning, estimate or opinion. Not evidence. |

Nothing in this audit is presented as user research, competitor research or
vendor benchmarking unless it was actually performed. Where such work has
**not** been done, the document says so and schedules it instead of inventing
findings. This applies in particular to §3 (business positioning), where no
competitor verification has been carried out.

---

## 0. Baseline

**[M]** Test suite, run at the start of this audit:

```
$ python -m pytest tests
300 passed in 15.45s
```

**[M]** Working tree clean at `fb8c56b`; no uncommitted changes.

**[M]** Codebase size:

| Layer | Lines | Files |
|---|---:|---:|
| `driftguard/` (analysis) | 5,666 | 16 |
| `ui/` (presentation) | 4,747 | 17 |
| `tests/` | 3,035 | 11 |
| `app.py` + `run.sh` | ~175 | 2 |
| **Total** | **~13,600** | **46** |

Test-to-production ratio is 3,035 : 10,413 ≈ **0.29**. **[A]** That is healthy
for an analytical codebase, and the tests are behavioural rather than
snapshot-based, which matters more than the ratio.

---

## 1. Existing architecture assessment

### 1.1 Layering — strong

**[M]** The analysis layer contains **zero** Streamlit imports:

```
$ grep -rn "import streamlit\|from ui" driftguard/
(no matches)
```

This is the single best structural property of the codebase. It means every
number the dashboard shows is reproducible headlessly, and it is why 300 tests
can run without a browser or a Streamlit runtime.

### 1.2 Dependency graph — mostly acyclic, one inversion

**[M]** Intra-package dependencies:

```
classification  -> config
data            -> config
drift           -> config
evaluation      -> config
features        -> config
matching        -> config
synthetic       -> config
scenarios       -> cocomo, config, drift
pipeline        -> classification, cocomo, config, drift, matching
validation      -> classification, config, evaluation
reporting       -> config, evaluation, pipeline      <-- inversion
pdf_report      -> config, evaluation, pipeline      <-- inversion
```

`config` is a shared kernel that everything depends on and which depends on
nothing. **[A]** Appropriate: it holds the taxonomy, lexicons and constants.

**[I]** The inversion: `reporting.py:23` and `pdf_report.py:38` import
`AnalysisResult` from `pipeline`. Report *generators* therefore depend on the
*orchestrator*. **[A]** This is a design smell rather than a defect — the
dependency is for a type only — but it blocks reusing the report writers
against any other result shape, and it is why `features.py` takes `result`
duck-typed instead (an inconsistency: two modules solve the same problem two
different ways).

**[M]** Eight deferred (function-level) imports exist, principally to avoid
import cycles:

```
classification.py -> matching      evaluation.py  -> drift, classification
features.py       -> classification, config       validation.py -> classification, config
scenarios.py      -> cocomo
```

**[A]** Each is individually defensible, but collectively they are a signal
that the module boundaries are not quite where the call graph wants them.

### 1.3 Code quality signals

**[M]** Analysis layer, measured by AST walk:

| Metric | Value |
|---|---|
| Functions | 161 |
| With type hints | **161 (100%)** |
| With docstrings | 85 (53%) |
| Classes | 24 |
| With docstrings | **23 (96%)** |
| Bare `except:` | **0** |
| Broad `except Exception` | 1, at `ui/pages/reports.py:79`, annotated `# noqa: BLE001` with a user-facing fallback |
| `raise` statements | 70 |
| `TODO` / `FIXME` / `HACK` | **0** |

**[A]** Unusually disciplined for a project of this age. The 53% function
docstring figure is lower than it looks: the undocumented functions are mostly
small private helpers and dataclass methods.

### 1.4 Maintainability hotspots

**[M]** Longest functions:

| Lines | Location | Function |
|---:|---|---|
| **481** | `driftguard/pdf_report.py:232` | `build_pdf_report` |
| 359 | `ui/pages/scenarios.py:18` | `render` |
| 316 | `ui/pages/overview.py:13` | `render` |
| 294 | `ui/pages/methodology.py:15` | `render` |
| 236 | `ui/pages/performance.py:14` | `render` |
| 233 | `ui/pages/drift_analysis.py:13` | `render` |
| 215 | `ui/sidebar.py:85` | `render` |
| 209 | `driftguard/reporting.py:36` | `build_markdown_report` |

**[A]** `build_pdf_report` at 481 lines is the clearest maintainability
liability in the repository. It is a single linear procedure that mixes
document structure, content selection and formatting. It is well covered by
tests (13 of them), so it is safe to refactor — but every future report change
means editing one very long function.

The page `render()` functions are a milder version of the same problem. They
are difficult to unit-test (they require a Streamlit runtime), which is why
**the UI layer has no direct test coverage at all** — see §7.3.

### 1.5 Architectural verdict

**[A]** The core architecture is sound and the layering decision was correct.
The weaknesses are not structural rot; they are the absence of the
infrastructure a longer-lived system needs — persistence, logging,
configuration, CI — plus two localised code-quality hotspots.

---

## 2. Existing functionality inventory

**[I]** Verified by reading source and the nine registered pages in
`app.py:38-52`.

### 2.1 Analysis capabilities

| Capability | Module | Tests |
|---|---|---|
| CSV + TXT baseline loading, column-alias normalisation | `data.py` | ✓ |
| Ticket CSV loading, date parsing, label validation | `data.py` | ✓ |
| Input validation: 12 hard rejections, 8 warning paths, 26 column aliases **[M]** | `data.py` | ✓ |
| TF-IDF matching with per-term evidence and runner-up margins | `matching.py` | ✓ |
| Four-class rule classification with visible lexicon | `classification.py` | ✓ |
| Logistic-regression classifier (learned baseline) | `classification.py` | ✓ |
| Human overrides, kept separate from predictions | `classification.py` | ✓ |
| Scope Drift Index, timeline, attribution | `drift.py` | ✓ |
| COCOMO II Post-Architecture, 5 SF + 17 EM + SCED dual table | `cocomo.py` | ✓ |
| REVL churn bridge (excludes new scope) | `drift.py` | ✓ |
| Accuracy, macro-F1, per-class, confusion, drift sensitivity | `evaluation.py` | ✓ |
| Stratified cross-validation | `evaluation.py` | ✓ |
| Nested CV + automatic leakage detection | `validation.py` | ✓ |
| Learning curve | `validation.py` | ✓ |
| Live methodology audit (8 invariants) | `validation.py` | ✓ |
| Sensitivity sweeps, scenarios, tornado, log elasticity | `scenarios.py` | ✓ |
| Synthetic generator, 5 scenarios, reproducible by seed | `synthetic.py` | ✓ |
| In-session project store and scope history | `projects.py` | ✓ |
| Markdown report | `reporting.py` | ✓ |
| PDF report (7 pages, ReportLab + Matplotlib) | `pdf_report.py` | ✓ |

### 2.2 Interface capabilities

Nine sections: Executive overview · Dataset explorer · Scope drift analysis ·
Scenarios · Model performance · Feature-level analysis · Ticket review ·
Reports and export · Methodology and about.

### 2.3 Functional gap found during this audit

**[I] [Defect] Uploaded projects are never registered in the project store.**

`ui/sidebar.py:52` (and the two sibling branches) assign
`st.session_state.baseline` directly:

```python
st.session_state.baseline = frame        # sidebar.py:52
```

whereas the sample and synthetic paths go through `state._install()`
(`state.py:78`, `state.py:112`), which is what calls `ProjectStore.add()`.

**Consequences, all verified by reading the call graph:**

1. An uploaded project does not appear in the project switcher, so multi-project
   switching works only between sample and synthetic projects.
2. Scope-history snapshots cannot be attached to an uploaded project —
   `ui/pages/drift_analysis.py` guards on `project_store.active is None` and
   shows a warning instead.
3. `st.session_state.project_name` keeps its default for uploads, so the
   context bar and every exported report title an uploaded project
   "DriftGuard analysis".

**[A]** Severity: **high for the multi-project feature**, because the feature
is effectively unavailable for the primary input path (upload). It is not a
data-corruption risk.

> ✅ **FIXED** under P0.2. The upload handler now routes through
> `state.install_uploaded_project()`, keyed on a content hash so Streamlit's
> repeated reruns cannot register duplicates. Project names are derived from
> the uploaded filename and are editable. Verified in the running application:
> an uploaded project appears in the switcher alongside the sample
> ("2 projects in this session") and accepts scope snapshots
> (`tests/test_upload_registration.py`, 27 tests).

---

## 3. Business and product positioning assessment

> **Verification status: NO competitor research has been performed.**
> Nothing in this section should be quoted as a finding about any named
> product. The brief requires that verified capabilities be distinguished from
> assumptions; at audit stage, **everything about competitors is an
> assumption**, and the verification work is scheduled as roadmap item **B1**.

### 3.1 What the product demonstrably is **[I]**

Read from the code rather than from aspiration, DriftGuard is:

- a **single-user, offline, run-on-demand analysis tool**;
- operating on **two CSV/TXT exports** rather than a live integration;
- producing a **governance artefact** (an auditable scope and cost argument);
- whose distinguishing property is that **every number is traceable to an
  input and every assumption is labelled as one**.

### 3.2 Positioning statement **[A]**

**[A]** For a delivery lead or business analyst who must argue about scope with
a sponsor and has only a requirements document and a ticket export, DriftGuard
converts an anecdotal disagreement into an auditable one — quantifying scope
change and costing it with a published parametric model, while stating
explicitly which inputs were measured and which were assumed.

### 3.3 Candidate differentiators **[A]**

These follow from implemented behaviour, but their *competitive* value is
unverified:

1. **Explainability to the individual word.** Every classification exposes its
   matched requirement, similarity, shared TF-IDF terms and the cue words that
   fired. **[I]** `ui/pages/review.py:260-400`.
2. **Adversarial honesty about its own validity.** The tool detects and reports
   leakage in its own evaluation rather than publishing the flattering number.
   **[I]** `validation.py:66-96`. **[A]** This is unusual and is probably the
   strongest academic differentiator.
3. **Offline and zero-infrastructure.** No API key, no cloud, no datastore.
4. **Costed in a defensible model.** COCOMO II with published constants and no
   fabricated calibration.

### 3.4 Honest weaknesses in the current positioning **[A]**

- **No integration.** Manual CSV export is a real adoption barrier.
- **Lexicon is domain-specific**, so out-of-the-box accuracy on a new project
  is unknown — and the audit cannot claim otherwise (§7.1).
- **Single-user, no persistence**, so it is a tool, not a system of record.
- **Validated on one synthetic project.** It is a demonstrated method, not a
  validated product, and the README should continue to say so.

### 3.5 Scheduled work (not performed)

| ID | Work | Why it cannot be asserted now |
|---|---|---|
| B1 | Competitor capability matrix | Requires reading current vendor documentation; none consulted |
| B2 | Blue/Red Ocean analysis | Depends on B1 |
| B3 | SWOT | Depends on B1 |
| B4 | Personas and user journeys | Requires contact with real users; none conducted |
| B5 | Business Model Canvas | Depends on B1–B4 |

---

## 4. UX and usability gap analysis

> **Method:** expert **heuristic inspection** of the implemented interface
> against Nielsen's ten heuristics, plus measured interaction cost. **No user
> testing, interviews or surveys have been conducted**, and none is reported
> here. Findings marked **[A]** are inspection judgements, not user research.

### 4.1 Heuristic evaluation

| # | Heuristic | Status | Evidence |
|---|---|---|---|
| 1 | Visibility of system status | **Good** | Spinners via `components.working()`; context bar; load warnings surfaced **[I]** |
| 2 | Match with the real world | **Good** | Domain vocabulary (scope, baseline, tickets); equations shown with real numbers substituted **[I]** `drift_analysis.py:34-44` |
| 3 | User control and freedom | **Partial** | Per-ticket Reset and Clear-all exist; **no undo for a batch of overrides**, no confirmation on "Clear all reviews" **[I]** `review.py:412-419` |
| 4 | Consistency and standards | **Good** | One theme module; charts routed through `components.chart()`; consistent number/date formatting **[I]** |
| 5 | Error prevention | **Strong** | 12 hard validations, 8 warning paths before analysis **[M]** |
| 6 | Recognition over recall | **Partial** | Assumptions shown beside figures; **but the sidebar has 7 collapsed expanders and the active settings are not summarised anywhere** **[I]** `sidebar.py` |
| 7 | Flexibility and efficiency | **Weak** | No keyboard shortcuts, no saved configurations, no URL state — every session re-enters settings by hand |
| 8 | Aesthetic and minimalist design | **Good** | Validated palette; table view paired with every chart |
| 9 | Help users recover from errors | **Strong** | Errors name the file, the column and the fix **[I]** `data.py` |
| 10 | Help and documentation | **Strong** | Methodology page, live audit, in-UI caveats |

### 4.2 Measured interaction cost — the most actionable UX finding

**[M]** There is **no caching anywhere** in the UI layer:

```
$ grep -rn "st.cache_data\|st.cache_resource" ui/ app.py
(no matches)
```

Streamlit re-executes the whole page on every widget interaction, so the
following is recomputed each time **[M]**:

| Computation | Cost | Recomputed on |
|---|---:|---|
| `run_analysis` | 28 ms | every interaction, every page |
| `cross_validated_scores` | 94 ms | Model Performance |
| `nested_cross_validate` | **933 ms** | Model Performance |
| `learning_curve` | 373 ms | Model Performance |
| `tornado` (17 sweeps) | 1 ms | Scenarios |

**[M]** The Model Performance page therefore costs **≈1.43 s of recomputation
for every widget change** — including merely switching the "Inspect classifier"
dropdown, which needs none of it.

**[A]** This is the clearest active-usability defect in the product. It is
invisible on the Overview page (28 ms) and unacceptable on Model Performance.

### 4.3 Accessibility — partially evidenced, largely untested

**[M] Verified:** the four-class categorical palette was validated
programmatically against colour-vision-deficiency and contrast gates; two
sub-3:1 fills are compensated by direct labels and a paired table view
(`driftguard/config.py:42-62` records the measurements).

**[A] NOT verified, and must not be claimed:**

- keyboard-only navigation has never been tested;
- no screen-reader pass (VoiceOver/NVDA) has been performed;
- focus order and visible focus states are unaudited;
- the `unsafe_allow_html` cards in `components.py` emit `<div>`s with no ARIA
  roles or heading semantics;
- no WCAG conformance level can currently be claimed at all.

### 4.4 Passive and predictive usability **[A]**

- **Passive:** severity bands, risk levels and the leakage verdict communicate
  state without the user asking. Good. Counter-example: the sidebar gives no
  indication that a non-default assumption is in force.
- **Predictive:** the application anticipates the "is this number trustworthy?"
  question better than most analytical tools (leakage flag, audit panel,
  assumption callouts). It does **not** anticipate "what did I change last
  time?" — there is no settings history.

---

## 5. Software modelling requirements

**[A]** The six required diagrams must reflect implemented behaviour. From this
inspection, each should contain the following — this is the specification for
the modelling work, not the diagrams themselves.

| Diagram | Must show | Source of truth |
|---|---|---|
| **Use case** | Actors: Delivery Lead/BA (primary), Reviewer, Sponsor (report consumer). ~14 use cases incl. *Load project*, *Generate synthetic project*, *Review and override classification*, *Record scope snapshot*, *Run sensitivity analysis*, *Export PDF report*, *Inspect methodology audit* | `app.py` page registry; `ui/sidebar.py` |
| **Class / domain model** | 24 classes **[M]**. Core: `AnalysisSettings`, `AnalysisResult`, `DriftResult`, `CocomoDrivers`, `CocomoEstimate`, `EstimateComparison`, `SizeConversion`, `MatchingThresholds`, `Decision`, `MatchEvidence`, `EvaluationResult`, `NestedValidationResult`, `Scenario`, `ScenarioComparison`, `ProjectRecord`, `ScopeSnapshot`, `ProjectStore`, `SyntheticProject`, `LoadReport`, `RequirementMatcher`, `RuleClassifier`, `MLClassifier` | `grep "^class" driftguard/*.py` |
| **Sequence** | The analysis path: `app.main` → `sidebar.render` → `pipeline.run_analysis` → `RequirementMatcher.fit/match_tickets` → `RuleClassifier.classify` → `apply_overrides` → `calculate_drift` → `revl_from_drift` → `cocomo.compare` → page render. Plus the override loop, which re-enters at `run_analysis` | `pipeline.py:57-198` |
| **ER** | *Currently no persistent schema exists.* The ERD must therefore model the **proposed** SQLite schema (§8) and be labelled as target-state, not as-built | §8 of this document |
| **Activity** | Load → validate (with warning/reject branches) → match → classify → optional human review loop → drift → estimate → report. Must include the validation failure paths, which are a third of the behaviour | `data.py`, `pipeline.py` |
| **State machine** | Two machines: (a) **Session**: `No project → Project loaded → Analysed → Reviewed → Snapshotted`; (b) **Ticket classification**: `Unclassified → Predicted → {Confirmed | Overridden} → Counted in SDI`, with the invariant that `predicted_class` is immutable after prediction | `state.py`, `classification.apply_overrides` |

**[A]** The ERD is the only one of the six that cannot be drawn from current
code. That is itself an audit finding: **the system has no persistent data
model.**

---

## 6. Architecture gap analysis

| Concern | Current state | Gap |
|---|---|---|
| Layer separation | **[M]** Clean; zero UI imports in analysis | None |
| Persistence | **[M]** None. `ProjectStore` is in-memory | **Critical for the multi-project feature** |
| Configuration | **[M]** No environment variables; all settings UI-only | No deploy-time configuration possible |
| Logging | **[M]** No `logging` import anywhere | **No audit trail of overrides or analyses** |
| Error handling | **[M]** 70 raises, 0 bare excepts, typed `DataValidationError` | Good; no central handler or error taxonomy beyond data loading |
| Packaging | **[M]** No `pyproject.toml`, no `setup.py` | Not installable; imports rely on CWD |
| Containerisation | **[M]** No `Dockerfile` | Deployment is "run this script on a machine with Python" |
| CI | **[M]** No `.github/` directory | **Tests only run when someone remembers** |
| Dependency pinning | **[I]** `requirements.txt` uses `>=` ranges only | Builds are not reproducible |
| API surface | **[M]** None; UI is the only entry point | Blocks future integration |
| Concurrency | **[I]** Single-process Streamlit; session state only | Fine for single user; no multi-user story |

---

## 7. Security and reliability assessment

### 7.1 Finding S-1 — HTML injection via unescaped interpolation **[M] CONFIRMED**

**Severity: Medium** (Low in the current single-user offline deployment;
**High** if DriftGuard is ever served to more than one person, which the
roadmap contemplates).

**[M]** Escaping is applied inconsistently:

| Location | `unsafe_allow_html` uses | `html.escape` calls |
|---|---:|---:|
| `ui/components.py` | 6 | **16** |
| `ui/pages/*.py` + `app.py` | 7 | **0** |

**[M] Proven exploit path.** A ticket id containing markup, as could appear in
any third-party ticket export, reaches the DOM unescaped:

```
ticket_id = "<img src=x onerror=alert(1)>"
  -> features.recommendations()  (features.py:318 interpolates ticket_id)
  -> "...three largest contributors (<img src=x onerror=alert(1)>, T-30..."
  -> ui/pages/overview.py:287   f-string into st.markdown(..., unsafe_allow_html=True)
  -> emitted verbatim
```

Executed during this audit; the payload survived to the rendered string
unescaped. The same value through `components._esc()` correctly yields
`&lt;img src=x onerror=alert(1)&gt;`.

**Other instances of the same class:** `ui/pages/review.py:336` (TF-IDF terms
from uploaded text), `app.py:113` (`project_name`).

**[A] Why it matters despite being a local app:** the threat model is not "the
user attacks themselves". It is "an analyst runs DriftGuard over a ticket
export supplied by a client or a vendor". That export is untrusted input.

> ✅ **FIXED** under P0.1. A `SafeHtml` marker type and an idempotent `esc()`
> now guard every render boundary, enforced by a static AST check over all
> `unsafe_allow_html` sites (`tests/test_html_escaping.py`, 18 tests). Verified
> in the running application: the payload produced **zero** injected `<img>`
> elements and rendered as literal text. See [ADR-0005](adr/0005-html-escaping-policy.md).

### 7.2 Finding S-2 — no audit trail **[M]**

**Severity: Medium** for a governance tool.

**[M]** No `logging` import exists anywhere in the codebase. Consequences:
reviewer overrides are not recorded with who/when/why; analyses leave no trace;
a disputed SDI figure cannot be reconstructed after the fact. **[A]** For a
product whose entire value proposition is *auditability*, the absence of an
audit log is a positioning inconsistency as much as a technical gap.

### 7.3 Finding S-3 — the UI layer is untested **[M]**

**[M]** All 300 tests target `driftguard/` plus pure helpers in `ui/theme.py`
and `ui/components.py`. No test executes a page `render()` function.

**[A]** This is a direct consequence of the 200–360-line render functions:
they require a Streamlit runtime. Every UI regression in this project's history
was found by a human driving a browser, not by the suite. That is the main
reliability gap.

### 7.4 Positive reliability findings

- **[M] Performance scales acceptably.** Measured on synthetic projects:

  | Requirements | Tickets | Similarity pairs | Analysis |
  |---:|---:|---:|---:|
  | 15 | 40 | 600 | 30 ms |
  | 40 | 150 | 6,000 | 50 ms |
  | 80 | 400 | 32,000 | 118 ms |
  | 150 | 1,000 | 150,000 | 278 ms |
  | 300 | 2,500 | 750,000 | **686 ms** |

  Per-ticket cost is flat at ≈0.27 ms. **[A]** No performance work is needed;
  the N×M similarity matrix is not a practical bottleneck at realistic sizes.

- **[M] Graceful degradation on dirty input** is tested
  (`test_dirty_scenario_still_completes_an_analysis`): malformed files produce
  warnings and a completed analysis, not a traceback.

- **[M] No network client** in the execution path — verified by grep for
  `requests`, `urllib`, `http`, `openai`, `anthropic` across `driftguard/`,
  `ui/` and `app.py`. The offline-first claim holds.

- **[M] No secrets or credentials** in the repository.

### 7.6 Finding S-4 — CSV formula injection in exports **[M] CONFIRMED**

**Severity: Low–Medium.** Discovered 20 November 2026 while implementing P0.1;
**not fixed**.

**[M]** Exported CSVs do not neutralise formula-leading cells. A ticket id of
`=HYPERLINK("http://evil","click")` survives `reporting.classified_export()`
and `data.frame_to_csv_bytes()` unescaped:

```
"=HYPERLINK(""http://evil"",""click"")",Add su...
```

A spreadsheet application opening that export may evaluate the cell as a
formula. **[A]** It is distinct from S-1: the sink is Excel or Sheets rather
than the DOM, so the S-1 fix does not address it. Standard mitigation is to
prefix a cell beginning `=`, `+`, `-` or `@` with an apostrophe, or to quote
defensively on export.

Scheduled as roadmap task **P0.6**.

### 7.5 Risks explicitly *not* found

To avoid implying a cleaner bill of health than was earned: this audit did
**not** perform dependency vulnerability scanning, SAST, fuzzing of the CSV
parser, or any penetration testing. Those are scheduled (roadmap **E4**).

---

## 8. Recommended SQLite integration design

**[A]** Recommended, with the offline-first constraint preserved: SQLite is a
single file, requires no server, ships with Python, and adds **no new
dependency**.

### 8.1 Design principles

1. **Persistence is optional.** The application must continue to run entirely
   in memory if no database path is configured. This preserves the current
   zero-infrastructure story and keeps the 300 existing tests valid unchanged.
2. **The domain layer must not import `sqlite3`.** Introduce a repository
   interface; `ProjectStore` becomes one implementation
   (`InMemoryProjectStore`), SQLite another (`SqliteProjectStore`). This is the
   smallest change that makes persistence swappable and testable.
3. **Analyses are append-only.** A snapshot is a historical record; it is never
   updated. This is what makes the audit trail trustworthy.
4. **Store inputs by content hash.** Two analyses of the same baseline should
   reference one stored copy, and a changed baseline must be detectable.

### 8.2 Proposed schema

```sql
CREATE TABLE project (
    project_id    TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT,
    created_at    TEXT NOT NULL,
    source        TEXT NOT NULL CHECK (source IN ('sample','upload','synthetic'))
);

CREATE TABLE dataset (                -- content-addressed input files
    dataset_id    TEXT PRIMARY KEY,   -- sha256 of the normalised frame
    project_id    TEXT NOT NULL REFERENCES project(project_id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK (kind IN ('baseline','tickets','labelled')),
    row_count     INTEGER NOT NULL,
    total_size    REAL,
    payload       TEXT NOT NULL,      -- CSV text, so it is inspectable
    loaded_at     TEXT NOT NULL,
    UNIQUE (project_id, kind, dataset_id)
);

CREATE TABLE analysis_run (           -- append-only
    run_id            TEXT PRIMARY KEY,
    project_id        TEXT NOT NULL REFERENCES project(project_id) ON DELETE CASCADE,
    baseline_id       TEXT NOT NULL REFERENCES dataset(dataset_id),
    tickets_id        TEXT NOT NULL REFERENCES dataset(dataset_id),
    run_at            TEXT NOT NULL,
    settings_json     TEXT NOT NULL,  -- full AnalysisSettings, for reproducibility
    sdi               REAL NOT NULL,
    baseline_size     REAL NOT NULL,
    revised_size      REAL NOT NULL,
    person_months     REAL NOT NULL,
    duration_months   REAL NOT NULL,
    revl_percent      REAL NOT NULL,
    label             TEXT,
    note              TEXT
);

CREATE TABLE ticket_classification (  -- per run, for attribution over time
    run_id            TEXT NOT NULL REFERENCES analysis_run(run_id) ON DELETE CASCADE,
    ticket_id         TEXT NOT NULL,
    predicted_class   TEXT NOT NULL,
    final_class       TEXT NOT NULL,
    confidence        REAL,
    similarity        REAL,
    matched_req       TEXT,
    net_contribution  REAL,
    PRIMARY KEY (run_id, ticket_id)
);

CREATE TABLE review_event (           -- append-only audit trail (addresses S-2)
    event_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id        TEXT NOT NULL REFERENCES project(project_id) ON DELETE CASCADE,
    ticket_id         TEXT NOT NULL,
    occurred_at       TEXT NOT NULL,
    actor             TEXT,           -- OS user; no auth system is proposed
    field             TEXT NOT NULL CHECK (field IN ('classification','size')),
    from_value        TEXT,
    to_value          TEXT,
    model_prediction  TEXT NOT NULL   -- preserved so evaluation stays honest
);

CREATE INDEX idx_run_project  ON analysis_run(project_id, run_at);
CREATE INDEX idx_event_ticket ON review_event(project_id, ticket_id, occurred_at);
```

### 8.3 Integrity rules **[A]**

- `review_event.model_prediction` is written at event time and never updated.
  This enforces in the schema the invariant the code already maintains: a human
  override must never contaminate model evaluation.
- `analysis_run` stores the complete `AnalysisSettings` as JSON, so any
  historical figure can be reproduced exactly — which is what makes the stored
  SDI defensible months later.
- Content-hashed `dataset_id` makes "the baseline changed between these two
  runs" detectable rather than a matter of recollection.
- Foreign keys with `ON DELETE CASCADE`, and `PRAGMA foreign_keys = ON` (off by
  default in SQLite — a real footgun worth an ADR).

### 8.4 Migration approach **[A]**

Schema version in `PRAGMA user_version`; forward-only numbered migrations
applied at startup. No migration framework — ~40 lines of code, no dependency.

### 8.5 What this does **not** solve

**[A]** SQLite gives durability and an audit trail. It does **not** give
multi-user concurrency, authentication or authorisation, and this audit
recommends **not** pretending otherwise. A shared deployment needs a different
conversation, and would invalidate the offline-first positioning.

---

## 9. Production-readiness gap matrix

Scored 0–5. **[A]** Scores are judgements; the evidence behind each is cited
above.

| Dimension | Score | Evidence | Gap to close |
|---|:---:|---|---|
| Correctness of core model | **5** | 300 tests; equations re-derived in tests; live audit of 8 invariants | — |
| Test coverage (analysis) | **5** | 3,035 test lines, behavioural | — |
| Test coverage (UI) | **1** | §7.3 — no page render is executed by any test | Extract testable logic; add smoke tests |
| Layer separation | **5** | §1.1 | — |
| Maintainability | **3** | §1.4 — one 481-line function, six 200+ line renders | Decompose hotspots |
| Input validation | **4** | 12 rejections, 8 warnings, dirty-data test | Fuzz the parser |
| Error handling | **4** | 0 bare excepts, typed errors | Central taxonomy |
| **Security** | **2** | §7.1 — proven injection path | Escape at every HTML boundary |
| **Audit logging** | **0** | §7.2 — no logging at all | Structured log + `review_event` |
| **Persistence** | **0** | §6 — in-memory only | SQLite (§8) |
| Multi-project | **2** | §2.3 — unavailable for uploads | Route uploads through `_install` |
| Performance | **5** | §7.4 — 686 ms at 2,500 tickets | — |
| UI responsiveness | **2** | §4.2 — 1.43 s recompute per interaction | Cache expensive calls |
| Accessibility | **2** | §4.3 — palette validated; everything else untested | Keyboard + screen-reader audit |
| Configuration | **1** | §6 — no env vars | Config layer |
| Packaging | **1** | §6 — no `pyproject.toml` | Package properly |
| CI | **0** | §6 — no `.github/` | GitHub Actions |
| Dependency reproducibility | **2** | `>=` ranges only | Pin, add lock |
| Documentation | **4** | README + 19-section guide + in-app methodology | Add the `docs/` set |
| API / integration | **0** | No programmatic entry point | Service layer |

**Weakest five:** audit logging (0), persistence (0), CI (0), API (0),
UI test coverage (1).

**[A] Overall:** DriftGuard is a **well-engineered analytical core wrapped in
an unhardened application shell**. The mathematics, the tests and the
intellectual honesty are genuinely strong. The gaps are almost entirely in the
surrounding systems engineering — which is precisely the transformation this
programme of work is meant to deliver.

---

## 10. Prioritised development roadmap

Full roadmap with acceptance criteria: **[`DEVELOPMENT_ROADMAP.md`](DEVELOPMENT_ROADMAP.md)**.

Priority summary:

| Priority | Items |
|---|---|
| **P0 — correctness and safety** | S-1 HTML escaping · upload/project-registration defect · CI pipeline |
| **P1 — the persistence story** | Repository abstraction · SQLite · audit logging · migrations |
| **P2 — engineering quality** | Decompose `build_pdf_report` · UI caching · UI smoke tests · packaging · dependency pinning |
| **P3 — documentation and modelling** | Six UML diagrams · architecture docs · ADRs · testing strategy |
| **P4 — business and UX research** | Competitor verification · Blue/Red Ocean · SWOT · personas · accessibility audit |
| **P5 — future** | Service/API layer · integrations |

---

## Appendix A — audit commands

Reproducible against commit `fb8c56b`:

```bash
python -m pytest tests                                   # 300 passed
wc -l driftguard/*.py ui/*.py ui/pages/*.py tests/*.py   # sizes
grep -rn "import streamlit\|from ui" driftguard/         # layer separation
grep -rn "st.cache_data\|st.cache_resource" ui/ app.py   # caching (none)
grep -rn "import logging\|logger\." driftguard/ ui/      # logging (none)
grep -rc "unsafe_allow_html" ui/*.py ui/pages/*.py       # injection surface
grep -rc "_esc(\|html.escape" ui/pages/*.py app.py       # escaping (zero)
ls .github Dockerfile pyproject.toml                     # CI/packaging (absent)
```

## Appendix B — explicitly preserved decisions

Verified present and unmodified by this audit:

| Decision | Location |
|---|---|
| REVL charges churn only, excluding new scope | `drift.py:270-305` |
| SDI = (added − removed) ÷ baseline | `drift.py:97-201` |
| COCOMO II.2000 constants A/B/C/D | `cocomo.py:42-45` |
| SCED dual table (effort multiplier + schedule %) | `cocomo.py:103-111` |
| Four-class taxonomy | `config.py:15-20` |
| Thresholds 0.18 / 0.45 | `config.py:55-70` |
| Leakage detection and nested CV | `validation.py` |
| Offline-first (no network client) | verified by grep |
| CSV/TXT import | `data.py` |
| PDF reporting | `pdf_report.py` |
| `run.sh` launcher | `run.sh` |
