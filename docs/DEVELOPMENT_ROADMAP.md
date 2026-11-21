# DriftGuard — Development Roadmap

**Version:** 1.0 · **Date:** 20 November 2026 · **Baseline commit:** `fb8c56b`
**Companion document:** [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md)

---

## 1. Purpose and constraints

This roadmap turns the audit's findings into sequenced, testable work. It is a
plan, not a record: nothing here has been implemented.

### 1.1 Constraints that bind every item

These are non-negotiable and every work item below was checked against them:

| Constraint | Enforcement |
|---|---|
| Preserve REVL churn rationale | No item touches `drift.revl_from_drift` |
| Preserve SDI calculation | No item touches `drift.calculate_drift` arithmetic |
| Preserve COCOMO II equations and constants | No item touches `cocomo.py` constants |
| Preserve classification taxonomy and thresholds | No item changes `config.CLASSES` or `MatchingThresholds` defaults |
| Preserve leakage detection and validation methodology | No item modifies `validation.py` logic |
| Preserve offline-first | **No item adds a network client.** SQLite is file-local and ships with Python |
| Preserve CSV/TXT import, PDF reporting, `run.sh` | Covered by existing tests, which must stay green |
| **300 existing tests must pass after every item** | Definition of done for all work |

### 1.2 Dependency budget

| Proposed addition | Verdict |
|---|---|
| `sqlite3` | **Approved** — Python standard library, no install, file-local |
| `hashlib`, `logging`, `json`, `pathlib` | **Approved** — standard library |
| ORM (SQLAlchemy etc.) | **Rejected** — the schema is 5 tables; an ORM is unnecessary weight |
| Migration framework (Alembic) | **Rejected** — `PRAGMA user_version` plus ~40 lines suffices |
| Test coverage tool (`pytest-cov`) | **Deferred to R2**, dev-only, not a runtime dependency |
| Anything requiring a network call at runtime | **Rejected on principle** |

---

## 2. Requirements specification

Requirements are derived from audit findings. **MoSCoW** priority; each has a
verifiable acceptance criterion.

### 2.1 Functional requirements

| ID | Requirement | Priority | Acceptance criterion | Audit ref |
|---|---|---|---|---|
| FR-01 | Uploaded projects shall register in the project store on the same path as sample and synthetic projects | **Must** ✅ | After uploading a baseline and tickets, the project appears in the switcher, accepts scope snapshots, and its name titles exported reports | §2.3 |
| FR-02 | All user-supplied text rendered into HTML shall be escaped | **Must** ✅ | A ticket id of `<img src=x onerror=alert(1)>` renders as literal text on every page; regression test asserts the payload appears escaped | §7.1 |
| FR-03 | The system shall optionally persist projects, analyses and review events to a local SQLite file | **Must** | With a database path configured, projects and history survive restart; with none configured, behaviour is byte-identical to today | §8 |
| FR-04 | Every reviewer override shall be recorded as an immutable audit event | **Must** | Each override writes one `review_event` row with timestamp, actor, before/after and the model's original prediction | §7.2 |
| FR-05 | The system shall record a structured application log | **Should** | Analyses, validation failures and persistence errors are logged with level and context to a configurable destination | §7.2 |
| FR-06 | The system shall expose a programmatic entry point for a full analysis | **Could** | A documented function/CLI accepts two file paths and returns or writes the analysis without Streamlit | §6 |
| FR-07 | Users shall be able to export and re-import a project archive | **Could** | A project exports to a single file and re-imports with history intact | §6 |
| FR-08 | CSV exports shall not produce cells a spreadsheet evaluates as formulas | **Should** | A ticket id of `=HYPERLINK(...)` exports inert | §7.6 |

### 2.2 Non-functional requirements

| ID | Requirement | Priority | Acceptance criterion | Audit ref |
|---|---|---|---|---|
| NFR-01 | No page interaction shall recompute analysis that has not changed | **Must** | Model Performance responds in <200 ms to a dropdown change (currently ≈1,430 ms) | §4.2 |
| NFR-02 | The test suite shall run automatically on every push and pull request | **Must** | A GitHub Actions run is green on `main`, and the badge is in the README | §6 |
| NFR-03 | Runtime dependencies shall be pinned to reproducible versions | **Must** | A fresh environment built from the pinned file passes all tests | §6 |
| NFR-04 | The UI layer shall have automated regression coverage | **Should** | Every page module has at least a smoke test proving it renders without exception for a loaded project | §7.3 |
| NFR-05 | No function shall exceed 120 lines | **Should** | `build_pdf_report` (481) and the six page renders are decomposed; a test asserts the limit | §1.4 |
| NFR-06 | The application shall be installable as a package | **Should** | `pip install -e .` works and `driftguard` imports without the CWD on `sys.path` | §6 |
| NFR-07 | Keyboard navigation and screen-reader behaviour shall be audited and documented | **Should** | A written accessibility report names the standard tested against and the result — pass or fail | §4.3 |
| NFR-08 | Analysis shall complete within 2 s for 2,500 tickets | **Must (already met)** | Measured 686 ms; a performance test guards against regression | §7.4 |

### 2.3 Documentation requirements

| ID | Deliverable | Priority |
|---|---|---|
| DR-01 | Six UML diagrams reflecting implemented behaviour (ERD as target state, labelled) | **Must** |
| DR-02 | Component and deployment diagrams | **Must** |
| DR-03 | Architecture decision records for every material decision | **Must** |
| DR-04 | Testing and QA strategy document | **Should** |
| DR-05 | Security and deployment document | **Should** |
| DR-06 | Business strategy pack — **requires research not yet performed** | **Should** |
| DR-07 | UX and usability document, separating inspection from user testing | **Should** |

---

## 3. Phased roadmap

Eight phases. Each ends with the full suite green and a commit.

---

### Phase P0 — Correctness and safety

**Rationale:** two confirmed defects and no safety net. Nothing else should be
built on top of these.

| # | Task | Files | Acceptance | Status |
|---|---|---|---|---|
| P0.1 | Escape all user data at HTML boundaries via a shared helper | `ui/safe_html.py` (new), `ui/components.py`, `overview.py`, `review.py`, `sidebar.py`, `app.py` | FR-02; regression test with the proven payload | ✅ **Done** |
| P0.2 | Route uploads through `state._install()` so they register as projects | `ui/sidebar.py`, `ui/state.py`, `driftguard/projects.py` | FR-01; test asserts an uploaded project appears in `ProjectStore` | ✅ **Done** |
| P0.3 | GitHub Actions: pytest on push and PR, Python 3.11/3.12/3.13 matrix | `.github/workflows/ci.yml` | NFR-02; green run on `main` | ⬜ Next |
| P0.4 | Pin runtime dependencies | `requirements.txt` | NFR-03 | ⬜ |
| P0.5 | ADR-0005 escaping policy, ADR-0006 CI | `docs/adr/` | DR-03 | 🟡 0005 done |
| P0.6 | **New.** Neutralise formula-leading cells in CSV exports (finding S-4) | `driftguard/reporting.py`, `driftguard/data.py` | A cell beginning `=`, `+`, `-` or `@` is exported inert; regression test | ⬜ |

**Exit:** 300 tests + new regressions green in CI. **Estimate [A]:** 1 day.

**Progress (20 Nov 2026).** P0.1 and P0.2 implemented and verified. Suite
went 300 → **345 passing**; the original 300 pass unmodified. P0.6 was added
after S-4 was discovered during P0.1 — see audit §7.6.

---

### Phase P1 — Persistence foundation

**Rationale:** the multi-project and history features are half-built without
it, and the audit trail is a positioning requirement, not a nice-to-have.

| # | Task | Files | Acceptance |
|---|---|---|---|
| P1.1 | Define `ProjectRepository` protocol; rename current class `InMemoryProjectStore` behind it | `driftguard/projects.py` | Existing 24 project tests pass **unchanged** |
| P1.2 | Schema + forward-only migrations via `PRAGMA user_version` | `driftguard/persistence/schema.py` | Schema matches audit §8.2; `PRAGMA foreign_keys = ON` asserted |
| P1.3 | `SqliteProjectRepository` implementing the protocol | `driftguard/persistence/sqlite_repo.py` | Same test suite runs against both implementations, parametrised |
| P1.4 | Content-hashed dataset storage | `driftguard/persistence/datasets.py` | Identical baselines stored once; a changed baseline is detected |
| P1.5 | Append-only `review_event` writes on every override | `driftguard/classification.py` (call site), repository | FR-04; test asserts `model_prediction` is never updated |
| P1.6 | Optional wiring: `DRIFTGUARD_DB` env var; absent ⇒ in-memory | `ui/state.py`, new `driftguard/settings.py` | FR-03; with no env var, behaviour is unchanged |
| P1.7 | ADR-0007 SQLite, ADR-0008 repository pattern, ADR-0009 append-only audit | `docs/adr/` | DR-03 |

**Exit:** restart-survival demonstrated; both repositories pass one shared
suite. **Estimate [A]:** 3–4 days.

> **Risk:** persistence is where silent data corruption lives. Mitigation:
> the repository suite is parametrised so the in-memory implementation acts as
> the oracle for the SQLite one.

---

### Phase P2 — Engineering quality

| # | Task | Files | Acceptance |
|---|---|---|---|
| P2.1 | Decompose `build_pdf_report` (481 lines) into section builders | `driftguard/pdf_report.py` | NFR-05; **all 13 PDF tests pass unchanged** — they are the safety net |
| P2.2 | Extract page logic into testable functions; renders become thin | `ui/pages/*.py` | NFR-05 |
| P2.3 | Cache expensive calls (`nested_cross_validate`, `learning_curve`, `cross_validated_scores`) keyed on inputs | `ui/pages/performance.py` | NFR-01; measured <200 ms interaction |
| P2.4 | Page smoke tests via `streamlit.testing.v1.AppTest` (ships with Streamlit — no new dependency) | `tests/test_ui_pages.py` | NFR-04 |
| P2.5 | `pyproject.toml` with package metadata | root | NFR-06 |
| P2.6 | Structured logging | `driftguard/logging_config.py` | FR-05 |
| P2.7 | Performance regression test at 2,500 tickets | `tests/test_performance.py` | NFR-08 |

**Exit:** no function >120 lines; UI has smoke coverage. **Estimate [A]:**
3 days.

---

### Phase P3 — Software modelling

All six diagrams as **Mermaid in Markdown** — version-controlled, diffable, no
binary assets, no new dependency.

| # | Deliverable | File |
|---|---|---|
| P3.1 | Use case diagram + narratives | `docs/modelling/USE_CASES.md` |
| P3.2 | Class/domain model (24 classes) | `docs/modelling/DOMAIN_MODEL.md` |
| P3.3 | Sequence diagrams: analysis path, override loop | `docs/modelling/SEQUENCE.md` |
| P3.4 | ERD — **labelled target state**, matching the P1 schema | `docs/modelling/ERD.md` |
| P3.5 | Activity diagram including validation failure branches | `docs/modelling/ACTIVITY.md` |
| P3.6 | State machines: session lifecycle, ticket classification | `docs/modelling/STATE.md` |

**Acceptance:** every element traces to a `file:line`. A diagram showing
unimplemented behaviour must be labelled target-state (this applies to the ERD
until P1 lands). **Estimate [A]:** 2 days.

---

### Phase P4 — System architecture documentation

| # | Deliverable | File |
|---|---|---|
| P4.1 | Component diagram: modules, dependencies, data flows | `docs/architecture/COMPONENTS.md` |
| P4.2 | Deployment diagram: local, and the SQLite-backed variant | `docs/architecture/DEPLOYMENT.md` |
| P4.3 | Runtime environment and configuration reference | `docs/architecture/RUNTIME.md` |
| P4.4 | ADR index kept current (0001-0004 written during the audit) | `docs/adr/README.md` |

**Estimate [A]:** 1 day.

---

### Phase P5 — Testing, QA, security and deployment docs

| # | Deliverable | File |
|---|---|---|
| P5.1 | Testing strategy: the pyramid, what is deliberately not tested, and why | `docs/quality/TESTING.md` |
| P5.2 | Security assessment: threat model, S-1 remediation, residual risk | `docs/security/SECURITY.md` |
| P5.3 | Deployment and operations guide | `docs/deployment/DEPLOYMENT.md` |
| P5.4 | Dependency vulnerability scan (`pip-audit`, dev-only) | `docs/security/DEPENDENCIES.md` |

**Estimate [A]:** 1.5 days.

---

### Phase P6 — UX and accessibility

| # | Task | Acceptance |
|---|---|---|
| P6.1 | Document the heuristic evaluation from audit §4, clearly labelled as expert inspection, **not** user testing | DR-07 |
| P6.2 | Keyboard navigation audit — record actual results, pass or fail | NFR-07 |
| P6.3 | Screen-reader pass (VoiceOver), semantics for the `unsafe_allow_html` cards | NFR-07 |
| P6.4 | User journey maps for the three actors | DR-07 |
| P6.5 | Fix defects found in P6.2–P6.3 | — |

> **Honesty requirement:** no WCAG conformance may be claimed until tested.
> If the audit fails, the document records the failure.

**Estimate [A]:** 2 days.

---

### Phase P7 — Business strategy

> **Blocked on research not yet performed.** The audit explicitly declined to
> invent competitor findings. This phase begins with verification.

| # | Task | Acceptance |
|---|---|---|
| P7.1 | Competitor capability matrix from **primary sources** (vendor docs), each row cited with URL and access date | Every capability claim has a citation; unverified rows marked "unverified" |
| P7.2 | Red Ocean analysis — the contested market | Depends on P7.1 |
| P7.3 | Blue Ocean analysis — uncontested space, with an ERRC grid | Depends on P7.1 |
| P7.4 | SWOT, with each item traced to audit evidence or a cited source | — |
| P7.5 | Personas and target market. **If no real users are interviewed, personas must be labelled hypothetical** | DR-06 |
| P7.6 | Business Model Canvas | — |
| P7.7 | Unique value proposition and differentiation | — |

**Estimate [A]:** 2 days, assuming desk research only.

---

### Phase P8 — Future integration (not scheduled)

Documented as direction, deliberately not committed:

- Service layer / CLI (FR-06) — prerequisite for anything below
- Project archive export/import (FR-07)
- Jira / Azure DevOps ingestion — **would introduce a network dependency and
  therefore requires an explicit decision to relax offline-first**
- Multi-user deployment — needs authn/authz; changes the product's nature
- Local sentence embeddings for paraphrase, from the prior roadmap

---

## 4. Sequencing and critical path

```
P0 (safety) ──► P1 (persistence) ──► P2 (quality) ──► P8 (future)
                      │                   │
                      └──► P3 (modelling) ─┴──► P4 (architecture) ──► P5 (QA/security docs)

P6 (UX/accessibility) ──── independent, can run in parallel after P0
P7 (business)         ──── independent, research-led, no code dependency
```

**Critical path:** P0 → P1 → P2. **[A]** ≈7–8 working days.
**Full programme [A]:** ≈15–16 working days.

**Why this order:** P0 fixes a proven injection path and a broken feature —
building on either would be building on sand. P1 unblocks the half-built
multi-project and history features and supplies the audit trail the product's
positioning already implies. P3/P4 depend on P1 only because the ERD should
document a schema that exists.

---

## 5. Definition of done

Every task:

1. All **300** pre-existing tests pass, unmodified.
2. New behaviour has a regression test; **fixed bugs have a test that fails
   without the fix**.
3. No new runtime dependency outside the §1.2 budget.
4. No change to the preserved decisions in §1.1.
5. Documentation updated in the same commit as the code.
6. Nothing claimed as working that has not been executed.

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Persistence introduces silent data corruption | Medium | High | Parametrise one suite across both repositories; in-memory is the oracle |
| Refactoring `build_pdf_report` changes output | Medium | Medium | 13 existing PDF tests assert content; they must pass unchanged |
| Caching serves stale results after an override | Medium | High | Key caches on the override dict and settings; test that an override busts the cache |
| Scope creep in the documentation phases | High | Low | Diagrams must trace to `file:line`; no speculative modelling |
| Business phase tempts fabricated findings | Medium | **High** | P7.1 requires citations; unverified rows must say so |
| Accessibility audit fails | Medium | Low | The honest outcome is a recorded failure plus remediation tasks |

**[A]** The fifth risk is the one worth naming plainly: a business-strategy
deliverable creates pressure to produce confident-sounding competitor claims.
The audit set the precedent by refusing to; P7 must hold that line.

---

## 7. Traceability

| Audit finding | Requirement | Phase |
|---|---|---|
| §7.1 injection (proven) | FR-02 | P0.1 ✅ |
| §2.3 uploads not registered | FR-01 | P0.2 ✅ |
| §7.6 CSV formula injection | FR-08 | P0.6 |
| §6 no CI | NFR-02 | P0.3 |
| §6 unpinned dependencies | NFR-03 | P0.4 |
| §6 no persistence | FR-03 | P1.2–P1.6 |
| §7.2 no audit trail | FR-04, FR-05 | P1.5, P2.6 |
| §1.4 481-line function | NFR-05 | P2.1 |
| §4.2 1.43 s recompute | NFR-01 | P2.3 |
| §7.3 UI untested | NFR-04 | P2.4 |
| §6 not packaged | NFR-06 | P2.5 |
| §5 modelling requirements | DR-01 | P3 |
| §6 architecture docs | DR-02 | P4 |
| §4.3 accessibility untested | NFR-07 | P6.2–P6.3 |
| §3.5 no competitor research | DR-06 | P7.1 |
| §6 no API | FR-06 | P8 |

Every audit finding maps to a requirement and a phase; no requirement exists
without a finding behind it.
