# ADR-0002: REVL charges churn only, excluding new scope

## Status
Accepted. Retrospective — recorded during the November 2026 audit.

## Context
COCOMO II's REVL factor inflates size inside the effort equation to account for
requirements evolution:

```
Size_effective = Size x (1 + REVL/100)
```

DriftGuard already computes a Scope Drift Index, so the obvious implementation
is to feed the SDI straight into REVL. That is what the first implementation
did, and it was wrong.

The revised size **already contains** the newly added scope. Inflating it again
by the same percentage charges for that work twice. On the sample project it
reported **+246% effort for +76% size growth** — a figure that would not
survive five minutes of scrutiny from a sponsor.

## Decision
REVL models **rework**, not growth. DriftGuard computes:

```
REVL% = 100 x (refinement-driven additions + removals) / baseline size
```

- **Refinements** are requirements that moved after being agreed — the textbook
  definition of volatility.
- **Removals** are work that was planned, possibly partly built, then dropped:
  the size leaves the revised total but the effort spent does not come back.
- **New scope is excluded**, because the larger revised size already carries it.

On the sample project this yields 27.4% rather than 75.8%, and effort growth of
+142.7% rather than +246%.

## Alternatives considered
- **REVL = SDI.** Rejected: double-counts, as above.
- **REVL = 0 always.** Rejected: understates the real cost of churn; a
  requirement rewritten three times genuinely costs more than one written once.
- **Make it a free parameter.** Rejected as the default — it would hide the
  modelling question behind a slider. It *is* available as an override.

## Consequences
**Makes easy**
- The effort figure is defensible: no quantity is charged twice.
- The reasoning is explainable in two sentences to a non-technical sponsor.

**Makes hard / costs**
- Removals are charged at full size, which assumes the work was built before
  being dropped. Where removals were cancelled before starting, this
  **overstates** rework. Stated in the UI, the README and the reports rather
  than hidden; parameterising it is a roadmap item.

**Protection**
`test_revl_excludes_new_scope_because_revised_size_already_carries_it` exists
solely to stop the double-count returning. The Methodology page re-derives the
figure live on whatever data is loaded.
