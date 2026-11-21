# Architecture Decision Records

An ADR records *why* a decision was made, what was rejected, and what it costs.

Two kinds live here:

- **Retrospective** (0001–0004): decisions already embodied in the code,
  written down during the November 2026 audit so the reasoning survives.
- **Accepted as work lands** (0005–0006): decisions taken while implementing
  the roadmap.
- **Forward** (0007+): decisions scheduled in the
  [roadmap](../DEVELOPMENT_ROADMAP.md), to be written when taken.

## Index

| ADR | Title | Status | Phase |
|---|---|---|---|
| [0001](0001-analysis-presentation-separation.md) | Separate analysis from presentation | Accepted | Retrospective |
| [0002](0002-revl-charges-churn-only.md) | REVL charges churn only, excluding new scope | Accepted | Retrospective |
| [0003](0003-offline-first.md) | Offline-first, no network client | Accepted | Retrospective |
| [0004](0004-report-leakage-rather-than-suppress.md) | Report evaluation leakage rather than suppress it | Accepted | Retrospective |
| [0005](0005-html-escaping-policy.md) | HTML escaping policy at render boundaries | **Accepted** | P0.1 |
| [0006](0006-continuous-integration.md) | Continuous integration and exact dependency pinning | **Accepted** | P0.3, P0.4 |
| 0007 | SQLite for optional durable persistence | Proposed | P1.7 |
| 0008 | Repository pattern for the project store | Proposed | P1.7 |
| 0009 | Append-only audit events | Proposed | P1.7 |

## Format

```markdown
# ADR-NNNN: Title

## Status
Proposed | Accepted | Superseded by ADR-NNNN

## Context
The forces at play. What made a decision necessary.

## Decision
What was decided, stated plainly.

## Alternatives considered
What was rejected, and why.

## Consequences
What this makes easy, what it makes hard, and what it costs.
```

An ADR is immutable once accepted. A changed decision is a new ADR that
supersedes the old one, so the history of reasoning stays readable.
