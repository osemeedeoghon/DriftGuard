# ADR-0001: Separate analysis from presentation

## Status
Accepted. Retrospective — recorded during the November 2026 audit.

## Context
DriftGuard is a Streamlit application. The path of least resistance in
Streamlit is to compute inside the page that displays the result, because
Streamlit re-runs the script top to bottom and there is no framework pressure
to do otherwise.

That path makes analytical code untestable: verifying a number requires a
browser and a running server, so in practice it does not get verified. For a
tool whose output is a governance argument about money and schedule, numbers
that cannot be tested are numbers that cannot be defended.

## Decision
All analysis lives in the `driftguard` package, which **must not import
Streamlit**. The `ui` package renders what `driftguard` returns and computes no
analytical figure itself.

## Alternatives considered
- **Compute in the page.** Rejected: untestable, and duplicated logic between
  the dashboard and the report writers.
- **Notebook-first with a thin UI.** Rejected: notebooks are poor at
  regression testing and at the human-review workflow the product needs.

## Consequences
**Makes easy**
- 300 tests run headlessly in ~15 s with no browser.
- Markdown and PDF reports reuse the same computation as the screen, so they
  cannot disagree with it.
- Any figure is reproducible in three lines of Python.

**Makes hard / costs**
- Two layers to navigate for a change that spans both.
- The UI layer has *no* automated coverage, because page `render()` functions
  need a Streamlit runtime. Audit §7.3 records this as the main reliability
  gap; roadmap NFR-04 addresses it.

**Verification**
`grep -rn "import streamlit" driftguard/` returns no matches. This is the
cheapest possible check and should be enforced in CI.
