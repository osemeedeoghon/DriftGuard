# ADR-0004: Report evaluation leakage rather than suppress it

## Status
Accepted. Retrospective — recorded during the November 2026 audit.

## Context
DriftGuard's rule classifier scores 95.6% accuracy / macro-F1 0.960 on the
45-ticket labelled set. That figure is in-sample: the similarity thresholds and
the keyword lexicon were both tuned on the same tickets used to score them.

Nested cross-validation was added to obtain an honest estimate. It returned a
**perfect** held-out macro-F1 of **1.000**, against 0.960 in-sample.

Held-out performance cannot legitimately exceed training performance. A result
like that is not a success; it is a symptom.

## Decision
When mean held-out performance meets or exceeds in-sample performance, the
tool **flags the run as a leakage indicator and refuses to present it as a
performance result**. It reports the mechanism alongside the flag: how many
decisions rested on the un-leaked similarity signal versus the leaked lexicon.

## Alternatives considered
- **Report 1.000 as the headline.** Rejected: it would be the most flattering
  number in the project and the least true.
- **Suppress the nested CV entirely.** Rejected: hiding a diagnostic because it
  is awkward is worse than showing it.
- **Keep tuning until the numbers look conventional.** Rejected: that is
  fitting the evaluation to the expectation.

## Consequences
**Makes easy**
- The tool is honest about its own validity, which is the same property it
  claims to give its users about their projects.
- The mechanism is measurable and shown: lowering the weak-match threshold
  shifts decisions from similarity (12 to 5 tickets) onto the hand-written
  lexicon (33 to 40).
- It justifies shipping the **more conservative** threshold (0.18) over the
  0.10 the search prefers.

**Makes hard / costs**
- There is no clean accuracy figure for the default classifier, and there
  cannot be one without an independently labelled corpus from another project.
  That is recorded as the highest-value future improvement.
- A reader skimming for a headline number will not find a comfortable one.
  Accepted deliberately.

**Protection**
`test_leakage_is_detected_on_the_sample_project` asserts the flag fires on this
dataset. If the scores ever separate properly, that test fails and forces the
dashboard wording to be revisited.
