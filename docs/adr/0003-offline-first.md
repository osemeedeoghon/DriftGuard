# ADR-0003: Offline-first, no network client

## Status
Accepted. Retrospective — recorded during the November 2026 audit.

## Context
The original brief imposed a hard constraint: no paid LLM API, API key, cloud
database or paid hosting in the execution path. The obvious implementation of
ticket-to-requirement matching in 2026 is an embedding API or an LLM call.

Beyond cost, there are two engineering reasons to refuse that: an LLM's
classification cannot be inspected term by term, and a network dependency makes
the tool unusable with client data that cannot leave a machine.

## Decision
No component of DriftGuard may make a network call. Matching uses local TF-IDF;
classification uses a visible keyword lexicon and a local logistic regression;
reports are rendered locally.

## Alternatives considered
- **Embedding API for semantic matching.** Rejected: cost, and it would make
  the evidence panel impossible — there is no "these terms drove the match" for
  a dense vector.
- **Local transformer model.** Not rejected on principle, and it is on the
  roadmap as a future improvement, but it would add a large dependency and
  weaken explainability. Not for the first version.

## Consequences
**Makes easy**
- Runs on an air-gapped machine; usable with confidential client data.
- Zero running cost; no key management; no rate limits.
- Every classification is inspectable down to the individual word.
- Fast: 686 ms for 2,500 tickets x 300 requirements (measured).

**Makes hard / costs**
- **TF-IDF cannot see paraphrase.** A ticket that restates a requirement in
  different words scores low and is misread as new scope. Both residual errors
  on the sample project are exactly this.
- The keyword lexicon is domain-specific and does not transfer between
  projects without re-tuning.

**Boundary**
Any future integration (Jira, Azure DevOps) would breach this ADR and requires
an explicit superseding decision, not an incremental commit.

**Verification**
`grep -rnE "import (requests|urllib|http|openai|anthropic)" driftguard/ ui/ app.py`
returns no matches.
