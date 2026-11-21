# ADR-0006: Continuous integration and exact dependency pinning

## Status
Accepted — 20 November 2026. Implements roadmap tasks P0.3 and P0.4.

## Context
Two audit findings in §6 share a root cause: the project had no mechanism that
ran anything automatically.

**No CI.** The suite ran when a human remembered to run it. That was tolerable
while the tests were purely behavioural, but P0.1 added a *static* guard — an
AST walk asserting that no `unsafe_allow_html` site interpolates unescaped
data. A static guard that nothing executes is decoration. The same applies to
the offline-first rule, which until now was a convention enforced by reviewer
attention.

**No pinning.** `requirements.txt` declared ranges (`pandas>=2.0`,
`streamlit>=1.30`). A fresh environment could therefore resolve a different
dependency set from the one the 345 tests were verified against, which makes
"the tests pass" a claim about nobody's machine in particular. It also makes a
bisect unreliable: a historical commit might not reproduce because its
transitive dependencies moved.

A third problem was latent: the project claims to support Python 3.11+, but
every test run to date had been on a single interpreter, 3.13.9. The claim was
untested.

## Decision

**Pin exactly.** `requirements.txt` pins each runtime dependency with `==` to
the version the suite was verified against. Test tooling moves to a new
`requirements-dev.txt`, which includes the runtime file, so a contributor
installs one thing and a deployment installs neither pytest nor a PDF parser.

**Run CI on GitHub Actions** for push and pull request against `main`, across
a matrix of Python **3.11, 3.12 and 3.13** with `fail-fast: false`, so one
incompatibility does not hide the state of the others. A second, dependency-free
job greps for a network client and fails the build if one appears.

**Enforce both in the suite.** `tests/test_requirements.py` asserts that every
requirement line is exactly pinned, that each pin matches the version actually
installed, that no forbidden network client is declared, and that the workflow
exists, triggers on push and PR, covers every supported Python version and
installs from the pinned file. A regression in the build configuration now
fails the tests rather than waiting for a broken build.

## Alternatives considered

- **Leave ranges, add a lockfile (`pip-tools`, `uv`, Poetry).** Rejected for
  now. A lockfile is the better long-term answer because it pins transitive
  dependencies too, which `==` on direct dependencies does not. It was rejected
  because it adds a tool and a workflow step for a project with seven runtime
  dependencies and no deployment pipeline. Recorded as a future improvement;
  this ADR should be superseded when one is adopted.
- **Pin to the latest release of everything.** Rejected: the pins must record
  what was *tested*, not what was newest at the time of writing. Pinning to
  something unverified would invert the point.
- **Test a single Python version.** Rejected: the project documents 3.11+
  support, and an untested claim is one the README should not make.
- **Self-hosted or scheduled runs.** Rejected as unnecessary: the repository is
  small and public, and GitHub-hosted runners are free for it.
- **Include `pytest` in `requirements.txt`.** Rejected: it would ship a test
  runner to every deployment and blur what the application actually needs.

## Consequences

**Makes easy**
- A fresh clone plus `pip install -r requirements-dev.txt` reproduces the
  verified environment exactly.
- The P0.1 escaping guard and the offline-first rule are now enforced by
  machine rather than by memory.
- The 3.11+ support claim is tested on every push instead of assumed.
- A failing CI run prints `pip list`, so the resolved environment is visible
  in the log rather than inferred from the pins.

**Makes hard / costs**
- **Upgrades become deliberate.** A dependency update is now an explicit commit
  that changes a pin, and `test_pinned_runtime_version_matches_the_installed_one`
  will fail until the local environment matches. That friction is the intended
  behaviour, but it is friction.
- **Transitive dependencies remain unpinned.** `==` on direct dependencies does
  not constrain what *they* pull in, so builds are substantially — not
  perfectly — reproducible. This is the honest limit of the approach and the
  reason a lockfile is recorded above as the eventual answer.
- Security patches in dependencies will not arrive on their own. A periodic
  review, or Dependabot, is the mitigation; neither is set up yet.

**Verification**
- `tests/test_requirements.py` — 20 tests, including a test that the pin regex
  itself rejects ranges, so the guard cannot pass vacuously.
- A fresh virtual environment was built from `requirements-dev.txt` and the
  full suite run inside it.
- The GitHub Actions run was watched to completion and its result recorded in
  the commit that introduced it.
