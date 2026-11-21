# ADR-0005: HTML escaping policy at render boundaries

## Status
Accepted — 20 November 2026. Implements roadmap task P0.1.

## Context
DriftGuard styles its dashboard with `st.markdown(..., unsafe_allow_html=True)`,
which emits a string into the DOM verbatim. There are 16 such call sites.

The November 2026 audit found escaping applied inconsistently: `ui/components.py`
escaped on every path (16 `html.escape` calls), while the page modules and
`app.py` escaped **nothing** across 7 call sites. The audit proved the
consequence by executing it — a ticket id of `<img src=x onerror=alert(1)>`
flowed from an uploaded CSV through `features.recommendations()` into
`ui/pages/overview.py` and reached the DOM unescaped (finding S-1).

Two further vectors existed: TF-IDF terms rendered as pills in
`ui/pages/review.py`, and the project name in the `app.py` context bar.

The threat model is not "a user attacks their own machine". It is an analyst
running DriftGuard over a ticket export supplied by a client or a vendor. That
file is untrusted input, and the roadmap contemplates deployments where more
than one person sees the result.

The obvious fix — escape at each render site — is fragile, because some
interpolated values are *deliberately* markup. A severity pill or a pre-built
card fragment must not be escaped, or the user sees angle brackets instead of a
badge. A policy of "escape everything" would break the UI; a policy of "escape
the data ones" relies on every future author correctly telling them apart.

## Decision
Introduce a `SafeHtml` marker type in `ui/safe_html.py`.

1. `SafeHtml` is a `str` subclass meaning "this value is safe to emit as HTML".
2. `esc(value)` is **idempotent**: it returns `SafeHtml` unchanged and escapes
   everything else, quotes included, so the result is safe in an attribute as
   well as in text.
3. Every function returning markup returns `SafeHtml` — `severity_pill` does.
4. Locals holding pre-built markup are named with an `_html` suffix.
5. `join_html` accepts only `SafeHtml` and raises on raw strings, so
   concatenation cannot silently reintroduce the problem.

Anything interpolated into a rendered string that is not an `esc()`-family
call, a `theme.*` design token, an `*_html` local, or a literal is, by
construction, unescaped data.

That property is **enforced statically**. `tests/test_html_escaping.py` walks
the AST of every file that renders HTML, finds each `unsafe_allow_html=True`
call, and asserts that every f-string interpolation inside it is provably safe
by those rules. A new unescaped interpolation fails CI rather than waiting to
be noticed.

## Alternatives considered

- **Escape at each call site, no marker type.** Rejected: it cannot express
  "this value is intentionally markup", so either badges break or the policy
  becomes a matter of author judgement — which is what failed the first time.
- **Adopt `markupsafe` / Jinja.** Rejected on the roadmap's dependency budget.
  `markupsafe` would be a better-tested version of 40 lines of standard-library
  code, but the surface we need is `escape` plus a marker type, and adding a
  runtime dependency for that is poor value.
- **Stop using `unsafe_allow_html`.** Rejected: it would mean abandoning the
  metric cards, severity pills and layout that carry the dashboard's design.
  The audit's own §4 records that visual hierarchy as a strength.
- **Sanitise on input instead of on output.** Rejected: escaping belongs at the
  boundary where the context is known. The same ticket id is safe in a CSV
  export and unsafe in HTML; stripping it at load would corrupt the data for
  every other consumer.

## Consequences

**Makes easy**
- Rendering untrusted text is safe by default, and *provably* so.
- The guard test localises the rule: a reviewer does not have to remember it,
  because CI remembers for them.
- `esc()` being idempotent means defensive double-wrapping is harmless, so the
  safe action is always available.

**Makes hard / costs**
- A naming convention (`*_html`) is now load-bearing. A local holding markup
  under a different name fails the guard. The failure message says what to do,
  which is the mitigation.
- The guard is deliberately conservative: it demands proof rather than
  inferring safety through assignments. This forced `ui.esc(colour)` on a value
  holding a theme constant, where escaping is a no-op. Uniformity was judged
  worth the redundant call, because a checker that tracks assignment flow is
  substantially more complex and fails open when it is wrong.
- `SafeHtml` subclasses `str`, so ordinary string operations on it return plain
  `str` and lose the marking. That is the conservative direction — the result
  gets escaped again — but it can surprise.

**Verification**
- `tests/test_html_escaping.py` — 18 tests. Includes the audit's exact payload
  driven through the real `recommendations()` path, a test asserting the
  payload still *reaches* that path (so the escaping test cannot pass
  vacuously), and a test that the guard's own predicate rejects unsafe input.
- Confirmed in the running application: with a ticket id carrying the payload
  uploaded through the UI, the rendered DOM contained **zero** injected `<img>`
  elements and displayed the payload as literal text.

## Residual risk
This ADR addresses HTML injection in the dashboard. It does **not** address
CSV injection in exported files — a cell beginning `=`, `+`, `-` or `@` can be
interpreted as a formula by a spreadsheet application opening a DriftGuard CSV
export. That is a separate finding, not covered here, and should be raised as
its own roadmap item.
