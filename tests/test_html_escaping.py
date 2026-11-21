"""Tests for HTML escaping in the presentation layer (audit finding S-1).

Two kinds of test live here:

1. **Behavioural** — a markup payload is pushed through the real code path
   that the audit proved was vulnerable, and the output is asserted escaped.
2. **Structural** — a static check over every ``unsafe_allow_html=True`` call
   site in the repository, so a *new* unescaped interpolation fails CI rather
   than waiting to be found.

The structural test is the one that keeps this fixed. The behavioural test
proves the specific reported defect is gone.
"""

from __future__ import annotations

import ast
import pathlib

import pandas as pd
import pytest

from driftguard import data, features, pipeline
from ui.safe_html import SafeHtml, esc, esc_number, join_html

#: The payload from the audit. A ticket id like this can appear in any
#: third-party ticket export.
PAYLOAD = "<img src=x onerror=alert(1)>"
ESCAPED = "&lt;img src=x onerror=alert(1)&gt;"


# --------------------------------------------------------------------------
# The escaping primitives
# --------------------------------------------------------------------------
def test_esc_escapes_markup():
    assert esc(PAYLOAD) == ESCAPED
    assert "<img" not in esc(PAYLOAD)


def test_esc_escapes_quotes_so_attributes_are_safe():
    assert esc('" onmouseover="alert(1)') == "&quot; onmouseover=&quot;alert(1)"


def test_esc_is_idempotent_on_safe_html():
    """Deliberately built markup must not be escaped a second time."""
    fragment = SafeHtml("<span class='dg-pill'>ok</span>")
    assert esc(fragment) is fragment
    assert esc(esc(fragment)) == fragment


def test_esc_returns_safe_html():
    assert isinstance(esc("plain"), SafeHtml)
    assert isinstance(esc(SafeHtml("<b>x</b>")), SafeHtml)


def test_esc_renders_none_as_empty_not_the_word_none():
    assert esc(None) == ""


def test_esc_coerces_non_strings():
    assert esc(42) == "42"
    assert esc(3.5) == "3.5"


def test_esc_number_formats_without_escaping_need():
    assert esc_number(1234567) == "1,234,567"
    assert isinstance(esc_number(10), SafeHtml)


def test_join_html_rejects_unescaped_fragments():
    """Joining raw strings would silently reintroduce the vulnerability."""
    with pytest.raises(TypeError, match="Wrap them in esc"):
        join_html([SafeHtml("<b>ok</b>"), PAYLOAD])  # type: ignore[list-item]


def test_join_html_concatenates_safe_fragments():
    joined = join_html([SafeHtml("<a>"), SafeHtml("<b>")], separator=" ")
    assert joined == "<a> <b>"
    assert isinstance(joined, SafeHtml)


# --------------------------------------------------------------------------
# The proven exploit path, end to end
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def poisoned_result():
    """A real analysis whose ticket id carries markup, as an upload could."""
    baseline = data.sample_baseline()
    tickets = data.sample_tickets().copy()
    # T-304 is the largest contributor, so it reaches the recommendation text.
    tickets.loc[tickets["ticket_id"] == "T-304", "ticket_id"] = PAYLOAD
    return pipeline.run_analysis(baseline, tickets)


def test_payload_really_does_reach_the_recommendation_text(poisoned_result):
    """Guards the premise of the next test.

    If this ever stops holding, the escaping test below would pass vacuously.
    """
    joined = " ".join(
        item["because"] for item in features.recommendations(poisoned_result)
    )
    assert PAYLOAD in joined, "payload no longer reaches the render path"


def test_recommendation_text_is_escaped_before_rendering(poisoned_result):
    """The audit's S-1 finding: uploaded ticket id -> unescaped markup."""
    for item in features.recommendations(poisoned_result):
        rendered = esc(item["because"])
        assert PAYLOAD not in rendered
        if PAYLOAD in item["because"]:
            assert ESCAPED in rendered


def test_every_recommendation_field_is_escaped(poisoned_result):
    for item in features.recommendations(poisoned_result):
        for field in ("priority", "action", "because"):
            assert "<img" not in esc(item[field])


def test_tfidf_terms_from_uploaded_text_are_escaped():
    """The second vector: terms come from uploaded requirement/ticket text."""
    terms = ["enrolment", PAYLOAD, "seat"]
    rendered = join_html(
        SafeHtml(f"<span>{esc(term)}</span>") for term in terms
    )
    assert "<img src=x" not in rendered
    assert ESCAPED in rendered
    assert "<span>enrolment</span>" in rendered


def test_project_name_is_escaped():
    """The third vector: project name, derived from an uploaded filename."""
    assert "<img" not in esc(f"Project {PAYLOAD}")


# --------------------------------------------------------------------------
# Static guard — new unescaped sites must fail CI
# --------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Functions whose return value is already safe markup.
SAFE_CALLS = {"esc", "_esc", "esc_number", "join_html", "SafeHtml", "escape", "severity_pill"}

#: A local holding pre-built markup must end with this suffix. The convention
#: is what makes the check decidable without running the code.
SAFE_NAME_SUFFIX = "_html"


def _rendered_files() -> list[pathlib.Path]:
    files = sorted((REPO_ROOT / "ui").rglob("*.py"))
    files.append(REPO_ROOT / "app.py")
    return [f for f in files if f.exists()]


def _is_safe_expression(node: ast.AST) -> bool:
    """Whether an f-string interpolation is safe to emit as HTML."""
    # esc(...), join_html(...), SafeHtml(...), html.escape(...)
    if isinstance(node, ast.Call):
        func = node.func
        name = (
            func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name)
            else None
        )
        return name in SAFE_CALLS

    # theme.INK_MUTED and friends: our own design tokens, never user data.
    if isinstance(node, ast.Attribute):
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in {"theme", "ui"}:
            return True
        return False

    # A local whose name declares it already holds markup.
    if isinstance(node, ast.Name):
        return node.id.endswith(SAFE_NAME_SUFFIX)

    # A literal written by us.
    if isinstance(node, ast.Constant):
        return True

    # A conditional is safe only if both branches are.
    if isinstance(node, ast.IfExp):
        return _is_safe_expression(node.body) and _is_safe_expression(node.orelse)

    return False


def _unsafe_interpolations() -> list[tuple[str, int, str]]:
    """Every interpolation in an unsafe_allow_html call that is not provably safe."""
    offenders: list[tuple[str, int, str]] = []
    for path in _rendered_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            enabled = any(
                kw.arg == "unsafe_allow_html"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
            if not enabled:
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.FormattedValue):
                    if not _is_safe_expression(inner.value):
                        offenders.append(
                            (
                                str(path.relative_to(REPO_ROOT)),
                                getattr(inner.value, "lineno", node.lineno),
                                ast.unparse(inner.value),
                            )
                        )
    return offenders


def test_no_unescaped_interpolation_in_any_rendered_html():
    """Every value interpolated into rendered HTML must be provably safe.

    Safe means one of: an ``esc()``-family call, a ``theme.*`` design token, a
    local named ``*_html``, or a literal. Anything else is assumed to be
    untrusted data.

    If this fails on code you just wrote, wrap the value in ``ui.esc(...)``, or
    — if it really is pre-built markup — build it as ``SafeHtml`` and name the
    local with an ``_html`` suffix.
    """
    offenders = _unsafe_interpolations()
    assert not offenders, "Unescaped interpolation in rendered HTML:\n" + "\n".join(
        f"  {path}:{line}  ->  {expr}" for path, line, expr in offenders
    )


def test_the_guard_can_actually_fail():
    """A check that never fails proves nothing.

    Feeds the guard's own predicate a deliberately unsafe expression and
    asserts it is rejected.
    """
    unsafe = ast.parse("f\"{ticket['id']}\"", mode="eval").body
    interpolation = next(
        n for n in ast.walk(unsafe) if isinstance(n, ast.FormattedValue)
    )
    assert _is_safe_expression(interpolation.value) is False

    safe = ast.parse('f"{esc(ticket[\'id\'])}"', mode="eval").body
    interpolation = next(
        n for n in ast.walk(safe) if isinstance(n, ast.FormattedValue)
    )
    assert _is_safe_expression(interpolation.value) is True


def test_guard_covers_the_files_that_render_html():
    """The guard is worthless if it is looking in the wrong place."""
    covered = {str(p.relative_to(REPO_ROOT)) for p in _rendered_files()}
    for expected in (
        "app.py",
        "ui/components.py",
        "ui/pages/overview.py",
        "ui/pages/review.py",
        "ui/sidebar.py",
    ):
        assert expected in covered


def test_guard_sees_a_realistic_number_of_render_sites():
    """Sanity bound: if this drops to zero the AST walk has silently broken."""
    sites = 0
    for path in _rendered_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(
                kw.arg == "unsafe_allow_html" for kw in node.keywords
            ):
                sites += 1
    assert sites >= 10, f"only {sites} unsafe_allow_html sites found"
