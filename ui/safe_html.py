"""HTML escaping for the presentation layer.

Why this module exists
----------------------
Streamlit's ``unsafe_allow_html=True`` emits a string into the DOM verbatim.
DriftGuard renders user-supplied text — ticket identifiers, titles, project
names, TF-IDF terms taken from uploaded files — into those strings. Before
this module existed, ``ui/components.py`` escaped consistently while the page
modules escaped nothing, and a ticket id of ``<img src=x onerror=alert(1)>``
reached the DOM unescaped (audit finding S-1).

The threat model is not "a user attacks their own machine". It is an analyst
running DriftGuard over a ticket export supplied by a client or a vendor. That
file is untrusted input.

The marker-type approach
------------------------
Escaping at the render site alone is fragile, because some interpolated values
are *deliberately* HTML — a severity pill, a pre-built card fragment — and
escaping those would render markup as visible angle brackets.

So :class:`SafeHtml` marks a string as already safe, and :func:`esc` is
idempotent: it passes ``SafeHtml`` through untouched and escapes everything
else. Any value that has not been through one of those two paths is, by
construction, unescaped data — which is exactly what the guard test in
``tests/test_html_escaping.py`` looks for.

Conventions this module relies on
---------------------------------
* Every function that returns markup returns :class:`SafeHtml`.
* Local variables holding pre-built markup are named with an ``_html`` suffix.
* Everything else interpolated into an ``unsafe_allow_html`` string is either
  ``esc(...)`` or a design token from :mod:`ui.theme`.

The guard test enforces all three statically, so a new unescaped site fails CI
rather than waiting to be noticed.
"""

from __future__ import annotations

import html
from typing import Iterable


class SafeHtml(str):
    """A string that is known to be safe to emit as HTML.

    Subclasses ``str`` so it can be interpolated into an f-string and passed
    to Streamlit with no further ceremony. The type is the whole point: it
    records that a value has already been escaped or was deliberately
    constructed as markup.

    Never construct this from unescaped user input. Use :func:`esc`.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SafeHtml({str.__repr__(self)})"


def esc(value: object) -> SafeHtml:
    """Escape ``value`` for inclusion in HTML text or an attribute.

    Idempotent: a :class:`SafeHtml` passes through unchanged, so wrapping an
    already-safe fragment is harmless. Everything else is coerced with ``str``
    and escaped, quotes included, which makes the result safe inside a quoted
    attribute as well as in text.

    ``None`` becomes an empty string rather than the text "None", because the
    alternative is a dashboard that prints "None" where a value is absent.
    """
    if isinstance(value, SafeHtml):
        return value
    if value is None:
        return SafeHtml("")
    return SafeHtml(html.escape(str(value), quote=True))


def esc_number(value: float | int, spec: str = ",") -> SafeHtml:
    """Format a number and mark it safe.

    A formatted number cannot contain markup, but routing it through this
    helper keeps every interpolation in a rendered string accounted for, so
    the guard test does not need a special case for arithmetic.
    """
    return SafeHtml(format(value, spec))


def join_html(fragments: Iterable[SafeHtml], separator: str = "") -> SafeHtml:
    """Concatenate already-safe fragments.

    Takes ``SafeHtml`` specifically: joining raw strings here would silently
    reintroduce the problem this module exists to prevent.
    """
    parts = list(fragments)
    unsafe = [p for p in parts if not isinstance(p, SafeHtml)]
    if unsafe:
        raise TypeError(
            "join_html expects SafeHtml fragments; received "
            f"{len(unsafe)} unescaped value(s). Wrap them in esc() first."
        )
    return SafeHtml(separator.join(parts))
