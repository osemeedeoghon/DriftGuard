"""Design tokens, global styling and Plotly defaults for the DriftGuard UI.

Presentation only -- nothing here computes an analysis number. Keeping the
tokens in one module means the dashboard reads as one system rather than a
collection of individually styled pages.

Colour decisions are recorded in ``driftguard.config.CLASS_COLOURS`` and in
``SEQUENTIAL_BLUE`` below; both were validated against the light chart surface
rather than picked by eye.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from driftguard.config import CLASS_COLOURS

# --------------------------------------------------------------------------
# Surfaces and ink
# --------------------------------------------------------------------------
PAGE_PLANE = "#f9f9f7"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
HAIRLINE = "rgba(11,11,11,0.10)"

#: Navy/charcoal sidebar.
SIDEBAR_TOP = "#0f172a"
SIDEBAR_BOTTOM = "#1e293b"
SIDEBAR_INK = "#e2e8f0"
SIDEBAR_INK_MUTED = "#94a3b8"
SIDEBAR_BORDER = "rgba(148,163,184,0.18)"

#: Accent. Deliberately not one of the four class colours.
ACCENT = "#2a78d6"
ACCENT_DARK = "#1c5cab"

# --------------------------------------------------------------------------
# Status palette (fixed, never themed, never reused as a series colour)
# --------------------------------------------------------------------------
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

#: Success/failure text ink (distinct from the fills above, for 4.5:1 text).
TEXT_GOOD = "#006300"
TEXT_BAD = "#a31515"

#: Single-hue ramp for magnitude encodings (heatmaps, ranked bars).
SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#0d366b",
]

#: De-emphasis grey, for "one series is the point, the rest are context".
CONTEXT_GREY = "#c9c8c2"

#: Label ink to use *inside* a filled mark.
INSIDE_LABEL_INK: dict[str, str] = {
    CLASS_COLOURS["Refinement"]: INK_PRIMARY,  # yellow needs dark text
}


def label_ink_for(fill: str) -> str:
    """Readable text colour for a label drawn on top of ``fill``."""
    return INSIDE_LABEL_INK.get(fill, "#ffffff")


def rgba(hex_colour: str, alpha: float) -> str:
    """Translucent fill as an ``rgba()`` string.

    Plotly rejects 8-digit ``#RRGGBBAA`` hex for fill properties, so
    translucency has to be expressed this way rather than by appending an
    alpha pair to a hex colour.
    """
    value = hex_colour.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex colour, got {hex_colour!r}.")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


# --------------------------------------------------------------------------
# Drift severity banding
# --------------------------------------------------------------------------
#: Presentational bands for reading the Scope Drift Index at a glance.
#:
#: These thresholds are a reading convention for this dashboard, not a
#: published industry standard, and the UI says so wherever they appear.
SEVERITY_BANDS: tuple[tuple[float, str, str, str], ...] = (
    (0.25, "Critical", STATUS_CRITICAL, "✖"),
    (0.10, "Serious", STATUS_SERIOUS, "▲"),
    (0.05, "Watch", STATUS_WARNING, "●"),
    (-0.10, "Stable", STATUS_GOOD, "✔"),
)


def severity_for(sdi: float) -> tuple[str, str, str]:
    """Return (label, colour, icon) for a Scope Drift Index value.

    Icon and label always accompany the colour, so severity is never carried
    by hue alone.
    """
    for threshold, label, colour, icon in SEVERITY_BANDS:
        if sdi >= threshold:
            return label, colour, icon
    return "De-scoped", ACCENT, "▼"


# --------------------------------------------------------------------------
# Number and date formatting
# --------------------------------------------------------------------------
def fmt_number(value: float, decimals: int = 0) -> str:
    """Thousands-separated fixed-decimal number."""
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}"


def fmt_signed(value: float, decimals: int = 1, suffix: str = "") -> str:
    """Signed number, for deltas, with an explicit + on positives."""
    if value is None:
        return "—"
    return f"{value:+,.{decimals}f}{suffix}"


def fmt_percent(value: float, decimals: int = 1, signed: bool = False) -> str:
    """Fraction rendered as a percentage."""
    if value is None:
        return "—"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:,.{decimals}f}%"


def fmt_date(value) -> str:
    """Dates are rendered one way across the whole dashboard: 07 Nov 2025."""
    if value is None:
        return "—"
    try:
        return value.strftime("%d %b %Y")
    except AttributeError:
        return str(value)


# --------------------------------------------------------------------------
# Plotly template
# --------------------------------------------------------------------------
TEMPLATE_NAME = "driftguard"


def register_template() -> None:
    """Register the shared Plotly template once per process."""
    if TEMPLATE_NAME in pio.templates:
        return

    template = go.layout.Template()
    template.layout = go.Layout(
        font={
            "family": (
                "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
                "Helvetica, Arial, sans-serif"
            ),
            "size": 13,
            "color": INK_SECONDARY,
        },
        title={
            "font": {"size": 15, "color": INK_PRIMARY},
            "x": 0,
            "xanchor": "left",
            "pad": {"b": 12},
        },
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=list(CLASS_COLOURS.values()),
        xaxis={
            "gridcolor": GRIDLINE,
            "linecolor": AXIS_LINE,
            "zerolinecolor": AXIS_LINE,
            "tickfont": {"color": INK_MUTED, "size": 12},
            "title": {"font": {"color": INK_SECONDARY, "size": 12}},
            "showgrid": False,
        },
        yaxis={
            "gridcolor": GRIDLINE,
            "linecolor": AXIS_LINE,
            "zerolinecolor": AXIS_LINE,
            "tickfont": {"color": INK_MUTED, "size": 12},
            "title": {"font": {"color": INK_SECONDARY, "size": 12}},
            "showgrid": True,
            "griddash": "solid",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": INK_SECONDARY, "size": 12},
            "title": {"text": ""},
        },
        margin={"l": 8, "r": 8, "t": 56, "b": 8},
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": HAIRLINE,
            "font": {"color": INK_PRIMARY, "size": 12},
        },
        hovermode="closest",
        colorscale={"sequential": [[i / 8, c] for i, c in enumerate(SEQUENTIAL_BLUE)]},
    )
    pio.templates[TEMPLATE_NAME] = template


def styled(figure: go.Figure, *, height: int | None = None, title: str | None = None) -> go.Figure:
    """Apply the shared template and common layout rules to a figure."""
    register_template()
    figure.update_layout(template=TEMPLATE_NAME)
    if height is not None:
        figure.update_layout(height=height)
    if title is not None:
        figure.update_layout(title=title)
    return figure


# --------------------------------------------------------------------------
# Global CSS
# --------------------------------------------------------------------------
_CSS = f"""
<style>
/* ---------- Page plane ---------- */
.stApp {{ background: {PAGE_PLANE}; }}

.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 1500px;
}}

html, body, [class*="st-"] {{
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Helvetica, Arial, sans-serif;
}}

/* ---------- Navy sidebar ---------- */
section[data-testid="stSidebar"] > div {{
    background: linear-gradient(180deg, {SIDEBAR_TOP} 0%, {SIDEBAR_BOTTOM} 100%);
    border-right: 1px solid {SIDEBAR_BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {SIDEBAR_INK}; }}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    color: {SIDEBAR_INK_MUTED} !important;
    font-size: 0.82rem;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: #ffffff !important;
    letter-spacing: -0.01em;
}}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    color: {SIDEBAR_INK} !important;
    font-weight: 500;
}}
/* Sidebar inputs read as dark-surface controls */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    background-color: rgba(255,255,255,0.06) !important;
    border-color: {SIDEBAR_BORDER} !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {{
    color: {SIDEBAR_INK_MUTED} !important;
}}
/* Secondary buttons (Browse files, Clear project) need a visible edge on dark */
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
    background-color: rgba(255,255,255,0.08) !important;
    border: 1px solid {SIDEBAR_BORDER} !important;
    color: {SIDEBAR_INK} !important;
}}
section[data-testid="stSidebar"] button[kind="secondary"]:hover,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{
    background-color: rgba(255,255,255,0.16) !important;
    border-color: rgba(148,163,184,0.45) !important;
    color: #ffffff !important;
}}
/* Expander headers read as section rows */
section[data-testid="stSidebar"] [data-testid="stExpander"] details {{
    background: rgba(255,255,255,0.04);
    border: 1px solid {SIDEBAR_BORDER};
    border-radius: 8px;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    color: {SIDEBAR_INK} !important;
    font-weight: 560;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
    color: #ffffff !important;
}}
section[data-testid="stSidebar"] hr {{ border-color: {SIDEBAR_BORDER}; }}

/* ---------- Typography ---------- */
h1 {{
    font-size: 1.75rem !important;
    font-weight: 650 !important;
    letter-spacing: -0.02em;
    color: {INK_PRIMARY};
}}
h2 {{
    font-size: 1.18rem !important;
    font-weight: 620 !important;
    letter-spacing: -0.01em;
    color: {INK_PRIMARY};
    margin-top: 0.4rem !important;
}}
h3 {{
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    color: {INK_PRIMARY};
}}

/* ---------- Cards ---------- */
.dg-card {{
    background: {SURFACE};
    border: 1px solid {HAIRLINE};
    border-radius: 10px;
    padding: 1.0rem 1.1rem;
    height: 100%;
}}
.dg-metric-label {{
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {INK_MUTED};
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}}
.dg-metric-value {{
    font-size: 1.85rem;
    font-weight: 660;
    letter-spacing: -0.02em;
    color: {INK_PRIMARY};
    line-height: 1.12;
}}
.dg-metric-delta {{ font-size: 0.84rem; font-weight: 550; margin-top: 0.3rem; }}
.dg-metric-foot {{ font-size: 0.78rem; color: {INK_MUTED}; margin-top: 0.3rem; }}

/* ---------- Hero figure ---------- */
.dg-hero {{
    background: {SURFACE};
    border: 1px solid {HAIRLINE};
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
}}
.dg-hero-value {{
    font-size: 3.2rem;
    font-weight: 680;
    letter-spacing: -0.035em;
    line-height: 1;
    color: {INK_PRIMARY};
}}

/* ---------- Severity pill ---------- */
.dg-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.22rem 0.62rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 620;
    border: 1px solid transparent;
}}

/* ---------- Section header ---------- */
.dg-section {{
    border-top: 1px solid {HAIRLINE};
    margin: 1.9rem 0 0.9rem 0;
    padding-top: 1.1rem;
}}
.dg-section-title {{
    font-size: 1.1rem;
    font-weight: 620;
    letter-spacing: -0.01em;
    color: {INK_PRIMARY};
}}
.dg-section-sub {{ font-size: 0.86rem; color: {INK_SECONDARY}; margin-top: 0.2rem; }}

/* ---------- Assumption / empty state ---------- */
.dg-note {{
    background: #fffdf5;
    border: 1px solid rgba(250,178,25,0.35);
    border-left: 3px solid {STATUS_WARNING};
    border-radius: 8px;
    padding: 0.75rem 0.95rem;
    font-size: 0.86rem;
    color: {INK_SECONDARY};
}}
.dg-empty {{
    background: {SURFACE};
    border: 1px dashed {AXIS_LINE};
    border-radius: 12px;
    padding: 2.4rem 1.5rem;
    text-align: center;
}}
.dg-empty-title {{ font-size: 1.02rem; font-weight: 620; color: {INK_PRIMARY}; }}
.dg-empty-body {{
    font-size: 0.88rem;
    color: {INK_SECONDARY};
    margin-top: 0.4rem;
    max-width: 46rem;
    margin-left: auto;
    margin-right: auto;
}}

/* ---------- Nav radio as a segmented control ---------- */
div[data-testid="stRadio"][aria-label="section-nav"] > div {{ gap: 0.3rem; }}

/* ---------- Tables ---------- */
[data-testid="stDataFrame"] {{ border-radius: 8px; }}

/* ---------- Hide Streamlit chrome for a cleaner demo ---------- */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}
</style>
"""


def inject_css() -> None:
    """Apply the global stylesheet. Safe to call on every rerun."""
    st.markdown(_CSS, unsafe_allow_html=True)
    register_template()
