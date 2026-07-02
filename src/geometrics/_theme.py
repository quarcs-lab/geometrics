"""Shared Plotly styling for geometrics figures.

This module centralizes the visual identity of every geometrics figure so the look is
consistent across notebooks, scripts, static exports and the apps:

* a **Tableau 10** qualitative palette for grouped series (:data:`COLOR_SEQUENCE`),
* cohesive Tableau-style continuous scales (:data:`DIVERGING_SCALE`,
  :data:`SEQUENTIAL_SCALE`),
* a presentation-friendly font stack and sizes (Arial/Helvetica, larger labels),
* a registered Plotly template (``"geometrics"``) layered on ``plotly_white`` and set as the
  default, so figures are styled even when a caller forgets :func:`apply_default_layout`,
* a high-resolution export config for crisp slide-ready PNGs (:data:`PLOTLY_CONFIG`).
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

__all__ = [
    "COLOR_SEQUENCE",
    "COLOR_SEQUENCE_COLORBLIND",
    "DIVERGING_SCALE",
    "DIVERGING_SCALE_COLORBLIND",
    "FONT_FAMILY",
    "FONT_SIZE_AXIS_TITLE",
    "FONT_SIZE_BASE",
    "FONT_SIZE_LEGEND",
    "FONT_SIZE_TICK",
    "FONT_SIZE_TITLE",
    "PLOTLY_CONFIG",
    "SEQUENTIAL_SCALE",
    "SEQUENTIAL_SCALE_COLORBLIND",
    "TEMPLATE_NAME",
    "TEMPLATE_NAME_DARK",
    "active_colorway",
    "active_diverging_scale",
    "active_sequential_scale",
    "apply_default_layout",
    "color_for",
    "diverging_color",
    "get_palette",
    "set_palette",
    "LISA_COLORS",
    "CLUB_COLORS",
    "MAP_SEQUENTIAL",
    "MAP_DIVERGING",
]

# --- Qualitative palette -------------------------------------------------------------
# The classic Tableau 10 palette: distinct, muted, and well-suited to projection on
# presentation slides. Used for grouped series via :func:`color_for`.
COLOR_SEQUENCE: list[str] = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#76B7B2",  # teal
    "#EDC948",  # yellow
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
]

# --- Continuous color scales ---------------------------------------------------------
# A Tableau-flavoured diverging scale (red <-> light neutral <-> blue), anchored at a
# near-white midpoint. Drives the correlation heatmap and the ellipse fill (see
# :func:`diverging_color`) so both styles look the same.
DIVERGING_SCALE: list[list[float | str]] = [
    [0.0, "#E15759"],  # strong negative -> Tableau red
    [0.25, "#F1A7A9"],
    [0.5, "#F5F5F5"],  # zero -> near-white
    [0.75, "#9FB8D4"],
    [1.0, "#4E79A7"],  # strong positive -> Tableau blue
]

# A Tableau-flavoured sequential blue ramp (light -> Tableau blue) for magnitude-only
# encodings such as the missing-values heatmap and continuous scatter color.
SEQUENTIAL_SCALE: list[list[float | str]] = [
    [0.0, "#F7FBFF"],
    [0.25, "#C6DBEF"],
    [0.5, "#90B5D6"],
    [0.75, "#5C8FBC"],
    [1.0, "#2E5C8A"],
]

# --- Colorblind-safe palette (opt-in via set_palette("colorblind")) -------------------
# The Okabe-Ito qualitative set (8 colors): the canonical colorblind-safe categorical
# palette (Okabe & Ito 2008). ``color_for`` wraps modulo its length, so 8 < 10 is fine.
COLOR_SEQUENCE_COLORBLIND: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#CC79A7",  # reddish purple
    "#999999",  # gray
]

# A colorblind-safe single-hue blue sequential ramp (monotonic in lightness).
SEQUENTIAL_SCALE_COLORBLIND: list[list[float | str]] = [
    [0.0, "#F7FBFF"],
    [0.25, "#C6DBEF"],
    [0.5, "#6BAED6"],
    [0.75, "#2171B5"],
    [1.0, "#08306B"],
]

# A colorblind-safe diverging scale (orange <-> neutral <-> blue), avoiding red/green so
# protan/deutan viewers can read sign.
DIVERGING_SCALE_COLORBLIND: list[list[float | str]] = [
    [0.0, "#D55E00"],  # strong negative -> Okabe-Ito vermillion
    [0.25, "#F0B488"],
    [0.5, "#F5F5F5"],  # zero -> near-white
    [0.75, "#80B1D3"],
    [1.0, "#0072B2"],  # strong positive -> Okabe-Ito blue
]

# --- Active-palette state ------------------------------------------------------------
# The public COLOR_SEQUENCE / SEQUENTIAL_SCALE / DIVERGING_SCALE constants are the frozen
# "default" snapshot. The *active* palette is chosen at runtime by :func:`set_palette` and
# read through the accessor functions below — so a toggle reaches modules that imported the
# scale constants by value at import time (they must call the accessors, not the constants).
_QUALITATIVE: dict[str, list[str]] = {
    "default": COLOR_SEQUENCE,
    "colorblind": COLOR_SEQUENCE_COLORBLIND,
}
_SEQUENTIAL: dict[str, list[list[float | str]]] = {
    "default": SEQUENTIAL_SCALE,
    "colorblind": SEQUENTIAL_SCALE_COLORBLIND,
}
_DIVERGING: dict[str, list[list[float | str]]] = {
    "default": DIVERGING_SCALE,
    "colorblind": DIVERGING_SCALE_COLORBLIND,
}
_ACTIVE_PALETTE: str = "default"


def active_colorway() -> list[str]:
    """Return the currently active qualitative palette (list of hex strings)."""
    return _QUALITATIVE[_ACTIVE_PALETTE]


def active_sequential_scale() -> list[list[float | str]]:
    """Return the currently active sequential color scale."""
    return _SEQUENTIAL[_ACTIVE_PALETTE]


def active_diverging_scale() -> list[list[float | str]]:
    """Return the currently active diverging color scale."""
    return _DIVERGING[_ACTIVE_PALETTE]


# --- Fonts -------------------------------------------------------------------------
# A modern sans (Inter) with system + Arial/Helvetica fallbacks: browsers and notebooks pick
# up the refreshed look, while the Arial fallback keeps static (Kaleido) exports identical
# across machines. Sizes follow a "presentation" tier so labels stay legible on slides.
FONT_FAMILY: str = (
    "Inter, -apple-system, 'Segoe UI', Roboto, Arial, Helvetica, sans-serif"
)
FONT_SIZE_BASE: int = 16
FONT_SIZE_TICK: int = 15
FONT_SIZE_AXIS_TITLE: int = 18
FONT_SIZE_TITLE: int = 22
FONT_SIZE_LEGEND: int = 15

TEMPLATE_NAME: str = "geometrics"
TEMPLATE_NAME_DARK: str = "geometrics_dark"

# Modebar / export config: emit crisp 2x PNGs suitable for slides.
PLOTLY_CONFIG: dict[str, object] = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "geometrics_figure",
        "scale": 2,
    },
}


def _build_template(*, dark: bool = False) -> go.layout.Template:
    """Construct a ``geometrics`` Plotly template (light by default, dark when ``dark=True``)."""
    font_color = "#e6e6e6" if dark else "#2a2a2a"
    # Softer gridlines with a slightly stronger zeroline reads as cleaner and more modern.
    grid = "rgba(255,255,255,0.10)" if dark else "rgba(0,0,0,0.06)"
    zeroline = "rgba(255,255,255,0.28)" if dark else "rgba(0,0,0,0.18)"
    legend_bg = "rgba(0,0,0,0.35)" if dark else "rgba(255,255,255,0.6)"
    hover_bg = "rgba(30,30,30,0.92)" if dark else "rgba(255,255,255,0.95)"
    axis = {
        "title": {"font": {"family": FONT_FAMILY, "size": FONT_SIZE_AXIS_TITLE}},
        "tickfont": {"family": FONT_FAMILY, "size": FONT_SIZE_TICK},
        "automargin": True,
        "gridcolor": grid,
        "zerolinecolor": zeroline,
        "zerolinewidth": 1,
    }
    template = go.layout.Template()
    template.layout = go.Layout(
        font={"family": FONT_FAMILY, "size": FONT_SIZE_BASE, "color": font_color},
        title={"font": {"family": FONT_FAMILY, "size": FONT_SIZE_TITLE}, "x": 0.02},
        colorway=active_colorway(),
        colorscale={
            "sequential": active_sequential_scale(),
            "diverging": active_diverging_scale(),
        },
        margin={"l": 70, "r": 30, "t": 60, "b": 60},
        legend={
            "bgcolor": legend_bg,
            "borderwidth": 0,
            "itemsizing": "constant",
            "font": {"family": FONT_FAMILY, "size": FONT_SIZE_LEGEND},
            "title": {"font": {"family": FONT_FAMILY, "size": FONT_SIZE_LEGEND}},
        },
        hoverlabel={
            "bgcolor": hover_bg,
            "bordercolor": grid,
            "align": "left",
            "font": {
                "family": FONT_FAMILY,
                "size": FONT_SIZE_TICK,
                "color": font_color,
            },
        },
        xaxis=dict(axis),
        yaxis=dict(axis),
    )
    # Shared colorbar geometry so heatmaps (correlation / missing / value) look uniform.
    template.data.heatmap = [
        go.Heatmap(colorbar={"thickness": 14, "len": 0.85, "outlinewidth": 0})
    ]
    return template


# Register the light + dark templates and make ``plotly_white + geometrics`` the process-wide
# default so even figures that bypass ``apply_default_layout`` pick up the geometrics look.
pio.templates[TEMPLATE_NAME] = _build_template()
pio.templates[TEMPLATE_NAME_DARK] = _build_template(dark=True)
pio.templates.default = f"plotly_white+{TEMPLATE_NAME}"

# The combined template strings applied to every figure for belt-and-suspenders styling.
_COMBINED_TEMPLATE = f"plotly_white+{TEMPLATE_NAME}"
_COMBINED_TEMPLATE_DARK = f"plotly_dark+{TEMPLATE_NAME_DARK}"


def set_palette(mode: str) -> None:
    """Switch the global geometrics color palette.

    The palette is **process-global**: it affects every subsequent geometrics figure —
    grouped series colors (via :func:`color_for`), the heatmap / scatter color scales, and
    the registered Plotly template's colorway. The default look is unchanged until you opt
    in, so existing figures keep their colors unless you call this.

    Parameters
    ----------
    mode
        ``"default"`` (the Tableau 10 palette, today's look) or ``"colorblind"`` (the
        Okabe-Ito colorblind-safe qualitative palette plus colorblind-safe sequential and
        diverging scales).

    Raises
    ------
    ValueError
        If ``mode`` is not a known palette.

    Examples
    --------
    Switch to the colorblind-safe palette, then restore the default:

    ```python
    import geometrics as gm

    gm.set_palette("colorblind")  # every later figure uses the Okabe-Ito palette
    print(gm.get_palette())
    gm.set_palette("default")  # restore the Tableau 10 default
    ```
    """
    global _ACTIVE_PALETTE
    if mode not in _QUALITATIVE:
        valid = ", ".join(sorted(_QUALITATIVE))
        raise ValueError(f"unknown palette {mode!r}; choose from {valid}")
    _ACTIVE_PALETTE = mode
    # Rebuild and re-register the templates so their baked-in colorway/colorscale reflect the
    # new palette; apply_default_layout applies these by name, so this makes the change take
    # effect for every subsequent figure.
    pio.templates[TEMPLATE_NAME] = _build_template()
    pio.templates[TEMPLATE_NAME_DARK] = _build_template(dark=True)


def get_palette() -> str:
    """Return the name of the currently active palette (``"default"`` / ``"colorblind"``)."""
    return _ACTIVE_PALETTE


def _supports_native_subtitle() -> bool:
    """Return ``True`` if the installed Plotly accepts ``title.subtitle`` (Plotly >= 5.22)."""
    try:
        go.layout.Title(subtitle={"text": "x"})
    except (ValueError, TypeError):
        return False
    return True


_NATIVE_SUBTITLE = _supports_native_subtitle()


def _make_title(title: str | None, subtitle: str | None) -> dict:
    """Build a Plotly ``title`` dict, using a native subtitle when supported else emulating it."""
    text = title or ""
    if subtitle is None:
        return {"text": text}
    if _NATIVE_SUBTITLE:
        return {"text": text, "subtitle": {"text": subtitle}}
    sub_size = int(FONT_SIZE_TITLE * 0.65)
    return {
        "text": f"{text}<br>"
        f"<span style='font-size:{sub_size}px;color:#888'>{subtitle}</span>"
    }


def apply_default_layout(
    fig: go.Figure,
    *,
    dark: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    **layout_kwargs: object,
) -> go.Figure:
    """Apply geometrics' default layout (Tableau theme, presentation fonts) to ``fig``.

    The geometrics template carries the palette, continuous scales, fonts and sizes; this
    function applies it explicitly (so per-figure output is correct regardless of the
    global default) and then forwards any extra ``layout_kwargs`` to
    :meth:`plotly.graph_objects.Figure.update_layout`.

    Parameters
    ----------
    fig
        The figure to style (modified in place and returned).
    dark
        Apply the dark template (``plotly_dark`` base) instead of the light one.
    title
        Optional main title for the figure. When both ``title`` and ``subtitle`` are
        ``None`` no title is set (the chart relies on its labelled axes, as before).
    subtitle
        Optional subtitle, rendered under the title (native when the installed Plotly
        supports ``title.subtitle``, otherwise emulated as a smaller second line).
    **layout_kwargs
        Extra keyword arguments forwarded to
        :meth:`plotly.graph_objects.Figure.update_layout`.
    """
    fig.update_layout(template=_COMBINED_TEMPLATE_DARK if dark else _COMBINED_TEMPLATE)
    if title is not None or subtitle is not None:
        fig.update_layout(title=_make_title(title, subtitle))
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig


def color_for(index: int) -> str:
    """Return the active-palette color for a 0-based series ``index`` (wraps around)."""
    cw = active_colorway()
    return cw[index % len(cw)]


def diverging_color(value: float) -> str:
    """Map a value in ``[-1, 1]`` to an ``rgb(...)`` string on :data:`DIVERGING_SCALE`.

    Used for the correlation ellipse fills so they match the heatmap's diverging scale.
    ``-1`` is Tableau red, ``0`` near-white, ``+1`` Tableau blue, with linear
    interpolation between the scale's anchor stops.
    """
    v = max(-1.0, min(1.0, value))
    pos = (v + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
    stops = active_diverging_scale()
    for i in range(len(stops) - 1):
        p0, c0 = float(stops[i][0]), str(stops[i][1])
        p1, c1 = float(stops[i + 1][0]), str(stops[i + 1][1])
        if pos <= p1:
            t = 0.0 if p1 == p0 else (pos - p0) / (p1 - p0)
            r0, g0, b0 = _hex_to_rgb(c0)
            r1, g1, b1 = _hex_to_rgb(c1)
            r = round(r0 + (r1 - r0) * t)
            g = round(g0 + (g1 - g0) * t)
            b = round(b0 + (b1 - b0) * t)
            return f"rgb({r},{g},{b})"
    return f"rgb{_hex_to_rgb(str(stops[-1][1]))}"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Convert a ``#rrggbb`` hex string to an ``(r, g, b)`` integer tuple."""
    h = value.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# Fixed categorical semantics for spatial-analysis maps. LISA colors follow the
# splot/GeoDa convention so cluster maps read identically across the ecosystem.
LISA_COLORS = {
    "High-High": "#d7191c",
    "Low-Low": "#2c7bb6",
    "Low-High": "#abd9e9",
    "High-Low": "#fdae61",
    "Not significant": "#d3d3d3",
}
CLUB_COLORS = (
    COLOR_SEQUENCE  # clubs are ordinal-categorical; reuse the qualitative cycle
)
MAP_SEQUENTIAL = SEQUENTIAL_SCALE
MAP_DIVERGING = DIVERGING_SCALE
