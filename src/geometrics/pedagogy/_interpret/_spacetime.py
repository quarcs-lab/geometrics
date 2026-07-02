"""Plain-language interpretation for the space-time descriptives vertical.

Duck-typed against :class:`geometrics._types.DistributionOverTimeResult` and
:class:`geometrics._types.SpacetimeHeatmapResult` (this module never imports the
result classes): the functions read ``df`` and the recorded options and return
Markdown.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from geometrics.pedagogy._format import fmt_num
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

__all__ = ["interpret_distribution_over_time", "interpret_spacetime_heatmap"]


def _density_moments(values: np.ndarray, density: np.ndarray) -> tuple[float, float]:
    """Return the (mean, standard deviation) of a density evaluated on a grid."""
    mass = float(trapezoid(density, values))
    if not np.isfinite(mass) or mass <= 0.0:
        return float("nan"), float("nan")
    mean = float(trapezoid(values * density, values) / mass)
    var = float(trapezoid((values - mean) ** 2 * density, values) / mass)
    return mean, float(np.sqrt(max(var, 0.0)))


def interpret_distribution_over_time(result: Any, *, lang: str = "en") -> str:
    """Interpret how a cross-sectional distribution shifts and spreads over time.

    Parameters
    ----------
    result
        A distribution-over-time result exposing the tidy evaluation frame ``df``
        (columns ``time``, ``value``, ``density``) plus ``var``, ``kind`` and
        ``relative``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of the shift in the center and the change in spread
        between the first and last period.
    """
    df = result.df
    var = str(getattr(result, "var", "the variable"))
    relative = bool(getattr(result, "relative", False))
    kind = str(getattr(result, "kind", "ridgeline"))

    periods = list(dict.fromkeys(df["time"]))
    first, last = periods[0], periods[-1]
    g0 = df.loc[df["time"] == first]
    g1 = df.loc[df["time"] == last]
    m0, s0 = _density_moments(
        g0["value"].to_numpy(dtype=float), g0["density"].to_numpy(dtype=float)
    )
    m1, s1 = _density_moments(
        g1["value"].to_numpy(dtype=float), g1["density"].to_numpy(dtype=float)
    )

    how = (
        "a ridgeline of one filled density per period (newest on top)"
        if kind == "ridgeline"
        else "a single density animated over the periods"
    )
    lines = [
        f"One kernel density of **{var}** per period, {first} to {last} "
        f"({len(periods)} periods), drawn as {how}."
    ]
    if relative:
        lines.append(
            "Values are divided by each period's cross-sectional mean, so **1.0 "
            "marks the period average**: mass piling up around 1 means units bunch "
            "near the average, while separate humps suggest groups of units "
            "clustering at distinct relative levels."
        )

    shift = m1 - m0
    scale = max(s0, abs(m0) * 0.01, 1e-12)
    if not np.isfinite(shift) or abs(shift) <= 0.05 * scale:
        shift_txt = (
            f"The center of the distribution is **essentially unchanged** between "
            f"{first} and {last} (density-weighted mean {fmt_num(m0)} to "
            f"{fmt_num(m1)})."
        )
    else:
        word = "higher" if shift > 0 else "lower"
        shift_txt = (
            f"The center of the distribution shifted **{word}** between {first} "
            f"and {last} (density-weighted mean {fmt_num(m0)} to {fmt_num(m1)})."
        )
    lines.append(shift_txt)

    ratio = s1 / s0 if np.isfinite(s0) and s0 > 0 else float("nan")
    if np.isfinite(ratio) and ratio < 0.95:
        lines.append(
            f"The **spread narrowed** (standard deviation {fmt_num(s0)} to "
            f"{fmt_num(s1)}): the cross-section became more alike over time — the "
            "distributional footprint of σ-convergence."
        )
    elif np.isfinite(ratio) and ratio > 1.05:
        lines.append(
            f"The **spread widened** (standard deviation {fmt_num(s0)} to "
            f"{fmt_num(s1)}): the cross-section pulled apart over time — "
            "dispersion grew rather than shrank."
        )
    else:
        lines.append(
            f"The **spread held roughly steady** (standard deviation {fmt_num(s0)} "
            f"to {fmt_num(s1)}): dispersion neither narrowed nor widened much."
        )

    return "\n\n".join([*lines, _ASSOC_NOTE])


def interpret_spacetime_heatmap(result: Any, *, lang: str = "en") -> str:
    """Interpret an entity-by-time heatmap: row order, persistence, and shading.

    Parameters
    ----------
    result
        A space-time heatmap result exposing the entity-by-time pivot ``df`` (rows
        in display order) plus ``var``, ``sort_by`` and ``relative``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of the value surface and how persistent unit rankings
        are between the first and last period.
    """
    df: pd.DataFrame = result.df
    var = str(getattr(result, "var", "the variable"))
    sort_by = str(getattr(result, "sort_by", "value"))
    relative = bool(getattr(result, "relative", False))

    n_units, n_periods = df.shape
    first, last = df.columns[0], df.columns[-1]
    order_txt = {
        "value": "rows are ordered by each unit's mean value, highest at the top",
        "name": "rows are in alphabetical order",
        "north_south": (
            "rows are ordered geographically from north (top) to south (bottom)"
        ),
        "east_west": (
            "rows are ordered geographically from west (top) to east (bottom)"
        ),
    }.get(sort_by, f"rows are ordered by {sort_by!r}")

    lines = [
        f"The heatmap shows **{var}** for {n_units:,} units across {n_periods} "
        f"periods ({first} to {last}); {order_txt}."
    ]
    if relative:
        lines.append(
            "Cells are divided by each period's cross-sectional mean (1.0 = the "
            "period average), so shading compares units **within** a period rather "
            "than tracking the overall level."
        )

    c0, c1 = df[first], df[last]
    both = c0.notna() & c1.notna()
    if int(both.sum()) >= 3:
        rho = float(c0[both].rank().corr(c1[both].rank()))
        if np.isfinite(rho):
            if rho >= 0.8:
                lines.append(
                    f"Unit rankings between {first} and {last} are **highly "
                    f"persistent** (rank correlation {fmt_num(rho)}): rows keep "
                    "their relative shading from left to right, so units mostly "
                    "hold their position in the distribution."
                )
            elif rho >= 0.4:
                lines.append(
                    f"Unit rankings between {first} and {last} are **moderately "
                    f"persistent** (rank correlation {fmt_num(rho)}): most rows "
                    "keep their relative shading, but some units move up or down "
                    "the distribution."
                )
            else:
                lines.append(
                    f"Unit rankings between {first} and {last} are **fluid** (rank "
                    f"correlation {fmt_num(rho)}): shading reshuffles across rows, "
                    "so units change position in the distribution considerably."
                )

    if sort_by in ("north_south", "east_west"):
        lines.append(
            "With rows in geographic order, horizontal bands of similar shading "
            "mean nearby units carry similar values — a visual cue of geographic "
            "grouping, not a test of spatial dependence."
        )

    return "\n\n".join([*lines, _ASSOC_NOTE])
