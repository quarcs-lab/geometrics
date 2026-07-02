"""Plain-language interpretation for the spatial-dependence vertical.

Duck-typed against :class:`geometrics._types.MoranPlotResult`,
:class:`geometrics._types.LisaClusterMapResult` and
:class:`geometrics._types.MoranOverTimeResult` (this module never imports the result
classes): each function reads the result's scalar fields and ``df`` and returns
Markdown.
"""

from __future__ import annotations

from typing import Any

from geometrics.pedagogy._format import fmt_num, is_significant, significance_phrase
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

__all__ = [
    "interpret_lisa_cluster_map",
    "interpret_moran_over_time",
    "interpret_moran_plot",
]


def _when(period: Any) -> str:
    """Return ``" in <period>"`` when a period is recorded, else an empty string."""
    return f" in {period}" if period is not None else ""


def _dependence_direction(moran_i: float, expected_i: float, p_sim: float) -> str:
    """One sentence on the direction of global spatial dependence (or its absence)."""
    if not is_significant(p_sim):
        return (
            "The pattern is statistically indistinguishable from spatial "
            "randomness: knowing a region's value tells you little about its "
            "neighbors' values."
        )
    if moran_i > expected_i:
        return (
            "The dependence is **positive**: similar values cluster in space — "
            "high values sit next to high values and low next to low, so the map "
            "shows contiguous patches rather than a random scatter."
        )
    return (
        "The dependence is **negative**: dissimilar values sit side by side — a "
        "checkerboard-like dispersion in which high values tend to neighbor low "
        "ones."
    )


def interpret_moran_plot(result: Any, *, lang: str = "en") -> str:
    """Interpret a Moran scatterplot: strength, direction and significance of clustering.

    Parameters
    ----------
    result
        A Moran-plot result exposing ``moran_i``, ``expected_i``, ``p_sim``,
        ``z_sim``, ``permutations``, ``var``, ``period``, ``w_spec`` and the
        per-entity ``df`` with a ``quadrant`` column.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of the global Moran test and the quadrant split.
    """
    moran_i = float(result.moran_i)
    expected_i = float(result.expected_i)
    p_sim = float(result.p_sim)
    permutations = int(result.permutations)
    var = str(getattr(result, "var", "the variable"))
    w_spec = str(getattr(result, "w_spec", "the spatial weights"))
    df = result.df
    n = len(df)

    lines = [
        f"Global Moran's I for **{var}**{_when(getattr(result, 'period', None))} "
        f"is {fmt_num(moran_i)}, against an expectation of {fmt_num(expected_i)} "
        f"under spatial randomness — {significance_phrase(p_sim)} (pseudo "
        f"p = {fmt_num(p_sim)} from {permutations:,} permutations, under "
        f"{w_spec}).",
        _dependence_direction(moran_i, expected_i, p_sim),
    ]
    if "quadrant" in df.columns and n:
        n_alike = int(df["quadrant"].isin(["HH", "LL"]).sum())
        lines.append(
            f"{n_alike:,} of {n:,} regions ({n_alike / n:.0%}) fall in the "
            "clustering quadrants of the scatter (High-High or Low-Low); the "
            "rest are surrounded by neighbors unlike themselves. The slope of "
            "the fitted line equals Moran's I under row-standardized weights, "
            "so a steeper line means stronger clustering."
        )
    return "\n\n".join([*lines, _ASSOC_NOTE])


def interpret_lisa_cluster_map(result: Any, *, lang: str = "en") -> str:
    """Interpret a LISA cluster map: local hot spots, cold spots and spatial outliers.

    Parameters
    ----------
    result
        A LISA result exposing ``moran_i``, ``p_sim_global``, ``alpha``,
        ``permutations``, the ``n_hh`` / ``n_ll`` / ``n_hl`` / ``n_lh`` / ``n_ns``
        counts, ``var``, ``period`` and ``w_spec``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of where values cluster locally and where they stand out.
    """
    moran_i = float(result.moran_i)
    p_global = float(result.p_sim_global)
    alpha = float(result.alpha)
    n_hh = int(result.n_hh)
    n_ll = int(result.n_ll)
    n_hl = int(result.n_hl)
    n_lh = int(result.n_lh)
    n_ns = int(result.n_ns)
    n = n_hh + n_ll + n_hl + n_lh + n_ns
    n_sig = n - n_ns
    var = str(getattr(result, "var", "the variable"))
    w_spec = str(getattr(result, "w_spec", "the spatial weights"))

    global_clause = (
        "consistent with overall clustering of similar values"
        if is_significant(p_global) and moran_i > 0
        else (
            "consistent with overall dispersion of dissimilar values"
            if is_significant(p_global)
            else "not significant globally, so any structure is purely local"
        )
    )
    lines = [
        f"Local Moran statistics (LISA) locate *where* **{var}**"
        f"{_when(getattr(result, 'period', None))} clusters or stands out, under "
        f"{w_spec}. The accompanying global Moran's I is {fmt_num(moran_i)} "
        f"(pseudo p = {fmt_num(p_global)}), {global_clause}.",
        f"At the {alpha:g} significance level, {n_sig:,} of {n:,} regions show "
        f"significant local association: **{n_hh:,} High-High** hot spots (high "
        f"values surrounded by high neighbors) and **{n_ll:,} Low-Low** cold "
        f"spots (low surrounded by low) mark clustering, while **{n_hl:,} "
        f"High-Low** and **{n_lh:,} Low-High** regions are spatial outliers that "
        f"break with their surroundings. The remaining {n_ns:,} regions are not "
        "significant — their local pattern is compatible with randomness.",
        "LISA pseudo p-values are computed region by region without a "
        "multiple-testing adjustment, so treat borderline clusters cautiously "
        "and read the map as descriptive of where dependence concentrates.",
    ]
    return "\n\n".join([*lines, _ASSOC_NOTE])


def interpret_moran_over_time(result: Any, *, lang: str = "en") -> str:
    """Interpret a Moran-over-time series: level, trajectory and per-period significance.

    Parameters
    ----------
    result
        A Moran-over-time result exposing ``var``, ``permutations``, ``w_spec``
        and a ``df`` with ``period``, ``moran_i`` and ``p_sim`` columns.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of how global spatial dependence evolves.
    """
    df = result.df
    var = str(getattr(result, "var", "the variable"))
    w_spec = str(getattr(result, "w_spec", "the spatial weights"))
    n_periods = len(df)
    first_p, last_p = df["period"].iloc[0], df["period"].iloc[-1]
    first_i = float(df["moran_i"].iloc[0])
    last_i = float(df["moran_i"].iloc[-1])
    n_sig = int((df["p_sim"] < 0.05).sum())
    change = last_i - first_i

    if abs(change) < 0.05:
        trajectory = (
            "The series is broadly **stable**: the degree to which similar "
            "values cluster in space changes little over the window."
        )
    elif change > 0:
        trajectory = (
            "The series **rises**: values cluster in space more strongly at the "
            "end of the window than at the start, so geography and the variable "
            "grow more aligned."
        )
    else:
        trajectory = (
            "The series **falls**: spatial clustering weakens over the window, "
            "moving the map closer to spatial randomness (or dispersion)."
        )

    lines = [
        f"Global Moran's I for **{var}** is tracked over {n_periods:,} periods "
        f"({first_p} to {last_p}) on a fixed set of regions, under {w_spec}: it "
        f"moves from {fmt_num(first_i)} to {fmt_num(last_i)}.",
        trajectory,
        f"Per-period permutation tests flag {n_sig:,} of {n_periods:,} periods "
        "as significant at the 5% level (filled markers in the figure); open "
        "markers are periods where the pattern is compatible with spatial "
        "randomness.",
    ]
    return "\n\n".join([*lines, _ASSOC_NOTE])
