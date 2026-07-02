"""Plain-language interpretation for the regional-inequality vertical.

Duck-typed against :class:`geometrics._types.InequalityOverTimeResult` and
:class:`geometrics._types.TheilDecompositionResult` (this module never imports the
result classes): the functions read the per-period ``df``, the trend ``summary`` and
the scalar fields, and return Markdown.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from geometrics.pedagogy._format import fmt_num, is_significant, significance_phrase
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

__all__ = ["interpret_inequality_over_time", "interpret_theil_decomposition"]

_MEASURE_NAMES = {
    "gini": "Gini index",
    "theil": "Theil index",
    "cv": "coefficient of variation",
}


def _first_last(series: pd.Series) -> tuple[float, float]:
    """Return the first and last finite values of ``series`` (``nan`` when none)."""
    finite = series.astype(float).dropna()
    if finite.empty:
        return float("nan"), float("nan")
    return float(finite.iloc[0]), float(finite.iloc[-1])


def interpret_inequality_over_time(result: Any, *, lang: str = "en") -> str:
    """Interpret the level and trend of cross-sectional inequality over time.

    Parameters
    ----------
    result
        An inequality-over-time result exposing ``df`` (per-period measures),
        ``summary`` (per-measure log-trend rows), ``var``, ``n_periods``,
        ``n_units`` and optionally ``w_spec``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of where inequality stands and which way it is moving.
    """
    df = result.df
    summary = result.summary
    var = str(getattr(result, "var", "the variable"))
    n_periods = int(getattr(result, "n_periods", len(df)))
    n_units = int(getattr(result, "n_units", 0))
    w_spec = getattr(result, "w_spec", None)

    first_t, last_t = df["time"].iloc[0], df["time"].iloc[-1]
    lines = [
        f"Cross-sectional inequality in **{var}** is tracked across {n_units:,} "
        f"units over {n_periods} periods ({first_t} to {last_t})."
    ]

    for measure, name in _MEASURE_NAMES.items():
        if measure not in df.columns:
            continue
        v0, v1 = _first_last(df[measure])
        if not (math.isfinite(v0) and math.isfinite(v1)):
            continue
        if math.isclose(v0, v1, rel_tol=1e-9, abs_tol=1e-12):
            move = "was essentially unchanged"
        elif v1 < v0:
            move = "fell"
        else:
            move = "rose"
        lines.append(
            f"The {name} {move} from {fmt_num(v0)} to {fmt_num(v1)} between the "
            "first and last period."
        )

    rows = summary if summary is not None else pd.DataFrame()
    for _, row in rows.iterrows():
        measure = str(row["measure"])
        name = _MEASURE_NAMES.get(measure, measure)
        slope = float(row["slope"])
        pvalue = float(row["pvalue"])
        if not math.isfinite(slope):
            lines.append(f"The trend of the {name} could not be estimated.")
            continue
        direction = "narrowing" if slope < 0 else "widening"
        verdict = (
            f"inequality by this measure is **{direction}**"
            if is_significant(pvalue)
            else "the movement is not distinguishable from a flat trend"
        )
        lines.append(
            f"The log-trend of the {name} is {fmt_num(slope)} per period "
            f"({significance_phrase(pvalue)}), so {verdict}."
        )

    if "gini_spatial" in df.columns:
        gs = df["gini_spatial"].astype(float).dropna()
        if not gs.empty:
            latest = float(gs.iloc[-1])
            share_txt = ""
            if "gini" in df.columns:
                g_latest = float(df["gini"].astype(float).iloc[-1])
                if math.isfinite(g_latest) and g_latest > 0:
                    share_txt = f" — about {latest / g_latest:.0%} of the overall Gini"
            spec_txt = f" under the weights ({w_spec})" if w_spec else ""
            lines.append(
                f"In the latest period, inequality between *neighboring* units"
                f"{spec_txt} contributes {fmt_num(latest)} to the Gini"
                f"{share_txt}; the rest comes from pairs that are not neighbors."
            )
        if "gini_spatial_p" in df.columns:
            p_last = float(df["gini_spatial_p"].astype(float).iloc[-1])
            if math.isfinite(p_last):
                if is_significant(p_last):
                    lines.append(
                        f"The permutation test (p = {fmt_num(p_last)}) indicates "
                        "the non-neighbor component of inequality is larger than "
                        "expected under spatial randomness — differences line up "
                        "with geography rather than being scattered among "
                        "neighbors."
                    )
                else:
                    lines.append(
                        f"The permutation test (p = {fmt_num(p_last)}) does not "
                        "distinguish the spatial split of inequality from what "
                        "random spatial arrangements produce."
                    )

    return "\n\n".join([*lines, _ASSOC_NOTE])


def interpret_theil_decomposition(result: Any, *, lang: str = "en") -> str:
    """Interpret the between/within split of the Theil index over time.

    Parameters
    ----------
    result
        A Theil-decomposition result exposing ``df`` (per-period ``theil``,
        ``between``, ``within``, ``between_share`` and optionally ``p_between``),
        ``var``, ``group`` and ``n_groups``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of how much inequality lies between versus within groups.
    """
    df = result.df
    var = str(getattr(result, "var", "the variable"))
    group = str(getattr(result, "group", "the partition"))
    n_groups = int(getattr(result, "n_groups", 0))

    last_t = df["time"].iloc[-1]
    theil = float(df["theil"].iloc[-1])
    share = float(df["between_share"].iloc[-1])

    lines = [
        f"The Theil index of **{var}** splits additively into inequality "
        f"**between** the {n_groups} {group} groups and inequality **within** them "
        "(between + within = total, exactly)."
    ]
    if math.isfinite(share):
        lines.append(
            f"In the latest period ({last_t}), the total Theil index is "
            f"{fmt_num(theil)}: about {share:.0%} of it lies between {group} "
            f"groups and {1 - share:.0%} within them — "
            + (
                f"differences across {group} means are the dominant layer of "
                "inequality."
                if share >= 0.5
                else f"most inequality plays out among units inside the same {group}."
            )
        )
    else:
        lines.append(
            f"In the latest period ({last_t}), total inequality is zero, so the "
            "between/within split is undefined."
        )

    s0, s1 = _first_last(df["between_share"])
    if math.isfinite(s0) and math.isfinite(s1) and len(df) > 1:
        if s1 > s0:
            trend = (
                f"rose from {s0:.0%} to {s1:.0%} — inequality is increasingly a "
                f"between-{group} phenomenon"
            )
        elif s1 < s0:
            trend = (
                f"fell from {s0:.0%} to {s1:.0%} — group means are pulling closer "
                "together relative to the differences inside groups"
            )
        else:
            trend = f"held steady at {s1:.0%}"
        lines.append(f"Over the window, the between-group share {trend}.")

    if "p_between" in df.columns:
        p_last = float(df["p_between"].iloc[-1])
        if math.isfinite(p_last):
            if is_significant(p_last):
                lines.append(
                    f"The permutation test on the between component (p = "
                    f"{fmt_num(p_last)}) indicates the observed grouping captures "
                    "more inequality than random reassignments of units to groups "
                    "typically do."
                )
            else:
                lines.append(
                    f"The permutation test on the between component (p = "
                    f"{fmt_num(p_last)}) does not distinguish the observed "
                    "grouping from random reassignments of units to groups."
                )

    return "\n\n".join([*lines, _ASSOC_NOTE])
