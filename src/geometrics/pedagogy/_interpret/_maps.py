"""Plain-language interpretation of choropleth map results.

Duck-typed against :class:`geometrics._types.ChoroplethMapResult` (this module never
imports it): the function reads ``var`` / ``period`` / ``scheme`` / ``k`` / ``bins`` /
``animated`` and the per-entity ``df`` and returns Markdown.
"""

from __future__ import annotations

from typing import Any

from geometrics.pedagogy._format import fmt_num
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

__all__ = ["interpret_choropleth_map"]


def interpret_choropleth_map(result: Any, *, lang: str = "en") -> str:
    """Interpret a choropleth map: the classing and the bottom/top class ranges."""
    var = str(getattr(result, "var", "the variable"))
    period = getattr(result, "period", None)
    scheme = getattr(result, "scheme", None)
    bins = tuple(getattr(result, "bins", ()) or ())
    animated = bool(getattr(result, "animated", False))
    df = result.df
    values = df[var] if var in df.columns else df.iloc[:, 1]
    n = int(values.notna().sum())
    vmin = float(values.min())
    vmax = float(values.max())
    n_units = int(df.iloc[:, 0].nunique())

    where = f"across {n_units:,} regions"
    when = f" in {period}" if (period is not None and not animated) else ""

    lines: list[str]
    if not bins:
        lines = [
            f"The map shades **{var}**{when} {where} on a continuous scale, from "
            f"{fmt_num(vmin)} (lightest) to {fmt_num(vmax)} (darkest).",
            "Patches of similar shading mark geographic groupings of similar "
            "values — a visual cue, not a test of spatial dependence.",
        ]
    else:
        k = len(bins)
        scheme_txt = (
            f"the {scheme} scheme" if scheme is not None else "user-defined breaks"
        )
        bottom_hi = float(bins[0])
        top_lo = float(bins[-2]) if k >= 2 else vmin
        top_hi = float(bins[-1])
        n_bottom = int((values <= bottom_hi).sum())
        n_top = int((values > top_lo).sum()) if k >= 2 else n
        lines = [
            f"The map groups **{var}**{when} {where} into {k} classes using "
            f"{scheme_txt} (k = {k}).",
            f"The bottom class runs from {fmt_num(vmin)} to {fmt_num(bottom_hi)} "
            f"({n_bottom:,} observations); the top class from {fmt_num(top_lo)} to "
            f"{fmt_num(top_hi)} ({n_top:,} observations).",
            "Regions in the same class share a color, so contiguous patches of "
            "similar shading suggest geographic grouping — a visual cue, not a "
            "test of spatial dependence.",
        ]
    if animated:
        lines.append(
            f"The animation starts at {period} and holds the class breaks fixed "
            "across periods (pooled classification), so a region changing color "
            "reflects a change in value, not a change in scale."
        )
    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)
