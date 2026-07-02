"""Plain-language interpretation of the convergence results (β, σ, clubs).

Duck-typed against :class:`geometrics._types.BetaConvergenceResult`,
:class:`geometrics._types.SigmaConvergenceResult` and
:class:`geometrics._types.ConvergenceClubsResult` (this module never imports them):
each function reads the result's scalar fields / frames and returns Markdown.
"""

from __future__ import annotations

import math
from typing import Any

from geometrics.pedagogy._format import fmt_num, significance_phrase
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

__all__ = [
    "interpret_beta_convergence",
    "interpret_sigma_convergence",
    "interpret_convergence_clubs",
]


def _z_pvalue(est: float, se: float) -> float:
    """Two-sided normal p-value of ``est / se`` (``nan`` for a non-positive ``se``)."""
    if not (math.isfinite(est) and math.isfinite(se)) or se <= 0.0:
        return float("nan")
    z = abs(est / se)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def interpret_beta_convergence(result: Any, *, lang: str = "en") -> str:
    """Interpret a β-convergence fit: slope, speed, half-life and spatial spillovers.

    Parameters
    ----------
    result
        A β-convergence result exposing ``model``, ``var``, ``horizon``, ``n_obs``,
        the ``beta_direct`` / ``beta_indirect`` / ``beta_total`` triple with their
        standard errors, ``rho`` / ``lam``, ``speed``, ``half_life`` and ``w_spec``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of the convergence slope and its decomposition.
    """
    model = str(getattr(result, "model", "ols"))
    var = str(getattr(result, "var", "the variable"))
    horizon = float(getattr(result, "horizon", float("nan")))
    n_obs = int(getattr(result, "n_obs", 0))
    total = float(getattr(result, "beta_total", float("nan")))
    se_total = float(getattr(result, "se_total", float("nan")))
    direct = float(getattr(result, "beta_direct", float("nan")))
    indirect = float(getattr(result, "beta_indirect", float("nan")))
    rho = float(getattr(result, "rho", float("nan")))
    lam = float(getattr(result, "lam", float("nan")))
    speed = float(getattr(result, "speed", float("nan")))
    half_life = float(getattr(result, "half_life", float("nan")))
    w_spec = getattr(result, "w_spec", None)

    sig = significance_phrase(_z_pvalue(total, se_total))
    lines = [
        f"Across {n_obs:,} units, the growth of **{var}** over a {fmt_num(horizon)}-"
        f"period window is associated with its initial log level with a total slope "
        f"of **{fmt_num(total)}** (SE {fmt_num(se_total)}), {sig}."
    ]

    if math.isfinite(total) and total < 0:
        lines.append(
            "The slope is negative — the β-convergence pattern: units that started "
            "lower tended to grow faster, narrowing initial gaps."
        )
        if math.isfinite(speed) and math.isfinite(half_life):
            lines.append(
                f"That slope implies a convergence speed of λ = {fmt_num(speed)} per "
                f"period and a half-life of about {fmt_num(half_life)} periods — the "
                "time for half of an initial gap to close at this pace."
            )
        elif not math.isfinite(speed):
            lines.append(
                "The slope does not map to a well-defined convergence speed here "
                "(1 + β·T is non-positive), so no half-life is reported."
            )
    elif math.isfinite(total):
        lines.append(
            "The slope is non-negative — no catch-up pattern: initially higher units "
            "grew at least as fast, consistent with persistence or divergence rather "
            "than convergence."
        )

    if model != "ols":
        spill = (
            f"a spillover (indirect) component of {fmt_num(indirect)} operating "
            "through neighboring units"
            if math.isfinite(indirect)
            else "no separate spillover component"
        )
        lines.append(
            f"The {model.upper()} decomposition splits the total into a direct "
            f"component of {fmt_num(direct)} (own initial level) and {spill}"
            + (f", under the weights: {w_spec}." if w_spec else ".")
        )
        if math.isfinite(rho):
            lines.append(
                f"The spatial-lag parameter ρ = {fmt_num(rho)} says each unit's "
                "growth moves together with its neighbors' growth, so part of the "
                "convergence pattern is shared across space rather than purely "
                "unit-by-unit."
            )
        if math.isfinite(lam):
            lines.append(
                f"The spatial-error parameter λ = {fmt_num(lam)} indicates spatially "
                "correlated unobservables: shocks to growth cluster geographically."
            )

    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_sigma_convergence(result: Any, *, lang: str = "en") -> str:
    """Interpret a σ-convergence fit: whether the cross-sectional dispersion narrows.

    Parameters
    ----------
    result
        A σ-convergence result exposing ``var``, ``n_units``, ``n_periods`` and the
        per-measure trend scalars (``std_slope`` / ``std_pvalue``, ``gini_slope``,
        ``cv_slope``).
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of the dispersion trend and measure agreement.
    """
    var = str(getattr(result, "var", "the variable"))
    n_units = int(getattr(result, "n_units", 0))
    n_periods = int(getattr(result, "n_periods", 0))
    std_slope = float(getattr(result, "std_slope", float("nan")))
    std_p = float(getattr(result, "std_pvalue", float("nan")))
    gini_slope = float(getattr(result, "gini_slope", float("nan")))
    cv_slope = float(getattr(result, "cv_slope", float("nan")))

    lines: list[str]
    if not math.isfinite(std_slope):
        lines = [
            f"The dispersion trend of **{var}** could not be estimated over the "
            f"{n_periods} periods available (see the result's notes)."
        ]
    else:
        direction = "narrowed" if std_slope < 0 else "widened"
        verdict = (
            "σ-convergence: the units are becoming more alike"
            if std_slope < 0
            else "σ-divergence: the units are drifting apart"
        )
        lines = [
            f"Across {n_units:,} units and {n_periods} periods, the cross-sectional "
            f"dispersion of **{var}** (standard deviation of its log) {direction} by "
            f"about {fmt_num(abs(std_slope) * 100)}% per period "
            f"({significance_phrase(std_p)}).",
            f"A negative log-dispersion trend is {verdict}.",
        ]

    finite = [s for s in (std_slope, gini_slope, cv_slope) if math.isfinite(s)]
    if len(finite) > 1:
        negatives = sum(1 for s in finite if s < 0)
        if negatives == len(finite):
            lines.append(
                "All estimated measures (standard deviation, Gini, coefficient of "
                "variation) trend downward, so the narrowing is not an artifact of "
                "one dispersion metric."
            )
        elif negatives == 0:
            lines.append(
                "All estimated measures trend upward, so the widening is consistent "
                "across dispersion metrics."
            )
        else:
            lines.append(
                "The dispersion measures disagree in sign — the verdict depends on "
                "the metric, so read the trend table before concluding either way."
            )

    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_convergence_clubs(result: Any, *, lang: str = "en") -> str:
    """Interpret a Phillips-Sul club-convergence run: one club, several, or none.

    Parameters
    ----------
    result
        A clubs result exposing ``var``, ``n_units``, ``n_periods``, ``n_clubs``,
        ``n_divergent``, ``global_beta``, ``global_tstat``, ``tcrit``, ``converged``
        and the per-club ``summary`` frame.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of how the panel splits into convergence clubs.
    """
    var = str(getattr(result, "var", "the variable"))
    n_units = int(getattr(result, "n_units", 0))
    n_periods = int(getattr(result, "n_periods", 0))
    n_clubs = int(getattr(result, "n_clubs", 0))
    n_divergent = int(getattr(result, "n_divergent", 0))
    tstat = float(getattr(result, "global_tstat", float("nan")))
    tcrit = float(getattr(result, "tcrit", -1.65))
    converged = bool(getattr(result, "converged", False))

    lines = []
    if converged:
        lines.append(
            f"The whole panel of {n_units:,} units passes the log(t) convergence "
            f"test for **{var}** (t = {fmt_num(tstat)} > {fmt_num(tcrit)} over "
            f"{n_periods} periods): the units form a **single convergence club**, "
            "all heading toward a common relative path."
        )
    else:
        lines.append(
            f"Global convergence of **{var}** is rejected for the {n_units:,} units "
            f"(log(t) t = {fmt_num(tstat)} ≤ {fmt_num(tcrit)}): the panel is not "
            "heading toward one common path."
        )
        if n_clubs > 0:
            lines.append(
                f"The Phillips-Sul clustering instead finds **{n_clubs} convergence "
                f"club{'s' if n_clubs != 1 else ''}** — groups whose members converge "
                "toward a shared, club-specific path while the clubs themselves stay "
                "apart."
            )
            summary = getattr(result, "summary", None)
            if summary is not None and len(summary) and "n_members" in summary.columns:
                club_rows = summary[summary["club"] != "Divergent"]
                if len(club_rows):
                    sizes = ", ".join(
                        f"{row['club']}: {int(row['n_members'])}"
                        for _, row in club_rows.iterrows()
                    )
                    lines.append(f"Club sizes — {sizes}.")
        else:
            lines.append(
                "No convergence clubs were found either: every unit follows its own "
                "path."
            )
        if n_divergent > 0:
            lines.append(
                f"{n_divergent} unit{'s' if n_divergent != 1 else ''} fit no club "
                "(the divergent group) and follow paths of their own."
            )

    lines.append(
        "Club membership is a description of co-movement in the relative transition "
        "paths, not an explanation of why the groups differ."
    )
    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)
