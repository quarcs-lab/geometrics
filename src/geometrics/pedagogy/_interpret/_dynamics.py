"""Plain-language interpretation for the distribution-dynamics vertical (giddy Markov).

Duck-typed against :class:`geometrics._types.MarkovTransitionsResult` and
:class:`geometrics._types.SpatialMarkovResult` (this module never imports the result
classes): the functions read the transition matrices and mobility scalars and return
Markdown.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from geometrics.pedagogy._format import fmt_num, is_significant, significance_phrase
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

__all__ = ["interpret_markov_transitions", "interpret_spatial_markov"]


def _diag(p: Any) -> np.ndarray:
    """Return the diagonal of a (labelled) transition-probability matrix as floats."""
    return np.diag(np.asarray(p, dtype=float))


def interpret_markov_transitions(result: Any, *, lang: str = "en") -> str:
    """Interpret a Markov transition analysis: persistence, mobility, the long-run mix.

    Parameters
    ----------
    result
        A Markov-transitions result exposing ``p``, ``states``, ``steady_state``,
        ``sojourn``, ``shorrocks`` / ``prais`` / ``bartholomew``, ``n_transitions``,
        ``k``, ``scheme`` and ``var``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of how sticky the states are and where the chain settles.
    """
    var = str(getattr(result, "var", "the variable"))
    k = int(getattr(result, "k", 0) or np.asarray(result.p).shape[0])
    scheme = {
        "quantiles": "quantile",
        "equal_interval": "equal-interval",
        "fisherjenks": "Fisher-Jenks",
        "user": "user-defined",
    }.get(
        str(getattr(result, "scheme", "quantiles")),
        str(getattr(result, "scheme", "quantiles")).replace("_", " "),
    )
    n_transitions = int(getattr(result, "n_transitions", 0))
    states = tuple(str(s) for s in getattr(result, "states", ()))
    diag = _diag(result.p)
    mean_stay = float(np.nanmean(diag))
    hi = int(np.nanargmax(diag))
    lo = int(np.nanargmin(diag))

    lines = [
        f"Pooling {n_transitions:,} period-to-period moves, **{var}** was discretized "
        f"into {k} {scheme} states and its movement summarized by a "
        "transition-probability matrix: each row gives the chance of ending the next "
        "period in each state, conditional on the current one.",
        f"On average a region stays in its current state with probability "
        f"{fmt_num(mean_stay)} — the diagonal is "
        + (
            "dominant, so positions in the distribution are highly persistent"
            if mean_stay >= 0.7
            else (
                "moderate, so regions change position with some regularity"
                if mean_stay >= 0.4
                else "weak, so regions churn rapidly across states"
            )
        )
        + f". The stickiest state is **{states[hi]}** (stay probability "
        f"{fmt_num(diag[hi])}); the most mobile is **{states[lo]}** "
        f"({fmt_num(diag[lo])}).",
    ]

    shorrocks = float(getattr(result, "shorrocks", float("nan")))
    prais = float(getattr(result, "prais", float("nan")))
    bartholomew = float(getattr(result, "bartholomew", float("nan")))
    if math.isfinite(shorrocks):
        lines.append(
            f"The Shorrocks mobility index is {fmt_num(shorrocks)} on a 0 (complete "
            "immobility: the identity matrix) to "
            f"{fmt_num(k / (k - 1))} (all mass off the diagonal) scale"
            + (
                f"; the Prais determinant index is {fmt_num(prais)} and the "
                f"Bartholomew index {fmt_num(bartholomew)}"
                if math.isfinite(prais) or math.isfinite(bartholomew)
                else ""
            )
            + "."
        )

    steady = np.asarray(getattr(result, "steady_state", []), dtype=float)
    if steady.size == len(states) and np.all(np.isfinite(steady)):
        top = int(np.argmax(steady))
        lines.append(
            "If these transition probabilities kept operating, the cross-section "
            "would settle into the ergodic (steady-state) mix with the largest "
            f"long-run share in **{states[top]}** ({fmt_num(steady[top])}, versus "
            f"{fmt_num(1.0 / k)} under an even split)."
        )
    else:
        lines.append(
            "The chain is reducible, so a single long-run (ergodic) distribution "
            "does not exist — see the result notes."
        )

    sojourn = np.asarray(getattr(result, "sojourn", []), dtype=float)
    if sojourn.size == len(states) and np.any(np.isfinite(sojourn)):
        longest = int(np.nanargmax(sojourn))
        lines.append(
            f"Expected sojourn times say a region entering **{states[longest]}** "
            f"remains there for about {fmt_num(sojourn[longest])} consecutive "
            "periods before moving."
        )

    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)


def interpret_spatial_markov(result: Any, *, lang: str = "en") -> str:
    """Interpret a spatial Markov analysis: how neighbors condition mobility.

    Parameters
    ----------
    result
        A spatial-Markov result exposing ``p_global``, ``p_conditional``,
        ``lr_stat`` / ``lr_p``, ``q_stat`` / ``q_p``, ``dof``, ``k``, ``m``,
        ``relative``, ``var`` and ``w_spec``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of the neighbor-conditioned transition dynamics.
    """
    var = str(getattr(result, "var", "the variable"))
    k = int(getattr(result, "k", 0) or np.asarray(result.p_global).shape[0])
    m = int(getattr(result, "m", 0) or len(result.p_conditional))
    w_spec = str(getattr(result, "w_spec", "the spatial weights"))
    relative = bool(getattr(result, "relative", False))

    lines = [
        f"The spatial Markov chain splits **{var}**'s {k}-state transition matrix by "
        f"the neighbors' position (spatial lag under {w_spec}), giving one matrix "
        f"per neighborhood class ({m} classes)"
        + (
            "; values were expressed relative to each period's mean first"
            if relative
            else ""
        )
        + ".",
        f"The unconditional matrix keeps regions in place with average probability "
        f"{fmt_num(float(np.nanmean(_diag(result.p_global))))}.",
    ]

    conditionals = tuple(getattr(result, "p_conditional", ()))
    if len(conditionals) >= 2:
        low = float(np.nanmean(_diag(conditionals[0])))
        high = float(np.nanmean(_diag(conditionals[-1])))
        lines.append(
            "Conditioning on context, regions surrounded by **low-value neighbors** "
            f"stay in place with average probability {fmt_num(low)}, while regions "
            f"surrounded by **high-value neighbors** do so with {fmt_num(high)} — "
            + (
                "similar persistence in both contexts."
                if not (math.isfinite(low) and math.isfinite(high))
                or abs(low - high) < 0.02
                else (
                    "movement is more common in prosperous neighborhoods."
                    if low > high
                    else "movement is more common in low-value neighborhoods."
                )
            )
        )

    lr_stat = float(getattr(result, "lr_stat", float("nan")))
    lr_p = float(getattr(result, "lr_p", float("nan")))
    q_stat = float(getattr(result, "q_stat", float("nan")))
    q_p = float(getattr(result, "q_p", float("nan")))
    dof = int(getattr(result, "dof", 0))
    if math.isfinite(lr_stat) or math.isfinite(q_stat):
        p_min = np.nanmin([lr_p, q_p])
        lines.append(
            f"The homogeneity tests (LR = {fmt_num(lr_stat)}, "
            f"p = {fmt_num(lr_p)}; Q = {fmt_num(q_stat)}, p = {fmt_num(q_p)}; "
            f"{dof} degrees of freedom) are {significance_phrase(float(p_min))}: "
            + (
                "transition dynamics **differ across neighborhood contexts** — a "
                "region's mobility is associated with the state of its neighbors."
                if is_significant(float(p_min))
                else "the data do not reject identical transition dynamics across "
                "neighborhood contexts."
            )
        )
    else:
        lines.append(
            "The LR / Q homogeneity tests could not be computed (some conditional "
            "matrices are too sparse) — see the result notes."
        )

    lines += ["", _ASSOC_NOTE]
    return "\n".join(lines)
