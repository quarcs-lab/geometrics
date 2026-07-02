"""Plain-language interpretation for the local-models vertical (GWR / MGWR).

Duck-typed against :class:`geometrics._types.GWRResult` and
:class:`geometrics._types.MGWRResult` (this module never imports the result
classes): the functions read the per-entity local-coefficient frame and the
bandwidth / correction scalars, and return Markdown.
"""

from __future__ import annotations

from typing import Any

from geometrics.pedagogy._format import fmt_num
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

__all__ = ["interpret_gwr", "interpret_mgwr"]


def _term_reading(df: Any, entity: str, term: str, critical_t: float) -> str:
    """Describe one term's local surface: range, where strongest, significance."""
    coefs = df[f"{term}_coef"]
    sig = df[f"{term}_significant"].astype(bool)
    n = len(df)
    n_sig = int(sig.sum())

    # "Strongest" = largest absolute local coefficient, preferring significant units.
    pool = df.loc[sig] if n_sig else df
    strongest = pool.loc[pool[f"{term}_coef"].abs().idxmax()]
    where = str(strongest[entity])
    peak = float(strongest[f"{term}_coef"])

    sig_txt = (
        f"{n_sig} of {n} units clear the corrected threshold (|t| ≥ "
        f"{fmt_num(critical_t)}); non-significant units are greyed on the map"
        if n_sig
        else f"no unit clears the corrected threshold (|t| ≥ {fmt_num(critical_t)}), "
        "so the surface should be read as noise"
    )
    return (
        f"The **{term}** association varies locally from {fmt_num(coefs.min())} to "
        f"{fmt_num(coefs.max())}; it is strongest around **{where}** (local "
        f"coefficient {fmt_num(peak)}). {sig_txt}."
    )


def interpret_gwr(result: Any, *, lang: str = "en") -> str:
    """Interpret a GWR fit: the bandwidth scale and each term's local surface.

    Parameters
    ----------
    result
        A GWR result exposing ``df`` (per-entity local coefficients with
        ``<term>_coef`` / ``<term>_significant`` columns and ``local_r2``),
        ``outcome``, ``covariates``, ``bw``, ``fixed``, ``kernel``, ``r2``,
        ``adj_alpha`` and ``critical_t``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of where and how the associations vary over space.
    """
    df = result.df
    entity = str(df.columns[0])
    outcome = str(getattr(result, "outcome", "the outcome"))
    covariates = tuple(getattr(result, "covariates", ()) or ())
    bw = float(result.bw)
    fixed = bool(getattr(result, "fixed", False))
    kernel = str(getattr(result, "kernel", "bisquare"))
    n = len(df)

    scale_txt = (
        f"a fixed {kernel} kernel of {fmt_num(bw)} metric units"
        if fixed
        else f"an adaptive {kernel} kernel of {bw:g} nearest neighbors (of {n} units)"
    )
    locality = (
        "a small bandwidth relative to the sample, so the surfaces pick up quite "
        "local variation"
        if (not fixed and bw <= n / 3)
        else "a broad bandwidth, so the surfaces vary smoothly and gradually"
    )
    lines = [
        f"GWR calibrated a separate weighted regression of **{outcome}** at each of "
        f"the {n} units, using {scale_txt} — {locality}. Overall R² is "
        f"{fmt_num(float(result.r2))}."
    ]
    lines.extend(
        _term_reading(df, entity, term, float(result.critical_t)) for term in covariates
    )
    if "local_r2" in df.columns:
        r2min = float(df["local_r2"].min())
        r2max = float(df["local_r2"].max())
        lines.append(
            f"Local R² ranges from {fmt_num(r2min)} to {fmt_num(r2max)}: the model "
            "describes the outcome better where it is higher."
        )
    lines.append(
        "Local significance uses the multiple-testing-corrected alpha "
        f"({fmt_num(float(result.adj_alpha))}) of da Silva & Fotheringham, so the "
        "flagged units are conservative."
    )
    return "\n\n".join([*lines, _ASSOC_NOTE])


def interpret_mgwr(result: Any, *, lang: str = "en") -> str:
    """Interpret an MGWR fit: each term's operating scale and local surface.

    Parameters
    ----------
    result
        An MGWR result exposing ``df`` (per-entity local coefficients with
        ``<term>_coef`` / ``<term>_significant`` columns), ``outcome``,
        ``covariates``, the per-term ``bw`` / ``critical_t`` dicts and ``r2``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of the per-term spatial scales and local patterns.
    """
    df = result.df
    entity = str(df.columns[0])
    outcome = str(getattr(result, "outcome", "the outcome"))
    covariates = tuple(getattr(result, "covariates", ()) or ())
    bw = dict(result.bw)
    critical_t = dict(result.critical_t)
    n = len(df)

    lines = [
        f"MGWR let every term in the local regression of **{outcome}** operate at "
        f"its own spatial scale (variables z-standardized, {n} units); overall R² "
        f"is {fmt_num(float(result.r2))}."
    ]

    scales = []
    for term, b in bw.items():
        reach = (
            "essentially global"
            if b >= 0.9 * n
            else ("broad" if b >= n / 2 else "local")
        )
        scales.append(f"**{term}** at {b:g} nearest neighbors ({reach})")
    lines.append(
        "Selected bandwidths: "
        + "; ".join(scales)
        + ". Smaller bandwidths mean the association shifts over shorter distances."
    )
    lines.extend(
        _term_reading(df, entity, term, float(critical_t[term]))
        for term in covariates
        if term in critical_t
    )
    lines.append(
        "Coefficients are on the standardized scale (MGWR requires z-standardized "
        "variables), so magnitudes are comparable across terms; significance uses "
        "per-term corrected alphas."
    )
    return "\n\n".join([*lines, _ASSOC_NOTE])
