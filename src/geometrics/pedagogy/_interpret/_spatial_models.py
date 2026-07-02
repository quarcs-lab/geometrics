"""Plain-language interpretation for the spatial-models vertical (spreg suite).

Duck-typed against :class:`geometrics._types.SpatialModelResult`,
:class:`geometrics._types.SpatialDiagnosticsResult` and
:class:`geometrics._types.WeightsRobustnessResult` (this module never imports the
result classes): each function reads the result's frames and scalars and returns
Markdown describing associations — never causal language.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from geometrics.pedagogy._format import fmt_num, significance_phrase
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

__all__ = [
    "interpret_spatial_model",
    "interpret_spatial_diagnostics",
    "interpret_weights_robustness",
]

_MODEL_PHRASES = {
    "ols": "non-spatial OLS",
    "lag": "spatial lag (SAR)",
    "error": "spatial error (SEM)",
    "slx": "SLX (spatially lagged regressors)",
    "durbin": "spatial Durbin (SDM)",
    "durbin_error": "spatial Durbin error (SDEM)",
}

#: Impact terms described in full before the summary truncates.
_MAX_IMPACT_TERMS = 3


def _z_p(est: float, se: float) -> float:
    """Two-sided normal p-value of ``est / se`` (NaN for a degenerate ``se``)."""
    if not (math.isfinite(est) and math.isfinite(se)) or se <= 0:
        return float("nan")
    z = abs(est / se)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def _param_p(df: Any, term_prefix: str, value: float) -> float:
    """Find the p-value of the spatial parameter row in the tidy frame.

    Matches the row whose term starts with ``term_prefix`` and whose estimate equals
    ``value`` (the parameter located from the result scalar), so it works whatever
    display label the outcome carried. Returns NaN when not found.
    """
    try:
        rows = df[
            df["term"].astype(str).str.startswith(term_prefix)
            & np.isclose(df["estimate"].astype(float), value)
        ]
        if len(rows):
            return float(rows.iloc[0]["p"])
    except (KeyError, TypeError, ValueError):
        pass
    return float("nan")


def interpret_spatial_model(result: Any, *, lang: str = "en") -> str:
    """Interpret a fitted spatial model: spatial parameters, impacts and fit.

    Parameters
    ----------
    result
        A spatial-model result exposing ``df`` (term/estimate/se/z/p), ``model``,
        ``method``, ``rho``, ``lam``, ``r2``, ``aic``, ``n_obs``, ``outcome``,
        ``w_spec`` and (optionally) the per-regressor ``impacts`` frame.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of the spatial dependence parameters and the
        direct/indirect/total associations.
    """
    model = str(getattr(result, "model", "durbin"))
    method = str(getattr(result, "method", "ml")).upper()
    outcome = str(getattr(result, "outcome", "the outcome"))
    n_obs = int(getattr(result, "n_obs", 0))
    w_spec = str(getattr(result, "w_spec", "the spatial weights"))
    rho = float(getattr(result, "rho", float("nan")))
    lam = float(getattr(result, "lam", float("nan")))
    r2 = float(getattr(result, "r2", float("nan")))
    aic = float(getattr(result, "aic", float("nan")))
    impacts = getattr(result, "impacts", None)
    df = getattr(result, "df", None)

    phrase = _MODEL_PHRASES.get(model, model)
    lines = [
        f"A **{phrase}** model of **{outcome}** was estimated by {method} on "
        f"{n_obs:,} units (weights: {w_spec})."
    ]

    if math.isfinite(rho):
        p_rho = _param_p(df, "W_", rho) if df is not None else float("nan")
        direction = (
            "outcomes move together with neighbors' outcomes"
            if rho > 0
            else "outcomes move opposite to neighbors' outcomes"
        )
        lines.append(
            f"The spatial autoregressive parameter is **ρ = {fmt_num(rho)}** "
            f"({significance_phrase(p_rho)}): {direction}, so any local difference "
            "is amplified through the neighborhood multiplier "
            f"1/(1-ρ) ≈ {fmt_num(1.0 / (1.0 - rho))} rather than staying put."
        )
    if math.isfinite(lam):
        p_lam = _param_p(df, "lambda", lam) if df is not None else float("nan")
        lines.append(
            f"The spatial error parameter is **λ = {fmt_num(lam)}** "
            f"({significance_phrase(p_lam)}): unmodeled shocks are spatially "
            "correlated across neighbors, which matters for standard errors more "
            "than for the coefficients themselves."
        )
    if not math.isfinite(rho) and not math.isfinite(lam):
        lines.append(
            "No spatial parameter enters this specification, so the coefficients "
            "read as ordinary regression associations under the usual assumptions."
        )

    if impacts is not None and len(impacts):
        for _, row in impacts.head(_MAX_IMPACT_TERMS).iterrows():
            term = str(row["term"])
            total, se_total = float(row["total"]), float(row["se_total"])
            direct, indirect = float(row["direct"]), float(row["indirect"])
            lines.append(
                f"For **{term}**, a one-unit higher value is associated with a "
                f"direct change of {fmt_num(direct)} in the same unit's {outcome} "
                f"and an indirect (spillover) change of {fmt_num(indirect)} summed "
                f"across its neighbors — a total association of "
                f"**{fmt_num(total)}** ({significance_phrase(_z_p(total, se_total))})."
            )
        if len(impacts) > _MAX_IMPACT_TERMS:
            lines.append(
                f"(Impacts for the remaining {len(impacts) - _MAX_IMPACT_TERMS} "
                "regressor(s) are in `result.impacts`.)"
            )
    elif model in ("ols", "error"):
        lines.append(
            "This specification has no direct/indirect impact decomposition; "
            "coefficients are read directly from the table."
        )

    fit_bits = []
    if math.isfinite(r2):
        fit_bits.append(f"(pseudo-)R² = {fmt_num(r2)}")
    if math.isfinite(aic):
        fit_bits.append(f"AIC = {fmt_num(aic, 4)}")
    if fit_bits:
        lines.append(
            "Model fit: " + ", ".join(fit_bits) + " (compare AIC across "
            "specifications estimated on the same sample)."
        )

    return "\n\n".join([*lines, _ASSOC_NOTE])


def interpret_spatial_diagnostics(result: Any, *, lang: str = "en") -> str:
    """Interpret the LM specification tests and the model they point to.

    Parameters
    ----------
    result
        A spatial-diagnostics result exposing ``df`` (test/statistic/df/p),
        ``moran_i_resid``, ``recommendation``, ``reasoning``, ``alpha`` and
        ``w_spec``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of the residual dependence tests and the
        Anselin-Florax recommendation.
    """
    df = result.df
    alpha = float(getattr(result, "alpha", 0.05))
    moran_i = float(getattr(result, "moran_i_resid", float("nan")))
    w_spec = str(getattr(result, "w_spec", "the spatial weights"))
    recommendation = str(getattr(result, "recommendation", ""))
    reasoning = str(getattr(result, "reasoning", ""))

    by_test = {
        str(row["test"]): (float(row["statistic"]), float(row["p"]))
        for _, row in df.iterrows()
    }
    p_moran = by_test.get("moran_residuals", (float("nan"), float("nan")))[1]

    lines = [
        f"Moran's I on the OLS residuals is **{fmt_num(moran_i)}** "
        f"({significance_phrase(p_moran)}) under {w_spec} — "
        + (
            "the residuals cluster in space, so the non-spatial model leaves "
            "spatial structure on the table."
            if (not math.isnan(p_moran)) and p_moran < alpha
            else "little sign that the non-spatial model leaves spatial structure "
            "behind."
        )
    ]

    sig_tests = [
        label
        for name, label in (
            ("lm_lag", "LM lag"),
            ("lm_error", "LM error"),
            ("robust_lm_lag", "robust LM lag"),
            ("robust_lm_error", "robust LM error"),
        )
        if name in by_test and by_test[name][1] < alpha
    ]
    if sig_tests:
        lines.append(
            f"At α = {alpha:g}, the tests that reject no-spatial-dependence are: "
            f"{', '.join(sig_tests)}. The *robust* forms are the decisive ones — "
            "each tests one dependence channel while allowing for the other."
        )
    else:
        lines.append(
            f"At α = {alpha:g}, none of the LM tests rejects the null of no "
            "spatial dependence."
        )

    lines.append(f"**Recommendation: `{recommendation}`** — {reasoning}")

    return "\n\n".join([*lines, _ASSOC_NOTE])


def interpret_weights_robustness(result: Any, *, lang: str = "en") -> str:
    """Interpret how stable the focal impacts are across weights specifications.

    Parameters
    ----------
    result
        A weights-robustness result exposing ``df`` (one row per weights
        specification with direct/indirect/total impacts and standard errors),
        ``baseline``, ``focal`` and ``model``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of the sign, magnitude and confidence-interval stability
        of the impacts across the alternative weights.
    """
    df = result.df
    baseline = str(getattr(result, "baseline", ""))
    focal = str(getattr(result, "focal", "the focal regressor"))
    model = str(getattr(result, "model", "durbin"))
    n_specs = len(df)

    totals = df["total"].astype(float)
    finite = totals[np.isfinite(totals)]
    base_row = df[df["weights"].astype(str) == baseline]
    base_total = float(base_row["total"].iloc[0]) if len(base_row) else float("nan")

    lines = [
        f"The {_MODEL_PHRASES.get(model, model)} model was re-estimated under "
        f"**{n_specs} alternative spatial weights** specifications; the table and "
        f"figure compare the direct, indirect and total impacts of **{focal}** "
        f"against the **{baseline}** baseline (total = {fmt_num(base_total)})."
    ]

    if len(finite):
        same_sign = bool((finite > 0).all() or (finite < 0).all())
        spread = float(finite.max() - finite.min())
        if same_sign:
            lines.append(
                f"The total impact keeps the **same sign in all {len(finite)} "
                f"specifications**, ranging from {fmt_num(finite.min())} to "
                f"{fmt_num(finite.max())} (spread {fmt_num(spread)}) — the "
                "qualitative conclusion does not hinge on the weights choice."
            )
        else:
            lines.append(
                f"The total impact **changes sign across specifications** "
                f"(from {fmt_num(finite.min())} to {fmt_num(finite.max())}), so the "
                "conclusion is sensitive to the weights choice and should be "
                "reported with that caveat."
            )
        if math.isfinite(base_total):
            se = df["se_total"].astype(float)
            covered = (
                (totals - 1.96 * se <= base_total) & (base_total <= totals + 1.96 * se)
            ) | ~np.isfinite(totals)
            if bool(covered.all()):
                lines.append(
                    "Every specification's 95% interval covers the baseline "
                    "estimate, so the alternatives are statistically "
                    "indistinguishable from the baseline."
                )
            else:
                outliers = ", ".join(
                    str(nm) for nm in df.loc[~covered, "weights"].head(3)
                )
                lines.append(
                    f"Specification(s) whose 95% interval misses the baseline "
                    f"estimate: {outliers} — inspect these in the figure before "
                    "leaning on the baseline numbers."
                )

    aic = df["aic"].astype(float)
    if np.isfinite(aic).any():
        best = df.loc[aic.idxmin(), "weights"]
        lines.append(
            f"By AIC the best-fitting specification is **{best}** "
            f"(AIC = {fmt_num(float(aic.min()), 4)}); AIC comparisons are only "
            "meaningful across models estimated on the same sample."
        )

    return "\n\n".join([*lines, _ASSOC_NOTE])
