"""Plain-language interpretation for the ``learn_*`` teaching sandboxes.

Duck-typed against :class:`geometrics._types.SandboxResult`: one dispatcher reads the
result's ``topic`` and ``summary`` and returns the demonstration's takeaway in
Markdown. Sandboxes plant their parameters, so — unusually for this package — the
truth is *known*; the interpretation still closes with the associational note because
the lesson transfers to real data, where it is not.
"""

from __future__ import annotations

from typing import Any

from geometrics.pedagogy._format import fmt_num, is_significant
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

__all__ = ["interpret_sandbox"]


def _spatial_autocorrelation(s: dict[str, float]) -> str:
    return (
        f"With the planted dependence at ρ = {fmt_num(s['rho'])}, Moran's I averages "
        f"{fmt_num(s['moran_focal'])} across the simulations, against a no-dependence "
        f"baseline of E[I] = {fmt_num(s['expected_i'])} (the ρ = 0 runs average "
        f"{fmt_num(s['moran_at_zero'])}). "
        f"{fmt_num(100 * s['share_significant_focal'], 3)}% of the focal-ρ runs are "
        "significant at 5% — as ρ rises, neighbors look alike and the statistic "
        "pulls away from its null."
    )


def _spatial_weights(s: dict[str, float]) -> str:
    return (
        "The same simulated field reads differently through different graphs: "
        f"Moran's I is {fmt_num(s['moran_queen'])} under queen contiguity (the "
        f"data-generating graph), {fmt_num(s['moran_rook'])} under rook and "
        f"{fmt_num(s['moran_knn'])} under k-nearest neighbors. The clustering is "
        "detected either way, but the magnitude shifts with what 'neighbor' means — "
        "conclusions should be checked across weights, not built on one W."
    )


def _local_moran(s: dict[str, float]) -> str:
    return (
        f"LISA recovers {fmt_num(100 * s['sensitivity_hot'], 3)}% of the planted hot "
        f"block as High-High and {fmt_num(100 * s['sensitivity_cold'], 3)}% of the "
        f"cold block as Low-Low, while {fmt_num(100 * s['false_positive_rate'], 3)}% "
        f"of the un-planted cells are flagged too (α = {fmt_num(s['alpha'])}, so "
        "some by-chance flags are expected across hundreds of local tests). Local "
        "maps locate candidates — they do not certify clusters."
    )


def _spatial_impacts(s: dict[str, float]) -> str:
    return (
        "The Monte-Carlo impact decomposition lands close to the planted truth: "
        f"direct {fmt_num(s['est_direct'])} vs {fmt_num(s['true_direct'])} true, "
        f"indirect {fmt_num(s['est_indirect'])} vs {fmt_num(s['true_indirect'])}, "
        f"total {fmt_num(s['est_total'])} vs {fmt_num(s['true_total'])} "
        f"(ρ̂ = {fmt_num(s['rho_hat'])} against a planted {fmt_num(s['rho'])}). "
        "This is why spatial-model coefficients are read through impacts: with "
        "feedback via ρ, β alone is not the marginal association."
    )


def _spatial_lag_model(s: dict[str, float]) -> str:
    return (
        f"OLS, which omits the spatial lag Wy, puts the slope at "
        f"{fmt_num(s['ols_coef'])} — off the planted β = {fmt_num(s['true_beta'])} "
        f"by {fmt_num(s['ols_bias'])} because the spatial multiplier is absorbed "
        f"into the coefficient. The ML spatial-lag model recovers "
        f"β̂ = {fmt_num(s['sar_beta'])} and ρ̂ = {fmt_num(s['sar_rho'])} "
        f"(planted ρ = {fmt_num(s['rho'])}) by modeling the dependence instead of "
        "ignoring it."
    )


def _beta_convergence(s: dict[str, float]) -> str:
    return (
        f"The growth-on-initial regression recovers the planted convergence: "
        f"β̂ = {fmt_num(s['est_beta'])} against a true {fmt_num(s['true_beta'])} "
        f"(SE {fmt_num(s['se_total'])}). That maps to a convergence speed of "
        f"{fmt_num(100 * s['speed'], 3)}% per period (true "
        f"{fmt_num(100 * s['true_speed'], 3)}%) and a half-life of "
        f"{fmt_num(s['half_life'])} periods (true {fmt_num(s['true_half_life'])}) — "
        "initially-poorer units grow faster by construction, and the estimator "
        "sees it."
    )


def _sigma_convergence(s: dict[str, float]) -> str:
    return (
        f"Dispersion contracts by the planted factor ρ = {fmt_num(s['rho'])} each "
        f"period, so the log-dispersion trend should be ln ρ = "
        f"{fmt_num(s['true_slope'])}; the fitted std trend is "
        f"{fmt_num(s['std_slope'])}. The Gini ({fmt_num(s['gini_slope'])}) and CV "
        f"({fmt_num(s['cv_slope'])}) trends track it only approximately because they "
        "are computed on levels rather than logs."
    )


def _convergence_clubs(s: dict[str, float]) -> str:
    converged = bool(s["converged"])
    global_read = (
        "the global log(t) test fails to reject convergence"
        if converged
        else f"the global log(t) test rejects overall convergence (t = {fmt_num(s['global_tstat'])})"
    )
    return (
        f"With {fmt_num(s['true_clubs'], 2)} clubs planted, {global_read} and the "
        f"clustering finds {fmt_num(s['detected_clubs'], 2)} clubs, assigning "
        f"{fmt_num(100 * s['accuracy'], 3)}% of units to the group that matches "
        "their planted club. Club convergence means the paths within each detected "
        "group contract while the groups themselves stay apart."
    )


def _markov_chains(s: dict[str, float]) -> str:
    return (
        f"Across {fmt_num(s['n_transitions'], 6)} observed transitions the estimated "
        f"matrix sits within {fmt_num(s['max_abs_error'])} of every planted "
        f"probability, and the implied long-run (ergodic) distribution is off by "
        f"{fmt_num(s['ergodic_l1_error'])} in total. Diagonal persistence averages "
        f"{fmt_num(s['mean_persistence_est'])} against a planted "
        f"{fmt_num(s['mean_persistence_true'])} — with enough transitions, the chain "
        "gives back what was planted."
    )


def _spatial_markov(s: dict[str, float]) -> str:
    lr_read = (
        "rejects spatial homogeneity"
        if is_significant(s["lr_p"])
        else "does not reject spatial homogeneity"
    )
    return (
        f"Mobility was planted to depend on the neighborhood (context effect "
        f"{fmt_num(s['contextual'])}), and the conditioned chains show it: a "
        f"middle-class unit moves up with probability "
        f"{fmt_num(s['up_prob_rich_nbrs'])} among rich neighbors but only "
        f"{fmt_num(s['up_prob_poor_nbrs'])} among poor ones (gap "
        f"{fmt_num(s['contextual_gap_est'])}). The LR test {lr_read} "
        f"(p = {fmt_num(s['lr_p'])})."
    )


def _theil_decomposition(s: dict[str, float]) -> str:
    return (
        "As the planted between-group gap widens, the Theil decomposition shifts "
        f"from within- to between-group inequality: the between share rises from "
        f"{fmt_num(s['between_share_min_gap'])} at the smallest gap to "
        f"{fmt_num(s['between_share_max_gap'])} at the largest, and the estimator "
        f"never strays more than {fmt_num(s['max_abs_share_error'])} from the "
        "independently computed truth. Between + within = total, exactly."
    )


_DISPATCH = {
    "spatial_autocorrelation": _spatial_autocorrelation,
    "spatial_weights": _spatial_weights,
    "local_moran": _local_moran,
    "spatial_impacts": _spatial_impacts,
    "spatial_lag_model": _spatial_lag_model,
    "beta_convergence": _beta_convergence,
    "sigma_convergence": _sigma_convergence,
    "convergence_clubs": _convergence_clubs,
    "markov_chains": _markov_chains,
    "spatial_markov": _spatial_markov,
    "theil_decomposition": _theil_decomposition,
}


def interpret_sandbox(result: Any, *, lang: str = "en") -> str:
    """Interpret a ``learn_*`` sandbox result in plain language.

    Parameters
    ----------
    result
        A sandbox result exposing ``topic`` and ``summary``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        The demonstration's Markdown takeaway.
    """
    renderer = _DISPATCH.get(str(result.topic))
    if renderer is None:
        raise KeyError(
            f"no sandbox interpretation registered for topic {result.topic!r}"
        )
    body = renderer(dict(result.summary))
    return (
        "**What this sandbox shows** — the data were simulated, so the truth is "
        f"known. {body}\n\n{_ASSOC_NOTE}"
    )
