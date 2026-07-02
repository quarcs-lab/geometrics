"""Spatial-econometrics sandboxes: planted spillovers and the omitted-lag bias.

Both simulate the spatial Durbin reduced form ``y = (I - rho W)^-1 (beta x + gamma Wx
+ eps)`` on a small lattice **with geometry** (their estimators are the real
:func:`geometrics.analyze_spatial_model`), then compare estimates against the planted
truth computed in closed form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from geometrics._theme import apply_default_layout, color_for
from geometrics._types import SandboxResult
from geometrics.sandbox._dgp import dense_w, lattice_gdf, queen_w_from_gdf, solve_sar
from geometrics.sandbox._validate import check_float, check_int

__all__ = ["learn_spatial_spillovers", "learn_omitted_spatial_lag"]


def _sdm_field(
    *,
    side: int,
    beta: float,
    gamma: float,
    rho: float,
    noise: float,
    seed: int,
):
    """Simulate the SDM reduced form on a lattice; return (gdf, w, w_dense, data)."""
    gdf = lattice_gdf(side)
    w = queen_w_from_gdf(gdf)
    ids = list(gdf["unit"])
    w_dense = dense_w(w, ids)

    rng = np.random.default_rng(seed)
    n = side * side
    x = rng.normal(0.0, 1.0, n)
    eps = rng.normal(0.0, noise, n)
    xb = beta * x + gamma * (w_dense @ x) + eps
    y = solve_sar(w_dense, xb, rho)
    data = pd.DataFrame({"unit": ids, "year": 2020, "x": x, "y": y})
    return gdf, w, w_dense, data


def _true_impacts(
    w_dense: np.ndarray, *, beta: float, gamma: float, rho: float
) -> tuple[float, float, float]:
    """Closed-form LeSage-Pace impacts of the planted SDM: (direct, indirect, total)."""
    n = w_dense.shape[0]
    multiplier = np.linalg.inv(np.eye(n) - rho * w_dense)
    partials = multiplier @ (beta * np.eye(n) + gamma * w_dense)
    direct = float(np.trace(partials) / n)
    total = float((beta + gamma) / (1.0 - rho))
    return direct, total - direct, total


def learn_spatial_spillovers(
    *,
    side: int = 10,
    beta: float = 1.0,
    gamma: float = 0.5,
    rho: float = 0.5,
    noise: float = 0.5,
    n_draws: int = 5000,
    seed: int = 0,
) -> SandboxResult:
    """Plant direct and indirect effects, then recover them as LeSage-Pace impacts.

    Simulates ``y = (I - rho W)^-1 (beta x + gamma Wx + eps)`` on a lattice, so the
    true impacts are known in closed form — direct = tr[(I-ρW)⁻¹(βI+γW)]/n and
    total = (β+γ)/(1-ρ) — then estimates a spatial Durbin model with
    :func:`geometrics.analyze_spatial_model` and compares its Monte-Carlo impact
    decomposition against the truth. This is why spatial-model coefficients are read
    through impacts: β alone is not the marginal effect once feedback via ρ exists.

    Parameters
    ----------
    side
        Lattice side length (n = side²).
    beta
        Planted own-place coefficient on ``x``.
    gamma
        Planted neighbor coefficient on ``Wx`` (drives spillovers alongside ρ).
    rho
        Planted spatial-lag parameter, |ρ| < 1.
    noise
        Standard deviation of the innovation.
    n_draws
        Monte-Carlo draws behind the estimated impact standard errors.
    seed
        Random seed (also passed to the estimator's Monte-Carlo step).

    Returns
    -------
    SandboxResult
        ``df`` (direct/indirect/total, estimated vs true), ``fig``, ``summary``,
        ``topic`` and the simulated cross-section in ``data``.

    Examples
    --------
    ```python
    import geometrics as gm

    res = gm.learn_spatial_spillovers(rho=0.7)
    res.df
    ```
    """
    from geometrics.spatial_models import analyze_spatial_model

    func = "learn_spatial_spillovers"
    side = check_int("side", side, minimum=5, func=func)
    beta = check_float("beta", beta, func=func)
    gamma = check_float("gamma", gamma, func=func)
    rho = check_float("rho", rho, minimum=-1.0, maximum=1.0, inclusive=False, func=func)
    noise = check_float("noise", noise, minimum=0.0, inclusive=False, func=func)
    n_draws = check_int("n_draws", n_draws, minimum=100, func=func)

    gdf, w, w_dense, data = _sdm_field(
        side=side, beta=beta, gamma=gamma, rho=rho, noise=noise, seed=seed
    )
    est = analyze_spatial_model(
        data,
        "y",
        "x",
        gdf=gdf,
        w=w,
        model="durbin",
        n_draws=n_draws,
        seed=seed,
        entity="unit",
        time="year",
    )
    assert est.impacts is not None
    impact_row = est.impacts.loc[est.impacts["term"] == "x"].iloc[0]

    true_direct, true_indirect, true_total = _true_impacts(
        w_dense, beta=beta, gamma=gamma, rho=rho
    )
    df = pd.DataFrame(
        {
            "effect": ["direct", "indirect", "total"],
            "estimate": [
                float(impact_row["direct"]),
                float(impact_row["indirect"]),
                float(impact_row["total"]),
            ],
            "se": [
                float(impact_row["se_direct"]),
                float(impact_row["se_indirect"]),
                float(impact_row["se_total"]),
            ],
            "true": [true_direct, true_indirect, true_total],
        }
    )

    fig = go.Figure(
        [
            go.Bar(
                x=df["effect"],
                y=df["estimate"],
                error_y={"type": "data", "array": df["se"], "visible": True},
                name="estimated (SDM impacts)",
                marker={"color": color_for(0)},
            ),
            go.Bar(
                x=df["effect"],
                y=df["true"],
                name="planted truth",
                marker={"color": color_for(9)},
            ),
        ]
    )
    apply_default_layout(
        fig,
        title="Spillovers you planted, impacts recovered",
        subtitle=(
            f"y = (I - {rho:g}W)^-1({beta:g}x + {gamma:g}Wx + eps) on the lattice; "
            f"SEs from {n_draws} Monte-Carlo draws"
        ),
        barmode="group",
        xaxis={"title": ""},
        yaxis={"title": "Impact of x on y"},
    )

    summary = {
        "true_direct": true_direct,
        "est_direct": float(impact_row["direct"]),
        "true_indirect": true_indirect,
        "est_indirect": float(impact_row["indirect"]),
        "true_total": true_total,
        "est_total": float(impact_row["total"]),
        "rho": float(rho),
        "rho_hat": float(est.rho),
    }
    return SandboxResult(
        df=df,
        fig=fig,
        summary=summary,
        topic="spatial_impacts",
        data=data,
        w_spec=est.w_spec,
    )


def learn_omitted_spatial_lag(
    *,
    side: int = 10,
    beta: float = 1.0,
    rho: float = 0.7,
    noise: float = 0.5,
    seed: int = 0,
) -> SandboxResult:
    """Show why ignoring spatial dependence biases OLS — and how SAR repairs it.

    Simulates ``y = (I - rho W)^-1 (beta x + eps)``: outcomes spill over through the
    spatial multiplier, so OLS on ``y ~ x`` (which omits the spatial lag ``Wy``)
    absorbs the feedback into its slope and overstates β. The ML spatial-lag (SAR)
    estimator models the dependence and recovers both β and ρ.

    Parameters
    ----------
    side
        Lattice side length (n = side²).
    beta
        Planted coefficient on ``x``.
    rho
        Planted spatial-lag parameter, |ρ| < 1 (drives the OLS bias).
    noise
        Standard deviation of the innovation.
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (OLS vs SAR vs true coefficient), ``fig``, ``summary``, ``topic`` and
        the simulated cross-section in ``data``.

    Examples
    --------
    ```python
    import geometrics as gm

    res = gm.learn_omitted_spatial_lag(rho=0.7)
    print(res.summary["ols_coef"], res.summary["sar_beta"])
    ```
    """
    from geometrics.spatial_models import analyze_spatial_model

    func = "learn_omitted_spatial_lag"
    side = check_int("side", side, minimum=5, func=func)
    beta = check_float("beta", beta, func=func)
    rho = check_float("rho", rho, minimum=-1.0, maximum=1.0, inclusive=False, func=func)
    noise = check_float("noise", noise, minimum=0.0, inclusive=False, func=func)

    gdf, w, _, data = _sdm_field(
        side=side, beta=beta, gamma=0.0, rho=rho, noise=noise, seed=seed
    )
    common = {"gdf": gdf, "w": w, "entity": "unit", "time": "year", "seed": seed}
    ols = analyze_spatial_model(data, "y", "x", model="ols", **common)
    sar = analyze_spatial_model(data, "y", "x", model="lag", **common)

    def _coef(res) -> float:
        return float(res.df.loc[res.df["term"] == "x", "estimate"].iloc[0])

    ols_coef = _coef(ols)
    sar_beta = _coef(sar)

    df = pd.DataFrame(
        {
            "model": ["OLS (omits Wy)", "SAR (ML)", "true value"],
            "x_coefficient": [ols_coef, sar_beta, float(beta)],
        }
    )
    fig = go.Figure(
        go.Bar(
            x=df["model"],
            y=df["x_coefficient"],
            marker={"color": [color_for(0), color_for(2), color_for(9)]},
        )
    )
    fig.add_hline(
        y=float(beta),
        line_dash="dash",
        line_color="rgba(0,0,0,0.5)",
        annotation_text="planted β",
    )
    apply_default_layout(
        fig,
        title="The omitted spatial lag inflates OLS",
        subtitle=f"y = (I - {rho:g}W)^-1({beta:g}x + eps); SAR also recovers ρ",
        xaxis={"title": ""},
        yaxis={"title": "Estimated coefficient on x"},
    )

    summary = {
        "true_beta": float(beta),
        "ols_coef": ols_coef,
        "sar_beta": sar_beta,
        "sar_rho": float(sar.rho),
        "ols_bias": ols_coef - float(beta),
        "rho": float(rho),
    }
    return SandboxResult(
        df=df,
        fig=fig,
        summary=summary,
        topic="spatial_lag_model",
        data=data,
        w_spec=sar.w_spec,
    )
