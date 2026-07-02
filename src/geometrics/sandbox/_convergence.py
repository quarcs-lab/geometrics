"""Convergence sandboxes: β at a planted speed, σ on a planted path, two clubs.

These are aspatial — the DGPs plant convergence dynamics in a plain panel and the
estimators are the real :func:`geometrics.analyze_beta_convergence`,
:func:`geometrics.analyze_sigma_convergence` and
:func:`geometrics.analyze_convergence_clubs`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from geometrics._theme import apply_default_layout, color_for
from geometrics._types import SandboxResult
from geometrics.sandbox._dgp import convergence_panel
from geometrics.sandbox._validate import check_float, check_int

__all__ = [
    "learn_beta_convergence",
    "learn_sigma_convergence",
    "learn_convergence_clubs",
]


def learn_beta_convergence(
    *,
    n_units: int = 60,
    n_periods: int = 6,
    convergence_rate: float = 0.02,
    growth_const: float = 0.05,
    noise: float = 0.005,
    seed: int = 0,
) -> SandboxResult:
    """Plant a convergence rate, then watch the growth-on-initial regression find it.

    Simulates ``log y_it = log y_i0 + t (a - b log y_i0) + noise`` so annualized
    growth is exactly ``a - b log y_i0`` (up to noise) and the β-convergence slope is
    ``-b`` by construction. The figure is the real
    :func:`geometrics.analyze_beta_convergence` scatter with the planted-truth line
    drawn on top; the summary compares estimated and true β, speed and half-life.

    Parameters
    ----------
    n_units
        Number of entities.
    n_periods
        Number of periods (the horizon is ``n_periods - 1``).
    convergence_rate
        The planted ``b`` > 0: poorer units grow faster by ``b`` per unit of initial
        log level, so the regression slope is ``-b``.
    growth_const
        The common growth constant ``a``.
    noise
        Standard deviation of the per-period log noise.
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (β / speed / half-life, estimated vs true), ``fig``, ``summary``,
        ``topic`` and the simulated panel in ``data``.

    Examples
    --------
    ```python
    import geometrics as gm

    res = gm.learn_beta_convergence(convergence_rate=0.03)
    print(res.summary["est_beta"], res.summary["true_beta"])
    ```
    """
    from geometrics.convergence import analyze_beta_convergence

    func = "learn_beta_convergence"
    n_units = check_int("n_units", n_units, minimum=12, func=func)
    n_periods = check_int("n_periods", n_periods, minimum=2, func=func)
    convergence_rate = check_float(
        "convergence_rate", convergence_rate, minimum=0.0, inclusive=False, func=func
    )
    growth_const = check_float("growth_const", growth_const, func=func)
    noise = check_float("noise", noise, minimum=0.0, func=func)

    ids = [f"r{i:03d}" for i in range(n_units)]
    periods = tuple(range(2000, 2000 + n_periods))
    panel = convergence_panel(
        ids,
        periods=periods,
        b=convergence_rate,
        a=growth_const,
        noise_sd=noise,
        seed=seed,
    )
    res = analyze_beta_convergence(
        panel, "gdppc", entity="unit", time="year", model="ols"
    )

    true_beta = -convergence_rate
    horizon = float(res.horizon)
    true_speed = -math.log1p(horizon * true_beta) / horizon
    true_half_life = math.log(2.0) / true_speed

    fig = res.fig
    x0 = float(res.df["initial"].min())
    x1 = float(res.df["initial"].max())
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[
                growth_const + true_beta * x0,
                growth_const + true_beta * x1,
            ],
            mode="lines",
            line={"color": color_for(9), "dash": "dash", "width": 2},
            name="planted truth",
        )
    )
    apply_default_layout(
        fig,
        title="β-convergence at a planted rate",
        subtitle=(
            f"growth = {growth_const:g} - {convergence_rate:g} x log y0 + noise; "
            f"the fitted slope should sit on the dashed truth"
        ),
    )

    df = pd.DataFrame(
        {
            "quantity": ["beta", "speed", "half_life"],
            "estimated": [
                float(res.beta_total),
                float(res.speed),
                float(res.half_life),
            ],
            "true": [true_beta, true_speed, true_half_life],
        }
    )
    summary = {
        "true_beta": true_beta,
        "est_beta": float(res.beta_total),
        "se_total": float(res.se_total),
        "speed": float(res.speed),
        "half_life": float(res.half_life),
        "true_speed": true_speed,
        "true_half_life": true_half_life,
    }
    return SandboxResult(
        df=df, fig=fig, summary=summary, topic="beta_convergence", data=panel
    )


def learn_sigma_convergence(
    *,
    n_units: int = 60,
    n_periods: int = 21,
    rho: float = 0.93,
    noise: float = 0.0,
    seed: int = 0,
) -> SandboxResult:
    """Plant a shrinking dispersion path, then watch the σ trend recover it.

    Simulates ``log y_it = mu + (log y_i0 - mu) rho^t`` so the cross-sectional
    standard deviation of ``log y`` contracts geometrically — ``sigma_t = sigma_0
    rho^t`` — and the log-dispersion trend of the standard deviation is exactly
    ``ln(rho)`` per period. :func:`geometrics.analyze_sigma_convergence` fits that
    trend (plus Gini and CV variants, which track it only approximately because they
    are computed on levels).

    Parameters
    ----------
    n_units
        Number of entities.
    n_periods
        Number of periods.
    rho
        Per-period contraction factor of deviations, 0 < ρ < 1 (smaller = faster
        σ-convergence; the planted trend slope is ``ln ρ``).
    noise
        Standard deviation of optional per-period log noise (0 keeps the geometric
        path exact).
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (trend slope per measure vs the planted ``ln ρ``), ``fig``,
        ``summary``, ``topic`` and the simulated panel in ``data``.

    Examples
    --------
    ```python
    import geometrics as gm

    res = gm.learn_sigma_convergence(rho=0.9)
    print(res.summary["std_slope"], res.summary["true_slope"])
    ```
    """
    from geometrics.convergence import analyze_sigma_convergence

    func = "learn_sigma_convergence"
    n_units = check_int("n_units", n_units, minimum=5, func=func)
    n_periods = check_int("n_periods", n_periods, minimum=3, func=func)
    rho = check_float("rho", rho, minimum=0.0, maximum=1.0, inclusive=False, func=func)
    noise = check_float("noise", noise, minimum=0.0, func=func)

    rng = np.random.default_rng(seed)
    mu = 9.0
    log_y0 = rng.normal(mu, 0.8, n_units)
    ids = [f"r{i:03d}" for i in range(n_units)]

    rows = []
    for t in range(n_periods):
        log_y = mu + (log_y0 - mu) * rho**t
        if noise > 0 and t:
            log_y = log_y + rng.normal(0.0, noise, n_units)
        rows.append(pd.DataFrame({"unit": ids, "year": 2000 + t, "y": np.exp(log_y)}))
    panel = pd.concat(rows, ignore_index=True)

    res = analyze_sigma_convergence(panel, "y", entity="unit", time="year")
    true_slope = math.log(rho)

    observed = res.df  # per-period dispersion table; time column carries its own name
    sigma0 = float(observed["std"].iloc[0])
    t_grid = np.arange(n_periods)
    fig = go.Figure(
        [
            go.Scatter(
                x=observed["year"],
                y=observed["std"],
                mode="lines+markers",
                name="observed std of log y",
                line={"color": color_for(0), "width": 3},
            ),
            go.Scatter(
                x=observed["year"],
                y=sigma0 * rho**t_grid,
                mode="lines",
                name="planted path σ0·ρ^t",
                line={"color": color_for(9), "dash": "dash"},
            ),
        ]
    )
    apply_default_layout(
        fig,
        title="σ-convergence on a planted contraction path",
        subtitle=f"deviations shrink by ρ = {rho:g} per period; trend slope = ln ρ = {true_slope:.3f}",
        xaxis={"title": "period"},
        yaxis={"title": "cross-sectional dispersion"},
    )

    df = pd.DataFrame(
        {
            "measure": ["std", "gini", "cv", "planted (ln rho)"],
            "trend_slope": [
                float(res.std_slope),
                float(res.gini_slope),
                float(res.cv_slope),
                true_slope,
            ],
        }
    )
    summary = {
        "rho": float(rho),
        "true_slope": true_slope,
        "std_slope": float(res.std_slope),
        "gini_slope": float(res.gini_slope),
        "cv_slope": float(res.cv_slope),
    }
    notes = (
        "The std trend matches ln(rho) exactly by construction; Gini and CV are "
        "computed on levels, so their trends track the planted slope only "
        "approximately.",
    )
    return SandboxResult(
        df=df,
        fig=fig,
        summary=summary,
        topic="sigma_convergence",
        data=panel,
        notes=notes,
    )


def learn_convergence_clubs(
    *,
    n_per_club: int = 15,
    levels: tuple[float, ...] = (10.0, 9.0),
    n_periods: int = 35,
    rho: float = 0.9,
    spread: float = 0.4,
    noise: float = 0.002,
    seed: int = 0,
) -> SandboxResult:
    """Plant convergence clubs, then watch the Phillips-Sul algorithm find them.

    Each club ``k`` converges to its own level: unit ``j`` follows ``x_jt =
    levels[k] + dev_j rho^t + noise`` with ``dev_j ~ U(-spread, spread)``, so within a
    club the transition paths contract while the between-club gaps persist — global
    convergence should be rejected and the clustering should recover the planted
    groups. The summary reports the detected club count and the assignment accuracy.

    Parameters
    ----------
    n_per_club
        Units per planted club.
    levels
        The clubs' long-run (log) levels — one entry per club, at least two.
    n_periods
        Number of periods (the log(t) test needs a long panel).
    rho
        Per-period contraction of within-club deviations, 0 < ρ < 1.
    spread
        Half-width of the uniform initial deviations around each club level.
    noise
        Standard deviation of the per-period noise.
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (unit, planted club, detected club), ``fig`` (the within-club
        transition paths from the real estimator), ``summary``, ``topic`` and the
        simulated panel in ``data``.

    Examples
    --------
    ```python
    import geometrics as gm

    res = gm.learn_convergence_clubs(levels=(10.0, 9.2, 8.5))
    print(res.summary["detected_clubs"], res.summary["accuracy"])
    ```
    """
    from geometrics.clubs import analyze_convergence_clubs

    func = "learn_convergence_clubs"
    n_per_club = check_int("n_per_club", n_per_club, minimum=5, func=func)
    if not isinstance(levels, (tuple, list)) or len(levels) < 2:
        raise ValueError(f"{func}: levels must supply at least two club levels")
    club_levels = [
        check_float(f"levels[{i}]", lv, func=func) for i, lv in enumerate(levels)
    ]
    n_periods = check_int("n_periods", n_periods, minimum=10, func=func)
    rho = check_float("rho", rho, minimum=0.0, maximum=1.0, inclusive=False, func=func)
    spread = check_float("spread", spread, minimum=0.0, inclusive=False, func=func)
    noise = check_float("noise", noise, minimum=0.0, func=func)

    rng = np.random.default_rng(seed)
    n_clubs = len(club_levels)
    n_total = n_clubs * n_per_club
    unit_levels = np.repeat(club_levels, n_per_club)
    true_club = np.repeat(np.arange(1, n_clubs + 1), n_per_club)
    ids = [f"c{k + 1}u{j:02d}" for k in range(n_clubs) for j in range(n_per_club)]
    devs = rng.uniform(-spread, spread, n_total)

    rows = []
    for t in range(n_periods):
        eps = rng.normal(0.0, noise, n_total) if noise > 0 else np.zeros(n_total)
        x = unit_levels + devs * rho**t + eps
        rows.append(pd.DataFrame({"unit": ids, "year": 2000 + t, "x": x}))
    panel = pd.concat(rows, ignore_index=True)

    res = analyze_convergence_clubs(panel, "x", entity="unit", time="year")

    membership = res.membership.set_index("entity")["club"]
    detected = [int(membership.get(u, 0)) for u in ids]
    df = pd.DataFrame({"unit": ids, "true_club": true_club, "detected_club": detected})

    # Modal-match accuracy: map each detected club to the planted club it mostly
    # contains, then score the share of units whose planted club matches that map.
    correct = 0
    for club_id, block in df.groupby("detected_club"):
        if club_id == 0:  # divergent group — never correct
            continue
        correct += int(block["true_club"].value_counts().iloc[0])
    accuracy = correct / n_total

    summary = {
        "true_clubs": float(n_clubs),
        "detected_clubs": float(res.n_clubs),
        "accuracy": float(accuracy),
        "global_tstat": float(res.global_tstat),
        "converged": float(res.converged),
    }
    fig = res.fig
    apply_default_layout(
        fig,
        title="Planted clubs, recovered by Phillips-Sul",
        subtitle=(
            f"{n_clubs} clubs x {n_per_club} units at levels "
            f"{tuple(round(lv, 2) for lv in club_levels)}; deviations shrink by ρ = {rho:g}"
        ),
    )
    return SandboxResult(
        df=df, fig=fig, summary=summary, topic="convergence_clubs", data=panel
    )
