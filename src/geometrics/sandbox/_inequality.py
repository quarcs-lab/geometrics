"""Inequality sandbox: a Theil decomposition with a planted between/within split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from geometrics._theme import apply_default_layout, color_for
from geometrics._types import SandboxResult
from geometrics.sandbox._validate import check_float, check_int

__all__ = ["learn_theil_decomposition"]


def _theil_between_share(values: np.ndarray, groups: np.ndarray) -> float:
    """Between-group share of the Theil-T index, computed directly from first principles.

    ``T = sum((y/mu)/n * ln(y/mu))`` decomposes exactly into a between part (group
    means vs the grand mean) and a within part (each group's own Theil, weighted by
    its income share). This is an independent implementation, so the sandbox's truth
    does not lean on the pysal estimator it is checking.
    """
    mu = values.mean()
    total = float(np.mean((values / mu) * np.log(values / mu)))
    between = 0.0
    for g in np.unique(groups):
        y_g = values[groups == g]
        share_n = len(y_g) / len(values)
        mu_g = y_g.mean()
        between += share_n * (mu_g / mu) * np.log(mu_g / mu)
    return float(between / total) if total > 0 else float("nan")


def learn_theil_decomposition(
    *,
    n_groups: int = 3,
    n_per_group: int = 40,
    gaps: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    within_sd: float = 0.5,
    jitter: float = 0.0,
    seed: int = 0,
) -> SandboxResult:
    """Plant a between/within inequality split, then watch Theil decompose it.

    Builds group log-income distributions on a deterministic quantile grid — group
    ``g`` at gap level ``tau`` is centered at ``g * tau`` with within-group spread
    ``within_sd`` — and sweeps ``gaps`` as successive "periods" of a panel, so one
    call to :func:`geometrics.analyze_theil_decomposition` traces how the
    between-group share rises with the planted gap. The truth is computed with an
    independent numpy implementation of the Theil-T decomposition.

    Parameters
    ----------
    n_groups
        Number of groups.
    n_per_group
        Units per group.
    gaps
        The planted between-group log gaps, swept as periods (at least two values).
    within_sd
        Within-group standard deviation of log income (drives the within share).
    jitter
        Standard deviation of optional extra log noise (0 keeps the grid exact).
    seed
        Random seed (only used when ``jitter > 0``).

    Returns
    -------
    SandboxResult
        ``df`` (per gap: Theil, between, within, estimated and true between share),
        ``fig``, ``summary``, ``topic`` and the simulated panel in ``data``.

    Examples
    --------
    ```python
    import geometrics as gm

    res = gm.learn_theil_decomposition(within_sd=0.3)
    res.df[["gap", "between_share_est", "between_share_true"]]
    ```
    """
    from scipy.stats import norm

    from geometrics.regional_inequality import analyze_theil_decomposition

    func = "learn_theil_decomposition"
    n_groups = check_int("n_groups", n_groups, minimum=2, func=func)
    n_per_group = check_int("n_per_group", n_per_group, minimum=5, func=func)
    if not isinstance(gaps, (tuple, list)) or len(gaps) < 2:
        raise ValueError(f"{func}: gaps must supply at least two gap values")
    gap_values = [
        check_float(f"gaps[{i}]", g, minimum=0.0, func=func) for i, g in enumerate(gaps)
    ]
    within_sd = check_float(
        "within_sd", within_sd, minimum=0.0, inclusive=False, func=func
    )
    jitter = check_float("jitter", jitter, minimum=0.0, func=func)

    rng = np.random.default_rng(seed)
    quantiles = norm.ppf((np.arange(n_per_group) + 0.5) / n_per_group)
    group_index = np.repeat(np.arange(n_groups), n_per_group)
    ids = [f"g{g + 1}u{i:02d}" for g in range(n_groups) for i in range(n_per_group)]

    frames = []
    truth_rows = []
    for period, tau in enumerate(gap_values):
        log_y = group_index * tau + within_sd * np.tile(quantiles, n_groups)
        if jitter > 0:
            log_y = log_y + rng.normal(0.0, jitter, len(log_y))
        y = np.exp(log_y)
        frames.append(
            pd.DataFrame(
                {
                    "unit": ids,
                    "period": period,
                    "group": [f"G{g + 1}" for g in group_index],
                    "gap": tau,
                    "y": y,
                }
            )
        )
        truth_rows.append(
            {
                "period": period,
                "gap": tau,
                "true_share": _theil_between_share(y, group_index),
            }
        )
    panel = pd.concat(frames, ignore_index=True)
    truth = pd.DataFrame(truth_rows)

    res = analyze_theil_decomposition(panel, "y", "group", entity="unit", time="period")
    est = res.df.rename(columns={"time": "period"})
    df = est.merge(truth, on="period", how="left")
    df = df.rename(
        columns={
            "between_share": "between_share_est",
            "true_share": "between_share_true",
        }
    )
    keep = [
        "period",
        "gap",
        "theil",
        "between",
        "within",
        "between_share_est",
        "between_share_true",
    ]
    df = df[[c for c in keep if c in df.columns]]

    fig = go.Figure(
        [
            go.Scatter(
                x=df["gap"],
                y=df["between_share_true"],
                mode="lines",
                name="planted truth",
                line={"color": color_for(9), "dash": "dash"},
            ),
            go.Scatter(
                x=df["gap"],
                y=df["between_share_est"],
                mode="markers",
                name="estimated (TheilD)",
                marker={"color": color_for(0), "size": 11},
            ),
        ]
    )
    apply_default_layout(
        fig,
        title="The between-group share rises with the planted gap",
        subtitle=(
            f"{n_groups} groups x {n_per_group} units; within-group log sd = "
            f"{within_sd:g}"
        ),
        xaxis={"title": "planted between-group log gap"},
        yaxis={"title": "between share of Theil", "range": [0, 1]},
    )

    errors = (df["between_share_est"] - df["between_share_true"]).abs()
    summary = {
        "n_groups": float(n_groups),
        "within_sd": float(within_sd),
        "max_abs_share_error": float(errors.max()),
        "between_share_min_gap": float(df["between_share_est"].iloc[0]),
        "between_share_max_gap": float(df["between_share_est"].iloc[-1]),
    }
    return SandboxResult(
        df=df, fig=fig, summary=summary, topic="theil_decomposition", data=panel
    )
