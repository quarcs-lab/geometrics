"""Distribution-dynamics sandboxes: a planted Markov chain and a spatial variant.

Both require the ``dynamics`` extra (giddy) — the import is checked first so the
``ImportError`` names the sandbox and the exact install command, matching the
``analyze_markov_*`` behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from geometrics._theme import apply_default_layout, color_for
from geometrics._types import SandboxResult
from geometrics.sandbox._dgp import (
    dense_w,
    ergodic_distribution,
    lattice_gdf,
    queen_w_from_gdf,
    simulate_markov_states,
)
from geometrics.sandbox._validate import check_float, check_int

__all__ = ["learn_markov_chains", "learn_spatial_markov"]

_DEFAULT_P = (
    (0.80, 0.15, 0.05),
    (0.10, 0.80, 0.10),
    (0.05, 0.15, 0.80),
)


def _check_transition_matrix(p: object, *, func: str) -> np.ndarray:
    """Validate and return ``p`` as a square row-stochastic array."""
    try:
        arr = np.asarray(p, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{func}: p must be a square matrix of floats") from exc
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        raise ValueError(f"{func}: p must be square with k >= 2, got shape {arr.shape}")
    if (arr < 0).any() or not np.allclose(arr.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError(f"{func}: p rows must be non-negative and sum to 1")
    return arr


def _states_to_values(rng: np.random.Generator, states: np.ndarray) -> np.ndarray:
    """Map integer states to well-separated continuous values (10s + 5 + noise)."""
    return 10.0 * states + 5.0 + rng.normal(0.0, 1.0, states.shape)


def learn_markov_chains(
    *,
    n_units: int = 100,
    n_periods: int = 30,
    p: tuple[tuple[float, ...], ...] = _DEFAULT_P,
    seed: int = 0,
) -> SandboxResult:
    """Plant a transition matrix, then watch the estimated chain recover it.

    Simulates ``n_units`` independent chains from the planted row-stochastic ``p``
    (started at its ergodic distribution), maps states to well-separated continuous
    values, and estimates the chain with :func:`geometrics.analyze_markov_transitions`
    using explicit class bins so the discretization is exact. The figure compares
    every estimated transition probability with its planted value; the summary
    reports the largest cell error and the ergodic-distribution error.

    Requires the ``dynamics`` extra (``pip install "geometrics[dynamics]"``).

    Parameters
    ----------
    n_units
        Number of independent chains (entities).
    n_periods
        Chain length; ``n_units * (n_periods - 1)`` transitions are observed.
    p
        The planted k x k row-stochastic transition matrix (k >= 2).
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (one row per matrix cell, planted vs estimated), ``fig``, ``summary``,
        ``topic`` and the simulated panel in ``data``.

    Examples
    --------
    ```python
    import geometrics as gm

    res = gm.learn_markov_chains(n_units=200)
    print(res.summary["max_abs_error"])
    ```
    """
    from geometrics.distribution_dynamics import (
        _import_giddy,
        analyze_markov_transitions,
    )

    func = "learn_markov_chains"
    _import_giddy(func)
    n_units = check_int("n_units", n_units, minimum=10, func=func)
    n_periods = check_int("n_periods", n_periods, minimum=3, func=func)
    p_true = _check_transition_matrix(p, func=func)
    k = p_true.shape[0]

    rng = np.random.default_rng(seed)
    states = simulate_markov_states(rng, p_true, n_units=n_units, n_periods=n_periods)
    values = _states_to_values(rng, states)

    ids = [f"r{i:03d}" for i in range(n_units)]
    panel = pd.DataFrame(
        {
            "unit": np.repeat(ids, n_periods),
            "year": np.tile(np.arange(2000, 2000 + n_periods), n_units),
            "y": values.ravel(),
        }
    )
    bins = [10.0 * s for s in range(1, k)]
    res = analyze_markov_transitions(
        panel, "y", entity="unit", time="year", k=k, bins=bins
    )

    p_est = res.p.to_numpy(dtype=float)
    labels = list(res.p.index)
    cells = []
    for i in range(k):
        for j in range(k):
            cells.append(
                {
                    "from_state": labels[i],
                    "to_state": labels[j],
                    "p_true": float(p_true[i, j]),
                    "p_est": float(p_est[i, j]),
                }
            )
    df = pd.DataFrame(cells)

    ergodic_true = ergodic_distribution(p_true)
    ergodic_est = res.steady_state.to_numpy(dtype=float)

    x_labels = [f"{row['from_state']} → {row['to_state']}" for _, row in df.iterrows()]
    fig = go.Figure(
        [
            go.Bar(
                x=x_labels,
                y=df["p_est"],
                name="estimated",
                marker={"color": color_for(0)},
            ),
            go.Bar(
                x=x_labels,
                y=df["p_true"],
                name="planted",
                marker={"color": color_for(9)},
            ),
        ]
    )
    apply_default_layout(
        fig,
        title="A planted Markov chain, recovered",
        subtitle=(
            f"{n_units} chains x {n_periods} periods = {res.n_transitions} "
            "transitions; explicit bins make the classification exact"
        ),
        barmode="group",
        xaxis={"title": "transition"},
        yaxis={"title": "probability"},
    )

    summary = {
        "k": float(k),
        "max_abs_error": float(np.abs(p_est - p_true).max()),
        "ergodic_l1_error": float(np.abs(ergodic_est - ergodic_true).sum()),
        "mean_persistence_true": float(np.diag(p_true).mean()),
        "mean_persistence_est": float(np.diag(p_est).mean()),
        "n_transitions": float(res.n_transitions),
    }
    return SandboxResult(
        df=df, fig=fig, summary=summary, topic="markov_chains", data=panel
    )


def learn_spatial_markov(
    *,
    side: int = 10,
    n_periods: int = 30,
    base_move: float = 0.10,
    contextual: float = 0.25,
    seed: int = 0,
) -> SandboxResult:
    """Plant neighbor-dependent mobility, then watch the spatial Markov test flag it.

    Simulates three-state dynamics on a lattice where each unit moves **toward its
    neighbors' average state** with probability ``base_move + contextual`` but away
    with only ``base_move`` — so transition probabilities genuinely depend on the
    spatial context (set ``contextual=0`` to restore homogeneity). Rey's spatial
    Markov (:func:`geometrics.analyze_spatial_markov`) conditions the transition
    matrix on the neighbors' class; its LR test should reject homogeneity and the
    upward-move probability should rise with richer neighbors.

    Requires the ``dynamics`` extra (``pip install "geometrics[dynamics]"``).

    Parameters
    ----------
    side
        Lattice side length (n = side²).
    n_periods
        Number of simulated periods.
    base_move
        Baseline per-period probability of moving a state in either direction.
    contextual
        Extra probability of moving *toward* the neighbors' average state — the
        planted spatial effect the test should detect.
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (upward/stay/downward probabilities of the middle state by neighbor
        class), ``fig``, ``summary``, ``topic`` and the simulated panel in ``data``.

    Examples
    --------
    ```python
    import geometrics as gm

    res = gm.learn_spatial_markov(contextual=0.25)
    print(res.summary["lr_p"], res.summary["contextual_gap_est"])
    ```
    """
    from geometrics.distribution_dynamics import _import_giddy, analyze_spatial_markov

    func = "learn_spatial_markov"
    _import_giddy(func)
    side = check_int("side", side, minimum=5, func=func)
    n_periods = check_int("n_periods", n_periods, minimum=5, func=func)
    base_move = check_float(
        "base_move", base_move, minimum=0.0, maximum=0.4, inclusive=False, func=func
    )
    contextual = check_float(
        "contextual", contextual, minimum=0.0, maximum=0.5, func=func
    )

    gdf = lattice_gdf(side)
    w = queen_w_from_gdf(gdf)
    ids = list(gdf["unit"])
    w_dense = dense_w(w, ids)
    n = side * side

    rng = np.random.default_rng(seed)
    states = np.empty((n, n_periods), dtype=int)
    states[:, 0] = rng.choice(3, size=n)
    p_toward = base_move + contextual
    for t in range(1, n_periods):
        current = states[:, t - 1]
        neighbor_mean = w_dense @ current
        direction = np.sign(neighbor_mean - current).astype(int)
        u = rng.random(n)
        step = np.zeros(n, dtype=int)
        has_direction = direction != 0
        step[has_direction & (u < p_toward)] = direction[has_direction & (u < p_toward)]
        away = has_direction & (u >= p_toward) & (u < p_toward + base_move)
        step[away] = -direction[away]
        # No pull (neighbors match): a symmetric random walk with the base rate.
        tie = ~has_direction
        step[tie & (u < base_move / 2)] = 1
        step[tie & (u >= base_move / 2) & (u < base_move)] = -1
        states[:, t] = np.clip(current + step, 0, 2)

    values = _states_to_values(rng, states)
    panel = pd.DataFrame(
        {
            "unit": np.repeat(ids, n_periods),
            "year": np.tile(np.arange(2000, 2000 + n_periods), n),
            "y": values.ravel(),
        }
    )
    res = analyze_spatial_markov(
        panel,
        "y",
        gdf=gdf,
        w=w,
        entity="unit",
        time="year",
        k=3,
        m=3,
        relative=False,
    )

    class_names = ("poor", "middle", "rich")
    rows = []
    for c, conditional in enumerate(res.p_conditional):
        cond = conditional.to_numpy(dtype=float)
        rows.append(
            {
                "neighbor_class": class_names[c],
                "p_up": float(cond[1, 2]),
                "p_stay": float(cond[1, 1]),
                "p_down": float(cond[1, 0]),
            }
        )
    df = pd.DataFrame(rows)
    up_probs = df["p_up"].to_numpy(dtype=float)
    pooled_up = float(res.p_global.to_numpy(dtype=float)[1, 2])

    fig = go.Figure(
        go.Bar(
            x=[f"{c} neighbors" for c in df["neighbor_class"]],
            y=df["p_up"],
            marker={"color": [color_for(i) for i in range(len(df))]},
        )
    )
    fig.add_hline(
        y=pooled_up,
        line_dash="dash",
        line_color="rgba(0,0,0,0.5)",
        annotation_text="pooled (ignoring neighbors)",
        # Left end clears the tall right-hand bar that otherwise sits under the label.
        annotation_position="top left",
    )
    apply_default_layout(
        fig,
        title="Mobility depends on the neighborhood",
        subtitle=(
            f"P(middle → rich) by neighbor class; planted context effect = "
            f"{contextual:g} (LR p = {res.lr_p:.3g})"
        ),
        xaxis={"title": ""},
        yaxis={"title": "P(move up | neighbor class)"},
    )

    summary = {
        "contextual": float(contextual),
        "base_move": float(base_move),
        "lr_stat": float(res.lr_stat),
        "lr_p": float(res.lr_p),
        "q_p": float(res.q_p),
        "up_prob_poor_nbrs": float(up_probs[0]),
        "up_prob_rich_nbrs": float(up_probs[2]),
        "contextual_gap_est": float(up_probs[2] - up_probs[0]),
    }
    return SandboxResult(
        df=df,
        fig=fig,
        summary=summary,
        topic="spatial_markov",
        data=panel,
        w_spec=res.w_spec,
    )
