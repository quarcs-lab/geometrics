"""ESDA sandboxes: spatial autocorrelation, weights choice, and LISA cluster recovery.

All three simulate on a square lattice with row-standardized contiguity weights and
never need geometry — the lattice heatmaps are drawn directly from the cell grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from geometrics._theme import (
    LISA_COLORS,
    active_sequential_scale,
    apply_default_layout,
    color_for,
)
from geometrics._types import SandboxResult
from geometrics.sandbox._dgp import dense_w, lattice_w, solve_sar
from geometrics.sandbox._validate import check_float, check_int

__all__ = [
    "learn_spatial_autocorrelation",
    "learn_spatial_weights",
    "learn_lisa_clusters",
]

# The ρ sweep behind learn_spatial_autocorrelation's right-hand panel. The focal ρ is
# added to (and highlighted on) this grid.
_RHO_GRID = (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9)

# esda's Moran_Local quadrant codes -> cluster names (1=HH, 2=LH, 3=LL, 4=HL), matching
# geometrics.explore_lisa_cluster_map.
_QUADRANT_FULL = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
_LISA_ORDER = ("High-High", "Low-Low", "Low-High", "High-Low", "Not significant")


def _field_frame(field: np.ndarray, side: int, value: str = "value") -> pd.DataFrame:
    """Long frame (unit, row, col, value) for a row-major lattice field."""
    rows, cols = np.divmod(np.arange(side * side), side)
    return pd.DataFrame(
        {"unit": np.arange(side * side), "row": rows, "col": cols, value: field}
    )


def learn_spatial_autocorrelation(
    *,
    side: int = 12,
    rho: float = 0.6,
    n_sims: int = 10,
    permutations: int = 199,
    seed: int = 0,
) -> SandboxResult:
    """See what spatial autocorrelation looks like — and how Moran's I tracks it.

    Simulates fields ``y = (I - rho W)^-1 eps`` on a ``side x side`` lattice with
    row-standardized queen weights, sweeping the planted dependence ρ over a grid that
    includes the focal ``rho``. The figure pairs the focal simulated field (left) with
    the Moran's I recovered at each planted ρ (right): at ρ = 0 the statistic sits at
    its null expectation E[I] = -1/(n-1); as ρ rises, neighbors look alike and I climbs.

    Parameters
    ----------
    side
        Lattice side length (n = side²).
    rho
        The focal planted spatial dependence, |ρ| < 1. The left panel draws a field
        at this value and the sweep curve highlights it.
    n_sims
        Simulated fields per ρ (the faint markers behind the mean curve).
    permutations
        Conditional permutations behind each Moran's I pseudo p-value.
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (one row per ρ and simulation), ``fig``, ``summary``, ``topic`` and the
        focal simulated field in ``data``.

    Examples
    --------
    The knob variation is the lesson — compare no dependence with strong dependence:

    ```python
    import geometrics as gm

    gm.learn_spatial_autocorrelation(rho=0.0).fig
    gm.learn_spatial_autocorrelation(rho=0.8).fig
    ```
    """
    from esda.moran import Moran
    from plotly.subplots import make_subplots

    func = "learn_spatial_autocorrelation"
    side = check_int("side", side, minimum=4, func=func)
    rho = check_float("rho", rho, minimum=-1.0, maximum=1.0, inclusive=False, func=func)
    n_sims = check_int("n_sims", n_sims, minimum=1, func=func)
    permutations = check_int("permutations", permutations, minimum=19, func=func)

    n = side * side
    w, w_spec = lattice_w(side, method="queen")
    w_dense = dense_w(w, list(range(n)))
    rng = np.random.default_rng(seed)

    grid = sorted(set(_RHO_GRID) | {rho, 0.0})
    rows = []
    focal_field: np.ndarray | None = None
    for rho_value in grid:
        for sim in range(n_sims):
            eps = rng.normal(0.0, 1.0, n)
            y = solve_sar(w_dense, eps, rho_value)
            mi = Moran(y, w, permutations=permutations)
            rows.append(
                {
                    "rho": rho_value,
                    "sim": sim,
                    "moran_i": float(mi.I),
                    "p_sim": float(mi.p_sim),
                }
            )
            if rho_value == rho and sim == 0:
                focal_field = y
    assert focal_field is not None
    df = pd.DataFrame(rows)

    expected_i = -1.0 / (n - 1)
    means = df.groupby("rho", sort=True)["moran_i"].mean()
    focal = df[df["rho"] == rho]

    fig = make_subplots(
        cols=2,
        rows=1,
        column_widths=[0.42, 0.58],
        subplot_titles=(f"One simulated field (ρ = {rho:g})", "Moran's I vs planted ρ"),
        horizontal_spacing=0.1,
    )
    fig.add_trace(
        go.Heatmap(
            z=focal_field.reshape(side, side),
            colorscale=active_sequential_scale(),
            showscale=False,
            hovertemplate="row %{y}, col %{x}<br>y = %{z:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["rho"],
            y=df["moran_i"],
            mode="markers",
            marker={"color": "rgba(120,120,120,0.35)", "size": 6},
            name="simulations",
            hovertemplate="ρ = %{x:g}<br>I = %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=list(means.index),
            y=list(means.to_numpy()),
            mode="lines+markers",
            line={"color": color_for(0), "width": 3},
            name="mean Moran's I",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=expected_i,
        line_dash="dash",
        line_color="rgba(0,0,0,0.5)",
        annotation_text="E[I] under no dependence",
        row=1,
        col=2,
    )
    fig.add_vline(x=rho, line_dash="dot", line_color=color_for(1), row=1, col=2)
    fig.update_yaxes(autorange="reversed", visible=False, row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="planted ρ", row=1, col=2)
    fig.update_yaxes(title_text="Moran's I", row=1, col=2)
    apply_default_layout(
        fig,
        title="Spatial autocorrelation, planted and recovered",
        subtitle=w_spec,
        showlegend=False,
    )

    summary = {
        "rho": float(rho),
        "moran_focal": float(focal["moran_i"].mean()),
        "moran_at_zero": float(df.loc[df["rho"] == 0.0, "moran_i"].mean()),
        "expected_i": float(expected_i),
        "share_significant_focal": float((focal["p_sim"] < 0.05).mean()),
        "n": float(n),
    }
    return SandboxResult(
        df=df,
        fig=fig,
        summary=summary,
        topic="spatial_autocorrelation",
        data=_field_frame(focal_field, side),
        w_spec=w_spec,
    )


def learn_spatial_weights(
    *,
    side: int = 12,
    rho: float = 0.6,
    k: int = 4,
    permutations: int = 199,
    seed: int = 0,
) -> SandboxResult:
    """See how the choice of spatial weights changes what "neighbors" means.

    Simulates one field with dependence ``rho`` under **queen** contiguity (the true
    graph), then re-tests the same field under queen, rook and k-nearest-neighbor
    weights. All three detect the clustering, but the statistic shifts with the graph —
    the substantive conclusion should not hinge on one W, which is why
    :func:`geometrics.analyze_spatial_model_by_weights` exists.

    Parameters
    ----------
    side
        Lattice side length (n = side²).
    rho
        Planted spatial dependence under the queen graph, |ρ| < 1.
    k
        Neighbors for the k-nearest-neighbor variant.
    permutations
        Conditional permutations behind each pseudo p-value.
    seed
        Random seed.

    Returns
    -------
    SandboxResult
        ``df`` (one row per weights choice), ``fig``, ``summary``, ``topic`` and the
        simulated field in ``data``.

    Examples
    --------
    ```python
    import geometrics as gm

    res = gm.learn_spatial_weights(rho=0.6, k=8)
    res.df
    ```
    """
    from esda.moran import Moran

    func = "learn_spatial_weights"
    side = check_int("side", side, minimum=4, func=func)
    rho = check_float("rho", rho, minimum=-1.0, maximum=1.0, inclusive=False, func=func)
    n = side * side
    k = check_int("k", k, minimum=1, func=func)
    if k >= n:
        raise ValueError(f"{func}: k must be < n = {n}, got {k}")
    permutations = check_int("permutations", permutations, minimum=19, func=func)

    w_true, w_spec = lattice_w(side, method="queen")
    w_dense = dense_w(w_true, list(range(n)))
    rng = np.random.default_rng(seed)
    y = solve_sar(w_dense, rng.normal(0.0, 1.0, n), rho)

    specs = [("queen", "queen"), ("rook", "rook"), (f"knn (k={k})", "knn")]
    rows = []
    for label, method in specs:
        w_alt, _ = lattice_w(side, method=method, k=k)
        mi = Moran(y, w_alt, permutations=permutations)
        rows.append(
            {
                "weights": label,
                "moran_i": float(mi.I),
                "p_sim": float(mi.p_sim),
                "mean_neighbors": float(np.mean(list(w_alt.cardinalities.values()))),
            }
        )
    df = pd.DataFrame(rows)
    moran_vals = df["moran_i"].to_numpy(dtype=float)
    neighbor_means = df["mean_neighbors"].to_numpy(dtype=float)

    fig = go.Figure(
        go.Bar(
            x=df["weights"],
            y=df["moran_i"],
            marker={"color": [color_for(i) for i in range(len(df))]},
            text=[f"~{m:.1f} neighbors" for m in neighbor_means],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=float(moran_vals[0]),
        line_dash="dash",
        line_color="rgba(0,0,0,0.5)",
        annotation_text="queen (the DGP's graph)",
    )
    apply_default_layout(
        fig,
        title="One field, three definitions of 'neighbor'",
        subtitle=f"field simulated with ρ = {rho:g} under queen contiguity",
        xaxis={"title": ""},
        yaxis={"title": "Moran's I"},
    )

    summary = {
        "rho": float(rho),
        "moran_queen": float(moran_vals[0]),
        "moran_rook": float(moran_vals[1]),
        "moran_knn": float(moran_vals[2]),
        "mean_neighbors_queen": float(neighbor_means[0]),
        "mean_neighbors_rook": float(neighbor_means[1]),
        "mean_neighbors_knn": float(neighbor_means[2]),
    }
    return SandboxResult(
        df=df,
        fig=fig,
        summary=summary,
        topic="spatial_weights",
        data=_field_frame(y, side),
        w_spec=w_spec,
    )


def learn_lisa_clusters(
    *,
    side: int = 12,
    block: int = 3,
    shift: float = 2.0,
    alpha: float = 0.05,
    permutations: int = 999,
    seed: int = 0,
) -> SandboxResult:
    """Plant hot and cold spots, then watch LISA find them (and sometimes cry wolf).

    Draws iid noise on a ``side x side`` lattice, shifts a ``block x block`` **hot**
    block up by ``shift`` and a **cold** block down by the same amount, and runs local
    Moran (LISA) with significance masking at ``alpha``. The map shows the recovered
    High-High / Low-Low clusters with the planted blocks outlined; the summary reports
    the hit rates and the false-positive share — a reminder that with hundreds of local
    tests, some cells are flagged by chance alone.

    Parameters
    ----------
    side
        Lattice side length (n = side²); must fit two disjoint blocks.
    block
        Side length of each planted block.
    shift
        Size of the planted level shift (in standard deviations of the noise).
    alpha
        Significance level masking the cluster labels (``p_sim < alpha``).
    permutations
        Conditional permutations behind the local pseudo p-values.
    seed
        Random seed (drives both the noise and the permutations).

    Returns
    -------
    SandboxResult
        ``df`` (one row per cell with planted and detected labels), ``fig``,
        ``summary``, ``topic`` and the raw field in ``data``.

    Examples
    --------
    ```python
    import geometrics as gm

    res = gm.learn_lisa_clusters(shift=2.0)
    print(res.summary["sensitivity_hot"], res.summary["false_positive_rate"])
    ```
    """
    from esda.moran import Moran_Local

    func = "learn_lisa_clusters"
    side = check_int("side", side, minimum=6, func=func)
    block = check_int("block", block, minimum=1, func=func)
    if 2 * block + 3 > side:
        raise ValueError(
            f"{func}: two {block}x{block} blocks do not fit on a {side}x{side} "
            "lattice with a margin — reduce block or increase side"
        )
    shift = check_float("shift", shift, minimum=0.0, func=func)
    alpha = check_float(
        "alpha", alpha, minimum=0.0, maximum=1.0, inclusive=False, func=func
    )
    permutations = check_int("permutations", permutations, minimum=99, func=func)

    n = side * side
    rng = np.random.default_rng(seed)
    field = rng.normal(0.0, 1.0, (side, side))
    planted = np.full((side, side), "none", dtype=object)
    hot = (slice(1, 1 + block), slice(1, 1 + block))
    cold = (slice(side - 1 - block, side - 1), slice(side - 1 - block, side - 1))
    field[hot] += shift
    planted[hot] = "hot"
    field[cold] -= shift
    planted[cold] = "cold"

    w, w_spec = lattice_w(side, method="queen")
    y = field.ravel()
    # np.errstate: esda's z_sim is a harmless 0/0 for degenerate cells.
    with np.errstate(divide="ignore", invalid="ignore"):
        lisa = Moran_Local(y, w, permutations=permutations, seed=seed)
    p_sim = np.asarray(lisa.p_sim, dtype=float)
    cluster = np.array(
        [
            _QUADRANT_FULL[int(q)] if p < alpha else "Not significant"
            for q, p in zip(np.asarray(lisa.q, dtype=int), p_sim, strict=True)
        ],
        dtype=object,
    )

    rows, cols = np.divmod(np.arange(n), side)
    df = pd.DataFrame(
        {
            "unit": np.arange(n),
            "row": rows,
            "col": cols,
            "value": y,
            "planted": planted.ravel(),
            "local_i": np.asarray(lisa.Is, dtype=float),
            "p_sim": p_sim,
            "cluster": cluster,
        }
    )

    flat_planted = planted.ravel()
    hot_mask = flat_planted == "hot"
    cold_mask = flat_planted == "cold"
    none_mask = flat_planted == "none"
    sensitivity_hot = float((cluster[hot_mask] == "High-High").mean())
    sensitivity_cold = float((cluster[cold_mask] == "Low-Low").mean())
    false_positive_rate = float((cluster[none_mask] != "Not significant").mean())

    codes = np.array([_LISA_ORDER.index(c) for c in cluster]).reshape(side, side)
    # A discrete colorscale: each category occupies an equal slice of [0, 1].
    n_cat = len(_LISA_ORDER)
    colorscale = []
    for i, label in enumerate(_LISA_ORDER):
        colorscale.append([i / n_cat, LISA_COLORS[label]])
        colorscale.append([(i + 1) / n_cat, LISA_COLORS[label]])
    fig = go.Figure(
        go.Heatmap(
            z=codes,
            zmin=-0.5,
            zmax=n_cat - 0.5,
            colorscale=colorscale,
            showscale=False,
            customdata=np.stack(
                [cluster.reshape(side, side), planted, p_sim.reshape(side, side)],
                axis=-1,
            ),
            hovertemplate=(
                "row %{y}, col %{x}<br>detected: %{customdata[0]}"
                "<br>planted: %{customdata[1]}<br>p_sim = %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        )
    )
    for block_slice, label in ((hot, "planted hot"), (cold, "planted cold")):
        fig.add_shape(
            type="rect",
            x0=block_slice[1].start - 0.5,
            x1=block_slice[1].stop - 0.5,
            y0=block_slice[0].start - 0.5,
            y1=block_slice[0].stop - 0.5,
            line={"color": "black", "width": 3},
        )
        fig.add_annotation(
            x=(block_slice[1].start + block_slice[1].stop) / 2 - 0.5,
            y=block_slice[0].start - 0.9,
            text=label,
            showarrow=False,
            font={"size": 12},
        )
    fig.update_yaxes(autorange="reversed", visible=False)
    fig.update_xaxes(visible=False)
    apply_default_layout(
        fig,
        title="Planted clusters, recovered by LISA",
        subtitle=f"{w_spec}; p < {alpha:g} on {permutations} permutations",
    )

    summary = {
        "shift": float(shift),
        "alpha": float(alpha),
        "sensitivity_hot": sensitivity_hot,
        "sensitivity_cold": sensitivity_cold,
        "false_positive_rate": false_positive_rate,
        "n_planted": float(hot_mask.sum() + cold_mask.sum()),
        "n_flagged": float((cluster != "Not significant").sum()),
    }
    notes = (
        "With n local tests, a share of about alpha of the un-planted cells is "
        "expected to be flagged by chance (and spatial smearing at block edges adds "
        "more) — LISA maps locate candidates, they do not certify clusters.",
    )
    return SandboxResult(
        df=df,
        fig=fig,
        summary=summary,
        topic="local_moran",
        data=df[["unit", "row", "col", "value"]].copy(),
        w_spec=w_spec,
        notes=notes,
    )
