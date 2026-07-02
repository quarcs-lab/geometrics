"""Shared data-generating processes for the ``learn_*`` sandboxes.

These helpers are the single home of the synthetic geographies and planted-parameter
processes used both by the sandboxes and by the test suite's known-answer fixtures
(``tests/conftest.py`` delegates here so the DGPs are exercised in one place):

- :func:`lattice_gdf` — a ``side x side`` lattice of square cells at plausible lon/lat,
  so contiguity weights and metric-CRS operations both work.
- :func:`lattice_w` — row-standardized queen/rook/knn weights on that lattice, with a
  human-readable ``w_spec`` string.
- :func:`dense_w` / :func:`solve_sar` — the aligned dense-``W`` array and the SAR
  reduced form ``y = (I - rho W)^-1 xb``.
- :func:`convergence_panel` — the planted β-convergence panel
  (``log y_it = log y_i0 + t (a - b log y_i0) + noise``).
- :func:`simulate_markov_states` / :func:`ergodic_distribution` — draws from a planted
  transition matrix and its stationary distribution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd
    from libpysal.weights import W

__all__ = [
    "lattice_gdf",
    "lattice_w",
    "queen_w_from_gdf",
    "dense_w",
    "solve_sar",
    "convergence_panel",
    "simulate_markov_states",
    "ergodic_distribution",
]


def lattice_gdf(
    side: int,
    *,
    lon0: float = 78.0,
    lat0: float = 20.0,
    step: float = 0.1,
    prefix: str = "u",
    entity: str = "unit",
) -> gpd.GeoDataFrame:
    """Return a ``side x side`` lattice of square cells with sequential entity ids.

    Cells are ``step``-degree squares anchored at (``lon0``, ``lat0``) — central India
    by default, so ``estimate_utm_crs()`` resolves for metric operations. Ids are
    zero-padded row-major (``u00 .. u63`` for ``side=8``), EPSG:4326.
    """
    import geopandas as gpd
    from shapely.geometry import box

    width = len(str(side * side - 1))
    cells = []
    ids = []
    for row in range(side):
        for col in range(side):
            x0 = lon0 + col * step
            y0 = lat0 + row * step
            cells.append(box(x0, y0, x0 + step, y0 + step))
            ids.append(f"{prefix}{row * side + col:0{width}d}")
    return gpd.GeoDataFrame({entity: ids}, geometry=cells, crs="EPSG:4326")


def lattice_w(side: int, *, method: str = "queen", k: int = 4) -> tuple[W, str]:
    """Row-standardized weights on a ``side x side`` lattice, without geometry.

    ``method`` is ``"queen"``, ``"rook"`` or ``"knn"`` (k nearest cell centers). The
    returned ``w_spec`` is a human-readable one-liner describing the graph. Ids are the
    row-major cell indices ``0 .. side*side - 1``.
    """
    from libpysal import weights

    n = side * side
    if method in ("queen", "rook"):
        w = weights.lat2W(side, side, rook=(method == "rook"), id_type="int")
        spec = f"{method} contiguity on a {side}x{side} lattice"
    elif method == "knn":
        rows, cols = np.divmod(np.arange(n), side)
        coords = np.column_stack([cols + 0.5, rows + 0.5]).astype(float)
        w = weights.KNN.from_array(coords, k=int(k))
        spec = f"{k} nearest neighbors on a {side}x{side} lattice"
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown lattice weights method {method!r}")
    w.transform = "r"
    return w, f"{spec}, row-standardized (n={n})"


def queen_w_from_gdf(gdf: gpd.GeoDataFrame, *, entity: str = "unit") -> W:
    """Row-standardized queen weights on ``gdf`` with ids from the entity column."""
    from libpysal import weights

    w = weights.Queen.from_dataframe(
        gdf, ids=list(gdf[entity]), use_index=False, silence_warnings=True
    )
    w.transform = "r"
    return w


def dense_w(w: W, ids: Sequence[Any]) -> np.ndarray:
    """Return the dense array of ``w`` reordered to match ``ids``."""
    w_dense, w_ids = w.full()
    order = [w_ids.index(i) for i in ids]
    return np.asarray(w_dense)[np.ix_(order, order)]


def solve_sar(w_dense: np.ndarray, xb: np.ndarray, rho: float) -> np.ndarray:
    """Return ``y = (I - rho W)^-1 xb`` — the SAR reduced form."""
    n = w_dense.shape[0]
    return np.linalg.solve(np.eye(n) - float(rho) * w_dense, xb)


def convergence_panel(
    ids: Sequence[Any],
    *,
    periods: Sequence[Any],
    b: float,
    a: float,
    noise_sd: float,
    seed: int,
    var: str = "gdppc",
    entity: str = "unit",
    time: str = "year",
    log_y0_mean: float = 9.0,
    log_y0_sd: float = 0.8,
) -> pd.DataFrame:
    """Long panel with planted β-convergence: ``growth = a - b log y0``.

    ``log y_it = log y_i0 + t (a - b log y_i0) + noise`` so annualized log growth
    regressed on the initial log level recovers ``-b`` exactly (up to noise), and the
    cross-sectional dispersion of ``log y`` shrinks mechanically over time. The value
    column holds strictly positive levels (``exp`` of the log path).
    """
    rng = np.random.default_rng(seed)
    n = len(ids)
    log_y0 = rng.normal(log_y0_mean, log_y0_sd, n)

    rows = []
    for t_index, period in enumerate(periods):
        drift = t_index * (a - b * log_y0)
        noise = rng.normal(0.0, noise_sd, n) if t_index else np.zeros(n)
        log_y = log_y0 + drift + noise
        rows.append(pd.DataFrame({entity: list(ids), time: period, var: np.exp(log_y)}))
    return pd.concat(rows, ignore_index=True)


def simulate_markov_states(
    rng: np.random.Generator,
    p: np.ndarray,
    *,
    n_units: int,
    n_periods: int,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate ``(n_units, n_periods)`` discrete states from transition matrix ``p``.

    ``init`` gives the initial-state probabilities (defaults to the ergodic
    distribution of ``p``, so the chain starts in steady state).
    """
    k = p.shape[0]
    if init is None:
        init = ergodic_distribution(p)
    states = np.empty((n_units, n_periods), dtype=int)
    states[:, 0] = rng.choice(k, size=n_units, p=init)
    cumulative = np.cumsum(p, axis=1)
    for t in range(1, n_periods):
        draws = rng.random(n_units)
        states[:, t] = (draws[:, None] > cumulative[states[:, t - 1]]).sum(axis=1)
    return states


def ergodic_distribution(p: np.ndarray) -> np.ndarray:
    """Return the stationary distribution of the row-stochastic matrix ``p``."""
    eigenvalues, eigenvectors = np.linalg.eig(p.T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    return pi / pi.sum()
