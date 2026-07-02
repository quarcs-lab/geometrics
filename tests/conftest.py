"""Shared synthetic fixtures: a lattice geography and planted-parameter panels.

The fixtures build small, fully deterministic datasets whose spatial and dynamic
properties are known in closed form, so tests can assert parameter recovery rather
than just shapes:

- ``grid_gdf`` — an 8x8 lattice of square cells placed at plausible lon/lat, so both
  contiguity weights and metric-CRS operations (``estimate_utm_crs``) work.
- ``sar_field`` — a cross-section generated from the SAR data-generating process
  ``y = (I - rho W)^-1 (X beta + eps)`` with rho = 0.6 on the lattice's
  row-standardized queen weights: Moran's I is known-positive and spreg estimators
  should recover ``rho`` and ``beta``.
- ``convergence_panel`` — a 6-period panel with planted beta-convergence
  (``growth = a - b log y0``, b = 0.02) and shrinking cross-sectional dispersion,
  so beta and sigma convergence are both detectable by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

RNG_SEED = 20260702

GRID_SIDE = 8
GRID_N = GRID_SIDE * GRID_SIDE
# 0.1-degree cells anchored in central India so estimate_utm_crs() resolves.
GRID_LON0, GRID_LAT0, GRID_STEP = 78.0, 20.0, 0.1

SAR_RHO = 0.6
SAR_BETA = (1.0, -0.5)

CONV_B = 0.02
CONV_A = 0.05
CONV_PERIODS = (2000, 2001, 2002, 2003, 2004, 2005)


@pytest.fixture(scope="session")
def grid_gdf():
    """8x8 lattice of square cells with entity ids u00..u63 (EPSG:4326)."""
    from geometrics.sandbox._dgp import lattice_gdf

    return lattice_gdf(
        GRID_SIDE, lon0=GRID_LON0, lat0=GRID_LAT0, step=GRID_STEP, prefix="u"
    )


@pytest.fixture(scope="session")
def grid_w(grid_gdf):
    """Row-standardized queen weights on the lattice, ids = the unit column."""
    from libpysal import weights

    w = weights.Queen.from_dataframe(
        grid_gdf, ids=list(grid_gdf["unit"]), use_index=False, silence_warnings=True
    )
    w.transform = "r"
    return w


@pytest.fixture(scope="session")
def sar_field(grid_gdf, grid_w):
    """Cross-section from the SAR DGP: y = (I - rho W)^-1 (X beta + eps), rho=0.6.

    Returns a long-form frame with one period (year 2020) and columns
    ``unit, year, x1, x2, y`` in the lattice's row order.
    """
    from geometrics.sandbox._dgp import dense_w, solve_sar

    rng = np.random.default_rng(RNG_SEED)
    ids = list(grid_gdf["unit"])
    w_dense = dense_w(grid_w, ids)

    x1 = rng.normal(0.0, 1.0, GRID_N)
    x2 = rng.normal(0.0, 1.0, GRID_N)
    eps = rng.normal(0.0, 0.5, GRID_N)
    xb = SAR_BETA[0] * x1 + SAR_BETA[1] * x2 + eps
    y = solve_sar(w_dense, xb, SAR_RHO)

    return pd.DataFrame({"unit": ids, "year": 2020, "x1": x1, "x2": x2, "y": y})


@pytest.fixture(scope="session")
def convergence_panel(grid_gdf):
    """Long panel with planted beta-convergence (b = 0.02) and shrinking dispersion.

    ``log y_it = log y_i0 + t * (a - b * log y_i0) + noise`` so annualized log growth
    regressed on the initial log level recovers ``-b``; the cross-sectional standard
    deviation of ``log y`` shrinks mechanically over time (sigma-convergence). The
    ``gdppc`` column is strictly positive (levels), suitable for Gini/Theil measures.
    """
    from geometrics.sandbox._dgp import convergence_panel as make_convergence_panel

    return make_convergence_panel(
        list(grid_gdf["unit"]),
        periods=CONV_PERIODS,
        b=CONV_B,
        a=CONV_A,
        noise_sd=0.005,
        seed=RNG_SEED + 1,
    )
