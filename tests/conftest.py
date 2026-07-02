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
    import geopandas as gpd
    from shapely.geometry import box

    cells = []
    ids = []
    for row in range(GRID_SIDE):
        for col in range(GRID_SIDE):
            x0 = GRID_LON0 + col * GRID_STEP
            y0 = GRID_LAT0 + row * GRID_STEP
            cells.append(box(x0, y0, x0 + GRID_STEP, y0 + GRID_STEP))
            ids.append(f"u{row * GRID_SIDE + col:02d}")
    return gpd.GeoDataFrame({"unit": ids}, geometry=cells, crs="EPSG:4326")


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
    rng = np.random.default_rng(RNG_SEED)
    ids = list(grid_gdf["unit"])
    w_dense, w_ids = grid_w.full()
    order = [w_ids.index(i) for i in ids]
    w_dense = w_dense[np.ix_(order, order)]

    x1 = rng.normal(0.0, 1.0, GRID_N)
    x2 = rng.normal(0.0, 1.0, GRID_N)
    eps = rng.normal(0.0, 0.5, GRID_N)
    xb = SAR_BETA[0] * x1 + SAR_BETA[1] * x2 + eps
    y = np.linalg.solve(np.eye(GRID_N) - SAR_RHO * w_dense, xb)

    return pd.DataFrame({"unit": ids, "year": 2020, "x1": x1, "x2": x2, "y": y})


@pytest.fixture(scope="session")
def convergence_panel(grid_gdf):
    """Long panel with planted beta-convergence (b = 0.02) and shrinking dispersion.

    ``log y_it = log y_i0 + t * (a - b * log y_i0) + noise`` so annualized log growth
    regressed on the initial log level recovers ``-b``; the cross-sectional standard
    deviation of ``log y`` shrinks mechanically over time (sigma-convergence). The
    ``gdppc`` column is strictly positive (levels), suitable for Gini/Theil measures.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    ids = list(grid_gdf["unit"])
    log_y0 = rng.normal(9.0, 0.8, GRID_N)

    rows = []
    for t_index, year in enumerate(CONV_PERIODS):
        drift = t_index * (CONV_A - CONV_B * log_y0)
        noise = rng.normal(0.0, 0.005, GRID_N) if t_index else np.zeros(GRID_N)
        log_y = log_y0 + drift + noise
        rows.append(pd.DataFrame({"unit": ids, "year": year, "gdppc": np.exp(log_y)}))
    return pd.concat(rows, ignore_index=True)
