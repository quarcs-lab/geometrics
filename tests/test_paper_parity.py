"""Paper-parity gate: reproduce Table 1 (Model 1) of the India NTL convergence study.

The published values are the Model 1 (unconditional, no state FE) column of Table 1 in
"Regional growth, convergence, and spatial spillovers in India: A reproducible view
from outer space" (quarcs-lab/project2025s-py), as hard-coded in the repository's
``replicate_india_sdm_convergence.py`` comparison block:

    Direct:   OLS -0.020 | SDM -0.021
    Indirect: OLS      - | SDM -0.001
    Total:    OLS -0.020 | SDM -0.022

The paper's dependent variable is the annualized per-capita NTL growth 1996-2010
(horizon T = 14) regressed on the initial (log) per-capita NTL; its weights are
6-nearest-neighbor from WGS84 district centroids, row-standardized. A two-period
parity panel is built so that :func:`geometrics.convergence.growth_cross_section`
reproduces the paper's dependent variable *exactly* (to float precision), which makes
``analyze_beta_convergence`` estimate the paper's regressions verbatim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geometrics.convergence import analyze_beta_convergence, growth_cross_section
from geometrics.weights import make_weights

pytestmark = pytest.mark.network

#: Published Table 1, Model 1 point estimates (replicate script's comparison block).
PUBLISHED_OLS_TOTAL = -0.020
PUBLISHED_SDM_DIRECT = -0.021
PUBLISHED_SDM_INDIRECT = -0.001
PUBLISHED_SDM_TOTAL = -0.022

#: The paper's growth horizon: 1996 -> 2010.
HORIZON = 14.0


@pytest.fixture(scope="module")
def india():
    """The case-study inputs (downloaded once, cached by pooch)."""
    from geometrics.data import load_india

    gdf, df, _df_dict = load_india()
    return gdf, df


@pytest.fixture(scope="module")
def parity_panel(india):
    """Two-period panel whose growth cross-section IS the paper's Table 1 sample.

    Value at 1996 is the paper's ``ntl_pc_1996``; value at 2010 is constructed as
    ``ntl_pc_1996 * exp(14 * growth_ntl_pc_9610)`` so that the annualized log growth
    over the window equals the paper's dependent variable exactly, and the initial
    log level equals ``log_ntl_pc_1996`` (up to the source file's float32 storage).
    """
    _, df = india
    base = df[df["year"] == 1996][
        ["statedist", "ntl_pc_1996", "log_ntl_pc_1996", "growth_ntl_pc_9610"]
    ].reset_index(drop=True)
    init = base["ntl_pc_1996"].astype(float)
    growth = base["growth_ntl_pc_9610"].astype(float)
    panel = pd.concat(
        [
            pd.DataFrame(
                {"statedist": base["statedist"], "year": 1996, "value": init.to_numpy()}
            ),
            pd.DataFrame(
                {
                    "statedist": base["statedist"],
                    "year": 2010,
                    "value": (init * np.exp(HORIZON * growth)).to_numpy(),
                }
            ),
        ],
        ignore_index=True,
    )
    return panel, base


def test_growth_cross_section_reproduces_paper_dv(parity_panel):
    panel, base = parity_panel
    cs = growth_cross_section(panel, "value", entity="statedist", time="year")
    assert len(cs) == 520
    merged = cs.merge(base, on="statedist", validate="1:1")
    # computed growth == the paper's dependent variable to float precision
    err_growth = np.abs(merged["growth"] - merged["growth_ntl_pc_9610"]).max()
    assert err_growth < 1e-12
    # log(initial) == the paper's initial log level (float32 storage tolerance)
    err_log = np.abs(np.log(merged["initial"]) - merged["log_ntl_pc_1996"]).max()
    assert err_log < 1e-5


def test_ols_model1_matches_published_beta(parity_panel):
    panel, _ = parity_panel
    res = analyze_beta_convergence(
        panel, "value", entity="statedist", time="year", model="ols"
    )
    assert res.horizon == HORIZON
    assert res.n_obs == 520
    # Table 1, Model 1 OLS: -0.020 (matches to 3 decimals)
    assert res.beta_total == pytest.approx(PUBLISHED_OLS_TOTAL, abs=5e-4)
    assert res.beta_direct == pytest.approx(PUBLISHED_OLS_TOTAL, abs=5e-4)
    assert res.speed > 0 and np.isfinite(res.half_life)


def test_sdm_model1_matches_published_impacts(parity_panel, india):
    gdf, _ = india
    panel, _ = parity_panel
    # The paper's W6nn: 6 nearest neighbors from WGS84 centroids (crs=None keeps the
    # geographic coordinates, reproducing the paper's KNN basis), row-standardized.
    w = make_weights(gdf, method="knn", k=6, crs=None)
    res = analyze_beta_convergence(
        panel,
        "value",
        entity="statedist",
        time="year",
        model="sdm",
        gdf=gdf,
        w=w,
        n_draws=2000,
    )
    assert res.n_obs == 520
    # Table 1, Model 1 SDM point estimates within 0.001 (MC SEs differ by draw).
    assert res.beta_direct == pytest.approx(PUBLISHED_SDM_DIRECT, abs=1e-3)
    assert res.beta_indirect == pytest.approx(PUBLISHED_SDM_INDIRECT, abs=1e-3)
    assert res.beta_total == pytest.approx(PUBLISHED_SDM_TOTAL, abs=1e-3)
    # the spatial-lag parameter and fit of the replication run
    assert 0.7 < res.rho < 0.9
    assert res.aic == pytest.approx(-2291.7, abs=5.0)
    assert res.impacts is not None and "log_initial" in set(res.impacts["term"])
    assert res.w_spec is not None and "6-nearest-neighbor" in res.w_spec
