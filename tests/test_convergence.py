"""Tests for ``growth_cross_section``, ``analyze_beta_convergence`` and ``analyze_sigma_convergence``."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from geometrics._types import BetaConvergenceResult, SigmaConvergenceResult
from geometrics._validation import GeometricsWarning
from geometrics.convergence import (
    analyze_beta_convergence,
    analyze_sigma_convergence,
    growth_cross_section,
)
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

# The conftest convergence_panel plants log y_it = log y_i0 + t*(a - b*log y_i0) + eps
# with b = 0.02, a = 0.05 over years 2000..2005, so the annualized log growth regressed
# on the initial log level recovers a slope of exactly -b up to the (tiny) noise term:
# growth_i = a - b*log y_i0 + eps_i,2005 / 5 with eps sd = 0.005 -> slope error < 1e-3.
PLANTED_B = 0.02
PANEL_T = 5.0  # 2005 - 2000


@pytest.fixture(scope="module")
def sar_growth_panel(grid_gdf, grid_w):
    """Two-period panel whose growth field follows a SAR DGP with rho = 0.5.

    growth = (I - 0.5 W)^-1 (a - b*log y0 + eps) is *exactly* the spatial-lag model
    analyze_beta_convergence(model='sar') estimates, so ML should recover rho.
    """
    rng = np.random.default_rng(7)
    ids = list(grid_gdf["unit"])
    w_dense, w_ids = grid_w.full()
    order = [w_ids.index(i) for i in ids]
    w_dense = w_dense[np.ix_(order, order)]
    n = len(ids)
    log_y0 = rng.normal(9.0, 0.8, n)
    eps = rng.normal(0.0, 0.002, n)
    growth = np.linalg.solve(np.eye(n) - 0.5 * w_dense, 0.05 - 0.02 * log_y0 + eps)
    horizon = 10.0
    return pd.concat(
        [
            pd.DataFrame({"unit": ids, "year": 2000, "y": np.exp(log_y0)}),
            pd.DataFrame(
                {"unit": ids, "year": 2010, "y": np.exp(log_y0 + horizon * growth)}
            ),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# growth_cross_section
# ---------------------------------------------------------------------------


def test_growth_cross_section_columns_and_formula(convergence_panel):
    cs = growth_cross_section(convergence_panel, "gdppc", entity="unit", time="year")
    assert list(cs.columns) == ["unit", "initial", "final", "growth"]
    assert len(cs) == 64
    # growth is exactly (log(final) - log(initial)) / T over the common window
    first = convergence_panel[convergence_panel["year"] == 2000].set_index("unit")
    last = convergence_panel[convergence_panel["year"] == 2005].set_index("unit")
    merged = cs.set_index("unit")
    expected = (
        np.log(last["gdppc"].reindex(merged.index))
        - np.log(first["gdppc"].reindex(merged.index))
    ) / PANEL_T
    assert np.allclose(merged["growth"], expected, atol=1e-12)
    assert np.allclose(merged["initial"], first["gdppc"].reindex(merged.index))
    # panel attrs are re-declared on the output
    assert cs.attrs["geometrics_panel"]["entity"] == "unit"


def test_growth_cross_section_annualize_false(convergence_panel):
    ann = growth_cross_section(convergence_panel, "gdppc", entity="unit", time="year")
    tot = growth_cross_section(
        convergence_panel, "gdppc", entity="unit", time="year", annualize=False
    )
    assert np.allclose(tot["growth"], ann["growth"] * PANEL_T)


def test_growth_cross_section_window_endpoints(convergence_panel):
    cs = growth_cross_section(
        convergence_panel, "gdppc", entity="unit", time="year", start=2000, end=2002
    )
    at_2002 = convergence_panel[convergence_panel["year"] == 2002].set_index("unit")
    assert np.allclose(
        cs.set_index("unit")["final"], at_2002["gdppc"].reindex(cs["unit"])
    )


def test_growth_cross_section_controls_at_initial_period(convergence_panel):
    df = convergence_panel.copy()
    df["ctrl"] = df["year"].astype(float)  # time-varying on purpose
    cs = growth_cross_section(df, "gdppc", ["ctrl"], entity="unit", time="year")
    assert (cs["ctrl"] == 2000.0).all()  # initial-period value carried


def test_growth_cross_section_validation(convergence_panel):
    with pytest.raises(KeyError, match="nope"):
        growth_cross_section(convergence_panel, "nope", entity="unit", time="year")
    df = convergence_panel.copy()
    df["txt"] = "a"
    with pytest.raises(TypeError, match="numeric"):
        growth_cross_section(df, "txt", entity="unit", time="year")
    with pytest.raises(ValueError, match="after start"):
        growth_cross_section(
            convergence_panel, "gdppc", entity="unit", time="year", start=2005, end=2000
        )
    with pytest.raises(ValueError, match="reserved"):
        growth_cross_section(
            df.assign(growth=1.0), "gdppc", ["growth"], entity="unit", time="year"
        )


def test_growth_cross_section_nonpositive_values_raise(convergence_panel):
    df = convergence_panel.copy()
    df.loc[(df["unit"] == "u00") & (df["year"] == 2000), "gdppc"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        growth_cross_section(df, "gdppc", entity="unit", time="year")


def test_growth_cross_section_duplicates_warn(convergence_panel):
    dirty = pd.concat(
        [convergence_panel.iloc[[0]], convergence_panel], ignore_index=True
    )
    with pytest.warns(GeometricsWarning, match="duplicate"):
        cs = growth_cross_section(dirty, "gdppc", entity="unit", time="year")
    assert len(cs) == 64


# ---------------------------------------------------------------------------
# analyze_beta_convergence — OLS on the planted panel
# ---------------------------------------------------------------------------


def test_beta_recovers_planted_slope(convergence_panel):
    res = analyze_beta_convergence(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    assert isinstance(res, BetaConvergenceResult)
    assert -0.025 < res.beta_total < -0.015
    # the DGP implies a slope of exactly -b up to noise sd(eps)/T ~ 1e-3
    assert res.beta_total == pytest.approx(-PLANTED_B, abs=3e-3)
    # speed / half-life derive from the total slope (Barro-Sala-i-Martin mapping)
    assert math.isfinite(res.speed) and res.speed > 0
    assert res.speed == pytest.approx(
        -math.log1p(res.beta_total * res.horizon) / res.horizon
    )
    assert math.isfinite(res.half_life)
    assert res.half_life == pytest.approx(math.log(2.0) / res.speed)


def test_beta_ols_result_surface(convergence_panel):
    res = analyze_beta_convergence(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    assert res.model == "ols"
    assert res.beta_direct == res.beta_total
    assert math.isnan(res.beta_indirect) and math.isnan(res.se_indirect)
    assert math.isnan(res.rho) and math.isnan(res.lam)
    assert res.impacts is None
    assert res.n_draws == 0
    assert res.w_spec is None
    assert res.fig_map is None
    assert res.fig_conditional is None
    assert res.horizon == PANEL_T
    assert res.n_obs == 64
    assert len(res.models) == 1
    assert list(res.df.columns) == ["unit", "initial", "final", "growth"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.beta_total = 0.0  # type: ignore[misc]


def test_beta_fig_and_summary(convergence_panel):
    res = analyze_beta_convergence(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    assert isinstance(res.fig, go.Figure)
    markers = [t for t in res.fig.data if t.mode == "markers"]
    assert markers and markers[0].hovertemplate.endswith("<extra></extra>")
    assert list(res.summary.columns) == ["metric", "ols"]
    assert "total" in set(res.summary["metric"])
    total = float(res.summary.set_index("metric").loc["total", "ols"])
    assert total == pytest.approx(res.beta_total)
    assert type(res.gt).__name__ == "GT"
    assert res.tidy() is res.summary
    assert res.glance().shape[0] == 1


def test_beta_fig_map_with_gdf(convergence_panel, grid_gdf):
    res = analyze_beta_convergence(
        convergence_panel, "gdppc", entity="unit", time="year", gdf=grid_gdf
    )
    assert isinstance(res.fig_map, go.Figure)
    assert len(res.fig_map.data) >= 1
    assert len(res.df) == 64  # all lattice units matched


def test_beta_conditional_controls(convergence_panel):
    rng = np.random.default_rng(3)
    df = convergence_panel.copy()
    ctrl = dict(zip(df["unit"].unique(), rng.normal(size=64), strict=True))
    df["z"] = df["unit"].map(ctrl)
    res = analyze_beta_convergence(df, "gdppc", ["z"], entity="unit", time="year")
    assert res.controls == ("z",)
    assert isinstance(res.fig_conditional, go.Figure)
    assert "z" in res.df.columns


def test_beta_fixed_effects_dummies(convergence_panel):
    df = convergence_panel.copy()
    df["zone"] = np.where(df["unit"].str[1:].astype(int) < 32, "north", "south")
    res = analyze_beta_convergence(
        df, "gdppc", entity="unit", time="year", fixed_effects="zone"
    )
    assert isinstance(res.fig_conditional, go.Figure)  # FE are partialled out too
    assert "zone" in res.df.columns


# ---------------------------------------------------------------------------
# analyze_beta_convergence — spatial models
# ---------------------------------------------------------------------------


def test_beta_sar_recovers_spatial_lag(sar_growth_panel, grid_gdf, grid_w):
    res = analyze_beta_convergence(
        sar_growth_panel,
        "y",
        entity="unit",
        time="year",
        model="sar",
        gdf=grid_gdf,
        w=grid_w,
        n_draws=400,
        seed=1,
    )
    assert res.model == "sar"
    assert math.isfinite(res.rho)
    assert 0.2 < res.rho < 0.8  # planted rho = 0.5
    assert res.beta_direct == pytest.approx(-0.02, abs=5e-3)
    # simple-method impacts: direct = b, total = b / (1 - rho)
    assert res.beta_total == pytest.approx(res.beta_direct / (1.0 - res.rho))
    assert res.impacts is not None
    assert set(res.impacts["term"]) == {"log_initial"}
    assert res.n_draws == 400
    assert res.w_spec is not None
    assert len(res.models) == 2
    assert list(res.summary.columns) == ["metric", "ols", "sar"]
    assert math.isfinite(res.se_total) and res.se_total > 0


def test_beta_sdm_and_slx_and_sem_run(sar_growth_panel, grid_gdf, grid_w):
    for model in ("sdm", "slx", "sem"):
        res = analyze_beta_convergence(
            sar_growth_panel,
            "y",
            entity="unit",
            time="year",
            model=model,
            gdf=grid_gdf,
            w=grid_w,
            n_draws=200,
            seed=1,
        )
        assert res.model == model
        assert res.impacts is not None
        assert math.isfinite(res.beta_total)
        if model == "sdm":
            assert math.isfinite(res.rho) and math.isfinite(res.beta_indirect)
        if model == "sem":
            assert math.isfinite(res.lam) and math.isnan(res.rho)
        if model == "slx":
            assert math.isfinite(res.beta_indirect)
            assert res.n_draws == 0


def test_beta_default_weights_warns(sar_growth_panel, grid_gdf):
    with pytest.warns(GeometricsWarning, match="no spatial weights"):
        res = analyze_beta_convergence(
            sar_growth_panel,
            "y",
            entity="unit",
            time="year",
            model="sar",
            gdf=grid_gdf,
            n_draws=200,
        )
    assert res.w_spec is not None
    assert any("defaulted" in n for n in res.notes)


# ---------------------------------------------------------------------------
# analyze_beta_convergence — validation
# ---------------------------------------------------------------------------


def test_beta_validation_errors(convergence_panel, grid_gdf):
    with pytest.raises(ValueError, match="needs entity geometry"):
        analyze_beta_convergence(
            convergence_panel, "gdppc", entity="unit", time="year", model="sar"
        )
    with pytest.raises(ValueError, match="unknown model"):
        analyze_beta_convergence(
            convergence_panel, "gdppc", entity="unit", time="year", model="gwr"
        )
    with pytest.raises(ValueError, match="vcov"):
        analyze_beta_convergence(
            convergence_panel, "gdppc", entity="unit", time="year", vcov="cluster"
        )
    with pytest.raises(KeyError, match="nope"):
        analyze_beta_convergence(convergence_panel, "nope", entity="unit", time="year")
    df = convergence_panel.copy()
    df["txt"] = "x"
    with pytest.raises(TypeError, match="numeric"):
        analyze_beta_convergence(df, "txt", entity="unit", time="year")


def test_beta_too_few_units_raises(convergence_panel):
    small = convergence_panel[convergence_panel["unit"].isin(["u00", "u01", "u02"])]
    with pytest.raises(ValueError, match="need at least"):
        analyze_beta_convergence(small, "gdppc", entity="unit", time="year")


def test_beta_zero_variance_initial_raises():
    df = pd.concat(
        [
            pd.DataFrame(
                {"unit": [f"r{i}" for i in range(12)], "year": 2000, "y": 5.0}
            ),
            pd.DataFrame(
                {
                    "unit": [f"r{i}" for i in range(12)],
                    "year": 2010,
                    "y": np.linspace(5.0, 9.0, 12),
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="zero variance"):
        analyze_beta_convergence(df, "y", entity="unit", time="year")


def test_beta_interpret_is_association_only(convergence_panel):
    res = analyze_beta_convergence(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    text = res.interpret()
    assert text.endswith(_ASSOC_NOTE)
    assert "causes" not in text
    assert "effect of" not in text
    assert "convergence" in text


# ---------------------------------------------------------------------------
# analyze_sigma_convergence
# ---------------------------------------------------------------------------


def test_sigma_detects_planted_convergence(convergence_panel):
    res = analyze_sigma_convergence(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    assert isinstance(res, SigmaConvergenceResult)
    # dispersion of log gdppc shrinks mechanically: sd_t = 0.8 * (1 - 0.02 t)
    assert res.std_slope < 0
    assert res.std_pvalue < 0.05
    assert res.gini_slope < 0
    assert res.cv_slope < 0
    assert bool(res.summary["converging"].all())
    assert res.n_units == 64
    assert res.n_periods == 6
    # the planted log-sd contracts ~2% per period
    assert res.std_slope == pytest.approx(-0.0204, abs=5e-3)


def test_sigma_result_surface(convergence_panel):
    res = analyze_sigma_convergence(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    assert list(res.df.columns) == ["year", "n_units", "mean", "std", "gini", "cv"]
    assert len(res.df) == 6
    assert isinstance(res.fig, go.Figure)
    assert len(res.fig.data) >= 2  # std + gini series (plus trend overlays)
    assert type(res.gt).__name__ == "GT"
    assert res.models  # fitted statsmodels trend models
    assert list(res.summary["measure"]) == ["std", "gini", "cv"]
    assert res.tidy() is res.summary
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.std_slope = 0.0  # type: ignore[misc]


def test_sigma_window_restriction(convergence_panel):
    res = analyze_sigma_convergence(
        convergence_panel, "gdppc", entity="unit", time="year", start=2001, end=2004
    )
    assert res.n_periods == 4
    assert res.df["year"].min() == 2001.0


def test_sigma_unbalanced_raises(convergence_panel):
    unbalanced = convergence_panel.drop(convergence_panel.index[0])
    with pytest.raises(ValueError, match="not balanced"):
        analyze_sigma_convergence(unbalanced, "gdppc", entity="unit", time="year")


def test_sigma_nonpositive_raises(convergence_panel):
    df = convergence_panel.copy()
    df.loc[df.index[0], "gdppc"] = -1.0
    with pytest.raises(ValueError, match="non-positive"):
        analyze_sigma_convergence(df, "gdppc", entity="unit", time="year")


def test_sigma_too_few_periods_raises(convergence_panel):
    short = convergence_panel[convergence_panel["year"] <= 2001]
    with pytest.raises(ValueError, match="periods"):
        analyze_sigma_convergence(short, "gdppc", entity="unit", time="year")


def test_sigma_duplicates_warn_and_note(convergence_panel):
    dirty = pd.concat(
        [convergence_panel.iloc[[0]], convergence_panel], ignore_index=True
    )
    with pytest.warns(GeometricsWarning, match="duplicate"):
        res = analyze_sigma_convergence(dirty, "gdppc", entity="unit", time="year")
    assert any("duplicate" in n for n in res.notes)


def test_sigma_validation_errors(convergence_panel):
    with pytest.raises(KeyError, match="nope"):
        analyze_sigma_convergence(convergence_panel, "nope", entity="unit", time="year")
    df = convergence_panel.copy()
    df["txt"] = "x"
    with pytest.raises(TypeError, match="numeric"):
        analyze_sigma_convergence(df, "txt", entity="unit", time="year")
    with pytest.raises(ValueError, match="vcov"):
        analyze_sigma_convergence(
            convergence_panel, "gdppc", entity="unit", time="year", vcov="cluster"
        )


def test_sigma_interpret_is_association_only(convergence_panel):
    res = analyze_sigma_convergence(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    text = res.interpret()
    assert text.endswith(_ASSOC_NOTE)
    assert "causes" not in text
    assert "effect of" not in text
    assert "σ-convergence" in text
