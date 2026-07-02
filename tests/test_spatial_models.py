"""Tests for the spreg suite: analyze_spatial_model / diagnostics / by_weights.

Known-answer style on the conftest SAR field (rho = 0.6, beta = (1.0, -0.5) on the
8x8 lattice's row-standardized queen weights), plus result-surface and validation
checks. Monte-Carlo draws are reduced (500) for speed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from geometrics._types import (
    SpatialDiagnosticsResult,
    SpatialModelResult,
    WeightsRobustnessResult,
)
from geometrics._validation import GeometricsWarning
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE
from geometrics.spatial_models import (
    analyze_spatial_diagnostics,
    analyze_spatial_model,
    analyze_spatial_model_by_weights,
)
from geometrics.weights import make_weights

N_DRAWS = 500

SAR_RHO = 0.6
SAR_BETA = {"x1": 1.0, "x2": -0.5}


def _lag_result(sar_field, grid_gdf, grid_w, **kwargs):
    defaults = dict(
        gdf=grid_gdf,
        w=grid_w,
        model="lag",
        entity="unit",
        time="year",
        n_draws=N_DRAWS,
    )
    defaults.update(kwargs)
    return analyze_spatial_model(sar_field, "y", ["x1", "x2"], **defaults)


# ---------------------------------------------------------------------------
# analyze_spatial_model — parameter recovery on the SAR DGP
# ---------------------------------------------------------------------------


def test_lag_recovers_rho(sar_field, grid_gdf, grid_w):
    res = _lag_result(sar_field, grid_gdf, grid_w)
    assert abs(res.rho - SAR_RHO) < 0.15
    assert np.isnan(res.lam)


def test_lag_recovers_betas_within_3_se(sar_field, grid_gdf, grid_w):
    res = _lag_result(sar_field, grid_gdf, grid_w)
    tidy = res.df.set_index("term")
    for term, beta in SAR_BETA.items():
        est, se = tidy.loc[term, "estimate"], tidy.loc[term, "se"]
        assert abs(est - beta) < 3 * se, f"{term}: {est} vs {beta} (se {se})"


def test_lag_result_surfaces(sar_field, grid_gdf, grid_w):
    res = _lag_result(sar_field, grid_gdf, grid_w)
    assert isinstance(res, SpatialModelResult)
    assert list(res.df.columns) == ["term", "estimate", "se", "z", "p"]
    # constant, x1, x2, and the spatial-lag parameter row
    assert len(res.df) == 4
    assert res.df["p"].between(0.0, 1.0).all()
    assert res.gt is not None
    assert res.impacts is not None
    assert list(res.impacts["term"]) == ["x1", "x2"]  # one row per regressor
    assert list(res.impacts.columns) == [
        "term",
        "direct",
        "se_direct",
        "indirect",
        "se_indirect",
        "total",
        "se_total",
    ]
    assert res.model == "lag"
    assert res.method == "ml"
    assert res.n_obs == 64
    assert res.n_draws == N_DRAWS
    assert res.outcome == "y"
    assert res.covariates == ("x1", "x2")
    assert res.period == 2020
    assert any("period" in note for note in res.notes)
    assert "n=64" in res.w_spec
    assert np.isfinite(res.r2)
    assert np.isfinite(res.log_likelihood)
    assert np.isfinite(res.aic)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.rho = 0.0


def test_lag_glance_and_tidy(sar_field, grid_gdf, grid_w):
    res = _lag_result(sar_field, grid_gdf, grid_w)
    assert res.tidy() is res.df
    glance = res.glance()
    assert len(glance) == 1
    assert glance.loc[0, "model"] == "lag"


def test_lag_interpret_is_association_only(sar_field, grid_gdf, grid_w):
    res = _lag_result(sar_field, grid_gdf, grid_w)
    text = res.interpret()
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)
    assert "ρ" in text


def test_lag_gm_method_runs(sar_field, grid_gdf, grid_w):
    res = _lag_result(sar_field, grid_gdf, grid_w, method="gm")
    assert res.method == "gm"
    assert abs(res.rho - SAR_RHO) < 0.25
    # GM has no likelihood: AIC / logll degrade to NaN with a note
    assert np.isnan(res.aic)
    assert np.isnan(res.log_likelihood)
    assert any("AIC" in note for note in res.notes)
    # rho row is still present in the tidy frame (labelled from name_y)
    assert len(res.df) == 4


def test_default_weights_warns(sar_field, grid_gdf):
    with pytest.warns(GeometricsWarning, match="no spatial weights supplied"):
        res = _lag_result(sar_field, grid_gdf, None, w=None)
    assert any("defaulted to" in note for note in res.notes)
    assert "queen contiguity" in res.w_spec


# ---------------------------------------------------------------------------
# analyze_spatial_model — durbin / slx / error variants
# ---------------------------------------------------------------------------


def test_durbin_total_impacts_match_coefficients(sar_field, grid_gdf, grid_w):
    res = _lag_result(sar_field, grid_gdf, grid_w, model="durbin")
    tidy = res.df.set_index("term")["estimate"]
    impacts = res.impacts.set_index("term")
    assert np.isfinite(res.rho)
    for term in ("x1", "x2"):
        expected = (tidy[term] + tidy[f"W_{term}"]) / (1.0 - res.rho)
        assert abs(impacts.loc[term, "total"] - expected) < 1e-8


def test_durbin_tidy_has_slx_terms(sar_field, grid_gdf, grid_w):
    res = _lag_result(sar_field, grid_gdf, grid_w, model="durbin")
    terms = list(res.df["term"])
    assert "W_x1" in terms and "W_x2" in terms
    assert res.model == "durbin"


def test_slx_impacts_are_analytic(sar_field, grid_gdf, grid_w):
    res = _lag_result(sar_field, grid_gdf, grid_w, model="slx")
    tidy = res.df.set_index("term")
    impacts = res.impacts.set_index("term")
    for term in ("x1", "x2"):
        assert impacts.loc[term, "direct"] == pytest.approx(
            tidy.loc[term, "estimate"], abs=1e-12
        )
        assert impacts.loc[term, "se_direct"] == pytest.approx(
            tidy.loc[term, "se"], abs=1e-12
        )
        assert impacts.loc[term, "indirect"] == pytest.approx(
            tidy.loc[f"W_{term}", "estimate"], abs=1e-12
        )
        assert impacts.loc[term, "total"] == pytest.approx(
            impacts.loc[term, "direct"] + impacts.loc[term, "indirect"], abs=1e-12
        )
    assert np.isnan(res.rho)  # no spatially lagged outcome in SLX
    assert res.n_draws == 0  # analytic impacts use no Monte-Carlo draws


def test_error_model_has_lambda_and_no_impacts(sar_field, grid_gdf, grid_w):
    res = _lag_result(sar_field, grid_gdf, grid_w, model="error")
    assert np.isfinite(res.lam)
    assert np.isnan(res.rho)
    assert res.impacts is None
    assert any("not defined" in note for note in res.notes)


def test_durbin_error_gm_raises(sar_field, grid_gdf, grid_w):
    with pytest.raises(ValueError, match="no GM estimator"):
        _lag_result(sar_field, grid_gdf, grid_w, model="durbin_error", method="gm")


def test_durbin_error_ml_runs(sar_field, grid_gdf, grid_w):
    res = _lag_result(sar_field, grid_gdf, grid_w, model="durbin_error")
    assert np.isfinite(res.lam)
    assert res.impacts is not None
    assert list(res.impacts["term"]) == ["x1", "x2"]


# ---------------------------------------------------------------------------
# analyze_spatial_model — validation order
# ---------------------------------------------------------------------------


def test_missing_column_raises_keyerror(sar_field, grid_gdf, grid_w):
    with pytest.raises(KeyError, match="nope"):
        analyze_spatial_model(
            sar_field, "y", ["nope"], gdf=grid_gdf, w=grid_w, entity="unit"
        )


def test_non_numeric_outcome_raises_typeerror(sar_field, grid_gdf, grid_w):
    bad = sar_field.assign(y=sar_field["y"].astype(str))
    with pytest.raises(TypeError, match="numeric"):
        analyze_spatial_model(bad, "y", ["x1"], gdf=grid_gdf, w=grid_w, entity="unit")


def test_unknown_model_raises_valueerror(sar_field, grid_gdf, grid_w):
    with pytest.raises(ValueError, match="unknown model"):
        _lag_result(sar_field, grid_gdf, grid_w, model="sarar")


def test_unknown_period_raises_valueerror(sar_field, grid_gdf, grid_w):
    with pytest.raises(ValueError, match="period"):
        _lag_result(sar_field, grid_gdf, grid_w, period=1999)


def test_zero_variance_covariate_raises_valueerror(sar_field, grid_gdf, grid_w):
    flat = sar_field.assign(x1=1.0)
    with pytest.raises(ValueError, match="zero variance"):
        analyze_spatial_model(
            flat, "y", ["x1", "x2"], gdf=grid_gdf, w=grid_w, entity="unit"
        )


def test_missing_roles_raise_valueerror(sar_field, grid_gdf, grid_w):
    with pytest.raises(ValueError, match="outcome"):
        analyze_spatial_model(sar_field, gdf=grid_gdf, w=grid_w, entity="unit")


# ---------------------------------------------------------------------------
# analyze_spatial_diagnostics
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diag_result(sar_field, grid_gdf, grid_w):
    return analyze_spatial_diagnostics(
        sar_field,
        "y",
        ["x1", "x2"],
        gdf=grid_gdf,
        w=grid_w,
        entity="unit",
        time="year",
    )


def test_diagnostics_frame_shape(diag_result):
    assert isinstance(diag_result, SpatialDiagnosticsResult)
    assert list(diag_result.df["test"]) == [
        "moran_residuals",
        "lm_lag",
        "lm_error",
        "robust_lm_lag",
        "robust_lm_error",
        "lm_sarma",
    ]
    assert list(diag_result.df.columns) == ["test", "statistic", "df", "p"]
    assert diag_result.df["p"].between(0.0, 1.0).all()
    assert diag_result.gt is not None


def test_diagnostics_recommend_lag_on_sar_dgp(diag_result):
    # The SAR DGP should trip robust LM lag but not robust LM error.
    by_test = diag_result.df.set_index("test")
    assert by_test.loc["robust_lm_lag", "p"] < 0.05
    assert by_test.loc["robust_lm_error", "p"] >= 0.05
    assert diag_result.recommendation in {
        "lag",
        "consider durbin (SDM) - it nests both",
    }
    assert diag_result.recommendation == "lag"
    assert diag_result.reasoning
    assert diag_result.moran_i_resid > 0
    assert diag_result.ols_model is not None
    assert diag_result.alpha == 0.05


def test_diagnostics_interpret_is_association_only(diag_result):
    text = diag_result.interpret()
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)
    assert "lag" in text


def test_diagnostics_bad_alpha_raises(sar_field, grid_gdf, grid_w):
    with pytest.raises(ValueError, match="alpha"):
        analyze_spatial_diagnostics(
            sar_field, "y", ["x1"], gdf=grid_gdf, w=grid_w, entity="unit", alpha=1.5
        )


# ---------------------------------------------------------------------------
# analyze_spatial_model_by_weights
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_weights(grid_gdf, grid_w):
    return {"queen": grid_w, "knn4": make_weights(grid_gdf, method="knn", k=4)}


@pytest.fixture(scope="module")
def robustness_result(sar_field, grid_gdf, two_weights):
    return analyze_spatial_model_by_weights(
        sar_field,
        "y",
        ["x1", "x2"],
        gdf=grid_gdf,
        weights=two_weights,
        entity="unit",
        time="year",
        n_draws=N_DRAWS,
    )


def test_by_weights_frame(robustness_result):
    res = robustness_result
    assert isinstance(res, WeightsRobustnessResult)
    assert len(res.df) == 2
    assert list(res.df["weights"]) == ["queen", "knn4"]
    for col in (
        "weights",
        "rho",
        "direct",
        "se_direct",
        "indirect",
        "se_indirect",
        "total",
        "se_total",
        "aic",
        "n_obs",
    ):
        assert col in res.df.columns
    assert res.baseline == "queen"
    assert res.focal == "x1"
    assert res.model == "durbin"
    assert (res.df["n_obs"] == 64).all()
    assert res.gt is not None


def test_by_weights_figure_has_three_facets(robustness_result):
    fig = robustness_result.fig
    assert isinstance(fig, go.Figure)
    titles = [a.text for a in fig.layout.annotations]
    for label in ("Direct", "Indirect", "Total"):
        assert label in titles
    assert fig.layout.xaxis3 is not None  # third facet exists
    # baseline dashed reference line in each facet
    dashed = [s for s in fig.layout.shapes if s.line.dash == "dash"]
    assert len(dashed) == 3
    # entity hover contract on the dot traces
    assert all(t.hovertemplate.endswith("<extra></extra>") for t in fig.data)
    assert all(t.customdata is not None for t in fig.data)


def test_by_weights_is_reproducible(
    sar_field, grid_gdf, two_weights, robustness_result
):
    again = analyze_spatial_model_by_weights(
        sar_field,
        "y",
        ["x1", "x2"],
        gdf=grid_gdf,
        weights=two_weights,
        entity="unit",
        time="year",
        n_draws=N_DRAWS,
    )
    pd.testing.assert_frame_equal(again.df, robustness_result.df)


def test_by_weights_interpret_is_association_only(robustness_result):
    text = robustness_result.interpret()
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)
    assert "queen" in text


def test_by_weights_rejects_models_without_impacts(sar_field, grid_gdf, two_weights):
    with pytest.raises(ValueError, match="impact decomposition"):
        analyze_spatial_model_by_weights(
            sar_field,
            "y",
            ["x1"],
            gdf=grid_gdf,
            weights=two_weights,
            model="ols",
            entity="unit",
        )


def test_by_weights_unknown_baseline_raises(sar_field, grid_gdf, two_weights):
    with pytest.raises(ValueError, match="baseline"):
        analyze_spatial_model_by_weights(
            sar_field,
            "y",
            ["x1"],
            gdf=grid_gdf,
            weights=two_weights,
            baseline="hexagon",
            entity="unit",
            n_draws=N_DRAWS,
        )


def test_by_weights_default_suite(sar_field, grid_gdf):
    with pytest.warns(GeometricsWarning):
        res = analyze_spatial_model_by_weights(
            sar_field,
            "y",
            ["x1", "x2"],
            gdf=grid_gdf,
            model="lag",
            entity="unit",
            time="year",
            n_draws=N_DRAWS,
        )
    assert list(res.df["weights"]) == [
        "knn4",
        "knn6",
        "knn8",
        "queen",
        "rook",
        "inv_distance",
        "inv_distance2",
    ]
    assert res.baseline == "knn4"
    assert res.df["total"].notna().all()
