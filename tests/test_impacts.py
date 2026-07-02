"""Tests for the spreg impact machinery (geometrics._impacts).

Ground truth on a small 5x5 rook lattice with known parameters, plus a fitted SDM on
the session ``sar_field`` fixture (SAR DGP, rho=0.6, beta=(1.0, -0.5)).

spreg 1.9.0 layout observed here (differs from the 1.8.5 the paper notebooks pinned,
which located terms positionally): for ``ML_Lag(slx_lags=1, name_x=["x1","x2"],
name_y="y")`` the fitted ``name_x`` is
``['CONSTANT', 'x1', 'x2', 'W_x1', 'W_x2', 'W_y']`` — the spatially lagged regressors
get the ``'W_'`` prefix and rho itself is labelled ``'W_<name_y>'`` as the LAST entry,
so ``len(name_x) == len(betas)`` and ``model.vm`` is the full covariance INCLUDING the
rho row/column.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from geometrics._impacts import (
    adi,
    analytic_slx_impacts,
    full_rank_lag_mask,
    locate_terms,
    mc_impacts,
    stars,
)


# --------------------------------------------------------------------- fixtures ---
@pytest.fixture(scope="module")
def rook_w():
    """Row-standardized 5x5 rook lattice weights."""
    from libpysal.weights import lat2W

    w = lat2W(5, 5, rook=True)
    w.transform = "r"
    return w


@pytest.fixture(scope="module")
def rook_dense(rook_w):
    return rook_w.full()[0]


@pytest.fixture(scope="module")
def fitted_sdm(sar_field, grid_w):
    """ML_Lag SDM (slx_lags=1) on the SAR-field fixture; returns (model, w_dense)."""
    import spreg

    y = sar_field[["y"]].to_numpy(dtype=float)
    x = sar_field[["x1", "x2"]].to_numpy(dtype=float)
    w_dense = grid_w.full()[0]
    mask = full_rank_lag_mask(x, w_dense)
    assert mask == [True, True]  # independent draws: both lags enter
    with contextlib.redirect_stdout(io.StringIO()):
        model = spreg.ML_Lag(
            y=y, x=x, w=grid_w, slx_lags=1, name_x=["x1", "x2"], name_y="y"
        )
    return model, w_dense


@pytest.fixture(scope="module")
def fitted_slx(sar_field, grid_w):
    """OLS SLX (slx_lags=1) on the SAR-field fixture."""
    import spreg

    y = sar_field[["y"]].to_numpy(dtype=float)
    x = sar_field[["x1", "x2"]].to_numpy(dtype=float)
    with contextlib.redirect_stdout(io.StringIO()):
        model = spreg.OLS(
            y=y, x=x, w=grid_w, slx_lags=1, name_x=["x1", "x2"], name_y="y"
        )
    return model


# -------------------------------------------------------------------------- adi ---
def test_adi_simple_is_one(rook_dense):
    eig = np.linalg.eigvals(rook_dense)
    assert adi("simple", 0.4, eig, 25) == 1.0


def test_adi_full_matches_direct_inverse(rook_dense):
    eig = np.linalg.eigvals(rook_dense)
    expected = float(np.diag(np.linalg.inv(np.eye(25) - 0.4 * rook_dense)).mean())
    assert abs(adi("full", 0.4, eig, 25) - expected) < 1e-10


def test_adi_power_approximates_full(rook_dense):
    eig = np.linalg.eigvals(rook_dense)
    expected = float(np.diag(np.linalg.inv(np.eye(25) - 0.4 * rook_dense)).mean())
    got = adi("power", 0.4, eig, 25, w_dense=rook_dense)
    assert abs(got - expected) < 1e-3


def test_adi_unknown_method_raises(rook_dense):
    eig = np.linalg.eigvals(rook_dense)
    with pytest.raises(ValueError, match="unknown impact method"):
        adi("bogus", 0.4, eig, 25)


def test_adi_power_requires_w_dense(rook_dense):
    eig = np.linalg.eigvals(rook_dense)
    with pytest.raises(ValueError, match="requires w_dense"):
        adi("power", 0.4, eig, 25)


def test_adi_full_validates_eigenvalue_count(rook_dense):
    eig = np.linalg.eigvals(rook_dense)
    with pytest.raises(ValueError, match="eigenvalues"):
        adi("full", 0.4, eig[:10], 25)


# ---------------------------------------------------------------- full_rank mask ---
def test_full_rank_lag_mask_duplicate_column_masked(rook_dense):
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=25)
    x = np.column_stack([x1, x1])  # duplicated column -> its lag adds no rank
    assert full_rank_lag_mask(x, rook_dense) == [True, False]


def test_full_rank_lag_mask_independent_all_true(rook_dense):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(25, 3))
    assert full_rank_lag_mask(x, rook_dense) == [True, True, True]


def test_full_rank_lag_mask_validates_shapes(rook_dense):
    with pytest.raises(ValueError, match="2-D"):
        full_rank_lag_mask(np.ones(25), rook_dense)
    with pytest.raises(ValueError, match="w_dense"):
        full_rank_lag_mask(np.ones((10, 2)), rook_dense)


# ----------------------------------------------------------------- locate_terms ---
def test_locate_terms_round_trip_on_sdm(fitted_sdm):
    model, _ = fitted_sdm
    # Observed spreg 1.9 layout:
    # ['CONSTANT', 'x1', 'x2', 'W_x1', 'W_x2', 'W_y'] with rho ('W_y') last and
    # vm covering all 6 rows of betas (rho included).
    terms = locate_terms(model, ["x1", "x2"])
    name_x = list(model.name_x)
    assert name_x[terms["const"]] == "CONSTANT"
    assert {name_x[i] for i in terms["x"].values()} == {"x1", "x2"}
    assert {name_x[i] for i in terms["wx"].values()} == {"W_x1", "W_x2"}
    assert terms["rho"] == len(model.betas.flatten()) - 1
    assert name_x[terms["rho"]] == f"W_{model.name_y}"
    assert terms["lam"] is None
    assert model.vm.shape == (len(model.betas), len(model.betas))


def test_locate_terms_missing_name_raises(fitted_sdm):
    model, _ = fitted_sdm
    with pytest.raises(KeyError, match="nope"):
        locate_terms(model, ["nope"])


def test_locate_terms_slx_has_no_rho(fitted_slx):
    terms = locate_terms(fitted_slx, ["x1", "x2"])
    assert terms["rho"] is None
    assert terms["lam"] is None
    assert set(terms["wx"]) == {"x1", "x2"}


def test_locate_terms_vm_shape_mismatch_raises(fitted_sdm):
    model, _ = fitted_sdm

    class Broken:
        name_x = model.name_x
        name_y = model.name_y
        betas = model.betas
        vm = model.vm[:3, :3]

    with pytest.raises(ValueError, match="vm"):
        locate_terms(Broken(), ["x1"])


# ------------------------------------------------------------------- mc_impacts ---
def test_mc_impacts_point_estimates_closed_form(fitted_sdm):
    model, w_dense = fitted_sdm
    out = mc_impacts(
        model, w_dense, ["x1", "x2"], has_slx=True, n_draws=200, seed=0
    ).set_index("term")

    terms = locate_terms(model, ["x1", "x2"])
    b = model.betas.flatten()
    rho = b[terms["rho"]]
    eig = np.linalg.eigvals(w_dense)
    adi_full = adi("full", rho, eig, w_dense.shape[0])
    for k in ["x1", "x2"]:
        beta_k = b[terms["x"][k]]
        gamma_k = b[terms["wx"][k]]
        total = (beta_k + gamma_k) / (1.0 - rho)
        direct = adi_full * beta_k
        assert abs(out.loc[k, "total"] - total) < 1e-10
        assert abs(out.loc[k, "direct"] - direct) < 1e-10
        assert abs(out.loc[k, "indirect"] - (total - direct)) < 1e-10


def test_mc_impacts_ses_positive_and_columns(fitted_sdm):
    model, w_dense = fitted_sdm
    out = mc_impacts(model, w_dense, ["x1", "x2"], has_slx=True, n_draws=500, seed=1)
    assert list(out.columns) == [
        "term",
        "direct",
        "se_direct",
        "indirect",
        "se_indirect",
        "total",
        "se_total",
    ]
    assert list(out["term"]) == ["x1", "x2"]
    for col in ["se_direct", "se_indirect", "se_total"]:
        assert (out[col] > 0).all()


def test_mc_impacts_seed_reproducible(fitted_sdm):
    model, w_dense = fitted_sdm
    a = mc_impacts(model, w_dense, ["x1", "x2"], has_slx=True, n_draws=300, seed=42)
    b = mc_impacts(model, w_dense, ["x1", "x2"], has_slx=True, n_draws=300, seed=42)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_mc_impacts_draw_count_stability(fitted_sdm):
    model, w_dense = fitted_sdm
    small = mc_impacts(model, w_dense, ["x1", "x2"], has_slx=True, n_draws=200, seed=7)
    large = mc_impacts(model, w_dense, ["x1", "x2"], has_slx=True, n_draws=2000, seed=7)
    # Point estimates do not depend on the draws at all.
    for col in ["direct", "indirect", "total"]:
        np.testing.assert_allclose(small[col], large[col], rtol=0, atol=0)
    # SEs from 200 vs 2000 draws agree within 30%.
    for col in ["se_direct", "se_indirect", "se_total"]:
        rel = np.abs(small[col].to_numpy() - large[col].to_numpy())
        rel = rel / large[col].to_numpy()
        assert (rel < 0.30).all()


def test_mc_impacts_masked_lag_gamma_zero(sar_field, grid_w):
    """A duplicated regressor's lag is masked; its gamma contributes 0 to the total."""
    import spreg

    y = sar_field[["y"]].to_numpy(dtype=float)
    x1 = sar_field["x1"].to_numpy(dtype=float)
    x = np.column_stack([x1, x1])
    w_dense = grid_w.full()[0]
    mask = full_rank_lag_mask(x, w_dense)
    assert mask == [True, False]
    # spreg cannot fit the duplicated column itself, so mirror the notebook design:
    # x2's own column is independent but its lag is dropped via slx_vars.
    x = sar_field[["x1", "x2"]].to_numpy(dtype=float)
    with contextlib.redirect_stdout(io.StringIO()):
        model = spreg.ML_Lag(
            y=y,
            x=x,
            w=grid_w,
            slx_lags=1,
            slx_vars=[True, False],
            name_x=["x1", "x2"],
            name_y="y",
        )
    out = mc_impacts(
        model,
        w_dense,
        ["x1", "x2"],
        has_slx=True,
        slx_mask=[True, False],
        n_draws=200,
        seed=0,
    ).set_index("term")
    terms = locate_terms(model, ["x1", "x2"])
    assert set(terms["wx"]) == {"x1"}
    b = model.betas.flatten()
    rho = b[terms["rho"]]
    # gamma_x2 = 0 -> total_x2 = beta_x2 / (1 - rho), exactly.
    expected = b[terms["x"]["x2"]] / (1.0 - rho)
    assert abs(out.loc["x2", "total"] - expected) < 1e-10


def test_mc_impacts_inconsistent_mask_raises(fitted_sdm):
    model, w_dense = fitted_sdm
    with pytest.raises(ValueError, match="slx_mask"):
        mc_impacts(
            model,
            w_dense,
            ["x1", "x2"],
            has_slx=True,
            slx_mask=[True, False],  # model actually lags both
            n_draws=10,
        )
    with pytest.raises(ValueError, match="slx_mask"):
        mc_impacts(
            model, w_dense, ["x1", "x2"], has_slx=True, slx_mask=[True], n_draws=10
        )


def test_mc_impacts_requires_spatial_lag(fitted_slx, grid_w):
    w_dense = grid_w.full()[0]
    with pytest.raises(ValueError, match="rho"):
        mc_impacts(fitted_slx, w_dense, ["x1", "x2"], has_slx=True, n_draws=10)


def test_mc_impacts_has_slx_false_with_wx_terms_raises(fitted_sdm):
    model, w_dense = fitted_sdm
    with pytest.raises(ValueError, match="has_slx"):
        mc_impacts(model, w_dense, ["x1", "x2"], has_slx=False, n_draws=10)


def test_mc_impacts_pure_lag_model(sar_field, grid_w):
    """On a plain ML_Lag (no SLX): total = beta / (1 - rho), direct = adi * beta."""
    import spreg

    y = sar_field[["y"]].to_numpy(dtype=float)
    x = sar_field[["x1", "x2"]].to_numpy(dtype=float)
    w_dense = grid_w.full()[0]
    with contextlib.redirect_stdout(io.StringIO()):
        model = spreg.ML_Lag(y=y, x=x, w=grid_w, name_x=["x1", "x2"], name_y="y")
    out = mc_impacts(
        model, w_dense, ["x1", "x2"], has_slx=False, n_draws=200, seed=0
    ).set_index("term")
    terms = locate_terms(model, ["x1", "x2"])
    b = model.betas.flatten()
    rho = b[terms["rho"]]
    # The fixture DGP (rho=0.6, beta=(1.0, -0.5)) should be roughly recovered.
    assert 0.3 < rho < 0.9
    eig = np.linalg.eigvals(w_dense)
    adi_full = adi("full", rho, eig, w_dense.shape[0])
    for k, planted in [("x1", 1.0), ("x2", -0.5)]:
        beta_k = b[terms["x"][k]]
        assert abs(out.loc[k, "total"] - beta_k / (1.0 - rho)) < 1e-10
        assert abs(out.loc[k, "direct"] - adi_full * beta_k) < 1e-10
        assert np.sign(out.loc[k, "total"]) == np.sign(planted)


def test_mc_impacts_method_simple_direct_is_beta(fitted_sdm):
    model, w_dense = fitted_sdm
    out = mc_impacts(
        model,
        w_dense,
        ["x1", "x2"],
        has_slx=True,
        n_draws=200,
        seed=0,
        method="simple",
    ).set_index("term")
    terms = locate_terms(model, ["x1", "x2"])
    b = model.betas.flatten()
    for k in ["x1", "x2"]:
        assert abs(out.loc[k, "direct"] - b[terms["x"][k]]) < 1e-12


# --------------------------------------------------------- analytic_slx_impacts ---
def test_analytic_slx_impacts_matches_vm(fitted_slx):
    out = analytic_slx_impacts(fitted_slx, ["x1", "x2"]).set_index("term")
    terms = locate_terms(fitted_slx, ["x1", "x2"])
    b = fitted_slx.betas.flatten()
    vm = np.asarray(fitted_slx.vm)
    for k in ["x1", "x2"]:
        i, j = terms["x"][k], terms["wx"][k]
        assert abs(out.loc[k, "direct"] - b[i]) < 1e-12
        assert abs(out.loc[k, "se_direct"] - np.sqrt(vm[i, i])) < 1e-12
        assert abs(out.loc[k, "indirect"] - b[j]) < 1e-12
        assert abs(out.loc[k, "se_indirect"] - np.sqrt(vm[j, j])) < 1e-12
        assert abs(out.loc[k, "total"] - (b[i] + b[j])) < 1e-12
        se_total = np.sqrt(vm[i, i] + vm[j, j] + 2 * vm[i, j])
        assert abs(out.loc[k, "se_total"] - se_total) < 1e-12


def test_analytic_slx_impacts_masked_lag(sar_field, grid_w):
    import spreg

    y = sar_field[["y"]].to_numpy(dtype=float)
    x = sar_field[["x1", "x2"]].to_numpy(dtype=float)
    with contextlib.redirect_stdout(io.StringIO()):
        model = spreg.OLS(
            y=y,
            x=x,
            w=grid_w,
            slx_lags=1,
            slx_vars=[True, False],
            name_x=["x1", "x2"],
            name_y="y",
        )
    out = analytic_slx_impacts(model, ["x1", "x2"], [True, False]).set_index("term")
    assert np.isnan(out.loc["x2", "indirect"])
    assert np.isnan(out.loc["x2", "se_indirect"])
    assert out.loc["x2", "total"] == out.loc["x2", "direct"]
    assert out.loc["x2", "se_total"] == out.loc["x2", "se_direct"]
    assert np.isfinite(out.loc["x1", "indirect"])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy tol notice inside spreg
def test_analytic_slx_impacts_durbin_error(sar_field, grid_w):
    """ML_Error with slx_lags=1 (Durbin error): lambda located, impacts analytic."""
    import spreg

    y = sar_field[["y"]].to_numpy(dtype=float)
    x = sar_field[["x1", "x2"]].to_numpy(dtype=float)
    with contextlib.redirect_stdout(io.StringIO()):
        model = spreg.ML_Error(
            y=y, x=x, w=grid_w, slx_lags=1, name_x=["x1", "x2"], name_y="y"
        )
    terms = locate_terms(model, ["x1", "x2"])
    assert terms["lam"] == len(model.betas.flatten()) - 1
    assert terms["rho"] is None
    out = analytic_slx_impacts(model, ["x1", "x2"]).set_index("term")
    b = model.betas.flatten()
    for k in ["x1", "x2"]:
        expected = b[terms["x"][k]] + b[terms["wx"][k]]
        assert abs(out.loc[k, "total"] - expected) < 1e-12


# ------------------------------------------------------------------------ stars ---
def test_stars_levels():
    assert stars(1.5, 1.0) == ""
    assert stars(1.7, 1.0) == "*"
    assert stars(2.0, 1.0) == "**"
    assert stars(2.6, 1.0) == "***"


def test_stars_boundaries():
    # two-sided normal critical values: 1.6449 (10%), 1.9600 (5%), 2.5758 (1%)
    assert stars(1.644, 1.0) == ""
    assert stars(1.646, 1.0) == "*"
    assert stars(1.959, 1.0) == "*"
    assert stars(1.961, 1.0) == "**"
    assert stars(2.575, 1.0) == "**"
    assert stars(2.577, 1.0) == "***"


def test_stars_sign_and_degenerate_se():
    assert stars(-2.6, 1.0) == "***"  # sign-symmetric
    assert stars(1.0, 0.0) == ""
    assert stars(1.0, -1.0) == ""
    assert stars(1.0, float("nan")) == ""
    assert stars(1.0, float("inf")) == ""
