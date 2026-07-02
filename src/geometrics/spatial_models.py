"""Spatial econometric models on cross-sections (the spreg suite).

Three ``analyze_*`` functions cover the spatial-econometrics workflow of the source
paper (notebooks c04/c07):

* :func:`analyze_spatial_model` — estimate one cross-sectional spatial model (OLS,
  spatial lag, spatial error, SLX, spatial Durbin, or spatial Durbin error) and, where
  defined, the LeSage-Pace direct/indirect/total impact decomposition per regressor.
* :func:`analyze_spatial_diagnostics` — the Lagrange-multiplier specification tests on
  OLS residuals plus Moran's I, with the Anselin-Florax model recommendation.
* :func:`analyze_spatial_model_by_weights` — re-estimate the same model under a suite
  of alternative spatial weights and compare the focal regressor's impacts (the c07
  robustness exercise), as a table and a three-facet dot-whisker figure.

All estimation is done by :mod:`spreg` (stdout suppressed); impacts are computed
in-package from ``betas`` + ``vm`` via :mod:`geometrics._impacts` (terms located by
their ``name_x`` labels, never scraped from printed summaries).
"""

from __future__ import annotations

import contextlib
import io
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt
from plotly.subplots import make_subplots

from geometrics._geo import _align_cross_section, resolve_gdf_entity
from geometrics._impacts import (
    analytic_slx_impacts,
    full_rank_lag_mask,
    locate_terms,
    mc_impacts,
    stars,
)
from geometrics._labels import resolve_label, resolve_labels
from geometrics._panel import resolve_panel
from geometrics._roles import resolve_roles
from geometrics._theme import apply_default_layout, color_for
from geometrics._types import (
    SpatialDiagnosticsResult,
    SpatialModelResult,
    WeightsRobustnessResult,
)
from geometrics._validation import (
    GeometricsWarning,
    ensure_dataframe,
    ensure_geodataframe,
)
from geometrics.weights import _default_weights, make_weights

if TYPE_CHECKING:
    import geopandas as gpd
    from great_tables import GT
    from libpysal.weights import W

__all__ = [
    "analyze_spatial_model",
    "analyze_spatial_diagnostics",
    "analyze_spatial_model_by_weights",
]

_MODELS = ("ols", "lag", "error", "slx", "durbin", "durbin_error")
_METHODS = ("ml", "gm")

#: Models whose impact decomposition needs the spatial multiplier (Monte-Carlo SEs).
_MC_IMPACT_MODELS = ("lag", "durbin")
#: Models whose impacts are read analytically off the coefficients.
_SLX_IMPACT_MODELS = ("slx", "durbin_error")
#: Models that include spatially lagged regressors (SLX terms).
_SLX_TERM_MODELS = ("slx", "durbin", "durbin_error")

_MODEL_LABELS = {
    "ols": "OLS",
    "lag": "spatial lag (SAR)",
    "error": "spatial error (SEM)",
    "slx": "SLX",
    "durbin": "spatial Durbin (SDM)",
    "durbin_error": "spatial Durbin error (SDEM)",
}

#: Ordered diagnostic tests reported by :func:`analyze_spatial_diagnostics`.
_DIAG_TESTS = (
    ("moran_residuals", "Moran's I (residuals)"),
    ("lm_lag", "LM lag"),
    ("lm_error", "LM error"),
    ("robust_lm_lag", "Robust LM lag"),
    ("robust_lm_error", "Robust LM error"),
    ("lm_sarma", "LM SARMA"),
)

#: The c07 alternative-weights suite built when ``weights`` is ``None``.
_DEFAULT_WEIGHTS_SUITE = (
    ("knn4", {"method": "knn", "k": 4}),
    ("knn6", {"method": "knn", "k": 6}),
    ("knn8", {"method": "knn", "k": 8}),
    ("queen", {"method": "queen"}),
    ("rook", {"method": "rook"}),
    ("inv_distance", {"method": "inverse_distance", "power": 1.0}),
    ("inv_distance2", {"method": "inverse_distance", "power": 2.0}),
)

_STARS_NOTE = "*** p<0.01, ** p<0.05, * p<0.10 (z-based)"


# ---------------------------------------------------------------------------
# shared private machinery
# ---------------------------------------------------------------------------


def _w_spec_of(w: W) -> str:
    """Return ``w.geometrics_meta['spec']`` or compose a short human description."""
    meta = dict(getattr(w, "geometrics_meta", {}) or {})
    spec = meta.get("spec")
    if spec:
        return str(spec)
    standardized = ", row-standardized" if str(w.transform).upper() == "R" else ""
    return (
        f"user-supplied W (mean {float(w.mean_neighbors):.2f} "
        f"neighbors{standardized}), n={w.n}"
    )


def _resolve_variables(
    df: pd.DataFrame,
    outcome: str | None,
    covariates: str | Sequence[str] | None,
    fixed_effects: str | None,
    *,
    func: str,
) -> tuple[str, list[str]]:
    """Resolve outcome/covariates via roles and validate columns and dtypes.

    Raises ``ValueError`` when the roles cannot be resolved, ``KeyError`` for missing
    columns and ``TypeError`` for non-numeric focal variables, in the library's
    standard validation order.
    """
    outcome, covs = resolve_roles(df, outcome, covariates)
    if outcome is None:
        raise ValueError(
            f"{func}: an outcome is required — pass outcome=... or declare it via "
            "set_roles(df, outcome=...)"
        )
    if not covs:
        raise ValueError(
            f"{func}: at least one covariate is required — pass covariates=... or "
            "declare them via set_roles(df, covariates=...)"
        )
    needed = [outcome, *covs] + ([fixed_effects] if fixed_effects is not None else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"{func}: column(s) not found in df: {missing}")
    for col in (outcome, *covs):
        if not pdt.is_numeric_dtype(df[col]):
            raise TypeError(f"{func}: column {col!r} needs to be numeric")
    if outcome in covs:
        raise ValueError(f"{func}: outcome {outcome!r} cannot also be a covariate")
    return outcome, list(covs)


def _check_variation(cross: pd.DataFrame, cols: Sequence[str], *, func: str) -> None:
    """Raise ``ValueError`` when a focal column has zero variance after alignment."""
    for col in cols:
        values = cross[col].to_numpy(dtype=float)
        if float(np.nanstd(values)) == 0.0:
            raise ValueError(
                f"{func}: column {col!r} has zero variance in the aligned "
                "cross-section — it cannot enter the model"
            )


def _design_matrix(
    cross: pd.DataFrame,
    covariates: Sequence[str],
    cov_names: Sequence[str],
    fixed_effects: str | None,
    *,
    func: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the regressor matrix (covariates + fixed-effect dummies) and its names.

    Returns ``(x, x_names, notes)`` where ``x_names`` starts with ``cov_names`` (the
    display names of the covariates, in order) followed by the dummy column names.
    """
    notes: list[str] = []
    x_df = cross[list(covariates)].astype(float).reset_index(drop=True)
    x_df.columns = list(cov_names)
    if fixed_effects is not None:
        dummies = pd.get_dummies(
            cross[fixed_effects].astype("category"),
            prefix=str(fixed_effects),
            drop_first=True,
        ).astype(float)
        if dummies.shape[1] == 0:
            msg = (
                f"{func}: fixed_effects column {fixed_effects!r} has a single level "
                "— no dummies were added"
            )
            warnings.warn(msg, GeometricsWarning, stacklevel=4)
            notes.append(msg)
        else:
            x_df = pd.concat([x_df, dummies.reset_index(drop=True)], axis=1)
    x_names = [str(c) for c in x_df.columns]
    if x_df.shape[0] <= x_df.shape[1] + 1:
        raise ValueError(
            f"{func}: only {x_df.shape[0]} observation(s) for {x_df.shape[1]} "
            "regressor(s) plus a constant — not enough to estimate"
        )
    return x_df.to_numpy(dtype=float), x_names, notes


def _fit_model(
    *,
    model: str,
    method: str,
    y: np.ndarray,
    x: np.ndarray,
    w: W,
    x_names: Sequence[str],
    y_name: str,
    slx_vars: Any = "All",
    spat_diag: bool = False,
) -> Any:
    """Fit the requested spreg estimator with stdout (and scipy noise) suppressed."""
    import spreg

    common: dict[str, Any] = {"name_x": list(x_names), "name_y": y_name}
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        # scipy's minimize_scalar in ML_Error warns about its own tolerance default.
        warnings.filterwarnings(
            "ignore", message="Method 'bounded'", category=RuntimeWarning
        )
        if model == "ols":
            return spreg.OLS(
                y,
                x,
                w=w,
                spat_diag=spat_diag,
                moran=spat_diag,
                robust="white",
                **common,
            )
        if model == "lag":
            if method == "ml":
                return spreg.ML_Lag(y, x, w=w, **common)
            return spreg.GM_Lag(y, x, w=w, robust="white", **common)
        if model == "error":
            if method == "ml":
                return spreg.ML_Error(y, x, w=w, **common)
            return spreg.GM_Error_Het(y, x, w=w, **common)
        if model == "slx":
            return spreg.OLS(y, x, w=w, slx_lags=1, slx_vars=slx_vars, **common)
        if model == "durbin":
            if method == "ml":
                return spreg.ML_Lag(y, x, w=w, slx_lags=1, slx_vars=slx_vars, **common)
            return spreg.GM_Lag(
                y, x, w=w, slx_lags=1, slx_vars=slx_vars, robust="white", **common
            )
        # durbin_error — ML only (the gm combination is rejected upstream).
        return spreg.ML_Error(y, x, w=w, slx_lags=1, slx_vars=slx_vars, **common)


def _slx_design(
    model: str, x: np.ndarray, w_dense: np.ndarray, x_names: Sequence[str], *, func: str
) -> tuple[list[bool] | None, Any, list[str]]:
    """Compute the full-rank SLX mask for models with spatially lagged regressors.

    Returns ``(mask, slx_vars, notes)`` — ``slx_vars`` is ``"All"`` unless some lag is
    masked, in which case the boolean list is passed to spreg (verified against
    spreg 1.9, which accepts a per-column boolean list for ``slx_vars``).
    """
    if model not in _SLX_TERM_MODELS:
        return None, "All", []
    mask = full_rank_lag_mask(x, w_dense)
    if all(mask):
        return mask, "All", []
    dropped = [name for name, keep in zip(x_names, mask, strict=True) if not keep]
    msg = (
        f"{func}: dropped the collinear spatial lag(s) of {dropped} from the "
        f"{model} design (full-rank check)"
    )
    warnings.warn(msg, GeometricsWarning, stacklevel=3)
    return mask, list(mask), [msg]


def _tidy_frame(fitted: Any, y_name: str) -> pd.DataFrame:
    """Build the tidy term/estimate/se/z/p frame from a fitted spreg model."""
    betas = np.asarray(fitted.betas, dtype=float).flatten()
    terms = [str(name) for name in fitted.name_x]
    if len(terms) == len(betas) - 1:
        # GM_Lag omits the lagged-dependent label from name_x; rho is the last beta.
        terms.append("W_" + str(getattr(fitted, "name_y", y_name)))
    if hasattr(fitted, "std_err"):
        se = np.asarray(fitted.std_err, dtype=float).flatten()
    else:  # pragma: no cover - every spreg 1.9 estimator used here exposes std_err
        se = np.sqrt(np.diag(np.asarray(fitted.vm, dtype=float)))
    stat_pairs = getattr(fitted, "z_stat", None)
    if stat_pairs is None:
        stat_pairs = fitted.t_stat  # OLS reports t statistics
    z = [float(pair[0]) for pair in stat_pairs]
    p = [float(pair[1]) for pair in stat_pairs]
    return pd.DataFrame({"term": terms, "estimate": betas, "se": se, "z": z, "p": p})


def _model_scalars(
    fitted: Any, x_names: Sequence[str], *, func: str
) -> tuple[float, float, float, float, float, int, list[str]]:
    """Extract ``(rho, lam, r2, logll, aic, n_obs, notes)`` from a fitted model."""
    notes: list[str] = []
    loc = locate_terms(fitted, list(x_names))
    betas = np.asarray(fitted.betas, dtype=float).flatten()
    rho = float(betas[loc["rho"]]) if loc["rho"] is not None else float("nan")
    lam = float(betas[loc["lam"]]) if loc["lam"] is not None else float("nan")
    if hasattr(fitted, "pr2"):
        r2 = float(fitted.pr2)
    elif hasattr(fitted, "r2"):
        r2 = float(fitted.r2)
    else:  # pragma: no cover - all spreg 1.9 estimators used here report (pseudo-)R2
        r2 = float("nan")
        notes.append(f"{func}: the estimator reports no (pseudo-)R² — recorded as NaN")
    logll = float(fitted.logll) if hasattr(fitted, "logll") else float("nan")
    aic = float(fitted.aic) if hasattr(fitted, "aic") else float("nan")
    if math.isnan(aic):
        notes.append(
            f"{func}: the GM estimator has no likelihood, so log-likelihood and AIC "
            "are recorded as NaN"
        )
    return rho, lam, r2, logll, aic, int(fitted.n), notes


def _impact_table(
    *,
    model: str,
    fitted: Any,
    w_dense: np.ndarray,
    cov_names: Sequence[str],
    slx_mask: list[bool] | None,
    n_draws: int,
    seed: int | None,
) -> pd.DataFrame | None:
    """Compute the per-covariate impact table for models where impacts are defined."""
    cov_mask = list(slx_mask[: len(cov_names)]) if slx_mask is not None else None
    if model in _MC_IMPACT_MODELS:
        return mc_impacts(
            fitted,
            w_dense,
            cov_names,
            has_slx=(model == "durbin"),
            slx_mask=cov_mask if model == "durbin" else None,
            n_draws=n_draws,
            seed=seed,
            method="full",
        )
    if model in _SLX_IMPACT_MODELS:
        return analytic_slx_impacts(fitted, cov_names, cov_mask)
    return None


def _fmt(value: float, digits: int = 4) -> str:
    """Format a scalar for the Great Tables displays (``--`` for non-finite)."""
    return f"{value:.{digits}f}" if np.isfinite(value) else "--"


def _coefficient_gt(
    tidy: pd.DataFrame,
    *,
    model: str,
    method: str,
    outcome_label: str,
    n_obs: int,
    r2: float,
    aic: float,
    w_spec: str,
    title: str | None,
) -> GT:
    """Render the coefficient table (estimates, stars, SEs) with a model header."""
    from great_tables import GT

    disp = pd.DataFrame(
        {
            "Term": tidy["term"],
            "Estimate": [
                f"{est:.4f}{stars(est, se)}"
                for est, se in zip(tidy["estimate"], tidy["se"], strict=True)
            ],
            "Std. Error": [f"({se:.4f})" for se in tidy["se"]],
            "z": [_fmt(z, 2) for z in tidy["z"]],
            "p": [_fmt(p, 4) for p in tidy["p"]],
        }
    )
    header = title or f"{_MODEL_LABELS[model].capitalize()} model of {outcome_label}"
    return (
        GT(disp, rowname_col="Term")
        .tab_header(
            title=header,
            subtitle=(
                f"{method.upper()} estimates | n = {n_obs} | "
                f"R² = {_fmt(r2, 3)} | AIC = {_fmt(aic, 1)}"
            ),
        )
        .tab_source_note(_STARS_NOTE)
        .tab_source_note(f"W: {w_spec}")
    )


def _restrict_w(w: W, kept: list[Any], *, func: str) -> W:
    """Restrict/reorder a weights object to the aligned unit ids (transform kept)."""
    from libpysal.weights.util import w_subset

    if list(w.id_order) == kept:
        return w
    not_in_w = [i for i in kept if i not in set(w.id_order)]
    if not_in_w:
        raise ValueError(
            f"{func}: w does not cover all aligned units (e.g. "
            f"{[str(i) for i in not_in_w[:5]]}) — build w from the same gdf"
        )
    transform = w.transform
    out = w_subset(w, kept, silence_warnings=True)
    out.transform = transform
    meta = dict(getattr(w, "geometrics_meta", {}) or {})
    if meta:
        meta["n"] = out.n
        out.geometrics_meta = meta
    return out


# ---------------------------------------------------------------------------
# analyze_spatial_model
# ---------------------------------------------------------------------------


def analyze_spatial_model(
    df: pd.DataFrame,
    outcome: str | None = None,
    covariates: str | Sequence[str] | None = None,
    *,
    gdf: gpd.GeoDataFrame,
    w: W | None = None,
    model: str = "durbin",
    method: str = "ml",
    period: Any = None,
    entity: str | None = None,
    time: str | None = None,
    fixed_effects: str | None = None,
    impacts: bool = True,
    n_draws: int = 10_000,
    seed: int | None = 20250620,
    spat_diag: bool = False,
    title: str | None = None,
) -> SpatialModelResult:
    """Estimate a cross-sectional spatial econometric model with impact decomposition.

    One period of the panel (the latest by default) is aligned to the geometry and
    weights, the requested :mod:`spreg` model is estimated, and — for models with a
    spatially lagged outcome or regressors — the per-covariate LeSage-Pace
    direct/indirect/total impacts are computed from ``betas`` + ``vm`` (Monte-Carlo
    standard errors for lag/Durbin models, analytic for SLX/Durbin-error).

    Parameters
    ----------
    df
        Long panel (or cross-section) holding the outcome and covariates per entity.
    outcome
        Dependent variable. Defaults to the outcome declared via
        :func:`geometrics.set_roles`.
    covariates
        Regressor column(s). Default to the covariates declared via
        :func:`geometrics.set_roles`.
    gdf
        Entity geometry carrying the same entity ids as ``df``.
    w
        ``libpysal`` weights aligned to the gdf ids. ``None`` builds the default
        weights (queen contiguity for polygons) with a
        :class:`~geometrics.GeometricsWarning`.
    model
        ``"ols"``, ``"lag"`` (SAR), ``"error"`` (SEM), ``"slx"``, ``"durbin"`` (SDM)
        or ``"durbin_error"`` (SDEM).
    method
        ``"ml"`` (maximum likelihood) or ``"gm"`` (method of moments / GMM; not
        available for ``durbin_error``). OLS-based models (``ols`` / ``slx``) ignore it.
    period
        Period to model when ``df`` has a time dimension; ``None`` uses the latest
        period and records a note.
    entity, time
        Panel identifiers; default to the ids declared via :func:`geometrics.set_panel`.
    fixed_effects
        Categorical column expanded to ``drop_first`` dummy regressors (e.g. state
        fixed effects). Dummies are never spatially lagged when their lag is collinear
        (full-rank check).
    impacts
        Compute the impact table where defined (lag/durbin: Monte-Carlo; slx/
        durbin_error: analytic). OLS and pure error models have no impact
        decomposition (``impacts`` is ``None``).
    n_draws
        Monte-Carlo draws for the impact standard errors.
    seed
        Seed for the Monte-Carlo draws (reproducible by default).
    spat_diag
        For ``model="ols"`` only: attach spreg's spatial diagnostics to the fitted
        object (see :func:`analyze_spatial_diagnostics` for the full workflow).
    title
        Header for the coefficient table. Defaults to the model and outcome labels.

    Returns
    -------
    SpatialModelResult
        Frozen result with the tidy coefficient frame (``df``), the Great Tables
        coefficient table (``gt``), the fitted spreg object (``model_obj``), the
        spatial parameters (``rho`` / ``lam``), fit scalars, the impact table
        (``impacts``) and ``w_spec``.

    Raises
    ------
    KeyError
        If a requested column is not in ``df``.
    TypeError
        If the outcome or a covariate is not numeric.
    ValueError
        For an unknown ``model`` / ``method``, the unsupported ``durbin_error`` +
        ``gm`` combination, an unknown ``period``, or too few / degenerate
        observations.

    Examples
    --------
    A spatial lag model on a small constructed lattice:

    ```python
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import box

    from geometrics.spatial_models import analyze_spatial_model
    from geometrics.weights import make_weights

    cells = [box(i % 4, i // 4, i % 4 + 1, i // 4 + 1) for i in range(16)]
    gdf = gpd.GeoDataFrame(
        {"id": [f"r{i}" for i in range(16)]}, geometry=cells, crs="EPSG:4326"
    )
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": gdf["id"], "x": rng.normal(size=16)})
    df["y"] = 2.0 * df["x"] + rng.normal(scale=0.1, size=16)
    res = analyze_spatial_model(
        df, "y", ["x"], gdf=gdf, w=make_weights(gdf), model="lag",
        entity="id", n_draws=200,
    )
    print(res.model, res.n_obs, res.impacts.shape)
    ```
    """
    func = "analyze_spatial_model"
    df = ensure_dataframe(df)
    gdf = ensure_geodataframe(gdf, func=func)
    entity, time = resolve_panel(df, entity, time, require_entity=True)
    outcome, covs = _resolve_variables(
        df, outcome, covariates, fixed_effects, func=func
    )
    model = str(model).lower()
    method = str(method).lower()
    if model not in _MODELS:
        raise ValueError(
            f"{func}: unknown model {model!r}; choose from {list(_MODELS)}"
        )
    if method not in _METHODS:
        raise ValueError(
            f"{func}: unknown method {method!r}; choose from {list(_METHODS)}"
        )
    if model == "durbin_error" and method == "gm":
        raise ValueError(
            f"{func}: the spatial Durbin error model has no GM estimator in spreg — "
            "use method='ml'"
        )
    if impacts and model in _MC_IMPACT_MODELS and n_draws < 2:
        raise ValueError(f"{func}: n_draws must be at least 2, got {n_draws}")

    notes: list[str] = []
    if w is None:
        w = _default_weights(gdf, func=func)
        notes.append(f"{func}: no weights supplied — defaulted to {_w_spec_of(w)}")

    cols = [outcome, *covs] + ([fixed_effects] if fixed_effects is not None else [])
    cross, w_aligned, info = _align_cross_section(
        df,
        gdf,
        cols,
        entity=entity,
        time=time,
        period=period,
        w=w,
        min_obs=len(covs) + 3,
        func=func,
    )
    notes.extend(info["notes"])
    period = info["period"]
    _check_variation(cross, [outcome, *covs], func=func)

    cov_names = resolve_labels(df, covs)
    y_name = resolve_label(df, outcome)
    y = cross[outcome].to_numpy(dtype=float).reshape(-1, 1)
    x, x_names, design_notes = _design_matrix(
        cross, covs, cov_names, fixed_effects, func=func
    )
    notes.extend(design_notes)

    w_dense = np.asarray(w_aligned.full()[0], dtype=float)
    slx_mask, slx_vars, mask_notes = _slx_design(model, x, w_dense, x_names, func=func)
    notes.extend(mask_notes)

    fitted = _fit_model(
        model=model,
        method=method,
        y=y,
        x=x,
        w=w_aligned,
        x_names=x_names,
        y_name=y_name,
        slx_vars=slx_vars,
        spat_diag=spat_diag,
    )

    tidy = _tidy_frame(fitted, y_name)
    rho, lam, r2, logll, aic, n_obs, scalar_notes = _model_scalars(
        fitted, x_names, func=func
    )
    notes.extend(scalar_notes)

    impacts_df: pd.DataFrame | None = None
    draws_used = 0
    if impacts:
        impacts_df = _impact_table(
            model=model,
            fitted=fitted,
            w_dense=w_dense,
            cov_names=cov_names,
            slx_mask=slx_mask,
            n_draws=n_draws,
            seed=seed,
        )
        if impacts_df is None:
            notes.append(
                f"{func}: direct/indirect impacts are not defined for the "
                f"{model!r} model"
            )
        elif model in _MC_IMPACT_MODELS:
            draws_used = n_draws

    w_spec = _w_spec_of(w_aligned)
    gt = _coefficient_gt(
        tidy,
        model=model,
        method=method,
        outcome_label=y_name,
        n_obs=n_obs,
        r2=r2,
        aic=aic,
        w_spec=w_spec,
        title=title,
    )

    return SpatialModelResult(
        df=tidy,
        gt=gt,
        model_obj=fitted,
        model=model,
        method=method,
        rho=rho,
        lam=lam,
        r2=r2,
        log_likelihood=logll,
        aic=aic,
        n_obs=n_obs,
        outcome=outcome,
        covariates=tuple(covs),
        period=period,
        w_spec=w_spec,
        impacts=impacts_df,
        n_draws=draws_used,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# analyze_spatial_diagnostics
# ---------------------------------------------------------------------------


def _diagnostics_recommendation(
    stats: dict[str, tuple[float, float]], alpha: float
) -> tuple[str, str]:
    """Apply the Anselin-Florax decision rule to the LM test results.

    ``stats`` maps test name to ``(statistic, p)``. Returns ``(recommendation,
    reasoning)`` where the recommendation is ``"ols"``, ``"lag"`` or ``"error"`` and
    the reasoning is a short plain-language explanation (mentioning the spatial
    Durbin model when both robust tests reject).
    """
    lm_lag_stat, lm_lag_p = stats["lm_lag"]
    lm_err_stat, lm_err_p = stats["lm_error"]
    r_lag_stat, r_lag_p = stats["robust_lm_lag"]
    r_err_stat, r_err_p = stats["robust_lm_error"]

    lag_sig, err_sig = lm_lag_p < alpha, lm_err_p < alpha
    r_lag_sig, r_err_sig = r_lag_p < alpha, r_err_p < alpha

    if not lag_sig and not err_sig:
        return "ols", (
            f"Neither simple LM test rejects the null of no spatial dependence at "
            f"alpha = {alpha:g} (LM lag p = {lm_lag_p:.3f}, LM error p = "
            f"{lm_err_p:.3f}). The OLS residuals look spatially random under this W, "
            "so a non-spatial specification is adequate. Treat this as consistency "
            "with independence under the chosen weights, not proof of it."
        )
    if r_lag_sig and not r_err_sig:
        return "lag", (
            f"At least one simple LM test rejects at alpha = {alpha:g}, so the robust "
            f"forms decide: robust LM lag remains significant (statistic = "
            f"{r_lag_stat:.2f}, p = {r_lag_p:.3g}) while robust LM error does not "
            f"(p = {r_err_p:.3f}). The Anselin-Florax rule reads this as dependence "
            "entering through the spatially lagged outcome, pointing to the spatial "
            "lag (SAR) model."
        )
    if r_err_sig and not r_lag_sig:
        return "error", (
            f"At least one simple LM test rejects at alpha = {alpha:g}, so the robust "
            f"forms decide: robust LM error remains significant (statistic = "
            f"{r_err_stat:.2f}, p = {r_err_p:.3g}) while robust LM lag does not "
            f"(p = {r_lag_p:.3f}). The Anselin-Florax rule reads this as spatially "
            "correlated disturbances, pointing to the spatial error (SEM) model."
        )
    if r_lag_sig and r_err_sig:
        winner = "lag" if r_lag_stat >= r_err_stat else "error"
        return winner, (
            f"Both robust LM tests reject at alpha = {alpha:g} (robust LM lag = "
            f"{r_lag_stat:.2f}, robust LM error = {r_err_stat:.2f}), so the larger "
            f"robust statistic wins and points to the {winner} model. With both "
            "channels active, consider durbin (SDM) - it nests both the lag and the "
            "spatially-lagged-X channels and lets the data separate them."
        )
    winner = (
        "lag"
        if (lag_sig and (not err_sig or lm_lag_stat >= lm_err_stat))
        else ("error")
    )
    return winner, (
        f"A simple LM test rejects at alpha = {alpha:g} (LM lag = {lm_lag_stat:.2f}, "
        f"p = {lm_lag_p:.3g}; LM error = {lm_err_stat:.2f}, p = {lm_err_p:.3g}) but "
        "neither robust form does, an ambiguous configuration. The larger simple "
        f"statistic points to the {winner} model; treat the choice as tentative and "
        "compare specifications directly."
    )


def _diagnostics_gt(
    disp_rows: pd.DataFrame, *, outcome_label: str, w_spec: str, recommendation: str
) -> GT:
    """Render the diagnostics table with the recommendation as a source note."""
    from great_tables import GT

    labels = dict(_DIAG_TESTS)
    disp = pd.DataFrame(
        {
            "Test": [labels[t] for t in disp_rows["test"]],
            "Statistic": [_fmt(s) for s in disp_rows["statistic"]],
            "df": [f"{d:.0f}" if np.isfinite(d) else "--" for d in disp_rows["df"]],
            "p-value": [_fmt(p) for p in disp_rows["p"]],
        }
    )
    return (
        GT(disp, rowname_col="Test")
        .tab_header(
            title=f"Spatial dependence diagnostics for {outcome_label}",
            subtitle=f"OLS residual tests | W: {w_spec}",
        )
        .tab_source_note(f"Anselin-Florax recommendation: {recommendation}")
    )


def analyze_spatial_diagnostics(
    df: pd.DataFrame,
    outcome: str | None = None,
    covariates: str | Sequence[str] | None = None,
    *,
    gdf: gpd.GeoDataFrame,
    w: W | None = None,
    period: Any = None,
    entity: str | None = None,
    time: str | None = None,
    fixed_effects: str | None = None,
    alpha: float = 0.05,
) -> SpatialDiagnosticsResult:
    """Run the LM specification tests on OLS residuals and recommend a spatial model.

    Estimates the non-spatial OLS benchmark, computes Moran's I on its residuals and
    the five Lagrange-multiplier tests (LM lag / LM error, their robust forms, and LM
    SARMA), then applies the Anselin-Florax decision rule: no LM rejection keeps OLS;
    otherwise the significant *robust* test picks the lag or error model, and when
    both robust tests reject the larger statistic wins (with a pointer to the spatial
    Durbin model, which nests both channels).

    Parameters
    ----------
    df
        Long panel (or cross-section) holding the outcome and covariates per entity.
    outcome, covariates
        Dependent variable and regressors; default to the roles declared via
        :func:`geometrics.set_roles`.
    gdf
        Entity geometry carrying the same entity ids as ``df``.
    w
        ``libpysal`` weights aligned to the gdf ids; ``None`` builds the default
        weights with a :class:`~geometrics.GeometricsWarning`.
    period
        Period to test; ``None`` uses the latest period and records a note.
    entity, time
        Panel identifiers; default to the ids declared via :func:`geometrics.set_panel`.
    fixed_effects
        Categorical column expanded to ``drop_first`` dummies in the OLS design.
    alpha
        Significance level for the decision rule.

    Returns
    -------
    SpatialDiagnosticsResult
        Frozen result with one row per test (``test``, ``statistic``, ``df``, ``p``),
        the rendered table, the residual Moran's I, the ``recommendation`` and its
        ``reasoning``, the fitted OLS benchmark and ``w_spec``.

    Raises
    ------
    KeyError
        If a requested column is not in ``df``.
    TypeError
        If the outcome or a covariate is not numeric.
    ValueError
        If ``alpha`` is not in (0, 1), the period is unknown, or the aligned
        cross-section is too small or degenerate.

    Examples
    --------
    Diagnostics on a small constructed lattice:

    ```python
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import box

    from geometrics.spatial_models import analyze_spatial_diagnostics
    from geometrics.weights import make_weights

    cells = [box(i % 4, i // 4, i % 4 + 1, i // 4 + 1) for i in range(16)]
    gdf = gpd.GeoDataFrame(
        {"id": [f"r{i}" for i in range(16)]}, geometry=cells, crs="EPSG:4326"
    )
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": gdf["id"], "x": rng.normal(size=16)})
    df["y"] = 2.0 * df["x"] + rng.normal(scale=0.1, size=16)
    res = analyze_spatial_diagnostics(df, "y", ["x"], gdf=gdf, w=make_weights(gdf), entity="id")
    print(res.recommendation)
    ```
    """
    from scipy import stats as scipy_stats
    from spreg import diagnostics_sp

    func = "analyze_spatial_diagnostics"
    df = ensure_dataframe(df)
    gdf = ensure_geodataframe(gdf, func=func)
    entity, time = resolve_panel(df, entity, time, require_entity=True)
    outcome, covs = _resolve_variables(
        df, outcome, covariates, fixed_effects, func=func
    )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"{func}: alpha needs to be in (0, 1), got {alpha}")

    notes: list[str] = []
    if w is None:
        w = _default_weights(gdf, func=func)
        notes.append(f"{func}: no weights supplied — defaulted to {_w_spec_of(w)}")

    cols = [outcome, *covs] + ([fixed_effects] if fixed_effects is not None else [])
    cross, w_aligned, info = _align_cross_section(
        df,
        gdf,
        cols,
        entity=entity,
        time=time,
        period=period,
        w=w,
        min_obs=len(covs) + 3,
        func=func,
    )
    notes.extend(info["notes"])
    _check_variation(cross, [outcome, *covs], func=func)

    cov_names = resolve_labels(df, covs)
    y_name = resolve_label(df, outcome)
    y = cross[outcome].to_numpy(dtype=float).reshape(-1, 1)
    x, x_names, design_notes = _design_matrix(
        cross, covs, cov_names, fixed_effects, func=func
    )
    notes.extend(design_notes)

    import spreg

    with contextlib.redirect_stdout(io.StringIO()):
        ols = spreg.OLS(y, x, name_x=x_names, name_y=y_name)
    lm = diagnostics_sp.LMtests(ols, w_aligned)
    moran = diagnostics_sp.MoranRes(ols, w_aligned, z=True)
    p_moran = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(float(moran.zI)))))

    rows = pd.DataFrame(
        {
            "test": [name for name, _ in _DIAG_TESTS],
            "statistic": [
                float(moran.I),
                float(lm.lml[0]),
                float(lm.lme[0]),
                float(lm.rlml[0]),
                float(lm.rlme[0]),
                float(lm.sarma[0]),
            ],
            "df": [float("nan"), 1.0, 1.0, 1.0, 1.0, 2.0],
            "p": [
                p_moran,
                float(lm.lml[1]),
                float(lm.lme[1]),
                float(lm.rlml[1]),
                float(lm.rlme[1]),
                float(lm.sarma[1]),
            ],
        }
    )
    stats_map = {
        name: (float(stat), float(p))
        for name, stat, p in zip(
            rows["test"], rows["statistic"], rows["p"], strict=True
        )
    }
    recommendation, reasoning = _diagnostics_recommendation(stats_map, alpha)

    w_spec = _w_spec_of(w_aligned)
    gt = _diagnostics_gt(
        rows, outcome_label=y_name, w_spec=w_spec, recommendation=recommendation
    )

    return SpatialDiagnosticsResult(
        df=rows,
        gt=gt,
        moran_i_resid=float(moran.I),
        recommendation=recommendation,
        reasoning=reasoning,
        ols_model=ols,
        alpha=float(alpha),
        w_spec=w_spec,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# analyze_spatial_model_by_weights
# ---------------------------------------------------------------------------


def _weights_suite(gdf: gpd.GeoDataFrame) -> dict[str, W]:
    """Build the default c07 alternative-weights suite from the geometry."""
    return {
        name: make_weights(gdf, **kwargs) for name, kwargs in _DEFAULT_WEIGHTS_SUITE
    }


def _robustness_fig(
    frame: pd.DataFrame,
    *,
    baseline: str,
    focal_label: str,
    model: str,
    title: str | None,
) -> go.Figure:
    """Build the three-facet (Direct/Indirect/Total) dot-whisker robustness figure."""
    effects = (("direct", "Direct"), ("indirect", "Indirect"), ("total", "Total"))
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=[label for _, label in effects],
        horizontal_spacing=0.06,
    )
    alt_color, base_color = color_for(0), color_for(1)
    is_base = frame["weights"].eq(baseline)
    for col_idx, (key, label) in enumerate(effects, start=1):
        for base_flag, sub in ((False, frame[~is_base]), (True, frame[is_base])):
            if sub.empty:
                continue
            est = sub[key].to_numpy(dtype=float)
            se = sub["se_" + key].to_numpy(dtype=float)
            color = base_color if base_flag else alt_color
            fig.add_trace(
                go.Scatter(
                    x=est,
                    y=sub["weights"],
                    mode="markers",
                    marker={
                        "color": color,
                        "size": 12 if base_flag else 9,
                        "symbol": "diamond" if base_flag else "circle",
                    },
                    error_x={
                        "type": "data",
                        "array": 1.96 * se,
                        "color": color,
                        "thickness": 1.5,
                        "width": 4,
                    },
                    customdata=np.column_stack(
                        [sub["weights"], est - 1.96 * se, est + 1.96 * se]
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        + label
                        + ": %{x:.4f}<br>95% CI: [%{customdata[1]:.4f}, "
                        "%{customdata[2]:.4f}]<extra></extra>"
                    ),
                    name="baseline" if base_flag else "alternative",
                    legendgroup="baseline" if base_flag else "alternative",
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
        base_value = float(frame.loc[is_base, key].iloc[0])
        if np.isfinite(base_value):
            fig.add_vline(
                x=base_value,
                line_dash="dash",
                line_color=base_color,
                line_width=1,
                row=1,
                col=col_idx,
            )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(list(frame["weights"]))),
        row=1,
        col=1,
    )
    apply_default_layout(
        fig,
        title=title or f"Robustness of the {focal_label} impacts to the weights choice",
        subtitle=(
            f"{_MODEL_LABELS[model].capitalize()} model | 95% CIs | "
            f"baseline = {baseline} (dashed)"
        ),
        # Legend under the plot: the title is long and the three subplot titles
        # (Direct/Indirect/Total) occupy the top band, so a top legend would collide.
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "xanchor": "center",
            "x": 0.5,
        },
        margin_b=90,
    )
    return fig


def _robustness_gt(
    frame: pd.DataFrame, *, baseline: str, focal_label: str, model: str
) -> GT:
    """Render the per-weights impact comparison table."""
    from great_tables import GT

    def _cell(est: float, se: float) -> str:
        if not np.isfinite(est):
            return "--"
        return f"{est:.4f}{stars(est, se)} ({se:.4f})"

    disp = pd.DataFrame(
        {
            "Weights": [
                f"{name} (baseline)" if name == baseline else str(name)
                for name in frame["weights"]
            ],
            "Direct": [
                _cell(e, s)
                for e, s in zip(frame["direct"], frame["se_direct"], strict=True)
            ],
            "Indirect": [
                _cell(e, s)
                for e, s in zip(frame["indirect"], frame["se_indirect"], strict=True)
            ],
            "Total": [
                _cell(e, s)
                for e, s in zip(frame["total"], frame["se_total"], strict=True)
            ],
            "rho": [_fmt(r, 3) for r in frame["rho"]],
            "AIC": [_fmt(a, 1) for a in frame["aic"]],
        }
    )
    return (
        GT(disp, rowname_col="Weights")
        .tab_header(
            title=f"Impacts of {focal_label} under alternative spatial weights",
            subtitle=f"{_MODEL_LABELS[model].capitalize()} model, re-estimated per W",
        )
        .tab_source_note(_STARS_NOTE)
    )


def analyze_spatial_model_by_weights(
    df: pd.DataFrame,
    outcome: str | None = None,
    covariates: str | Sequence[str] | None = None,
    *,
    gdf: gpd.GeoDataFrame,
    weights: Mapping[str, W] | None = None,
    baseline: str | None = None,
    focal: str | None = None,
    model: str = "durbin",
    period: Any = None,
    entity: str | None = None,
    time: str | None = None,
    fixed_effects: str | None = None,
    n_draws: int = 10_000,
    seed: int | None = 20250620,
    title: str | None = None,
) -> WeightsRobustnessResult:
    """Re-estimate a spatial model under alternative weights and compare the impacts.

    The weights-choice robustness check of the source paper (notebook c07): the same
    model is re-estimated under each spatial weights specification, and the focal
    regressor's direct/indirect/total impacts are compared across specifications in a
    table and a three-facet dot-whisker figure (95% Monte-Carlo confidence intervals,
    baseline highlighted with a dashed reference line).

    Parameters
    ----------
    df
        Long panel (or cross-section) holding the outcome and covariates per entity.
    outcome, covariates
        Dependent variable and regressors; default to the roles declared via
        :func:`geometrics.set_roles`.
    gdf
        Entity geometry carrying the same entity ids as ``df``.
    weights
        Mapping of specification name to ``libpysal`` weights. ``None`` builds the
        paper's suite from the geometry: 4/6/8-nearest-neighbor, queen and rook
        contiguity, and inverse distance with powers 1 and 2 (all row-standardized).
    baseline
        Name of the reference specification (highlighted in the figure). Defaults to
        the first key of ``weights``.
    focal
        Covariate whose impacts are compared. Defaults to the first covariate.
    model
        ``"lag"``, ``"slx"``, ``"durbin"`` (default) or ``"durbin_error"`` — a model
        with a defined impact decomposition. Estimated by ML.
    period
        Period to model; ``None`` uses the latest period and records a note.
    entity, time
        Panel identifiers; default to the ids declared via :func:`geometrics.set_panel`.
    fixed_effects
        Categorical column expanded to ``drop_first`` dummies in each design.
    n_draws
        Monte-Carlo draws for the impact standard errors (the RNG is re-seeded per
        weights specification, so rows are reproducible individually).
    seed
        Seed for the Monte-Carlo draws.
    title
        Figure title. Defaults to a title naming the focal regressor.

    Returns
    -------
    WeightsRobustnessResult
        Frozen result with one row per specification (``weights``, ``rho``,
        ``direct`` / ``indirect`` / ``total`` and their standard errors, ``aic``,
        ``n_obs``, ``w_spec``), the dot-whisker figure and the comparison table.

    Raises
    ------
    KeyError
        If a requested column is not in ``df``.
    TypeError
        If the outcome or a covariate is not numeric.
    ValueError
        For a model without impacts (``ols`` / ``error``), an empty ``weights``
        mapping, an unknown ``baseline`` or ``focal``, or degenerate data.

    Examples
    --------
    Compare queen contiguity against 4-nearest-neighbor weights:

    ```python
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import box

    from geometrics.spatial_models import analyze_spatial_model_by_weights
    from geometrics.weights import make_weights

    cells = [box(i % 4, i // 4, i % 4 + 1, i // 4 + 1) for i in range(16)]
    gdf = gpd.GeoDataFrame(
        {"id": [f"r{i}" for i in range(16)]}, geometry=cells, crs="EPSG:4326"
    )
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": gdf["id"], "x": rng.normal(size=16)})
    df["y"] = 2.0 * df["x"] + rng.normal(scale=0.1, size=16)
    res = analyze_spatial_model_by_weights(
        df, "y", ["x"], gdf=gdf, model="lag", entity="id", n_draws=200,
        weights={
            "queen": make_weights(gdf, method="queen"),
            "knn4": make_weights(gdf, method="knn", k=4),
        },
    )
    print(res.baseline, list(res.df["weights"]))
    ```
    """
    func = "analyze_spatial_model_by_weights"
    df = ensure_dataframe(df)
    gdf = ensure_geodataframe(gdf, func=func)
    entity, time = resolve_panel(df, entity, time, require_entity=True)
    outcome, covs = _resolve_variables(
        df, outcome, covariates, fixed_effects, func=func
    )
    model = str(model).lower()
    if model not in _MODELS:
        raise ValueError(
            f"{func}: unknown model {model!r}; choose from {list(_MODELS)}"
        )
    if model not in (*_MC_IMPACT_MODELS, *_SLX_IMPACT_MODELS):
        raise ValueError(
            f"{func}: model {model!r} has no impact decomposition to compare — "
            f"choose from {[*_MC_IMPACT_MODELS, *_SLX_IMPACT_MODELS]}"
        )
    if model in _MC_IMPACT_MODELS and n_draws < 2:
        raise ValueError(f"{func}: n_draws must be at least 2, got {n_draws}")
    focal = focal if focal is not None else covs[0]
    if focal not in covs:
        raise ValueError(
            f"{func}: focal {focal!r} needs to be one of the covariates {covs}"
        )

    notes: list[str] = []
    if weights is None:
        weights = _weights_suite(gdf)
        notes.append(
            f"{func}: no weights supplied — built the default suite "
            f"{list(weights)} from the geometry"
        )
    weights = dict(weights)
    if not weights:
        raise ValueError(f"{func}: weights needs at least one specification")
    baseline = baseline if baseline is not None else next(iter(weights))
    if baseline not in weights:
        raise ValueError(
            f"{func}: baseline {baseline!r} is not one of the weights "
            f"specifications {list(weights)}"
        )

    cols = [outcome, *covs] + ([fixed_effects] if fixed_effects is not None else [])
    cross, _, info = _align_cross_section(
        df,
        gdf,
        cols,
        entity=entity,
        time=time,
        period=period,
        w=None,
        min_obs=len(covs) + 3,
        func=func,
    )
    notes.extend(info["notes"])
    _check_variation(cross, [outcome, *covs], func=func)

    cov_names = resolve_labels(df, covs)
    y_name = resolve_label(df, outcome)
    focal_label = cov_names[covs.index(focal)]
    y = cross[outcome].to_numpy(dtype=float).reshape(-1, 1)
    x, x_names, design_notes = _design_matrix(
        cross, covs, cov_names, fixed_effects, func=func
    )
    notes.extend(design_notes)
    gdf_entity = resolve_gdf_entity(gdf)
    kept = list(cross[gdf_entity])

    rows: list[dict[str, Any]] = []
    for name, w_raw in weights.items():
        w_i = _restrict_w(w_raw, kept, func=func)
        w_dense = np.asarray(w_i.full()[0], dtype=float)
        slx_mask, slx_vars, mask_notes = _slx_design(
            model, x, w_dense, x_names, func=func
        )
        notes.extend(f"{note} [weights: {name}]" for note in mask_notes)
        fitted = _fit_model(
            model=model,
            method="ml",
            y=y,
            x=x,
            w=w_i,
            x_names=x_names,
            y_name=y_name,
            slx_vars=slx_vars,
        )
        rho, _, _, _, aic, n_obs, _ = _model_scalars(fitted, x_names, func=func)
        # Re-seed per specification so each row reproduces independently.
        impact = _impact_table(
            model=model,
            fitted=fitted,
            w_dense=w_dense,
            cov_names=[focal_label],
            slx_mask=(
                [slx_mask[cov_names.index(focal_label)]]
                if slx_mask is not None
                else None
            ),
            n_draws=n_draws,
            seed=seed,
        )
        assert impact is not None  # guarded by the model check above
        row = impact.iloc[0]
        rows.append(
            {
                "weights": name,
                "rho": rho,
                "direct": float(row["direct"]),
                "se_direct": float(row["se_direct"]),
                "indirect": float(row["indirect"]),
                "se_indirect": float(row["se_indirect"]),
                "total": float(row["total"]),
                "se_total": float(row["se_total"]),
                "aic": aic,
                "n_obs": n_obs,
                "w_spec": _w_spec_of(w_i),
            }
        )

    frame = pd.DataFrame(rows)
    fig = _robustness_fig(
        frame, baseline=baseline, focal_label=focal_label, model=model, title=title
    )
    gt = _robustness_gt(frame, baseline=baseline, focal_label=focal_label, model=model)

    return WeightsRobustnessResult(
        df=frame,
        fig=fig,
        gt=gt,
        baseline=baseline,
        focal=focal,
        model=model,
        notes=tuple(notes),
    )
