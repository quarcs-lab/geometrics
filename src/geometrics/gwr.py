"""Local spatial models: GWR and multiscale GWR (mgwr).

Where the global spatial models (:mod:`geometrics.models`) estimate one coefficient
per regressor for the whole map, :func:`analyze_gwr` fits **geographically weighted
regression**: a separate, distance-weighted regression at every location, so each
covariate's association with the outcome becomes a *surface* rather than a single
number. :func:`analyze_mgwr` relaxes GWR's single shared bandwidth and lets every
term operate at its own spatial scale (multiscale GWR).

Both functions align the panel cross-section to the entity geometry, compute
calibration coordinates from **metric-CRS centroids** (never geographic degrees),
select the bandwidth(s) by golden-section search when not supplied, and apply the
da Silva & Fotheringham multiple-testing correction when flagging significant local
coefficients. Every local surface is returned as a themed diverging choropleth with
non-significant units greyed out.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from geometrics._common import entity_display_map
from geometrics._geo import _align_cross_section, ensure_metric_crs, resolve_gdf_entity
from geometrics._labels import resolve_label
from geometrics._mapping import continuous_map
from geometrics._panel import resolve_entity_name, resolve_panel
from geometrics._roles import resolve_roles
from geometrics._types import GWRResult, MGWRResult
from geometrics._validation import (
    GeometricsWarning,
    ensure_dataframe,
    ensure_geodataframe,
)

if TYPE_CHECKING:
    import geopandas as gpd
    import plotly.graph_objects as go
    from great_tables import GT

__all__ = ["analyze_gwr", "analyze_mgwr"]

_GWR_FUNC = "analyze_gwr"
_MGWR_FUNC = "analyze_mgwr"

#: Name used for the intercept term in result frames, figures and tables.
_CONST = "const"

#: Above this many units the golden-section bandwidth search gets expensive.
_LARGE_N = 3000

#: Kernels mgwr implements.
_KERNELS = ("bisquare", "gaussian", "exponential")

#: Bandwidth-selection criteria mgwr implements.
_CRITERIA = ("AICc", "AIC", "BIC", "CV")

#: Nominal alpha levels of the rows of ``GWRResults.adj_alpha`` /
#: ``MGWRResults.adj_alpha_j`` (mgwr 2.2 tabulates the da Silva-Fotheringham
#: correction for exactly these levels).
_MGWR_NOMINAL_ALPHAS = (0.1, 0.05, 0.001)


# ---------------------------------------------------------------------------
# Shared preparation
# ---------------------------------------------------------------------------


def _zscore(a: np.ndarray) -> np.ndarray:
    """Return ``a`` z-standardized column-wise (population std, matching mgwr docs)."""
    return (a - a.mean(axis=0)) / a.std(axis=0)


def _prepare_local(
    df: pd.DataFrame,
    outcome: str | None,
    covariates: str | Sequence[str] | None,
    *,
    gdf: gpd.GeoDataFrame,
    period: Any,
    entity: str | None,
    time: str | None,
    kernel: str,
    criterion: str,
    func: str,
) -> tuple[
    gpd.GeoDataFrame,
    str,
    str,
    str,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Any,
    list[str],
    dict[str, str],
]:
    """Validate and align the inputs shared by :func:`analyze_gwr` / :func:`analyze_mgwr`.

    Runs the standard validation cascade, aligns the requested cross-section to the
    geometry, and computes the calibration coordinates from metric-CRS centroids.

    Returns
    -------
    tuple
        ``(cross, gdf_entity, entity, outcome, covs, coords, y, x, period, notes,
        display)`` — the aligned GeoDataFrame cross-section, the geometry-side and
        data-side entity ids, the resolved outcome/covariates, the ``(n, 2)`` metric
        coordinates, the ``(n, 1)`` outcome and ``(n, k)`` covariate arrays, the
        resolved period, the accumulated notes, and the entity display-name map.
    """
    df = ensure_dataframe(df)
    gdf = ensure_geodataframe(gdf, func=func)
    outcome, covs = resolve_roles(df, outcome, covariates)
    if outcome is None:
        raise ValueError(
            f"{func}: an outcome is required — pass outcome=... or declare it via "
            "set_roles(df, outcome=...)"
        )
    if not covs:
        raise ValueError(
            f"{func}: at least one covariate is required — pass covariates=[...] or "
            "declare them via set_roles(df, covariates=...)"
        )
    entity, time = resolve_panel(df, entity, time, require_entity=True)
    assert entity is not None  # require_entity=True guarantees it

    cols = [outcome, *covs]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{func}: column(s) not found in df: {missing}")
    for c in cols:
        if not pdt.is_numeric_dtype(df[c]):
            raise TypeError(f"{func}: {c!r} needs to be numeric")

    if outcome in covs:
        raise ValueError(f"{func}: outcome {outcome!r} cannot also be a covariate")
    if len(set(covs)) != len(covs):
        raise ValueError(f"{func}: covariates contain duplicates: {list(covs)}")
    if kernel not in _KERNELS:
        raise ValueError(f"{func}: kernel {kernel!r} is not one of {list(_KERNELS)}")
    if criterion not in _CRITERIA:
        raise ValueError(
            f"{func}: criterion {criterion!r} is not one of {list(_CRITERIA)}"
        )

    cross, _, meta = _align_cross_section(
        df,
        gdf,
        cols,
        entity=entity,
        time=time,
        period=period,
        min_obs=len(cols) + 2,
        func=func,
    )
    notes = [str(n) for n in (meta.get("notes") or ())]
    resolved_period = meta.get("period")

    for c in cols:
        if float(cross[c].std(ddof=0)) == 0.0:
            raise ValueError(
                f"{func}: {c!r} has zero variance in the aligned cross-section"
            )

    n = len(cross)
    if n > _LARGE_N:
        message = (
            f"{n} units — the golden-section bandwidth search can be slow at this "
            "size; consider passing an explicit bandwidth (bw=...)"
        )
        warnings.warn(f"{func}: {message}", GeometricsWarning, stacklevel=3)
        notes.append(message)

    # Calibration coordinates: metric-CRS centroids, never lon/lat degrees.
    gdf_entity = resolve_gdf_entity(gdf)
    centroids = ensure_metric_crs(cross, func=func).geometry.centroid
    coords = np.column_stack(
        [centroids.x.to_numpy(dtype=float), centroids.y.to_numpy(dtype=float)]
    )
    y = cross[outcome].to_numpy(dtype=float).reshape(-1, 1)
    x = cross[list(covs)].to_numpy(dtype=float)
    display = entity_display_map(df, entity, resolve_entity_name(df))
    return (
        cross,
        gdf_entity,
        entity,
        outcome,
        list(covs),
        coords,
        y,
        x,
        resolved_period,
        notes,
        display,
    )


def _term_label(df: pd.DataFrame, term: str) -> str:
    """Return the display label for ``term`` (``"Intercept"`` for the constant)."""
    return "Intercept" if term == _CONST else resolve_label(df, term)


def _local_frame(
    entity: str,
    ids: Sequence[Any],
    terms: Sequence[str],
    params: np.ndarray,
    bse: np.ndarray,
    tvalues: np.ndarray,
    significant: np.ndarray,
    extra: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build the tidy one-row-per-entity local-coefficient frame."""
    data: dict[str, Any] = {entity: list(ids)}
    for j, term in enumerate(terms):
        data[f"{term}_coef"] = params[:, j]
        data[f"{term}_se"] = bse[:, j]
        data[f"{term}_t"] = tvalues[:, j]
    data.update(extra)
    for j, term in enumerate(terms):
        data[f"{term}_significant"] = significant[:, j]
    return pd.DataFrame(data)


def _coefficient_maps(
    cross: gpd.GeoDataFrame,
    gdf_entity: str,
    df: pd.DataFrame,
    terms: Sequence[str],
    params: np.ndarray,
    significant: np.ndarray,
    *,
    tiles: str | None,
    title: str | None,
    prefix: str,
    display: dict[str, str],
) -> dict[str, go.Figure]:
    """Build one diverging local-coefficient map per term (non-significant greyed)."""
    figs: dict[str, go.Figure] = {}
    for j, term in enumerate(terms):
        label = _term_label(df, term)
        fig_title = (
            f"{title} — {label}"
            if title is not None
            else f"{prefix} local coefficient: {label}"
        )
        figs[term] = continuous_map(
            cross,
            params[:, j],
            entity=gdf_entity,
            diverging=True,
            midpoint=0.0,
            mask=~significant[:, j],
            tiles=tiles,
            title=fig_title,
            hover_names=display,
            colorbar_title=f"{label} (local)",
        )
    return figs


# ---------------------------------------------------------------------------
# analyze_gwr
# ---------------------------------------------------------------------------


def _gwr_gt(
    outcome_label: str,
    *,
    bw: float,
    fixed: bool,
    kernel: str,
    bw_source: str,
    aicc: float,
    r2: float,
    enp: float,
    alpha: float,
    adj_alpha: float,
    critical_t: float,
    n: int,
) -> GT:
    """Render the global GWR summary as a Great Tables object."""
    from great_tables import GT

    bw_txt = (
        f"{bw:g} (fixed distance, metric CRS units)"
        if fixed
        else f"{bw:g} nearest neighbors (adaptive)"
    )
    disp = pd.DataFrame(
        {
            "Quantity": [
                "Bandwidth",
                "Kernel",
                "AICc",
                "R²",
                "Effective number of parameters (ENP)",
                f"Corrected alpha (nominal {alpha:g})",
                "Critical |t|",
                "Observations",
            ],
            "Value": [
                f"{bw_txt}, {bw_source}",
                kernel,
                f"{aicc:.4g}",
                f"{r2:.4g}",
                f"{enp:.4g}",
                f"{adj_alpha:.4g}",
                f"{critical_t:.4g}",
                f"{n:d}",
            ],
        }
    )
    return (
        GT(disp, rowname_col="Quantity")
        .tab_header(
            title=f"GWR: {outcome_label}",
            subtitle="geographically weighted regression (local model)",
        )
        .tab_source_note(
            "Local significance applies the da Silva & Fotheringham (2016) "
            "multiple-testing correction (corrected alpha → critical t)."
        )
    )


def analyze_gwr(
    df: pd.DataFrame,
    outcome: str | None = None,
    covariates: str | Sequence[str] | None = None,
    *,
    gdf: gpd.GeoDataFrame,
    period: Any = None,
    entity: str | None = None,
    time: str | None = None,
    bw: float | None = None,
    fixed: bool = False,
    kernel: str = "bisquare",
    criterion: str = "AICc",
    standardize: bool = False,
    alpha: float = 0.05,
    tiles: str | None = None,
    title: str | None = None,
) -> GWRResult:
    """Fit a geographically weighted regression and map each local surface.

    A separate distance-weighted regression of ``outcome`` on ``covariates`` is
    calibrated at every entity (mgwr's ``GWR``), so each term's coefficient becomes
    a local surface. The bandwidth is selected by golden-section search on
    ``criterion`` when ``bw`` is ``None``; local significance applies the
    da Silva & Fotheringham multiple-testing correction (corrected alpha →
    critical t) at the nominal ``alpha`` level, and non-significant units are
    greyed on the coefficient maps.

    Parameters
    ----------
    df
        Long panel (or cross section) holding the outcome and covariates per entity.
    outcome
        Numeric outcome column. Defaults to the outcome declared via
        :func:`geometrics.set_roles`.
    covariates
        Numeric covariate column(s). Default to the covariates declared via
        :func:`geometrics.set_roles`.
    gdf
        Entity geometry; must carry the same entity-id column as ``df``.
        Calibration coordinates are the polygon centroids in a metric CRS.
    period
        Period to analyze. Defaults to the latest period when ``df`` has a time
        dimension (a note records this).
    entity, time
        Panel identifiers; default to the ids declared via
        :func:`geometrics.set_panel`.
    bw
        Bandwidth: number of nearest neighbors when ``fixed=False`` (adaptive), a
        distance in metric-CRS units when ``fixed=True``. ``None`` (default)
        selects it by golden-section search on ``criterion``.
    fixed
        Use a fixed-distance kernel instead of an adaptive nearest-neighbor one.
    kernel
        Kernel weighting function: ``"bisquare"`` (default), ``"gaussian"`` or
        ``"exponential"``.
    criterion
        Bandwidth-selection criterion: ``"AICc"`` (default), ``"AIC"``, ``"BIC"``
        or ``"CV"``.
    standardize
        Z-standardize the outcome and covariates before fitting, so local
        coefficients are comparable across terms (a note records this).
    alpha
        Nominal significance level for the corrected local t-tests.
    tiles
        MapLibre base-map style for the coefficient maps, or ``None`` (default)
        for the vector backend (deterministic PNG export).
    title
        Title for the local-R² map; per-term maps append the term label.

    Returns
    -------
    GWRResult
        Frozen result with the per-entity local frame (``df``), one diverging
        coefficient map per term (``figs``), the local-R² map (``fig``), the
        global summary table (``gt``) and the fitted mgwr results (``model_obj``).

    Raises
    ------
    KeyError
        If the outcome or a covariate is not a column of ``df``.
    TypeError
        If a focal variable is not numeric, or ``df``/``gdf`` have the wrong type.
    ValueError
        If roles cannot be resolved, arguments are invalid, or too few complete
        observations remain after alignment.

    Examples
    --------
    A 3x3 lattice with an explicit bandwidth (skipping the search):

    ```python
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    from geometrics.gwr import analyze_gwr

    gdf = gpd.GeoDataFrame(
        {"cell": [f"c{i}" for i in range(9)]},
        geometry=[box(78 + i % 3, 20 + i // 3, 79 + i % 3, 21 + i // 3) for i in range(9)],
        crs="EPSG:4326",
    )
    df = pd.DataFrame(
        {
            "cell": [f"c{i}" for i in range(9)],
            "x1": [0.1, 0.9, 0.4, 0.7, 0.2, 0.8, 0.5, 0.3, 0.6],
            "y": [0.2, 1.4, 0.8, 1.5, 0.6, 2.0, 1.4, 1.0, 1.9],
        }
    )
    res = analyze_gwr(df, "y", ["x1"], gdf=gdf, entity="cell", bw=8, tiles=None)
    print(res.bw, sorted(res.figs))
    ```
    """
    (
        cross,
        gdf_entity,
        entity,
        outcome,
        covs,
        coords,
        y,
        x,
        resolved_period,
        notes,
        display,
    ) = _prepare_local(
        df,
        outcome,
        covariates,
        gdf=gdf,
        period=period,
        entity=entity,
        time=time,
        kernel=kernel,
        criterion=criterion,
        func=_GWR_FUNC,
    )
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"{_GWR_FUNC}: alpha needs to be in (0, 1), got {alpha!r}")
    if bw is not None and float(bw) <= 0:
        raise ValueError(f"{_GWR_FUNC}: bw needs to be positive, got {bw!r}")

    if standardize:
        y = _zscore(y)
        x = _zscore(x)
        notes.append(
            "outcome and covariates were z-standardized — local coefficients are on "
            "the standardized scale"
        )

    from mgwr.gwr import GWR

    if bw is None:
        from mgwr.sel_bw import Sel_BW

        selector = Sel_BW(coords, y, x, fixed=fixed, kernel=kernel)
        with contextlib.redirect_stdout(io.StringIO()):
            bw_value = float(selector.search(criterion=criterion))
        bw_source = f"selected by {criterion} golden-section search"
    else:
        bw_value = float(bw)
        bw_source = "user-specified"
    notes.append(f"bandwidth {bw_value:g} ({bw_source})")

    model = GWR(coords, y, x, bw_value, fixed=fixed, kernel=kernel)
    with contextlib.redirect_stdout(io.StringIO()):
        results = model.fit()

    # Multiple-testing correction: GWRResults.adj_alpha is a cached property whose
    # rows tabulate the da Silva-Fotheringham corrected alpha at the nominal levels
    # in _MGWR_NOMINAL_ALPHAS; for other levels the same alpha*k/ENP formula applies.
    adj_rows = np.asarray(results.adj_alpha, dtype=float)
    matched = np.flatnonzero(np.isclose(np.asarray(_MGWR_NOMINAL_ALPHAS), alpha))
    if matched.size:
        adj_alpha = float(adj_rows[int(matched[0])])
    else:
        adj_alpha = float(alpha) * float(results.k) / float(results.ENP)
    critical_t = float(results.critical_tval(alpha=adj_alpha))

    terms = [_CONST, *covs]
    params = np.asarray(results.params, dtype=float)
    bse = np.asarray(results.bse, dtype=float)
    tvalues = np.asarray(results.tvalues, dtype=float)
    significant = np.abs(tvalues) >= critical_t
    local_r2 = np.asarray(results.localR2, dtype=float).ravel()

    tidy = _local_frame(
        entity,
        list(cross[gdf_entity]),
        terms,
        params,
        bse,
        tvalues,
        significant,
        {"local_r2": local_r2},
    )

    outcome_label = resolve_label(df, outcome)
    figs = _coefficient_maps(
        cross,
        gdf_entity,
        df,
        terms,
        params,
        significant,
        tiles=tiles,
        title=title,
        prefix="GWR",
        display=display,
    )
    fig = continuous_map(
        cross,
        local_r2,
        entity=gdf_entity,
        tiles=tiles,
        title=title if title is not None else f"GWR local R²: {outcome_label}",
        hover_names=display,
        colorbar_title="local R²",
    )
    gt = _gwr_gt(
        outcome_label,
        bw=bw_value,
        fixed=fixed,
        kernel=kernel,
        bw_source=bw_source,
        aicc=float(results.aicc),
        r2=float(results.R2),
        enp=float(results.ENP),
        alpha=float(alpha),
        adj_alpha=adj_alpha,
        critical_t=critical_t,
        n=len(cross),
    )
    return GWRResult(
        df=tidy,
        figs=figs,
        fig=fig,
        gt=gt,
        bw=bw_value,
        fixed=fixed,
        kernel=kernel,
        aicc=float(results.aicc),
        r2=float(results.R2),
        adj_alpha=adj_alpha,
        critical_t=critical_t,
        model_obj=results,
        outcome=outcome,
        covariates=tuple(covs),
        period=resolved_period,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# analyze_mgwr
# ---------------------------------------------------------------------------


def _mgwr_gt(
    outcome_label: str, *, kernel: str, criterion: str, aicc: float, r2: float, n: int
) -> GT:
    """Render the global MGWR summary as a Great Tables object."""
    from great_tables import GT

    disp = pd.DataFrame(
        {
            "Quantity": [
                "Kernel",
                "Bandwidth criterion",
                "AICc",
                "R²",
                "Observations",
            ],
            "Value": [kernel, criterion, f"{aicc:.4g}", f"{r2:.4g}", f"{n:d}"],
        }
    )
    return (
        GT(disp, rowname_col="Quantity")
        .tab_header(
            title=f"MGWR: {outcome_label}",
            subtitle="multiscale GWR — variables z-standardized",
        )
        .tab_source_note(
            "Each term operates at its own bandwidth (see the bandwidth table); "
            "local significance uses per-term corrected alphas."
        )
    )


def _mgwr_bw_gt(
    df: pd.DataFrame,
    terms: Sequence[str],
    bw: dict[str, float],
    enp_j: np.ndarray,
    adj_alpha: dict[str, float],
    critical_t: dict[str, float],
    n: int,
) -> GT:
    """Render the per-term MGWR bandwidth table as a Great Tables object."""
    from great_tables import GT

    disp = pd.DataFrame(
        {
            "Term": [_term_label(df, t) for t in terms],
            "Bandwidth": [f"{bw[t]:g}" for t in terms],
            "ENP": [f"{float(e):.4g}" for e in enp_j],
            "Corrected alpha": [f"{adj_alpha[t]:.4g}" for t in terms],
            "Critical |t|": [f"{critical_t[t]:.4g}" for t in terms],
        }
    )
    return (
        GT(disp, rowname_col="Term")
        .tab_header(
            title="MGWR bandwidths",
            subtitle=f"adaptive nearest-neighbor bandwidths out of {n} units",
        )
        .tab_source_note(
            "Smaller bandwidths mean the association varies over shorter distances "
            "(a more local process); a bandwidth near n is effectively global."
        )
    )


def analyze_mgwr(
    df: pd.DataFrame,
    outcome: str | None = None,
    covariates: str | Sequence[str] | None = None,
    *,
    gdf: gpd.GeoDataFrame,
    period: Any = None,
    entity: str | None = None,
    time: str | None = None,
    kernel: str = "bisquare",
    criterion: str = "AICc",
    max_iter: int = 200,
    tiles: str | None = None,
    title: str | None = None,
) -> MGWRResult:
    """Fit a multiscale GWR: every term gets its own spatial scale (bandwidth).

    MGWR relaxes GWR's single shared bandwidth: a backfitting algorithm (mgwr's
    ``Sel_BW(multi=True)`` + ``MGWR``) selects one adaptive bandwidth per term, so
    each covariate's association can vary at its own spatial scale. Following the
    MGWR requirement, the outcome and covariates are **always z-standardized**, so
    local coefficients are on the standardized scale (a note records this). Local
    significance applies per-term da Silva & Fotheringham corrected alphas at the
    5% nominal level.

    Parameters
    ----------
    df
        Long panel (or cross section) holding the outcome and covariates per entity.
    outcome
        Numeric outcome column. Defaults to the outcome declared via
        :func:`geometrics.set_roles`.
    covariates
        Numeric covariate column(s). Default to the covariates declared via
        :func:`geometrics.set_roles`.
    gdf
        Entity geometry; must carry the same entity-id column as ``df``.
        Calibration coordinates are the polygon centroids in a metric CRS.
    period
        Period to analyze. Defaults to the latest period when ``df`` has a time
        dimension (a note records this).
    entity, time
        Panel identifiers; default to the ids declared via
        :func:`geometrics.set_panel`.
    kernel
        Kernel weighting function: ``"bisquare"`` (default), ``"gaussian"`` or
        ``"exponential"``.
    criterion
        Bandwidth-selection criterion: ``"AICc"`` (default), ``"AIC"``, ``"BIC"``
        or ``"CV"``.
    max_iter
        Maximum number of multiscale backfitting iterations.
    tiles
        MapLibre base-map style for the coefficient maps, or ``None`` (default)
        for the vector backend (deterministic PNG export).
    title
        Title for the residual map; per-term maps append the term label.

    Returns
    -------
    MGWRResult
        Frozen result with the per-entity local frame (``df``), one diverging
        coefficient map per term (``figs``), the residual map (``fig`` — mgwr does
        not define a local R² under multiple bandwidths), the summary and
        bandwidth tables (``gt`` / ``gt_bw``), the per-term ``bw`` /
        ``adj_alpha`` / ``critical_t`` dicts and the fitted mgwr results
        (``model_obj``).

    Raises
    ------
    KeyError
        If the outcome or a covariate is not a column of ``df``.
    TypeError
        If a focal variable is not numeric, or ``df``/``gdf`` have the wrong type.
    ValueError
        If roles cannot be resolved, arguments are invalid, or too few complete
        observations remain after alignment.

    Examples
    --------
    A 3x3 lattice cross-section (bandwidths selected by backfitting):

    ```python
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    from geometrics.gwr import analyze_mgwr

    gdf = gpd.GeoDataFrame(
        {"cell": [f"c{i}" for i in range(9)]},
        geometry=[box(78 + i % 3, 20 + i // 3, 79 + i % 3, 21 + i // 3) for i in range(9)],
        crs="EPSG:4326",
    )
    df = pd.DataFrame(
        {
            "cell": [f"c{i}" for i in range(9)],
            "x1": [0.1, 0.9, 0.4, 0.7, 0.2, 0.8, 0.5, 0.3, 0.6],
            "y": [0.2, 1.4, 0.8, 1.5, 0.6, 2.0, 1.4, 1.0, 1.9],
        }
    )
    res = analyze_mgwr(df, "y", ["x1"], gdf=gdf, entity="cell", tiles=None)
    print(sorted(res.bw))
    ```
    """
    (
        cross,
        gdf_entity,
        entity,
        outcome,
        covs,
        coords,
        y,
        x,
        resolved_period,
        notes,
        display,
    ) = _prepare_local(
        df,
        outcome,
        covariates,
        gdf=gdf,
        period=period,
        entity=entity,
        time=time,
        kernel=kernel,
        criterion=criterion,
        func=_MGWR_FUNC,
    )
    if int(max_iter) < 1:
        raise ValueError(
            f"{_MGWR_FUNC}: max_iter needs to be at least 1, got {max_iter!r}"
        )

    # MGWR requires standardized variables (Fotheringham, Yang & Kang 2017).
    y = _zscore(y)
    x = _zscore(x)
    notes.append(
        "MGWR requires z-standardized variables — the outcome and covariates were "
        "standardized, so local coefficients are on the standardized scale"
    )

    from mgwr.gwr import MGWR
    from mgwr.sel_bw import Sel_BW

    selector = Sel_BW(coords, y, x, multi=True, fixed=False, kernel=kernel)
    with contextlib.redirect_stdout(io.StringIO()):
        bws = np.asarray(
            selector.search(criterion=criterion, max_iter_multi=int(max_iter)),
            dtype=float,
        )
        results = MGWR(coords, y, x, selector, fixed=False, kernel=kernel).fit()

    terms = [_CONST, *covs]
    bw = {term: float(b) for term, b in zip(terms, bws, strict=True)}
    notes.append(
        "per-term bandwidths selected by multiscale backfitting: "
        + ", ".join(f"{t}={bw[t]:g}" for t in terms)
    )

    # Per-term multiple-testing correction: MGWRResults.adj_alpha_j rows are terms,
    # columns the nominal levels in _MGWR_NOMINAL_ALPHAS; critical_tval() returns
    # the per-term critical t at the 5% nominal level (column 1).
    adj_alpha_j = np.asarray(results.adj_alpha_j, dtype=float)[:, 1]
    critical_j = np.asarray(results.critical_tval(), dtype=float).ravel()
    adj_alpha = {t: float(a) for t, a in zip(terms, adj_alpha_j, strict=True)}
    critical_t = {t: float(c) for t, c in zip(terms, critical_j, strict=True)}

    params = np.asarray(results.params, dtype=float)
    bse = np.asarray(results.bse, dtype=float)
    tvalues = np.asarray(results.tvalues, dtype=float)
    significant = np.abs(tvalues) >= critical_j[np.newaxis, :]
    residuals = np.asarray(results.resid_response, dtype=float).ravel()

    tidy = _local_frame(
        entity,
        list(cross[gdf_entity]),
        terms,
        params,
        bse,
        tvalues,
        significant,
        {"residual": residuals},
    )

    outcome_label = resolve_label(df, outcome)
    figs = _coefficient_maps(
        cross,
        gdf_entity,
        df,
        terms,
        params,
        significant,
        tiles=tiles,
        title=title,
        prefix="MGWR",
        display=display,
    )
    # mgwr's localR2 is not implemented for multiple bandwidths, so the headline
    # map shows the residual surface instead (standardized scale, diverging at 0).
    fig = continuous_map(
        cross,
        residuals,
        entity=gdf_entity,
        diverging=True,
        midpoint=0.0,
        tiles=tiles,
        title=(
            title
            if title is not None
            else f"MGWR residuals: {outcome_label} (standardized scale)"
        ),
        hover_names=display,
        colorbar_title="residual",
    )
    gt = _mgwr_gt(
        outcome_label,
        kernel=kernel,
        criterion=criterion,
        aicc=float(results.aicc),
        r2=float(results.R2),
        n=len(cross),
    )
    gt_bw = _mgwr_bw_gt(
        df,
        terms,
        bw,
        np.asarray(results.ENP_j, dtype=float),
        adj_alpha,
        critical_t,
        len(cross),
    )
    return MGWRResult(
        df=tidy,
        figs=figs,
        fig=fig,
        gt=gt,
        gt_bw=gt_bw,
        bw=bw,
        kernel=kernel,
        aicc=float(results.aicc),
        r2=float(results.R2),
        adj_alpha=adj_alpha,
        critical_t=critical_t,
        model_obj=results,
        outcome=outcome,
        covariates=tuple(covs),
        period=resolved_period,
        notes=tuple(notes),
    )
