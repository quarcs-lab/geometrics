"""Global and local spatial autocorrelation: Moran scatterplot, LISA map, Moran over time.

Three ESDA views of spatial dependence in one variable:

* :func:`explore_moran_plot` — the Moran scatterplot with global Moran's I. Under
  row-standardized weights the OLS slope of the spatial lag on the (standardized)
  variable *is* Moran's I, so the fitted line makes the statistic visible.
* :func:`explore_lisa_cluster_map` — Local Moran statistics (LISA) mapped as the
  GeoDa-style cluster map (High-High, Low-Low, Low-High, High-Low, Not significant),
  alongside the cluster-colored Moran scatterplot.
* :func:`explore_moran_over_time` — global Moran's I recomputed period by period on a
  fixed entity set, tracking how spatial dependence evolves.

All three align data, geometry and weights through
:func:`geometrics._geo._align_cross_section` (or its panel-wide logic), so W rows and
data rows can never fall out of order.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt

from geometrics._common import entity_display_map
from geometrics._geo import _align_cross_section, _first_ids, resolve_gdf_entity
from geometrics._labels import resolve_label
from geometrics._mapping import categorical_map
from geometrics._panel import resolve_entity_name, resolve_panel
from geometrics._theme import LISA_COLORS, apply_default_layout, color_for
from geometrics._types import (
    LisaClusterMapResult,
    MoranOverTimeResult,
    MoranPlotResult,
)
from geometrics._validation import (
    GeometricsWarning,
    ensure_dataframe,
    ensure_geodataframe,
)
from geometrics.weights import _default_weights

if TYPE_CHECKING:
    import geopandas as gpd
    from libpysal.weights import W

__all__ = [
    "explore_lisa_cluster_map",
    "explore_moran_over_time",
    "explore_moran_plot",
]

# esda's Moran_Local quadrant codes -> short labels (1=HH, 2=LH, 3=LL, 4=HL).
_QUADRANT_SHORT = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
# Short quadrant labels -> the fixed LISA cluster names (and thus colors).
_QUADRANT_FULL = {
    "HH": "High-High",
    "LH": "Low-High",
    "LL": "Low-Low",
    "HL": "High-Low",
}
# Quadrant colors reuse the ecosystem-fixed LISA palette so the scatter and the
# cluster map read identically (label-to-color is fixed, never cycled).
_QUADRANT_COLORS = {short: LISA_COLORS[full] for short, full in _QUADRANT_FULL.items()}
# Legend/trace order of the LISA cluster map (clusters first, outliers, then grey).
_LISA_ORDER = ("High-High", "Low-Low", "Low-High", "High-Low", "Not significant")

_REFLINE = {"dash": "dash", "color": "rgba(0,0,0,0.35)", "width": 1}


def _w_spec(w: W) -> str:
    """Return ``w.geometrics_meta['spec']``, or compose a short human description."""
    meta = dict(getattr(w, "geometrics_meta", {}) or {})
    spec = meta.get("spec")
    if spec:
        return str(spec)
    standardized = ", row-standardized" if str(w.transform).upper() == "R" else ""
    return (
        f"user-supplied W (mean {float(w.mean_neighbors):.2f} "
        f"neighbors{standardized}), n={w.n}"
    )


def _validate_var(df: pd.DataFrame, var: str, *, func: str) -> None:
    """Raise ``KeyError`` for a missing ``var`` and ``TypeError`` for a non-numeric one."""
    if var not in df.columns:
        raise KeyError(f"{func}: column {var!r} not found in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"{func}: var {var!r} needs to be numeric")


def _validate_permutations(permutations: int, *, func: str) -> None:
    """Raise ``ValueError`` unless at least one conditional permutation is requested."""
    if int(permutations) < 1:
        raise ValueError(
            f"{func}: permutations needs to be >= 1 to compute a pseudo p-value"
        )


def _zscore(y: np.ndarray, var: str, *, func: str) -> np.ndarray:
    """Standardize ``y`` to zero mean and unit variance (``ValueError`` on zero spread)."""
    std = float(y.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        raise ValueError(
            f"{func}: {var!r} has zero variance in the aligned cross-section — "
            "spatial autocorrelation is undefined for a constant"
        )
    return (y - float(y.mean())) / std


def _note_islands(w: W, notes: list[str], *, func: str, extra: str = "") -> None:
    """Warn (and record) when ``w`` leaves units with no neighbors at all.

    An island's spatial lag is zero by convention, and a zero row also breaks the
    exact identity between the Moran scatter's OLS slope and Moran's I (the row can
    no longer sum to one).
    """
    islands = list(w.islands)
    if not islands:
        return
    msg = (
        f"{func}: {len(islands)} unit(s) have no neighbors under the supplied "
        f"weights (e.g. {_first_ids(islands)}) — their spatial lag is zero by "
        f"convention{extra}"
    )
    warnings.warn(msg, GeometricsWarning, stacklevel=3)
    notes.append(msg)


def _quadrant_codes(z: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """Return Moran-scatter quadrant codes (1=HH, 2=LH, 3=LL, 4=HL) by sign of z and Wz."""
    high_z = z > 0
    high_lag = lag > 0
    return np.where(
        high_z & high_lag,
        1,
        np.where(~high_z & high_lag, 2, np.where(~high_z & ~high_lag, 3, 4)),
    )


def _moran_scatter(
    *,
    z: np.ndarray,
    lag: np.ndarray,
    labels: Sequence[str],
    categories: Sequence[str],
    colors: Mapping[str, str],
    display: Sequence[str],
    moran_i: float,
    expected_i: float,
    p_sim: float,
    var_label: str,
    w_spec: str,
    title: str,
    legend_title: str,
) -> go.Figure:
    """Build the themed Moran scatterplot: category-colored points + OLS fit line.

    One trace per category (fixed label-to-color, empty categories drawn for stable
    legends), the OLS fit of ``lag`` on ``z`` (whose slope equals Moran's I under
    row-standardized weights), dashed zero reference lines, and a stat-box
    annotation reporting I, E[I] and the permutation pseudo p-value.
    """
    fig = go.Figure()
    for cat in categories:
        members = [i for i, lab in enumerate(labels) if lab == cat]
        fig.add_trace(
            go.Scatter(
                x=[float(z[i]) for i in members],
                y=[float(lag[i]) for i in members],
                mode="markers",
                name=cat,
                marker={
                    "color": colors.get(cat, color_for(0)),
                    "size": 9,
                    "line": {"color": "white", "width": 0.5},
                },
                customdata=[
                    [display[i], float(z[i]), float(lag[i]), cat] for i in members
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{var_label} (z): %{{customdata[1]:.3f}}<br>"
                    "spatial lag (z): %{customdata[2]:.3f}<br>"
                    "%{customdata[3]}<extra></extra>"
                ),
            )
        )
    slope, intercept = (float(c) for c in np.polyfit(z, lag, 1))
    xs = np.array([float(z.min()), float(z.max())])
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=intercept + slope * xs,
            mode="lines",
            line={"color": "#2a2a2a", "width": 2},
            name="OLS fit",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_hline(y=0.0, line=dict(_REFLINE))
    fig.add_vline(x=0.0, line=dict(_REFLINE))
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        text=(
            f"Moran's I = {moran_i:.3f}<br>"
            f"E[I] = {expected_i:.3f}<br>"
            f"p (perm) = {p_sim:.3f}"
        ),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
    )
    apply_default_layout(
        fig,
        title=title,
        subtitle=w_spec,
        xaxis={"title": f"{var_label} (z-score)"},
        yaxis={"title": f"Spatial lag of {var_label} (z-score)"},
        legend_title_text=legend_title,
    )
    return fig


def explore_moran_plot(
    df: pd.DataFrame,
    var: str,
    *,
    gdf: gpd.GeoDataFrame,
    w: W | None = None,
    period: Any = None,
    entity: str | None = None,
    time: str | None = None,
    permutations: int = 999,
    seed: int | None = 12345,
    title: str | None = None,
) -> MoranPlotResult:
    """Draw the Moran scatterplot and test global spatial autocorrelation in ``var``.

    The panel is aligned to the geometry for one cross-section (the latest period by
    default), the variable is z-standardized, and its row-standardized spatial lag is
    plotted against it, colored by scatter quadrant (HH, LH, LL, HL). The OLS slope of
    the fitted line equals global Moran's I under row-standardized weights, whose
    significance is assessed with ``permutations`` conditional permutations
    (:class:`esda.moran.Moran`).

    Parameters
    ----------
    df
        Long panel (or cross-section) holding ``var`` per entity.
    var
        Numeric column of ``df`` to test.
    gdf
        Entity geometry; must carry the same entity-id column as ``df``.
    w
        ``libpysal`` weights aligned to the gdf entity ids. ``None`` builds the
        default weights (queen contiguity for polygons, 6-nearest-neighbor
        otherwise) with a :class:`~geometrics.GeometricsWarning`. esda
        row-standardizes the weights for the statistic (its ``transformation="r"``
        convention), which is also what makes the scatter slope equal Moran's I.
    period
        Period to analyze. Defaults to the latest period when ``df`` has a time
        dimension (a note records this).
    entity, time
        Panel identifiers; default to the ids declared via
        :func:`geometrics.set_panel`.
    permutations
        Number of conditional permutations behind ``p_sim`` / ``z_sim``.
    seed
        esda's global :class:`~esda.moran.Moran` draws its permutations from NumPy's
        **global** random state and exposes no seed argument, so when ``seed`` is not
        ``None`` geometrics calls ``numpy.random.seed(seed)`` immediately before the
        test to make ``p_sim`` reproducible. Pass ``None`` to leave the global state
        untouched.
    title
        Figure title. Defaults to ``"Moran scatterplot: <label> (<period>)"``.

    Returns
    -------
    MoranPlotResult
        Frozen result with ``df`` (``entity``, standardized ``value``, spatial
        ``lag``, ``quadrant``), the quadrant-colored scatter ``fig``, the global
        Moran scalars (``moran_i``, ``expected_i``, ``p_sim``, ``z_sim``) and
        ``w_spec``.

    Examples
    --------
    Moran's I on a four-cell strip where value increases smoothly west to east:

    ```python
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    from geometrics.dependence import explore_moran_plot
    from geometrics.weights import make_weights

    gdf = gpd.GeoDataFrame(
        {"region": ["a", "b", "c", "d"]},
        geometry=[box(i, 0, i + 1, 1) for i in range(4)],
        crs="EPSG:4326",
    )
    df = pd.DataFrame({"region": ["a", "b", "c", "d"], "gdppc": [1.0, 2.0, 3.0, 4.0]})
    res = explore_moran_plot(
        df, "gdppc", gdf=gdf, w=make_weights(gdf), entity="region", permutations=99
    )
    print(res.df["quadrant"].tolist(), round(res.moran_i, 3))
    ```
    """
    func = "explore_moran_plot"
    df = ensure_dataframe(df)
    gdf = ensure_geodataframe(gdf, func=func)
    entity, time = resolve_panel(df, entity, time, require_entity=True)
    assert entity is not None  # require_entity=True guarantees it
    _validate_var(df, var, func=func)
    _validate_permutations(permutations, func=func)

    notes: list[str] = []
    if w is None:
        w = _default_weights(gdf, func=func)
        notes.append(
            f"{func}: no weights supplied — defaulted to {w.geometrics_meta['spec']}"
        )

    cross, w_aligned, info = _align_cross_section(
        df,
        gdf,
        [var],
        entity=entity,
        time=time,
        period=period,
        w=w,
        min_obs=3,
        func=func,
    )
    assert w_aligned is not None  # w is never None past the default above
    resolved_period = info.get("period")
    notes.extend(str(n) for n in info.get("notes") or ())
    _note_islands(
        w_aligned,
        notes,
        func=func,
        extra=" and the scatter's OLS slope no longer equals Moran's I exactly",
    )

    from esda.moran import Moran

    y = cross[var].to_numpy(dtype=float)
    z = _zscore(y, var, func=func)
    if seed is not None:
        np.random.seed(seed)
    mi = Moran(y, w_aligned, permutations=int(permutations))

    from libpysal.weights import lag_spatial

    lag = np.asarray(lag_spatial(w_aligned, z), dtype=float)
    quadrant = [_QUADRANT_SHORT[int(q)] for q in _quadrant_codes(z, lag)]

    gdf_entity = resolve_gdf_entity(gdf)
    display_map = entity_display_map(df, entity, resolve_entity_name(df))
    ids = cross[gdf_entity].to_numpy()
    display = [display_map.get(str(u), str(u)) for u in ids]

    tidy = pd.DataFrame({"entity": ids, "value": z, "lag": lag, "quadrant": quadrant})
    w_spec = _w_spec(w_aligned)
    var_label = resolve_label(df, var)
    if title is None:
        title = f"Moran scatterplot: {var_label}" + (
            f" ({resolved_period})" if resolved_period is not None else ""
        )
    fig = _moran_scatter(
        z=z,
        lag=lag,
        labels=quadrant,
        categories=tuple(_QUADRANT_FULL),
        colors=_QUADRANT_COLORS,
        display=display,
        moran_i=float(mi.I),
        expected_i=float(mi.EI),
        p_sim=float(mi.p_sim),
        var_label=var_label,
        w_spec=w_spec,
        title=title,
        legend_title="Quadrant",
    )
    return MoranPlotResult(
        df=tidy,
        fig=fig,
        moran_i=float(mi.I),
        expected_i=float(mi.EI),
        p_sim=float(mi.p_sim),
        z_sim=float(mi.z_sim),
        permutations=int(permutations),
        var=var,
        period=resolved_period,
        w_spec=w_spec,
        notes=tuple(notes),
    )


def explore_lisa_cluster_map(
    df: pd.DataFrame,
    var: str,
    *,
    gdf: gpd.GeoDataFrame,
    w: W | None = None,
    period: Any = None,
    entity: str | None = None,
    time: str | None = None,
    permutations: int = 999,
    seed: int | None = 12345,
    alpha: float = 0.05,
    tiles: str | None = "carto-positron",
    title: str | None = None,
) -> LisaClusterMapResult:
    """Map local Moran (LISA) clusters of ``var`` and the matching Moran scatterplot.

    :class:`esda.moran.Moran_Local` assigns each entity a scatter quadrant (via its
    ``q``: 1=HH, 2=LH, 3=LL, 4=HL) and a permutation pseudo p-value; entities with
    ``p_sim < alpha`` receive their quadrant's cluster label (High-High hot spots,
    Low-Low cold spots, Low-High / High-Low spatial outliers) and everything else is
    ``"Not significant"``. The cluster map uses the ecosystem-fixed LISA colors
    (GeoDa / splot convention); ``fig_scatter`` is the same Moran scatter as
    :func:`explore_moran_plot`, colored by cluster. Global Moran's I accompanies the
    local statistics (``moran_i``, ``p_sim_global``).

    Parameters
    ----------
    df
        Long panel (or cross-section) holding ``var`` per entity.
    var
        Numeric column of ``df`` to analyze.
    gdf
        Entity geometry; must carry the same entity-id column as ``df``.
    w
        ``libpysal`` weights aligned to the gdf entity ids. ``None`` builds the
        default weights (queen contiguity for polygons, 6-nearest-neighbor
        otherwise) with a :class:`~geometrics.GeometricsWarning`.
    period
        Period to analyze. Defaults to the latest period when ``df`` has a time
        dimension (a note records this).
    entity, time
        Panel identifiers; default to the ids declared via
        :func:`geometrics.set_panel`.
    permutations
        Number of conditional permutations behind the local and global pseudo
        p-values.
    seed
        Reproducibility seed. ``Moran_Local`` accepts it directly; the global
        :class:`~esda.moran.Moran` has no seed argument, so ``numpy.random.seed(seed)``
        is set immediately before it (see :func:`explore_moran_plot`). ``None``
        leaves the random state untouched.
    alpha
        Significance level masking the cluster labels (``p_sim < alpha``).
    tiles
        MapLibre base-map style for the cluster map (default ``"carto-positron"``)
        or ``None`` for the vector backend (deterministic PNG export).
    title
        Cluster-map title. Defaults to ``"LISA clusters: <label> (<period>)"``; the
        scatter always uses its own composed title.

    Returns
    -------
    LisaClusterMapResult
        Frozen result with the per-entity frame (``entity``, standardized ``value``,
        ``lag``, ``local_i``, ``quadrant``, ``p_sim``, ``cluster``), the cluster map
        ``fig``, the cluster-colored ``fig_scatter``, the global test scalars, the
        per-class counts and ``w_spec``.

    Examples
    --------
    LISA on a four-cell strip (tiny n, so nothing is significant at 5%):

    ```python
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    from geometrics.dependence import explore_lisa_cluster_map
    from geometrics.weights import make_weights

    gdf = gpd.GeoDataFrame(
        {"region": ["a", "b", "c", "d"]},
        geometry=[box(i, 0, i + 1, 1) for i in range(4)],
        crs="EPSG:4326",
    )
    df = pd.DataFrame({"region": ["a", "b", "c", "d"], "gdppc": [1.0, 2.0, 3.0, 4.0]})
    res = explore_lisa_cluster_map(
        df, "gdppc", gdf=gdf, w=make_weights(gdf), entity="region",
        permutations=99, tiles=None,
    )
    print(res.df["cluster"].tolist())
    ```
    """
    func = "explore_lisa_cluster_map"
    df = ensure_dataframe(df)
    gdf = ensure_geodataframe(gdf, func=func)
    entity, time = resolve_panel(df, entity, time, require_entity=True)
    assert entity is not None  # require_entity=True guarantees it
    _validate_var(df, var, func=func)
    _validate_permutations(permutations, func=func)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"{func}: alpha needs to be in (0, 1), got {alpha!r}")

    notes: list[str] = []
    if w is None:
        w = _default_weights(gdf, func=func)
        notes.append(
            f"{func}: no weights supplied — defaulted to {w.geometrics_meta['spec']}"
        )

    cross, w_aligned, info = _align_cross_section(
        df,
        gdf,
        [var],
        entity=entity,
        time=time,
        period=period,
        w=w,
        min_obs=3,
        func=func,
    )
    assert w_aligned is not None  # w is never None past the default above
    resolved_period = info.get("period")
    notes.extend(str(n) for n in info.get("notes") or ())
    _note_islands(w_aligned, notes, func=func)

    from esda.moran import Moran, Moran_Local

    y = cross[var].to_numpy(dtype=float)
    z = _zscore(y, var, func=func)
    if seed is not None:
        np.random.seed(seed)
    mi = Moran(y, w_aligned, permutations=int(permutations))
    # np.errstate: an island's simulated local statistics are all zero, making
    # esda's z_sim a harmless 0/0 for that unit — silence the numpy warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        lisa = Moran_Local(y, w_aligned, permutations=int(permutations), seed=seed)

    from libpysal.weights import lag_spatial

    lag = np.asarray(lag_spatial(w_aligned, z), dtype=float)
    quad_codes = np.asarray(lisa.q, dtype=int)
    quadrant = [_QUADRANT_SHORT[int(q)] for q in quad_codes]
    p_sim = np.asarray(lisa.p_sim, dtype=float)
    cluster = [
        _QUADRANT_FULL[qs] if p < float(alpha) else "Not significant"
        for qs, p in zip(quadrant, p_sim, strict=True)
    ]
    counts = {label: cluster.count(label) for label in _LISA_ORDER}

    gdf_entity = resolve_gdf_entity(gdf)
    display_map = entity_display_map(df, entity, resolve_entity_name(df))
    ids = cross[gdf_entity].to_numpy()
    display = [display_map.get(str(u), str(u)) for u in ids]

    tidy = pd.DataFrame(
        {
            "entity": ids,
            "value": z,
            "lag": lag,
            "local_i": np.asarray(lisa.Is, dtype=float),
            "quadrant": quadrant,
            "p_sim": p_sim,
            "cluster": cluster,
        }
    )
    w_spec = _w_spec(w_aligned)
    var_label = resolve_label(df, var)
    when = f" ({resolved_period})" if resolved_period is not None else ""
    if title is None:
        title = f"LISA clusters: {var_label}{when}"
    fig = categorical_map(
        cross,
        cluster,
        entity=gdf_entity,
        colors=LISA_COLORS,
        category_order=list(_LISA_ORDER),
        tiles=tiles,
        hover_names=display_map,
    )
    apply_default_layout(
        fig,
        title=title,
        subtitle=f"{w_spec}; p < {float(alpha):g} on {int(permutations)} permutations",
    )
    fig_scatter = _moran_scatter(
        z=z,
        lag=lag,
        labels=cluster,
        categories=_LISA_ORDER,
        colors=LISA_COLORS,
        display=display,
        moran_i=float(mi.I),
        expected_i=float(mi.EI),
        p_sim=float(mi.p_sim),
        var_label=var_label,
        w_spec=w_spec,
        title=f"Moran scatterplot: {var_label}{when}",
        legend_title="Cluster",
    )
    return LisaClusterMapResult(
        df=tidy,
        fig=fig,
        fig_scatter=fig_scatter,
        moran_i=float(mi.I),
        p_sim_global=float(mi.p_sim),
        alpha=float(alpha),
        permutations=int(permutations),
        n_hh=counts["High-High"],
        n_ll=counts["Low-Low"],
        n_hl=counts["High-Low"],
        n_lh=counts["Low-High"],
        n_ns=counts["Not significant"],
        var=var,
        period=resolved_period,
        w_spec=w_spec,
        notes=tuple(notes),
    )


def explore_moran_over_time(
    df: pd.DataFrame,
    var: str,
    *,
    gdf: gpd.GeoDataFrame,
    w: W | None = None,
    entity: str | None = None,
    time: str | None = None,
    permutations: int = 999,
    seed: int | None = 12345,
    title: str | None = None,
) -> MoranOverTimeResult:
    """Track global Moran's I in ``var`` period by period on a fixed entity set.

    The long panel is pivoted to one row per entity and one column per period, and a
    **single** entity set — the entities with complete data across every kept period —
    is used throughout, so the same (possibly restricted) weights apply to every
    period and the Moran's I values are comparable over time. Periods with no data at
    all and entities with incomplete series are dropped with a note. Each period's
    test uses ``permutations`` conditional permutations (:class:`esda.moran.Moran`).

    Parameters
    ----------
    df
        Long panel holding ``var`` per (entity, period).
    var
        Numeric column of ``df`` to track.
    gdf
        Entity geometry; must carry the same entity-id column as ``df``.
    w
        ``libpysal`` weights aligned to the gdf entity ids. ``None`` builds the
        default weights (queen contiguity for polygons, 6-nearest-neighbor
        otherwise) with a :class:`~geometrics.GeometricsWarning`. When entities drop,
        the weights are restricted (``w_subset``) and their transform re-applied.
    entity, time
        Panel identifiers; default to the ids declared via
        :func:`geometrics.set_panel`. Both are required here.
    permutations
        Number of conditional permutations behind each period's ``p_sim`` / ``z_sim``.
    seed
        esda's :class:`~esda.moran.Moran` draws its permutations from NumPy's
        **global** random state and exposes no seed argument, so when ``seed`` is not
        ``None`` geometrics calls ``numpy.random.seed(seed)`` immediately before
        *each* period's test — every period's p-value is then reproducible on its
        own, independent of which other periods are present. ``None`` leaves the
        random state untouched.
    title
        Figure title. Defaults to ``"Global Moran's I over time: <label>"``.

    Returns
    -------
    MoranOverTimeResult
        Frozen result with one row per period (``period``, ``moran_i``, ``z_sim``,
        ``p_sim``, ``n_obs``), the line-and-marker ``fig`` (filled markers flag
        ``p_sim < 0.05``; the dashed line marks E[I] under spatial randomness) and
        ``w_spec``.

    Examples
    --------
    Two periods on a four-cell strip with a smooth west-east gradient:

    ```python
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    from geometrics.dependence import explore_moran_over_time
    from geometrics.weights import make_weights

    gdf = gpd.GeoDataFrame(
        {"region": ["a", "b", "c", "d"]},
        geometry=[box(i, 0, i + 1, 1) for i in range(4)],
        crs="EPSG:4326",
    )
    df = pd.DataFrame(
        {
            "region": ["a", "b", "c", "d"] * 2,
            "year": [2000] * 4 + [2010] * 4,
            "gdppc": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
        }
    )
    res = explore_moran_over_time(
        df, "gdppc", gdf=gdf, w=make_weights(gdf), entity="region", time="year",
        permutations=99,
    )
    print(res.df[["period", "n_obs"]].to_dict("list"))
    ```
    """
    func = "explore_moran_over_time"
    df = ensure_dataframe(df)
    gdf = ensure_geodataframe(gdf, func=func)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # require_* guarantee both
    _validate_var(df, var, func=func)
    _validate_permutations(permutations, func=func)

    notes: list[str] = []
    if w is None:
        w = _default_weights(gdf, func=func)
        notes.append(
            f"{func}: no weights supplied — defaulted to {w.geometrics_meta['spec']}"
        )
    gdf_entity = resolve_gdf_entity(gdf)

    sub = df
    dup = sub.duplicated(subset=[entity, time])
    if bool(dup.any()):
        sub = sub.drop_duplicates(subset=[entity, time], keep="first")
        notes.append(
            f"{func}: kept the first of {int(dup.sum())} duplicate "
            f"({entity!r}, {time!r}) row(s)"
        )

    gdf_ids = list(gdf[gdf_entity])
    if set(w.id_order) != set(gdf_ids):
        raise ValueError(
            f"{func}: w ids do not match gdf.{gdf_entity} ids — build w from the "
            "same gdf (e.g. make_weights(gdf))"
        )
    extra = set(sub[entity].dropna()) - set(gdf_ids)
    if extra:
        msg = (
            f"{func}: {len(extra)} df entit(ies) not in the geometry were ignored "
            f"(e.g. {_first_ids(sorted(extra, key=str))})"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)

    wide = sub.pivot(index=entity, columns=time, values=var).reindex(index=gdf_ids)
    periods = sorted(wide.columns.tolist())
    empty = [p for p in periods if wide[p].isna().all()]
    if empty:
        msg = (
            f"{func}: dropped {len(empty)} period(s) with no {var!r} data "
            f"({_first_ids(empty)})"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)
        periods = [p for p in periods if p not in set(empty)]
    if not periods:
        raise ValueError(f"{func}: no period of {time!r} has any {var!r} data")

    # ONE aligned entity set: complete cases intersected across periods, so the
    # same W applies to every period and the series is comparable over time.
    complete = wide[periods].dropna()
    n_dropped = len(wide) - len(complete)
    if n_dropped:
        msg = (
            f"{func}: dropped {n_dropped} of {len(wide)} unit(s) without complete "
            f"{var!r} data across all {len(periods)} period(s) — the same entity "
            "set (and weights) is reused for every period"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)
    if len(complete) < 3:
        raise ValueError(
            f"{func}: only {len(complete)} unit(s) have complete {var!r} data "
            "across all periods; need at least 3"
        )

    kept = list(complete.index)
    if list(w.id_order) != kept:
        from libpysal.weights.util import w_subset

        transform = w.transform
        w = w_subset(w, kept, silence_warnings=True)
        w.transform = transform
        meta = dict(getattr(w, "geometrics_meta", {}) or {})
        if meta:
            meta["n"] = w.n
            w.geometrics_meta = meta
        notes.append(
            f"{func}: restricted the spatial weights to the {len(kept)} aligned "
            f"unit(s) and re-applied transform {transform!r}"
        )
    _note_islands(w, notes, func=func)

    from esda.moran import Moran

    values = complete.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    skipped: list[Any] = []
    for j, p in enumerate(periods):
        y = values[:, j]
        if float(y.std(ddof=0)) == 0.0:
            skipped.append(p)
            continue
        if seed is not None:
            np.random.seed(seed)
        mi = Moran(y, w, permutations=int(permutations))
        rows.append(
            {
                "period": p,
                "moran_i": float(mi.I),
                "z_sim": float(mi.z_sim),
                "p_sim": float(mi.p_sim),
                "n_obs": len(y),
            }
        )
    if skipped:
        msg = (
            f"{func}: skipped {len(skipped)} period(s) where {var!r} has zero "
            f"variance ({_first_ids(skipped)})"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)
    if not rows:
        raise ValueError(
            f"{func}: {var!r} has zero variance in every period — spatial "
            "autocorrelation is undefined for a constant"
        )
    tidy = pd.DataFrame(rows, columns=["period", "moran_i", "z_sim", "p_sim", "n_obs"])

    w_spec = _w_spec(w)
    var_label = resolve_label(df, var)
    if title is None:
        title = f"Global Moran's I over time: {var_label}"

    # Single series: no legend box (the title names it); filled vs open markers
    # encode per-period permutation significance at the 5% level.
    significant = tidy["p_sim"].to_numpy() < 0.05
    period_labels = [str(p) for p in tidy["period"]]
    fig = go.Figure(
        go.Scatter(
            x=period_labels,
            y=tidy["moran_i"].to_numpy(),
            mode="lines+markers",
            line={"color": color_for(0), "width": 2},
            marker={
                "size": 11,
                "color": color_for(0),
                "symbol": ["circle" if s else "circle-open" for s in significant],
                "line": {"color": "white", "width": 1},
            },
            customdata=np.column_stack(
                [
                    period_labels,
                    tidy["p_sim"].to_numpy(),
                    tidy["n_obs"].to_numpy(),
                ]
            ),
            hovertemplate=(
                f"{time}: %{{customdata[0]}}<br>"
                "Moran's I: %{y:.3f}<br>"
                "p (perm): %{customdata[1]:.3f}<br>"
                "n: %{customdata[2]}<extra></extra>"
            ),
            name="Moran's I",
            showlegend=False,
        )
    )
    expected_i = -1.0 / (len(kept) - 1)
    fig.add_hline(
        y=expected_i,
        line=dict(_REFLINE),
        annotation_text=f"E[I] = {expected_i:.3f}",
        annotation_position="bottom right",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.02,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        text="filled markers: p (perm) < 0.05",
        font={"size": 12},
        bgcolor="rgba(255,255,255,0.7)",
    )
    apply_default_layout(
        fig,
        title=title,
        subtitle=f"{w_spec}; {int(permutations)} permutations per period",
        xaxis={
            "title": time,
            "type": "category",
            "categoryorder": "array",
            "categoryarray": period_labels,
        },
        yaxis={"title": "Moran's I"},
    )
    return MoranOverTimeResult(
        df=tidy,
        fig=fig,
        var=var,
        permutations=int(permutations),
        w_spec=w_spec,
        notes=tuple(notes),
    )
