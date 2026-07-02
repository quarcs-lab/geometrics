"""Spatial weights: construction, sensible defaults, and the connectivity map.

The spatial weights matrix ``W`` encodes who is a neighbor of whom — the "geography" a
spatial statistic conditions on. :func:`make_weights` builds the standard families
(contiguity, k-nearest neighbors, distance band, inverse distance) with the library's
conventions baked in (entity ids as the weight ids, islands attached, row
standardization, a human-readable ``spec`` recorded on the object), and
:func:`explore_connectivity_map` draws the resulting graph so the connectivity structure
can be *seen* before it is trusted.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from geometrics._geo import ensure_metric_crs, resolve_gdf_entity
from geometrics._theme import apply_default_layout, color_for
from geometrics._types import ConnectivityMapResult
from geometrics._validation import GeometricsWarning, ensure_geodataframe

if TYPE_CHECKING:
    import geopandas as gpd
    from libpysal.weights import W

__all__ = ["explore_connectivity_map", "make_weights"]

_METHODS = ("queen", "rook", "knn", "distance_band", "inverse_distance")

_EDGE_COLOR = "rgba(78,121,167,0.45)"  # Tableau blue at low opacity
_POLYGON_FILL = "#e8e8e8"
_POLYGON_LINE = "#ffffff"


def _metric_coords(gdf: gpd.GeoDataFrame, crs: Any) -> tuple[np.ndarray, str | None]:
    """Return centroid coordinates (and the CRS used) for distance-based weights."""
    proj = ensure_metric_crs(gdf, crs, func="make_weights")
    with warnings.catch_warnings():
        # crs=None deliberately reproduces geographic-centroid behavior (the source
        # paper's k-NN basis); silence geopandas' advisory about it.
        warnings.filterwarnings(
            "ignore", message="Geometry is in a geographic CRS", category=UserWarning
        )
        cent = proj.geometry.centroid
    coords = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])
    return coords, (proj.crs.to_string() if proj.crs is not None else None)


def _knn_weights(gdf: gpd.GeoDataFrame, k: int, ids: list, crs: Any) -> W:
    """Build a k-nearest-neighbor W from (metric) centroids with the entity ids."""
    from libpysal.weights import KNN

    coords, _ = _metric_coords(gdf, crs)
    return KNN(coords, k=k, ids=ids, silence_warnings=True)


def _compose_spec(
    *,
    method: str,
    k: int | None,
    threshold: float | None,
    power: float | None,
    islands_attached: list,
    row_standardized: bool,
    n: int,
    crs: Any,
) -> str:
    """Compose the human-readable one-line description recorded as ``spec``."""
    if method in ("queen", "rook"):
        desc = f"{method} contiguity"
        if islands_attached:
            desc += f" ({len(islands_attached)} island(s) attached to nearest neighbor)"
    elif method == "knn":
        basis = "geographic centroids" if crs is None else "metric centroids"
        desc = f"{k}-nearest-neighbor ({basis})"
    elif method == "distance_band":
        desc = f"distance band (threshold {threshold:.6g})"
    else:
        desc = f"inverse distance (power {power:g}, band threshold {threshold:.6g})"
    if row_standardized:
        desc += ", row-standardized"
    return f"{desc}, n={n}"


def make_weights(
    gdf: gpd.GeoDataFrame,
    *,
    method: str = "queen",
    k: int = 6,
    threshold: float | None = None,
    power: float = 1.0,
    row_standardize: bool = True,
    attach_islands: bool = True,
    entity: str | None = None,
    crs: Any = "auto",
) -> W:
    """Build a spatial weights matrix from geometry with the library's conventions.

    The weights are keyed by the gdf's entity ids (so results are auditable by unit),
    contiguity islands are attached to their nearest neighbor, rows are standardized to
    sum to one, and a machine- and human-readable description is stored on
    ``w.geometrics_meta`` (``spec`` is the one-liner every spatial result records as
    ``w_spec``).

    Parameters
    ----------
    gdf
        Geometry frame (see :func:`geometrics.read_gdf`); its entity column supplies
        the weight ids.
    method
        ``"queen"`` / ``"rook"`` (shared-boundary contiguity), ``"knn"``
        (k-nearest-neighbor centroids), ``"distance_band"`` (binary within a radius) or
        ``"inverse_distance"`` (:math:`1/d^{p}` within a radius).
    k
        Number of neighbors for ``method="knn"``.
    threshold
        Radius for the distance-based methods. ``None`` uses the smallest distance that
        leaves no unit isolated (``min_threshold_distance``).
    power
        Distance-decay exponent :math:`p` for ``method="inverse_distance"``.
    row_standardize
        Standardize each row of ``W`` to sum to one (the convention for spatial lags).
    attach_islands
        For contiguity methods, connect units with no shared-boundary neighbor to
        their nearest neighbor (a :class:`~geometrics.GeometricsWarning` names them).
    entity
        Entity id column of ``gdf``; resolved automatically when ``None``.
    crs
        CRS handling for centroid distances (knn / distance methods): ``"auto"``
        projects to an estimated UTM CRS, ``None`` keeps the raw coordinates
        (reproducing lat/lon-centroid k-NN analyses), anything else is passed to
        ``to_crs``.

    Returns
    -------
    libpysal.weights.W
        The weights, with ``w.geometrics_meta`` recording ``method``, ``k``,
        ``threshold``, ``power``, ``crs``, ``islands_attached``,
        ``row_standardized``, ``n`` and the human-readable ``spec``.

    Raises
    ------
    ValueError
        For an unknown ``method``, duplicate entity ids, or an out-of-range ``k``.

    Examples
    --------
    Queen contiguity on a two-cell map (each cell has one neighbor):

    ```python
    import geopandas as gpd
    from shapely.geometry import box

    from geometrics.weights import make_weights

    gdf = gpd.GeoDataFrame(
        {"region": ["A", "B"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    w = make_weights(gdf, method="queen")
    (w.neighbors["A"], w.geometrics_meta["spec"])
    ```
    """
    from libpysal import weights as lw
    from libpysal.weights.util import attach_islands as _attach
    from libpysal.weights.util import min_threshold_distance

    gdf = ensure_geodataframe(gdf, func="make_weights")
    if method not in _METHODS:
        raise ValueError(
            f"make_weights: unknown method {method!r}; choose from {list(_METHODS)}"
        )
    entity = resolve_gdf_entity(gdf, entity)
    ids = list(gdf[entity])
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"make_weights: entity column {entity!r} has duplicate ids — weights "
            "need one row per unique unit"
        )
    n = len(gdf)

    islands_attached: list = []
    resolved_threshold: float | None = None
    crs_used: Any = None

    if method in ("queen", "rook"):
        builder = lw.Queen if method == "queen" else lw.Rook
        w = builder.from_dataframe(gdf, ids=ids, use_index=False, silence_warnings=True)
        if w.islands:
            if attach_islands:
                islands_attached = list(w.islands)
                warnings.warn(
                    f"make_weights: {len(islands_attached)} unit(s) have no {method} "
                    f"neighbor (islands: {islands_attached[:5]}) — attached each to "
                    "its nearest neighbor",
                    GeometricsWarning,
                    stacklevel=2,
                )
                w = _attach(w, _knn_weights(gdf, 1, ids, crs), silence_warnings=True)
            else:
                warnings.warn(
                    f"make_weights: {len(w.islands)} unit(s) have no {method} "
                    f"neighbor and were left as islands: {list(w.islands)[:5]}",
                    GeometricsWarning,
                    stacklevel=2,
                )
    elif method == "knn":
        if not 1 <= k < n:
            raise ValueError(f"make_weights: k={k} needs to satisfy 1 <= k < n (n={n})")
        w = _knn_weights(gdf, k, ids, crs)
        crs_used = _metric_coords(gdf, crs)[1]
    else:
        coords, crs_used = _metric_coords(gdf, crs)
        resolved_threshold = (
            float(min_threshold_distance(coords))
            if threshold is None
            else float(threshold)
        )
        with np.errstate(divide="ignore"):
            w = lw.DistanceBand(
                coords,
                threshold=resolved_threshold,
                binary=(method == "distance_band"),
                alpha=-float(power),
                ids=ids,
                silence_warnings=True,
            )
        if w.islands:
            warnings.warn(
                f"make_weights: threshold {resolved_threshold:.6g} leaves "
                f"{len(w.islands)} unit(s) with no neighbor: {list(w.islands)[:5]} — "
                "increase threshold= or leave it None",
                GeometricsWarning,
                stacklevel=2,
            )

    if row_standardize:
        w.transform = "r"

    w.geometrics_meta = {
        "method": method,
        "k": k if method == "knn" else None,
        "threshold": resolved_threshold,
        "power": power if method == "inverse_distance" else None,
        "crs": crs_used,
        "islands_attached": islands_attached,
        "row_standardized": bool(row_standardize),
        "n": n,
        "spec": _compose_spec(
            method=method,
            k=k,
            threshold=resolved_threshold,
            power=power,
            islands_attached=islands_attached,
            row_standardized=bool(row_standardize),
            n=n,
            crs=crs_used if method != "knn" else (None if crs is None else crs_used),
        ),
    }
    return w


def _default_weights(
    gdf: gpd.GeoDataFrame, *, entity: str | None = None, func: str = ""
) -> W:
    """Build the default W used when a spatial function receives ``w=None``.

    Polygon geometry gets queen contiguity (islands attached); anything else (points,
    lines, mixed) gets 6-nearest-neighbor weights. A
    :class:`~geometrics.GeometricsWarning` announces the choice so the default is
    never silent.
    """
    gdf = ensure_geodataframe(gdf, func=func or "_default_weights")
    geom_types = set(gdf.geometry.geom_type.dropna())
    polygonal = bool(geom_types) and geom_types <= {"Polygon", "MultiPolygon"}
    if polygonal:
        w = make_weights(gdf, method="queen", entity=entity)
    else:
        w = make_weights(gdf, method="knn", k=6, entity=entity)
    prefix = f"{func}: " if func else ""
    warnings.warn(
        f"{prefix}no spatial weights supplied — defaulting to "
        f"{w.geometrics_meta['spec']}; pass w=make_weights(...) to control this",
        GeometricsWarning,
        stacklevel=3,
    )
    return w


def _describe_w(w: W) -> str:
    """Compose a short human description for a W without ``geometrics_meta``."""
    standardized = ", row-standardized" if str(w.transform).upper() == "R" else ""
    return (
        f"user-supplied W (mean {float(w.mean_neighbors):.2f} "
        f"neighbors{standardized}), n={w.n}"
    )


def _geojson_and_ids(gdf: gpd.GeoDataFrame, ids: list) -> tuple[dict, list[str]]:
    """Return a GeoJSON mapping and string feature ids for the choropleth layers."""
    str_ids = [str(i) for i in ids]
    geo = gdf[[gdf.geometry.name]].set_index(pd.Index(str_ids))
    return json.loads(geo.to_json()), str_ids


def _edge_coordinates(
    w: W, xs: dict, ys: dict
) -> tuple[list[float | None], list[float | None]]:
    """Build None-separated edge polylines for every unique neighbor pair in ``w``."""
    lons: list[float | None] = []
    lats: list[float | None] = []
    seen: set[tuple] = set()
    for i in w.id_order:
        for j in w.neighbors[i]:
            key = (i, j) if str(i) <= str(j) else (j, i)
            if key in seen:
                continue
            seen.add(key)
            lons.extend([xs[i], xs[j], None])
            lats.extend([ys[i], ys[j], None])
    return lons, lats


def _map_zoom(bounds: np.ndarray) -> float:
    """Approximate a MapLibre zoom level that frames ``bounds`` (lon/lat degrees)."""
    minx, miny, maxx, maxy = (float(b) for b in bounds)
    span = max(maxx - minx, (maxy - miny) * 1.6, 1e-4)
    return float(np.clip(np.log2(360.0 / span) - 0.5, 1.0, 13.0))


def explore_connectivity_map(
    gdf: gpd.GeoDataFrame,
    *,
    w: W | None = None,
    entity: str | None = None,
    tiles: str | None = "carto-positron",
    title: str | None = None,
) -> ConnectivityMapResult:
    """Draw the spatial weights graph over the map and summarize its connectivity.

    The figure overlays the neighbor graph (edges between adjacent centroids, one node
    per unit) on a light-grey polygon layer, and the companion histogram shows the
    neighbor-cardinality distribution — the standard visual audit of a ``W`` before
    any spatial statistic is computed on it.

    Parameters
    ----------
    gdf
        Geometry frame (see :func:`geometrics.read_gdf`).
    w
        ``libpysal`` weights aligned to the gdf entity ids. ``None`` builds the
        default weights (queen contiguity for polygons, 6-nearest-neighbor otherwise)
        with a :class:`~geometrics.GeometricsWarning`.
    entity
        Entity id column of ``gdf``; resolved automatically when ``None``.
    tiles
        MapLibre basemap style (e.g. ``"carto-positron"``). ``None`` draws a vector
        (tile-free) figure suitable for deterministic PNG export.
    title
        Figure title (a default naming the weights is used when ``None``).

    Returns
    -------
    ConnectivityMapResult
        The per-entity neighbor-cardinality frame, the graph figure (``fig``), the
        cardinality histogram (``fig_hist``), the connectivity scalars and ``w_spec``.

    Examples
    --------
    Connectivity of a two-cell map (each unit has exactly one neighbor):

    ```python
    import geopandas as gpd
    from shapely.geometry import box

    from geometrics.weights import explore_connectivity_map, make_weights

    gdf = gpd.GeoDataFrame(
        {"region": ["A", "B"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    res = explore_connectivity_map(gdf, w=make_weights(gdf), tiles=None)
    (res.n_units, res.mean_neighbors)
    ```
    """
    gdf = ensure_geodataframe(gdf, func="explore_connectivity_map")
    entity = resolve_gdf_entity(gdf, entity)
    notes: list[str] = []

    if w is None:
        w = _default_weights(gdf, entity=entity, func="explore_connectivity_map")
        notes.append(
            "explore_connectivity_map: no weights supplied — defaulted to "
            f"{w.geometrics_meta['spec']}"
        )
    ids = list(gdf[entity])
    if w.n != len(gdf) or set(w.id_order) != set(ids):
        raise ValueError(
            f"explore_connectivity_map: w ids do not match gdf.{entity} ids "
            f"(w.n={w.n}, gdf n={len(gdf)}) — build w from the same gdf"
        )

    meta = dict(getattr(w, "geometrics_meta", {}) or {})
    w_spec = meta.get("spec") or _describe_w(w)
    islands = (
        tuple(meta["islands_attached"])
        if meta.get("islands_attached")
        else tuple(w.islands)
    )

    cards = [int(w.cardinalities[i]) for i in ids]
    df = pd.DataFrame({entity: ids, "n_neighbors": cards})

    # --- the connectivity graph figure -------------------------------------------
    plot_gdf = gdf
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        plot_gdf = gdf.to_crs("EPSG:4326")
    elif gdf.crs is None:
        notes.append(
            "explore_connectivity_map: gdf has no CRS — coordinates drawn as given"
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Geometry is in a geographic CRS", category=UserWarning
        )
        cent = plot_gdf.geometry.centroid
    xs = dict(zip(ids, cent.x.to_numpy(), strict=True))
    ys = dict(zip(ids, cent.y.to_numpy(), strict=True))
    edge_x, edge_y = _edge_coordinates(w, xs, ys)
    node_x = [xs[i] for i in ids]
    node_y = [ys[i] for i in ids]
    customdata = np.column_stack([[str(i) for i in ids], cards])
    hover = "%{customdata[0]}<br>neighbors: %{customdata[1]}<extra></extra>"
    geom_types = set(plot_gdf.geometry.geom_type.dropna())
    polygonal = bool(geom_types) and geom_types <= {"Polygon", "MultiPolygon"}

    fig = go.Figure()
    if tiles is not None:
        if polygonal:
            geojson, str_ids = _geojson_and_ids(plot_gdf, ids)
            fig.add_trace(
                go.Choroplethmap(
                    geojson=geojson,
                    locations=str_ids,
                    z=[0.0] * len(str_ids),
                    colorscale=[[0.0, _POLYGON_FILL], [1.0, _POLYGON_FILL]],
                    showscale=False,
                    marker={
                        "line": {"color": _POLYGON_LINE, "width": 1},
                        "opacity": 0.6,
                    },
                    hoverinfo="skip",
                    name="units",
                )
            )
        fig.add_trace(
            go.Scattermap(
                lon=edge_x,
                lat=edge_y,
                mode="lines",
                line={"color": _EDGE_COLOR, "width": 1},
                hoverinfo="skip",
                showlegend=False,
                name="edges",
            )
        )
        fig.add_trace(
            go.Scattermap(
                lon=node_x,
                lat=node_y,
                mode="markers",
                marker={"size": 7, "color": color_for(0)},
                customdata=customdata,
                hovertemplate=hover,
                showlegend=False,
                name="units",
            )
        )
        bounds = plot_gdf.total_bounds
        fig.update_layout(
            map={
                "style": tiles,
                "center": {
                    "lon": float((bounds[0] + bounds[2]) / 2),
                    "lat": float((bounds[1] + bounds[3]) / 2),
                },
                "zoom": _map_zoom(bounds),
            }
        )
    else:
        if polygonal:
            geojson, str_ids = _geojson_and_ids(plot_gdf, ids)
            fig.add_trace(
                go.Choropleth(
                    geojson=geojson,
                    locations=str_ids,
                    z=[0.0] * len(str_ids),
                    colorscale=[[0.0, _POLYGON_FILL], [1.0, _POLYGON_FILL]],
                    showscale=False,
                    marker={"line": {"color": _POLYGON_LINE, "width": 1}},
                    hoverinfo="skip",
                    name="units",
                )
            )
        fig.add_trace(
            go.Scattergeo(
                lon=edge_x,
                lat=edge_y,
                mode="lines",
                line={"color": _EDGE_COLOR, "width": 1},
                hoverinfo="skip",
                showlegend=False,
                name="edges",
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lon=node_x,
                lat=node_y,
                mode="markers",
                marker={"size": 7, "color": color_for(0)},
                customdata=customdata,
                hovertemplate=hover,
                showlegend=False,
                name="units",
            )
        )
        fig.update_geos(visible=False, fitbounds="locations")

    apply_default_layout(
        fig,
        title=title if title is not None else "Spatial connectivity structure",
        subtitle=w_spec,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
    )

    # --- the cardinality histogram ------------------------------------------------
    counts = pd.Series(cards).value_counts().sort_index()
    fig_hist = go.Figure(
        go.Bar(
            x=counts.index.to_numpy(),
            y=counts.to_numpy(),
            marker={"color": color_for(0)},
            hovertemplate="neighbors: %{x}<br>units: %{y}<extra></extra>",
            name="units",
        )
    )
    apply_default_layout(
        fig_hist,
        title="Neighbor cardinality distribution",
        subtitle=w_spec,
        xaxis={"title": "Number of neighbors", "dtick": 1},
        yaxis={"title": "Number of units"},
        bargap=0.05,
        showlegend=False,
    )

    return ConnectivityMapResult(
        df=df,
        fig=fig,
        fig_hist=fig_hist,
        n_units=int(w.n),
        mean_neighbors=float(w.mean_neighbors),
        min_neighbors=int(w.min_neighbors),
        max_neighbors=int(w.max_neighbors),
        pct_nonzero=float(w.pct_nonzero),
        n_components=int(w.n_components),
        islands=islands,
        w_spec=w_spec,
        notes=tuple(notes),
    )
