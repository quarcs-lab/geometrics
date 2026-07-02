"""Shared Plotly choropleth builders used by every mapping function.

Three building blocks render entity polygons as themed Plotly choropleths:

* :func:`classified_map` — a classed map (mapclassify schemes) with one
  legend-togglable trace per class;
* :func:`categorical_map` — fixed label-to-color categories (LISA clusters,
  convergence clubs);
* :func:`continuous_map` — a single continuous trace with a colorbar, optionally
  diverging and/or with a light-grey mask layer (e.g. non-significant GWR
  coefficients).

Every builder renders through one of two backends, switched by ``tiles``:

* ``tiles=<style name>`` — :class:`plotly.graph_objects.Choroplethmap` traces over a
  MapLibre base map (``layout.map``; Plotly >= 5.24 — never the deprecated
  ``*mapbox`` family), with the view centered and zoomed to the data's bounds;
* ``tiles=None`` — :class:`plotly.graph_objects.Choropleth` vector traces on an
  invisible ``layout.geo`` fitted to the locations (deterministic PNG export).

Geometry reaches Plotly via :func:`geojson_interface`, which reprojects to WGS84 and
(by default) simplifies in a metric CRS first so figures stay light.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import plotly.colors as pcolors
import plotly.graph_objects as go

from geometrics._theme import (
    MAP_DIVERGING,
    MAP_SEQUENTIAL,
    apply_default_layout,
    color_for,
)
from geometrics._validation import ensure_geodataframe

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "geojson_interface",
    "classified_map",
    "categorical_map",
    "continuous_map",
]

# Light grey for masked / not-significant polygons (matches LISA "Not significant").
_MASK_GREY = "#d3d3d3"
# Maps carry their own base layer, so the plot area runs nearly edge to edge.
_MAP_MARGIN = {"l": 10, "r": 10, "t": 60, "b": 10}
_POLYGON_EDGE = {"color": "white", "width": 0.5}


def _wgs84_simplified(
    gdf: gpd.GeoDataFrame,
    entity: str,
    simplify: float | str | None,
    *,
    func: str = "geojson_interface",
) -> gpd.GeoDataFrame:
    """Return ``gdf`` reduced to entity + geometry, simplified, in EPSG:4326.

    Simplification happens *before* the WGS84 reprojection, in a metric CRS
    (``estimate_utm_crs``) so the tolerance is in meters: ``"auto"`` uses the maximum
    bounding-box dimension divided by 2000, a float is an explicit tolerance, and
    ``None`` skips simplification. Topology is preserved. A ``gdf`` without a CRS is
    used as-is (its coordinates are assumed to already be in degrees).
    """
    gdf = ensure_geodataframe(gdf, func=func)
    if entity not in gdf.columns:
        raise KeyError(f"{func}: entity column {entity!r} not found in gdf")
    if len(gdf) == 0:
        raise ValueError(f"{func}: gdf has no rows to map")
    geom_col = gdf.geometry.name
    g = gdf[[entity, geom_col]].copy()
    if simplify is not None:
        metric = (
            g.to_crs(g.estimate_utm_crs())
            if g.crs is not None and g.crs.is_geographic
            else g
        )
        if simplify == "auto":
            minx, miny, maxx, maxy = metric.total_bounds
            tolerance = max(maxx - minx, maxy - miny) / 2000.0
        else:
            tolerance = float(simplify)
        if tolerance > 0:
            metric = metric.copy() if metric is g else metric
            metric[geom_col] = metric.geometry.simplify(
                tolerance, preserve_topology=True
            )
        g = metric
    if g.crs is not None:
        g = g.to_crs("EPSG:4326")
    return g


def geojson_interface(
    gdf: gpd.GeoDataFrame,
    entity: str,
    *,
    simplify: float | str | None = "auto",
) -> tuple[dict, list]:
    """Convert ``gdf`` into a Plotly-ready GeoJSON FeatureCollection keyed by entity.

    The geometry is (optionally) simplified in a metric CRS and reprojected to WGS84;
    each feature carries ``id`` = its entity value (via ``set_index(entity)``), the
    key Plotly's choropleth traces match ``locations`` against.

    Parameters
    ----------
    gdf
        Geometry with an entity-id column.
    entity
        Name of the entity-id column in ``gdf``.
    simplify
        ``"auto"`` (default) simplifies with a tolerance of the maximum metric
        bounding-box dimension divided by 2000; a float is an explicit tolerance in
        the metric CRS's units (meters); ``None`` disables simplification.

    Returns
    -------
    tuple of (dict, list)
        The FeatureCollection ``dict`` and the list of entity ids, in ``gdf`` row
        order.

    Examples
    --------
    Two square cells become two identifiable features:

    ```python
    import geopandas as gpd
    from shapely.geometry import box

    from geometrics._mapping import geojson_interface

    gdf = gpd.GeoDataFrame(
        {"id": ["a", "b"]},
        geometry=[box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)],
        crs="EPSG:4326",
    )
    fc, ids = geojson_interface(gdf, "id", simplify=None)
    print(ids, fc["features"][0]["id"])
    ```
    """
    g = _wgs84_simplified(gdf, entity, simplify)
    fc = dict(g.set_index(entity).__geo_interface__)
    return fc, list(g[entity])


def _geo_inputs(
    gdf: gpd.GeoDataFrame,
    entity: str,
    simplify: float | str | None,
    *,
    func: str,
) -> tuple[dict, list, gpd.GeoDataFrame]:
    """Return the FeatureCollection, entity ids and the WGS84 frame actually drawn."""
    g = _wgs84_simplified(gdf, entity, simplify, func=func)
    fc = dict(g.set_index(entity).__geo_interface__)
    return fc, list(g[entity]), g


def _center_zoom(bounds: ArrayLike) -> tuple[dict[str, float], float]:
    """Compute a MapLibre ``center`` / ``zoom`` fitting the WGS84 ``bounds``.

    ``bounds`` is ``(minx, miny, maxx, maxy)`` in degrees; the zoom fits the wider of
    the longitude/latitude spans into a single map view with a small padding.
    """
    minx, miny, maxx, maxy = (float(b) for b in np.asarray(bounds, dtype=float))
    center = {"lon": (minx + maxx) / 2.0, "lat": (miny + maxy) / 2.0}
    span_lon = max(maxx - minx, 1e-5)
    span_lat = max(maxy - miny, 1e-5)
    zoom = min(math.log2(360.0 / span_lon), math.log2(170.0 / span_lat)) - 0.4
    return center, float(min(max(zoom, 0.0), 14.0))


def _polygon_trace(
    tiles: str | None, **kwargs: Any
) -> go.Choroplethmap | go.Choropleth:
    """Build one choropleth trace on the backend selected by ``tiles``."""
    return go.Choroplethmap(**kwargs) if tiles is not None else go.Choropleth(**kwargs)


def _finish_map(
    fig: go.Figure,
    *,
    tiles: str | None,
    bounds: ArrayLike,
    title: str | None,
    legend_title: str | None = None,
) -> go.Figure:
    """Apply the backend layout (tiles or vector geo) and the geometrics theme."""
    if tiles is not None:
        center, zoom = _center_zoom(bounds)
        fig.update_layout(map={"style": tiles, "center": center, "zoom": zoom})
    else:
        fig.update_layout(geo={"visible": False, "fitbounds": "locations"})
    apply_default_layout(fig, title=title, margin=dict(_MAP_MARGIN))
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    return fig


def _display_names(
    ids: Sequence[Any], hover_names: Mapping[Any, str] | None
) -> list[str]:
    """Map entity ids to display names via ``hover_names`` (``str(id)`` fallback)."""
    if hover_names is None:
        return [str(i) for i in ids]
    return [str(hover_names.get(str(i), hover_names.get(i, i))) for i in ids]


def _class_labels(bins: Sequence[float], vmin: float) -> list[str]:
    """Build ``'lo - hi'`` interval labels for classes with upper bounds ``bins``."""
    edges = [float(vmin), *(float(b) for b in bins)]
    return [f"{edges[i]:.4g} - {edges[i + 1]:.4g}" for i in range(len(bins))]


def _class_colors(k: int) -> list[str]:
    """Sample ``k`` class colors evenly from the sequential map scale."""
    points = [0.5] if k == 1 else [i / (k - 1) for i in range(k)]
    return pcolors.sample_colorscale(MAP_SEQUENTIAL, points)


def _clean_values(values: ArrayLike, n: int, *, func: str) -> np.ndarray:
    """Validate ``values`` (float, complete, one per gdf row) and return the array."""
    vals = np.asarray(values, dtype=float).ravel()
    if len(vals) != n:
        raise ValueError(f"{func}: values has {len(vals)} entries but gdf has {n} rows")
    if np.isnan(vals).any():
        raise ValueError(
            f"{func}: values contain missing entries — align df and gdf (dropping "
            "missing rows) before mapping"
        )
    return vals


def classified_map(
    gdf: gpd.GeoDataFrame,
    values: ArrayLike,
    *,
    entity: str,
    scheme: str | None = "fisherjenks",
    k: int = 5,
    bins: Sequence[float] | None = None,
    tiles: str | None = "carto-positron",
    title: str | None = None,
    legend_title: str | None = None,
    hover_names: Mapping[Any, str] | None = None,
    simplify: float | str | None = "auto",
) -> tuple[go.Figure, tuple[float, ...]]:
    """Build a classed choropleth: one legend-togglable trace per class.

    Values are classified with :func:`mapclassify.classify` (or
    :class:`mapclassify.UserDefined` when explicit ``bins`` are given) and each class
    is drawn as its own single-color trace — clicking a legend entry toggles the
    class. Class colors are sampled evenly from the geometrics sequential map scale.

    Parameters
    ----------
    gdf
        Geometry with the ``entity`` id column, one row per value.
    values
        Numeric values in ``gdf`` row order (no missing entries).
    entity
        Name of the entity-id column in ``gdf``.
    scheme
        A mapclassify scheme name (``"fisherjenks"``, ``"quantiles"``, ...), or
        ``None`` for a single continuous trace with a colorbar.
    k
        Number of classes for ``scheme`` (ignored when ``bins`` are given).
    bins
        Explicit upper class bounds; overrides ``scheme`` / ``k``.
    tiles
        MapLibre base-map style (``go.Choroplethmap``) or ``None`` for the vector
        ``go.Choropleth`` backend.
    title
        Figure title.
    legend_title
        Legend title (also names the hovered value).
    hover_names
        Optional mapping from entity id to a display name for the hover label.
    simplify
        Geometry simplification forwarded to :func:`geojson_interface`.

    Returns
    -------
    tuple of (plotly.graph_objects.Figure, tuple of float)
        The themed figure and the upper class bounds actually applied (empty for a
        continuous map).

    Examples
    --------
    Three cells in two classes:

    ```python
    import geopandas as gpd
    from shapely.geometry import box

    from geometrics._mapping import classified_map

    gdf = gpd.GeoDataFrame(
        {"id": ["a", "b", "c"]},
        geometry=[box(i, 0, i + 1, 1) for i in range(3)],
        crs="EPSG:4326",
    )
    fig, bins = classified_map(gdf, [1.0, 2.0, 9.0], entity="id", k=2, tiles=None)
    print(len(fig.data), bins)
    ```
    """
    import mapclassify

    fc, ids, g = _geo_inputs(gdf, entity, simplify, func="classified_map")
    vals = _clean_values(values, len(g), func="classified_map")

    if scheme is None and bins is None:
        fig = continuous_map(
            g,
            vals,
            entity=entity,
            tiles=tiles,
            title=title,
            hover_names=hover_names,
            colorbar_title=legend_title,
            simplify=None,
        )
        return fig, ()

    if bins is not None:
        classifier = mapclassify.UserDefined(vals, bins=list(bins))
    else:
        classifier = mapclassify.classify(vals, scheme, k=k)
    upper = tuple(float(b) for b in classifier.bins)
    labels = _class_labels(upper, float(vals.min()))
    colors = _class_colors(len(upper))
    display = _display_names(ids, hover_names)
    value_name = legend_title or "value"
    yb = np.asarray(classifier.yb)

    fig = go.Figure()
    for ci in range(len(upper)):
        members = np.flatnonzero(yb == ci)
        fig.add_trace(
            _polygon_trace(
                tiles,
                geojson=fc,
                locations=[ids[j] for j in members],
                z=[ci] * len(members),
                colorscale=[[0.0, colors[ci]], [1.0, colors[ci]]],
                showscale=False,
                name=labels[ci],
                showlegend=True,
                marker={"line": dict(_POLYGON_EDGE)},
                customdata=[[display[j], float(vals[j])] for j in members],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{value_name}: %{{customdata[1]:.6g}}<extra></extra>"
                ),
            )
        )
    _finish_map(
        fig, tiles=tiles, bounds=g.total_bounds, title=title, legend_title=legend_title
    )
    return fig, upper


def categorical_map(
    gdf: gpd.GeoDataFrame,
    labels: ArrayLike,
    *,
    entity: str,
    colors: Mapping[str, str],
    category_order: Sequence[str] | None = None,
    tiles: str | None = "carto-positron",
    title: str | None = None,
    hover_names: Mapping[Any, str] | None = None,
    simplify: float | str | None = "auto",
) -> go.Figure:
    """Build a categorical choropleth: one fixed-color trace per category.

    Used for maps whose classes carry fixed semantics (LISA cluster types,
    convergence-club membership), so the label-to-color assignment never depends on
    the data.

    Parameters
    ----------
    gdf
        Geometry with the ``entity`` id column, one row per label.
    labels
        Category label per ``gdf`` row.
    entity
        Name of the entity-id column in ``gdf``.
    colors
        Mapping from category label to fill color; categories missing from the
        mapping fall back to the qualitative palette.
    category_order
        Legend/trace order. Categories listed here are drawn even when empty (stable
        trace counts across animation frames); default is order of first appearance.
    tiles
        MapLibre base-map style (``go.Choroplethmap``) or ``None`` for the vector
        ``go.Choropleth`` backend.
    title
        Figure title.
    hover_names
        Optional mapping from entity id to a display name for the hover label.
    simplify
        Geometry simplification forwarded to :func:`geojson_interface`.

    Returns
    -------
    plotly.graph_objects.Figure
        The themed categorical map.

    Examples
    --------
    Two categories with fixed colors:

    ```python
    import geopandas as gpd
    from shapely.geometry import box

    from geometrics._mapping import categorical_map

    gdf = gpd.GeoDataFrame(
        {"id": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    fig = categorical_map(
        gdf,
        ["High-High", "Not significant"],
        entity="id",
        colors={"High-High": "#d7191c", "Not significant": "#d3d3d3"},
        tiles=None,
    )
    print([trace.name for trace in fig.data])
    ```
    """
    fc, ids, g = _geo_inputs(gdf, entity, simplify, func="categorical_map")
    labs = [str(x) for x in np.asarray(labels, dtype=object).ravel()]
    if len(labs) != len(g):
        raise ValueError(
            f"categorical_map: labels has {len(labs)} entries but gdf has {len(g)} rows"
        )
    if category_order is not None:
        categories = [str(c) for c in category_order]
    else:
        categories = list(dict.fromkeys(labs))
    display = _display_names(ids, hover_names)

    fig = go.Figure()
    for ci, cat in enumerate(categories):
        color = colors.get(cat, color_for(ci))
        members = [j for j, lab in enumerate(labs) if lab == cat]
        fig.add_trace(
            _polygon_trace(
                tiles,
                geojson=fc,
                locations=[ids[j] for j in members],
                z=[ci] * len(members),
                colorscale=[[0.0, color], [1.0, color]],
                showscale=False,
                name=cat,
                showlegend=True,
                marker={"line": dict(_POLYGON_EDGE)},
                customdata=[[display[j], labs[j]] for j in members],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"
                ),
            )
        )
    return _finish_map(fig, tiles=tiles, bounds=g.total_bounds, title=title)


def continuous_map(
    gdf: gpd.GeoDataFrame,
    values: ArrayLike,
    *,
    entity: str,
    diverging: bool = False,
    midpoint: float | None = None,
    mask: ArrayLike | None = None,
    mask_label: str = "Not significant",
    tiles: str | None = None,
    title: str | None = None,
    hover_names: Mapping[Any, str] | None = None,
    colorbar_title: str | None = None,
    simplify: float | str | None = "auto",
) -> go.Figure:
    """Build a continuous choropleth: a single trace with a colorbar.

    Colors come from the geometrics sequential scale, or the diverging scale when
    ``diverging=True`` (anchored at ``midpoint`` via ``zmid`` when given). Entities
    flagged by ``mask`` are pulled out of the continuous trace and drawn as a
    separate light-grey layer labelled ``mask_label`` — e.g. non-significant local
    coefficients on a GWR surface.

    Parameters
    ----------
    gdf
        Geometry with the ``entity`` id column, one row per value.
    values
        Numeric values in ``gdf`` row order (no missing entries).
    entity
        Name of the entity-id column in ``gdf``.
    diverging
        Use the diverging map scale instead of the sequential one.
    midpoint
        Value anchoring the diverging scale's midpoint (sets ``zmid``).
    mask
        Optional boolean array (``gdf`` row order); ``True`` rows are drawn grey.
    mask_label
        Legend label for the masked layer.
    tiles
        MapLibre base-map style (``go.Choroplethmap``) or ``None`` (default) for the
        vector ``go.Choropleth`` backend.
    title
        Figure title.
    hover_names
        Optional mapping from entity id to a display name for the hover label.
    colorbar_title
        Colorbar title (also names the hovered value).
    simplify
        Geometry simplification forwarded to :func:`geojson_interface`.

    Returns
    -------
    plotly.graph_objects.Figure
        The themed continuous map.

    Examples
    --------
    A diverging surface with one masked cell:

    ```python
    import geopandas as gpd
    from shapely.geometry import box

    from geometrics._mapping import continuous_map

    gdf = gpd.GeoDataFrame(
        {"id": ["a", "b", "c"]},
        geometry=[box(i, 0, i + 1, 1) for i in range(3)],
        crs="EPSG:4326",
    )
    fig = continuous_map(
        gdf,
        [-1.0, 0.5, 2.0],
        entity="id",
        diverging=True,
        midpoint=0.0,
        mask=[False, True, False],
    )
    print([trace.name for trace in fig.data])
    ```
    """
    fc, ids, g = _geo_inputs(gdf, entity, simplify, func="continuous_map")
    vals = _clean_values(values, len(g), func="continuous_map")
    if mask is None:
        masked = np.zeros(len(g), dtype=bool)
    else:
        masked = np.asarray(mask, dtype=bool).ravel()
        if len(masked) != len(g):
            raise ValueError(
                f"continuous_map: mask has {len(masked)} entries but gdf has "
                f"{len(g)} rows"
            )
    display = _display_names(ids, hover_names)
    value_name = colorbar_title or "value"

    keep = np.flatnonzero(~masked)
    trace_kwargs: dict[str, Any] = {
        "geojson": fc,
        "locations": [ids[j] for j in keep],
        "z": [float(vals[j]) for j in keep],
        "colorscale": MAP_DIVERGING if diverging else MAP_SEQUENTIAL,
        "colorbar": {
            "title": {"text": colorbar_title or ""},
            "thickness": 14,
            "len": 0.85,
            "outlinewidth": 0,
        },
        "name": value_name,
        "showlegend": False,
        "marker": {"line": dict(_POLYGON_EDGE)},
        "customdata": [[display[j], float(vals[j])] for j in keep],
        "hovertemplate": (
            "<b>%{customdata[0]}</b><br>"
            f"{value_name}: %{{customdata[1]:.6g}}<extra></extra>"
        ),
    }
    if midpoint is not None:
        trace_kwargs["zmid"] = float(midpoint)
    fig = go.Figure(_polygon_trace(tiles, **trace_kwargs))
    if masked.any():
        hidden = np.flatnonzero(masked)
        fig.add_trace(
            _polygon_trace(
                tiles,
                geojson=fc,
                locations=[ids[j] for j in hidden],
                z=[0] * len(hidden),
                colorscale=[[0.0, _MASK_GREY], [1.0, _MASK_GREY]],
                showscale=False,
                name=mask_label,
                showlegend=True,
                marker={"line": dict(_POLYGON_EDGE)},
                customdata=[[display[j], mask_label] for j in hidden],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"
                ),
            )
        )
    return _finish_map(fig, tiles=tiles, bounds=g.total_bounds, title=title)
