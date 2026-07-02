"""Choropleth mapping of one panel variable across entities.

:func:`explore_choropleth_map` joins a long panel to entity geometry and draws a
classed (mapclassify) or continuous Plotly choropleth for one period — or an
animation across all periods with pooled, fixed class breaks so colors stay
comparable over time.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from geometrics._common import entity_display_map
from geometrics._geo import _align_cross_section, resolve_gdf_entity
from geometrics._labels import resolve_label
from geometrics._mapping import _class_labels, _wgs84_simplified, classified_map
from geometrics._panel import resolve_entity_name, resolve_panel
from geometrics._types import ChoroplethMapResult
from geometrics._validation import (
    GeometricsWarning,
    ensure_dataframe,
    ensure_geodataframe,
)

if TYPE_CHECKING:
    import geopandas as gpd
    import plotly.graph_objects as go

__all__ = ["explore_choropleth_map"]

_FUNC = "explore_choropleth_map"


def _hover_value(value: Any) -> str:
    """Format one extra hover value compactly (floats to 6 significant digits)."""
    if isinstance(value, float) and not isinstance(value, bool):
        return f"{value:.6g}"
    return str(value)


def _append_hover(
    traces: Sequence[go.Choropleth | go.Choroplethmap],
    extra: dict[str, list[str]],
    labels: Sequence[str],
) -> None:
    """Append extra hover columns to each trace's customdata and hovertemplate.

    ``extra`` maps ``str(entity id)`` to the pre-formatted extra values; the builders
    always emit customdata rows ``[display_name, value_or_label]``, so the extras are
    appended from index 2 on.
    """
    lines = "".join(
        f"<br>{label}: %{{customdata[{2 + j}]}}" for j, label in enumerate(labels)
    )
    for trace in traces:
        locations = trace.locations
        if locations is None or len(locations) == 0:
            continue
        base = [list(row) for row in trace.customdata]
        trace.customdata = [
            [*base[i], *extra[str(loc)]] for i, loc in enumerate(locations)
        ]
        trace.hovertemplate = trace.hovertemplate.replace(
            "<extra></extra>", f"{lines}<extra></extra>"
        )


def _class_column(
    values: np.ndarray, bins: tuple[float, ...], vmin: float
) -> pd.Series:
    """Return the interval label of each value's class (``NA`` when unclassified)."""
    if not bins:
        return pd.Series([pd.NA] * len(values), dtype="object")
    labels = _class_labels(bins, vmin)
    yb = np.clip(
        np.searchsorted(np.asarray(bins, dtype=float), values, side="left"),
        0,
        len(bins) - 1,
    )
    return pd.Series([labels[i] for i in yb], dtype="object")


def explore_choropleth_map(
    df: pd.DataFrame,
    var: str,
    *,
    gdf: gpd.GeoDataFrame,
    period: Any = None,
    animate: bool = False,
    entity: str | None = None,
    time: str | None = None,
    scheme: str | None = "fisherjenks",
    k: int = 5,
    bins: Sequence[float] | None = None,
    tiles: str | None = "carto-positron",
    hover: str | Sequence[str] | None = None,
    simplify: float | str | None = "auto",
    title: str | None = None,
) -> ChoroplethMapResult:
    """Map one variable across entities as a classed (or continuous) choropleth.

    The panel ``df`` is aligned to the entity geometry ``gdf`` for one cross section
    (the latest period by default) and drawn as a Plotly choropleth with one
    legend-togglable trace per class. With ``animate=True`` every period becomes an
    animation frame, classified on **pooled** breaks (computed from all periods
    together) so colors are comparable over time.

    Parameters
    ----------
    df
        Long panel (or cross section) holding ``var`` per entity.
    var
        Numeric column of ``df`` to map.
    gdf
        Entity geometry; must carry the same entity-id column as ``df``.
    period
        Period to map. Defaults to the latest period when ``df`` has a time
        dimension (a note records this). Ignored when ``animate=True``.
    animate
        Draw every period as an animation frame with a slider and play button.
    entity, time
        Panel identifiers; default to the ids declared via
        :func:`geometrics.set_panel`.
    scheme
        A mapclassify scheme name (``"fisherjenks"``, ``"quantiles"``, ...), or
        ``None`` for a continuous colorbar map.
    k
        Number of classes for ``scheme`` (ignored when ``bins`` are given).
    bins
        Explicit upper class bounds (overrides ``scheme`` / ``k``).
    tiles
        MapLibre base-map style (default ``"carto-positron"``) or ``None`` for the
        vector backend (deterministic PNG export).
    hover
        Extra ``df`` column(s) appended to the hover box.
    simplify
        Geometry simplification: ``"auto"`` (metric tolerance = max bounding-box
        dimension / 2000), a float tolerance in meters, or ``None`` to disable.
    title
        Figure title. Defaults to the variable label plus the mapped period.

    Returns
    -------
    ChoroplethMapResult
        Frozen result with ``df`` (entity, value, class label), ``fig``,
        ``gdf_plotted`` (the WGS84 geometry actually drawn, value and class
        attached), the applied ``bins``, and ``notes``.

    Examples
    --------
    Map a small two-period panel (latest period by default):

    ```python
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    from geometrics.maps import explore_choropleth_map

    gdf = gpd.GeoDataFrame(
        {"region": ["a", "b", "c", "d"]},
        geometry=[box(i % 2, i // 2, i % 2 + 1, i // 2 + 1) for i in range(4)],
        crs="EPSG:4326",
    )
    df = pd.DataFrame(
        {
            "region": ["a", "b", "c", "d"] * 2,
            "year": [2000] * 4 + [2010] * 4,
            "gdppc": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
        }
    )
    res = explore_choropleth_map(
        df, "gdppc", gdf=gdf, entity="region", time="year", k=2, tiles=None
    )
    print(res.period, res.bins)
    ```
    """
    df = ensure_dataframe(df)
    gdf = ensure_geodataframe(gdf, func=_FUNC)
    entity, time = resolve_panel(df, entity, time, require_entity=True)
    assert entity is not None  # require_entity=True guarantees it
    hover_cols = [hover] if isinstance(hover, str) else list(hover or [])
    missing = [c for c in (var, *hover_cols) if c not in df.columns]
    if missing:
        raise KeyError(f"{_FUNC}: column(s) not found in df: {missing}")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"{_FUNC}: var {var!r} needs to be numeric")
    if animate and time is None:
        raise ValueError(
            f"{_FUNC}: animate=True needs a time id — pass time=... or call "
            "set_panel(df, time=...)"
        )

    notes: list[str] = []
    if animate and period is not None:
        message = "period= is ignored when animate=True (all periods are drawn)"
        warnings.warn(f"{_FUNC}: {message}", GeometricsWarning, stacklevel=2)
        notes.append(message)
        period = None

    var_label = resolve_label(df, var)
    hover_labels = [resolve_label(df, c) for c in hover_cols]
    display = entity_display_map(df, entity, resolve_entity_name(df))
    cols = [var, *hover_cols]
    # The aligned cross sections carry the geometry side's entity column.
    gdf_entity = resolve_gdf_entity(gdf)

    if animate:
        assert time is not None  # validated above
        return _animated(
            df,
            gdf,
            var,
            entity=entity,
            gdf_entity=gdf_entity,
            time=time,
            cols=cols,
            scheme=scheme,
            k=k,
            bins=bins,
            tiles=tiles,
            simplify=simplify,
            title=title,
            var_label=var_label,
            hover_cols=hover_cols,
            hover_labels=hover_labels,
            display=display,
            notes=notes,
        )

    cross, _, meta = _align_cross_section(
        df, gdf, cols, entity=entity, time=time, period=period, func=_FUNC
    )
    resolved_period = meta.get("period")
    notes.extend(str(n) for n in (meta.get("notes") or ()))
    if (
        period is None
        and time is not None
        and resolved_period is not None
        and not any("period" in n.lower() for n in notes)
    ):
        message = (
            f"period not specified — mapping the latest period ({resolved_period})"
        )
        warnings.warn(f"{_FUNC}: {message}", GeometricsWarning, stacklevel=2)
        notes.append(message)

    plotted = _wgs84_simplified(cross, gdf_entity, simplify, func=_FUNC)
    values = cross[var].to_numpy(dtype=float)
    if title is None:
        title = (
            f"{var_label} ({resolved_period})"
            if resolved_period is not None
            else var_label
        )
    fig, upper = classified_map(
        plotted,
        values,
        entity=gdf_entity,
        scheme=scheme,
        k=k,
        bins=list(bins) if bins is not None else None,
        tiles=tiles,
        title=title,
        legend_title=var_label,
        hover_names=display,
        simplify=None,
    )
    if hover_cols:
        extra = {
            str(row[gdf_entity]): [_hover_value(row[c]) for c in hover_cols]
            for _, row in cross.iterrows()
        }
        _append_hover(fig.data, extra, hover_labels)

    classes = _class_column(values, upper, float(values.min()))
    tidy = pd.DataFrame(
        {
            entity: cross[gdf_entity].to_numpy(),
            var: values,
            "class": classes.to_numpy(),
        }
    )
    plotted = plotted.assign(**{var: values, "class": classes.to_numpy()})
    return ChoroplethMapResult(
        df=tidy,
        fig=fig,
        gdf_plotted=plotted,
        var=var,
        period=resolved_period,
        scheme=scheme,
        k=len(upper) if upper else int(k),
        bins=upper,
        animated=False,
        notes=tuple(notes),
    )


def _animated(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    var: str,
    *,
    entity: str,
    gdf_entity: str,
    time: str,
    cols: list[str],
    scheme: str | None,
    k: int,
    bins: Sequence[float] | None,
    tiles: str | None,
    simplify: float | str | None,
    title: str | None,
    var_label: str,
    hover_cols: list[str],
    hover_labels: list[str],
    display: dict[str, str],
    notes: list[str],
) -> ChoroplethMapResult:
    """Build the multi-period frame animation behind ``animate=True``."""
    import mapclassify
    import plotly.graph_objects as go

    periods = sorted(pd.unique(df[time].dropna()))
    if not periods:
        raise ValueError(f"{_FUNC}: no periods found in time column {time!r}")

    # One aligned cross section per period; suppress per-period advisory warnings
    # (their notes are collected below instead of firing len(periods) times).
    sections: list[pd.DataFrame] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometricsWarning)
        for p in periods:
            cross, _, meta = _align_cross_section(
                df, gdf, cols, entity=entity, time=time, period=p, func=_FUNC
            )
            sections.append(cross)
            for n in meta.get("notes") or ():
                if str(n) not in notes:
                    notes.append(str(n))

    # Pooled class breaks: classify all periods together so colors are comparable.
    all_values = np.concatenate([c[var].to_numpy(dtype=float) for c in sections])
    if bins is not None:
        upper = tuple(
            float(b) for b in mapclassify.UserDefined(all_values, list(bins)).bins
        )
    elif scheme is not None:
        upper = tuple(
            float(b) for b in mapclassify.classify(all_values, scheme, k=k).bins
        )
    else:
        upper = ()
    pooled_min = float(all_values.min())
    labels = _class_labels(upper, pooled_min) if upper else []

    # Simplify the full geometry once; every frame subsets these WGS84 polygons.
    base = _wgs84_simplified(gdf, gdf_entity, simplify, func=_FUNC)

    frame_figs: list[go.Figure] = []
    tidy_parts: list[pd.DataFrame] = []
    for p, cross in zip(periods, sections, strict=True):
        plotted = base.merge(
            cross[[gdf_entity, *cols]].drop_duplicates(gdf_entity), on=gdf_entity
        )
        values = plotted[var].to_numpy(dtype=float)
        fig_p, _ = classified_map(
            plotted,
            values,
            entity=gdf_entity,
            scheme=scheme,
            k=k,
            bins=list(upper) if upper else None,
            tiles=tiles,
            legend_title=var_label,
            hover_names=display,
            simplify=None,
        )
        for i, trace in enumerate(fig_p.data):
            if labels:
                trace.name = labels[i]
            elif scheme is None:
                trace.update(zmin=pooled_min, zmax=float(all_values.max()))
        if hover_cols:
            extra = {
                str(row[gdf_entity]): [_hover_value(row[c]) for c in hover_cols]
                for _, row in plotted.iterrows()
            }
            _append_hover(fig_p.data, extra, hover_labels)
        frame_figs.append(fig_p)
        classes = _class_column(values, upper, pooled_min)
        tidy_parts.append(
            pd.DataFrame(
                {
                    entity: plotted[gdf_entity].to_numpy(),
                    time: p,
                    var: values,
                    "class": classes.to_numpy(),
                }
            )
        )

    fig = frame_figs[0]
    fig.frames = [
        go.Frame(data=frame.data, name=str(p))
        for frame, p in zip(frame_figs, periods, strict=True)
    ]
    frame_args = {
        "frame": {"duration": 700, "redraw": True},
        "transition": {"duration": 0},
    }
    step_args = {
        "mode": "immediate",
        "frame": {"duration": 0, "redraw": True},
        "transition": {"duration": 0},
    }
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "xanchor": "left",
                "y": -0.08,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {**frame_args, "fromcurrent": True}],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.16,
                "xanchor": "left",
                "y": -0.06,
                "yanchor": "top",
                "len": 0.84,
                "currentvalue": {"prefix": f"{time}: "},
                "steps": [
                    {
                        "label": str(p),
                        "method": "animate",
                        "args": [[str(p)], step_args],
                    }
                    for p in periods
                ],
            }
        ],
    )
    if title is None:
        title = f"{var_label} ({periods[0]} to {periods[-1]})"
    fig.update_layout(title={"text": title})

    message = (
        f"animated over {len(periods)} periods ({periods[0]} to {periods[-1]}) with "
        "pooled class breaks so colors are comparable across frames"
    )
    notes.append(message)

    first = tidy_parts[0]
    plotted_first = base.merge(
        sections[0][[gdf_entity, var]].drop_duplicates(gdf_entity), on=gdf_entity
    ).assign(**{"class": first["class"].to_numpy()})
    return ChoroplethMapResult(
        df=pd.concat(tidy_parts, ignore_index=True),
        fig=fig,
        gdf_plotted=plotted_first,
        var=var,
        period=periods[0],
        scheme=scheme,
        k=len(upper) if upper else int(k),
        bins=upper,
        animated=True,
        notes=tuple(notes),
    )
