"""Space-time descriptives: distribution dynamics and entity-by-time heatmaps.

:func:`explore_distribution_over_time` estimates one Gaussian kernel density of a
variable per period on a shared grid and draws the densities as a **ridgeline** (one
filled density per period, newest on top) or as a single **animated** density with a
period slider. With ``relative=True`` the variable is first divided by its
cross-sectional mean per period — the distribution-dynamics convention popularized by
giddy — so 1.0 marks the period average and the plot isolates changes in *shape*
(polarization, convergence clubs) from changes in the overall level.

:func:`explore_spacetime_heatmap` pivots the long panel to an entity-by-time matrix
and draws it as a heatmap: every unit keeps its own row, so persistence (rows keeping
their shading left to right) and mobility (rows changing shade) are visible unit by
unit. Rows can be ordered by mean value, alphabetically, or geographically
(north-south / east-west centroid order, which needs a ``gdf``).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt

from geometrics._common import entity_display_map
from geometrics._geo import ensure_metric_crs, resolve_gdf_entity
from geometrics._labels import resolve_label
from geometrics._panel import resolve_entity_name, resolve_panel
from geometrics._theme import MAP_SEQUENTIAL, apply_default_layout, color_for
from geometrics._types import DistributionOverTimeResult, SpacetimeHeatmapResult
from geometrics._validation import (
    GeometricsWarning,
    ensure_dataframe,
    ensure_geodataframe,
)

if TYPE_CHECKING:
    import geopandas as gpd

__all__ = ["explore_distribution_over_time", "explore_spacetime_heatmap"]

_FUNC_DIST = "explore_distribution_over_time"
_FUNC_HEAT = "explore_spacetime_heatmap"

#: Number of evaluation points on the shared density grid.
_GRID_POINTS = 512
#: Grid padding beyond the data range, in units of the widest kernel bandwidth, so
#: every period's density integrates to ~1 over the grid.
_PAD_BANDWIDTHS = 4.0
#: Vertical distance between ridgeline baselines, as a fraction of the tallest
#: density peak (< 1 so consecutive ridges overlap a little, joyplot style).
_RIDGE_STEP = 0.6

_SORTS = ("value", "name", "north_south", "east_west")

#: Max length for heatmap y-axis tick labels before eliding (full name stays in hover),
#: so a few long entity ids don't blow out the left margin.
_MAX_TICK_LABEL = 26


def _truncate_label(text: str, limit: int = _MAX_TICK_LABEL) -> str:
    """Shorten ``text`` to ``limit`` chars with an ellipsis (used for axis tick labels)."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _content_xrange(
    grid: np.ndarray,
    dens: dict[Any, np.ndarray],
    *,
    frac: float = 0.01,
    margin: float = 0.03,
) -> list[float]:
    """Trim the density grid to where mass actually lives (drop the long near-zero tails).

    Returns ``[lo, hi]`` spanning the grid points where any period's density exceeds
    ``frac`` of the global peak, plus a small ``margin``. Keeps a lone high outlier from
    stretching the x-axis so far that the bulk of the distribution collapses to the left.
    """
    peak = max(float(d.max()) for d in dens.values())
    thresh = frac * peak
    mask = np.zeros(grid.size, dtype=bool)
    for d in dens.values():
        mask |= np.asarray(d) >= thresh
    if not mask.any():
        return [float(grid[0]), float(grid[-1])]
    idx = np.flatnonzero(mask)
    pad = int(margin * grid.size)
    lo = max(int(idx[0]) - pad, 0)
    hi = min(int(idx[-1]) + pad, grid.size - 1)
    return [float(grid[lo]), float(grid[hi])]


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a ``#rrggbb`` hex string to an ``rgba(...)`` string with ``alpha``."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _complete_cases(
    df: pd.DataFrame, cols: list[str], notes: list[str], func: str
) -> pd.DataFrame:
    """Drop rows with missing values in ``cols``, warning and recording a note."""
    n_before = len(df)
    out = df.dropna(subset=cols)
    n_dropped = n_before - len(out)
    if n_dropped:
        pct = n_dropped / n_before if n_before else 0.0
        msg = (
            f"{func}: dropped {n_dropped} of {n_before} row(s) ({pct:.0%}) with "
            f"missing values in {cols}"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=3)
        notes.append(msg)
    return out


# ---------------------------------------------------------------------------
# explore_distribution_over_time
# ---------------------------------------------------------------------------


def explore_distribution_over_time(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    relative: bool = False,
    periods: Sequence[Any] | None = None,
    kind: Literal["ridgeline", "animated"] = "ridgeline",
    bandwidth: float | str | None = None,
    title: str | None = None,
) -> DistributionOverTimeResult:
    """Track how the cross-sectional distribution of one variable evolves over time.

    A Gaussian kernel density of ``var`` is estimated per period
    (:class:`scipy.stats.gaussian_kde`) and evaluated on a single grid shared by all
    periods, so the densities are directly comparable. ``kind="ridgeline"`` stacks
    one filled density per period with a subtle vertical offset (newest period on
    top); ``kind="animated"`` shows a single density animated over the periods with
    a play button and slider.

    Parameters
    ----------
    df
        Long panel holding ``var`` per entity and period.
    var
        Numeric column of ``df`` whose distribution is tracked.
    entity, time
        Panel identifiers; default to the ids declared via
        :func:`geometrics.set_panel`. A ``time`` id is required.
    relative
        Divide ``var`` by its cross-sectional mean per period before density
        estimation (the distribution-dynamics convention): 1.0 marks the period
        average and a dashed vertical line is drawn at 1.
    periods
        Subset of periods to include (default: all periods in ``df``). Unknown
        periods raise :class:`ValueError`.
    kind
        ``"ridgeline"`` (stacked filled densities, one trace per period) or
        ``"animated"`` (one density trace animated over periods with a slider).
    bandwidth
        Kernel bandwidth passed to :class:`scipy.stats.gaussian_kde` as
        ``bw_method`` (a scalar factor or ``"scott"`` / ``"silverman"``).
        ``None`` uses scipy's default (Scott's rule).
    title
        Figure title. Defaults to a description built from the variable label.

    Returns
    -------
    DistributionOverTimeResult
        Frozen result with the tidy evaluation frame ``df`` (columns ``time``,
        ``value``, ``density``), the themed ``fig``, and ``notes``.

    Raises
    ------
    KeyError
        If ``var`` is not a column of ``df``.
    TypeError
        If ``var`` is not numeric.
    ValueError
        If no ``time`` id resolves, ``kind`` is unknown, a requested period is
        absent, or a period has fewer than 2 distinct values.

    Examples
    --------
    Ridgeline of a small two-period panel:

    ```python
    import pandas as pd

    from geometrics.spacetime import explore_distribution_over_time

    df = pd.DataFrame(
        {
            "region": list("abcdefgh") * 2,
            "year": [2000] * 8 + [2010] * 8,
            "gdppc": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            + [2.0, 2.5, 3.5, 4.5, 5.0, 6.5, 7.0, 7.5],
        }
    )
    res = explore_distribution_over_time(df, "gdppc", entity="region", time="year")
    len(res.fig.data)
    ```
    """
    df = ensure_dataframe(df)
    _, time = resolve_panel(df, entity, time, require_time=True)
    assert time is not None  # require_time=True guarantees it
    if var not in df.columns:
        raise KeyError(f"{_FUNC_DIST}: column {var!r} not found in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"{_FUNC_DIST}: var {var!r} needs to be numeric")
    if kind not in ("ridgeline", "animated"):
        raise ValueError(
            f"{_FUNC_DIST}: kind needs to be 'ridgeline' or 'animated', got {kind!r}"
        )

    notes: list[str] = []
    sub = df[[time, var]].copy()
    sub = _complete_cases(sub, [time, var], notes, _FUNC_DIST)

    available = sorted(pd.unique(sub[time]).tolist())
    if not available:
        raise ValueError(
            f"{_FUNC_DIST}: time column {time!r} has no non-missing periods"
        )
    if periods is None:
        use = available
    else:
        unknown = [p for p in periods if p not in set(available)]
        if unknown:
            raise ValueError(
                f"{_FUNC_DIST}: period(s) {unknown} not found in {time!r}; "
                f"available periods: {available}"
            )
        use = sorted(dict.fromkeys(periods))

    # Per-period samples (relative = divided by the period's cross-sectional mean).
    samples: dict[Any, np.ndarray] = {}
    for p in use:
        vals = sub.loc[sub[time] == p, var].to_numpy(dtype=float)
        if vals.size < 2 or np.ptp(vals) == 0.0:
            raise ValueError(
                f"{_FUNC_DIST}: {var!r} needs at least 2 distinct values in period "
                f"{p!r} to estimate a density"
            )
        if relative:
            mean = float(vals.mean())
            if mean == 0.0:
                raise ValueError(
                    f"{_FUNC_DIST}: the cross-sectional mean of {var!r} in period "
                    f"{p!r} is zero — relative=True is undefined"
                )
            vals = vals / mean
        samples[p] = vals

    # One KDE per period, evaluated on a grid shared by all periods and padded by a
    # few bandwidths so each density integrates to ~1 over the grid.
    from scipy.stats import gaussian_kde

    kdes = {p: gaussian_kde(v, bw_method=bandwidth) for p, v in samples.items()}
    bw_max = max(float(np.sqrt(k.covariance[0, 0])) for k in kdes.values())
    pad = _PAD_BANDWIDTHS * bw_max
    lo = min(float(v.min()) for v in samples.values()) - pad
    hi = max(float(v.max()) for v in samples.values()) + pad
    grid = np.linspace(lo, hi, _GRID_POINTS)
    dens = {p: np.asarray(kdes[p](grid), dtype=float) for p in use}

    tidy = pd.concat(
        [pd.DataFrame({"time": p, "value": grid, "density": dens[p]}) for p in use],
        ignore_index=True,
    )

    var_label = resolve_label(df, var)
    time_label = resolve_label(df, time)
    x_label = f"{var_label} (relative to the period mean)" if relative else var_label
    if title is None:
        title = f"Distribution of {var_label} by {time_label}"
        if relative:
            title += " (relative to the period mean)"

    if kind == "ridgeline":
        fig = _ridgeline_figure(use, grid, dens, x_label=x_label, time_label=time_label)
    else:
        fig = _animated_figure(use, grid, dens, x_label=x_label, time_label=time_label)
    if relative:
        fig.add_vline(x=1.0, line_dash="dash", line_color="#888888", line_width=1)
    apply_default_layout(fig, title=title)

    return DistributionOverTimeResult(
        df=tidy,
        fig=fig,
        var=var,
        kind=kind,
        relative=relative,
        notes=tuple(notes),
    )


def _ridgeline_figure(
    periods: list[Any],
    grid: np.ndarray,
    dens: dict[Any, np.ndarray],
    *,
    x_label: str,
    time_label: str,
) -> go.Figure:
    """Stack one filled density per period with a subtle vertical offset.

    Baselines rise with recency so the newest period sits on top; traces are added
    newest first so the older (lower) ridges paint over the tails of the ones above,
    the classic joyplot layering.
    """
    peak = max(float(d.max()) for d in dens.values())
    step = _RIDGE_STEP * peak
    baseline = {p: i * step for i, p in enumerate(periods)}

    fig = go.Figure()
    for i, p in reversed(list(enumerate(periods))):
        d = dens[p]
        color = color_for(i)
        # A closed polygon (density curve out, flat baseline back) so the fill sits
        # on the period's own baseline rather than on y=0.
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([grid, grid[::-1]]),
                y=np.concatenate([d + baseline[p], np.full(grid.size, baseline[p])]),
                mode="lines",
                line={"color": color, "width": 1.5},
                fill="toself",
                fillcolor=_rgba(color, 0.55),
                name=str(p),
                showlegend=False,
                customdata=np.concatenate([d, d[::-1]]),
                hovertemplate=(
                    f"{time_label}={p}<br>{x_label}=%{{x:.4g}}"
                    "<br>density=%{customdata:.4g}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        xaxis={"title": {"text": x_label}, "range": _content_xrange(grid, dens)},
        yaxis={
            "title": {"text": time_label},
            "tickmode": "array",
            "tickvals": [baseline[p] for p in periods],
            "ticktext": [str(p) for p in periods],
            "showgrid": False,
            "zeroline": False,
        },
    )
    return fig


def _animated_figure(
    periods: list[Any],
    grid: np.ndarray,
    dens: dict[Any, np.ndarray],
    *,
    x_label: str,
    time_label: str,
) -> go.Figure:
    """Animate a single filled density over the periods with a play button and slider.

    Axis ranges are fixed across frames so motion reflects the distribution, not a
    rescaling of the axes.
    """
    peak = max(float(d.max()) for d in dens.values())
    color = color_for(0)

    def _trace(p: Any) -> go.Scatter:
        return go.Scatter(
            x=grid,
            y=dens[p],
            mode="lines",
            line={"color": color, "width": 2},
            fill="tozeroy",
            fillcolor=_rgba(color, 0.35),
            name=str(p),
            showlegend=False,
            hovertemplate=(
                f"{x_label}=%{{x:.4g}}<br>density=%{{y:.4g}}<extra></extra>"
            ),
        )

    fig = go.Figure(
        data=[_trace(periods[0])],
        frames=[go.Frame(data=[_trace(p)], name=str(p)) for p in periods],
    )
    frame_args = {
        "frame": {"duration": 700, "redraw": False},
        "transition": {"duration": 250},
    }
    step_args = {
        "mode": "immediate",
        "frame": {"duration": 0, "redraw": False},
        "transition": {"duration": 0},
    }
    fig.update_layout(
        # Bottom margin so the Play button + slider (placed just below the axis) have room.
        margin={"b": 120},
        xaxis={"title": {"text": x_label}, "range": _content_xrange(grid, dens)},
        yaxis={"title": {"text": "Density"}, "range": [0.0, 1.08 * peak]},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "xanchor": "left",
                "y": -0.22,
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
                "y": -0.2,
                "yanchor": "top",
                "len": 0.84,
                "currentvalue": {"prefix": f"{time_label}: "},
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
    return fig


# ---------------------------------------------------------------------------
# explore_spacetime_heatmap
# ---------------------------------------------------------------------------


def explore_spacetime_heatmap(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    gdf: gpd.GeoDataFrame | None = None,
    sort_by: Literal["value", "name", "north_south", "east_west"] = "value",
    relative: bool = False,
    title: str | None = None,
) -> SpacetimeHeatmapResult:
    """Draw one variable as an entity-by-time heatmap (every unit keeps its row).

    The long panel is pivoted to one row per entity and one column per period and
    drawn as a heatmap on the library's sequential scale, so persistence (rows that
    keep their shading left to right) and mobility (rows that change shade) are
    visible unit by unit.

    Parameters
    ----------
    df
        Long panel holding ``var`` per entity and period.
    var
        Numeric column of ``df`` to draw.
    entity, time
        Panel identifiers; default to the ids declared via
        :func:`geometrics.set_panel`. Both are required.
    gdf
        Entity geometry, required for the geographic row orders
        (``sort_by="north_south"`` / ``"east_west"``); entities are matched on the
        gdf's entity-id column.
    sort_by
        Row order: ``"value"`` (mean value per unit, highest first), ``"name"``
        (alphabetical), ``"north_south"`` (metric-CRS centroid latitude, north
        first) or ``"east_west"`` (centroid longitude, west first).
    relative
        Divide each column by its period mean, so 1.0 marks the period average and
        shading compares units within a period rather than tracking the level.
    title
        Figure title. Defaults to a description built from the variable label.

    Returns
    -------
    SpacetimeHeatmapResult
        Frozen result with ``df`` (the entity-by-time pivot in display order,
        entities as the index), the themed ``fig``, and ``notes``.

    Raises
    ------
    KeyError
        If ``var`` is not a column of ``df``.
    TypeError
        If ``var`` is not numeric, or ``gdf`` is not a GeoDataFrame.
    ValueError
        If the panel ids do not resolve, ``sort_by`` is unknown, a geographic sort
        is requested without a ``gdf``, or no complete observations remain.

    Examples
    --------
    Heatmap of a small two-period panel:

    ```python
    import pandas as pd

    from geometrics.spacetime import explore_spacetime_heatmap

    df = pd.DataFrame(
        {
            "region": ["a", "b", "a", "b"],
            "year": [2000, 2000, 2010, 2010],
            "gdppc": [1.0, 2.0, 1.5, 2.5],
        }
    )
    res = explore_spacetime_heatmap(df, "gdppc", entity="region", time="year")
    res.df.shape
    ```
    """
    df = ensure_dataframe(df)
    if gdf is not None:
        gdf = ensure_geodataframe(gdf, func=_FUNC_HEAT)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # required above
    if var not in df.columns:
        raise KeyError(f"{_FUNC_HEAT}: column {var!r} not found in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"{_FUNC_HEAT}: var {var!r} needs to be numeric")
    if sort_by not in _SORTS:
        raise ValueError(
            f"{_FUNC_HEAT}: sort_by needs to be one of {list(_SORTS)}, got {sort_by!r}"
        )
    if sort_by in ("north_south", "east_west") and gdf is None:
        raise ValueError(
            f"{_FUNC_HEAT}: sort_by={sort_by!r} orders rows by geographic "
            "centroid — pass gdf=..."
        )

    notes: list[str] = []
    display = entity_display_map(df, entity, resolve_entity_name(df))

    sub = df[[entity, time, var]].copy()
    sub = _complete_cases(sub, [entity, time, var], notes, _FUNC_HEAT)
    dup = sub.duplicated(subset=[entity, time])
    if bool(dup.any()):
        sub = sub.drop_duplicates(subset=[entity, time], keep="first")
        msg = (
            f"{_FUNC_HEAT}: kept the first of {int(dup.sum())} duplicate "
            f"({entity!r}, {time!r}) row(s)"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)
    if sub.empty:
        raise ValueError(f"{_FUNC_HEAT}: no complete observations of {var!r}")

    wide = sub.pivot(index=entity, columns=time, values=var)
    periods = sorted(wide.columns.tolist())
    wide = wide.reindex(columns=periods)
    if bool(wide.isna().any().any()):
        n_gaps = int(wide.isna().sum().sum())
        msg = (
            f"{_FUNC_HEAT}: the panel is unbalanced — {n_gaps} missing cell(s) are "
            "left blank in the heatmap"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)

    if relative:
        col_means = wide.mean(axis=0)
        bad = [p for p, m in col_means.items() if not np.isfinite(m) or m == 0.0]
        if bad:
            raise ValueError(
                f"{_FUNC_HEAT}: relative=True is undefined — the period mean of "
                f"{var!r} is zero or undefined in period(s) {bad}"
            )
        wide = wide / col_means  # divides each column by its own period mean

    if sort_by == "value":
        order = wide.mean(axis=1).sort_values(ascending=False, kind="stable").index
        wide = wide.loc[order]
    elif sort_by == "name":
        wide = wide.loc[sorted(wide.index, key=str)]
    else:
        assert gdf is not None  # validated above
        wide = _geographic_order(wide, gdf, sort_by, entity=entity, notes=notes)

    var_label = resolve_label(df, var)
    time_label = resolve_label(df, time)
    z_label = f"{var_label} / period mean" if relative else var_label
    if title is None:
        title = f"Space-time heatmap of {var_label}"
        if relative:
            title += " (relative to the period mean)"

    full_labels = [display.get(str(i), str(i)) for i in wide.index]
    x_labels = [str(p) for p in periods]
    # Elide long entity ids for the axis (so a few very long names don't blow out the
    # left margin and detach the left-anchored title) but keep the full name in hover.
    # Where truncation would collide, keep the full label so heatmap rows never merge —
    # Plotly still auto-thins which labels are drawn.
    trunc = [_truncate_label(lbl) for lbl in full_labels]
    y_labels = [
        full if trunc.count(t) > 1 else t
        for full, t in zip(full_labels, trunc, strict=True)
    ]
    customdata = np.tile(
        np.asarray(full_labels, dtype=object)[:, None], (1, len(x_labels))
    )
    z = wide.to_numpy(dtype=float)
    # Robust color range: clip to the 2nd/98th percentiles so a handful of outliers
    # don't compress every other cell to near-white.
    finite = z[np.isfinite(z)]
    zmin = float(np.percentile(finite, 2)) if finite.size else None
    zmax = float(np.percentile(finite, 98)) if finite.size else None
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            zmin=zmin,
            zmax=zmax,
            colorscale=MAP_SEQUENTIAL,
            colorbar={"title": {"text": z_label}},
            customdata=customdata,
            hoverongaps=False,
            hovertemplate=(
                f"%{{customdata}}<br>{time_label}: %{{x}}"
                f"<br>{z_label}: %{{z:.4g}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis={
            "title": {"text": time_label},
            "type": "category",
            "categoryorder": "array",
            "categoryarray": x_labels,
        },
        # Reversed so the first row of the pivot is drawn at the top of the map.
        yaxis={"title": {"text": ""}, "autorange": "reversed"},
    )
    apply_default_layout(fig, title=title)

    return SpacetimeHeatmapResult(
        df=wide,
        fig=fig,
        var=var,
        sort_by=sort_by,
        relative=relative,
        notes=tuple(notes),
    )


def _geographic_order(
    wide: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    sort_by: str,
    *,
    entity: str,
    notes: list[str],
) -> pd.DataFrame:
    """Order the pivot rows by metric-CRS centroid latitude/longitude.

    Entities are matched to ``gdf`` with a plain merge on the entity ids; units the
    geometry does not cover cannot be positioned and are dropped with a warning.
    ``"north_south"`` puts the northernmost unit first, ``"east_west"`` the
    westernmost.
    """
    gdf_entity = resolve_gdf_entity(gdf)
    metric = ensure_metric_crs(gdf, func=_FUNC_HEAT)
    cent = metric.geometry.centroid
    coords = pd.DataFrame(
        {
            "_gm_key": metric[gdf_entity].to_numpy(),
            "_gm_y": cent.y.to_numpy(),
            "_gm_x": cent.x.to_numpy(),
        }
    )
    keyed = wide.reset_index().merge(
        coords, how="left", left_on=entity, right_on="_gm_key"
    )
    unmatched = keyed["_gm_key"].isna()
    if bool(unmatched.all()):
        raise ValueError(
            f"{_FUNC_HEAT}: df.{entity} and gdf.{gdf_entity} share no ids — cannot "
            "order rows geographically"
        )
    if bool(unmatched.any()):
        examples = list(keyed.loc[unmatched, entity].astype(str).head(5))
        msg = (
            f"{_FUNC_HEAT}: dropped {int(unmatched.sum())} entit(ies) not found in "
            f"gdf.{gdf_entity} (e.g. {examples}) — they cannot be ordered "
            "geographically"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=3)
        notes.append(msg)
        keyed = keyed.loc[~unmatched]
    key, ascending = ("_gm_y", False) if sort_by == "north_south" else ("_gm_x", True)
    keyed = keyed.sort_values(key, ascending=ascending, kind="stable")
    ordered = keyed.drop(columns=["_gm_key", "_gm_y", "_gm_x"]).set_index(entity)
    ordered.columns.name = wide.columns.name
    return ordered
