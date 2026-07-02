"""Regional inequality over time: level measures, trends, and decompositions.

:func:`analyze_inequality_over_time` tracks how *cross-sectional* inequality in a panel
variable evolves: for every period it computes the requested measures — the **Gini
index**, the **Theil index** and the **coefficient of variation** — and then tests
whether inequality is narrowing by regressing the **log** of each measure on time
(a negative slope is falling inequality, the inequality-narrative complement of
σ-convergence). When geometry is supplied, the per-period **spatial Gini
decomposition** of Rey & Smith (2013) is added: the share of overall inequality owed
to *neighbor* pairs, with permutation-based inference on the non-neighbor component.

:func:`analyze_theil_decomposition` splits the Theil index **between** and **within**
the groups of a partition column (e.g. states containing districts) per period, with
optional permutation inference on the between-group component. The additive identity
``T = between + within`` holds exactly, so the ``between_share`` traces how much of
total inequality is a *regional* (group-level) phenomenon.

Both functions require the measured variable on a strictly positive scale for
entropy-based measures (the Theil index takes logarithms of shares); the Gini index
is tolerant of zeros.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt
from plotly.subplots import make_subplots

from geometrics._geo import _align_cross_section, resolve_gdf_entity
from geometrics._labels import resolve_label
from geometrics._panel import resolve_panel
from geometrics._theme import apply_default_layout, color_for
from geometrics._types import InequalityOverTimeResult, TheilDecompositionResult
from geometrics._validation import (
    GeometricsWarning,
    ensure_dataframe,
    ensure_geodataframe,
)
from geometrics.weights import _default_weights, _describe_w

if TYPE_CHECKING:
    import geopandas as gpd
    from great_tables import GT
    from libpysal.weights import W

__all__ = ["analyze_inequality_over_time", "analyze_theil_decomposition"]

_MEASURES = ("gini", "theil", "cv")
_MEASURE_LABELS = {
    "gini": "Gini index",
    "theil": "Theil index",
    "cv": "Coefficient of variation",
    "gini_spatial": "Spatial Gini (neighbor component)",
}

#: Fixed seed for the spatial-Gini permutation inference (``Gini_Spatial`` draws from
#: NumPy's global RNG and exposes no seed parameter, so the p-values are made
#: reproducible by seeding once per call).
_SPATIAL_GINI_SEED = 12345


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _measure_value(measure: str, y: np.ndarray) -> float:
    """Compute one inequality measure of the cross-section ``y`` (``nan`` if undefined).

    ``gini`` uses :class:`inequality.gini.Gini` (attribute ``.g``), ``theil``
    :class:`inequality.theil.Theil` (attribute ``.T``), and ``cv`` is the sample
    standard deviation (``ddof=1``) over the mean.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if measure == "gini":
            from inequality.gini import Gini

            value = float(Gini(y).g)
        elif measure == "theil":
            from inequality.theil import Theil

            value = float(Theil(y).T)
        else:
            mean = float(np.mean(y))
            value = float(np.std(y, ddof=1)) / mean if mean != 0.0 else float("nan")
    return value if math.isfinite(value) else float("nan")


def _positional_binary_w(ids: list[Any], w: W) -> W:
    """Rebuild ``w`` as a binary W keyed by the row position of ``ids``.

    :class:`inequality.gini.Gini_Spatial` indexes the value array *positionally* with
    the neighbor keys and counts neighbor pairs from ``w.s0``, so it needs a binary
    weights object whose ids are ``0..n-1`` in the row order of the data. Only the
    neighbor structure of ``w`` is used (weight values are discarded).
    """
    from libpysal.weights import W as _W

    pos = {eid: k for k, eid in enumerate(ids)}
    neighbors = {pos[i]: [pos[j] for j in w.neighbors[i]] for i in ids}
    return _W(neighbors, id_order=list(range(len(ids))), silence_warnings=True)


def _numeric_time(periods: list[Any]) -> tuple[np.ndarray, bool]:
    """Return a numeric time axis for the trend fit (rank fallback for labels).

    Returns ``(t, ranked)`` where ``t`` is the numeric time value per period and
    ``ranked`` flags that non-numeric period labels were replaced by their rank.
    """
    t = pd.to_numeric(pd.Series(periods), errors="coerce")
    if bool(t.notna().all()):
        return t.to_numpy(dtype=float), False
    return np.arange(len(periods), dtype=float), True


def _log_trend(
    t: np.ndarray, values: np.ndarray
) -> tuple[Any | None, float, float, float, float, int]:
    """Fit ``log(values) ~ t`` by OLS with HC1 standard errors (statsmodels).

    Returns ``(model, slope, se, pvalue, r2, n_used)``; the model is ``None`` and the
    scalars ``nan`` when fewer than three periods have a positive, finite value (the
    log needs a strictly positive measure). A **negative** slope is falling
    inequality.
    """
    import statsmodels.api as sm

    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v) & (v > 0.0) & np.isfinite(t)
    n_used = int(ok.sum())
    nan = float("nan")
    if n_used < 3:
        return None, nan, nan, nan, nan, n_used
    design = sm.add_constant(t[ok])
    model = sm.OLS(np.log(v[ok]), design).fit(cov_type="HC1")
    with np.errstate(divide="ignore", invalid="ignore"):
        # rsquared divides by the centered TSS, which is zero for a flat measure.
        r2 = float(model.rsquared)
    return (
        model,
        float(model.params[1]),
        float(model.bse[1]),
        float(model.pvalues[1]),
        r2 if math.isfinite(r2) else float("nan"),
        n_used,
    )


def _fmt(value: float, *, digits: int = 4) -> str:
    """Format a table cell compactly, with an em-dash for missing values."""
    return "—" if not math.isfinite(value) else f"{value:.{digits}g}"


def _first_ids(values: Any, k: int = 5) -> list[str]:
    """Return up to ``k`` distinct ids of ``values`` as strings (for error messages)."""
    return list(dict.fromkeys(str(v) for v in values))[:k]


def _require_positive(
    work: pd.DataFrame, var: str, entity: str, *, func: str, why: str
) -> None:
    """Raise ``ValueError`` naming the entities where ``var`` is not strictly positive."""
    bad = work.loc[work[var] <= 0.0, entity]
    if len(bad):
        raise ValueError(
            f"{func}: {why} needs strictly positive {var!r} values, but "
            f"{len(bad)} row(s) are <= 0 (entities: {_first_ids(bad)})"
        )


def _drop_var_missing(
    work: pd.DataFrame, cols: list[str], notes: list[str], *, func: str
) -> pd.DataFrame:
    """Complete-case filter on ``cols`` with an advisory warning + note when rows drop."""
    n_missing = int(work[cols].isna().any(axis=1).sum())
    if n_missing:
        msg = (
            f"{func}: dropped {n_missing} of {len(work)} row(s) with missing values "
            f"in {cols} (complete cases per period)"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=3)
        notes.append(msg)
        work = work.dropna(subset=cols)
    return work


def _dedupe_first(
    work: pd.DataFrame, keys: list[str], notes: list[str], *, func: str
) -> pd.DataFrame:
    """Keep the first of duplicate ``keys`` rows, recording a note when any drop."""
    dup = work.duplicated(subset=keys)
    if bool(dup.any()):
        work = work.drop_duplicates(subset=keys, keep="first")
        notes.append(
            f"{func}: kept the first of {int(dup.sum())} duplicate row(s) per {keys}"
        )
    return work


def _window_periods(
    periods: list[Any], start: Any, end: Any, *, func: str
) -> list[Any]:
    """Restrict sorted ``periods`` to the ``[start, end]`` window (inclusive)."""
    out = [
        p
        for p in periods
        if (start is None or p >= start) and (end is None or p <= end)
    ]
    if not out:
        raise ValueError(
            f"{func}: no periods left in the window start={start!r}, end={end!r}; "
            f"available periods: {periods}"
        )
    return out


# ---------------------------------------------------------------------------
# analyze_inequality_over_time
# ---------------------------------------------------------------------------


def _scale_groups(tab: pd.DataFrame, series: list[str]) -> dict[str, bool]:
    """Assign each plotted series to the primary/secondary axis by scale.

    The first series anchors the primary axis; a later series moves to the secondary
    axis when its median level differs from the anchor's by more than a factor of 5.
    """
    secondary: dict[str, bool] = {}
    anchor = float("nan")
    for name in series:
        med = float(np.nanmedian(np.abs(tab[name].to_numpy(dtype=float))))
        if not math.isfinite(anchor):
            anchor = med if math.isfinite(med) and med > 0 else anchor
            secondary[name] = False
            continue
        if math.isfinite(med) and med > 0:
            ratio = med / anchor
            secondary[name] = bool(ratio > 5.0 or ratio < 0.2)
        else:
            secondary[name] = False
    return secondary


def _inequality_fig(
    tab: pd.DataFrame,
    measures: tuple[str, ...],
    trends: dict[str, tuple[Any | None, float, float, float, float, int]],
    t_num: np.ndarray,
    *,
    time_label: str,
    var_label: str,
    w_spec: str | None,
    title: str | None,
) -> go.Figure:
    """Build the measures-over-time figure with dashed fitted log-trends."""
    series = [m for m in measures if m in tab.columns]
    if "gini_spatial" in tab.columns:
        series.append("gini_spatial")
    secondary = _scale_groups(tab, series)
    if "gini_spatial" in secondary and "gini" in secondary:
        # The neighbor component lives on the Gini scale; keep them on one axis.
        secondary["gini_spatial"] = secondary["gini"]
    use_secondary = any(secondary.values())
    fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])
    x = tab["time"].tolist()

    for i, name in enumerate(series):
        label = _MEASURE_LABELS.get(name, name)
        color = color_for(i)
        dash = "dot" if name == "gini_spatial" else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=tab[name].to_numpy(dtype=float),
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 2, "dash": dash},
                marker={"color": color, "size": 7},
                hovertemplate=(
                    f"{time_label} = %{{x}}<br>{label} = %{{y:.4g}}<extra></extra>"
                ),
            ),
            secondary_y=secondary[name] if use_secondary else None,
        )
        trend = trends.get(name)
        if trend is not None and trend[0] is not None:
            b0 = float(trend[0].params[0])
            b1 = trend[1]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.exp(b0 + b1 * t_num),
                    mode="lines",
                    line={"color": color, "width": 1, "dash": "dash"},
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{label} trend",
                ),
                secondary_y=secondary[name] if use_secondary else None,
            )

    primary = [_MEASURE_LABELS.get(s, s) for s in series if not secondary[s]]
    fig.update_yaxes(title_text=", ".join(primary), secondary_y=False)
    if use_secondary:
        second = [_MEASURE_LABELS.get(s, s) for s in series if secondary[s]]
        fig.update_yaxes(title_text=", ".join(second), secondary_y=True)
    apply_default_layout(
        fig,
        title=title
        if title is not None
        else f"Regional inequality over time: {var_label}",
        subtitle=w_spec,
        xaxis={"title": time_label},
        # Horizontal legend under the plot so it clears the right (secondary) axis title.
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
        },
        margin_b=104,
    )
    return fig


def _inequality_summary_and_gt(
    measures: tuple[str, ...],
    trends: dict[str, tuple[Any | None, float, float, float, float, int]],
    *,
    var_label: str,
    n_periods: int,
    n_units: int,
    w_spec: str | None,
) -> tuple[pd.DataFrame, GT]:
    """Build the per-measure trend ``summary`` frame and its Great Tables rendering."""
    from great_tables import GT

    def converging(measure: str) -> bool:
        slope, pvalue = trends[measure][1], trends[measure][3]
        return bool(
            math.isfinite(slope)
            and math.isfinite(pvalue)
            and slope < 0.0
            and pvalue < 0.05
        )

    summary = pd.DataFrame(
        {
            "measure": list(measures),
            "slope": [trends[m][1] for m in measures],
            "se": [trends[m][2] for m in measures],
            "pvalue": [trends[m][3] for m in measures],
            "r2": [trends[m][4] for m in measures],
            "converging": [converging(m) for m in measures],
        }
    )

    disp = pd.DataFrame(
        {
            "Measure": [_MEASURE_LABELS.get(m, m) for m in measures],
            "Trend (per period)": [_fmt(trends[m][1]) for m in measures],
            "Std. error": [_fmt(trends[m][2]) for m in measures],
            "p-value": [_fmt(trends[m][3], digits=3) for m in measures],
            "R²": [_fmt(trends[m][4], digits=3) for m in measures],
            "Narrowing": [
                "—"
                if not math.isfinite(trends[m][1])
                else ("yes" if converging(m) else "no")
                for m in measures
            ],
        }
    )
    subtitle = f"{n_periods} periods, {n_units} units"
    if w_spec is not None:
        subtitle += f" — W: {w_spec}"
    gt = (
        GT(disp, rowname_col="Measure")
        .tab_header(title=f"Inequality trends: {var_label}", subtitle=subtitle)
        .tab_source_note(
            "Trend = OLS slope of ln(measure) on time with HC1 standard errors. A "
            "negative, significant slope means cross-sectional inequality is "
            "narrowing over time."
        )
    )
    return summary, gt


def analyze_inequality_over_time(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    measures: Sequence[str] = ("gini", "theil"),
    gdf: gpd.GeoDataFrame | None = None,
    w: W | None = None,
    permutations: int = 99,
    start: Any = None,
    end: Any = None,
    title: str | None = None,
) -> InequalityOverTimeResult:
    """Track cross-sectional inequality measures over time and test their trend.

    For every period the function computes the requested inequality measures of
    ``var`` across units — the **Gini index** (:class:`inequality.gini.Gini`), the
    **Theil index** (:class:`inequality.theil.Theil`) and the **coefficient of
    variation** (sample std over mean) — then regresses the **log** of each measure
    on time (OLS, HC1 standard errors). A negative, significant slope means
    inequality is narrowing: the inequality-narrative complement of σ-convergence.

    When geometry is supplied (``gdf``, with ``w`` optional), the per-period
    **spatial Gini decomposition** of Rey & Smith (2013)
    (:class:`inequality.gini.Gini_Spatial`) is added: ``gini_spatial`` is the
    component of the overall Gini owed to *neighbor* pairs under ``w`` (so
    ``gini_spatial <= gini``, with the remainder owed to non-neighbor pairs), and
    ``gini_spatial_p`` is the permutation pseudo p-value testing whether the
    non-neighbor component exceeds its expectation under spatial randomness. Units
    are aligned to the geometry per period with the **same** entity set across
    periods (the intersection of the per-period complete cases).

    Parameters
    ----------
    df
        Long-form panel data frame.
    var
        Numeric variable whose cross-sectional inequality is tracked (e.g. GDP per
        capita in levels). Used as supplied; the Theil index requires strictly
        positive values.
    entity, time
        Panel identifiers. Default to those declared via
        :func:`geometrics.set_panel`.
    measures
        Measures to compute per period, from ``"gini"``, ``"theil"`` and ``"cv"``.
    gdf
        Geometry frame (see :func:`geometrics.read_gdf`) enabling the spatial Gini
        decomposition. ``None`` skips it.
    w
        ``libpysal`` weights aligned to the ``gdf`` entity ids (only its neighbor
        structure is used, as a binary graph). ``None`` with a ``gdf`` builds the
        default weights (queen contiguity for polygons) with a
        :class:`~geometrics.GeometricsWarning`.
    permutations
        Number of permutations for the spatial-Gini inference (``0`` disables it;
        ``gini_spatial_p`` is then ``NaN``).
    start, end
        Optional first and last period to include (inclusive, on the scale of the
        time column). Default to the full range.
    title
        Title for the figure.

    Returns
    -------
    InequalityOverTimeResult
        Per-period measures ``df`` (``time``, ``n_units`` and one column per
        measure, plus ``gini_spatial`` / ``gini_spatial_p`` when spatial); the
        measures-over-time ``fig`` with dashed fitted trends; the per-measure trend
        table ``gt`` / ``summary`` (``measure``, ``slope``, ``se``, ``pvalue``,
        ``r2``, ``converging``); the fitted trend ``models``; ``n_periods`` /
        ``n_units``; and ``w_spec`` describing the weights (``None`` without
        geometry).

    Raises
    ------
    KeyError
        If ``var`` is not a column of ``df``.
    TypeError
        If ``var`` is not numeric.
    ValueError
        For unknown ``measures``, a Theil request on non-positive values (the
        offending entities are named), ``w`` without ``gdf``, fewer than two
        periods, or a period with fewer than two complete observations.

    Notes
    -----
    ``Gini_Spatial`` draws permutations from NumPy's global RNG and has no seed
    parameter, so the global seed is set to a fixed value (12345) at the start of
    the spatial loop to make ``gini_spatial_p`` reproducible.

    Examples
    --------
    Inequality trend across three regions over three years:

    ```python
    import pandas as pd

    from geometrics.regional_inequality import analyze_inequality_over_time

    df = pd.DataFrame(
        {
            "region": ["A", "B", "C"] * 3,
            "year": [2000] * 3 + [2001] * 3 + [2002] * 3,
            "gdppc": [10.0, 20.0, 40.0, 12.0, 21.0, 38.0, 14.0, 22.0, 36.0],
        }
    )
    res = analyze_inequality_over_time(
        df, "gdppc", entity="region", time="year", measures=("gini", "theil")
    )
    (res.df["gini"].round(3).tolist(), bool(res.summary.loc[0, "converging"]))
    ```
    """
    func = "analyze_inequality_over_time"
    df = ensure_dataframe(df)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # guaranteed by require_* above
    if var not in df.columns:
        raise KeyError(f"{func}: column {var!r} not found in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"{func}: {var!r} needs to be numeric")
    measures = tuple(str(m).lower() for m in measures)
    unknown = [m for m in measures if m not in _MEASURES]
    if not measures or unknown:
        raise ValueError(
            f"{func}: unknown measure(s) {unknown}; choose from {list(_MEASURES)}"
        )
    if w is not None and gdf is None:
        raise ValueError(
            f"{func}: w was given without gdf — pass the geometry so units can be "
            "aligned to the weights"
        )
    if gdf is not None:
        gdf = ensure_geodataframe(gdf, func=func)

    var_label = resolve_label(df, var)
    time_label = resolve_label(df, time)
    notes: list[str] = []

    work = df[[entity, time, var]].copy()
    work = work.dropna(subset=[entity, time])
    if work.empty:
        raise ValueError(f"{func}: no rows with non-missing {entity!r} and {time!r}")
    periods = _window_periods(
        sorted(pd.unique(work[time]).tolist()), start, end, func=func
    )
    work = work.loc[work[time].isin(periods)]
    work = _dedupe_first(work, [entity, time], notes, func=func)
    work = _drop_var_missing(work, [var], notes, func=func)
    periods = [p for p in periods if p in set(work[time])]
    if len(periods) < 2:
        raise ValueError(
            f"{func}: needs at least 2 periods to track inequality over time; "
            f"got {len(periods)}"
        )

    counts = work.groupby(time, observed=True)[entity].nunique()
    thin = counts[counts < 2]
    if len(thin):
        raise ValueError(
            f"{func}: period(s) {_first_ids(thin.index)} have fewer than 2 complete "
            "observations — inequality needs a cross-section per period"
        )

    if "theil" in measures:
        _require_positive(work, var, entity, func=func, why="the Theil index")

    sets = {p: set(work.loc[work[time] == p, entity]) for p in periods}
    union: set[Any] = set().union(*sets.values())
    common: set[Any] = set.intersection(*sets.values())
    if common != union:
        msg = (
            f"{func}: the panel is unbalanced — {len(union - common)} of "
            f"{len(union)} unit(s) are missing in some period(s)"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)

    spatial = gdf is not None
    w_spec: str | None = None
    rows: list[dict[str, Any]] = []

    if spatial:
        assert gdf is not None
        if w is None:
            w = _default_weights(gdf, func=func)
            notes.append(f"{func}: no weights supplied — defaulted to the built W")
        meta = dict(getattr(w, "geometrics_meta", {}) or {})
        w_spec = str(meta.get("spec") or _describe_w(w))

        if common != union:
            notes.append(
                f"{func}: spatial decomposition restricted to the {len(common)} "
                "unit(s) observed in every period"
            )
        keep = common
        gdf_entity = resolve_gdf_entity(gdf)
        gdf_ids = set(gdf[gdf_entity].dropna())
        gdf_use = gdf
        if keep & gdf_ids:
            unmatched = keep - gdf_ids
            if unmatched:
                msg = (
                    f"{func}: {len(unmatched)} unit(s) not in the geometry were "
                    f"dropped (e.g. {_first_ids(sorted(unmatched, key=str))})"
                )
                warnings.warn(msg, GeometricsWarning, stacklevel=2)
                notes.append(msg)
                keep &= gdf_ids
            if gdf_ids - keep:
                gdf_use = gdf.loc[gdf[gdf_entity].isin(keep)]
        work = work.loc[work[entity].isin(keep)]

        if permutations:
            np.random.seed(_SPATIAL_GINI_SEED)  # Gini_Spatial has no seed parameter
        align_notes: list[str] = []
        for p in periods:
            aligned, w_p, info = _align_cross_section(
                work,
                gdf_use,
                [var],
                entity=entity,
                time=time,
                period=p,
                w=w,
                min_obs=2,
                func=func,
            )
            align_notes.extend(info["notes"])
            y = aligned[var].to_numpy(dtype=float)
            row: dict[str, Any] = {"time": p, "n_units": len(aligned)}
            for m in measures:
                row[m] = _measure_value(m, y)
            from inequality.gini import Gini_Spatial

            w_pos = _positional_binary_w(list(aligned[gdf_entity]), w_p)
            gs = Gini_Spatial(y, w_pos, permutations=int(permutations))
            row["gini_spatial"] = float(gs.wg / gs.den)
            row["gini_spatial_p"] = float(gs.p_sim) if permutations else float("nan")
            rows.append(row)
        notes.extend(dict.fromkeys(align_notes))
        if not permutations:
            notes.append(
                f"{func}: permutations=0 — gini_spatial_p is NaN (no inference)"
            )
    else:
        for p in periods:
            y = work.loc[work[time] == p, var].to_numpy(dtype=float)
            row = {"time": p, "n_units": int(y.size)}
            for m in measures:
                row[m] = _measure_value(m, y)
            rows.append(row)

    tab = pd.DataFrame(rows)
    for m in measures:
        if tab[m].isna().any():
            notes.append(
                f"{func}: {m} is undefined (NaN) in some period(s) — check the "
                f"scale of {var!r}"
            )

    t_num, ranked = _numeric_time(tab["time"].tolist())
    if ranked:
        notes.append(
            f"{func}: {time!r} labels are non-numeric — trends fitted on the "
            "period rank"
        )
    trends = {m: _log_trend(t_num, tab[m].to_numpy(dtype=float)) for m in measures}
    if "gini_spatial" in tab.columns:
        trends["gini_spatial"] = _log_trend(
            t_num, tab["gini_spatial"].to_numpy(dtype=float)
        )
    for m in measures:
        if trends[m][0] is None:
            notes.append(
                f"{func}: fewer than 3 periods with a positive {m} — its trend was "
                "not estimated"
            )

    n_units = int(work[entity].nunique())
    fig = _inequality_fig(
        tab,
        measures,
        trends,
        t_num,
        time_label=time_label,
        var_label=var_label,
        w_spec=w_spec,
        title=title,
    )
    summary, gt = _inequality_summary_and_gt(
        measures,
        trends,
        var_label=var_label,
        n_periods=len(periods),
        n_units=n_units,
        w_spec=w_spec,
    )
    models = [trends[m][0] for m in measures if trends[m][0] is not None]

    return InequalityOverTimeResult(
        df=tab,
        fig=fig,
        gt=gt,
        summary=summary,
        models=models,
        var=var,
        n_periods=len(periods),
        n_units=n_units,
        w_spec=w_spec,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# analyze_theil_decomposition
# ---------------------------------------------------------------------------


def _theil_fig(
    tab: pd.DataFrame,
    *,
    time_label: str,
    var_label: str,
    group_label: str,
    title: str | None,
) -> go.Figure:
    """Build the stacked between/within area with the between-share line on top."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = tab["time"].tolist()
    for i, part in enumerate(("between", "within")):
        color = color_for(i)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=tab[part].to_numpy(dtype=float),
                mode="lines",
                name=part.capitalize(),
                stackgroup="theil",
                line={"color": color, "width": 1},
                hovertemplate=(
                    f"{time_label} = %{{x}}<br>{part} = %{{y:.4g}}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
    share_color = color_for(3)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=tab["between_share"].to_numpy(dtype=float),
            mode="lines+markers",
            name="Between share",
            line={"color": share_color, "width": 2, "dash": "dot"},
            marker={"color": share_color, "size": 7},
            hovertemplate=(
                f"{time_label} = %{{x}}<br>between share = %{{y:.1%}}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text=f"Theil index of {var_label}", secondary_y=False)
    fig.update_yaxes(
        title_text="Between-group share",
        tickformat=".0%",
        range=[0, 1],
        secondary_y=True,
    )
    apply_default_layout(
        fig,
        title=title
        if title is not None
        else f"Theil decomposition of {var_label} by {group_label}",
        xaxis={"title": time_label},
        # Horizontal legend under the plot so it clears the right (secondary) axis title.
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
        },
        margin_b=104,
    )
    return fig


def _theil_gt(
    tab: pd.DataFrame, *, var_label: str, group_label: str, n_groups: int
) -> GT:
    """Render the per-period decomposition table with Great Tables."""
    from great_tables import GT

    disp = pd.DataFrame(
        {
            "Period": [str(t) for t in tab["time"]],
            "Theil": [_fmt(v) for v in tab["theil"]],
            "Between": [_fmt(v) for v in tab["between"]],
            "Within": [_fmt(v) for v in tab["within"]],
            "Between share": [
                "—" if not math.isfinite(v) else f"{v:.1%}"
                for v in tab["between_share"]
            ],
        }
    )
    if "p_between" in tab.columns:
        disp["p (between)"] = [_fmt(v, digits=3) for v in tab["p_between"]]
    gt = (
        GT(disp, rowname_col="Period")
        .tab_header(
            title=f"Theil decomposition: {var_label}",
            subtitle=f"between / within {n_groups} {group_label} group(s)",
        )
        .tab_source_note(
            "Between + within sum to the total Theil index. The between share is "
            "the fraction of overall inequality owed to differences across group "
            "means."
        )
    )
    return gt


def analyze_theil_decomposition(
    df: pd.DataFrame,
    var: str,
    group: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    permutations: int = 0,
    seed: int = 12345,
    title: str | None = None,
) -> TheilDecompositionResult:
    """Decompose the Theil index between and within a group partition, per period.

    For every period the Theil index of ``var`` across units is split additively
    (:class:`inequality.theil.TheilD`) into a **between-group** component (inequality
    across the mean levels of the ``group`` partition, e.g. states) and a
    **within-group** component (inequality among units inside each group):
    ``theil = between + within`` exactly. The ``between_share`` tracks how much of
    total inequality is a group-level phenomenon. With ``permutations > 0`` the
    between component gets a permutation pseudo p-value
    (:class:`inequality.theil.TheilDSim`): units are randomly reassigned to groups
    and ``p_between`` reports how often a random partition yields a between share at
    least as large.

    Parameters
    ----------
    df
        Long-form panel data frame.
    var
        Numeric variable to decompose (strictly positive — the Theil index takes
        logarithms of shares).
    group
        Partition column (e.g. a state id for district units). It must be constant
        within each entity across periods, and define at least two groups.
    entity, time
        Panel identifiers. Default to those declared via
        :func:`geometrics.set_panel`.
    permutations
        Number of permutations for the between-component inference (``0`` disables
        it and omits the ``p_between`` column).
    seed
        Seed for the permutation draws. ``TheilDSim`` has no seed parameter and
        draws from NumPy's global RNG, so ``np.random.seed(seed)`` is called once
        before the per-period loop.
    title
        Title for the figure.

    Returns
    -------
    TheilDecompositionResult
        Per-period frame ``df`` (``time``, ``theil``, ``between``, ``within``,
        ``between_share``, plus ``p_between`` when ``permutations > 0``); the
        stacked between/within area ``fig`` with the between-share line on the
        secondary axis; the per-period ``gt`` table; and ``group`` / ``n_groups`` /
        ``permutations``.

    Raises
    ------
    KeyError
        If ``var`` or ``group`` is not a column of ``df``.
    TypeError
        If ``var`` is not numeric.
    ValueError
        If ``group`` varies within an entity (the offenders are named), defines
        fewer than two groups, or ``var`` has non-positive values (the offending
        entities are named).

    Examples
    --------
    Two states with two districts each, over two years:

    ```python
    import pandas as pd

    from geometrics.regional_inequality import analyze_theil_decomposition

    df = pd.DataFrame(
        {
            "district": ["d1", "d2", "d3", "d4"] * 2,
            "state": ["north", "north", "south", "south"] * 2,
            "year": [2000] * 4 + [2001] * 4,
            "income": [10.0, 12.0, 30.0, 36.0, 11.0, 13.0, 33.0, 40.0],
        }
    )
    res = analyze_theil_decomposition(
        df, "income", "state", entity="district", time="year"
    )
    res.df[["time", "between_share"]].round(3)
    ```
    """
    func = "analyze_theil_decomposition"
    df = ensure_dataframe(df)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # guaranteed by require_* above
    missing = [c for c in (var, group) if c not in df.columns]
    if missing:
        raise KeyError(f"{func}: column(s) not found in df: {missing}")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"{func}: {var!r} needs to be numeric")

    var_label = resolve_label(df, var)
    group_label = resolve_label(df, group)
    time_label = resolve_label(df, time)
    notes: list[str] = []

    work = df[list(dict.fromkeys([entity, time, var, group]))].copy()
    work = work.dropna(subset=[entity, time])
    if work.empty:
        raise ValueError(f"{func}: no rows with non-missing {entity!r} and {time!r}")
    work = _dedupe_first(work, [entity, time], notes, func=func)
    work = _drop_var_missing(work, list(dict.fromkeys([var, group])), notes, func=func)
    if work.empty:
        raise ValueError(f"{func}: no complete rows for {var!r} and {group!r}")

    if group != entity:  # the entity itself is trivially constant within itself
        varying = work.groupby(entity, observed=True)[group].nunique()
        offenders = varying[varying > 1]
        if len(offenders):
            raise ValueError(
                f"{func}: {group!r} needs to be constant within each entity, but "
                f"{len(offenders)} entit(ies) change group over time "
                f"(e.g. {_first_ids(offenders.index)})"
            )
    n_groups = int(work[group].nunique())
    if n_groups < 2:
        raise ValueError(
            f"{func}: {group!r} defines only {n_groups} group — the between/within "
            "decomposition needs at least 2"
        )
    _require_positive(work, var, entity, func=func, why="the Theil index")

    from inequality.theil import TheilD, TheilDSim

    if permutations:
        np.random.seed(seed)  # TheilDSim has no seed parameter
    periods = sorted(pd.unique(work[time]).tolist())
    rows: list[dict[str, Any]] = []
    for p in periods:
        sub = work.loc[work[time] == p]
        y = sub[var].to_numpy(dtype=float)
        partition = np.asarray(sub[group].astype(str))
        td = TheilD(y, partition)
        total = float(td.T)
        between = float(np.asarray(td.bg).reshape(-1)[0])
        within = float(np.asarray(td.wg).reshape(-1)[0])
        row: dict[str, Any] = {
            "time": p,
            "theil": total,
            "between": between,
            "within": within,
            "between_share": between / total if total > 0.0 else float("nan"),
        }
        if permutations:
            tds = TheilDSim(y, partition, permutations=int(permutations))
            row["p_between"] = float(np.asarray(tds.bg_pvalue).reshape(-1)[0])
        rows.append(row)
    tab = pd.DataFrame(rows)
    if tab["between_share"].isna().any():
        notes.append(
            f"{func}: total inequality is zero in some period(s) — the between "
            "share is undefined (NaN) there"
        )

    fig = _theil_fig(
        tab,
        time_label=time_label,
        var_label=var_label,
        group_label=group_label,
        title=title,
    )
    gt = _theil_gt(tab, var_label=var_label, group_label=group_label, n_groups=n_groups)

    return TheilDecompositionResult(
        df=tab,
        fig=fig,
        gt=gt,
        group=group,
        n_groups=n_groups,
        permutations=int(permutations),
        var=var,
        notes=tuple(notes),
    )
