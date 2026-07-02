"""Distribution dynamics: discrete Markov transition analysis on the giddy stack.

The distribution-dynamics tradition (Quah 1993; Rey 2001) studies *movement within the
cross-sectional distribution* rather than its average: each region's value is discretized
into ``k`` states (income classes) and the panel's period-to-period moves are summarized
by a Markov transition-probability matrix, its ergodic (steady-state) distribution,
expected sojourn times and scalar mobility indices.

:func:`analyze_markov_transitions` runs that classic (a-spatial) workflow from a long
panel. :func:`analyze_spatial_markov` runs Rey's **spatial Markov** extension: the same
transition matrix, but *conditioned on the spatial lag* of each region's neighbors, plus
the Bickenbach-Bode LR / Q homogeneity tests of whether transition dynamics differ across
neighborhood contexts.

Both functions need the optional ``giddy`` dependency::

    pip install "geometrics[dynamics]"
"""

from __future__ import annotations

import contextlib
import io
import math
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from great_tables import GT
from pandas.api import types as pdt
from plotly.subplots import make_subplots

from geometrics._geo import _align_panel_wide
from geometrics._labels import resolve_label
from geometrics._panel import resolve_panel
from geometrics._theme import active_sequential_scale, apply_default_layout
from geometrics._types import MarkovTransitionsResult, SpatialMarkovResult
from geometrics._validation import (
    GeometricsWarning,
    ensure_dataframe,
    ensure_geodataframe,
)

if TYPE_CHECKING:
    import geopandas as gpd
    from libpysal.weights import W

__all__ = ["analyze_markov_transitions", "analyze_spatial_markov"]

#: Canonical mapclassify scheme names accepted by ``analyze_markov_transitions``
#: (spelling is normalized, so ``"FisherJenks"`` / ``"fisher-jenks"`` also work).
_SCHEMES = ("quantiles", "equal_interval", "fisher_jenks")


def _import_giddy(func: str) -> Any:
    """Import and return :mod:`giddy`, raising a helpful ``ImportError`` if absent."""
    try:
        import giddy
        import giddy.markov
        import giddy.mobility
    except ImportError as exc:
        raise ImportError(
            f'{func} requires the dynamics extra: pip install "geometrics[dynamics]"'
        ) from exc
    return giddy


def _normalize_scheme(scheme: str, *, func: str) -> str:
    """Return the canonical scheme name for ``scheme`` or raise ``ValueError``."""
    key = str(scheme).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "quantiles": "quantiles",
        "quantile": "quantiles",
        "equal_interval": "equal_interval",
        "equalinterval": "equal_interval",
        "fisher_jenks": "fisher_jenks",
        "fisherjenks": "fisher_jenks",
    }
    if key not in aliases:
        raise ValueError(
            f"{func}: unknown scheme {scheme!r}; choose from {list(_SCHEMES)}"
        )
    return aliases[key]


def _panel_wide(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str,
    time: str,
    func: str,
) -> tuple[np.ndarray, list, list, list[str]]:
    """Reshape a long panel to a balanced wide ``(n, t)`` array (no geometry needed).

    Rows follow the first-appearance order of the entities; columns are the sorted
    periods. Returns ``(values, ids, periods, notes)`` and raises ``ValueError`` when
    the panel is unbalanced.
    """
    notes: list[str] = []
    sub = df
    dup = sub.duplicated(subset=[entity, time])
    if bool(dup.any()):
        sub = sub.drop_duplicates(subset=[entity, time], keep="first")
        msg = (
            f"{func}: kept the first of {int(dup.sum())} duplicate "
            f"({entity!r}, {time!r}) row(s)"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=3)
        notes.append(msg)

    ids = list(dict.fromkeys(sub[entity].dropna()))
    wide = sub.pivot(index=entity, columns=time, values=var)
    periods = sorted(wide.columns.tolist())
    wide = wide.reindex(index=ids, columns=periods)
    incomplete = [str(i) for i in wide.index[wide.isna().any(axis=1)]]
    if incomplete:
        raise ValueError(
            f"{func}: needs a balanced panel of {var!r} over periods {periods} — "
            f"{len(incomplete)} entit(ies) are incomplete (e.g. {incomplete[:5]})"
        )
    return wide.to_numpy(dtype=float), ids, periods, notes


def _make_relative(values: np.ndarray, *, func: str) -> np.ndarray:
    """Divide each period (column) of ``values`` by its cross-sectional mean."""
    means = values.mean(axis=0)
    if not np.all(np.isfinite(means)) or bool(np.any(means == 0.0)):
        raise ValueError(
            f"{func}: relative=True needs a non-zero, finite cross-sectional mean "
            "in every period"
        )
    return values / means


def _discretize(
    values: np.ndarray,
    *,
    k: int,
    scheme: str,
    bins: Sequence[float] | None,
    per_period: bool,
    func: str,
) -> tuple[np.ndarray, int, str, list[str]]:
    """Discretize the ``(n, t)`` array into class ids.

    Returns ``(class_ids, k_eff, scheme_used, notes)``. With ``bins`` the classes are
    fixed :class:`mapclassify.UserDefined` intervals (identical in every period);
    otherwise each period's column is classified on its own when ``per_period`` is
    ``True``, or the pooled ``n*t`` values are classified once when it is ``False``.
    """
    import mapclassify

    notes: list[str] = []
    pooled = values.ravel()
    if bins is not None:
        classifier = mapclassify.UserDefined(pooled, list(bins))
        k_eff = int(classifier.k)
        class_ids = np.asarray(classifier.yb).reshape(values.shape)
        return class_ids, k_eff, "user_defined", notes

    scheme = _normalize_scheme(scheme, func=func)
    builders: dict[str, Any] = {
        "quantiles": mapclassify.Quantiles,
        "equal_interval": mapclassify.EqualInterval,
        "fisher_jenks": mapclassify.FisherJenks,
    }
    builder = builders[scheme]
    if per_period:
        columns = [
            np.asarray(builder(values[:, j], k=k).yb) for j in range(values.shape[1])
        ]
        class_ids = np.column_stack(columns)
        k_eff = k
    else:
        classifier = builder(pooled, k=k)
        k_eff = int(classifier.k)
        class_ids = np.asarray(classifier.yb).reshape(values.shape)
        if k_eff != k:
            msg = (
                f"{func}: the pooled {scheme} classification produced {k_eff} "
                f"classes instead of the requested {k} (ties in the data)"
            )
            warnings.warn(msg, GeometricsWarning, stacklevel=3)
            notes.append(msg)
    return class_ids, k_eff, scheme, notes


def _state_labels(k: int, scheme: str) -> list[str]:
    """Return the state labels: ``Q1..Qk`` for quantiles, ``C1..Ck`` otherwise."""
    prefix = "Q" if scheme == "quantiles" else "C"
    return [f"{prefix}{i + 1}" for i in range(k)]


def _guarded(getter: Callable[[], float]) -> float:
    """Return ``float(getter())`` or ``nan`` when the computation degenerates."""
    try:
        return float(getter())
    except Exception:  # giddy raises assorted numeric errors on degenerate chains
        return float("nan")


def _steady_state_vector(mk: Any, k: int) -> tuple[np.ndarray, bool]:
    """Return the ergodic distribution of ``mk`` as a ``(k,)`` vector (nan if undefined).

    The second element flags degeneracy: giddy returns a multi-row array for a
    reducible chain, in which case a single ergodic distribution does not exist.
    """
    try:
        arr = np.asarray(mk.steady_state, dtype=float)
    except Exception:  # reducible / singular chains
        return np.full(k, np.nan), True
    arr = arr.squeeze()
    if arr.shape != (k,):
        return np.full(k, np.nan), True
    return arr, False


def _transition_heatmap(
    p: np.ndarray, states: Sequence[str], *, coloraxis: str | None = None
) -> go.Heatmap:
    """Build the annotated transition-probability heatmap trace.

    With ``coloraxis`` the trace binds to a figure-level shared color axis (small
    multiples); without it the trace carries its own sequential scale and colorbar.
    """
    trace = go.Heatmap(
        z=p,
        x=list(states),
        y=list(states),
        xgap=1,
        ygap=1,
        texttemplate="%{z:.2f}",
        hovertemplate="from %{y} → %{x}: %{z:.3f}<extra></extra>",
    )
    if coloraxis is not None:
        trace.coloraxis = coloraxis
    else:
        trace.colorscale = active_sequential_scale()
        trace.zmin = 0.0
        trace.zmax = 1.0
        trace.colorbar = {"title": "P(next | current)"}
    return trace


def analyze_markov_transitions(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    k: int = 5,
    scheme: str = "quantiles",
    bins: Sequence[float] | None = None,
    per_period: bool = True,
    relative: bool = False,
    title: str | None = None,
) -> MarkovTransitionsResult:
    r"""Estimate a discrete Markov chain of movement between distribution states.

    Each region's ``var`` is discretized into ``k`` states (per period by default, so a
    state is a *rank* within that period's cross-section) and every period-to-period
    move is pooled into a ``k``-by-``k`` transition-probability matrix
    (:class:`giddy.markov.Markov`). The result carries the ergodic (steady-state)
    distribution, expected sojourn times, and the Shorrocks / Prais / Bartholomew
    mobility indices of the matrix.

    Parameters
    ----------
    df
        Long-form panel with entity, time and ``var`` columns. The panel must be
        balanced in ``var`` (every entity observed in every period).
    var
        Numeric variable whose distribution dynamics are analyzed.
    entity
        Entity (unit) id column; defaults to the panel declared via
        :func:`geometrics.set_panel`.
    time
        Time id column; defaults to the declared panel.
    k
        Number of states (classes) to discretize into (default 5).
    scheme
        Classification scheme: ``"quantiles"`` (default), ``"equal_interval"`` or
        ``"fisher_jenks"``. Ignored when ``bins`` is given.
    bins
        Explicit upper class bounds (:class:`mapclassify.UserDefined`); the same fixed
        intervals apply in every period and ``scheme`` / ``per_period`` are ignored.
    per_period
        Classify each period's cross-section separately (default ``True``, the
        distribution-dynamics convention: states are positions *within* the period's
        distribution). ``False`` pools all ``n*t`` values into one classification.
    relative
        Divide ``var`` by its cross-sectional mean per period first (so 1.0 marks the
        period average). Default ``False``.
    title
        Figure title (a default naming the variable is used when ``None``).

    Returns
    -------
    MarkovTransitionsResult
        The long panel with each (entity, period) ``state``, the labelled transition
        matrix ``p`` and ``counts``, the annotated heatmap ``fig``, the summary table
        ``gt``, the ``steady_state`` and ``sojourn`` series, and the ``shorrocks`` /
        ``prais`` / ``bartholomew`` mobility indices.

    Raises
    ------
    ImportError
        If the optional ``giddy`` dependency is not installed.
    KeyError
        If ``var`` is not a column of ``df``.
    TypeError
        If ``var`` is not numeric.
    ValueError
        If ``k < 2``, the scheme is unknown, the panel is unbalanced, or fewer than
        two periods are observed.

    Notes
    -----
    Mobility indices use :func:`giddy.mobility.markov_mobility` measure codes:
    ``shorrocks`` is measure ``"P"`` (the trace index :math:`(k - \mathrm{tr}\,P)/(k-1)`),
    ``prais`` is measure ``"D"`` (the determinant index :math:`1 - |\det P|`), and
    ``bartholomew`` is measure ``"B1"`` (the trace index weighted by the first period's
    observed state distribution).

    Examples
    --------
    Three groups of regions that keep their income rank from year to year:

    ```python
    import numpy as np
    import pandas as pd

    from geometrics.distribution_dynamics import analyze_markov_transitions

    rng = np.random.default_rng(0)
    units = [f"r{i}" for i in range(9)]
    base = np.repeat([1.0, 2.0, 3.0], 3)
    df = pd.DataFrame(
        [
            {"region": u, "year": y, "income": b + rng.normal(0, 0.5)}
            for y in (2000, 2001, 2002, 2003)
            for u, b in zip(units, base)
        ]
    )
    res = analyze_markov_transitions(df, "income", entity="region", time="year", k=3)
    res.p.round(2)
    ```
    """
    func = "analyze_markov_transitions"
    df = ensure_dataframe(df)
    giddy = _import_giddy(func)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None
    if var not in df.columns:
        raise KeyError(f"{func}: column {var!r} not found in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"{func}: {var!r} needs to be numeric")
    if int(k) < 2:
        raise ValueError(f"{func}: k={k} needs to be at least 2")
    k = int(k)

    values, ids, periods, notes = _panel_wide(
        df, var, entity=entity, time=time, func=func
    )
    if len(periods) < 2:
        raise ValueError(
            f"{func}: needs at least two periods to observe transitions "
            f"(found {len(periods)})"
        )
    if float(np.ptp(values)) == 0.0:
        raise ValueError(f"{func}: {var!r} has zero variance — nothing to classify")

    if relative:
        values = _make_relative(values, func=func)
        notes.append(
            f"{func}: values divided by the cross-sectional mean of each period "
            "(relative=True)"
        )

    class_ids, k_eff, scheme_used, class_notes = _discretize(
        values, k=k, scheme=scheme, bins=bins, per_period=per_period, func=func
    )
    notes.extend(class_notes)
    states = _state_labels(k_eff, scheme_used)

    # --- estimate (giddy prints chain summaries; keep stdout clean) ---------------
    with contextlib.redirect_stdout(io.StringIO()):
        mk = giddy.markov.Markov(
            class_ids.astype(int),
            classes=np.arange(k_eff),
            fill_empty_classes=True,
            summary=False,
        )
        p_arr = np.asarray(mk.p, dtype=float)
        counts_arr = np.asarray(mk.transitions, dtype=float)
        steady_arr, degenerate = _steady_state_vector(mk, k_eff)
        try:
            sojourn_arr = np.asarray(mk.sojourn_time, dtype=float).reshape(k_eff)
        except Exception:  # degenerate chains
            sojourn_arr = np.full(k_eff, np.nan)
        first_period = np.bincount(class_ids[:, 0].astype(int), minlength=k_eff)
        ini = first_period / first_period.sum()
        shorrocks = _guarded(lambda: giddy.mobility.markov_mobility(p_arr, measure="P"))
        prais = _guarded(lambda: giddy.mobility.markov_mobility(p_arr, measure="D"))
        bartholomew = _guarded(
            lambda: giddy.mobility.markov_mobility(p_arr, measure="B1", ini=ini)
        )

    empty_rows = [states[i] for i in range(k_eff) if counts_arr[i].sum() == 0]
    if empty_rows:
        msg = (
            f"{func}: state(s) {empty_rows} were never observed as an origin — their "
            "transition row was filled as absorbing (probability 1 of staying)"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)
    if degenerate:
        msg = (
            f"{func}: the chain is reducible (multiple recurrent classes), so a "
            "single ergodic steady-state distribution does not exist — reported as NaN"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)

    p = pd.DataFrame(p_arr, index=list(states), columns=list(states))
    counts = pd.DataFrame(
        counts_arr.astype(int), index=list(states), columns=list(states)
    )
    steady_state = pd.Series(steady_arr, index=list(states), name="steady_state")
    sojourn = pd.Series(sojourn_arr, index=list(states), name="sojourn")
    n_transitions = int(counts_arr.sum())

    long = pd.DataFrame(
        {
            entity: np.repeat(np.asarray(ids, dtype=object), len(periods)),
            time: np.tile(np.asarray(periods, dtype=object), len(ids)),
            var: values.ravel(),
            "state": [states[int(c)] for c in class_ids.ravel()],
        }
    )

    # --- themed figure -------------------------------------------------------------
    var_label = resolve_label(df, var)
    scheme_desc = {"user_defined": "user-defined bins"}.get(
        scheme_used, f"{scheme_used.replace('_', ' ')} classes"
    )
    subtitle = (
        f"{k_eff} states ({scheme_desc}, "
        f"{'per-period' if per_period and bins is None else 'pooled'} classification), "
        f"{n_transitions:,} transitions"
        + (", relative to the period mean" if relative else "")
    )
    fig = go.Figure(_transition_heatmap(p_arr, states))
    fig.update_yaxes(autorange="reversed")
    apply_default_layout(
        fig,
        title=title if title is not None else f"Markov transitions — {var_label}",
        subtitle=subtitle,
        xaxis={"title": "State at t+1"},
        yaxis={"title": "State at t"},
    )

    # --- summary table ---------------------------------------------------------------
    tidy_rows = [
        {
            "section": "Mobility indices",
            "measure": "Shorrocks trace index (giddy measure 'P')",
            "value": shorrocks,
        },
        {
            "section": "Mobility indices",
            "measure": "Prais determinant index (giddy measure 'D')",
            "value": prais,
        },
        {
            "section": "Mobility indices",
            "measure": "Bartholomew index (giddy measure 'B1')",
            "value": bartholomew,
        },
        *(
            {
                "section": "Ergodic (steady-state) distribution",
                "measure": f"Long-run share in {state}",
                "value": float(steady_state[state]),
            }
            for state in states
        ),
        *(
            {
                "section": "Expected sojourn time (periods)",
                "measure": f"Consecutive periods in {state}",
                "value": float(sojourn[state]),
            }
            for state in states
        ),
    ]
    summary = pd.DataFrame(tidy_rows)
    # Great Tables cannot format infinities (absorbing-state sojourn times): show NA.
    summary["value"] = summary["value"].replace([np.inf, -np.inf], np.nan)
    gt = (
        GT(summary, groupname_col="section")
        .tab_header(title=f"Markov transition summary — {var_label}", subtitle=subtitle)
        .cols_label(measure="Measure", value="Value")
        .fmt_number(columns="value", decimals=3)
    )

    return MarkovTransitionsResult(
        df=long,
        p=p,
        counts=counts,
        fig=fig,
        gt=gt,
        states=tuple(states),
        steady_state=steady_state,
        sojourn=sojourn,
        shorrocks=shorrocks,
        prais=prais,
        bartholomew=bartholomew,
        n_transitions=n_transitions,
        k=k_eff,
        scheme=scheme_used,
        var=var,
        notes=tuple(notes),
    )


def analyze_spatial_markov(
    df: pd.DataFrame,
    var: str,
    *,
    gdf: gpd.GeoDataFrame,
    w: W | None = None,
    entity: str | None = None,
    time: str | None = None,
    k: int = 5,
    m: int | None = None,
    fixed: bool = True,
    relative: bool = True,
    title: str | None = None,
) -> SpatialMarkovResult:
    """Estimate a spatial Markov chain: transitions conditioned on the neighbors' state.

    Rey's (2001) spatial Markov splits the classic transition matrix by the *spatial
    lag* of each region — the (weighted) average of its neighbors — discretized into
    ``m`` classes. One ``k``-by-``k`` matrix per neighbor class shows whether upward
    or downward moves happen at different rates in rich versus poor neighborhoods, and
    the Bickenbach-Bode LR / Q tests ask whether those conditional dynamics differ
    from the pooled (unconditional) matrix.

    Parameters
    ----------
    df
        Long-form panel with entity, time and ``var`` columns. The panel must be
        balanced in ``var`` (every entity observed in every period).
    var
        Numeric variable whose distribution dynamics are analyzed.
    gdf
        Geometry frame carrying the entity ids (see :func:`geometrics.read_gdf`); the
        panel is aligned to the weights' row order through it.
    w
        ``libpysal`` weights aligned to the gdf entity ids. ``None`` builds the
        default weights (queen contiguity for polygons, 6-nearest-neighbor otherwise)
        with a :class:`~geometrics.GeometricsWarning`.
    entity
        Entity (unit) id column of ``df``; defaults to the declared panel.
    time
        Time id column; defaults to the declared panel.
    k
        Number of states for the variable itself (default 5).
    m
        Number of classes for the spatial lag (default: same as ``k``).
    fixed
        Pool the ``n*t`` values into one quantile classification (default ``True``,
        giddy's convention); ``False`` re-classifies each period separately.
    relative
        Divide ``var`` by its cross-sectional mean per period first (default ``True``,
        the distribution-dynamics convention for income data).
    title
        Figure title (a default naming the variable is used when ``None``).

    Returns
    -------
    SpatialMarkovResult
        The long panel with each (entity, period) ``state`` and ``neighbor_state``,
        the unconditional ``p_global``, the tuple of ``m`` conditional matrices
        ``p_conditional`` (with ``steady_states`` stacking their ergodic
        distributions), the small-multiple heatmap ``fig``, the homogeneity-test table
        ``gt``, and the LR / Q statistics with their p-values and ``dof``.

    Raises
    ------
    ImportError
        If the optional ``giddy`` dependency is not installed.
    KeyError
        If ``var`` is not a column of ``df``.
    TypeError
        If ``var`` is not numeric.
    ValueError
        If ``k < 2`` or ``m < 2``, the panel is unbalanced, fewer than two periods are
        observed, or the weights ids do not match the geometry.

    Examples
    --------
    A 3x3 lattice where income levels follow a smooth spatial gradient:

    ```python
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import box

    from geometrics.distribution_dynamics import analyze_spatial_markov
    from geometrics.weights import make_weights

    cells = [box(x, y, x + 1, y + 1) for y in range(3) for x in range(3)]
    gdf = gpd.GeoDataFrame(
        {"region": [f"r{i}" for i in range(9)]}, geometry=cells, crs="EPSG:4326"
    )
    w = make_weights(gdf, method="queen")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        [
            {"region": f"r{i}", "year": y, "income": 1.0 + i / 4 + rng.normal(0, 0.3)}
            for y in (2000, 2001, 2002, 2003)
            for i in range(9)
        ]
    )
    res = analyze_spatial_markov(
        df, "income", gdf=gdf, w=w, entity="region", time="year", k=2, m=2
    )
    res.p_global.round(2)
    ```
    """
    func = "analyze_spatial_markov"
    df = ensure_dataframe(df)
    gdf = ensure_geodataframe(gdf, func=func)
    giddy = _import_giddy(func)
    if int(k) < 2:
        raise ValueError(f"{func}: k={k} needs to be at least 2")
    k = int(k)
    m_classes = k if m is None else int(m)
    if m_classes < 2:
        raise ValueError(f"{func}: m={m} needs to be at least 2")

    notes: list[str] = []
    if w is None:
        from geometrics.weights import _default_weights

        w = _default_weights(gdf, func=func)
        notes.append(
            f"{func}: no weights supplied — defaulted to {w.geometrics_meta['spec']}"
        )
    meta = dict(getattr(w, "geometrics_meta", {}) or {})
    if meta.get("spec"):
        w_spec = str(meta["spec"])
    else:
        from geometrics.weights import _describe_w

        w_spec = _describe_w(w)

    values, ids, periods, info = _align_panel_wide(
        df, gdf, var, w=w, entity=entity, time=time, func=func
    )
    notes.extend(str(n) for n in info.get("notes") or ())
    entity_col, time_col = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity_col is not None and time_col is not None
    if len(periods) < 2:
        raise ValueError(
            f"{func}: needs at least two periods to observe transitions "
            f"(found {len(periods)})"
        )
    if float(np.ptp(values)) == 0.0:
        raise ValueError(f"{func}: {var!r} has zero variance — nothing to classify")

    if relative:
        values = _make_relative(values, func=func)
        notes.append(
            f"{func}: values divided by the cross-sectional mean of each period "
            "(relative=True)"
        )

    var_label = resolve_label(df, var)

    # --- estimate (giddy prints chain summaries; keep stdout clean) ---------------
    with contextlib.redirect_stdout(io.StringIO()):
        sm = giddy.markov.Spatial_Markov(
            values,
            w,
            k=k,
            m=m_classes,
            fixed=fixed,
            fill_empty_classes=True,
            variable_name=str(var_label),
        )
        p_global_arr = np.asarray(sm.p, dtype=float)
        p_cond_arr = np.asarray(sm.P, dtype=float)
        class_ids = np.asarray(sm.class_ids, dtype=int)
        lclass_ids = np.asarray(sm.lclass_ids, dtype=int)
        try:
            raw_steady: Any = sm.S
        except Exception:
            raw_steady = None
        lr_stat = _guarded(lambda: sm.LR)
        lr_p = _guarded(lambda: sm.LR_p_value)
        q_stat = _guarded(lambda: sm.Q)
        q_p = _guarded(lambda: sm.Q_p_value)
        dof_f = _guarded(lambda: sm.dof_hom)

    k_eff = int(p_global_arr.shape[0])
    m_eff = int(p_cond_arr.shape[0])
    states = _state_labels(k_eff, "quantiles")
    lag_states = _state_labels(m_eff, "quantiles")
    lag_labels = [f"Neighbors in {s}" for s in lag_states]

    dof = int(dof_f) if math.isfinite(dof_f) else 0
    if not (math.isfinite(lr_stat) and math.isfinite(q_stat)) or dof == 0:
        msg = (
            f"{func}: the LR / Q homogeneity tests could not be (fully) computed — "
            "some conditional transition matrices are too sparse (degenerate cells)"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)

    # Steady states per neighbor class; giddy returns a list / object array when a
    # conditional chain is reducible — those rows are undefined and reported as NaN.
    steady_rows: list[np.ndarray] = []
    degenerate_lags: list[str] = []
    for i in range(m_eff):
        row = np.full(k_eff, np.nan)
        if raw_steady is not None:
            try:
                arr = np.asarray(raw_steady[i], dtype=float).squeeze()
            except Exception:
                arr = np.full(k_eff, np.nan)
            if arr.shape == (k_eff,):
                row = arr
        if not np.all(np.isfinite(row)):
            degenerate_lags.append(lag_labels[i])
            row = np.full(k_eff, np.nan)
        steady_rows.append(row)
    if degenerate_lags:
        msg = (
            f"{func}: no unique ergodic distribution for {degenerate_lags} (the "
            "conditional chain is reducible) — reported as NaN"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)
    steady_states = pd.DataFrame(
        np.vstack(steady_rows), index=lag_labels, columns=list(states)
    )

    p_global = pd.DataFrame(p_global_arr, index=list(states), columns=list(states))
    p_conditional = tuple(
        pd.DataFrame(p_cond_arr[i], index=list(states), columns=list(states))
        for i in range(m_eff)
    )

    long = pd.DataFrame(
        {
            entity_col: np.repeat(np.asarray(ids, dtype=object), len(periods)),
            time_col: np.tile(np.asarray(periods, dtype=object), len(ids)),
            var: values.ravel(),
            "state": [states[c] for c in class_ids.ravel()],
            "neighbor_state": [lag_states[c] for c in lclass_ids.ravel()],
        }
    )

    # --- small-multiple heatmaps sharing one coloraxis ------------------------------
    n_cols = min(m_eff, 3)
    n_rows = math.ceil(m_eff / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=lag_labels,
        horizontal_spacing=0.06,
        vertical_spacing=0.16 if n_rows > 1 else 0.08,
    )
    for i in range(m_eff):
        sub_row, sub_col = i // n_cols + 1, i % n_cols + 1
        heat = _transition_heatmap(p_cond_arr[i], states, coloraxis="coloraxis")
        fig.add_trace(heat, row=sub_row, col=sub_col)
        fig.update_yaxes(autorange="reversed", row=sub_row, col=sub_col)
        if sub_col == 1:
            fig.update_yaxes(title_text="State at t", row=sub_row, col=sub_col)
        if sub_row == n_rows:
            fig.update_xaxes(title_text="State at t+1", row=sub_row, col=sub_col)
    apply_default_layout(
        fig,
        title=title
        if title is not None
        else f"Spatial Markov transitions — {var_label}",
        subtitle=w_spec,
        coloraxis={
            "colorscale": active_sequential_scale(),
            "cmin": 0.0,
            "cmax": 1.0,
            "colorbar": {"title": "P(next | current)", "thickness": 14, "len": 0.85},
        },
    )

    # --- homogeneity-test table ------------------------------------------------------
    tests = pd.DataFrame(
        [
            {
                "test": "Likelihood ratio (LR)",
                "statistic": lr_stat,
                "dof": dof,
                "p_value": lr_p,
            },
            {
                "test": "Chi-square (Q)",
                "statistic": q_stat,
                "dof": dof,
                "p_value": q_p,
            },
        ]
    )
    gt = (
        GT(tests)
        .tab_header(
            title=f"Spatial homogeneity tests — {var_label}",
            subtitle=(
                f"H0: transition dynamics identical across the {m_eff} neighbor "
                f"classes — {w_spec}"
            ),
        )
        .cols_label(test="Test", statistic="Statistic", dof="dof", p_value="p-value")
        .fmt_number(columns="statistic", decimals=3)
        .fmt_number(columns="p_value", decimals=4)
    )

    return SpatialMarkovResult(
        df=long,
        p_global=p_global,
        p_conditional=p_conditional,
        steady_states=steady_states,
        fig=fig,
        gt=gt,
        lr_stat=lr_stat,
        lr_p=lr_p,
        q_stat=q_stat,
        q_p=q_p,
        dof=dof,
        k=k_eff,
        m=m_eff,
        relative=bool(relative),
        var=var,
        w_spec=w_spec,
        notes=tuple(notes),
    )
