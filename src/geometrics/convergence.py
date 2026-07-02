"""Convergence analysis: growth cross-sections, β-convergence and σ-convergence.

:func:`growth_cross_section` builds the one-row-per-unit growth table every convergence
analysis starts from: each unit's initial and final level of a variable over a common
window and the (annualized) **log growth** between them, with controls attached at their
initial-period values.

:func:`analyze_beta_convergence` regresses that growth on the **initial log level** — the
canonical Barro-Sala-i-Martin test. A **negative** slope β is convergence (units that
start lower grow faster). Beyond plain OLS, the same regression can be estimated with
the standard spatial econometric family on a cross-section aligned to entity geometry:
``"sar"`` (spatial lag), ``"sem"`` (spatial error), ``"slx"`` (spatially lagged
regressors) and ``"sdm"`` (spatial Durbin), with the LeSage-Pace direct/indirect/total
impact decomposition of the initial-level term and Monte-Carlo impact inference — the
workflow of the source paper's Table 1.

:func:`analyze_sigma_convergence` takes the complementary **σ-convergence** view: it
tracks the *cross-sectional dispersion* of the variable's **log** over time — the
standard deviation, the Gini index and the coefficient of variation — and tests whether
that dispersion shrinks via an OLS trend of the **log dispersion** on time. A negative
trend is σ-convergence (the distribution is narrowing).

Both analyses take the variable in **levels** (e.g. GDP per capita, nighttime lights per
capita) and handle the logs internally, so the growth rate is the annualized
log-difference and the x-axis of the β-convergence scatter is the initial log level.
"""

from __future__ import annotations

import contextlib
import io
import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from pandas.api import types as pdt
from plotly.subplots import make_subplots

from geometrics._common import entity_display_map
from geometrics._geo import _align_cross_section, resolve_gdf_entity
from geometrics._impacts import (
    analytic_slx_impacts,
    full_rank_lag_mask,
    mc_impacts,
    stars,
)
from geometrics._labels import resolve_label
from geometrics._mapping import classified_map
from geometrics._panel import resolve_entity_name, resolve_panel, set_panel
from geometrics._theme import apply_default_layout, color_for
from geometrics._types import BetaConvergenceResult, SigmaConvergenceResult
from geometrics._validation import (
    GeometricsWarning,
    ensure_dataframe,
    ensure_geodataframe,
)

if TYPE_CHECKING:
    import geopandas as gpd
    from libpysal.weights import W

__all__ = [
    "growth_cross_section",
    "analyze_beta_convergence",
    "analyze_sigma_convergence",
]

#: Estimators supported by :func:`analyze_beta_convergence`.
_BETA_MODELS = ("ols", "sar", "sem", "slx", "sdm")

#: Name of the focal regressor (the initial log level) in designs and impact tables.
_FOCAL = "log_initial"

#: Column names reserved by the growth cross-section.
_RESERVED = ("initial", "final", "growth")

#: Row order of the numeric ``summary`` frame of :func:`analyze_beta_convergence`.
_BETA_METRICS = (
    "direct",
    "se_direct",
    "indirect",
    "se_indirect",
    "total",
    "se_total",
    "rho",
    "lambda",
    "r2",
    "aic",
    "n_obs",
    "speed",
    "half_life",
)


def _as_list(value: Sequence[str] | str | None) -> list[str]:
    """Normalize an optional name-or-names argument to a plain list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _speed_of_convergence(beta: float, horizon: float) -> float:
    """Return the structural speed ``lambda = -log1p(beta*T) / T`` implied by ``beta``.

    Positive lambda means convergence. Returns ``nan`` when ``1 + beta*T <= 0`` (the
    mapping is undefined) or the horizon is non-positive.
    """
    arg = 1.0 + beta * horizon
    if horizon <= 0.0 or not math.isfinite(arg) or arg <= 0.0:
        return float("nan")
    return -math.log1p(beta * horizon) / horizon


def _half_life(speed: float) -> float:
    """Return the half-life ``ln 2 / lambda`` (periods to close half a gap); ``nan`` if ``lambda <= 0``."""
    if not math.isfinite(speed) or speed <= 0.0:
        return float("nan")
    return math.log(2.0) / speed


def _cross_section(
    df: pd.DataFrame,
    var: str,
    carry: list[str],
    entity: str,
    time: str,
    start: float | None,
    end: float | None,
    annualize: bool,
    func: str,
) -> tuple[pd.DataFrame, float, float, float]:
    """Build the one-row-per-unit growth cross-section over a common window.

    Returns ``(cs, horizon, t0, t1)`` where ``cs`` has columns ``entity``, ``initial``,
    ``final``, ``growth`` (the log-difference, divided by the horizon when
    ``annualize``) and one column per ``carry`` name (its initial-period value).
    """
    cols = list(dict.fromkeys([entity, time, var, *carry]))
    sub = df[cols].copy()
    sub[time] = pd.to_numeric(sub[time], errors="coerce")
    years = sub[time].dropna()
    if years.empty:
        raise ValueError(f"{func}: time column {time!r} has no numeric values")
    t0 = float(start) if start is not None else float(years.min())
    t1 = float(end) if end is not None else float(years.max())
    if t1 <= t0:
        raise ValueError(f"{func}: end ({t1:g}) must be after start ({t0:g})")
    horizon = t1 - t0

    clash = [c for c in carry if c in _RESERVED]
    if clash:
        raise ValueError(
            f"{func}: control/fixed-effect name(s) {clash} collide with the reserved "
            f"cross-section columns {list(_RESERVED)} — rename them"
        )

    # groupby(...).first() skips NaN per column, so a missing-valued duplicate row
    # cannot evict an otherwise-valid observation of the same unit.
    init = sub[sub[time] == t0].groupby(entity, observed=True).first()
    fin = sub[sub[time] == t1].groupby(entity, observed=True).first()
    if init.empty or fin.empty:
        raise ValueError(
            f"{func}: no units with data at both start ({t0:g}) and end ({t1:g}); "
            "check the periods present in the panel or pass start=/end="
        )

    cs = pd.DataFrame(index=init.index)
    cs["initial"] = init[var]
    cs["final"] = fin[var].reindex(cs.index)
    for c in carry:
        cs[c] = init[c]
    cs = cs.dropna(subset=["initial", "final"])
    if cs.empty:
        raise ValueError(
            f"{func}: no units with a non-missing {var!r} at both {t0:g} and {t1:g}"
        )

    non_positive = (cs["initial"] <= 0.0) | (cs["final"] <= 0.0)
    if bool(non_positive.any()):
        offending = list(cs.index[non_positive].astype(str)[:5])
        raise ValueError(
            f"{func}: {var!r} has non-positive endpoint values for "
            f"{int(non_positive.sum())} unit(s) (e.g. {offending}) — growth is the "
            "log-difference, so the variable must be strictly positive (pass levels, "
            "not logs or demeaned values)"
        )

    log_diff = np.log(cs["final"].to_numpy(dtype=float)) - np.log(
        cs["initial"].to_numpy(dtype=float)
    )
    cs["growth"] = log_diff / horizon if annualize else log_diff
    cs = cs.reset_index()
    return cs[[entity, "initial", "final", "growth", *carry]], horizon, t0, t1


def growth_cross_section(
    df: pd.DataFrame,
    var: str,
    controls: Sequence[str] | str | None = None,
    *,
    entity: str | None = None,
    time: str | None = None,
    start: float | None = None,
    end: float | None = None,
    annualize: bool = True,
) -> pd.DataFrame:
    """Build the per-unit growth cross-section a convergence analysis starts from.

    For each unit observed at both endpoints of a common window, the function records
    the ``initial`` and ``final`` level of ``var`` and the log growth between them:
    ``growth = (log(final) - log(initial)) / T`` when ``annualize`` (the average
    per-period log growth over the horizon ``T = end - start``), or the raw
    log-difference otherwise. Controls are attached at their **initial-period** values.

    Parameters
    ----------
    df
        Long panel data frame.
    var
        Numeric, strictly positive variable in **levels** (e.g. GDP per capita); the
        log is taken internally.
    controls
        Optional column name(s) whose initial-period values are carried into the
        cross-section (the conditional-convergence controls).
    entity, time
        Panel identifiers. Default to those declared via :func:`geometrics.set_panel`.
    start, end
        First and last period of the growth window. Default to the earliest and latest
        period in the panel; only units observed at **both** endpoints are kept.
    annualize
        Divide the log-difference by the horizon ``T`` (default). ``False`` returns
        the total log growth over the window.

    Returns
    -------
    pandas.DataFrame
        One row per unit with columns ``entity``, ``initial``, ``final``, ``growth``
        and one column per control, with the panel entity re-declared on
        ``df.attrs`` (:func:`geometrics.set_panel`).

    Raises
    ------
    KeyError
        If ``var`` or a control is not a column of ``df``.
    TypeError
        If ``var`` or a control is not numeric.
    ValueError
        If the window is empty or inverted, no unit spans it, or ``var`` has
        non-positive endpoint values (the log is undefined).

    Examples
    --------
    Two units over a 10-period window; the low-income unit grows faster:

    ```python
    import pandas as pd

    from geometrics.convergence import growth_cross_section

    df = pd.DataFrame(
        {
            "region": ["A", "A", "B", "B"],
            "year": [2000, 2010, 2000, 2010],
            "gdppc": [1000.0, 2000.0, 4000.0, 5000.0],
        }
    )
    cs = growth_cross_section(df, "gdppc", entity="region", time="year")
    cs[["region", "initial", "final", "growth"]]
    ```
    """
    df = ensure_dataframe(df)
    controls = _as_list(controls)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # guaranteed by require_* above

    missing = [c for c in [var, *controls] if c not in df.columns]
    if missing:
        raise KeyError(f"growth_cross_section: column(s) not found in df: {missing}")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"growth_cross_section: {var!r} needs to be numeric")
    for c in controls:
        if not pdt.is_numeric_dtype(df[c]):
            raise TypeError(f"growth_cross_section: control {c!r} needs to be numeric")

    n_dup = int(df.duplicated([entity, time]).sum())
    if n_dup:
        warnings.warn(
            f"growth_cross_section: found {n_dup} duplicate ({entity!r}, {time!r}) "
            "row(s); the first non-missing value of each is used",
            GeometricsWarning,
            stacklevel=2,
        )

    cs, _horizon, _t0, _t1 = _cross_section(
        df, var, controls, entity, time, start, end, annualize, "growth_cross_section"
    )
    return set_panel(cs, entity=entity)


# ------------------------------------------------------------------------------------
# β-convergence
# ------------------------------------------------------------------------------------


def _fit_baseline_ols(
    y: np.ndarray, x_df: pd.DataFrame, vcov: str
) -> tuple[Any, float, float, float, float, int]:
    """Fit the statsmodels OLS baseline; return ``(model, beta, se, r2, aic, n)``."""
    design = sm.add_constant(x_df.astype(float), has_constant="add")
    cov = "HC1" if vcov == "hetero" else "nonrobust"
    res = sm.OLS(y, design).fit(cov_type=cov)
    return (
        res,
        float(res.params[_FOCAL]),
        float(res.bse[_FOCAL]),
        float(res.rsquared),
        float(res.aic),
        int(res.nobs),
    )


def _residualize(target: np.ndarray, others: pd.DataFrame) -> np.ndarray:
    """Return the residuals of ``target`` on ``others`` (plus a constant) via OLS."""
    design = sm.add_constant(others.astype(float), has_constant="add")
    return np.asarray(sm.OLS(target, design).fit().resid, dtype=float)


def _fit_spatial(
    model: str,
    y: np.ndarray,
    x: np.ndarray,
    names: list[str],
    w: W,
    n_draws: int,
    seed: int,
) -> tuple[Any, pd.DataFrame, list[bool] | None]:
    """Estimate the requested spreg model; return ``(model, impacts, slx_mask)``.

    ``"sar"`` / ``"sdm"`` use ``spreg.ML_Lag`` (the SDM adds ``slx_lags=1`` restricted
    to the full-rank lag mask) with Monte-Carlo impact inference; ``"sem"`` uses
    ``spreg.ML_Error`` and ``"slx"`` ``spreg.OLS(slx_lags=1)`` with analytic impacts.
    Impacts use the ``'simple'`` average-direct-impact multiplier (``direct = b``),
    matching Stata's ``estat impact`` and the source paper's Table 1.
    """
    import spreg

    w_dense = w.full()[0]
    mask: list[bool] | None = None
    slx_vars: Any = "All"
    if model in ("slx", "sdm"):
        mask = full_rank_lag_mask(x, w_dense)
        slx_vars = "All" if all(mask) else mask

    with contextlib.redirect_stdout(io.StringIO()):
        if model == "sar":
            fitted = spreg.ML_Lag(y=y, x=x, w=w, name_x=names, name_y="growth")
        elif model == "sdm":
            fitted = spreg.ML_Lag(
                y=y,
                x=x,
                w=w,
                slx_lags=1,
                slx_vars=slx_vars,
                name_x=names,
                name_y="growth",
            )
        elif model == "sem":
            fitted = spreg.ML_Error(y=y, x=x, w=w, name_x=names, name_y="growth")
        else:  # slx
            fitted = spreg.OLS(
                y=y,
                x=x,
                w=w,
                slx_lags=1,
                slx_vars=slx_vars,
                name_x=names,
                name_y="growth",
            )

    if model == "sar":
        impacts = mc_impacts(
            fitted,
            w_dense,
            names,
            has_slx=False,
            n_draws=n_draws,
            seed=seed,
            method="simple",
        )
    elif model == "sdm":
        impacts = mc_impacts(
            fitted,
            w_dense,
            names,
            has_slx=True,
            slx_mask=mask,
            n_draws=n_draws,
            seed=seed,
            method="simple",
        )
    else:  # sem / slx: no spatial multiplier -> analytic impacts
        impacts = analytic_slx_impacts(fitted, names, mask)
    return fitted, impacts, mask


def _scatter_fig(
    xv: np.ndarray,
    yv: np.ndarray,
    entities: np.ndarray,
    x_label: str,
    y_label: str,
    stat_lines: list[str],
    title: str | None,
    subtitle: str | None,
) -> go.Figure:
    """Growth-vs-initial scatter: OLS fit + 95% band, unit hover, stat box."""
    finite = np.isfinite(xv) & np.isfinite(yv)
    xv, yv, entities = xv[finite], yv[finite], entities[finite]
    order = np.argsort(xv)
    xs, ys, ents = xv[order], yv[order], entities[order]

    fig = go.Figure()
    if xs.size >= 3 and np.ptp(xs) > 0:
        design = sm.add_constant(xs, has_constant="add")
        pred = sm.OLS(ys, design).fit().get_prediction(design)
        fit = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([xs, xs[::-1]]),
                y=np.concatenate([ci[:, 1], ci[:, 0][::-1]]),
                fill="toself",
                fillcolor="rgba(0,0,0,0.12)",
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=False,
                name="ci",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=fit,
                mode="lines",
                line={"color": color_for(2), "width": 2},
                name="fit",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker={
                "color": color_for(0),
                "size": 8,
                "opacity": 0.75,
                "line": {"color": "white", "width": 0.5},
            },
            customdata=ents,
            hovertemplate=(
                "%{customdata}<br>initial=%{x:.4g}<br>growth=%{y:.4g}<extra></extra>"
            ),
            name="units",
            showlegend=False,
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.7)",
        text="<br>".join(stat_lines),
    )
    apply_default_layout(
        fig,
        title=title,
        subtitle=subtitle,
        xaxis={"title": x_label},
        yaxis={"title": y_label},
    )
    return fig


def _stat_lines(
    beta: float, se: float, r2: float, n: int, speed: float, hl: float
) -> list[str]:
    """Compose the stat-box lines of the β-convergence scatter."""
    lines = [f"β = {beta:.4g}", f"SE = {se:.4g}"]
    if math.isfinite(r2):
        lines.append(f"R² = {r2:.3f}")
    lines.append(f"N = {n:,}")
    if math.isfinite(speed):
        lines.append(f"speed λ = {speed:.4g}")
    if math.isfinite(hl):
        lines.append(f"half-life = {hl:.4g}")
    return lines


def _fmt_est(est: float, se: float) -> str:
    """Format one estimate cell as ``est*** (se)`` (``—`` when not finite)."""
    if not math.isfinite(est):
        return "—"
    cell = f"{est:.4f}"
    if math.isfinite(se):
        cell = f"{cell}{stars(est, se)} ({se:.4f})"
    return cell


def _fmt_plain(value: float, *, integer: bool = False, digits: int = 3) -> str:
    """Format one plain numeric cell (``—`` when not finite)."""
    if not math.isfinite(value):
        return "—"
    if integer:
        return f"{round(value):,}"
    return f"{value:.{digits}g}"


def _beta_summary_and_gt(
    columns: dict[str, list[float]],
    var_label: str,
    horizon: float,
    w_spec: str | None,
) -> tuple[pd.DataFrame, Any]:
    """Build the numeric ``summary`` frame and its Great-Tables rendering.

    ``columns`` maps a column label (``"ols"`` and, for spatial models, the model
    name) to its 13 metric values in :data:`_BETA_METRICS` order.
    """
    from great_tables import GT

    summary = pd.DataFrame({"metric": list(_BETA_METRICS), **columns})

    display_rows = [
        ("Direct β", "est", "direct", "se_direct"),
        ("Indirect β", "est", "indirect", "se_indirect"),
        ("Total β", "est", "total", "se_total"),
        ("ρ (spatial lag)", "plain", "rho", None),
        ("λ (spatial error)", "plain", "lambda", None),
        ("R²", "plain", "r2", None),
        ("AIC", "plain", "aic", None),
        ("N", "int", "n_obs", None),
        ("Speed of convergence (λ)", "plain", "speed", None),
        ("Half-life", "plain", "half_life", None),
    ]
    idx = {m: i for i, m in enumerate(_BETA_METRICS)}
    disp: dict[str, list[str]] = {"Metric": [r[0] for r in display_rows]}
    for col, values in columns.items():
        cells = []
        for _label, kind, metric, se_metric in display_rows:
            if kind == "est":
                cells.append(
                    _fmt_est(
                        values[idx[metric]],
                        values[idx[se_metric]] if se_metric else float("nan"),
                    )
                )
            elif kind == "int":
                cells.append(_fmt_plain(values[idx[metric]], integer=True))
            else:
                cells.append(_fmt_plain(values[idx[metric]], digits=4))
        disp[col.upper()] = cells

    subtitle = f"growth over a {horizon:g}-period horizon vs. initial log level"
    if w_spec is not None:
        subtitle += f" — W: {w_spec}"
    gt = (
        GT(pd.DataFrame(disp), rowname_col="Metric")
        .tab_header(title=f"β-convergence: {var_label}", subtitle=subtitle)
        .tab_source_note(
            "β < 0 indicates convergence. Direct/Indirect/Total follow the "
            "LeSage-Pace impact decomposition of the initial-level term (Monte-Carlo "
            "standard errors for SAR/SDM). Speed λ = -ln(1 + β·T)/T per period; "
            "half-life = ln 2 / λ. *** p<0.01, ** p<0.05, * p<0.10."
        )
    )
    return summary, gt


def analyze_beta_convergence(
    df: pd.DataFrame,
    var: str,
    controls: Sequence[str] | str | None = None,
    *,
    entity: str | None = None,
    time: str | None = None,
    start: float | None = None,
    end: float | None = None,
    model: str = "ols",
    gdf: gpd.GeoDataFrame | None = None,
    w: W | None = None,
    fixed_effects: Sequence[str] | str | None = None,
    vcov: str = "hetero",
    n_draws: int = 10_000,
    seed: int = 20250620,
    min_obs: int = 10,
    title: str | None = None,
) -> BetaConvergenceResult:
    """β-convergence of a panel variable, from plain OLS to the spatial Durbin model.

    Builds the growth cross-section (:func:`growth_cross_section`) over a common window
    and regresses each unit's annualized log growth on its **initial log level** (plus
    initial-period ``controls`` and ``fixed_effects`` dummies). A **negative** slope β
    is convergence: units that start lower grow faster. The slope maps to a speed
    ``λ = -ln(1 + β·T)/T`` and a half-life ``ln 2 / λ``.

    With a spatial ``model`` the cross-section is first aligned to the entity geometry
    ``gdf`` (and the weights ``w``) so rows, polygons and weights always match, and the
    regression is estimated with :mod:`spreg`: ``"sar"`` (ML spatial lag), ``"sem"``
    (ML spatial error), ``"slx"`` (spatially lagged regressors) or ``"sdm"`` (spatial
    Durbin). The initial-level term is decomposed into **direct**, **indirect**
    (spillover) and **total** impacts with Monte-Carlo standard errors from ``n_draws``
    draws (the ``impacts`` table covers every regressor); speed and half-life derive
    from the **total** impact. An OLS baseline on the same sample is always reported
    alongside.

    Parameters
    ----------
    df
        Long panel data frame.
    var
        Numeric, strictly positive variable in **levels** (the log is taken
        internally; growth is the annualized log-difference).
    controls
        Optional control name(s), entering at their initial-period values
        (conditional convergence).
    entity, time
        Panel identifiers. Default to those declared via :func:`geometrics.set_panel`.
    start, end
        Growth window endpoints; default to the earliest and latest period.
    model
        ``"ols"`` (default), ``"sar"``, ``"sem"``, ``"slx"`` or ``"sdm"``.
    gdf
        Entity geometry (required for the spatial models; optional for OLS, where it
        only adds the growth map and restricts the sample to matched units).
    w
        ``libpysal`` weights aligned to the ``gdf`` ids. ``None`` builds the default
        weights (queen contiguity for polygons, 6-nearest-neighbor otherwise) with a
        :class:`~geometrics.GeometricsWarning`.
    fixed_effects
        Optional categorical column name(s) (e.g. a state id) entered as dummy
        variables (first level dropped), valued at the initial period.
    vcov
        Standard errors of the OLS baseline: ``"hetero"`` (HC1, default) or
        ``"iid"``. The spatial models use their spreg (ML) covariance.
    n_draws
        Monte-Carlo draws for the SAR/SDM impact standard errors.
    seed
        Seed for the Monte-Carlo draws (reproducible).
    min_obs
        Minimum number of cross-section units required.
    title
        Title for the growth-vs-initial scatter.

    Returns
    -------
    BetaConvergenceResult
        The growth cross-section ``df``; the scatter ``fig``, the Frisch-Waugh-Lovell
        partial scatter ``fig_conditional`` (``None`` without controls/fixed effects)
        and the growth choropleth ``fig_map`` (``None`` without ``gdf``); the estimate
        table ``gt`` / ``summary`` (OLS column always, an impact column for spatial
        models); the fitted ``models``; the ``beta_direct`` / ``beta_indirect`` /
        ``beta_total`` triple with standard errors; ``rho`` / ``lam``; ``speed`` and
        ``half_life``; and the per-regressor ``impacts`` table for spatial models.

    Raises
    ------
    KeyError
        If ``var``, a control or a fixed-effect column is missing from ``df``.
    TypeError
        If ``var`` or a control is not numeric.
    ValueError
        For an unknown ``model`` / ``vcov``, a spatial model without ``gdf``, an
        empty or inverted window, non-positive values of ``var``, too few units, or a
        zero-variance initial level.

    Examples
    --------
    Unconditional convergence across 12 regions over one decade (β < 0 by
    construction — low-income regions grow faster):

    ```python
    import numpy as np
    import pandas as pd

    from geometrics.convergence import analyze_beta_convergence

    ids = [f"r{i}" for i in range(12)]
    y0 = np.linspace(1000.0, 12000.0, 12)
    growth = 0.05 - 0.02 * np.log(y0) / np.log(y0).mean()  # poorer -> faster
    df = pd.concat(
        [
            pd.DataFrame({"region": ids, "year": 2000, "gdppc": y0}),
            pd.DataFrame(
                {"region": ids, "year": 2010, "gdppc": y0 * np.exp(10 * growth)}
            ),
        ]
    )
    res = analyze_beta_convergence(df, "gdppc", entity="region", time="year")
    round(res.beta_total, 4), res.model
    ```
    """
    df = ensure_dataframe(df)
    controls = _as_list(controls)
    fe_cols = _as_list(fixed_effects)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # guaranteed by require_* above
    func = "analyze_beta_convergence"

    missing = [c for c in [var, *controls, *fe_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"{func}: column(s) not found in df: {missing}")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"{func}: {var!r} needs to be numeric")
    for c in controls:
        if not pdt.is_numeric_dtype(df[c]):
            raise TypeError(f"{func}: control {c!r} needs to be numeric")
    if model not in _BETA_MODELS:
        raise ValueError(
            f"{func}: unknown model {model!r}; choose from {list(_BETA_MODELS)}"
        )
    if vcov not in ("hetero", "iid"):
        raise ValueError(f"{func}: vcov must be 'hetero' or 'iid', got {vcov!r}")
    spatial = model != "ols"
    if spatial and gdf is None:
        raise ValueError(
            f"{func}: model={model!r} needs entity geometry — pass gdf=... "
            "(and optionally w=make_weights(gdf, ...))"
        )
    if _FOCAL in {*controls, *fe_cols}:
        raise ValueError(
            f"{func}: the name {_FOCAL!r} is reserved for the initial log level"
        )

    var_label = resolve_label(df, var)
    ent_disp = entity_display_map(df, entity, resolve_entity_name(df))
    notes: list[str] = []

    n_dup = int(df.duplicated([entity, time]).sum())
    if n_dup:
        msg = (
            f"{func}: found {n_dup} duplicate ({entity!r}, {time!r}) row(s); the "
            "first non-missing value of each is used"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)

    cs, horizon, t0, t1 = _cross_section(
        df, var, [*controls, *fe_cols], entity, time, start, end, True, func
    )
    need = ["initial", "growth", *controls, *fe_cols]
    n_before = len(cs)
    data = cs.dropna(subset=need)
    if len(data) < n_before:
        msg = (
            f"{func}: dropped {n_before - len(data)} of {n_before} unit(s) with "
            f"missing values in {need}"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)

    # --- geometry / weights alignment (spatial models, or OLS with a map) ----------
    w_used: W | None = None
    w_spec: str | None = None
    gdf_entity: str | None = None
    aligned: pd.DataFrame
    if gdf is not None:
        gdf = ensure_geodataframe(gdf, func=func)
        gdf_entity = resolve_gdf_entity(gdf)
        if spatial and w is None:
            w = _default_weights_for(gdf, func)
            notes.append(
                f"{func}: no spatial weights supplied — defaulted to "
                f"{w.geometrics_meta['spec']}"
            )
        if not spatial and w is not None:
            msg = f"{func}: w is ignored for model='ols'"
            warnings.warn(msg, GeometricsWarning, stacklevel=2)
            notes.append(msg)
            w = None
        aligned, w_used, info = _align_cross_section(
            data,
            gdf,
            cols=["initial", "final", "growth", *controls, *fe_cols],
            entity=entity,
            w=w,
            min_obs=max(3, min_obs),
            func=func,
        )
        notes.extend(info["notes"])
        est = pd.DataFrame(aligned.drop(columns=aligned.geometry.name))
        ent_col = gdf_entity
    else:
        aligned = data
        est = data
        ent_col = entity
    if w_used is not None:
        meta = dict(getattr(w_used, "geometrics_meta", {}) or {})
        w_spec = meta.get("spec") or _describe_w_fallback(w_used)

    if len(est) < max(3, min_obs):
        raise ValueError(
            f"{func}: only {len(est)} unit(s) with data at both endpoints; need at "
            f"least {max(3, min_obs)}. Try a different start/end or lower min_obs."
        )
    log_initial = np.log(est["initial"].to_numpy(dtype=float))
    if np.ptp(log_initial) <= 1e-10 * (1.0 + float(np.abs(log_initial).max())):
        raise ValueError(
            f"{func}: initial {var!r} has (near) zero variance across units; a "
            "convergence slope is not identified"
        )

    # --- design -------------------------------------------------------------------
    x_df = pd.DataFrame({_FOCAL: log_initial}, index=est.index)
    for c in controls:
        x_df[c] = est[c].to_numpy(dtype=float)
    dummy_names: list[str] = []
    for c in fe_cols:
        dummies = pd.get_dummies(est[c].astype(str), prefix=c, drop_first=True).astype(
            float
        )
        for name in dummies.columns:
            x_df[str(name)] = dummies[name].to_numpy()
            dummy_names.append(str(name))
    names = [_FOCAL, *controls, *dummy_names]
    y = est["growth"].to_numpy(dtype=float)

    # --- estimation: OLS baseline always; the spatial model when requested ---------
    ols_res, beta_o, se_o, r2_o, aic_o, n_o = _fit_baseline_ols(y, x_df, vcov)
    speed_o = _speed_of_convergence(beta_o, horizon)
    hl_o = _half_life(speed_o)
    models: list[Any] = [ols_res]

    impacts: pd.DataFrame | None = None
    if spatial:
        assert w_used is not None  # guaranteed by the alignment path above
        fitted, impacts, _mask = _fit_spatial(
            model,
            y.reshape(-1, 1),
            x_df.to_numpy(dtype=float),
            names,
            w_used,
            n_draws,
            seed,
        )
        models.append(fitted)
        focal = impacts.loc[impacts["term"] == _FOCAL].iloc[0]
        beta_direct = float(focal["direct"])
        beta_indirect = float(focal["indirect"])
        beta_total = float(focal["total"])
        se_direct = float(focal["se_direct"])
        se_indirect = float(focal["se_indirect"])
        se_total = float(focal["se_total"])
        rho = float(getattr(fitted, "rho", float("nan")))
        lam = float(getattr(fitted, "lam", float("nan")))
        r2 = float(getattr(fitted, "pr2", getattr(fitted, "r2", float("nan"))))
        aic = float(getattr(fitted, "aic", float("nan")))
        n_obs = int(fitted.n)
    else:
        beta_direct = beta_total = beta_o
        beta_indirect = float("nan")
        se_direct = se_total = se_o
        se_indirect = float("nan")
        rho = lam = float("nan")
        r2, aic, n_obs = r2_o, aic_o, n_o

    speed = _speed_of_convergence(beta_total, horizon)
    hl = _half_life(speed)
    if not math.isfinite(speed):
        notes.append(
            f"{func}: the speed of convergence is undefined (1 + β·T ≤ 0 for "
            f"β = {beta_total:.4g}, T = {horizon:g})"
        )
    elif not math.isfinite(hl):
        notes.append(
            f"{func}: the half-life is undefined (speed λ = {speed:.4g} ≤ 0 — no "
            "convergence)"
        )

    # --- table ----------------------------------------------------------------------
    columns: dict[str, list[float]] = {
        "ols": [
            beta_o,
            se_o,
            float("nan"),
            float("nan"),
            beta_o,
            se_o,
            float("nan"),
            float("nan"),
            r2_o,
            aic_o,
            float(n_o),
            speed_o,
            hl_o,
        ]
    }
    if spatial:
        columns[model] = [
            beta_direct,
            se_direct,
            beta_indirect,
            se_indirect,
            beta_total,
            se_total,
            rho,
            lam,
            r2,
            aic,
            float(n_obs),
            speed,
            hl,
        ]
    summary, gt = _beta_summary_and_gt(columns, var_label, horizon, w_spec)

    # --- figures ----------------------------------------------------------------------
    ents = (
        est[ent_col].map(lambda u: ent_disp.get(str(u), str(u))).to_numpy(dtype=object)
    )
    subtitle = model.upper() if w_spec is None else f"{model.upper()} — W: {w_spec}"
    fig = _scatter_fig(
        log_initial,
        y,
        ents,
        f"Initial log {var_label}",
        f"Growth of {var_label} (annualized log growth)",
        _stat_lines(beta_total, se_total, r2, n_obs, speed, hl),
        title if title is not None else f"β-convergence: {var_label}",
        subtitle,
    )

    fig_conditional: go.Figure | None = None
    partial_cols = [*controls, *dummy_names]
    if partial_cols:
        x_res = _residualize(log_initial, x_df[partial_cols])
        y_res = _residualize(y, x_df[partial_cols])
        fig_conditional = _scatter_fig(
            x_res,
            y_res,
            ents,
            f"Residualized initial log {var_label}",
            f"Residualized growth of {var_label}",
            _stat_lines(beta_o, se_o, r2_o, n_o, speed_o, hl_o),
            f"Conditional β-convergence (FWL): {var_label}",
            "controls and fixed effects partialled out (OLS baseline)",
        )

    fig_map: go.Figure | None = None
    if gdf is not None:
        assert gdf_entity is not None  # resolved alongside gdf above
        fig_map, _bins = classified_map(
            aligned,  # type: ignore[arg-type]  # GeoDataFrame on this path
            est["growth"].to_numpy(dtype=float),
            entity=gdf_entity,
            scheme="fisherjenks",
            k=5,
            title=f"Growth of {var_label}, {t0:g}-{t1:g}",
            legend_title="growth",
            hover_names=ent_disp,
        )

    out_df = pd.DataFrame(
        {
            entity: est[ent_col].to_numpy(),
            **{
                c: est[c].to_numpy()
                for c in ["initial", "final", "growth", *controls, *fe_cols]
            },
        }
    )

    return BetaConvergenceResult(
        df=out_df,
        fig=fig,
        fig_conditional=fig_conditional,
        fig_map=fig_map,
        gt=gt,
        summary=summary,
        models=models,
        model=model,
        var=var,
        controls=tuple(controls),
        horizon=float(horizon),
        beta_direct=beta_direct,
        beta_indirect=beta_indirect,
        beta_total=beta_total,
        se_direct=se_direct,
        se_indirect=se_indirect,
        se_total=se_total,
        rho=rho,
        lam=lam,
        r2=r2,
        aic=aic,
        n_obs=n_obs,
        speed=speed,
        half_life=hl,
        impacts=impacts,
        n_draws=n_draws if model in ("sar", "sdm") else 0,
        w_spec=w_spec,
        notes=tuple(notes),
    )


def _default_weights_for(gdf: gpd.GeoDataFrame, func: str) -> W:
    """Build the library-default weights for ``gdf`` (thin wrapper for import order)."""
    from geometrics.weights import _default_weights

    return _default_weights(gdf, func=func)


def _describe_w_fallback(w: W) -> str:
    """Describe a user-supplied W without ``geometrics_meta`` (thin wrapper)."""
    from geometrics.weights import _describe_w

    return _describe_w(w)


# ------------------------------------------------------------------------------------
# σ-convergence
# ------------------------------------------------------------------------------------

_SIGMA_MEASURES = ("std", "gini", "cv")
_SIGMA_LABELS = {
    "std": "Standard deviation",
    "gini": "Gini index",
    "cv": "Coefficient of variation",
}


def _gini(x: np.ndarray) -> float:
    """Return the Gini coefficient of ``x`` (relative mean absolute difference / 2).

    Uses the sorted-order identity ``G = 2*sum(i*x_(i)) / (n*sum x) - (n + 1)/n`` on
    the values sorted ascending. Returns ``nan`` for fewer than two finite values, a
    non-positive sum, or any negative value (the index is only defined on non-negative
    data).
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2 or bool(np.any(v < 0.0)):
        return float("nan")
    total = float(v.sum())
    if total <= 0.0:
        return float("nan")
    v = np.sort(v)
    idx = np.arange(1, n + 1, dtype=float)
    return float(2.0 * float(np.sum(idx * v)) / (n * total) - (n + 1.0) / n)


def _balance_offenders(
    work: pd.DataFrame, entity: str, time: str
) -> tuple[int, int, int, int]:
    """Return ``(n_units, n_periods, units_missing, periods_missing)`` describing balance.

    A panel is **balanced** when every unit is observed in every period — i.e. both
    ``units_missing`` and ``periods_missing`` are zero.
    """
    n_periods = int(work[time].nunique())
    n_units = int(work[entity].nunique())
    per_unit = work.groupby(entity, observed=True)[time].nunique()
    per_period = work.groupby(time, observed=True)[entity].nunique()
    units_missing = int((per_unit < n_periods).sum())
    periods_missing = int((per_period < n_units).sum())
    return n_units, n_periods, units_missing, periods_missing


def _period_table(work: pd.DataFrame, value_col: str, time: str) -> pd.DataFrame:
    """One row per period: ``time``, ``n_units``, ``mean``, ``std`` (ddof=1), ``gini``, ``cv``."""
    rows: list[dict[str, float]] = []
    for t in sorted(work[time].unique()):
        v = work.loc[work[time] == t, value_col].to_numpy(dtype=float)
        mean = float(np.mean(v))
        std = float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
        rows.append(
            {
                time: float(t),
                "n_units": int(v.size),
                "mean": mean,
                "std": std,
                "gini": _gini(v),
                "cv": std / mean if mean != 0.0 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(time).reset_index(drop=True)


def _dispersion_trend(
    tab: pd.DataFrame, measure: str, time: str, vcov: str
) -> tuple[Any | None, float, float, float, float, float, int]:
    """Fit ``log(<measure>) ~ time`` over periods with a positive, finite dispersion.

    Returns ``(model, intercept, slope, se, pvalue, r2, n_used)``. ``model`` is
    ``None`` and the scalars ``nan`` when fewer than three usable periods remain
    (``log`` needs a positive dispersion). The slope is the average proportional change
    in the dispersion per period; a **negative** slope is σ-convergence.
    """
    nan = float("nan")
    disp = tab[measure].to_numpy(dtype=float)
    tv = tab[time].to_numpy(dtype=float)
    ok = np.isfinite(disp) & (disp > 0.0)
    n_used = int(ok.sum())
    if n_used < 3:
        return None, nan, nan, nan, nan, nan, n_used
    design = pd.DataFrame({"const": 1.0, "t": tv[ok]})
    cov = "HC1" if vcov == "hetero" else "nonrobust"
    model = sm.OLS(np.log(disp[ok]), design).fit(cov_type=cov)
    return (
        model,
        float(model.params["const"]),
        float(model.params["t"]),
        float(model.bse["t"]),
        float(model.pvalues["t"]),
        float(model.rsquared),
        n_used,
    )


def _sigma_annotation(trends: dict[str, tuple], n_periods: int) -> list[str]:
    """Build the annotation-box lines reporting each measure's log-dispersion trend."""

    def line(measure: str, label: str) -> str:
        slope, pval = trends[measure][2], trends[measure][4]
        if not math.isfinite(slope):
            return f"{label}: not estimated"
        verdict = "converging" if slope < 0 else "diverging"
        ptxt = f", p = {pval:.2g}" if math.isfinite(pval) else ""
        return f"{label} trend = {slope:.3g}/period ({verdict}{ptxt})"

    return [line("std", "Std"), line("gini", "Gini"), f"periods = {n_periods}"]


def _sigma_fig(
    tab: pd.DataFrame,
    trends: dict[str, tuple],
    time: str,
    time_label: str,
    var_label: str,
    title: str | None,
) -> go.Figure:
    """Build the dual-axis figure: std (left axis) and Gini (right axis) over time."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    tv = tab[time].to_numpy(dtype=float)
    std_color, gini_color = color_for(0), color_for(1)

    fig.add_trace(
        go.Scatter(
            x=tv,
            y=tab["std"].to_numpy(dtype=float),
            mode="lines+markers",
            name="Std. dev.",
            line={"color": std_color, "width": 2},
            marker={"color": std_color, "size": 7},
            hovertemplate=f"{time_label} = %{{x}}<br>std = %{{y:.4g}}<extra></extra>",
        ),
        secondary_y=False,
    )
    gini_v = tab["gini"].to_numpy(dtype=float)
    has_gini = bool(np.any(np.isfinite(gini_v)))
    if has_gini:
        fig.add_trace(
            go.Scatter(
                x=tv,
                y=gini_v,
                mode="lines+markers",
                name="Gini index",
                line={"color": gini_color, "width": 2},
                marker={"color": gini_color, "size": 7},
                hovertemplate=(
                    f"{time_label} = %{{x}}<br>Gini = %{{y:.4g}}<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

    # Dashed exp(log-trend) overlays so the fitted convergence path is visible.
    for measure, color, sec in (("std", std_color, False), ("gini", gini_color, True)):
        model = trends[measure][0]
        if model is None or (measure == "gini" and not has_gini):
            continue
        b0, b1 = float(trends[measure][1]), float(trends[measure][2])
        fig.add_trace(
            go.Scatter(
                x=tv,
                y=np.exp(b0 + b1 * tv),
                mode="lines",
                line={"color": color, "width": 1, "dash": "dash"},
                hoverinfo="skip",
                showlegend=False,
                name=f"{measure} trend",
            ),
            secondary_y=sec,
        )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.7)",
        text="<br>".join(_sigma_annotation(trends, len(tab))),
    )
    apply_default_layout(
        fig,
        title=title,
        xaxis={"title": time_label},
    )
    fig.update_yaxes(
        title_text=f"Std. dev. of log {var_label}", color=std_color, secondary_y=False
    )
    fig.update_yaxes(
        title_text=f"Gini of log {var_label}", color=gini_color, secondary_y=True
    )
    return fig


def _sigma_summary_and_gt(
    trends: dict[str, tuple],
    var_label: str,
    n_periods: int,
    n_units: int,
) -> tuple[pd.DataFrame, Any]:
    """Build the numeric ``summary`` frame and its Great-Tables trend rendering."""
    from great_tables import GT

    def converging(slope: float) -> bool:
        return bool(slope < 0) if math.isfinite(slope) else False

    summary = pd.DataFrame(
        {
            "measure": list(_SIGMA_MEASURES),
            "slope": [trends[m][2] for m in _SIGMA_MEASURES],
            "se": [trends[m][3] for m in _SIGMA_MEASURES],
            "pvalue": [trends[m][4] for m in _SIGMA_MEASURES],
            "r2": [trends[m][5] for m in _SIGMA_MEASURES],
            "n_periods_used": [trends[m][6] for m in _SIGMA_MEASURES],
            "converging": [converging(trends[m][2]) for m in _SIGMA_MEASURES],
        }
    )

    def fmt(value: float, *, dp: bool = False) -> str:
        if not math.isfinite(value):
            return "—"
        return f"{value:.3f}" if dp else f"{value:.4g}"

    disp = pd.DataFrame(
        {
            "Measure": [_SIGMA_LABELS[m] for m in _SIGMA_MEASURES],
            "Trend (per period)": [fmt(trends[m][2]) for m in _SIGMA_MEASURES],
            "Std. error": [fmt(trends[m][3]) for m in _SIGMA_MEASURES],
            "p-value": [fmt(trends[m][4], dp=True) for m in _SIGMA_MEASURES],
            "σ-convergence": [
                "—"
                if not math.isfinite(trends[m][2])
                else ("yes" if trends[m][2] < 0 else "no")
                for m in _SIGMA_MEASURES
            ],
        }
    )
    gt = (
        GT(disp, rowname_col="Measure")
        .tab_header(
            title=f"σ-convergence: {var_label}",
            subtitle=(
                f"trend of log dispersion of log {var_label} over {n_periods} "
                f"periods, {n_units} units"
            ),
        )
        .tab_source_note(
            "A negative trend in log dispersion is σ-convergence (the cross-sectional "
            "distribution is narrowing). Trend = OLS slope of ln(dispersion) on time; "
            "dispersion is measured on the log of the variable."
        )
    )
    return summary, gt


def analyze_sigma_convergence(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    start: float | None = None,
    end: float | None = None,
    min_periods: int = 3,
    vcov: str = "hetero",
    title: str | None = None,
) -> SigmaConvergenceResult:
    r"""σ-convergence: track and test the cross-sectional dispersion of a panel variable.

    For each period the function measures how spread out the **log** of ``var`` is
    across units — the standard deviation (the classic σ), the Gini index and the
    coefficient of variation — and then asks whether that dispersion shrinks over time
    by regressing the **log dispersion** on time. A **negative** trend slope is
    σ-convergence: the cross-sectional distribution is narrowing (units are becoming
    more alike). This is the distributional complement to β-convergence
    (:func:`analyze_beta_convergence`).

    The variable is taken in **levels** and logged internally (so pass GDP per capita,
    not log GDP per capita). The panel must be **balanced** (every unit present in
    every period) so the dispersion is comparable across periods.

    Parameters
    ----------
    df
        Long panel data frame.
    var
        Numeric, strictly positive variable in levels whose log dispersion is tracked.
    entity, time
        Panel identifiers. Default to those declared via :func:`geometrics.set_panel`.
    start, end
        Optional first and last period to include. Default to the full range; the
        retained window must still be balanced.
    min_periods
        Minimum number of periods required to estimate a dispersion trend (at
        least 3).
    vcov
        Standard errors of the trend regressions: ``"hetero"`` (HC1, default) or
        ``"iid"``. Does not change the point estimates.
    title
        Title for the dual-axis figure.

    Returns
    -------
    SigmaConvergenceResult
        The per-period dispersion table ``df`` (``time``, ``n_units``, ``mean``,
        ``std``, ``gini``, ``cv`` — all on log values); the dual-axis ``fig`` (std on
        the left axis, Gini on the right, with fitted trend overlays); the trend table
        ``gt`` / ``summary``; the fitted trend ``models``; and the headline trend
        scalars (``std_slope`` / ``std_se`` / ``std_pvalue`` / ``std_r2`` plus the
        ``gini_*`` and ``cv_*`` counterparts). ``notes`` records any degraded measure.

    Raises
    ------
    KeyError
        If ``var`` is not a column of ``df``.
    TypeError
        If ``var`` is not numeric.
    ValueError
        If no usable rows remain, ``var`` has non-positive values (the log is
        undefined), the panel is unbalanced, or there are too few units/periods.

    Notes
    -----
    For a dispersion measure :math:`D_t` computed cross-sectionally at each period
    ``t``, the trend is the OLS slope ``b`` in :math:`\ln D_t = a + b t +
    \varepsilon_t`, so ``b`` is the average proportional change in dispersion per
    period and ``b < 0`` is σ-convergence. The standard deviation uses ``ddof = 1``;
    the Gini index is the relative mean absolute difference over twice the mean; the
    coefficient of variation is the standard deviation over the mean. See Barro &
    Sala-i-Martin, *Economic Growth*, ch. 11.

    Examples
    --------
    Dispersion shrinks by construction (each unit moves halfway to the mean):

    ```python
    import numpy as np
    import pandas as pd

    from geometrics.convergence import analyze_sigma_convergence

    ids = [f"r{i}" for i in range(8)]
    y0 = np.linspace(1000.0, 8000.0, 8)
    frames = [
        pd.DataFrame(
            {
                "region": ids,
                "year": 2000 + t,
                "gdppc": np.exp(
                    np.log(y0).mean() + (np.log(y0) - np.log(y0).mean()) * 0.8**t
                ),
            }
        )
        for t in range(5)
    ]
    res = analyze_sigma_convergence(
        pd.concat(frames), "gdppc", entity="region", time="year"
    )
    res.std_slope < 0
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # guaranteed by require_* above
    func = "analyze_sigma_convergence"

    if var not in df.columns:
        raise KeyError(f"{func}: column {var!r} not found in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"{func}: {var!r} needs to be numeric")
    if vcov not in ("hetero", "iid"):
        raise ValueError(f"{func}: vcov must be 'hetero' or 'iid', got {vcov!r}")

    var_label = resolve_label(df, var)
    time_label = resolve_label(df, time)
    notes: list[str] = []

    work = df[[entity, time, var]].copy()
    work[time] = pd.to_numeric(work[time], errors="coerce")
    work = work.dropna(subset=[time, var])
    if work.empty:
        raise ValueError(
            f"{func}: no rows with both a numeric {time!r} and a non-missing {var!r}"
        )
    if start is not None:
        work = work[work[time] >= float(start)]
    if end is not None:
        work = work[work[time] <= float(end)]
    if work.empty:
        raise ValueError(f"{func}: no rows remain in the [start, end] window")

    before = len(work)
    work = work.groupby([time, entity], observed=True, as_index=False).first()
    if len(work) < before:
        msg = (
            f"{func}: found duplicate ({entity!r}, {time!r}) rows; kept the first of "
            f"each ({before - len(work)} dropped)"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)

    if bool((work[var] <= 0.0).any()):
        raise ValueError(
            f"{func}: {var!r} has non-positive values — dispersion is measured on "
            "log values, so the variable must be strictly positive (pass levels)"
        )
    work["_gm_log"] = np.log(work[var].to_numpy(dtype=float))

    n_units, n_periods, units_missing, periods_missing = _balance_offenders(
        work, entity, time
    )
    if units_missing or periods_missing:
        raise ValueError(
            f"{func}: panel is not balanced: {units_missing} of {n_units} units are "
            f"missing in some period and {periods_missing} of {n_periods} periods "
            "are missing some units. σ-convergence compares dispersion across a fixed "
            "set of units; restrict to a balanced window with start=/end= or drop "
            "the offending units."
        )
    if n_units < 2:
        raise ValueError(
            f"{func}: need >= 2 units to measure cross-sectional dispersion; "
            f"got {n_units}"
        )
    if n_periods < max(3, min_periods):
        raise ValueError(
            f"{func}: need >= {max(3, min_periods)} periods to estimate a dispersion "
            f"trend; got {n_periods}"
        )

    tab = _period_table(work, "_gm_log", time)

    # Degradation guards: the Gini needs non-negative values; the CV a strictly
    # positive mean. Both are evaluated on the *log* values.
    if bool((work["_gm_log"] < 0.0).any()):
        tab["gini"] = float("nan")
        msg = (
            f"{func}: log {var!r} has negative values (units below 1 in levels); the "
            "Gini index is undefined on them and was set to NaN"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)
    means = tab["mean"].to_numpy(dtype=float)
    max_abs = float(np.max(np.abs(means)))
    if bool(np.min(means) <= 1e-10 * (1.0 + max_abs)):
        tab["cv"] = float("nan")
        msg = (
            f"{func}: a period mean of log {var!r} is not strictly positive; the "
            "coefficient of variation is undefined and was set to NaN"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)

    trends = {m: _dispersion_trend(tab, m, time, vcov) for m in _SIGMA_MEASURES}
    for m in _SIGMA_MEASURES:
        if trends[m][0] is None:
            notes.append(
                f"{func}: fewer than 3 periods with a positive {m}; its trend was "
                "not estimated"
            )

    fig = _sigma_fig(
        tab,
        trends,
        time,
        time_label,
        var_label,
        title if title is not None else f"σ-convergence: {var_label}",
    )
    summary, gt = _sigma_summary_and_gt(trends, var_label, n_periods, n_units)
    models = [trends[m][0] for m in _SIGMA_MEASURES if trends[m][0] is not None]

    return SigmaConvergenceResult(
        df=tab,
        fig=fig,
        gt=gt,
        summary=summary,
        models=models,
        var=var,
        entity=entity,
        time=time,
        n_periods=n_periods,
        n_units=n_units,
        std_slope=trends["std"][2],
        std_se=trends["std"][3],
        std_pvalue=trends["std"][4],
        std_r2=trends["std"][5],
        gini_slope=trends["gini"][2],
        gini_se=trends["gini"][3],
        gini_pvalue=trends["gini"][4],
        gini_r2=trends["gini"][5],
        cv_slope=trends["cv"][2],
        cv_se=trends["cv"][3],
        cv_pvalue=trends["cv"][4],
        cv_r2=trends["cv"][5],
        notes=tuple(notes),
    )
