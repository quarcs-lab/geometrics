"""Phillips-Sul convergence clubs: the log(t) test and data-driven club clustering.

:func:`analyze_convergence_clubs` runs the **Phillips-Sul (2007/2009) log(t)
club-convergence** workflow end to end: it smooths each unit's series with the
**Hodrick-Prescott filter** (``lambda = 400`` for annual data), forms the **relative
transition path** ``h_it = x_it / mean_i(x_it)``, runs the **log(t) regression test**
for the whole panel and — if global convergence is rejected — applies the data-driven
**clustering algorithm** to split the units into convergence **clubs**, then **merges**
adjacent clubs that jointly converge (the Phillips-Sul 2009 rule). It is a faithful
port of the Stata ``psecta`` package (Du 2017): the log(t) statistic uses the
Phillips-Sul scalar-long-run-variance HAC (Andrews 1991 quadratic-spectral kernel with
an AR(1) automatic bandwidth), which standard OLS engines do not provide, so that one
statistic is computed in NumPy here.

The variable is used **as supplied** (pass *log* GDP per capita / log labor
productivity — the Phillips-Sul convention); the panel must be **balanced** (the HP
filter needs a gap-free series). With entity geometry (``gdf``) the result also carries
a club-membership choropleth.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api import types as pdt
from plotly.subplots import make_subplots

from geometrics._common import entity_display_map
from geometrics._geo import resolve_gdf_entity
from geometrics._labels import resolve_label
from geometrics._mapping import categorical_map
from geometrics._panel import resolve_entity_name, resolve_panel
from geometrics._theme import CLUB_COLORS, apply_default_layout
from geometrics._types import ConvergenceClubsResult
from geometrics._validation import (
    GeometricsWarning,
    ensure_dataframe,
    ensure_geodataframe,
)

if TYPE_CHECKING:
    import geopandas as gpd

__all__ = ["analyze_convergence_clubs"]

#: One-sided 5% critical value for the log(t) convergence test.
_TCRIT = -1.65

#: Grey for the divergent (club-less) group, matching the map mask color.
_DIVERGENT_COLOR = "#d3d3d3"

#: Merge modes: Phillips-Sul (2009) merging applied iteratively, one pass, or none.
_MERGE_MODES = ("ps", "single", "none")


# ------------------------------------------------------------------------------------
# The Phillips-Sul numerical core (a line-faithful port of psecta's Mata source)
# ------------------------------------------------------------------------------------


def _sround(x: float) -> int:
    """Round half **away from zero**, matching Stata's ``round()`` (not banker's rounding).

    The log(t) trimming uses ``round(r*T)``; Python's built-in ``round`` rounds halves
    to even, which would silently shift the discarded fraction by one period at the
    half-way point.
    """
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _andrews_lrv(x: np.ndarray) -> float:
    """Long-run variance of a 1-D series via the Andrews (1991) quadratic-spectral HAC.

    A verbatim port of the Mata ``_andrs`` (itself a translation of Donggyu Sul's GAUSS
    code) used inside the Phillips-Sul log(t) test. The bandwidth is the AR(1)-based
    automatic choice ``band = 1.3221 (a2 * m)^(1/5)`` with ``a2 = 4 b1^2 / (1 - b1)^4``;
    the autocovariances run over the first ``m - 1`` terms and the variance is
    normalised by ``m - 1`` (both exactly as in the reference). Returns ``nan`` for a
    series too short or with no first-order variation.
    """
    v = np.asarray(x, dtype=float).ravel()
    m = v.size
    if m < 3:
        return float("nan")
    x1, y1 = v[:-1], v[1:]
    denom = float(np.dot(x1, x1))
    if denom <= 0.0:
        return float("nan")
    b1 = float(np.dot(x1, y1) / denom)  # AR(1) coefficient
    if b1 == 1.0:
        return float("nan")
    a2 = 4.0 * b1**2 / (1.0 - b1) ** 4
    band = 1.3221 * (a2 * m) ** 0.2
    if not math.isfinite(band) or band <= 0.0:
        return float("nan")
    t = m - 1
    j = np.arange(1, m, dtype=float)  # 1 .. m-1
    jb = j / band
    jband = jb * (1.2 * math.pi)
    # Quadratic-spectral kernel weights.
    kern = (np.sin(jband) / jband - np.cos(jband)) / ((jb * math.pi) ** 2 * 12.0) * 25.0
    lam = 0.0
    for i in range(1, t):  # i = 1 .. t-1
        c = float(np.dot(v[: t - i], v[i:t]))
        lam += 2.0 * c * kern[i - 1] / t
    sigm = float(np.dot(v, v)) / t
    return sigm + lam


def _log_t_test(mat: np.ndarray, r: float) -> tuple[float, float]:
    """Phillips-Sul log(t) convergence test on a units-by-time matrix.

    Forms the relative transition ``h_it = x_it / mean_i(x_it)`` and the
    cross-sectional variance ``H_t = mean_i (h_it - 1)^2``, then runs the regression

    ``log(H_1 / H_t) - 2 log(log t) = a + b log t + e``,    ``t = [rT] .. T``

    discarding the first ``round(r*T)`` periods. ``b = 2*alpha`` so a one-sided
    ``t_b > -1.65`` fails to reject convergence. The standard error is the Phillips-Sul
    scalar-long-run-variance HAC ``V = (X'X)^{-1} * omega`` with ``omega`` from
    :func:`_andrews_lrv` on the demeaned residuals. Returns ``(b, t_b)``; either is
    ``nan`` when the test is not estimable. Port of the Mata ``_reglogt``.
    """
    m = np.asarray(mat, dtype=float)
    n_units, big_t = m.shape
    if n_units < 1 or big_t < 2:
        return float("nan"), float("nan")
    xcm = m.mean(axis=0)  # cross-sectional mean per period
    with np.errstate(divide="ignore", invalid="ignore"):
        h = m / xcm
        h_var = np.mean((h - 1.0) ** 2, axis=0)  # H_t
        logt = np.log(np.arange(1, big_t + 1, dtype=float))
        y = np.log(h_var[0] / h_var) - 2.0 * np.log(logt)
    start = _sround(r * big_t)  # discard the first `start` periods (0-based slice)
    if big_t - start < 4:
        return float("nan"), float("nan")
    design = np.column_stack([logt[start:], np.ones(big_t - start)])
    ys = y[start:]
    if not (np.all(np.isfinite(ys)) and np.all(np.isfinite(design))):
        return float("nan"), float("nan")
    xtx = design.T @ design
    try:
        b = np.linalg.solve(xtx, design.T @ ys)
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return float("nan"), float("nan")
    resid = ys - design @ b
    resid = resid - resid.mean()
    omega = _andrews_lrv(resid)
    var0 = float(xtx_inv[0, 0]) * omega
    if not math.isfinite(var0) or var0 <= 0.0:
        return float(b[0]), float("nan")
    return float(b[0]), float(b[0] / math.sqrt(var0))


def _hp_trend(mat: np.ndarray, lamb: float) -> np.ndarray:
    """Return the Hodrick-Prescott **trend** of each row of ``mat`` (one unit per row).

    Mirrors the Stata ``pfilter ..., method(hp)`` step: the filter is applied to each
    unit's time series independently and the trend (not the cycle) is kept. Requires a
    gap-free series.
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter

    out = np.empty_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        _, trend = hpfilter(mat[i], lamb=lamb)
        out[i] = np.asarray(trend, dtype=float)
    return out


def _relative_transition(mat: np.ndarray) -> np.ndarray:
    """Return ``h_it = x_it / mean_i(x_it)`` (cross-sectional mean is 1 in every period)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return mat / mat.mean(axis=0)


def _sort_order(sub: np.ndarray, fr: float) -> np.ndarray:
    """Return indices sorting units by the cross-section criterion, **descending** (Step 1).

    ``fr == 0`` sorts by the last-period value; ``fr > 0`` sorts by the mean of the
    last ``(1 - fr)`` fraction of periods (the high-volatility option of the
    reference).
    """
    big_t = sub.shape[1]
    if fr <= 0.0:
        key = sub[:, -1]
    else:
        # Mata `_findclub` averages observation columns (trunc((1-fr)*(T-1))+2)..T of a
        # matrix whose first column is the id; here `sub` has no id column, so big_t ==
        # Mata's (T-1) and the faithful 0-based start period is trunc((1-fr)*big_t).
        p_start = math.trunc((1.0 - fr) * big_t)
        key = sub[:, p_start:].mean(axis=1)
    return np.argsort(-key, kind="stable")


def _find_one_club(
    sub: np.ndarray,
    ids: np.ndarray,
    r: float,
    tcrit: float,
    cr: float,
    incr: float,
    max_cr: float,
    fr: float,
    adjust: bool,
) -> list[int]:
    """Find the highest-ranked convergence club in a subgroup (Steps 1-3 of Phillips-Sul).

    ``sub`` is the units-by-time matrix of the subgroup and ``ids`` the units'
    positional ids in the full panel. Returns the member ids (possibly empty when no
    club exists). Port of the Mata ``_findclub``: cross-section sort, core-group
    formation by maximum t-statistic, then a sieve of the complement — either the
    original PS-2007 ``cr``-increment rule or the Schnurbus et al. (2016) ``adjust``
    refinement.
    """
    n_units = sub.shape[0]
    if n_units < 2:
        return []
    order = _sort_order(sub, fr)
    s = sub[order]
    sid = ids[order]

    # Step 2.1 - first successive pair (k, k+1) whose log(t) t-stat exceeds tcrit.
    tt = -100.0
    core_start = 0
    found = False
    while core_start < n_units - 1:
        _, tt = _log_t_test(s[core_start : core_start + 2], r)
        if math.isfinite(tt) and tt > tcrit:
            found = True
            break
        if not math.isfinite(tt):  # `.` in Mata stops the search (treated as failure)
            break
        core_start += 1
    if not found:
        return []

    # Step 2.2 - extend the core upward, keeping the prefix with the maximum t-stat.
    ts_by_end: dict[int, float] = {}
    end = core_start + 1
    last_tt = tt
    while end <= n_units - 1 and last_tt > tcrit:
        _, last_tt = _log_t_test(s[core_start : end + 1], r)
        if not math.isfinite(last_tt):
            break
        ts_by_end[end] = last_tt
        end += 1
    core_end = (
        max(ts_by_end, key=lambda e: ts_by_end[e]) if ts_by_end else core_start + 1
    )
    core_pos = list(range(core_start, core_end + 1))
    core_set = set(core_pos)

    # Step 3.1/3.2 - sieve the complement, adding each unit whose core+unit t-stat
    # exceeds cr.
    complement = [p for p in range(n_units) if p not in core_set]

    def club_tstat(positions: list[int]) -> float:
        _, t = _log_t_test(s[np.array(positions)], r)
        return t

    club_pos = list(core_pos)
    for p in complement:
        t = club_tstat([*core_pos, p])
        if math.isfinite(t) and t > cr:
            club_pos.append(p)

    # Step 3.3 - if the assembled club fails the joint test, refine it.
    club_t = club_tstat(club_pos)
    only_core = len(club_pos) == len(core_pos)
    if math.isfinite(club_t) and club_t <= tcrit and not only_core:
        if not adjust:  # original PS-2007: raise cr until the club converges
            cur_cr = cr
            while (not math.isfinite(club_t) or club_t <= tcrit) and cur_cr < max_cr:
                cur_cr += incr
                club_pos = list(core_pos)
                for p in complement:
                    t = club_tstat([*core_pos, p])
                    if math.isfinite(t) and t > cur_cr:
                        club_pos.append(p)
                club_t = club_tstat(club_pos)
            if not math.isfinite(club_t) or club_t <= tcrit:
                club_pos = list(core_pos)
        else:  # Schnurbus et al. (2016): add the best candidate one at a time
            candidates = [p for p in club_pos if p not in core_set]
            club_pos = list(core_pos)
            remaining = list(candidates)
            while remaining:
                # Score each candidate by the t-stat of the *growing* club plus that
                # candidate (not the core alone), so the stopping rule sees the club
                # degrade as members accumulate; stop before an addition would push it
                # below tcrit.
                scored = [(club_tstat([*club_pos, p]), p) for p in remaining]
                best_t, best_p = max(scored, key=lambda it: it[0])
                if not math.isfinite(best_t) or best_t <= tcrit:
                    break
                club_pos.append(best_p)
                remaining = [p for p in candidates if p not in set(club_pos)]
            # Never return a club that fails its own joint test (as the PS-2007 branch
            # does).
            final_t = club_tstat(club_pos)
            if not math.isfinite(final_t) or final_t <= tcrit:
                club_pos = list(core_pos)

    return [int(sid[p]) for p in club_pos]


def _get_clusters(
    mat: np.ndarray,
    r: float,
    tcrit: float,
    cr: float,
    incr: float,
    max_cr: float,
    fr: float,
    adjust: bool,
) -> dict[int, int]:
    """Recursively partition the panel into convergence clubs (Phillips-Sul Step 4).

    Returns a ``{unit_id: club}`` mapping with clubs numbered ``1..K`` from the
    highest-ranked group down; units left unassigned form the (divergent) residual
    group. Called only after the whole-panel log(t) test has rejected global
    convergence. Port of the Mata ``_getcluster``.
    """
    remaining = list(range(mat.shape[0]))
    club_of: dict[int, int] = {}
    club = 0
    while True:
        sub_ids = np.array(remaining)
        members = _find_one_club(
            mat[sub_ids], sub_ids, r, tcrit, cr, incr, max_cr, fr, adjust
        )
        if not members:
            break  # the remaining units do not form a further club (divergent)
        club += 1
        for cid in members:
            club_of[cid] = club
        member_set = set(members)
        remaining = [i for i in remaining if i not in member_set]
        if len(remaining) < 2:
            break
        # Does the whole remainder converge as a single final club?
        _, tt = _log_t_test(mat[np.array(remaining)], r)
        if math.isfinite(tt) and tt > tcrit:
            club += 1
            for cid in remaining:
                club_of[cid] = club
            break
    return club_of


def _merge_once(
    mat: np.ndarray, club_of: dict[int, int], r: float, tcrit: float
) -> tuple[dict[int, int], bool]:
    """One adjacent-club merging pass (Phillips-Sul 2009). Port of Stata's ``icheckmerge``.

    Walks the clubs in rank order, absorbing club ``k+1`` into the running merged
    block when their joint log(t) test converges, else starting a new block. Returns
    the relabelled ``{unit_id: club}`` and whether any merge happened.
    """
    members_by_club: dict[int, list[int]] = {}
    for cid, c in club_of.items():
        members_by_club.setdefault(c, []).append(cid)
    clubs = sorted(members_by_club)
    n_clubs = len(clubs)
    new_label = {clubs[0]: 1}
    running = list(members_by_club[clubs[0]])
    j = 1
    for k in range(1, n_clubs):
        cand = members_by_club[clubs[k]]
        _, tt = _log_t_test(mat[np.array(running + cand)], r)
        if math.isfinite(tt) and tt > tcrit:
            new_label[clubs[k]] = j
            running = running + cand
        else:
            j += 1
            new_label[clubs[k]] = j
            running = list(cand)
    return {cid: new_label[c] for cid, c in club_of.items()}, j < n_clubs


def _merge_clubs(
    mat: np.ndarray, club_of: dict[int, int], r: float, tcrit: float, mode: str
) -> dict[int, int]:
    """Merge adjacent clubs per ``mode`` (``"ps"`` iterative / ``"single"`` / ``"none"``)."""
    if mode == "none" or len(set(club_of.values())) < 2:
        return club_of
    if mode == "single":
        return _merge_once(mat, club_of, r, tcrit)[0]
    cur = club_of  # "ps": iterate until no pass merges (each pass drops >= 1 club)
    for _ in range(len(set(club_of.values()))):
        cur, merged = _merge_once(mat, cur, r, tcrit)
        if not merged:
            break
    return cur


# ------------------------------------------------------------------------------------
# Presentation helpers
# ------------------------------------------------------------------------------------


def _club_color(club: int) -> str:
    """Theme color for a club label (1-based); the divergent group (0) renders grey."""
    if club == 0:
        return _DIVERGENT_COLOR
    return CLUB_COLORS[(club - 1) % len(CLUB_COLORS)]


def _club_name(club: int) -> str:
    """Human label for a club number (0 is the non-converging 'Divergent' group)."""
    return "Divergent" if club == 0 else f"Club {club}"


def _clubs_long_frame(
    entities: np.ndarray,
    times: np.ndarray,
    trend: np.ndarray,
    relative: np.ndarray,
    club_of: dict[int, int],
    entity: str,
    time: str,
) -> pd.DataFrame:
    """Tidy long frame: one row per (unit, period) with ``value``, ``relative``, ``club``."""
    n_units, n_t = trend.shape
    rows = {
        entity: np.repeat(entities, n_t),
        time: np.tile(times, n_units),
        "value": trend.reshape(-1),
        "relative": relative.reshape(-1),
        "club": np.repeat([club_of.get(i, 0) for i in range(n_units)], n_t),
    }
    return pd.DataFrame(rows)


def _clubs_avg_fig(
    long: pd.DataFrame,
    entity: str,
    time: str,
    time_label: str,
    var_label: str,
    title: str | None,
) -> go.Figure:
    """Within-club **average** relative-transition paths (the headline figure)."""
    fig = go.Figure()
    for club in sorted(long["club"].unique()):
        sub = long[long["club"] == club]
        avg = sub.groupby(time, observed=True)["relative"].mean().sort_index()
        n_members = sub[entity].nunique()
        fig.add_trace(
            go.Scatter(
                x=avg.index.to_numpy(dtype=float),
                y=avg.to_numpy(dtype=float),
                mode="lines+markers",
                name=f"{_club_name(int(club))} (n={n_members})",
                line={
                    "color": _club_color(int(club)),
                    "width": 2.5,
                    "dash": "dot" if club == 0 else "solid",
                },
                marker={"color": _club_color(int(club)), "size": 6},
                hovertemplate=(
                    f"{_club_name(int(club))}<br>{time_label}=%{{x}}<br>"
                    "relative=%{y:.3f}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(0,0,0,0.4)")
    apply_default_layout(
        fig,
        title=title if title is not None else f"Convergence clubs: {var_label}",
        xaxis={"title": time_label},
        yaxis={"title": f"Relative {var_label} (cross-sectional mean = 1)"},
    )
    return fig


def _clubs_paths_fig(
    long: pd.DataFrame,
    entity: str,
    time: str,
    time_label: str,
    var_label: str,
    disp: dict[str, str] | None = None,
) -> go.Figure:
    """All units' relative-transition paths, colored by club (one legend entry per club)."""
    disp = disp or {}
    fig = go.Figure()
    seen: set[int] = set()
    for ent, sub in long.groupby(entity, observed=True, sort=False):
        sub = sub.sort_values(time)
        club = int(sub["club"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=sub[time].to_numpy(dtype=float),
                y=sub["relative"].to_numpy(dtype=float),
                mode="lines",
                line={"color": _club_color(club), "width": 1},
                opacity=0.55,
                legendgroup=_club_name(club),
                name=_club_name(club),
                showlegend=club not in seen,
                customdata=np.full(len(sub), disp.get(str(ent), str(ent))),
                hovertemplate=(
                    f"%{{customdata}} ({_club_name(club)})<br>{time_label}=%{{x}}<br>"
                    "relative=%{y:.3f}<extra></extra>"
                ),
            )
        )
        seen.add(club)
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(0,0,0,0.4)")
    apply_default_layout(
        fig,
        title=f"Relative transition paths by club: {var_label}",
        xaxis={"title": time_label},
        yaxis={"title": f"Relative {var_label} (cross-sectional mean = 1)"},
    )
    return fig


def _clubs_facets_fig(
    long: pd.DataFrame,
    entity: str,
    time: str,
    time_label: str,
    var_label: str,
) -> go.Figure:
    """Small-multiple panels (one per club) of member paths with the club mean overlaid."""
    clubs = sorted(int(c) for c in long["club"].unique())
    n = len(clubs)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    titles = []
    for c in clubs:
        members = long[long["club"] == c][entity].nunique()
        titles.append(f"{_club_name(c)} (n={members})")
    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=titles, shared_yaxes=True
    )
    for idx, club in enumerate(clubs):
        row, col = idx // ncols + 1, idx % ncols + 1
        sub = long[long["club"] == club]
        for _ent, g in sub.groupby(entity, observed=True, sort=False):
            g = g.sort_values(time)
            fig.add_trace(
                go.Scatter(
                    x=g[time].to_numpy(dtype=float),
                    y=g["relative"].to_numpy(dtype=float),
                    mode="lines",
                    line={"color": _club_color(club), "width": 0.8},
                    opacity=0.4,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        avg = sub.groupby(time, observed=True)["relative"].mean().sort_index()
        fig.add_trace(
            go.Scatter(
                x=avg.index.to_numpy(dtype=float),
                y=avg.to_numpy(dtype=float),
                mode="lines",
                line={"color": _club_color(club), "width": 2.5},
                showlegend=False,
                hovertemplate="relative=%{y:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
    apply_default_layout(
        fig, title=f"Convergence clubs ({var_label}): member paths by club"
    )
    fig.update_xaxes(title_text=time_label, row=nrows)
    return fig


def _clubs_summary_and_gt(
    club_of: dict[int, int],
    club_stats: dict[int, tuple[float, float, int]],
    entities: np.ndarray,
    var_label: str,
    n_units: int,
    n_periods: int,
    tcrit: float,
) -> tuple[pd.DataFrame, Any, pd.DataFrame]:
    """Build the per-club ``summary`` frame, its GT rendering, and the membership frame.

    ``club_stats`` maps a club number to ``(beta, tstat, n_members)``; club ``0`` (if
    present) is the divergent residual group, listed last.
    """
    from great_tables import GT

    members_by_club: dict[int, list[str]] = {}
    for cid, c in sorted(club_of.items()):
        members_by_club.setdefault(c, []).append(str(entities[cid]))
    for cid in range(n_units):  # never-assigned units are the divergent group (club 0)
        if cid not in club_of:
            members_by_club.setdefault(0, []).append(str(entities[cid]))

    order = [c for c in sorted(members_by_club) if c != 0] + (
        [0] if 0 in members_by_club else []
    )

    def member_str(names: list[str], limit: int = 8) -> str:
        names = sorted(names)
        if len(names) <= limit:
            return ", ".join(names)
        return ", ".join(names[:limit]) + f", ... (+{len(names) - limit})"

    nan3 = (float("nan"), float("nan"), 0)
    summary = pd.DataFrame(
        {
            "club": [_club_name(c) for c in order],
            "n_members": [len(members_by_club[c]) for c in order],
            "beta": [club_stats.get(c, nan3)[0] for c in order],
            "tstat": [club_stats.get(c, nan3)[1] for c in order],
            "converging": [
                bool(club_stats.get(c, nan3)[1] > tcrit) if c != 0 else False
                for c in order
            ],
            "members": [member_str(members_by_club[c]) for c in order],
        }
    )

    def fmt(value: float) -> str:
        return "—" if not math.isfinite(value) else f"{value:.3f}"

    disp = pd.DataFrame(
        {
            "Group": summary["club"],
            "N": summary["n_members"],
            "log(t) b": [fmt(v) for v in summary["beta"]],
            "t-stat": [fmt(v) for v in summary["tstat"]],
            "Converges": [
                "—" if g == "Divergent" else ("yes" if c else "no")
                for g, c in zip(summary["club"], summary["converging"], strict=True)
            ],
            "Members": summary["members"],
        }
    )
    gt = (
        GT(disp, rowname_col="Group")
        .tab_header(
            title=f"Convergence clubs: {var_label}",
            subtitle=(
                f"Phillips-Sul log(t) clustering over {n_periods} periods, "
                f"{n_units} units"
            ),
        )
        .tab_source_note(
            f"Each club's log(t) t-stat exceeds {tcrit:g} (the convergence "
            "threshold); b = 2*alpha is the within-club convergence speed. The "
            "Divergent group does not form a convergence club."
        )
    )

    membership = pd.DataFrame(
        {
            "entity": [str(entities[i]) for i in range(n_units)],
            "club": [club_of.get(i, 0) for i in range(n_units)],
        }
    )
    membership["club_label"] = membership["club"].map(_club_name)
    membership = membership.sort_values(["club", "entity"]).reset_index(drop=True)
    return summary, gt, membership


def _clubs_map(
    gdf: gpd.GeoDataFrame,
    membership: pd.DataFrame,
    var_label: str,
    tiles: str | None,
    ent_disp: dict[str, str],
    func: str,
    notes: list[str],
) -> go.Figure:
    """Build the club-membership categorical choropleth (fixed club colors)."""
    gdf = ensure_geodataframe(gdf, func=func)
    gdf_entity = resolve_gdf_entity(gdf)
    club_by_id = dict(
        zip(membership["entity"], membership["club"].astype(int), strict=True)
    )
    keys = gdf[gdf_entity].astype(str)
    matched = keys.isin(club_by_id)
    if not bool(matched.any()):
        raise ValueError(
            f"{func}: the panel entities and gdf.{gdf_entity} share no ids — pass a "
            "gdf keyed by the same entity ids"
        )
    n_unmatched_gdf = int((~matched).sum())
    n_unmatched_df = len(set(club_by_id) - set(keys))
    if n_unmatched_gdf or n_unmatched_df:
        msg = (
            f"{func}: club map covers the matched units only — {n_unmatched_gdf} gdf "
            f"unit(s) without club data and {n_unmatched_df} panel unit(s) without "
            "geometry were left off the map"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=3)
        notes.append(msg)
    sub = gdf.loc[matched]
    labels = [_club_name(club_by_id[str(i)]) for i in sub[gdf_entity]]
    clubs_present = sorted({c for c in club_by_id.values() if c != 0})
    category_order = [_club_name(c) for c in clubs_present]
    if 0 in set(club_by_id.values()):
        category_order.append(_club_name(0))
    colors = {_club_name(c): _club_color(c) for c in [*clubs_present, 0]}
    return categorical_map(
        sub,
        labels,
        entity=gdf_entity,
        colors=colors,
        category_order=category_order,
        tiles=tiles,
        title=f"Convergence-club membership: {var_label}",
        hover_names=ent_disp,
    )


# ------------------------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------------------------


def analyze_convergence_clubs(
    df: pd.DataFrame,
    var: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    gdf: gpd.GeoDataFrame | None = None,
    hp_filter: bool = True,
    hp_lambda: float = 400.0,
    trim: float = 0.3,
    tcrit: float = _TCRIT,
    cr: float = 0.0,
    increment: float = 0.05,
    max_cr: float = 3.0,
    fraction: float = 0.0,
    adjust: bool = False,
    merge: str = "ps",
    tiles: str | None = "carto-positron",
    title: str | None = None,
) -> ConvergenceClubsResult:
    r"""Phillips-Sul log(t) convergence test and data-driven club clustering for a panel.

    Runs the full club-convergence workflow on one variable: optionally smooth each
    unit's series with the **Hodrick-Prescott filter** (``lambda = 400`` for annual
    data); form the **relative transition path** ``h_it = x_it / mean_i(x_it)``; run
    the **log(t) regression test** for the whole panel; and, when global convergence is
    rejected, apply the **clustering algorithm** to split the units into convergence
    **clubs**, then **merge** adjacent clubs that jointly converge. This answers the
    descriptive question "do these units form one converging group, several catch-up
    clubs, or none?".

    The variable is used **as supplied** — no log is taken — so for the canonical
    income case pass *log* GDP per capita (or log labor productivity). The panel must
    be **balanced** (every unit present in every period) because the HP filter needs a
    gap-free series.

    Parameters
    ----------
    df
        Balanced panel data frame.
    var
        Numeric variable to analyse (e.g. ``"log_gdppc"``). Used as supplied.
    entity, time
        Panel identifiers. Default to those declared via :func:`geometrics.set_panel`.
    gdf
        Optional entity geometry; when given, the result carries a club-membership
        choropleth ``fig_map`` (``None`` otherwise).
    hp_filter
        Apply the Hodrick-Prescott filter per unit and analyse the **trend**
        (default). ``False`` analyses the variable as given (already smooth).
    hp_lambda
        HP smoothing parameter (``400`` for annual data, the convergence-literature
        default).
    trim
        Initiating sample fraction ``r`` of the log(t) regression: the first
        ``round(r*T)`` periods are discarded. Phillips-Sul recommend ``0.3`` for
        small/moderate ``T`` and ``0.2`` for large ``T``.
    tcrit
        One-sided convergence critical value for the t-statistic (``-1.65``, the 5%
        level).
    cr
        Sieve inclusion threshold ``c*`` for adding units to a core group.
    increment
        Increment by which ``cr`` is raised (original PS-2007 refinement rule) when
        the assembled club fails its joint test.
    max_cr
        Ceiling for the raised ``cr``.
    fraction
        Cross-section sort key: ``0`` (default) sorts by the last period; ``> 0``
        sorts by the mean of the last ``(1 - fraction)`` share of periods (for noisy
        endpoints).
    adjust
        Use the Schnurbus et al. (2016) club refinement (add the best candidate one at
        a time) instead of the original Phillips-Sul ``cr``-increment rule.
    merge
        Adjacent-club merging after clustering: ``"ps"`` (default) applies the
        Phillips-Sul (2009) merge test iteratively until no clubs merge, ``"single"``
        does one pass, ``"none"`` reports the raw clusters.
    tiles
        MapLibre basemap style for ``fig_map`` (``None`` draws the vector backend).
    title
        Title for the headline figure.

    Returns
    -------
    ConvergenceClubsResult
        The tidy long ``df`` (``entity``, ``time``, ``value`` = HP trend, ``relative``
        = ``h_it``, ``club`` with ``0`` = divergent); the within-club average figure
        ``fig``; the all-paths figure ``fig_paths``; the per-club small-multiples
        ``fig_clubs``; the membership choropleth ``fig_map`` (``None`` without
        ``gdf``); the classification table ``gt`` / ``summary`` and the ``membership``
        frame; the whole-panel ``global_beta`` / ``global_tstat`` and ``converged``
        flag; and the club counts and run parameters.

    Raises
    ------
    KeyError
        If ``var`` is not a column of ``df``.
    TypeError
        If ``var`` is not numeric.
    ValueError
        If ``trim`` is out of ``(0, 1)``, ``merge`` is unknown, the panel is
        unbalanced or too short/small, the per-period cross-sectional mean is (near)
        zero, or the global log(t) test is not estimable.

    Notes
    -----
    The log(t) test regresses, for :math:`t = [rT] \ldots T`,

    .. math:: \log(H_1 / H_t) - 2 \log(\log t) = a + b \log t + \varepsilon_t,

    where :math:`H_t = N^{-1} \sum_i (h_{it} - 1)^2` is the cross-sectional variance of
    the relative transition paths. Under the null of convergence ``b = 2*alpha >= 0``;
    a one-sided ``t_b > -1.65`` fails to reject it. The standard error is the
    Phillips-Sul scalar long-run variance form with an Andrews (1991)
    quadratic-spectral HAC of the residuals. The clustering sorts units by their final
    value, forms a core group by maximising ``t_b``, sieves in the remaining units,
    and recurses on the residual; adjacent clubs are then merged when they jointly
    converge. This is a faithful port of the Stata ``psecta`` package (Du 2017); see
    Phillips & Sul (2007, 2009) and Schnurbus et al. (2016).

    Examples
    --------
    Two planted clubs (units converge within their group, not across groups):

    ```python
    import numpy as np
    import pandas as pd

    from geometrics.clubs import analyze_convergence_clubs

    rng = np.random.default_rng(0)
    rows = []
    for k, mu in enumerate((10.0, 8.5), start=1):
        for j in range(10):
            dev = rng.uniform(-0.4, 0.4)
            for t in range(1, 31):
                rows.append((f"c{k}u{j}", t, mu + dev * 0.9 ** (t - 1)))
    df = pd.DataFrame(rows, columns=["unit", "year", "log_y"])
    res = analyze_convergence_clubs(df, "log_y", entity="unit", time="year")
    res.n_clubs, res.converged
    ```
    """
    df = ensure_dataframe(df)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    assert entity is not None and time is not None  # guaranteed by require_* above
    func = "analyze_convergence_clubs"

    if var not in df.columns:
        raise KeyError(f"{func}: column {var!r} not found in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"{func}: {var!r} needs to be numeric")
    if not 0.0 < trim < 1.0:
        raise ValueError(f"{func}: trim must be in (0, 1); got {trim}")
    if merge not in _MERGE_MODES:
        raise ValueError(
            f"{func}: unknown merge mode {merge!r}; choose from {list(_MERGE_MODES)}"
        )

    var_label = resolve_label(df, var)
    time_label = resolve_label(df, time)
    # Entity-name display map ("Name (id)"), built before the slice/groupby drops attrs.
    ent_disp = entity_display_map(df, entity, resolve_entity_name(df))
    notes: list[str] = []

    work = df[[entity, time, var]].copy()
    work[time] = pd.to_numeric(work[time], errors="coerce")
    work = work.dropna(subset=[time, var])
    if work.empty:
        raise ValueError(
            f"{func}: no rows with both a numeric {time!r} and a non-missing {var!r}"
        )

    before = len(work)
    work = work.groupby([entity, time], observed=True, as_index=False).first()
    if len(work) < before:
        msg = (
            f"{func}: found duplicate ({entity!r}, {time!r}) rows; kept the first of "
            f"each ({before - len(work)} dropped)"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=2)
        notes.append(msg)

    wide = work.pivot(index=entity, columns=time, values=var).sort_index(axis=1)
    if bool(wide.isna().to_numpy().any()):
        n_missing = int(wide.isna().to_numpy().sum())
        raise ValueError(
            f"{func}: panel is not balanced: {n_missing} (entity, time) cells are "
            "missing. Club convergence needs a gap-free series per unit (the HP "
            "filter cannot span gaps); restrict to a balanced window or drop the "
            "offending units."
        )
    entities = wide.index.to_numpy()
    times = wide.columns.to_numpy(dtype=float)
    n_units, n_periods = wide.shape
    if n_units < 2:
        raise ValueError(
            f"{func}: need >= 2 units to form convergence clubs; got {n_units}"
        )
    if n_periods - _sround(trim * n_periods) < 4:
        raise ValueError(
            f"{func}: too few periods ({n_periods}) for a log(t) test trimmed at "
            f"trim={trim:g}: only {n_periods - _sround(trim * n_periods)} remain "
            f"after discarding the first {_sround(trim * n_periods)}; need >= 4. "
            "Use more periods or a smaller trim."
        )

    raw = wide.to_numpy(dtype=float)
    trend = _hp_trend(raw, hp_lambda) if hp_filter else raw

    # The relative transition divides by the per-period cross-sectional mean, so a
    # mean at or near zero (e.g. a demeaned / centered / growth variable) blows it up
    # to inf and silently corrupts every downstream frame and figure. Phillips-Sul
    # expects a strictly-positive variable (levels or log-levels); reject a
    # (near-)zero or sign-changing mean up front.
    xcm = trend.mean(axis=0)
    scale = float(np.nanmax(np.abs(trend))) if trend.size else 0.0
    if not np.all(np.isfinite(xcm)) or bool(
        np.any(np.abs(xcm) <= 1e-9 * (1.0 + scale))
    ):
        raise ValueError(
            f"{func}: the per-period cross-sectional mean of {var!r} is at or near "
            "zero in some period, so the relative transition h_it = x_it / "
            "mean_i(x_it) is undefined. Club convergence expects a strictly-positive "
            "variable (levels or log-levels), not a demeaned/centered or "
            "sign-changing series."
        )
    relative = _relative_transition(trend)

    global_beta, global_tstat = _log_t_test(trend, trim)
    if not math.isfinite(global_tstat):
        raise ValueError(
            f"{func}: the log(t) convergence test for {var!r} is not estimable: the "
            "cross-sectional dispersion of the relative transitions is (near) zero "
            "in every period — the units are already identical (trivially "
            "converged), so the test statistic is undefined."
        )
    converged = bool(global_tstat > tcrit)

    if converged:
        club_of = {i: 1 for i in range(n_units)}
    else:
        club_of = _get_clusters(
            trend, trim, tcrit, cr, increment, max_cr, fraction, adjust
        )
        club_of = _merge_clubs(trend, club_of, trim, tcrit, merge)

    # Per-club log(t) statistics (and the divergent group, if any has >= 2 members).
    club_stats: dict[int, tuple[float, float, int]] = {}
    members_by_club: dict[int, list[int]] = {}
    for cid, c in club_of.items():
        members_by_club.setdefault(c, []).append(cid)
    divergent_ids = [i for i in range(n_units) if i not in club_of]
    n_clubs = len(members_by_club)
    for c, ids in members_by_club.items():
        b, t = _log_t_test(trend[np.array(ids)], trim)
        club_stats[c] = (b, t, len(ids))
    if len(divergent_ids) >= 2:
        b, t = _log_t_test(trend[np.array(divergent_ids)], trim)
        club_stats[0] = (b, t, len(divergent_ids))
    elif len(divergent_ids) == 1:
        club_stats[0] = (float("nan"), float("nan"), 1)

    long = _clubs_long_frame(entities, times, trend, relative, club_of, entity, time)
    summary, gt, membership = _clubs_summary_and_gt(
        club_of, club_stats, entities, var_label, n_units, n_periods, tcrit
    )

    fig = _clubs_avg_fig(long, entity, time, time_label, var_label, title)
    fig_paths = _clubs_paths_fig(long, entity, time, time_label, var_label, ent_disp)
    fig_clubs = _clubs_facets_fig(long, entity, time, time_label, var_label)
    fig_map: go.Figure | None = None
    if gdf is not None:
        fig_map = _clubs_map(gdf, membership, var_label, tiles, ent_disp, func, notes)

    if converged:
        notes.append(
            f"{func}: the whole panel converges (global log(t) t-stat > {tcrit:g}); "
            "it forms a single club"
        )
    elif n_clubs == 0:
        notes.append(
            f"{func}: global convergence is rejected and no convergence clubs were "
            "found; all units diverge"
        )

    return ConvergenceClubsResult(
        df=long,
        fig=fig,
        fig_paths=fig_paths,
        fig_clubs=fig_clubs,
        fig_map=fig_map,
        gt=gt,
        summary=summary,
        membership=membership,
        var=var,
        entity=entity,
        time=time,
        n_units=n_units,
        n_periods=n_periods,
        n_clubs=n_clubs,
        n_divergent=len(divergent_ids),
        global_beta=global_beta,
        global_tstat=global_tstat,
        converged=converged,
        hp_lambda=float(hp_lambda) if hp_filter else float("nan"),
        trim=float(trim),
        tcrit=float(tcrit),
        method="adjust" if adjust else "ps",
        merge=merge,
        notes=tuple(notes),
    )
