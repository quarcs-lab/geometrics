"""Tests for the Phillips-Sul club-convergence port (``analyze_convergence_clubs``)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from geometrics._theme import CLUB_COLORS
from geometrics._types import ConvergenceClubsResult
from geometrics._validation import GeometricsWarning
from geometrics.clubs import (
    _andrews_lrv,
    _log_t_test,
    _sround,
    analyze_convergence_clubs,
)
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

# ---------------------------------------------------------------------------
# Verification backbone: a panel with a *planted* two-club structure. Each unit in
# club k sits at a distinct long-run level plus an idiosyncratic deviation that decays
# geometrically, so units converge within their club while the distinct levels keep
# the whole panel from converging. Unit ids reuse the lattice ids (u00..u63) so the
# club map can be drawn on grid_gdf.
# ---------------------------------------------------------------------------

N_YEARS = 35
LEVELS = (10.0, 8.6)  # club 1 (high) and club 2 (low) long-run levels


def _two_club_panel(
    ids: list[str], seed: int = 1
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Panel with two planted clubs on the given unit ids; returns ``(df, truth)``."""
    rng = np.random.default_rng(seed)
    half = len(ids) // 2
    rows: list[tuple[str, int, float]] = []
    truth: dict[str, int] = {}
    for j, uid in enumerate(ids):
        club = 1 if j < half else 2
        truth[uid] = club
        mu = LEVELS[club - 1]
        dev = float(rng.uniform(-0.4, 0.4))
        for t in range(1, N_YEARS + 1):
            rows.append(
                (uid, t, mu + dev * 0.9 ** (t - 1) + float(rng.normal(0, 0.002)))
            )
    return pd.DataFrame(rows, columns=["unit", "year", "x"]), truth


def _single_club_panel(n: int = 40, seed: int = 2) -> pd.DataFrame:
    """One converging group: all units approach a common level."""
    df, _ = _two_club_panel([f"s{i:02d}" for i in range(n)], seed=seed)
    df["x"] = df["x"] - df["unit"].map(
        {f"s{i:02d}": (LEVELS[1] - LEVELS[0] if i >= n // 2 else 0.0) for i in range(n)}
    )
    return df


def _recovery_accuracy(membership: pd.DataFrame, truth: dict[str, int]) -> float:
    """Best-match fraction of units placed in their planted club (label-invariant)."""
    detected = dict(zip(membership["entity"], membership["club"], strict=True))
    by_detected: dict[int, list[int]] = {}
    for uid, det in detected.items():
        by_detected.setdefault(int(det), []).append(truth[uid])
    correct = 0
    for det, trues in by_detected.items():
        if det == 0:
            continue
        modal = max(set(trues), key=trues.count)
        correct += sum(1 for tc in trues if tc == modal)
    return correct / len(truth)


@pytest.fixture(scope="module")
def two_club_panel(grid_gdf):
    return _two_club_panel(list(grid_gdf["unit"]))


# ---------------------------------------------------------------------------
# Pure-helper unit tests (the psecta-faithful numerical core)
# ---------------------------------------------------------------------------


def test_sround_rounds_half_away_from_zero():
    assert _sround(7.5) == 8
    assert _sround(2.5) == 3  # Python round(2.5) == 2 (banker's rounding)
    assert _sround(-2.5) == -3
    assert _sround(2.4) == 2


def test_andrews_lrv_recovers_white_noise_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 2.0, size=600)  # iid -> long-run variance == variance == 4
    assert _andrews_lrv(x) == pytest.approx(4.0, rel=0.2)
    assert math.isnan(_andrews_lrv(np.array([1.0, 2.0])))  # too short


def test_log_t_test_matches_independent_ols_slope():
    rng = np.random.default_rng(3)
    mat = 10.0 + np.cumsum(rng.normal(0.0, 0.05, size=(6, 30)), axis=1)
    b, t = _log_t_test(mat, 0.3)
    big_t = mat.shape[1]
    h_var = np.mean((mat / mat.mean(0) - 1.0) ** 2, axis=0)
    logt = np.log(np.arange(1, big_t + 1))
    with np.errstate(divide="ignore"):
        y = np.log(h_var[0] / h_var) - 2.0 * np.log(logt)
    start = _sround(0.3 * big_t)
    design = np.column_stack([logt[start:], np.ones(big_t - start)])
    b_ref = float(np.linalg.lstsq(design, y[start:], rcond=None)[0][0])
    assert b == pytest.approx(b_ref, abs=1e-9)
    assert math.isfinite(t)


def test_log_t_test_flags_convergence_and_divergence():
    # Vanishing deviations -> converges (t > -1.65); distinct levels -> diverges.
    conv = np.array(
        [
            [10.0 + (0.5 if i % 2 else -0.5) * 0.9**t for t in range(35)]
            for i in range(20)
        ]
    )
    assert _log_t_test(conv, 0.3)[1] > -1.65
    rng = np.random.default_rng(1)
    div = np.array([[8.0 if i < 10 else 12.0] * 35 for i in range(20)], dtype=float)
    div = div + rng.normal(0.0, 1e-3, div.shape)
    assert _log_t_test(div, 0.3)[1] <= -1.65


# ---------------------------------------------------------------------------
# Mathematical validity: recover the planted clubs
# ---------------------------------------------------------------------------


def test_recovers_two_planted_clubs(two_club_panel):
    df, truth = two_club_panel
    res = analyze_convergence_clubs(df, "x", entity="unit", time="year")
    assert isinstance(res, ConvergenceClubsResult)
    assert res.converged is False  # distinct levels => global convergence rejected
    assert res.global_tstat <= res.tcrit
    assert res.n_clubs == 2
    assert _recovery_accuracy(res.membership, truth) >= 0.90
    # every detected club passes its own convergence test
    clubs = res.summary[res.summary["club"] != "Divergent"]
    assert bool(clubs["converging"].all())
    assert res.n_divergent + int(res.summary["n_members"].sum()) >= 64


def test_whole_panel_converges_to_single_club():
    res = analyze_convergence_clubs(
        _single_club_panel(), "x", entity="unit", time="year"
    )
    assert res.converged is True
    assert res.n_clubs == 1
    assert res.n_divergent == 0
    assert (res.membership["club"] == 1).all()
    assert any("single club" in n for n in res.notes)


def test_merging_does_not_increase_club_count(two_club_panel):
    df, _ = two_club_panel
    counts = {
        mode: analyze_convergence_clubs(
            df, "x", entity="unit", time="year", merge=mode
        ).n_clubs
        for mode in ("none", "single", "ps")
    }
    assert counts["ps"] <= counts["single"] <= counts["none"]


def test_adjust_refinement_runs(two_club_panel):
    df, truth = two_club_panel
    res = analyze_convergence_clubs(df, "x", entity="unit", time="year", adjust=True)
    assert res.method == "adjust"
    assert res.n_clubs >= 2
    assert _recovery_accuracy(res.membership, truth) >= 0.90


# ---------------------------------------------------------------------------
# Result surface
# ---------------------------------------------------------------------------


def test_clubs_result_surface(two_club_panel):
    df, _ = two_club_panel
    res = analyze_convergence_clubs(df, "x", entity="unit", time="year")
    assert list(res.df.columns) == ["unit", "year", "value", "relative", "club"]
    assert len(res.df) == 64 * N_YEARS
    assert res.n_units == 64
    assert res.n_periods == N_YEARS
    assert list(res.membership.columns) == ["entity", "club", "club_label"]
    assert res.method == "ps"
    assert res.merge == "ps"
    assert res.hp_lambda == 400.0
    assert res.trim == 0.3
    assert res.tcrit == -1.65
    assert isinstance(res.fig, go.Figure)
    assert isinstance(res.fig_paths, go.Figure)
    assert isinstance(res.fig_clubs, go.Figure)
    assert res.fig_map is None  # no gdf supplied
    assert type(res.gt).__name__ == "GT"
    assert res.tidy() is res.summary
    # relative transition paths average to 1 in every period
    per_period = res.df.groupby("year")["relative"].mean()
    assert np.allclose(per_period, 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.n_clubs = 0  # type: ignore[misc]


def test_clubs_fig_map_categorical(two_club_panel, grid_gdf):
    df, _ = two_club_panel
    res = analyze_convergence_clubs(
        df, "x", entity="unit", time="year", gdf=grid_gdf, tiles=None
    )
    assert isinstance(res.fig_map, go.Figure)
    names = [trace.name for trace in res.fig_map.data]
    assert names == ["Club 1", "Club 2"]
    # club colors come from the theme's CLUB_COLORS cycle
    first_colors = [trace.colorscale[0][1] for trace in res.fig_map.data]
    assert first_colors == [CLUB_COLORS[0], CLUB_COLORS[1]]
    assert all(
        trace.hovertemplate.endswith("<extra></extra>") for trace in res.fig_map.data
    )


def test_clubs_map_partial_overlap_warns(two_club_panel, grid_gdf):
    df, _ = two_club_panel
    sub_gdf = grid_gdf.iloc[:50]
    with pytest.warns(GeometricsWarning, match="matched units only"):
        res = analyze_convergence_clubs(
            df, "x", entity="unit", time="year", gdf=sub_gdf, tiles=None
        )
    assert res.fig_map is not None
    assert any("matched units only" in n for n in res.notes)


def test_clubs_hp_filter_off(two_club_panel):
    df, _ = two_club_panel
    res = analyze_convergence_clubs(
        df, "x", entity="unit", time="year", hp_filter=False
    )
    assert math.isnan(res.hp_lambda)
    assert res.n_clubs >= 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_clubs_unbalanced_panel_raises(two_club_panel):
    df, _ = two_club_panel
    unbalanced = df.drop(df.index[0])
    with pytest.raises(ValueError, match="not balanced"):
        analyze_convergence_clubs(unbalanced, "x", entity="unit", time="year")


def test_clubs_validation_errors(two_club_panel):
    df, _ = two_club_panel
    with pytest.raises(KeyError, match="nope"):
        analyze_convergence_clubs(df, "nope", entity="unit", time="year")
    bad = df.copy()
    bad["txt"] = "a"
    with pytest.raises(TypeError, match="numeric"):
        analyze_convergence_clubs(bad, "txt", entity="unit", time="year")
    with pytest.raises(ValueError, match="trim"):
        analyze_convergence_clubs(df, "x", entity="unit", time="year", trim=1.5)
    with pytest.raises(ValueError, match="merge"):
        analyze_convergence_clubs(df, "x", entity="unit", time="year", merge="bogus")
    single = df[df["unit"] == "u00"]
    with pytest.raises(ValueError, match=">= 2 units"):
        analyze_convergence_clubs(single, "x", entity="unit", time="year")
    short = df[df["year"] <= 5]
    with pytest.raises(ValueError, match="too few periods"):
        analyze_convergence_clubs(short, "x", entity="unit", time="year")


def test_clubs_near_zero_mean_raises(two_club_panel):
    df, _ = two_club_panel
    demeaned = df.copy()
    demeaned["x"] = demeaned["x"] - demeaned.groupby("year")["x"].transform("mean")
    with pytest.raises(ValueError, match="near zero"):
        analyze_convergence_clubs(demeaned, "x", entity="unit", time="year")


def test_clubs_duplicates_warn_and_note(two_club_panel):
    df, _ = two_club_panel
    dirty = pd.concat([df.iloc[[0]], df], ignore_index=True)
    with pytest.warns(GeometricsWarning, match="duplicate"):
        res = analyze_convergence_clubs(dirty, "x", entity="unit", time="year")
    assert any("duplicate" in n for n in res.notes)


def test_clubs_interpret_is_association_only(two_club_panel):
    df, _ = two_club_panel
    res = analyze_convergence_clubs(df, "x", entity="unit", time="year")
    text = res.interpret()
    assert text.endswith(_ASSOC_NOTE)
    assert "causes" not in text
    assert "effect of" not in text
    assert "club" in text
