"""Tests for the distribution-dynamics vertical (giddy Markov analysis)."""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from great_tables import GT

from geometrics._validation import GeometricsWarning
from geometrics.distribution_dynamics import (
    analyze_markov_transitions,
    analyze_spatial_markov,
)
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

pytestmark = pytest.mark.dynamics

pytest.importorskip("giddy")

N_UNITS = 60
N_PERIODS = 10


@pytest.fixture(scope="module")
def planted_chain():
    """3-state block DGP: units drift inside fixed bands, so terciles are sticky.

    Bands at 0 / 10 / 20 with noise sd 3 keep each unit in its own tercile with
    probability around 0.9 per period (occasional boundary crossings).
    """
    rng = np.random.default_rng(20260702)
    base = np.repeat([0.0, 10.0, 20.0], N_UNITS // 3)
    units = [f"r{i:02d}" for i in range(N_UNITS)]
    rows = []
    for year in range(2000, 2000 + N_PERIODS):
        vals = base + rng.normal(0.0, 3.0, N_UNITS)
        rows.append(pd.DataFrame({"unit": units, "year": year, "y": vals}))
    return pd.concat(rows, ignore_index=True)


def _hide_giddy(monkeypatch):
    """Make ``import giddy`` raise ImportError inside the current process."""
    for mod in [m for m in sys.modules if m == "giddy" or m.startswith("giddy.")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setitem(sys.modules, "giddy", None)


# ---------------------------------------------------------------------------
# analyze_markov_transitions
# ---------------------------------------------------------------------------


def test_markov_planted_chain_recovers_persistence(planted_chain):
    res = analyze_markov_transitions(
        planted_chain, "y", entity="unit", time="year", k=3
    )
    p = res.p.to_numpy()
    assert p.shape == (3, 3)
    assert float(np.diag(p).mean()) > 0.7
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert res.steady_state.sum() == pytest.approx(1.0, abs=1e-9)
    assert (res.sojourn > 0).all()
    assert 0.0 <= res.shorrocks <= 1.5
    assert res.n_transitions == N_UNITS * (N_PERIODS - 1)
    assert res.counts.to_numpy().sum() == res.n_transitions


def test_markov_result_surface(planted_chain):
    res = analyze_markov_transitions(
        planted_chain, "y", entity="unit", time="year", k=3
    )
    assert res.states == ("Q1", "Q2", "Q3")
    assert res.k == 3
    assert res.scheme == "quantiles"
    assert res.var == "y"
    # long panel: one row per (unit, period) with the discretized state
    assert list(res.df.columns) == ["unit", "year", "y", "state"]
    assert len(res.df) == N_UNITS * N_PERIODS
    assert set(res.df["state"]) <= {"Q1", "Q2", "Q3"}
    # annotated heatmap + GT summary
    assert isinstance(res.fig, go.Figure)
    assert isinstance(res.fig.data[0], go.Heatmap)
    assert res.fig.data[0].texttemplate == "%{z:.2f}"
    assert res.fig.data[0].hovertemplate.endswith("<extra></extra>")
    assert isinstance(res.gt, GT)
    # frozen dataclass
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.k = 4
    # tidy(): long transition-probability frame
    tidy = res.tidy()
    assert list(tidy.columns) == ["state_from", "state_to", "probability"]
    assert len(tidy) == 9


def test_markov_relative_divides_by_period_mean(planted_chain):
    res = analyze_markov_transitions(
        planted_chain, "y", entity="unit", time="year", k=3, relative=True
    )
    assert any("relative" in n for n in res.notes)
    # relative values average to 1 within each period
    means = res.df.groupby("year")["y"].mean()
    np.testing.assert_allclose(means.to_numpy(), 1.0, atol=1e-12)
    np.testing.assert_allclose(res.p.to_numpy().sum(axis=1), 1.0, atol=1e-9)


def test_markov_pooled_classification(planted_chain):
    res = analyze_markov_transitions(
        planted_chain, "y", entity="unit", time="year", k=3, per_period=False
    )
    assert res.k == 3
    np.testing.assert_allclose(res.p.to_numpy().sum(axis=1), 1.0, atol=1e-9)


def test_markov_user_defined_bins(planted_chain):
    res = analyze_markov_transitions(
        planted_chain, "y", entity="unit", time="year", bins=[5.0, 15.0]
    )
    assert res.scheme == "user_defined"
    assert res.k == 3  # (-inf, 5], (5, 15], (15, max]
    assert res.states == ("C1", "C2", "C3")
    assert float(np.diag(res.p.to_numpy()).mean()) > 0.7


def test_markov_scheme_variants(planted_chain):
    res = analyze_markov_transitions(
        planted_chain, "y", entity="unit", time="year", k=3, scheme="equal_interval"
    )
    assert res.scheme == "equal_interval"
    assert res.states == ("C1", "C2", "C3")
    with pytest.raises(ValueError, match="scheme"):
        analyze_markov_transitions(
            planted_chain, "y", entity="unit", time="year", scheme="nope"
        )


def test_markov_unbalanced_panel_raises(planted_chain):
    unbalanced = planted_chain.iloc[:-1]
    with pytest.raises(ValueError, match="balanced"):
        analyze_markov_transitions(unbalanced, "y", entity="unit", time="year", k=3)


def test_markov_validation_errors(planted_chain):
    with pytest.raises(KeyError, match="nope"):
        analyze_markov_transitions(planted_chain, "nope", entity="unit", time="year")
    bad = planted_chain.assign(y=planted_chain["unit"])
    with pytest.raises(TypeError, match="numeric"):
        analyze_markov_transitions(bad, "y", entity="unit", time="year")
    with pytest.raises(ValueError, match="k="):
        analyze_markov_transitions(planted_chain, "y", entity="unit", time="year", k=1)
    with pytest.raises(TypeError, match="DataFrame"):
        analyze_markov_transitions("not a df", "y", entity="unit", time="year")


def test_markov_interpret_association_only(planted_chain):
    res = analyze_markov_transitions(
        planted_chain, "y", entity="unit", time="year", k=3
    )
    text = res.interpret()
    assert isinstance(text, str)
    assert text.endswith(_ASSOC_NOTE)
    assert "causes" not in text.lower()
    assert "effect of" not in text.lower()
    assert "Shorrocks" in text


# ---------------------------------------------------------------------------
# analyze_spatial_markov
# ---------------------------------------------------------------------------


def test_spatial_markov_shapes_on_convergence_panel(
    convergence_panel, grid_gdf, grid_w
):
    res = analyze_spatial_markov(
        convergence_panel,
        "gdppc",
        gdf=grid_gdf,
        w=grid_w,
        entity="unit",
        time="year",
        k=3,
        m=3,
    )
    assert res.k == 3
    assert res.m == 3
    assert res.relative is True  # the default
    assert any("relative" in n for n in res.notes)
    assert res.p_global.shape == (3, 3)
    np.testing.assert_allclose(res.p_global.to_numpy().sum(axis=1), 1.0, atol=1e-9)
    assert len(res.p_conditional) == 3
    for cond in res.p_conditional:
        assert cond.shape == (3, 3)
        np.testing.assert_allclose(cond.to_numpy().sum(axis=1), 1.0, atol=1e-9)
    # steady states: each row sums to 1, or is all-NaN with an explanatory note
    assert res.steady_states.shape == (3, 3)
    for _, row in res.steady_states.iterrows():
        if row.isna().all():
            assert any("ergodic" in n for n in res.notes)
        else:
            assert row.sum() == pytest.approx(1.0, abs=1e-6)
    # LR / Q: finite (with dof) or NaN with an explanatory note
    for stat in (res.lr_stat, res.q_stat):
        if np.isfinite(stat):
            assert res.dof > 0
        else:
            assert any("homogeneity" in n for n in res.notes)
    assert res.w_spec  # grid_w has no geometrics_meta -> composed description
    assert "n=64" in res.w_spec


def test_spatial_markov_df_and_fig(convergence_panel, grid_gdf, grid_w):
    res = analyze_spatial_markov(
        convergence_panel,
        "gdppc",
        gdf=grid_gdf,
        w=grid_w,
        entity="unit",
        time="year",
        k=3,
        m=3,
    )
    assert list(res.df.columns) == ["unit", "year", "gdppc", "state", "neighbor_state"]
    assert len(res.df) == 64 * 6
    assert set(res.df["state"]) <= {"Q1", "Q2", "Q3"}
    assert set(res.df["neighbor_state"]) <= {"Q1", "Q2", "Q3"}
    # one heatmap per neighbor class, all bound to the shared coloraxis
    heatmaps = [t for t in res.fig.data if isinstance(t, go.Heatmap)]
    assert len(heatmaps) == 3
    assert all(t.coloraxis == "coloraxis" for t in heatmaps)
    titles = [a["text"] for a in res.fig.layout.annotations]
    assert "Neighbors in Q1" in titles
    assert isinstance(res.gt, GT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.m = 5


def test_spatial_markov_mobile_panel_steady_states(grid_gdf, grid_w):
    # A mobile 64-unit panel (noisy bands) keeps every conditional chain irreducible,
    # so all conditional steady states are well defined and sum to one.
    rng = np.random.default_rng(7)
    units = list(grid_gdf["unit"])
    rows = []
    base = rng.permutation(np.repeat([0.0, 10.0, 20.0, 30.0], 16))
    for year in range(2000, 2008):
        rows.append(
            pd.DataFrame(
                {
                    "unit": units,
                    "year": year,
                    "y": base + rng.normal(0.0, 6.0, 64),
                }
            )
        )
    panel = pd.concat(rows, ignore_index=True)
    res = analyze_spatial_markov(
        panel,
        "y",
        gdf=grid_gdf,
        w=grid_w,
        entity="unit",
        time="year",
        k=2,
        m=2,
        relative=False,
    )
    assert res.relative is False
    assert res.steady_states.notna().all().all()
    np.testing.assert_allclose(res.steady_states.to_numpy().sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(res.lr_stat)
    assert np.isfinite(res.q_stat)
    assert res.dof > 0


def test_spatial_markov_default_weights_warns(convergence_panel, grid_gdf):
    with pytest.warns(GeometricsWarning, match="no spatial weights supplied"):
        res = analyze_spatial_markov(
            convergence_panel,
            "gdppc",
            gdf=grid_gdf,
            entity="unit",
            time="year",
            k=3,
            m=3,
        )
    assert "queen contiguity" in res.w_spec


def test_spatial_markov_unbalanced_panel_raises(convergence_panel, grid_gdf, grid_w):
    unbalanced = convergence_panel.iloc[:-1]
    with pytest.raises(ValueError, match="balanced"):
        analyze_spatial_markov(
            unbalanced, "gdppc", gdf=grid_gdf, w=grid_w, entity="unit", time="year"
        )


def test_spatial_markov_interpret_association_only(convergence_panel, grid_gdf, grid_w):
    res = analyze_spatial_markov(
        convergence_panel,
        "gdppc",
        gdf=grid_gdf,
        w=grid_w,
        entity="unit",
        time="year",
        k=3,
        m=3,
    )
    text = res.interpret()
    assert isinstance(text, str)
    assert text.endswith(_ASSOC_NOTE)
    assert "causes" not in text.lower()
    assert "effect of" not in text.lower()
    assert "neighbor" in text.lower()


# ---------------------------------------------------------------------------
# lazy giddy import
# ---------------------------------------------------------------------------


def test_markov_missing_giddy_raises_helpful_error(monkeypatch, planted_chain):
    _hide_giddy(monkeypatch)
    with pytest.raises(ImportError, match=r"geometrics\[dynamics\]"):
        analyze_markov_transitions(planted_chain, "y", entity="unit", time="year")


def test_spatial_markov_missing_giddy_raises_helpful_error(
    monkeypatch, convergence_panel, grid_gdf, grid_w
):
    _hide_giddy(monkeypatch)
    with pytest.raises(
        ImportError, match=r"analyze_spatial_markov requires the dynamics extra"
    ):
        analyze_spatial_markov(
            convergence_panel,
            "gdppc",
            gdf=grid_gdf,
            w=grid_w,
            entity="unit",
            time="year",
        )
