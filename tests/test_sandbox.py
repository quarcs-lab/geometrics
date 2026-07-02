"""Known-answer and result-surface tests for the ``learn_*`` teaching sandboxes.

Every sandbox plants its parameters, so the tests assert *recovery* — the estimate
sits within tolerance of the planted truth — plus the house result-surface contract
(frozen dataclass, themed figure, finite summary, registered topic, association-only
interpretation, seed determinism).
"""

from __future__ import annotations

import dataclasses
import math
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import geometrics as gm
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

# The nine sandboxes that run without the dynamics extra, with fast small-knob calls
# for the parametrized surface tests (module-scoped cache below).
_FAST_CALLS = {
    "learn_spatial_autocorrelation": {"side": 6, "n_sims": 3, "permutations": 49},
    "learn_spatial_weights": {"side": 6, "permutations": 49},
    "learn_lisa_clusters": {"side": 8, "block": 2, "permutations": 99},
    "learn_spatial_spillovers": {"side": 7, "n_draws": 200},
    "learn_omitted_spatial_lag": {"side": 7},
    "learn_beta_convergence": {"n_units": 30, "n_periods": 4},
    "learn_sigma_convergence": {"n_units": 20, "n_periods": 6},
    "learn_convergence_clubs": {"n_per_club": 8, "n_periods": 20},
    "learn_theil_decomposition": {"n_per_group": 12, "gaps": (0.0, 0.5, 1.0)},
}

_CACHE: dict[str, gm.SandboxResult] = {}


def _fast(name: str) -> gm.SandboxResult:
    """Compute (once) and return the fast-knob result for ``name``."""
    if name not in _CACHE:
        _CACHE[name] = getattr(gm, name)(**_FAST_CALLS[name])
    return _CACHE[name]


def _hide_giddy(monkeypatch):
    """Make ``import giddy`` raise ImportError inside the current process."""
    for mod in [m for m in sys.modules if m == "giddy" or m.startswith("giddy.")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setitem(sys.modules, "giddy", None)


# ---------------------------------------------------------------------------
# Known-answer: each sandbox recovers what it planted (default knobs)
# ---------------------------------------------------------------------------


def test_spatial_autocorrelation_tracks_planted_rho():
    res = gm.learn_spatial_autocorrelation(rho=0.9)
    s = res.summary
    # At rho = 0 the mean I sits near its null expectation ...
    assert abs(s["moran_at_zero"] - s["expected_i"]) < 0.05
    # ... and at strong dependence it pulls far away and is reliably significant.
    assert s["moran_focal"] > 0.4
    assert s["share_significant_focal"] >= 0.9
    # The sweep is monotone on the planted grid's non-negative side.
    means = res.df.groupby("rho")["moran_i"].mean()
    positive = means[means.index >= 0.0]
    assert list(positive.index) == sorted(positive.index)
    assert (positive.diff().dropna() > 0).all()


def test_spatial_weights_all_graphs_detect_the_clustering():
    res = gm.learn_spatial_weights(rho=0.6)
    assert (res.df["moran_i"] > 0).all()
    assert (res.df["p_sim"] < 0.05).all()
    # Queen has more neighbors than rook on a lattice.
    s = res.summary
    assert s["mean_neighbors_queen"] > s["mean_neighbors_rook"]


def test_lisa_recovers_planted_blocks():
    res = gm.learn_lisa_clusters()
    s = res.summary
    assert s["sensitivity_hot"] >= 0.6
    assert s["sensitivity_cold"] >= 0.6
    # Allow spatial smearing at block edges: 3x the nominal alpha.
    assert s["false_positive_rate"] <= 3 * s["alpha"]


def test_spillovers_recover_planted_impacts():
    res = gm.learn_spatial_spillovers()
    s = res.summary
    assert abs(s["est_direct"] - s["true_direct"]) < 0.2
    assert abs(s["est_indirect"] - s["true_indirect"]) < 0.4
    assert abs(s["est_total"] - s["true_total"]) < 0.5
    assert abs(s["rho_hat"] - s["rho"]) < 0.2


def test_omitted_lag_ols_is_more_biased_than_sar():
    res = gm.learn_omitted_spatial_lag()
    s = res.summary
    assert abs(s["ols_coef"] - s["true_beta"]) > abs(s["sar_beta"] - s["true_beta"])
    assert s["ols_bias"] > 0.1  # the multiplier inflates OLS upward here
    assert abs(s["sar_rho"] - s["rho"]) < 0.2


def test_beta_convergence_recovers_planted_rate():
    res = gm.learn_beta_convergence()
    s = res.summary
    assert abs(s["est_beta"] - s["true_beta"]) < 2e-3
    assert abs(s["speed"] - s["true_speed"]) < 2e-3
    assert math.isfinite(s["half_life"]) and s["half_life"] > 0


def test_sigma_convergence_recovers_ln_rho():
    res = gm.learn_sigma_convergence()
    s = res.summary
    assert abs(s["std_slope"] - s["true_slope"]) < 0.01
    assert abs(s["gini_slope"] - s["true_slope"]) < 0.05
    assert abs(s["cv_slope"] - s["true_slope"]) < 0.05


def test_convergence_clubs_recovers_two_clubs():
    res = gm.learn_convergence_clubs()
    s = res.summary
    assert s["detected_clubs"] == 2.0
    assert s["accuracy"] >= 0.9
    assert s["converged"] == 0.0  # global convergence rejected by construction


def test_theil_decomposition_matches_independent_truth():
    res = gm.learn_theil_decomposition()
    s = res.summary
    assert s["max_abs_share_error"] < 0.02
    # The between share rises monotonically with the planted gap.
    shares = res.df["between_share_est"].to_numpy()
    assert (np.diff(shares) > 0).all()
    assert shares[0] < 0.05  # no gap -> essentially all within


@pytest.mark.dynamics
def test_markov_chains_recover_planted_matrix():
    pytest.importorskip("giddy")
    res = gm.learn_markov_chains(n_units=200)
    s = res.summary
    assert s["max_abs_error"] <= 0.1
    assert s["ergodic_l1_error"] <= 0.15
    assert abs(s["mean_persistence_est"] - s["mean_persistence_true"]) < 0.05


@pytest.mark.dynamics
def test_spatial_markov_detects_planted_context_effect():
    pytest.importorskip("giddy")
    res = gm.learn_spatial_markov()
    s = res.summary
    assert s["lr_p"] < 0.05
    assert s["up_prob_rich_nbrs"] > s["up_prob_poor_nbrs"]
    assert s["contextual_gap_est"] > 0.05


@pytest.mark.dynamics
def test_spatial_markov_null_is_homogeneous():
    pytest.importorskip("giddy")
    res = gm.learn_spatial_markov(contextual=0.0)
    assert res.summary["lr_p"] > 0.05


# ---------------------------------------------------------------------------
# Result surface (parametrized over the nine giddy-free sandboxes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_FAST_CALLS))
def test_surface_frozen_and_typed(name):
    res = _fast(name)
    assert isinstance(res, gm.SandboxResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.topic = "oops"
    assert isinstance(res.fig, go.Figure) and len(res.fig.data) >= 1
    assert isinstance(res.df, pd.DataFrame) and not res.df.empty
    assert isinstance(res.data, pd.DataFrame) and not res.data.empty
    assert res.tidy() is res.df


@pytest.mark.parametrize("name", sorted(_FAST_CALLS))
def test_surface_summary_is_finite(name):
    res = _fast(name)
    assert res.summary, "summary must not be empty"
    for key, value in res.summary.items():
        assert isinstance(value, float), f"{key} is not a float"
        assert math.isfinite(value), f"{key} is not finite"


@pytest.mark.parametrize("name", sorted(_FAST_CALLS))
def test_surface_topic_and_explain(name):
    res = _fast(name)
    assert res.topic in gm.list_topics()
    explainer = res.explain()
    assert explainer.topic == res.topic


@pytest.mark.parametrize("name", sorted(_FAST_CALLS))
def test_surface_interpret_is_associational(name):
    text = _fast(name).interpret()
    assert isinstance(text, str) and text
    assert _ASSOC_NOTE in text
    assert " causes " not in text
    assert "effect of" not in text.replace("context effect", "")


@pytest.mark.parametrize(
    "name", ["learn_spatial_autocorrelation", "learn_beta_convergence"]
)
def test_surface_seed_determinism(name):
    kwargs = _FAST_CALLS[name]
    first = getattr(gm, name)(**kwargs)
    again = getattr(gm, name)(**kwargs)
    other = getattr(gm, name)(**kwargs, seed=99)
    assert first.summary == again.summary
    assert not first.data.equals(other.data)


# ---------------------------------------------------------------------------
# Validation and gating
# ---------------------------------------------------------------------------


def test_type_errors_on_non_numeric_knobs():
    with pytest.raises(TypeError):
        gm.learn_spatial_autocorrelation(rho="high")
    with pytest.raises(TypeError):
        gm.learn_beta_convergence(n_units=12.5)


def test_value_errors_on_bad_knobs():
    with pytest.raises(ValueError):
        gm.learn_spatial_autocorrelation(rho=1.0)
    with pytest.raises(ValueError):
        gm.learn_spatial_autocorrelation(side=2)
    with pytest.raises(ValueError):
        gm.learn_spatial_weights(side=4, k=16)
    with pytest.raises(ValueError):
        gm.learn_lisa_clusters(side=8, block=4)
    with pytest.raises(ValueError):
        gm.learn_lisa_clusters(alpha=1.5)
    with pytest.raises(ValueError):
        gm.learn_sigma_convergence(rho=1.2)
    with pytest.raises(ValueError):
        gm.learn_convergence_clubs(levels=(10.0,))
    with pytest.raises(ValueError):
        gm.learn_theil_decomposition(gaps=(0.5,))


def test_markov_knob_validation():
    # The giddy gate fires before knob validation (matching analyze_markov_*), so
    # these run only where the dynamics extra is installed.
    pytest.importorskip("giddy")
    with pytest.raises(TypeError):
        gm.learn_markov_chains(p="not a matrix")
    with pytest.raises(ValueError):
        gm.learn_markov_chains(p=((0.9, 0.2), (0.1, 0.9)))


def test_markov_sandboxes_raise_helpful_import_error(monkeypatch):
    _hide_giddy(monkeypatch)
    with pytest.raises(ImportError, match=r"geometrics\[dynamics\]"):
        gm.learn_markov_chains()
    with pytest.raises(ImportError, match=r"geometrics\[dynamics\]"):
        gm.learn_spatial_markov()


def test_lisa_notes_mention_multiple_testing():
    res = _fast("learn_lisa_clusters")
    assert any("chance" in note for note in res.notes)


def test_llms_groups_include_learn():
    sys.path.insert(0, "tools")
    try:
        from build_llms_txt import _api_groups

        groups = dict(_api_groups())
    finally:
        sys.path.pop(0)
    assert len(groups["learn_*"]) == 11
    assert all(n.startswith("learn_") for n in groups["learn_*"])
    assert not any(n.startswith("learn_") for n in groups["utilities"])
