"""Frozen result dataclasses returned by the public analysis functions.

The ``explore_*`` / ``analyze_*`` functions return these typed wrappers: small, typed,
immutable dataclasses that expose the underlying :class:`pandas.DataFrame` alongside the
rendered object (a Plotly ``Figure``, a Great Tables ``GT``, or both).

Many result types also mix in :class:`geometrics.pedagogy.Interpretable`, which adds a
small broom-style surface: ``interpret()`` (plain-language reading of the result),
``explain()`` (the concept explainer for the method) and, where meaningful, ``tidy()`` /
``glance()``.

Spatial results carry ``w_spec`` — a human-readable, single-string description of the
spatial weights that produced them (method, island handling, standardization, n) — so a
result is always auditable without the ``W`` object in hand.

The ``interpret`` methods import their ``interpret_*`` implementation lazily (inside the
method) rather than at module top: the interpret functions live in per-domain modules
under :mod:`geometrics.pedagogy` and are duck-typed against these classes, so the lazy
import keeps this module importable independent of them and rules out import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from geometrics.pedagogy import Interpretable
from geometrics.pedagogy import explain as _explain

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd
    import plotly.graph_objects as go
    from great_tables import GT

    from geometrics.pedagogy import Explainer

__all__ = [
    # ===== EXPLORE =====
    "ChoroplethMapResult",
    "ConnectivityMapResult",
    "MoranPlotResult",
    "LisaClusterMapResult",
    "MoranOverTimeResult",
    "DistributionOverTimeResult",
    "SpacetimeHeatmapResult",
    # ===== ANALYZE =====
    "BetaConvergenceResult",
    "SigmaConvergenceResult",
    "ConvergenceClubsResult",
    "SpatialModelResult",
    "SpatialDiagnosticsResult",
    "WeightsRobustnessResult",
    "MarkovTransitionsResult",
    "SpatialMarkovResult",
    "InequalityOverTimeResult",
    "TheilDecompositionResult",
    "GWRResult",
    "MGWRResult",
]


# ---------------------------------------------------------------------------
# EXPLORE results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChoroplethMapResult(Interpretable):
    """Result of :func:`geometrics.explore_choropleth_map`.

    ``df`` is the per-entity frame behind the map (entity, value, class label);
    ``gdf_plotted`` is the WGS84 geometry actually drawn (values and class attached),
    ready for reuse. ``bins`` holds the upper class bounds when a classification
    ``scheme`` is applied (empty for continuous maps). ``animated`` flags a multi-period
    frame animation, in which case ``period`` is the first period drawn.
    """

    df: pd.DataFrame
    fig: go.Figure
    gdf_plotted: gpd.GeoDataFrame
    var: str
    period: Any
    scheme: str | None
    k: int
    bins: tuple[float, ...]
    animated: bool = False
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the mapped distribution and its class breaks."""
        from geometrics.pedagogy._interpret import interpret_choropleth_map

        return interpret_choropleth_map(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for choropleth classification schemes."""
        return _explain("choropleth_classification", lang=lang)


@dataclass(frozen=True)
class ConnectivityMapResult(Interpretable):
    """Result of :func:`geometrics.explore_connectivity_map`.

    ``df`` is the per-entity neighbor-cardinality frame; ``fig`` draws the weights graph
    over the (grey) polygon layer and ``fig_hist`` the cardinality histogram. The scalars
    summarize the connectivity structure of the ``W`` described by ``w_spec``; ``islands``
    lists entity ids that had no neighbors before any island attachment.
    """

    df: pd.DataFrame
    fig: go.Figure
    fig_hist: go.Figure
    n_units: int
    mean_neighbors: float
    min_neighbors: int
    max_neighbors: int
    pct_nonzero: float
    n_components: int
    islands: tuple[Any, ...]
    w_spec: str
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the connectivity health of the weights graph."""
        from geometrics.pedagogy._interpret import interpret_connectivity_map

        return interpret_connectivity_map(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for spatial weights matrices."""
        return _explain("spatial_weights", lang=lang)


@dataclass(frozen=True)
class MoranPlotResult(Interpretable):
    """Result of :func:`geometrics.explore_moran_plot`.

    ``df`` holds one row per entity (``entity``, ``value``, ``lag``, ``quadrant``) with
    the standardized variable and its spatial lag. ``moran_i`` is global Moran's I with
    its permutation-based pseudo p-value ``p_sim`` and standardized ``z_sim`` over
    ``permutations`` conditional permutations; ``expected_i`` is E[I] under the null.
    """

    df: pd.DataFrame
    fig: go.Figure
    moran_i: float
    expected_i: float
    p_sim: float
    z_sim: float
    permutations: int
    var: str
    period: Any
    w_spec: str
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the strength and significance of clustering."""
        from geometrics.pedagogy._interpret import interpret_moran_plot

        return interpret_moran_plot(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for spatial autocorrelation and Moran's I."""
        return _explain("spatial_autocorrelation", lang=lang)

    def glance(self) -> pd.DataFrame:
        """One-row summary of the global Moran test."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "var": self.var,
                    "moran_i": self.moran_i,
                    "expected_i": self.expected_i,
                    "z_sim": self.z_sim,
                    "p_sim": self.p_sim,
                    "permutations": self.permutations,
                    "n_obs": len(self.df),
                }
            ]
        )


@dataclass(frozen=True)
class LisaClusterMapResult(Interpretable):
    """Result of :func:`geometrics.explore_lisa_cluster_map`.

    ``df`` holds one row per entity (``entity``, ``value``, ``lag``, ``local_i``,
    ``quadrant``, ``p_sim``, ``cluster``) where ``cluster`` is the significance-masked
    label (High-High, Low-Low, Low-High, High-Low, or Not significant at ``alpha``).
    ``fig`` is the cluster map; ``fig_scatter`` the quadrant-colored Moran scatterplot.
    ``moran_i`` / ``p_sim_global`` report the accompanying global test, and the ``n_*``
    counts tally entities per significant cluster class (``n_ns`` = not significant).
    """

    df: pd.DataFrame
    fig: go.Figure
    fig_scatter: go.Figure
    moran_i: float
    p_sim_global: float
    alpha: float
    permutations: int
    n_hh: int
    n_ll: int
    n_hl: int
    n_lh: int
    n_ns: int
    var: str
    period: Any
    w_spec: str
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the local clusters and spatial outliers."""
        from geometrics.pedagogy._interpret import interpret_lisa_cluster_map

        return interpret_lisa_cluster_map(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for local Moran statistics (LISA)."""
        return _explain("local_moran", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the per-entity LISA frame."""
        return self.df


@dataclass(frozen=True)
class MoranOverTimeResult(Interpretable):
    """Result of :func:`geometrics.explore_moran_over_time`.

    ``df`` holds one row per period (``period``, ``moran_i``, ``z_sim``, ``p_sim``,
    ``n_obs``) tracking how global spatial dependence in ``var`` evolves under the
    weights described by ``w_spec``.
    """

    df: pd.DataFrame
    fig: go.Figure
    var: str
    permutations: int
    w_spec: str
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the trajectory of spatial dependence."""
        from geometrics.pedagogy._interpret import interpret_moran_over_time

        return interpret_moran_over_time(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for spatial autocorrelation and Moran's I."""
        return _explain("spatial_autocorrelation", lang=lang)


@dataclass(frozen=True)
class DistributionOverTimeResult(Interpretable):
    """Result of :func:`geometrics.explore_distribution_over_time`.

    ``df`` is the tidy evaluation frame behind the densities (``time``, ``value``,
    ``density``). When ``relative`` is ``True`` the variable was divided by its
    cross-sectional mean per period before density estimation (the distribution-dynamics
    convention), so 1.0 marks the period average.
    """

    df: pd.DataFrame
    fig: go.Figure
    var: str
    kind: str
    relative: bool
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of how the cross-sectional distribution evolves."""
        from geometrics.pedagogy._interpret import interpret_distribution_over_time

        return interpret_distribution_over_time(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for distribution dynamics."""
        return _explain("distribution_dynamics", lang=lang)


@dataclass(frozen=True)
class SpacetimeHeatmapResult(Interpretable):
    """Result of :func:`geometrics.explore_spacetime_heatmap`.

    ``df`` is the entity-by-time pivot actually drawn (rows in display order).
    ``sort_by`` records the row ordering used (``"value"``, ``"name"``,
    ``"north_south"`` or ``"east_west"``; the geographic orderings need a ``gdf``).
    """

    df: pd.DataFrame
    fig: go.Figure
    var: str
    sort_by: str
    relative: bool
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the space-time value surface."""
        from geometrics.pedagogy._interpret import interpret_spacetime_heatmap

        return interpret_spacetime_heatmap(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for distribution dynamics."""
        return _explain("distribution_dynamics", lang=lang)


# ---------------------------------------------------------------------------
# ANALYZE results — convergence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaConvergenceResult(Interpretable):
    """Result of :func:`geometrics.analyze_beta_convergence`.

    ``df`` is the per-unit growth cross-section the analysis is built on (``entity``,
    ``initial``, ``final``, annualized ``growth``, and any initial-period controls).
    ``fig`` is the growth-vs-initial scatter with the fitted line; ``fig_conditional``
    the Frisch-Waugh-Lovell partial scatter once controls are partialled out (``None``
    without controls); ``fig_map`` the growth choropleth (``None`` without a ``gdf``).
    ``gt`` / ``summary`` hold the estimate table.

    ``model`` records the estimator (``"ols"``, ``"sar"``, ``"sem"``, ``"slx"`` or
    ``"sdm"``). The scalar triple ``beta_direct`` / ``beta_indirect`` / ``beta_total``
    reports the LeSage-Pace impact decomposition of the initial-level coefficient (for
    OLS the three collapse to β with ``beta_indirect`` = ``nan``), with matching
    ``se_*`` from ``n_draws`` Monte-Carlo draws for the spatial models. ``rho`` is the
    spatial-lag parameter (``nan`` where absent) and ``lam`` the spatial-error parameter.
    ``speed`` and ``half_life`` derive from the **total** effect over horizon ``T``
    (Barro-Sala-i-Martin). ``impacts`` is the per-regressor impact table for spatial
    models (``None`` for OLS).
    """

    df: pd.DataFrame
    fig: go.Figure
    fig_conditional: go.Figure | None
    fig_map: go.Figure | None
    gt: GT
    summary: pd.DataFrame
    models: list[Any]
    model: str
    var: str
    controls: tuple[str, ...]
    horizon: float
    beta_direct: float
    beta_indirect: float
    beta_total: float
    se_direct: float
    se_indirect: float
    se_total: float
    rho: float
    lam: float
    r2: float
    aic: float
    n_obs: int
    speed: float
    half_life: float
    impacts: pd.DataFrame | None = None
    n_draws: int = 0
    w_spec: str | None = None
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the convergence slope, spillovers, speed."""
        from geometrics.pedagogy._interpret import interpret_beta_convergence

        return interpret_beta_convergence(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for β-convergence (spatial variant when applicable)."""
        topic = "beta_convergence" if self.model == "ols" else "spatial_convergence"
        return _explain(topic, lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the estimate table (direct/indirect/total where spatial)."""
        return self.summary

    def glance(self) -> pd.DataFrame:
        """One-row summary of the convergence fit."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "model": self.model,
                    "var": self.var,
                    "beta_total": self.beta_total,
                    "se_total": self.se_total,
                    "rho": self.rho,
                    "r2": self.r2,
                    "aic": self.aic,
                    "n_obs": self.n_obs,
                    "horizon": self.horizon,
                    "speed": self.speed,
                    "half_life": self.half_life,
                }
            ]
        )


@dataclass(frozen=True)
class SigmaConvergenceResult(Interpretable):
    """Result of :func:`geometrics.analyze_sigma_convergence`.

    ``df`` is the per-period dispersion table (one row per period: ``time``,
    ``n_units``, ``mean``, ``std``, ``gini``, ``cv``). ``fig`` is the dual-axis time
    series with fitted trend overlays; ``gt`` / ``summary`` the trend table (one row per
    measure: ``measure``, ``slope``, ``se``, ``pvalue``, ``r2``, ``n_periods_used``,
    ``converging``). ``models`` holds the fitted statsmodels trend model(s).

    The scalar fields report the OLS trend of each measure's **log dispersion** on time:
    a **negative** slope is σ-convergence. A measure's scalars are ``nan`` when its
    trend could not be estimated (see ``notes``).
    """

    df: pd.DataFrame
    fig: go.Figure
    gt: GT
    summary: pd.DataFrame
    models: list[Any]
    var: str
    entity: str
    time: str
    n_periods: int
    n_units: int
    std_slope: float
    std_se: float
    std_pvalue: float
    std_r2: float
    gini_slope: float = float("nan")
    gini_se: float = float("nan")
    gini_pvalue: float = float("nan")
    gini_r2: float = float("nan")
    cv_slope: float = float("nan")
    cv_se: float = float("nan")
    cv_pvalue: float = float("nan")
    cv_r2: float = float("nan")
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of whether and how fast dispersion narrows."""
        from geometrics.pedagogy._interpret import interpret_sigma_convergence

        return interpret_sigma_convergence(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for σ-convergence."""
        return _explain("sigma_convergence", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the per-measure trend table."""
        return self.summary


@dataclass(frozen=True)
class ConvergenceClubsResult(Interpretable):
    """Result of :func:`geometrics.analyze_convergence_clubs`.

    ``df`` is the tidy long panel behind the analysis: one row per (unit, period) with
    ``value`` (the HP-filtered trend, or the raw variable when unfiltered), ``relative``
    (the relative transition ``h_it``) and ``club`` (``0`` = divergent group). ``fig``
    is the within-club average transition-path figure; ``fig_paths`` overlays every
    unit's path colored by club; ``fig_clubs`` is the per-club small-multiples panel;
    ``fig_map`` the club-membership choropleth (``None`` without a ``gdf``). ``gt`` /
    ``summary`` hold the classification table and ``membership`` the tidy
    entity-to-club frame.

    The scalars report the panel dimensions, the whole-panel log(t) test
    (``global_beta``, ``global_tstat``, ``converged``), the club counts, and the run
    parameters (``hp_lambda``, ``trim``, ``tcrit``, ``method``, ``merge``).
    """

    df: pd.DataFrame
    fig: go.Figure
    fig_paths: go.Figure
    fig_clubs: go.Figure
    fig_map: go.Figure | None
    gt: GT
    summary: pd.DataFrame
    membership: pd.DataFrame
    var: str
    entity: str
    time: str
    n_units: int
    n_periods: int
    n_clubs: int
    n_divergent: int
    global_beta: float
    global_tstat: float
    converged: bool
    hp_lambda: float
    trim: float
    tcrit: float
    method: str
    merge: str
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of how the panel splits into convergence clubs."""
        from geometrics.pedagogy._interpret import interpret_convergence_clubs

        return interpret_convergence_clubs(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for club convergence."""
        return _explain("convergence_clubs", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the per-club classification table."""
        return self.summary


# ---------------------------------------------------------------------------
# ANALYZE results — spatial econometric models (spreg)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialModelResult(Interpretable):
    """Result of :func:`geometrics.analyze_spatial_model`.

    ``df`` is the tidy coefficient frame (``term``, ``estimate``, ``se``, ``z``,
    ``p``); ``model_obj`` the fitted spreg object. ``model`` is one of ``"ols"``,
    ``"lag"``, ``"error"``, ``"slx"``, ``"durbin"``, ``"durbin_error"`` and ``method``
    is ``"ml"`` or ``"gm"``. ``rho`` / ``lam`` are the spatial parameters (``nan``
    where absent). ``impacts`` holds the per-regressor LeSage-Pace direct/indirect/total
    table with Monte-Carlo standard errors from ``n_draws`` draws (``None`` where
    impacts are undefined for the model).
    """

    df: pd.DataFrame
    gt: GT
    model_obj: Any
    model: str
    method: str
    rho: float
    lam: float
    r2: float
    log_likelihood: float
    aic: float
    n_obs: int
    outcome: str
    covariates: tuple[str, ...]
    period: Any
    w_spec: str
    impacts: pd.DataFrame | None = None
    n_draws: int = 0
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the spatial parameters and impacts."""
        from geometrics.pedagogy._interpret import interpret_spatial_model

        return interpret_spatial_model(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for the estimated spatial model."""
        topic = {
            "ols": "lm_diagnostics",
            "lag": "spatial_lag_model",
            "error": "spatial_error_model",
            "slx": "slx_model",
            "durbin": "spatial_durbin_model",
            "durbin_error": "spatial_durbin_model",
        }.get(self.model, "spatial_durbin_model")
        return _explain(topic, lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the tidy coefficient frame."""
        return self.df

    def glance(self) -> pd.DataFrame:
        """One-row model-level summary."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "model": self.model,
                    "method": self.method,
                    "rho": self.rho,
                    "lam": self.lam,
                    "r2": self.r2,
                    "log_likelihood": self.log_likelihood,
                    "aic": self.aic,
                    "n_obs": self.n_obs,
                }
            ]
        )


@dataclass(frozen=True)
class SpatialDiagnosticsResult(Interpretable):
    """Result of :func:`geometrics.analyze_spatial_diagnostics`.

    ``df`` holds one row per test (Moran's I on residuals, LM lag, LM error, robust LM
    lag, robust LM error, LM SARMA) with ``statistic``, ``df`` and ``p``.
    ``recommendation`` applies the Anselin-Florax decision rule (``"ols"``, ``"lag"``,
    ``"error"`` or ``"consider durbin"``) and ``reasoning`` spells out why.
    """

    df: pd.DataFrame
    gt: GT
    moran_i_resid: float
    recommendation: str
    reasoning: str
    ols_model: Any
    alpha: float
    w_spec: str
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the LM diagnostics and the model they point to."""
        from geometrics.pedagogy._interpret import interpret_spatial_diagnostics

        return interpret_spatial_diagnostics(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for Lagrange-multiplier specification tests."""
        return _explain("lm_diagnostics", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the per-test frame."""
        return self.df


@dataclass(frozen=True)
class WeightsRobustnessResult(Interpretable):
    """Result of :func:`geometrics.analyze_spatial_model_by_weights`.

    ``df`` holds one row per weights specification (``weights``, ``rho``, ``direct``,
    ``indirect``, ``total`` with their Monte-Carlo standard errors, ``aic``, ``n_obs``)
    for the focal regressor ``focal``. ``fig`` is the three-facet dot-whisker figure
    with the ``baseline`` specification marked.
    """

    df: pd.DataFrame
    fig: go.Figure
    gt: GT
    baseline: str
    focal: str
    model: str
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of how stable the impacts are across weights."""
        from geometrics.pedagogy._interpret import interpret_weights_robustness

        return interpret_weights_robustness(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for weights-choice robustness."""
        return _explain("weights_robustness", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the per-weights impact frame."""
        return self.df


# ---------------------------------------------------------------------------
# ANALYZE results — distribution dynamics (giddy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkovTransitionsResult(Interpretable):
    """Result of :func:`geometrics.analyze_markov_transitions`.

    ``df`` is the long panel with the discretized ``state`` per (entity, time); ``p``
    the K-by-K transition-probability matrix (labelled by ``states``) and ``counts``
    the raw transition counts. ``steady_state`` is the ergodic distribution and
    ``sojourn`` the expected time in each state. ``shorrocks`` / ``prais`` /
    ``bartholomew`` are mobility indices of ``p``.
    """

    df: pd.DataFrame
    p: pd.DataFrame
    counts: pd.DataFrame
    fig: go.Figure
    gt: GT
    states: tuple[str, ...]
    steady_state: pd.Series
    sojourn: pd.Series
    shorrocks: float
    prais: float
    bartholomew: float
    n_transitions: int
    k: int
    scheme: str
    var: str
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of persistence, mobility and the long-run mix."""
        from geometrics.pedagogy._interpret import interpret_markov_transitions

        return interpret_markov_transitions(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for Markov transition analysis."""
        return _explain("markov_chains", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the transition-probability matrix as a tidy long frame."""
        long = self.p.melt(
            ignore_index=False, var_name="state_to", value_name="probability"
        ).reset_index(names="state_from")
        return long[["state_from", "state_to", "probability"]]


@dataclass(frozen=True)
class SpatialMarkovResult(Interpretable):
    """Result of :func:`geometrics.analyze_spatial_markov`.

    ``df`` is the long panel with each entity's ``state`` and its neighbors'
    ``neighbor_state``; ``p_global`` the unconditional K-by-K matrix and
    ``p_conditional`` the tuple of m neighbor-conditioned matrices (``steady_states``
    stacks their ergodic distributions). ``lr_stat`` / ``q_stat`` (with p-values and
    ``dof``) test whether transition dynamics are homogeneous across neighbor classes —
    rejection means neighbors condition mobility.
    """

    df: pd.DataFrame
    p_global: pd.DataFrame
    p_conditional: tuple[pd.DataFrame, ...]
    steady_states: pd.DataFrame
    fig: go.Figure
    gt: GT
    lr_stat: float
    lr_p: float
    q_stat: float
    q_p: float
    dof: int
    k: int
    m: int
    relative: bool
    var: str
    w_spec: str
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of how neighbors condition transition dynamics."""
        from geometrics.pedagogy._interpret import interpret_spatial_markov

        return interpret_spatial_markov(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for the spatial Markov chain."""
        return _explain("spatial_markov", lang=lang)


# ---------------------------------------------------------------------------
# ANALYZE results — regional inequality (PySAL inequality)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InequalityOverTimeResult(Interpretable):
    """Result of :func:`geometrics.analyze_inequality_over_time`.

    ``df`` holds one row per period with the requested measures (``gini``, ``theil``,
    ``cv``; plus ``gini_spatial`` and ``gini_spatial_p`` when a ``w`` enables the
    spatial Gini decomposition). ``gt`` / ``summary`` hold the per-measure log-trend
    table (``measure``, ``slope``, ``se``, ``pvalue``, ``r2``, ``converging``) —
    the inequality-narrative complement of σ-convergence.
    """

    df: pd.DataFrame
    fig: go.Figure
    gt: GT
    summary: pd.DataFrame
    models: list[Any]
    var: str
    n_periods: int
    n_units: int
    w_spec: str | None = None
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the inequality level and trend."""
        from geometrics.pedagogy._interpret import interpret_inequality_over_time

        return interpret_inequality_over_time(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for the Gini index."""
        return _explain("gini", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the per-measure trend table."""
        return self.summary


@dataclass(frozen=True)
class TheilDecompositionResult(Interpretable):
    """Result of :func:`geometrics.analyze_theil_decomposition`.

    ``df`` holds one row per period (``time``, ``theil``, ``between``, ``within``,
    ``between_share``; plus ``p_between`` when permutation inference is requested).
    ``group`` is the partition column (e.g. states) with ``n_groups`` groups.
    """

    df: pd.DataFrame
    fig: go.Figure
    gt: GT
    group: str
    n_groups: int
    permutations: int
    var: str
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the between/within inequality split."""
        from geometrics.pedagogy._interpret import interpret_theil_decomposition

        return interpret_theil_decomposition(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for the Theil between/within decomposition."""
        return _explain("theil_decomposition", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the per-period decomposition frame."""
        return self.df


# ---------------------------------------------------------------------------
# ANALYZE results — local models (mgwr)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GWRResult(Interpretable):
    """Result of :func:`geometrics.analyze_gwr`.

    ``df`` holds one row per entity with the local coefficient, standard error and
    t-value per term, ``local_r2``, and a ``<term>_significant`` flag applying the
    multiple-testing-corrected threshold (``adj_alpha`` → ``critical_t``). ``figs``
    maps each term to its local-coefficient map (non-significant entities greyed);
    ``fig`` is the local-R² map. ``bw`` is the selected bandwidth (``fixed`` distance
    or adaptive neighbor count).
    """

    df: pd.DataFrame
    figs: dict[str, go.Figure]
    fig: go.Figure
    gt: GT
    bw: float
    fixed: bool
    kernel: str
    aicc: float
    r2: float
    adj_alpha: float
    critical_t: float
    model_obj: Any
    outcome: str
    covariates: tuple[str, ...]
    period: Any
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of where and how associations vary over space."""
        from geometrics.pedagogy._interpret import interpret_gwr

        return interpret_gwr(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for geographically weighted regression."""
        return _explain("gwr", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the per-entity local-coefficient frame."""
        return self.df

    def glance(self) -> pd.DataFrame:
        """One-row model-level summary."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "bw": self.bw,
                    "fixed": self.fixed,
                    "kernel": self.kernel,
                    "aicc": self.aicc,
                    "r2": self.r2,
                    "adj_alpha": self.adj_alpha,
                    "critical_t": self.critical_t,
                    "n_obs": len(self.df),
                }
            ]
        )


@dataclass(frozen=True)
class MGWRResult(Interpretable):
    """Result of :func:`geometrics.analyze_mgwr`.

    Like :class:`GWRResult` but multiscale: ``bw`` maps each term to its own
    bandwidth and ``gt_bw`` renders the bandwidth table. Variables are always
    z-standardized for MGWR, so local coefficients are on the standardized scale
    (recorded in ``notes``).
    """

    df: pd.DataFrame
    figs: dict[str, go.Figure]
    fig: go.Figure
    gt: GT
    gt_bw: GT
    bw: dict[str, float]
    kernel: str
    aicc: float
    r2: float
    adj_alpha: dict[str, float]
    critical_t: dict[str, float]
    model_obj: Any
    outcome: str
    covariates: tuple[str, ...]
    period: Any
    notes: tuple[str, ...] = ()

    def interpret(self, *, lang: str = "en") -> str:
        """Plain-language reading of the per-term operating scales and local patterns."""
        from geometrics.pedagogy._interpret import interpret_mgwr

        return interpret_mgwr(self, lang=lang)

    def explain(self, *, lang: str = "en") -> Explainer:
        """Concept explainer for multiscale GWR."""
        return _explain("mgwr", lang=lang)

    def tidy(self) -> pd.DataFrame:
        """Return the per-entity local-coefficient frame."""
        return self.df
