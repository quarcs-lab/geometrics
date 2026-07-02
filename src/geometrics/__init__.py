"""geometrics: regional growth, convergence, and inequality on the PySAL stack.

geometrics wraps the standard analyses of the regional convergence literature —
exploratory spatial data analysis, β/σ/club convergence, spatial econometric models,
distribution dynamics, inequality decomposition, and local (GWR) models — into
illustrative, easy-to-apply functions built on libpysal, esda, giddy, inequality,
mapclassify, spreg, and mgwr.

Three inputs drive everything: a geometry with only the entity ID (``read_gdf``),
a long-form panel (``set_panel`` / ``set_labels``), and a data dictionary
(``df_dict``, inferable with ``build_data_dict``). Every public function returns a
frozen result dataclass with ``.df``, ``.fig`` and/or ``.gt``, plain-language
``.interpret()``, and a concept ``.explain()``.
"""

from geometrics import data
from geometrics._data_dict import build_data_dict
from geometrics._geo import read_gdf
from geometrics._labels import resolve_label, set_labels
from geometrics._panel import resolve_panel, set_panel
from geometrics._roles import set_roles
from geometrics._theme import get_palette, set_palette
from geometrics._types import (
    BetaConvergenceResult,
    ChoroplethMapResult,
    ConnectivityMapResult,
    ConvergenceClubsResult,
    DistributionOverTimeResult,
    GWRResult,
    InequalityOverTimeResult,
    LisaClusterMapResult,
    MarkovTransitionsResult,
    MGWRResult,
    MoranOverTimeResult,
    MoranPlotResult,
    SandboxResult,
    SigmaConvergenceResult,
    SpacetimeHeatmapResult,
    SpatialDiagnosticsResult,
    SpatialMarkovResult,
    SpatialModelResult,
    TheilDecompositionResult,
    WeightsRobustnessResult,
)
from geometrics.clubs import analyze_convergence_clubs
from geometrics.convergence import (
    analyze_beta_convergence,
    analyze_sigma_convergence,
    growth_cross_section,
)
from geometrics.dependence import (
    explore_lisa_cluster_map,
    explore_moran_over_time,
    explore_moran_plot,
)
from geometrics.distribution_dynamics import (
    analyze_markov_transitions,
    analyze_spatial_markov,
)
from geometrics.gwr import analyze_gwr, analyze_mgwr
from geometrics.maps import explore_choropleth_map
from geometrics.pedagogy import Explainer, explain, list_topics
from geometrics.regional_inequality import (
    analyze_inequality_over_time,
    analyze_theil_decomposition,
)
from geometrics.sandbox import (
    learn_beta_convergence,
    learn_convergence_clubs,
    learn_lisa_clusters,
    learn_markov_chains,
    learn_omitted_spatial_lag,
    learn_sigma_convergence,
    learn_spatial_autocorrelation,
    learn_spatial_markov,
    learn_spatial_spillovers,
    learn_spatial_weights,
    learn_theil_decomposition,
)
from geometrics.spacetime import (
    explore_distribution_over_time,
    explore_spacetime_heatmap,
)
from geometrics.spatial_models import (
    analyze_spatial_diagnostics,
    analyze_spatial_model,
    analyze_spatial_model_by_weights,
)
from geometrics.weights import explore_connectivity_map, make_weights

__version__ = "0.1.3"

__all__ = [
    # ===== EXPLORE =====
    # maps
    "explore_choropleth_map",
    # spatial weights
    "explore_connectivity_map",
    # spatial dependence (ESDA)
    "explore_moran_plot",
    "explore_lisa_cluster_map",
    "explore_moran_over_time",
    # space-time dynamics
    "explore_distribution_over_time",
    "explore_spacetime_heatmap",
    # ===== ANALYZE =====
    # convergence
    "analyze_beta_convergence",
    "analyze_sigma_convergence",
    "analyze_convergence_clubs",
    # spatial econometric models (spreg)
    "analyze_spatial_model",
    "analyze_spatial_diagnostics",
    "analyze_spatial_model_by_weights",
    # distribution dynamics (giddy)
    "analyze_markov_transitions",
    "analyze_spatial_markov",
    # regional inequality (PySAL inequality)
    "analyze_inequality_over_time",
    "analyze_theil_decomposition",
    # local models (mgwr)
    "analyze_gwr",
    "analyze_mgwr",
    # ===== LEARN =====
    # concept sandboxes (simulate from a known DGP; never for user data)
    "learn_spatial_autocorrelation",
    "learn_spatial_weights",
    "learn_lisa_clusters",
    "learn_spatial_spillovers",
    "learn_omitted_spatial_lag",
    "learn_beta_convergence",
    "learn_sigma_convergence",
    "learn_convergence_clubs",
    "learn_markov_chains",
    "learn_spatial_markov",
    "learn_theil_decomposition",
    # ===== UTILITIES =====
    "read_gdf",
    "make_weights",
    "growth_cross_section",
    "set_panel",
    "resolve_panel",
    "set_labels",
    "resolve_label",
    "set_roles",
    "build_data_dict",
    "set_palette",
    "get_palette",
    "explain",
    "list_topics",
    "Explainer",
    # ===== DATA =====
    "data",
    # ===== RESULT TYPES =====
    "ChoroplethMapResult",
    "ConnectivityMapResult",
    "MoranPlotResult",
    "LisaClusterMapResult",
    "MoranOverTimeResult",
    "DistributionOverTimeResult",
    "SpacetimeHeatmapResult",
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
    "SandboxResult",
]
