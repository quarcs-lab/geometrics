"""Plain-language interpretation of result objects.

Each ``interpret_*`` function takes a *duck-typed* result object (it reads ``.df`` /
scalar fields) and returns a Markdown string. Keeping the logic here — rather than in
``geometrics._types`` — keeps the result dataclasses thin and avoids an import cycle,
since this package never imports ``geometrics._types``.

The functions live in per-domain submodules (one per feature module) and are resolved
lazily through module ``__getattr__`` (PEP 562), so
``from geometrics.pedagogy._interpret import interpret_moran_plot`` works while keeping
each domain's implementation independently importable.

Design rule: interpretations describe *associations*, never causal effects. The word
"causes" and the phrase "effect of" must not appear; a closing note points users to the
``correlation_vs_causation`` explainer.
"""

from __future__ import annotations

import importlib
from typing import Any

from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE, _MAX_VARS

__all__ = ["_ASSOC_NOTE", "_MAX_VARS"]

# interpret_* name -> owning per-domain submodule (each feature module's vertical
# implements its own file; this mapping is the single registry).
_SUBMODULES = {
    "interpret_choropleth_map": "_maps",
    "interpret_connectivity_map": "_weights",
    "interpret_moran_plot": "_dependence",
    "interpret_lisa_cluster_map": "_dependence",
    "interpret_moran_over_time": "_dependence",
    "interpret_distribution_over_time": "_spacetime",
    "interpret_spacetime_heatmap": "_spacetime",
    "interpret_beta_convergence": "_convergence",
    "interpret_sigma_convergence": "_convergence",
    "interpret_convergence_clubs": "_convergence",
    "interpret_spatial_model": "_spatial_models",
    "interpret_spatial_diagnostics": "_spatial_models",
    "interpret_weights_robustness": "_spatial_models",
    "interpret_markov_transitions": "_dynamics",
    "interpret_spatial_markov": "_dynamics",
    "interpret_inequality_over_time": "_inequality",
    "interpret_theil_decomposition": "_inequality",
    "interpret_gwr": "_gwr",
    "interpret_mgwr": "_gwr",
}


def __getattr__(name: str) -> Any:
    """Resolve ``interpret_*`` functions from their per-domain submodule."""
    submodule = _SUBMODULES.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{submodule}", __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    """Expose the lazily resolved names alongside the module globals."""
    return sorted(set(globals()) | set(_SUBMODULES))
