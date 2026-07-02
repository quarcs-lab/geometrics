"""Teaching sandboxes that *generate* data to make a spatial concept tangible.

Unlike the ``explore_*`` / ``analyze_*`` functions (which summarize *your* data), the
``learn_*`` functions simulate data from a known data-generating process so a learner
can see a concept in action and turn the knobs: spatial autocorrelation and the choice
of weights, LISA cluster recovery, spillovers and the omitted-lag bias, β/σ/club
convergence, Markov dynamics (plain and spatial), and the Theil decomposition. Each
returns a :class:`~geometrics.SandboxResult` whose ``summary`` holds the scalar facts
the demonstration turns on (planted values, estimates, and their gaps), so they are
easy to test and to read back.

The two ``learn_*markov*`` sandboxes require the ``dynamics`` extra
(``pip install "geometrics[dynamics]"``), matching the ``analyze_markov_*`` pair.
"""

from __future__ import annotations

from geometrics.sandbox._convergence import (
    learn_beta_convergence,
    learn_convergence_clubs,
    learn_sigma_convergence,
)
from geometrics.sandbox._dynamics import learn_markov_chains, learn_spatial_markov
from geometrics.sandbox._esda import (
    learn_lisa_clusters,
    learn_spatial_autocorrelation,
    learn_spatial_weights,
)
from geometrics.sandbox._inequality import learn_theil_decomposition
from geometrics.sandbox._models import (
    learn_omitted_spatial_lag,
    learn_spatial_spillovers,
)

__all__ = [
    # ESDA
    "learn_spatial_autocorrelation",
    "learn_spatial_weights",
    "learn_lisa_clusters",
    # Spatial models
    "learn_spatial_spillovers",
    "learn_omitted_spatial_lag",
    # Convergence
    "learn_beta_convergence",
    "learn_sigma_convergence",
    "learn_convergence_clubs",
    # Distribution dynamics (dynamics extra)
    "learn_markov_chains",
    "learn_spatial_markov",
    # Inequality
    "learn_theil_decomposition",
]
