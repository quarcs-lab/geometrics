"""Concept-explainer content.

Importing this package registers every shipped topic into the
:mod:`geometrics.pedagogy._registry`. Each submodule calls ``register_topic`` at import time.
"""

from __future__ import annotations

from geometrics.pedagogy._text import (
    convergence,
    correlation,
    dynamics,
    inequality,
    models,
    spatial,
)

__all__ = [
    "convergence",
    "correlation",
    "dynamics",
    "inequality",
    "models",
    "spatial",
]
