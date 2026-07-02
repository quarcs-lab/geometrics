"""geometrics' pedagogy layer: concept explainers + plain-language result interpretation.

Two complementary pieces:

* **Explainers** — data-independent teaching content (`explain("beta_convergence")`,
  `list_topics()`), reusable in notebooks, docs and the apps.
* **Interpretation** — data-dependent prose attached to result objects via the
  :class:`Interpretable` mixin (``result.interpret()`` / ``result.explain()``).

Importing this package also registers the shipped explainer topics (via :mod:`._text`).
"""

from __future__ import annotations

from geometrics.pedagogy import (
    _text,  # noqa: F401  (import registers the explainer topics)
)
from geometrics.pedagogy._mixin import Interpretable
from geometrics.pedagogy._registry import (
    Explainer,
    explain,
    list_topics,
    register_topic,
)

__all__ = [
    "Explainer",
    "Interpretable",
    "explain",
    "list_topics",
    "register_topic",
]
