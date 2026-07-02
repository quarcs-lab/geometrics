"""Building blocks shared by the per-domain interpret modules.

Design rule: interpretations describe *associations*, never causal effects. The word
"causes" and the phrase "effect of" must not appear; a closing note points users to the
``correlation_vs_causation`` explainer.
"""

from __future__ import annotations

__all__ = ["_ASSOC_NOTE", "_MAX_VARS"]

_ASSOC_NOTE = (
    "_These are associations, not causal effects. A causal reading needs a research "
    "design — see `explain('correlation_vs_causation')`._"
)
_MAX_VARS = 6
