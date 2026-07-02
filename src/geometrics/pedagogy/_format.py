"""Markdown-safe helpers for turning numbers into plain-language phrases.

These are deliberately small and deterministic so the interpretation strings can be
snapshot-tested by asserting on stable substrings.
"""

from __future__ import annotations

import math

__all__ = [
    "direction_word",
    "fmt_num",
    "is_significant",
    "sign_word",
    "significance_phrase",
]


def fmt_num(value: float, digits: int = 3) -> str:
    """Format a number compactly, guarding against NaN / infinities.

    Parameters
    ----------
    value
        The number to format.
    digits
        Significant digits (passed to the ``g`` format).

    Returns
    -------
    str
        e.g. ``"6.41"``, or ``"NaN"`` / ``"infinite"`` for non-finite input.
    """
    x = float(value)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "infinite"
    return f"{x:.{digits}g}"


def sign_word(value: float) -> str:
    """Return ``"positive"``, ``"negative"`` or ``"essentially zero"`` for ``value``."""
    x = float(value)
    if math.isnan(x) or x == 0.0:
        return "essentially zero"
    return "positive" if x > 0 else "negative"


def direction_word(value: float) -> str:
    """Return ``"higher"`` / ``"lower"`` / ``"unchanged"`` for the sign of ``value``."""
    x = float(value)
    if math.isnan(x) or x == 0.0:
        return "unchanged"
    return "higher" if x > 0 else "lower"


def is_significant(p_value: float, level: float = 0.05) -> bool:
    """Return ``True`` if ``p_value`` is below ``level`` (default 5%)."""
    p = float(p_value)
    return (not math.isnan(p)) and p < level


def significance_phrase(p_value: float) -> str:
    """Translate a p-value into a conventional significance phrase.

    Parameters
    ----------
    p_value
        Two-sided p-value.

    Returns
    -------
    str
        ``"statistically significant at the 1%/5%/10% level"`` or
        ``"not statistically significant at conventional levels"``.
    """
    p = float(p_value)
    if math.isnan(p):
        return "of undetermined significance"
    if p < 0.01:
        return "statistically significant at the 1% level"
    if p < 0.05:
        return "statistically significant at the 5% level"
    if p < 0.10:
        return "statistically significant at the 10% level"
    return "not statistically significant at conventional levels"
