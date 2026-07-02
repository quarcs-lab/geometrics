"""Knob validation shared by the ``learn_*`` sandboxes.

The sandboxes take scalar knobs rather than DataFrames, so the house validation order
(type errors first, then value errors) reduces to these two checks per knob.
"""

from __future__ import annotations

from numbers import Integral, Real

__all__ = ["check_int", "check_float"]


def check_int(name: str, value: object, *, minimum: int, func: str) -> int:
    """Return ``value`` as ``int`` or raise ``TypeError`` / ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{func}: {name} must be an integer, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{func}: {name} must be >= {minimum}, got {int(value)}")
    return int(value)


def check_float(
    name: str,
    value: object,
    *,
    func: str,
    minimum: float | None = None,
    maximum: float | None = None,
    inclusive: bool = True,
) -> float:
    """Return ``value`` as ``float`` or raise ``TypeError`` / ``ValueError``.

    Bounds are inclusive by default; ``inclusive=False`` makes both strict (used for
    open intervals like ``0 < rho < 1``).
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{func}: {name} must be a number, got {value!r}")
    out = float(value)
    if minimum is not None and (out < minimum if inclusive else out <= minimum):
        op = ">=" if inclusive else ">"
        raise ValueError(f"{func}: {name} must be {op} {minimum:g}, got {out:g}")
    if maximum is not None and (out > maximum if inclusive else out >= maximum):
        op = "<=" if inclusive else "<"
        raise ValueError(f"{func}: {name} must be {op} {maximum:g}, got {out:g}")
    return out
