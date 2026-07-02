"""Plain-language interpretation for the spatial-weights vertical (connectivity map).

Duck-typed against :class:`geometrics._types.ConnectivityMapResult` (this module never
imports the result classes): it reads the connectivity scalars and returns Markdown.
"""

from __future__ import annotations

from typing import Any

from geometrics.pedagogy._format import fmt_num
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

__all__ = ["interpret_connectivity_map"]


def interpret_connectivity_map(result: Any, *, lang: str = "en") -> str:
    """Interpret the connectivity health of a spatial weights graph.

    Parameters
    ----------
    result
        A connectivity-map result exposing ``n_units``, ``mean_neighbors``,
        ``min_neighbors``, ``max_neighbors``, ``pct_nonzero``, ``n_components``,
        ``islands`` and ``w_spec``.
    lang
        Language code (only ``"en"`` is shipped).

    Returns
    -------
    str
        A Markdown reading of the graph's density, balance and connectedness.
    """
    n = int(result.n_units)
    mean_nb = float(result.mean_neighbors)
    min_nb = int(result.min_neighbors)
    max_nb = int(result.max_neighbors)
    pct = float(result.pct_nonzero)
    components = int(result.n_components)
    islands = tuple(getattr(result, "islands", ()) or ())
    w_spec = str(getattr(result, "w_spec", "the spatial weights"))

    lines = [
        f"The spatial weights graph (**{w_spec}**) links {n:,} units with "
        f"{fmt_num(mean_nb)} neighbors each on average (ranging from {min_nb} to "
        f"{max_nb}); {fmt_num(pct)}% of all possible unit pairs are connected, so "
        "each spatial lag averages over a "
        + ("small, local" if mean_nb <= 8 else "fairly broad")
        + " set of neighbors."
    ]

    if components == 1:
        lines.append(
            "The graph forms a **single connected component** — every unit can reach "
            "every other through a chain of neighbors, which is what global spatial "
            "statistics (Moran's I, spatial models) implicitly assume."
        )
    else:
        lines.append(
            f"The graph splits into **{components} disconnected components**: values "
            "cannot be associated across components through the weights, so global "
            "statistics mix separate sub-graphs. Consider a denser specification "
            "(k-NN or a wider distance band) if a single connected map is intended."
        )

    if islands:
        shown = ", ".join(str(i) for i in islands[:5])
        lines.append(
            f"{len(islands)} unit(s) initially had **no neighbors at all** (islands: "
            f"{shown}). Islands receive an undefined spatial lag, so geometrics "
            "attaches them to their nearest neighbor by default."
        )
    else:
        lines.append(
            "No unit is an island: every unit has at least one neighbor, so spatial "
            "lags are defined everywhere."
        )

    if max_nb >= 3 * max(min_nb, 1):
        lines.append(
            f"Cardinalities are quite uneven (from {min_nb} to {max_nb} neighbors): "
            "densely connected units get smoother, more averaged lags than sparsely "
            "connected ones — worth keeping in mind when comparing local statistics."
        )

    return "\n\n".join([*lines, _ASSOC_NOTE])
