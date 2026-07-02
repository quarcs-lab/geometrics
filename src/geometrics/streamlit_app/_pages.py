"""The page registry: specs, gates, and the per-module navigation builder.

A page spec is ``(title, icon, url_path, render_fn, gate)``. Gates keep pages honest:
a page that cannot run on the active dataset (single period, too few units, missing
optional extra) simply disappears from the navigation rather than erroring.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

import streamlit as st

from geometrics.streamlit_app._data import Active

__all__ = ["PAGE_SPECS", "MODULE_OF", "selected_specs", "build_pages"]

PageGate = Callable[[Active], bool] | None


def _is_panel(active: Active) -> bool:
    return active.is_panel


def _periods_at_least(n: int) -> Callable[[Active], bool]:
    return lambda active: len(active.periods) >= n


def _n_at_least(n: int) -> Callable[[Active], bool]:
    return lambda active: active.n_entities >= n


def _has_factor(active: Active) -> bool:
    return len(active.factor_vars) > 0


def _has_dynamics(active: Active) -> bool:
    return importlib.util.find_spec("giddy") is not None and len(active.periods) >= 3


def _clubs_ready(active: Active) -> bool:
    return len(active.periods) >= 5


def _gwr_ready(active: Active) -> bool:
    return active.n_entities >= 30 and active.is_panel


# Filled in lazily to avoid importing the page modules (and geometrics' heavy deps)
# before Streamlit needs them.
def _specs() -> list[tuple]:
    from geometrics.streamlit_app import _pages_analyze as pa
    from geometrics.streamlit_app import _pages_explore as pe
    from geometrics.streamlit_app import _pages_learn as pl

    return [
        # Explore
        ("Choropleth map", "🗺️", "map", pe.page_map, None),
        ("Connectivity (W)", "🕸️", "connectivity", pe.page_connectivity, None),
        (
            "Spatial autocorrelation",
            "📈",
            "autocorrelation",
            pe.page_autocorrelation,
            None,
        ),
        ("Moran over time", "⏱️", "moran_time", pe.page_moran_time, _is_panel),
        (
            "Distributions over time",
            "🌊",
            "distributions",
            pe.page_distributions,
            _is_panel,
        ),
        # Analyze
        ("Convergence (β and σ)", "📉", "convergence", pa.page_convergence, _is_panel),
        ("Convergence clubs", "🧩", "clubs", pa.page_clubs, _clubs_ready),
        ("Spatial model", "🧮", "spatial_model", pa.page_spatial_model, None),
        ("Weights robustness", "⚖️", "by_weights", pa.page_by_weights, None),
        (
            "Distribution dynamics",
            "🔁",
            "markov",
            pa.page_markov,
            _has_dynamics,
        ),
        ("Inequality", "📊", "inequality", pa.page_inequality, _is_panel),
        ("Local models (GWR)", "📍", "gwr", pa.page_gwr, _gwr_ready),
        # Learn
        ("Concept sandboxes", "🧪", "sandboxes", pl.page_sandboxes, None),
        ("Concept explainers", "📚", "explainers", pl.page_explainers, None),
    ]


PAGE_SPECS = _specs  # callable, resolved at navigation-build time

# url_path -> module. The three apps each render only their module's pages; the
# combined navigation (module=None) shows everything.
MODULE_OF: dict[str, str] = {
    "map": "explore",
    "connectivity": "explore",
    "autocorrelation": "explore",
    "moran_time": "explore",
    "distributions": "explore",
    "convergence": "analyze",
    "clubs": "analyze",
    "spatial_model": "analyze",
    "by_weights": "analyze",
    "markov": "analyze",
    "inequality": "analyze",
    "gwr": "analyze",
    "sandboxes": "learn",
    "explainers": "learn",
}


def _admits(gate: PageGate, active: Active | None) -> bool:
    """Whether a gate admits the active context (Learn pages take ``None``)."""
    if gate is None:
        return True
    if active is None:
        return False
    return bool(gate(active))


def selected_specs(active: Active | None, module: str | None = None) -> list[tuple]:
    """Return the page specs available for ``active``, optionally per module."""
    out = []
    for spec in _specs():
        if module is not None and MODULE_OF.get(spec[2]) != module:
            continue
        if MODULE_OF.get(spec[2]) == "learn":
            out.append(spec)  # Learn pages need no dataset
        elif active is not None and _admits(spec[4], active):
            out.append(spec)
    return out


def build_pages(active: Active | None, module: str | None = None) -> list:
    """Return the ``st.Page`` list for the navigation."""

    def _bind(func: Callable, needs_active: bool) -> Callable:
        if not needs_active:
            return func
        return lambda: func(active)

    pages = []
    for title, icon, url, func, _ in selected_specs(active, module):
        needs_active = MODULE_OF.get(url) != "learn"
        pages.append(
            st.Page(_bind(func, needs_active), title=title, icon=icon, url_path=url)
        )
    return pages
