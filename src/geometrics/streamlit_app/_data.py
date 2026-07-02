"""Bundled datasets, caching, and the active-dataset context for the apps.

The lean apps work exclusively on the bundled case studies (no upload): the registry
maps a display name to a ``geometrics.data`` loader, and everything downstream hangs
off the frozen :class:`Active` context the sidebar builds. Three caches keep
interactions snappy:

- :func:`load_dataset` (``st.cache_data``) — the ``(gdf, df, df_dict)`` trio with
  labels/panel declared, cached per dataset;
- :func:`get_weights` (``st.cache_resource``) — the spatial weights per
  (dataset, method, k);
- :func:`run_cached` (``st.cache_resource``) — slow estimator calls keyed on
  primitives only (the frames and W are re-derived from the other caches, so no
  DataFrame hashing ever happens).

``DATASETS`` is a plain module dict so the test-suite can monkeypatch a synthetic
entry and drive the whole app offline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd
    from libpysal.weights import W

__all__ = [
    "DATASETS",
    "Active",
    "load_dataset",
    "get_weights",
    "run_cached",
]

# Display name -> loader spec. "loader" is resolved from geometrics.data by name (or a
# callable injected by tests); "note" is the sidebar caption.
DATASETS: dict[str, dict[str, Any]] = {
    "India — 520 districts, night lights 1996-2010": {
        "loader": "load_india",
        "note": "DMSP-OLS satellite nighttime lights (Mendez, Kabiraj & Li).",
    },
    "Bolivia — 112 provinces, GDP pc 2012-2022": {
        "loader": "load_bolivia",
        "note": "PWT-anchored local GDP (Rossi-Hansberg & Zhang; PWT 11.0).",
    },
    "Bolivia — 9 departments, GDP pc 2012-2022": {
        "loader": "load_bolivia_departments",
        "note": "PWT-anchored local GDP at department level.",
    },
    "India — 32 states, 1992 (single year)": {
        "loader": "load_india_states",
        "note": "A small single-period demo — panel pages hide themselves.",
    },
}

WEIGHTS_METHODS = ("queen", "knn")


def _resolve_loader(spec: dict[str, Any]) -> Callable:
    """Return the loader callable for a dataset spec (name in geometrics.data, or a callable)."""
    loader = spec["loader"]
    if callable(loader):
        return loader
    from geometrics import data

    return getattr(data, loader)


@st.cache_data(show_spinner="Downloading the dataset (cached after the first run)…")
def load_dataset(name: str):
    """Load one bundled trio and declare labels + panel; cached per dataset name."""
    import geometrics as gm

    gdf, df, df_dict = _resolve_loader(DATASETS[name])()
    df = gm.set_labels(df, df_dict, set_panel=True)
    return gdf, df, df_dict


@st.cache_resource(show_spinner="Building the spatial weights…")
def get_weights(name: str, method: str, k: int) -> W:
    """Build (and cache) the weights for a dataset under one (method, k) choice."""
    import geometrics as gm

    gdf, _, _ = load_dataset(name)
    if method == "knn":
        return gm.make_weights(gdf, method="knn", k=int(k))
    return gm.make_weights(gdf, method=method)


@st.cache_resource(show_spinner="Estimating… (cached per knob combination)")
def run_cached(
    fn_name: str,
    dataset: str,
    w_method: str,
    w_k: int,
    needs: tuple[str, ...] = ("df",),
    **kwargs: Any,
):
    """Run a geometrics function with cached inputs, keyed on primitives only.

    ``needs`` names the data arguments to inject: ``"df"`` (first positional),
    ``"gdf"`` and/or ``"w"`` (keyword). All remaining ``kwargs`` must be hashable
    primitives — they form the cache key together with the dataset/weights choice.
    """
    import geometrics as gm

    gdf, df, _ = load_dataset(dataset)
    call_kwargs = dict(kwargs)
    if "gdf" in needs:
        call_kwargs["gdf"] = gdf
    if "w" in needs:
        call_kwargs["w"] = get_weights(dataset, w_method, w_k)
    fn = getattr(gm, fn_name)
    return fn(df, **call_kwargs) if "df" in needs else fn(**call_kwargs)


@dataclass(frozen=True)
class Active:
    """Everything a page needs about the currently selected dataset and weights."""

    name: str
    gdf: gpd.GeoDataFrame
    df: pd.DataFrame
    df_dict: pd.DataFrame
    entity: str
    time: str
    periods: tuple[Any, ...]
    numeric_vars: tuple[str, ...]
    factor_vars: tuple[str, ...]
    outcome: str
    labels: dict[str, str] = field(default_factory=dict)
    w_method: str = "queen"
    w_k: int = 6

    @property
    def n_entities(self) -> int:
        """Number of geometry rows (entities on the map)."""
        return len(self.gdf)

    @property
    def is_panel(self) -> bool:
        """At least two periods observed."""
        return len(self.periods) >= 2

    @property
    def w(self) -> W:
        """The cached weights for the current (method, k) choice."""
        return get_weights(self.name, self.w_method, self.w_k)

    def label(self, var: str) -> str:
        """Human label for a variable ('Label (var)') used by the pickers."""
        label = self.labels.get(var)
        return f"{label} ({var})" if label and label != var else var


def build_active(name: str, *, w_method: str, w_k: int) -> Active:
    """Assemble the :class:`Active` context for one dataset selection."""
    import geometrics as gm

    gdf, df, df_dict = load_dataset(name)
    entity, time = gm.resolve_panel(df, None, None)
    periods: tuple[Any, ...] = ()
    if time is not None and time in df.columns:
        periods = tuple(sorted(df[time].dropna().unique().tolist()))

    types = df_dict.set_index("var_name")["type"]
    labels = df_dict.set_index("var_name")["label"].to_dict()
    numeric = tuple(
        v for v in df_dict["var_name"] if types.get(v) == "numeric" and v in df.columns
    )
    factors = tuple(
        v for v in df_dict["var_name"] if types.get(v) == "factor" and v in df.columns
    )
    roles = df_dict.set_index("var_name").get("role")
    outcome = numeric[0] if numeric else ""
    if roles is not None:
        flagged = [v for v in numeric if str(roles.get(v, "")) == "outcome"]
        if flagged:
            outcome = flagged[0]

    return Active(
        name=name,
        gdf=gdf,
        df=df,
        df_dict=df_dict,
        entity=str(entity),
        time=str(time),
        periods=periods,
        numeric_vars=numeric,
        factor_vars=factors,
        outcome=outcome,
        labels=labels,
        w_method=w_method,
        w_k=int(w_k),
    )
