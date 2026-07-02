"""The Explore app's pages — maps, weights, ESDA, and space-time views."""

from __future__ import annotations

from typing import Literal, cast

import streamlit as st

from geometrics.streamlit_app._common import compute, show_result
from geometrics.streamlit_app._data import Active, run_cached

__all__ = [
    "page_map",
    "page_connectivity",
    "page_autocorrelation",
    "page_moran_time",
    "page_distributions",
]

_SCHEMES = {
    "Fisher-Jenks": "fisherjenks",
    "Quantiles": "quantiles",
    "Equal interval": "equalinterval",
    "Continuous": None,
}


def _pick_var(active: Active, *, key: str, label: str = "Variable") -> str:
    """Render a labelled numeric-variable picker defaulted to the outcome role."""
    options = list(active.numeric_vars)
    default = options.index(active.outcome) if active.outcome in options else 0
    return st.selectbox(
        label, options, index=default, key=key, format_func=active.label
    )


def _pick_period(active: Active, *, key: str):
    """Render a period picker defaulted to the latest period."""
    options = list(active.periods)
    return st.selectbox(
        f"Period ({active.time})", options, index=len(options) - 1, key=key
    )


def page_map(active: Active) -> None:
    """Classified / animated choropleth of one variable."""
    import geometrics as gm

    st.header("Choropleth map")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        var = _pick_var(active, key="map_var")
    with c2:
        scheme_name = st.selectbox("Scheme", list(_SCHEMES), key="map_scheme")
    with c3:
        k = st.slider("Classes", 3, 9, 5, key="map_k")
    with c4:
        animate = st.checkbox(
            "Animate", key="map_animate", disabled=not active.is_panel
        )

    kwargs = {"gdf": active.gdf, "scheme": _SCHEMES[scheme_name], "k": int(k)}
    if animate:
        kwargs["animate"] = True
    else:
        kwargs["period"] = _pick_period(active, key="map_period")
    res = compute(lambda: gm.explore_choropleth_map(active.df, var, **kwargs))
    if res is not None:
        show_result(res, show_df=True)


def page_connectivity(active: Active) -> None:
    """Draw the weights graph so it can be inspected before it is trusted."""
    import geometrics as gm

    st.header("Connectivity of the spatial weights")
    st.caption(
        "The sidebar's W drives every spatial page — this graph is what it looks "
        "like. Change the method or k there and watch the structure move."
    )
    res = compute(lambda: gm.explore_connectivity_map(active.gdf, w=active.w))
    if res is not None:
        show_result(res, show_df=True)
        st.subheader("Neighbor cardinalities")
        st.plotly_chart(res.fig_hist, width="stretch")


def page_autocorrelation(active: Active) -> None:
    """Global Moran scatterplot + the LISA cluster map."""
    import geometrics as gm

    st.header("Spatial autocorrelation")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        var = _pick_var(active, key="ac_var")
    with c2:
        period = _pick_period(active, key="ac_period")
    with c3:
        permutations = st.selectbox("Permutations", (99, 999), key="ac_perm")
    with c4:
        alpha = st.selectbox("LISA α", (0.10, 0.05, 0.01), index=1, key="ac_alpha")

    st.subheader("Is it clustered? (global Moran)")
    moran = compute(
        lambda: gm.explore_moran_plot(
            active.df,
            var,
            gdf=active.gdf,
            w=active.w,
            period=period,
            permutations=int(permutations),
        )
    )
    if moran is not None:
        show_result(moran)

    st.subheader("Where exactly? (LISA)")
    lisa = compute(
        lambda: gm.explore_lisa_cluster_map(
            active.df,
            var,
            gdf=active.gdf,
            w=active.w,
            period=period,
            permutations=int(permutations),
            alpha=float(alpha),
        )
    )
    if lisa is not None:
        show_result(lisa, show_df=True)


def page_moran_time(active: Active) -> None:
    """Global Moran's I period by period."""
    st.header("Moran's I over time")
    var = _pick_var(active, key="mt_var")
    permutations = st.selectbox("Permutations", (99, 999), key="mt_perm")
    res = compute(
        lambda: run_cached(
            "explore_moran_over_time",
            active.name,
            active.w_method,
            active.w_k,
            needs=("df", "gdf", "w"),
            var=var,
            permutations=int(permutations),
        )
    )
    if res is not None:
        show_result(res, show_df=True)


def page_distributions(active: Active) -> None:
    """Ridgeline of the cross-sectional density + the entity-by-time heatmap."""
    import geometrics as gm

    st.header("The whole distribution over time")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        var = _pick_var(active, key="dist_var")
    with c2:
        kind = cast(
            Literal["ridgeline", "animated"],
            st.selectbox("View", ("ridgeline", "animated"), key="dist_kind"),
        )
    with c3:
        relative = st.checkbox("Relative to the period mean", key="dist_rel")

    res = compute(
        lambda: gm.explore_distribution_over_time(
            active.df, var, kind=kind, relative=relative
        )
    )
    if res is not None:
        show_result(res)

    st.subheader("Every entity, every period")
    sort_by = cast(
        Literal["value", "name", "north_south", "east_west"],
        st.selectbox(
            "Sort rows by",
            ("value", "name", "north_south", "east_west"),
            key="dist_sort",
        ),
    )
    heat = compute(
        lambda: gm.explore_spacetime_heatmap(
            active.df, var, gdf=active.gdf, sort_by=sort_by, relative=relative
        )
    )
    if heat is not None:
        show_result(heat)
