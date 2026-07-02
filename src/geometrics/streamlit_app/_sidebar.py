"""The lean sidebar: pick a bundled dataset and configure the spatial weights.

No upload, no filters — the apps run on the bundled case studies. The Learn app skips
the sidebar entirely (its pages need no data).
"""

from __future__ import annotations

import streamlit as st

from geometrics.streamlit_app._data import (
    DATASETS,
    WEIGHTS_METHODS,
    Active,
    build_active,
)

__all__ = ["render_sidebar"]


def render_sidebar(module: str | None) -> Active | None:
    """Render the sidebar and return the active dataset context (None for Learn)."""
    if module == "learn":
        with st.sidebar:
            st.markdown("**geometrics — Learn**")
            st.caption(
                "Concept sandboxes simulate their own data — no dataset needed. "
                "Docs: [quarcs-lab.github.io/geometrics]"
                "(https://quarcs-lab.github.io/geometrics/)"
            )
        return None

    with st.sidebar:
        st.markdown("**Dataset**")
        name = st.selectbox(
            "Bundled case study",
            list(DATASETS),
            key="dataset",
            label_visibility="collapsed",
        )
        st.caption(DATASETS[name]["note"])

        st.markdown("**Spatial weights (W)**")
        method = st.selectbox("Method", WEIGHTS_METHODS, key="w_method")
        k = 6
        if method == "knn":
            # One fewer than the entity count is the hard upper bound for knn.
            active_probe = build_active(name, w_method="queen", w_k=6)
            k_max = max(2, min(12, active_probe.n_entities - 1))
            k = st.slider("k neighbors", 2, k_max, min(6, k_max), key="w_k")

        active = build_active(name, w_method=method, w_k=k)
        st.caption(
            f"{active.n_entities} entities x {len(active.periods)} period(s); "
            f"entity `{active.entity}`, time `{active.time}`"
        )
        st.divider()
        st.caption(
            "Docs: [quarcs-lab.github.io/geometrics]"
            "(https://quarcs-lab.github.io/geometrics/) · every figure's reading is "
            "associational, never causal."
        )
    return active
