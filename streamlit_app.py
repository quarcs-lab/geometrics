"""Streamlit Community Cloud entry point: a chooser for the three geometrics apps.

geometrics ships three module apps — **Explore**, **Analyze** and **Learn**. Deploy
one of them directly with its dedicated script (``app_explore.py`` /
``app_analyze.py`` / ``app_learn.py``), or point Streamlit Community Cloud at this
file to get a small landing page that lets the user pick a module. You can also pin
this entry to a single module by setting the ``GEOMETRICS_MODULE`` environment
variable (``explore`` / ``analyze`` / ``learn``).

Run locally with::

    streamlit run streamlit_app.py
"""

import os

import streamlit as st

from geometrics.streamlit_app._entry import run_app

_MODULES = {
    "explore": "🗺️ Explore — maps, weights, Moran/LISA, space-time views",
    "analyze": "🧮 Analyze — convergence, spatial models, dynamics, inequality",
    "learn": "📚 Learn — concept sandboxes and explainers",
}

_module = os.environ.get("GEOMETRICS_MODULE") or st.query_params.get("module")

if _module in _MODULES:
    run_app(module=_module)
else:
    st.set_page_config(page_title="geometrics", page_icon="🗺️", layout="centered")
    st.title("geometrics")
    st.write(
        "Explore, analyze and learn regional growth, convergence and inequality — "
        "spatially. Choose a module to open:"
    )
    for key, label in _MODULES.items():
        if st.button(label, use_container_width=True):
            st.query_params["module"] = key
            st.rerun()
    st.caption(
        "Tip: deploy one app directly with `app_explore.py` / `app_analyze.py` / "
        "`app_learn.py`, or set the `GEOMETRICS_MODULE` environment variable."
    )
