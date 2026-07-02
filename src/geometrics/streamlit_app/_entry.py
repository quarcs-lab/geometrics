"""The app body executed on every Streamlit rerun.

``run_app`` is import-safe (no module-level side effects), so it can be driven both by
Streamlit's script runner (via :mod:`geometrics.streamlit_app._run` or the root
``app_*.py`` wrappers) and by ``streamlit.testing.v1.AppTest`` in the test-suite.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from geometrics.streamlit_app._pages import build_pages
from geometrics.streamlit_app._sidebar import render_sidebar

__all__ = ["run_app", "GEOMETRICS_MODULE_ENV"]

#: Environment variable the launchers use to pin the subprocess to one module.
GEOMETRICS_MODULE_ENV = "GEOMETRICS_MODULE"

_TITLES = {
    "explore": "geometrics — Explore regional data",
    "analyze": "geometrics — Analyze convergence & inequality",
    "learn": "geometrics — Learn spatial analysis",
}
_EMOJI_ICON = "🗺️"


def _page_icon() -> Any:
    """Return the packaged lattice favicon for the tab, falling back to an emoji."""
    try:
        from importlib.resources import files
        from io import BytesIO

        from PIL import Image

        data = files("geometrics").joinpath("_assets/favicon.png").read_bytes()
        return Image.open(BytesIO(data))
    except Exception:
        return _EMOJI_ICON


def run_app(module: str | None = None) -> None:
    """Render the sidebar and run the multipage navigation for one module.

    ``module`` is ``"explore"``, ``"analyze"`` or ``"learn"``; ``None`` falls back to
    the ``GEOMETRICS_MODULE`` environment variable (set by the launchers), and if that
    is unset shows every page (the combined navigation).
    """
    if module is None:
        module = os.environ.get(GEOMETRICS_MODULE_ENV) or None

    st.set_page_config(
        page_title=_TITLES.get(module or "", "geometrics"),
        page_icon=_page_icon(),
        layout="wide",
    )
    active = render_sidebar(module)
    st.navigation(build_pages(active, module)).run()
