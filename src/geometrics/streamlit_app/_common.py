"""Shared result-rendering helpers for the app pages.

Every geometrics result renders the same way — the figure, the plain-language
``.interpret()`` right under it (the apps' teaching contract), and expanders for the
publication table / tidy frame / concept explainer. Estimation errors degrade to a
friendly ``st.info`` so an ungated edge case never crashes a page.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

import streamlit as st

from geometrics._theme import PLOTLY_CONFIG

__all__ = ["compute", "show_result", "show_sandbox"]


def compute(make: Callable[[], Any]) -> Any:
    """Run ``make()`` and turn estimation errors into a friendly message (returns None)."""
    try:
        return make()
    except ImportError as exc:
        st.info(f"Missing optional dependency: {exc}")
    except (KeyError, TypeError, ValueError) as exc:
        st.info(f"Not available for this selection: {exc}")
    return None


def show_result(
    res: Any,
    *,
    fig: Any = None,
    show_gt: bool = False,
    show_df: bool = False,
) -> None:
    """Render one result: figure -> interpret -> notes -> expanders."""
    figure = fig if fig is not None else getattr(res, "fig", None)
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG)
    with suppress(NotImplementedError):
        st.markdown(res.interpret())
    for note in getattr(res, "notes", ()):  # advisory degradation, surfaced
        st.caption(f"⚠️ {note}")
    if show_gt and getattr(res, "gt", None) is not None:
        with st.expander("📋 Publication table"):
            st.html(res.gt.as_raw_html())
    if show_df and getattr(res, "df", None) is not None:
        with st.expander("🧮 Tidy data behind the figure"):
            st.dataframe(res.df, width="stretch", hide_index=True)
    with suppress(NotImplementedError):
        explainer_md = res.explain().to_markdown()
        with st.expander("❓ What is this? (method explainer)"):
            st.markdown(explainer_md)


def show_sandbox(res: Any) -> None:
    """Render one learn_* sandbox result: figure -> summary -> interpret -> extras."""
    st.plotly_chart(res.fig, width="stretch", config=PLOTLY_CONFIG)
    st.dataframe(res.df, width="stretch", hide_index=True)
    st.markdown(res.interpret())
    for note in getattr(res, "notes", ()):
        st.caption(f"⚠️ {note}")
    st.download_button(
        "Download the simulated data (CSV)",
        res.data.to_csv(index=False).encode(),
        file_name=f"{res.topic}_simulated.csv",
        mime="text/csv",
    )
    with st.expander("❓ What is this? (method explainer)"):
        st.markdown(res.explain().to_markdown())
