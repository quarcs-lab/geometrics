"""Headless smoke tests for the three Streamlit apps.

The suite drives the pages with ``streamlit.testing.v1.AppTest`` on a synthetic
lattice dataset injected through the plain :data:`geometrics.streamlit_app._data
.DATASETS` registry — fully offline, so these run in the default ``make test`` pass.
``st.navigation`` cannot be driven by AppTest, so each test renders the sidebar + one
page function directly (the expdpy pattern).
"""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from geometrics.streamlit_app import (
    _data,
    build_command,
)
from geometrics.streamlit_app._pages import MODULE_OF, selected_specs

pytestmark = pytest.mark.app

_TOY = "Toy — 25 lattice units (test)"
_TOY_SINGLE = "Toy — single period (test)"


def _toy_loader():
    """A 5x5 lattice with a 6-period planted-convergence panel and a factor column."""
    import numpy as np

    import geometrics as gm
    from geometrics.sandbox._dgp import convergence_panel, lattice_gdf

    gdf = lattice_gdf(5, prefix="t", entity="unit")
    ids = list(gdf["unit"])
    df = convergence_panel(
        ids, periods=tuple(range(2000, 2006)), b=0.02, a=0.05, noise_sd=0.005, seed=7
    )
    # A second numeric column (covariate pickers need one besides the outcome) and a
    # three-valued grouping (two values would be typed "logical", not "factor").
    rng = np.random.default_rng(11)
    df["pop"] = rng.uniform(1e4, 1e6, len(df))
    df["zone"] = [("north", "center", "south")[int(u[1:]) // 9] for u in df["unit"]]
    df_dict = gm.build_data_dict(df, entity="unit", time="year")
    df_dict.loc[df_dict["var_name"] == "gdppc", "role"] = "outcome"
    return gdf, df, df_dict


def _toy_single_loader():
    """The same lattice observed in a single period (panel pages must hide)."""
    gdf, df, df_dict = _toy_loader()
    df = df[df["year"] == 2000].reset_index(drop=True)
    return gdf, df, df_dict


@pytest.fixture(autouse=True)
def _toy_datasets(monkeypatch):
    """Replace the bundled-dataset registry with offline synthetic entries."""
    monkeypatch.setattr(
        _data,
        "DATASETS",
        {
            _TOY: {"loader": _toy_loader, "note": "synthetic lattice (offline)"},
            _TOY_SINGLE: {"loader": _toy_single_loader, "note": "single period"},
        },
    )


def _active(name: str = _TOY, **kwargs) -> _data.Active:
    return _data.build_active(
        name, w_method=kwargs.get("w_method", "queen"), w_k=kwargs.get("w_k", 4)
    )


def _page_app(page_func: str, dataset: str = _TOY) -> AppTest:
    """An AppTest that builds the toy Active and renders one page function."""
    script = textwrap.dedent(
        f"""
        from geometrics.streamlit_app import _data
        from geometrics.streamlit_app import _pages_analyze, _pages_explore, _pages_learn

        active = _data.build_active({dataset!r}, w_method="queen", w_k=4)
        module, func = {page_func!r}.split(".")
        page = getattr(
            {{"e": _pages_explore, "a": _pages_analyze, "l": _pages_learn}}[module], func
        )
        page(active) if module != "l" else page()
        """
    )
    return AppTest.from_string(script, default_timeout=120)


# ---------------------------------------------------------------------------
# The page registry and gating
# ---------------------------------------------------------------------------


def test_selected_specs_partition_by_module():
    active = _active()
    urls = {m: [s[2] for s in selected_specs(active, m)] for m in MODULE_OF.values()}
    assert urls["explore"] == [
        "map",
        "connectivity",
        "autocorrelation",
        "moran_time",
        "distributions",
    ]
    assert "convergence" in urls["analyze"] and "inequality" in urls["analyze"]
    assert urls["learn"] == ["sandboxes", "explainers"]
    # No page belongs to two modules.
    all_urls = [u for urls_m in urls.values() for u in urls_m]
    assert len(all_urls) == len(set(all_urls))


def test_panel_pages_hide_on_a_single_period():
    single = _active(_TOY_SINGLE)
    urls = [s[2] for s in selected_specs(single, "explore")]
    assert "moran_time" not in urls and "distributions" not in urls
    assert "map" in urls
    analyze_urls = [s[2] for s in selected_specs(single, "analyze")]
    assert "convergence" not in analyze_urls and "clubs" not in analyze_urls


def test_gwr_gate_requires_enough_units():
    # 25 toy units < the 30-unit floor -> the GWR page hides itself.
    assert "gwr" not in [s[2] for s in selected_specs(_active(), "analyze")]


def test_learn_pages_need_no_dataset():
    assert [s[2] for s in selected_specs(None, "learn")] == ["sandboxes", "explainers"]
    assert selected_specs(None, "explore") == []


# ---------------------------------------------------------------------------
# Explore pages
# ---------------------------------------------------------------------------


def test_page_map_renders_figure_and_interpret():
    at = _page_app("e.page_map").run()
    assert not at.exception
    assert at.get("plotly_chart"), "the choropleth figure should render"
    assert any("class" in m.value.lower() for m in at.markdown)


def test_page_connectivity_renders():
    at = _page_app("e.page_connectivity").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 2  # graph + cardinality histogram


def test_page_autocorrelation_renders_moran_and_lisa():
    at = _page_app("e.page_autocorrelation").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 2
    text = " ".join(m.value for m in at.markdown).lower()
    assert "moran" in text


def test_page_distributions_renders():
    at = _page_app("e.page_distributions").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 2  # ridgeline + heatmap


# ---------------------------------------------------------------------------
# Analyze pages
# ---------------------------------------------------------------------------


def test_page_convergence_renders_beta_and_sigma():
    at = _page_app("a.page_convergence").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown).lower()
    assert "converg" in text
    assert len(at.get("plotly_chart")) >= 2


def test_page_spatial_model_renders_with_diagnostics():
    at = _page_app("a.page_spatial_model").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Recommendation" in text or "recommendation" in text.lower()


def test_page_inequality_renders_trend_and_theil():
    at = _page_app("a.page_inequality").run()
    assert not at.exception
    assert at.get("plotly_chart")
    # The toy factor column drives the Theil section.
    headers = " ".join(h.value for h in at.subheader)
    assert "Theil" in headers


@pytest.mark.dynamics
def test_page_markov_renders_with_giddy():
    pytest.importorskip("giddy")
    at = _page_app("a.page_markov").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown).lower()
    assert "transition" in text or "markov" in text


# ---------------------------------------------------------------------------
# Learn pages
# ---------------------------------------------------------------------------


def test_page_sandboxes_lists_and_runs():
    at = _page_app("l.page_sandboxes").run()
    assert not at.exception
    picker = at.selectbox(key="sandbox_choice")
    assert len(picker.options) in (9, 11)  # 9 without giddy, 11 with
    assert at.get("plotly_chart"), "the default sandbox figure should render"
    assert any("sandbox shows" in m.value for m in at.markdown)


def test_page_sandboxes_slider_changes_output():
    at = _page_app("l.page_sandboxes").run()
    before = " ".join(m.value for m in at.markdown)
    at.slider(key="sa_rho").set_value(0.0).run()
    after = " ".join(m.value for m in at.markdown)
    assert not at.exception
    assert before != after


def test_page_explainers_search_filters():
    at = _page_app("l.page_explainers").run()
    assert not at.exception
    assert len(at.selectbox(key="explainer_topic").options) >= 25
    at.text_input(key="explainer_search").set_value("moran").run()
    filtered = at.selectbox(key="explainer_topic").options
    assert 0 < len(filtered) < 25


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


def test_build_command_flags():
    cmd = build_command("entry.py", {"port": 8601, "launch_browser": False})
    joined = " ".join(cmd)
    assert "streamlit run entry.py" in joined.replace(f"{cmd[0]} -m ", "")
    assert "--server.port 8601" in joined
    assert "--server.headless true" in joined


def test_launcher_returns_command_without_running():
    import geometrics.streamlit_app as apps

    cmd = apps.LearnApp(run=False, port=8765)
    assert cmd[-3].endswith("_run.py") or cmd[4].endswith("_run.py")
    assert "--server.port" in cmd
