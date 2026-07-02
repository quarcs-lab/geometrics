"""Tests for the spatial-dependence vertical (Moran plot, LISA map, Moran over time).

The SAR field fixture plants rho = 0.6 on the lattice's row-standardized queen
weights, so global Moran's I in ``y`` is strongly positive by construction and the
LISA map must find High-High clusters. Note that ``grid_w`` (raw queen on the
0.1-degree lattice) contains one genuine island (``u63``: floating-point edge gaps),
so the exact scatter-slope identity is asserted under ``make_weights`` (which
attaches the island, making every row sum to one).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from geometrics._theme import LISA_COLORS
from geometrics._types import (
    LisaClusterMapResult,
    MoranOverTimeResult,
    MoranPlotResult,
)
from geometrics._validation import GeometricsWarning
from geometrics.dependence import (
    explore_lisa_cluster_map,
    explore_moran_over_time,
    explore_moran_plot,
)
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE
from geometrics.weights import make_weights

CANONICAL = ("High-High", "Low-Low", "Low-High", "High-Low", "Not significant")
QUADRANTS = {"HH", "LH", "LL", "HL"}


@pytest.fixture(scope="module")
def attached_w(grid_gdf):
    """Queen weights with the lattice's one island attached: every row sums to one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometricsWarning)
        return make_weights(grid_gdf, method="queen")


@pytest.fixture(scope="module")
def moran_res(sar_field, grid_gdf, grid_w):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometricsWarning)  # grid_w island advisory
        return explore_moran_plot(
            sar_field, "y", gdf=grid_gdf, w=grid_w, entity="unit", time="year"
        )


@pytest.fixture(scope="module")
def lisa_res(sar_field, grid_gdf, grid_w):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometricsWarning)  # grid_w island advisory
        return explore_lisa_cluster_map(
            sar_field,
            "y",
            gdf=grid_gdf,
            w=grid_w,
            entity="unit",
            time="year",
            tiles=None,
        )


@pytest.fixture(scope="module")
def over_time_res(convergence_panel, grid_gdf, grid_w):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometricsWarning)  # grid_w island advisory
        return explore_moran_over_time(
            convergence_panel,
            "gdppc",
            gdf=grid_gdf,
            w=grid_w,
            entity="unit",
            time="year",
        )


# ---------------------------------------------------------------------------
# explore_moran_plot
# ---------------------------------------------------------------------------


def test_moran_plot_detects_planted_dependence(moran_res):
    assert isinstance(moran_res, MoranPlotResult)
    assert moran_res.moran_i > 0.3  # rho = 0.6 makes clustering strong
    assert moran_res.p_sim <= 0.01
    assert moran_res.z_sim > 2.0
    assert moran_res.expected_i == pytest.approx(-1.0 / 63.0)
    assert moran_res.permutations == 999
    assert moran_res.var == "y"
    assert moran_res.period == 2020
    assert any("latest period" in note for note in moran_res.notes)


def test_moran_plot_slope_equals_moran_i(sar_field, grid_gdf, attached_w):
    res = explore_moran_plot(
        sar_field, "y", gdf=grid_gdf, w=attached_w, entity="unit", time="year"
    )
    slope = float(np.polyfit(res.df["value"], res.df["lag"], 1)[0])
    assert slope == pytest.approx(res.moran_i, abs=1e-6)


def test_moran_plot_quadrants_partition_all_units(moran_res):
    df = moran_res.df
    assert list(df.columns) == ["entity", "value", "lag", "quadrant"]
    assert len(df) == 64
    assert set(df["entity"]) == {f"u{i:02d}" for i in range(64)}
    assert df["quadrant"].notna().all()
    assert set(df["quadrant"]) <= QUADRANTS
    assert int(df["quadrant"].value_counts().sum()) == 64
    # The value column is the z-standardized variable.
    assert df["value"].mean() == pytest.approx(0.0, abs=1e-12)
    assert df["value"].std(ddof=0) == pytest.approx(1.0, abs=1e-12)


def test_moran_plot_figure_surface(moran_res):
    fig = moran_res.fig
    assert isinstance(fig, go.Figure)
    # Four fixed quadrant traces (stable legend, empty quadrants drawn) + OLS line.
    assert [trace.name for trace in fig.data] == ["HH", "LH", "LL", "HL", "OLS fit"]
    by_name = {trace.name: trace for trace in fig.data}
    assert by_name["HH"].marker.color == LISA_COLORS["High-High"]
    assert by_name["LL"].marker.color == LISA_COLORS["Low-Low"]
    hovered = {row[0] for trace in fig.data[:4] for row in (trace.customdata or [])}
    assert hovered == set(moran_res.df["entity"].astype(str))
    for trace in fig.data[:4]:
        assert trace.hovertemplate.endswith("<extra></extra>")
    # Dashed zero reference lines and the stat-box annotation.
    assert len(fig.layout.shapes) == 2
    assert all(shape.line.dash == "dash" for shape in fig.layout.shapes)
    annotations = " ".join(a.text for a in fig.layout.annotations)
    assert "Moran's I" in annotations
    assert "E[I]" in annotations
    assert f"{moran_res.moran_i:.3f}" in annotations


def test_moran_plot_result_is_frozen(moran_res):
    with pytest.raises(dataclasses.FrozenInstanceError):
        moran_res.var = "other"  # type: ignore[misc]


def test_moran_plot_seed_reproducible(sar_field, grid_gdf, attached_w):
    kwargs = dict(gdf=grid_gdf, w=attached_w, entity="unit", time="year")
    res1 = explore_moran_plot(sar_field, "y", **kwargs)
    res2 = explore_moran_plot(sar_field, "y", **kwargs)
    assert res1.p_sim == res2.p_sim
    assert res1.z_sim == res2.z_sim


def test_moran_plot_default_w_warns_and_notes(sar_field, grid_gdf):
    with pytest.warns(GeometricsWarning, match="no spatial weights"):
        res = explore_moran_plot(sar_field, "y", gdf=grid_gdf, entity="unit")
    assert any("defaulted to" in note for note in res.notes)
    assert "queen" in res.w_spec


def test_moran_plot_w_spec_fallback_for_plain_w(moran_res):
    # grid_w carries no geometrics_meta, so the spec is composed on the fly.
    assert "user-supplied W" in moran_res.w_spec
    assert "row-standardized" in moran_res.w_spec
    assert "n=64" in moran_res.w_spec


def test_moran_plot_glance_row(moran_res):
    row = moran_res.glance().iloc[0]
    assert row["moran_i"] == moran_res.moran_i
    assert row["n_obs"] == 64


def test_moran_plot_validation_errors(sar_field, grid_gdf, grid_w):
    with pytest.raises(KeyError, match="nope"):
        explore_moran_plot(sar_field, "nope", gdf=grid_gdf, w=grid_w, entity="unit")
    bad = sar_field.assign(txt="x")
    with pytest.raises(TypeError, match="numeric"):
        explore_moran_plot(bad, "txt", gdf=grid_gdf, w=grid_w, entity="unit")
    with pytest.raises(TypeError, match="GeoDataFrame"):
        explore_moran_plot(sar_field, "y", gdf=pd.DataFrame(), w=grid_w, entity="unit")
    const = sar_field.assign(flat=1.0)
    with pytest.raises(ValueError, match="variance"), warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometricsWarning)
        explore_moran_plot(const, "flat", gdf=grid_gdf, w=grid_w, entity="unit")
    with pytest.raises(ValueError, match="permutations"):
        explore_moran_plot(
            sar_field, "y", gdf=grid_gdf, w=grid_w, entity="unit", permutations=0
        )


# ---------------------------------------------------------------------------
# explore_lisa_cluster_map
# ---------------------------------------------------------------------------


def test_lisa_counts_partition_all_units(lisa_res):
    assert isinstance(lisa_res, LisaClusterMapResult)
    assert set(lisa_res.df["cluster"]) <= set(CANONICAL)
    total = (
        lisa_res.n_hh + lisa_res.n_ll + lisa_res.n_hl + lisa_res.n_lh + lisa_res.n_ns
    )
    assert total == 64
    assert lisa_res.n_hh > 0  # the SAR DGP plants high-value clusters
    assert lisa_res.alpha == 0.05
    assert lisa_res.permutations == 999
    assert lisa_res.period == 2020


def test_lisa_frame_columns_and_masking(lisa_res):
    df = lisa_res.df
    assert list(df.columns) == [
        "entity",
        "value",
        "lag",
        "local_i",
        "quadrant",
        "p_sim",
        "cluster",
    ]
    assert len(df) == 64
    assert set(df["quadrant"]) <= QUADRANTS
    assert df["p_sim"].between(0.0, 1.0).all()
    # The cluster label is the significance-masked quadrant.
    sig = df["p_sim"] < lisa_res.alpha
    assert (df.loc[~sig, "cluster"] == "Not significant").all()
    full = {"HH": "High-High", "LH": "Low-High", "LL": "Low-Low", "HL": "High-Low"}
    assert (df.loc[sig, "cluster"] == df.loc[sig, "quadrant"].map(full)).all()
    assert lisa_res.n_hh == int((df["cluster"] == "High-High").sum())
    assert lisa_res.n_ns == int((df["cluster"] == "Not significant").sum())


def test_lisa_seed_reproducible(sar_field, grid_gdf, grid_w):
    kwargs = dict(gdf=grid_gdf, w=grid_w, entity="unit", time="year", tiles=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometricsWarning)
        res1 = explore_lisa_cluster_map(sar_field, "y", **kwargs)
        res2 = explore_lisa_cluster_map(sar_field, "y", **kwargs)
    assert np.array_equal(res1.df["p_sim"], res2.df["p_sim"])
    assert res1.df["cluster"].tolist() == res2.df["cluster"].tolist()
    assert res1.p_sim_global == res2.p_sim_global


def test_lisa_map_has_fixed_categorical_traces(lisa_res):
    fig = lisa_res.fig
    assert len(fig.data) <= 5
    assert [trace.name for trace in fig.data] == list(CANONICAL)
    by_name = {trace.name: trace for trace in fig.data}
    for label in CANONICAL:
        assert by_name[label].colorscale[0][1] == LISA_COLORS[label]
    assert all(isinstance(trace, go.Choropleth) for trace in fig.data)  # tiles=None
    drawn = [loc for trace in fig.data for loc in trace.locations]
    assert sorted(drawn) == sorted(lisa_res.df["entity"].astype(str))


def test_lisa_map_tiles_backend(sar_field, grid_gdf, attached_w):
    res = explore_lisa_cluster_map(
        sar_field, "y", gdf=grid_gdf, w=attached_w, entity="unit", time="year"
    )
    assert all(isinstance(trace, go.Choroplethmap) for trace in res.fig.data)
    assert res.fig.layout.map.style == "carto-positron"


def test_lisa_scatter_colored_by_cluster(lisa_res):
    fig = lisa_res.fig_scatter
    assert isinstance(fig, go.Figure)
    assert [trace.name for trace in fig.data] == [*CANONICAL, "OLS fit"]
    by_name = {trace.name: trace for trace in fig.data}
    assert by_name["High-High"].marker.color == LISA_COLORS["High-High"]
    n_points = sum(len(trace.x) for trace in fig.data[:5])
    assert n_points == 64
    for trace in fig.data[:5]:
        assert trace.hovertemplate.endswith("<extra></extra>")


def test_lisa_global_test_matches_moran_plot(lisa_res, moran_res):
    # Same variable, weights and seed: the global test is identical.
    assert lisa_res.moran_i == pytest.approx(moran_res.moran_i, abs=1e-12)
    assert lisa_res.p_sim_global == moran_res.p_sim
    assert lisa_res.p_sim_global <= 0.01


def test_lisa_stricter_alpha_masks_more(sar_field, grid_gdf, grid_w, lisa_res):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometricsWarning)
        strict = explore_lisa_cluster_map(
            sar_field,
            "y",
            gdf=grid_gdf,
            w=grid_w,
            entity="unit",
            time="year",
            alpha=0.001,
            tiles=None,
        )
    assert strict.n_ns >= lisa_res.n_ns


def test_lisa_default_w_warns(sar_field, grid_gdf):
    with pytest.warns(GeometricsWarning, match="no spatial weights"):
        res = explore_lisa_cluster_map(
            sar_field, "y", gdf=grid_gdf, entity="unit", tiles=None
        )
    assert any("defaulted to" in note for note in res.notes)


def test_lisa_validation_errors(sar_field, grid_gdf, grid_w):
    with pytest.raises(ValueError, match="alpha"):
        explore_lisa_cluster_map(
            sar_field, "y", gdf=grid_gdf, w=grid_w, entity="unit", alpha=1.5
        )
    with pytest.raises(KeyError, match="nope"):
        explore_lisa_cluster_map(
            sar_field, "nope", gdf=grid_gdf, w=grid_w, entity="unit"
        )


def test_lisa_tidy_returns_frame(lisa_res):
    assert lisa_res.tidy() is lisa_res.df


# ---------------------------------------------------------------------------
# explore_moran_over_time
# ---------------------------------------------------------------------------


def test_moran_over_time_one_row_per_period(over_time_res):
    assert isinstance(over_time_res, MoranOverTimeResult)
    df = over_time_res.df
    assert list(df.columns) == ["period", "moran_i", "z_sim", "p_sim", "n_obs"]
    assert len(df) == 6
    assert df["period"].tolist() == list(range(2000, 2006))
    assert df["p_sim"].between(0.0, 1.0).all()
    assert (df["n_obs"] == 64).all()
    assert df["moran_i"].notna().all()
    assert over_time_res.var == "gdppc"
    assert over_time_res.permutations == 999
    assert "n=64" in over_time_res.w_spec


def test_moran_over_time_figure_encodes_significance(over_time_res):
    fig = over_time_res.fig
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.mode == "lines+markers"
    assert list(trace.x) == [str(p) for p in over_time_res.df["period"]]
    expected = [
        "circle" if p < 0.05 else "circle-open" for p in over_time_res.df["p_sim"]
    ]
    assert list(trace.marker.symbol) == expected
    assert trace.hovertemplate.endswith("<extra></extra>")
    # The E[I] reference line and the marker-key annotation.
    assert len(fig.layout.shapes) == 1
    texts = " ".join(a.text for a in fig.layout.annotations)
    assert "E[I]" in texts
    assert "p (perm)" in texts


def test_moran_over_time_incomplete_entity_dropped_everywhere(
    convergence_panel, grid_gdf, grid_w
):
    holed = convergence_panel[
        ~((convergence_panel["unit"] == "u07") & (convergence_panel["year"] == 2003))
    ]
    with pytest.warns(GeometricsWarning, match="complete"):
        res = explore_moran_over_time(
            holed, "gdppc", gdf=grid_gdf, w=grid_w, entity="unit", time="year"
        )
    assert (res.df["n_obs"] == 63).all()  # one fixed entity set for every period
    assert any("restricted the spatial weights" in note for note in res.notes)


def test_moran_over_time_seed_reproducible(convergence_panel, grid_gdf, attached_w):
    kwargs = dict(gdf=grid_gdf, w=attached_w, entity="unit", time="year")
    res1 = explore_moran_over_time(convergence_panel, "gdppc", **kwargs)
    res2 = explore_moran_over_time(convergence_panel, "gdppc", **kwargs)
    assert res1.df["p_sim"].tolist() == res2.df["p_sim"].tolist()


def test_moran_over_time_default_w_warns(convergence_panel, grid_gdf):
    with pytest.warns(GeometricsWarning, match="no spatial weights"):
        res = explore_moran_over_time(
            convergence_panel, "gdppc", gdf=grid_gdf, entity="unit", time="year"
        )
    assert len(res.df) == 6


def test_moran_over_time_requires_time_id(sar_field, grid_gdf, grid_w):
    with pytest.raises(ValueError, match="time id"):
        explore_moran_over_time(
            sar_field.drop(columns="year"), "y", gdf=grid_gdf, w=grid_w, entity="unit"
        )


def test_moran_over_time_validation_errors(convergence_panel, grid_gdf, grid_w):
    with pytest.raises(KeyError, match="nope"):
        explore_moran_over_time(
            convergence_panel,
            "nope",
            gdf=grid_gdf,
            w=grid_w,
            entity="unit",
            time="year",
        )
    bad = convergence_panel.assign(txt="x")
    with pytest.raises(TypeError, match="numeric"):
        explore_moran_over_time(
            bad, "txt", gdf=grid_gdf, w=grid_w, entity="unit", time="year"
        )


# ---------------------------------------------------------------------------
# interpret()
# ---------------------------------------------------------------------------


def _assert_association_only(text: str) -> None:
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)


def test_interpret_moran_plot_mentions_clustering(moran_res):
    text = moran_res.interpret()
    assert "y" in text
    assert "cluster" in text.lower()  # positive dependence -> clustering direction
    assert "positive" in text.lower()
    assert f"{moran_res.moran_i:.3g}" in text
    _assert_association_only(text)


def test_interpret_lisa_cluster_map(lisa_res):
    text = lisa_res.interpret()
    assert "High-High" in text
    assert "Low-Low" in text
    assert str(lisa_res.n_ns) in text
    assert "outlier" in text.lower()
    _assert_association_only(text)


def test_interpret_moran_over_time(over_time_res):
    text = over_time_res.interpret()
    assert "gdppc" in text
    assert "cluster" in text.lower()
    assert "2000" in text and "2005" in text
    _assert_association_only(text)
