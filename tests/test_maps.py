"""Tests for the shared map builders (``_mapping``) and ``explore_choropleth_map``."""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from geometrics._mapping import (
    categorical_map,
    classified_map,
    continuous_map,
    geojson_interface,
)
from geometrics._theme import LISA_COLORS, MAP_DIVERGING, MAP_SEQUENTIAL
from geometrics._types import ChoroplethMapResult
from geometrics.maps import explore_choropleth_map
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE


@pytest.fixture(scope="module")
def grid_values():
    return np.arange(64, dtype=float)


def _first_colors(fig):
    """Return the first colorscale color of every trace."""
    return [trace.colorscale[0][1] for trace in fig.data]


# ---------------------------------------------------------------------------
# geojson_interface
# ---------------------------------------------------------------------------


def test_geojson_interface_features_carry_entity_ids(grid_gdf):
    fc, ids = geojson_interface(grid_gdf, "unit")
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 64
    assert ids == list(grid_gdf["unit"])
    assert {f["id"] for f in fc["features"]} == set(ids)


@pytest.mark.parametrize("simplify", ["auto", None, 50.0])
def test_geojson_interface_simplify_modes(grid_gdf, simplify):
    fc, ids = geojson_interface(grid_gdf, "unit", simplify=simplify)
    assert len(fc["features"]) == 64
    assert ids == list(grid_gdf["unit"])


def test_geojson_interface_missing_entity_raises(grid_gdf):
    with pytest.raises(KeyError, match="nope"):
        geojson_interface(grid_gdf, "nope")


# ---------------------------------------------------------------------------
# classified_map
# ---------------------------------------------------------------------------


def test_classified_map_one_trace_per_class(grid_gdf, grid_values):
    fig, bins = classified_map(grid_gdf, grid_values, entity="unit", k=5)
    assert len(fig.data) == 5
    assert all(isinstance(trace, go.Choroplethmap) for trace in fig.data)
    assert len(bins) == 5
    assert all(b1 > b0 for b0, b1 in pairwise(bins))
    # Every entity is drawn exactly once across the class traces.
    drawn = [loc for trace in fig.data for loc in trace.locations]
    assert sorted(drawn) == sorted(grid_gdf["unit"])


def test_classified_map_legend_labels_contain_interval_bounds(grid_gdf, grid_values):
    fig, bins = classified_map(grid_gdf, grid_values, entity="unit", k=5)
    for i, trace in enumerate(fig.data):
        assert f"{bins[i]:.4g}" in trace.name  # upper bound
    assert f"{grid_values.min():.4g}" in fig.data[0].name  # bottom lower bound
    assert " - " in fig.data[0].name


def test_classified_map_tiles_backend_layout(grid_gdf, grid_values):
    fig, _ = classified_map(
        grid_gdf, grid_values, entity="unit", tiles="carto-positron"
    )
    assert fig.layout.map.style == "carto-positron"
    assert fig.layout.map.zoom is not None
    assert fig.layout.map.center.lon == pytest.approx(78.4)


def test_classified_map_vector_backend(grid_gdf, grid_values):
    fig, bins = classified_map(grid_gdf, grid_values, entity="unit", tiles=None)
    assert all(isinstance(trace, go.Choropleth) for trace in fig.data)
    assert fig.layout.geo.visible is False
    assert fig.layout.geo.fitbounds == "locations"
    assert len(bins) == 5


def test_classified_map_user_defined_bins(grid_gdf, grid_values):
    fig, bins = classified_map(
        grid_gdf, grid_values, entity="unit", bins=[10.0, 40.0, 63.0], tiles=None
    )
    assert bins == (10.0, 40.0, 63.0)
    assert len(fig.data) == 3


def test_classified_map_scheme_none_is_continuous(grid_gdf, grid_values):
    fig, bins = classified_map(
        grid_gdf, grid_values, entity="unit", scheme=None, tiles=None
    )
    assert bins == ()
    assert len(fig.data) == 1
    assert fig.data[0].colorscale[0][1] == MAP_SEQUENTIAL[0][1]
    assert fig.data[0].showscale is not False  # colorbar shown


def test_classified_map_hover_names_and_customdata(grid_gdf, grid_values):
    names = {"u00": "Alpha (u00)"}
    fig, _ = classified_map(
        grid_gdf, grid_values, entity="unit", hover_names=names, tiles=None
    )
    trace = next(t for t in fig.data if "u00" in t.locations)
    row = list(trace.customdata[list(trace.locations).index("u00")])
    assert row[0] == "Alpha (u00)"
    assert row[1] == 0.0
    assert trace.hovertemplate.endswith("<extra></extra>")


def test_classified_map_rejects_bad_values(grid_gdf, grid_values):
    with pytest.raises(ValueError, match="missing"):
        classified_map(
            grid_gdf, np.r_[grid_values[:-1], np.nan], entity="unit", tiles=None
        )
    with pytest.raises(ValueError, match="63"):
        classified_map(grid_gdf, grid_values[:-1], entity="unit", tiles=None)


# ---------------------------------------------------------------------------
# categorical_map
# ---------------------------------------------------------------------------


def test_categorical_map_fixed_colors_and_order(grid_gdf):
    labels = ["High-High" if i < 20 else "Not significant" for i in range(64)]
    fig = categorical_map(
        grid_gdf,
        labels,
        entity="unit",
        colors=LISA_COLORS,
        category_order=list(LISA_COLORS),
        tiles=None,
    )
    assert [trace.name for trace in fig.data] == list(LISA_COLORS)
    by_name = {trace.name: trace for trace in fig.data}
    assert by_name["High-High"].colorscale[0][1] == LISA_COLORS["High-High"]
    assert len(by_name["High-High"].locations) == 20
    assert len(by_name["Low-Low"].locations) == 0  # empty category still drawn
    assert all(isinstance(trace, go.Choropleth) for trace in fig.data)


def test_categorical_map_default_order_and_tiles(grid_gdf):
    labels = ["b"] * 32 + ["a"] * 32
    fig = categorical_map(
        grid_gdf, labels, entity="unit", colors={"a": "#111111", "b": "#222222"}
    )
    assert [trace.name for trace in fig.data] == ["b", "a"]  # appearance order
    assert all(isinstance(trace, go.Choroplethmap) for trace in fig.data)
    row = list(fig.data[0].customdata[0])
    assert row == ["u00", "b"]


# ---------------------------------------------------------------------------
# continuous_map
# ---------------------------------------------------------------------------


def test_continuous_map_mask_draws_grey_layer(grid_gdf, grid_values):
    mask = np.zeros(64, dtype=bool)
    mask[:10] = True
    fig = continuous_map(grid_gdf, grid_values, entity="unit", mask=mask)
    assert len(fig.data) == 2
    main, grey = fig.data
    assert len(main.locations) == 54
    assert grey.name == "Not significant"
    assert grey.colorscale[0][1] == "#d3d3d3"
    assert grey.showlegend is True
    assert list(grey.customdata[0])[1] == "Not significant"


def test_continuous_map_diverging_scale_and_midpoint(grid_gdf, grid_values):
    fig = continuous_map(
        grid_gdf, grid_values - 30.0, entity="unit", diverging=True, midpoint=0.0
    )
    trace = fig.data[0]
    assert trace.zmid == 0.0
    assert [c for _, c in trace.colorscale] == [c for _, c in MAP_DIVERGING]


def test_continuous_map_default_sequential(grid_gdf, grid_values):
    fig = continuous_map(grid_gdf, grid_values, entity="unit")
    assert isinstance(fig.data[0], go.Choropleth)  # tiles=None default
    assert [c for _, c in fig.data[0].colorscale] == [c for _, c in MAP_SEQUENTIAL]


# ---------------------------------------------------------------------------
# explore_choropleth_map
# ---------------------------------------------------------------------------


def test_explore_defaults_to_latest_period_with_note(convergence_panel, grid_gdf):
    res = explore_choropleth_map(
        convergence_panel, "gdppc", gdf=grid_gdf, entity="unit", time="year", tiles=None
    )
    assert isinstance(res, ChoroplethMapResult)
    assert res.period == 2005
    assert any("period" in note.lower() for note in res.notes)
    assert res.animated is False
    assert res.scheme == "fisherjenks"
    assert res.k == 5
    assert isinstance(res.bins, tuple)
    assert len(res.bins) == 5
    assert all(isinstance(b, float) for b in res.bins)
    assert all(b1 > b0 for b0, b1 in pairwise(res.bins))


def test_explore_result_frames_are_typed(convergence_panel, grid_gdf):
    res = explore_choropleth_map(
        convergence_panel, "gdppc", gdf=grid_gdf, entity="unit", time="year", tiles=None
    )
    assert isinstance(res.df, pd.DataFrame)
    assert list(res.df.columns) == ["unit", "gdppc", "class"]
    assert len(res.df) == 64
    assert res.df["class"].notna().all()
    assert isinstance(res.gdf_plotted, gpd.GeoDataFrame)
    assert res.gdf_plotted.crs.to_epsg() == 4326
    assert {"unit", "gdppc", "class"} <= set(res.gdf_plotted.columns)
    assert isinstance(res.fig, go.Figure)
    assert all(isinstance(trace, go.Choropleth) for trace in res.fig.data)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.var = "other"  # type: ignore[misc]


def test_explore_customdata_carries_entity_ids(convergence_panel, grid_gdf):
    res = explore_choropleth_map(
        convergence_panel, "gdppc", gdf=grid_gdf, entity="unit", time="year", tiles=None
    )
    hovered = {row[0] for trace in res.fig.data for row in (trace.customdata or [])}
    assert hovered == set(convergence_panel["unit"].astype(str))
    for trace in res.fig.data:
        assert trace.hovertemplate.endswith("<extra></extra>")


def test_explore_explicit_period_and_tiles(convergence_panel, grid_gdf):
    res = explore_choropleth_map(
        convergence_panel,
        "gdppc",
        gdf=grid_gdf,
        entity="unit",
        time="year",
        period=2000,
    )
    assert res.period == 2000
    assert all(isinstance(trace, go.Choroplethmap) for trace in res.fig.data)
    assert res.fig.layout.map.style == "carto-positron"


def test_explore_animate_builds_frames_and_slider(convergence_panel, grid_gdf):
    res = explore_choropleth_map(
        convergence_panel,
        "gdppc",
        gdf=grid_gdf,
        entity="unit",
        time="year",
        animate=True,
        tiles=None,
    )
    assert res.animated is True
    assert res.period == 2000  # first period drawn
    assert len(res.fig.frames) == 6
    assert [frame.name for frame in res.fig.frames] == [
        str(y) for y in range(2000, 2006)
    ]
    assert len(res.fig.layout.sliders) == 1
    assert len(res.fig.layout.sliders[0].steps) == 6
    assert len(res.fig.layout.updatemenus) == 1
    labels = [button.label for button in res.fig.layout.updatemenus[0].buttons]
    assert any("Play" in label for label in labels)
    # Pooled fixed bins: every frame carries one trace per class with stable names.
    assert len(res.bins) == 5
    for frame in res.fig.frames:
        assert len(frame.data) == 5
        assert [t.name for t in frame.data] == [t.name for t in res.fig.data]
    # The tidy frame stacks all periods.
    assert list(res.df.columns) == ["unit", "year", "gdppc", "class"]
    assert res.df["year"].nunique() == 6
    assert len(res.df) == 64 * 6


def test_explore_animate_ignores_period_with_warning(convergence_panel, grid_gdf):
    with pytest.warns(UserWarning, match="ignored"):
        res = explore_choropleth_map(
            convergence_panel,
            "gdppc",
            gdf=grid_gdf,
            entity="unit",
            time="year",
            period=2003,
            animate=True,
            tiles=None,
        )
    assert res.period == 2000
    assert any("ignored" in note for note in res.notes)


@pytest.mark.parametrize("simplify", ["auto", None])
def test_explore_simplify_modes(convergence_panel, grid_gdf, simplify):
    res = explore_choropleth_map(
        convergence_panel,
        "gdppc",
        gdf=grid_gdf,
        entity="unit",
        time="year",
        simplify=simplify,
        tiles=None,
    )
    assert len(res.fig.data) == 5
    assert len(res.gdf_plotted) == 64


def test_explore_hover_appends_customdata(convergence_panel, grid_gdf):
    df = convergence_panel.assign(rank=convergence_panel["gdppc"].rank())
    res = explore_choropleth_map(
        df, "gdppc", gdf=grid_gdf, entity="unit", time="year", hover="rank", tiles=None
    )
    trace = next(t for t in res.fig.data if len(t.locations))
    assert len(trace.customdata[0]) == 3
    assert "rank: %{customdata[2]}" in trace.hovertemplate
    assert trace.hovertemplate.endswith("<extra></extra>")


def test_explore_validation_errors(convergence_panel, grid_gdf):
    with pytest.raises(KeyError, match="nope"):
        explore_choropleth_map(
            convergence_panel, "nope", gdf=grid_gdf, entity="unit", time="year"
        )
    bad = convergence_panel.assign(txt="x")
    with pytest.raises(TypeError, match="numeric"):
        explore_choropleth_map(bad, "txt", gdf=grid_gdf, entity="unit", time="year")
    with pytest.raises(TypeError, match="GeoDataFrame"):
        explore_choropleth_map(
            convergence_panel, "gdppc", gdf=pd.DataFrame(), entity="unit"
        )
    cross_section = convergence_panel[convergence_panel["year"] == 2005]
    with pytest.raises(ValueError, match="time"):
        explore_choropleth_map(
            cross_section, "gdppc", gdf=grid_gdf, entity="unit", animate=True
        )


def test_explore_scheme_none_continuous_result(convergence_panel, grid_gdf):
    res = explore_choropleth_map(
        convergence_panel,
        "gdppc",
        gdf=grid_gdf,
        entity="unit",
        time="year",
        scheme=None,
        tiles=None,
    )
    assert res.bins == ()
    assert res.scheme is None
    assert len(res.fig.data) == 1
    assert res.df["class"].isna().all()


def test_explore_gdf_entity_named_differently(convergence_panel, grid_gdf):
    renamed = grid_gdf.rename(columns={"unit": "region"})
    res = explore_choropleth_map(
        convergence_panel,
        "gdppc",
        gdf=renamed,
        entity="unit",
        time="year",
        tiles=None,
    )
    assert list(res.df.columns) == ["unit", "gdppc", "class"]
    assert len(res.df) == 64
    assert "region" in res.gdf_plotted.columns


def test_interpret_choropleth_map(convergence_panel, grid_gdf):
    res = explore_choropleth_map(
        convergence_panel, "gdppc", gdf=grid_gdf, entity="unit", time="year", tiles=None
    )
    text = res.interpret()
    assert "gdppc" in text
    assert "2005" in text
    assert "fisherjenks" in text
    assert f"{res.bins[0]:.3g}"[:3] in text  # bottom class upper bound mentioned
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)


def test_interpret_animated_mentions_pooling(convergence_panel, grid_gdf):
    res = explore_choropleth_map(
        convergence_panel,
        "gdppc",
        gdf=grid_gdf,
        entity="unit",
        time="year",
        animate=True,
        tiles=None,
    )
    text = res.interpret()
    assert "pooled" in text
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)
