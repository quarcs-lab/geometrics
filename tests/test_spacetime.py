"""Tests for the space-time descriptives (``geometrics.spacetime``)."""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from geometrics._panel import set_panel
from geometrics._types import DistributionOverTimeResult, SpacetimeHeatmapResult
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE
from geometrics.spacetime import (
    explore_distribution_over_time,
    explore_spacetime_heatmap,
)

YEARS = list(range(2000, 2006))


# ---------------------------------------------------------------------------
# explore_distribution_over_time
# ---------------------------------------------------------------------------


def test_density_integrates_to_one_per_period(convergence_panel):
    res = explore_distribution_over_time(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    assert isinstance(res, DistributionOverTimeResult)
    assert list(res.df.columns) == ["time", "value", "density"]
    assert sorted(res.df["time"].unique()) == YEARS
    for _, g in res.df.groupby("time"):
        integral = np.trapezoid(g["density"].to_numpy(), g["value"].to_numpy())
        assert integral == pytest.approx(1.0, abs=0.05)


def test_relative_centers_each_period_at_one(convergence_panel):
    res = explore_distribution_over_time(
        convergence_panel, "gdppc", entity="unit", time="year", relative=True
    )
    assert res.relative is True
    for _, g in res.df.groupby("time"):
        v = g["value"].to_numpy()
        d = g["density"].to_numpy()
        mean = np.trapezoid(v * d, v) / np.trapezoid(d, v)
        assert mean == pytest.approx(1.0, abs=0.02)
    # The giddy convention itself: value / period mean has per-period mean 1.
    rel = convergence_panel.groupby("year")["gdppc"].transform(lambda s: s / s.mean())
    per_period = rel.groupby(convergence_panel["year"]).mean().to_numpy()
    np.testing.assert_allclose(per_period, 1.0)
    # A dashed guide marks the period average at 1.
    assert any(s.x0 == 1.0 and s.x1 == 1.0 for s in res.fig.layout.shapes)


def test_ridgeline_has_one_trace_per_period(convergence_panel):
    res = explore_distribution_over_time(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    assert res.kind == "ridgeline"
    assert len(res.fig.data) == 6
    assert all(isinstance(t, go.Scatter) for t in res.fig.data)
    assert {t.name for t in res.fig.data} == {str(y) for y in YEARS}
    for t in res.fig.data:
        assert t.hovertemplate.endswith("<extra></extra>")
    # Newest period on top: baselines (y ticks) increase with the period.
    tickvals = list(res.fig.layout.yaxis.tickvals)
    assert list(res.fig.layout.yaxis.ticktext) == [str(y) for y in YEARS]
    assert all(b > a for a, b in pairwise(tickvals))


def test_animated_builds_frames_and_slider(convergence_panel):
    res = explore_distribution_over_time(
        convergence_panel, "gdppc", entity="unit", time="year", kind="animated"
    )
    assert res.kind == "animated"
    assert len(res.fig.data) == 1
    assert len(res.fig.frames) == 6
    assert [f.name for f in res.fig.frames] == [str(y) for y in YEARS]
    assert len(res.fig.layout.sliders) == 1
    assert len(res.fig.layout.sliders[0].steps) == 6
    labels = [button.label for button in res.fig.layout.updatemenus[0].buttons]
    assert any("Play" in label for label in labels)
    # Fixed axis ranges so the animation is comparable across frames.
    assert res.fig.layout.yaxis.range[0] == 0.0
    assert res.fig.layout.yaxis.range[1] >= res.df["density"].max()


def test_periods_subset(convergence_panel):
    res = explore_distribution_over_time(
        convergence_panel, "gdppc", entity="unit", time="year", periods=[2005, 2000]
    )
    assert sorted(res.df["time"].unique()) == [2000, 2005]
    assert len(res.fig.data) == 2
    with pytest.raises(ValueError, match="1999"):
        explore_distribution_over_time(
            convergence_panel, "gdppc", entity="unit", time="year", periods=[1999]
        )


def test_bandwidth_is_passed_through(convergence_panel):
    kwargs = {"entity": "unit", "time": "year"}
    wide = explore_distribution_over_time(
        convergence_panel, "gdppc", bandwidth=1.0, **kwargs
    )
    narrow = explore_distribution_over_time(
        convergence_panel, "gdppc", bandwidth=0.1, **kwargs
    )
    # A narrower bandwidth produces a spikier (higher-peaked) density.
    assert narrow.df["density"].max() > wide.df["density"].max()
    for _, g in narrow.df.groupby("time"):
        integral = np.trapezoid(g["density"].to_numpy(), g["value"].to_numpy())
        assert integral == pytest.approx(1.0, abs=0.05)


def test_distribution_missing_rows_warn_and_note(convergence_panel):
    df = convergence_panel.copy()
    df.loc[df.index[:3], "gdppc"] = np.nan
    with pytest.warns(UserWarning, match="missing"):
        res = explore_distribution_over_time(df, "gdppc", entity="unit", time="year")
    assert any("missing" in note for note in res.notes)


def test_distribution_validation_errors(convergence_panel):
    with pytest.raises(KeyError, match="nope"):
        explore_distribution_over_time(
            convergence_panel, "nope", entity="unit", time="year"
        )
    bad = convergence_panel.assign(txt="x")
    with pytest.raises(TypeError, match="numeric"):
        explore_distribution_over_time(bad, "txt", entity="unit", time="year")
    with pytest.raises(ValueError, match="kind"):
        explore_distribution_over_time(
            convergence_panel, "gdppc", entity="unit", time="year", kind="violin"
        )
    with pytest.raises(ValueError, match="time"):
        explore_distribution_over_time(convergence_panel, "gdppc", entity="unit")
    constant = pd.DataFrame(
        {
            "unit": list("abcd") * 2,
            "year": [2000] * 4 + [2001] * 4,
            "v": [1.0] * 8,
        }
    )
    with pytest.raises(ValueError, match="distinct"):
        explore_distribution_over_time(constant, "v", entity="unit", time="year")


def test_distribution_result_is_frozen(convergence_panel):
    res = explore_distribution_over_time(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.var = "other"  # type: ignore[misc]


def test_interpret_distribution_mentions_shift_and_spread(convergence_panel):
    res = explore_distribution_over_time(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    text = res.interpret()
    assert "gdppc" in text
    assert "spread" in text
    assert "narrowed" in text  # planted shrinking dispersion
    assert "lower" in text  # planted negative drift at the mean initial level
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)


def test_interpret_distribution_relative_mentions_period_average(convergence_panel):
    res = explore_distribution_over_time(
        convergence_panel, "gdppc", entity="unit", time="year", relative=True
    )
    text = res.interpret()
    assert "1.0" in text
    assert "spread" in text
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)


# ---------------------------------------------------------------------------
# explore_spacetime_heatmap
# ---------------------------------------------------------------------------


def test_heatmap_pivot_shape_and_default_sort(convergence_panel):
    res = explore_spacetime_heatmap(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    assert isinstance(res, SpacetimeHeatmapResult)
    assert res.df.shape == (64, 6)
    assert list(res.df.columns) == YEARS
    assert res.sort_by == "value"
    assert res.relative is False
    # 'value' order: row means descending (highest first).
    means = res.df.mean(axis=1).to_numpy()
    assert all(a >= b for a, b in pairwise(means))
    assert len(res.fig.data) == 1
    trace = res.fig.data[0]
    assert isinstance(trace, go.Heatmap)
    assert trace.hovertemplate.endswith("<extra></extra>")
    assert list(trace.x) == [str(y) for y in YEARS]
    # The first pivot row is drawn at the top of the heatmap.
    assert res.fig.layout.yaxis.autorange == "reversed"


def test_heatmap_sort_north_south(convergence_panel, grid_gdf):
    res = explore_spacetime_heatmap(
        convergence_panel,
        "gdppc",
        entity="unit",
        time="year",
        gdf=grid_gdf,
        sort_by="north_south",
    )
    metric = grid_gdf.to_crs(grid_gdf.estimate_utm_crs())
    northing = dict(zip(metric["unit"], metric.geometry.centroid.y, strict=True))
    ys = [northing[u] for u in res.df.index]
    assert all(a >= b for a, b in pairwise(ys))  # north to south, monotone
    assert ys[0] > ys[-1]
    assert res.df.shape == (64, 6)


def test_heatmap_sort_east_west(convergence_panel, grid_gdf):
    res = explore_spacetime_heatmap(
        convergence_panel,
        "gdppc",
        entity="unit",
        time="year",
        gdf=grid_gdf,
        sort_by="east_west",
    )
    metric = grid_gdf.to_crs(grid_gdf.estimate_utm_crs())
    easting = dict(zip(metric["unit"], metric.geometry.centroid.x, strict=True))
    xs = [easting[u] for u in res.df.index]
    assert all(a <= b for a, b in pairwise(xs))  # west to east, monotone
    assert xs[0] < xs[-1]


def test_heatmap_geographic_sort_requires_gdf(convergence_panel):
    with pytest.raises(ValueError, match="gdf"):
        explore_spacetime_heatmap(
            convergence_panel,
            "gdppc",
            entity="unit",
            time="year",
            sort_by="north_south",
        )
    with pytest.raises(ValueError, match="gdf"):
        explore_spacetime_heatmap(
            convergence_panel, "gdppc", entity="unit", time="year", sort_by="east_west"
        )


def test_heatmap_relative_column_means_are_one(convergence_panel):
    res = explore_spacetime_heatmap(
        convergence_panel, "gdppc", entity="unit", time="year", relative=True
    )
    assert res.relative is True
    np.testing.assert_allclose(res.df.mean(axis=0).to_numpy(), 1.0)


def test_heatmap_sort_by_name_is_alphabetical(convergence_panel):
    res = explore_spacetime_heatmap(
        convergence_panel, "gdppc", entity="unit", time="year", sort_by="name"
    )
    assert list(res.df.index) == sorted(convergence_panel["unit"].unique())


def test_heatmap_y_labels_use_entity_names(convergence_panel):
    df = convergence_panel.copy()
    df["name"] = df["unit"].str.upper()
    set_panel(df, entity="unit", time="year", entity_name="name")
    res = explore_spacetime_heatmap(df, "gdppc")
    assert res.df.shape == (64, 6)
    y = list(res.fig.data[0].y)
    assert "U00 (u00)" in y
    assert all("(" in label for label in y)
    # The pivot itself keeps the raw entity ids.
    assert "u00" in list(res.df.index)


def test_heatmap_duplicates_and_gaps_warn_with_notes(convergence_panel):
    dup = pd.concat([convergence_panel, convergence_panel.iloc[[0]]], ignore_index=True)
    with pytest.warns(UserWarning, match="duplicate"):
        res = explore_spacetime_heatmap(dup, "gdppc", entity="unit", time="year")
    assert res.df.shape == (64, 6)
    assert any("duplicate" in note for note in res.notes)

    gappy = convergence_panel[
        ~((convergence_panel["unit"] == "u00") & (convergence_panel["year"] == 2003))
    ]
    with pytest.warns(UserWarning, match="unbalanced"):
        res = explore_spacetime_heatmap(gappy, "gdppc", entity="unit", time="year")
    assert res.df.shape == (64, 6)
    assert int(res.df.isna().sum().sum()) == 1
    assert any("unbalanced" in note for note in res.notes)


def test_heatmap_validation_errors(convergence_panel):
    with pytest.raises(KeyError, match="nope"):
        explore_spacetime_heatmap(convergence_panel, "nope", entity="unit", time="year")
    bad = convergence_panel.assign(txt="x")
    with pytest.raises(TypeError, match="numeric"):
        explore_spacetime_heatmap(bad, "txt", entity="unit", time="year")
    with pytest.raises(ValueError, match="sort_by"):
        explore_spacetime_heatmap(
            convergence_panel, "gdppc", entity="unit", time="year", sort_by="latitude"
        )
    with pytest.raises(ValueError, match="time"):
        explore_spacetime_heatmap(convergence_panel, "gdppc", entity="unit")
    with pytest.raises(TypeError, match="GeoDataFrame"):
        explore_spacetime_heatmap(
            convergence_panel,
            "gdppc",
            entity="unit",
            time="year",
            gdf=pd.DataFrame(),
            sort_by="north_south",
        )


def test_heatmap_result_is_frozen(convergence_panel):
    res = explore_spacetime_heatmap(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.sort_by = "name"  # type: ignore[misc]


def test_interpret_heatmap(convergence_panel, grid_gdf):
    res = explore_spacetime_heatmap(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    text = res.interpret()
    assert "gdppc" in text
    assert "64" in text
    assert "highest at the top" in text
    assert "persistent" in text  # planted DGP preserves the ranking
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)

    res_ns = explore_spacetime_heatmap(
        convergence_panel,
        "gdppc",
        entity="unit",
        time="year",
        gdf=grid_gdf,
        sort_by="north_south",
    )
    text_ns = res_ns.interpret()
    assert "north" in text_ns
    assert text_ns.endswith(_ASSOC_NOTE)


def test_interpret_heatmap_relative_mentions_period_average(convergence_panel):
    res = explore_spacetime_heatmap(
        convergence_panel, "gdppc", entity="unit", time="year", relative=True
    )
    text = res.interpret()
    assert "1.0" in text
    assert text.endswith(_ASSOC_NOTE)
