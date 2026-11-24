"""Loader contract and integration with the pinned municipal geography."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from geometrics import set_labels
from geometrics.data import (
    _bolivia343,
    _bolivia_population,
    _registry,
    load_bolivia343,
    load_bolivia_grid,
)

DICT_COLUMNS = ["var_name", "var_def", "label", "type", "role", "can_be_na"]


def test_bolivia343_uses_grid_and_boundary_cache(monkeypatch, tmp_path):
    import geometrics.data as data

    path = tmp_path / "map343.geojson"
    map343 = gpd.GeoDataFrame(
        {"geo_id": ["BO-X"], "name": ["Example"]},
        geometry=[box(-65, -16, -64.75, -15.75)],
        crs=4326,
    )
    map343.to_file(path, driver="GeoJSON")
    grid = gpd.GeoDataFrame(
        {"cell": ["1.2.3"]}, geometry=map343.geometry.copy(), crs=4326
    )
    source_panel = pd.DataFrame({"cell": ["1.2.3"], "year": [2012]})
    result = SimpleNamespace(
        gdf=map343[["geo_id", "geometry"]],
        panel=pd.DataFrame(
            {
                "geo_id": ["BO-X"],
                "name": ["Example"],
                "year": [2012],
                "gdp_pwt": [5.0],
                "pop_pwt": [0.002],
                "gdppc": [2500.0],
                "ln_gdppc": [np.log(2500.0)],
                "coverage_fraction": [1.0],
                "coverage_status": ["complete"],
                "n_source_cells": [1],
                "n_censored_cells": [0],
                "censored_area_fraction": [0.0],
                "censored_pop_fraction": [0.0],
                "censor_status": ["none"],
            }
        ),
    )
    calls = []

    def build(actual_grid, actual_panel, actual_map):
        assert actual_grid is grid
        assert actual_panel is source_panel
        assert actual_map["geo_id"].tolist() == ["BO-X"]
        calls.append("built")
        return result

    monkeypatch.setattr(data, "load_bolivia_grid", lambda: (grid, source_panel, None))
    monkeypatch.setattr(_registry, "_fetch_bolivia343_map", lambda: path)
    monkeypatch.setattr(_bolivia343, "build_bolivia343", build)

    def attach(actual):
        assert actual is result.panel
        return actual.assign(
            pop_ine_harmonized=1000,
            pop_ine_census2012=800,
            pop_ine_method="ine_yearbook_projection",
            pop_ine_group="BO-X",
        )

    monkeypatch.setattr(_bolivia_population, "attach_bolivia343_population", attach)
    gdf, panel, dictionary = load_bolivia343()
    assert calls == ["built"]
    assert gdf is result.gdf
    assert list(dictionary["var_name"]) == list(panel.columns)
    assert panel.at[0, "gdppc"] == 5000
    assert panel.at[0, "gdppc_census_2012"] == 6250
    assert panel.at[0, "gdppc_pwt"] == 2500
    assert panel.at[0, "ln_gdppc"] == np.log(5000)
    assert result.panel.at[0, "gdppc"] == 2500
    assert list(dictionary.columns) == DICT_COLUMNS
    assert dictionary.query("type == 'entity'")["var_name"].tolist() == ["geo_id"]
    assert dictionary.query("type == 'time'")["var_name"].tolist() == ["year"]


def test_bolivia343_boundary_fetch_error_reports_source(monkeypatch):
    class FailingCache:
        abspath = Path("/example/empty-cache")

        @staticmethod
        def fetch(name):
            assert name == "BOL343/map343.geojson"
            raise ValueError("hash mismatch")

    monkeypatch.setattr(_registry, "_POOCH_BOL343", FailingCache())
    with pytest.raises(
        _registry.GeometricsDataError, match=r"map343\.geojson"
    ) as error:
        _registry._fetch_bolivia343_map()
    assert "hash mismatch" in str(error.value.__cause__)


@pytest.mark.network
def test_bolivia343_pinned_inputs_and_panel():
    gdf, panel, dictionary = load_bolivia343()
    assert list(gdf.columns) == ["geo_id", "geometry"]
    assert len(gdf) == gdf["geo_id"].nunique() == 343
    assert gdf.crs.to_epsg() == 4326
    assert gdf.geometry.is_valid.all()
    assert len(panel) == 3773
    assert not panel.duplicated(["geo_id", "year"]).any()
    assert panel.groupby("geo_id")["year"].nunique().eq(11).all()
    assert set(panel["geo_id"]) == set(gdf["geo_id"])
    assert sorted(panel["year"].unique()) == list(range(2012, 2023))
    assert list(dictionary["var_name"]) == list(panel.columns)
    set_labels(panel, dictionary, set_panel=True)

    measures = ["gdp_pwt", "pop_pwt"]
    annual = panel.groupby("year")[measures].sum()
    _, source, _ = load_bolivia_grid()
    # The baseline preserves all mass from matched cells; unmatched source
    # cells are excluded and separately accounted for by the builder.
    source_totals = source.groupby("year")[measures].sum()
    assert (annual <= source_totals + 1e-8).all().all()
    assert (annual > 0).all().all()
    positive_pop = panel["pop_pwt"] > 0
    assert np.allclose(
        panel.loc[positive_pop, "gdppc_pwt"],
        panel.loc[positive_pop, "gdp_pwt"] / panel.loc[positive_pop, "pop_pwt"],
    )
    assert panel.loc[~positive_pop, "gdppc_pwt"].isna().all()
    assert panel["pop_ine"].gt(0).all()
    np.testing.assert_allclose(panel.gdppc, 1e6 * panel.gdp_pwt / panel.pop_ine)
    assert panel.pop_census_2012.notna().eq(panel.year.eq(2012)).all()
    p = panel.query("year == 2012")
    np.testing.assert_allclose(
        p.gdppc_census_2012,
        1e6 * p.gdp_pwt / p.pop_census_2012.astype(float),
    )
    assert panel.groupby("year").pop_ine.sum().loc[2019] == 11_512_468
    assert panel.loc[~(panel["gdppc"] > 0), "ln_gdppc"].isna().all()


def test_bundled_population_roster_and_totals():
    p = _bolivia_population.load_bolivia343_population()
    assert p.geo_id.nunique() == 343
    assert len(p) == 3773 and not p.duplicated(["geo_id", "year"]).any()
    assert p.groupby("year").pop_ine_harmonized.sum().tolist() == [
        10356978,
        10521247,
        10685994,
        10851103,
        11016438,
        11181861,
        11347241,
        11512468,
        11677406,
        11841955,
        12006031,
    ]
    assert p.pop_ine_census2012.notna().eq(p.year.eq(2012)).all()
    assert p.pop_ine_census2012.sum() == 10_059_856
    methods = p.query("year == 2012").pop_ine_method.value_counts().to_dict()
    assert methods == {"ine_yearbook_projection": 333, "census_share_interpolated": 10}
    assert (
        p.query("pop_ine_method == 'census_share_interpolated'").pop_ine_group.nunique()
        == 4
    )


def test_attach_population_rejects_unmatched_keys():
    p = _bolivia_population.load_bolivia343_population()[["geo_id", "year"]].copy()
    p.loc[p.index[0], "geo_id"] = "unknown"
    with pytest.raises(ValueError):
        _bolivia_population.attach_bolivia343_population(p)
