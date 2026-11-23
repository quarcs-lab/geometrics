"""Independent, hand-computable tests for the map343 construction."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from geometrics.data._bolivia343 import build_bolivia343, prepare_grid


def test_area_allocation_ledgers_and_missing_semantics():
    grid = gpd.GeoDataFrame(
        {"cell": ["shared", "censor", "empty", "unmapped"]},
        geometry=[
            box(0, 0, 10, 10),
            box(10, 0, 20, 10),
            box(20, 0, 30, 10),
            box(40, 0, 50, 10),
        ],
        crs="EPSG:6933",
    )
    values = pd.DataFrame(
        {
            "cell": grid.cell,
            "year": 2022,
            "gdp_pwt": [160.0, 0.0, 0.0, 12.0],
            "pop_pwt": [8.0, 2.0, 0.0, 1.0],
            "is_cell_censored": [0, 1, 0, 0],
        }
    )
    municipalities = gpd.GeoDataFrame(
        {"geo_id": list("abcde"), "name": list("ABCDE")},
        geometry=[
            box(0, 0, 3, 20),
            box(3, 0, 8, 10),
            box(10, 0, 20, 10),
            box(20, 0, 30, 10),
            box(50, 0, 60, 10),
        ],
        crs="EPSG:6933",
    )
    out = build_bolivia343(grid, values, municipalities)
    panel = out.panel.set_index("geo_id")
    assert list(out.gdf.columns) == ["geo_id", "geometry"]
    assert panel.coverage_status.tolist() == [
        "partial",
        "complete",
        "complete",
        "complete",
        "no_support",
    ]
    np.testing.assert_allclose(panel.coverage_fraction, [0.5, 1.0, 1.0, 1.0, 0.0])
    # 30 and 50 m² out of 80 supported; 20 m² outside both municipalities.
    np.testing.assert_allclose(
        panel.gdp_pwt, [60.0, 100.0, 0.0, 0.0, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        panel.pop_pwt, [3.0, 5.0, 2.0, 0.0, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        panel.gdppc, [20.0, 20.0, 0.0, np.nan, np.nan], equal_nan=True
    )
    assert panel.censor_status.tolist() == ["none", "none", "all", "none", "no_support"]
    assert panel.ln_gdppc.isna().tolist() == [False, False, True, True, True]
    assert (
        panel.loc["c", "censored_area_fraction"]
        == panel.loc["c", "censored_pop_fraction"]
        == 1.0
    )
    assert np.isnan(panel.loc["d", "censored_pop_fraction"])
    assert out.weights.query("cell == 'unmapped'").geo_id.isna().all()
    ledger = out.ledger.set_index("method")
    assert ledger.loc["covered_area", "source_gdp_pwt"] == 172.0
    assert ledger.loc["covered_area", "unassigned_gdp_pwt"] == 12.0
    assert ledger.loc["full_footprint", "unassigned_gdp_pwt"] == pytest.approx(44.0)
    assert ledger.loc["centroid", "unassigned_gdp_pwt"] == 12.0
    assert ledger.loc["full_footprint", "unassigned_pop_pwt"] == pytest.approx(2.6)
    assert (ledger.gdp_residual.abs() < 1e-10).all()
    assert (ledger.pop_residual.abs() < 1e-10).all()


def test_prepared_grid_never_joins_on_nonunique_parent_cell():
    geometry = gpd.GeoDataFrame(
        {"longitude": [-65.0, -64.75], "latitude": [-18.0, -18.0]},
        geometry=[
            box(-65.125, -18.125, -64.875, -17.875),
            box(-64.875, -18.125, -64.625, -17.875),
        ],
        crs="EPSG:4326",
    )
    values = pd.DataFrame(
        {
            "cell_id": [10, 10],
            "subcell_id": [1, 1],
            "subcell_id_0_25": [1, 2],
            "year": [2012, 2012],
            "longitude": geometry.longitude,
            "latitude": geometry.latitude,
        }
    )
    built, panel = prepare_grid(geometry, values)
    assert built.cell.tolist() == ["10.1.1", "10.1.2"]
    assert sorted(built.cell) == sorted(panel.cell)
