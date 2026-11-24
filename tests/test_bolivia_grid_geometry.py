"""Guard the original quarter-degree coordinates and fixed-parent aggregation."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from geometrics.data import load_bolivia_harmonized
from geometrics.data._bolivia343 import canonical_grid_footprints
from geometrics.data._bolivia_harmonized import rollup_map343


def test_origin_is_southwest_not_center():
    gdf = gpd.GeoDataFrame(
        {"cell": ["g.1.2"]},
        geometry=[box(-65.125, -18.125, -64.875, -17.875)],
        crs=4326,
    )
    panel = pd.DataFrame(
        {
            "cell": ["g.1.2"] * 2,
            "year": [2012, 2013],
            "longitude": [-65, -65],
            "latitude": [-18.0, -18.0],
        }
    )
    out = canonical_grid_footprints(gdf, panel)
    assert out.geometry.iloc[0].bounds == (-65, -18, -64.75, -17.75)
    panel.loc[1, "latitude"] = -17.75
    with pytest.raises(ValueError, match="southwest coordinates"):
        canonical_grid_footprints(gdf, panel)


def test_rollup_sums_levels_preserves_missing_and_recomputes_ratio():
    children = ["M1", "M2", "M3", "M4"]
    gdf = gpd.GeoDataFrame(
        {"geo_id": children},
        geometry=[box(i, 0, i + 1, 1) for i in range(4)],
        crs=4326,
    )
    units = pd.DataFrame(
        {
            "geo_id": [*children, "P1", "P2"],
            "scale": ["map343"] * 4 + ["province"] * 2,
            "name": [*children, "First", "Second"],
            "province_id": ["P1", "P1", "P1", "P2", None, None],
        }
    )
    panel = pd.DataFrame(
        {
            "geo_id": children,
            "year": [2012] * 4,
            "gdp_pwt": [5.0, 9.0, np.nan, np.nan],
            "pop_pwt": [1.0, 3.0, np.nan, np.nan],
            "pop_ine": [10, 20, 30, 40],
            "pop_ine_method": [
                "census_share_interpolated",
                "ine_yearbook_projection",
                "ine_yearbook_projection",
                "ine_yearbook_projection",
            ],
            "pop_census_2012": pd.array([11, 22, 33, 44], dtype="Int64"),
        }
    )
    geo, result = rollup_map343(gdf, panel, units, "adm2")
    assert geo["geo_id"].tolist() == ["P1", "P2"]
    assert geo.geometry.iloc[0].equals(box(0, 0, 3, 1))
    assert result["gdp_pwt"].iloc[0] == 14
    assert result["pop_pwt"].iloc[0] == 4
    assert result["gdppc_pwt"].iloc[0] == 3.5
    assert result["gdppc"].isna().all()
    assert result["gdppc_census_2012"].isna().all()
    assert result["pop_ine"].tolist() == [60, 40]
    assert result["n_population_split_units"].tolist() == [1, 0]
    assert result["n_supported_municipal_units"].tolist() == [2, 0]
    assert result["gdp_pwt"].iloc[1:].isna().all()


@pytest.mark.network
def test_harmonized_public_loader_agrees_across_scales():
    levels = [("adm1", 9), ("adm2", 113), ("adm3", 343)]
    totals = []
    for level, n in levels:
        gdf, df, dictionary = load_bolivia_harmonized(level)
        assert len(gdf) == n
        assert list(df.columns) == dictionary["var_name"].tolist()
        assert len(df) == 11 * n
        assert set(gdf["geo_id"]) == set(df["geo_id"])
        assert df["pop_census_2012"].notna().eq(df["year"].eq(2012)).all()
        assert int(df.loc[df.year.eq(2012), "pop_census_2012"].sum()) == 10_059_856
        np.testing.assert_allclose(df.gdppc, 1e6 * df.gdp_pwt / df.pop_ine)
        totals.append(df.groupby("year")[["gdp_pwt", "pop_pwt", "pop_ine"]].sum())
    for t in totals[1:]:
        np.testing.assert_allclose(t, totals[0], rtol=1e-12)
