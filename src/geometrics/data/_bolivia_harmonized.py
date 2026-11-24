"""Exact rollups of the modeled Bolivia map343 panel."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

_IDS = {"adm1": ("department", "department_id"), "adm2": ("province", "province_id")}
_LEVELS = ["gdp_pwt", "pop_pwt", "pop_ine", "pop_census_2012"]


def rollup_map343(
    gdf: gpd.GeoDataFrame,
    panel: pd.DataFrame,
    units: pd.DataFrame,
    level: str,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Union children and add levels using the published Observatory hierarchy."""
    if level not in _IDS:
        raise ValueError("A map343 rollup must be adm1 or adm2")
    scope, parent_col = _IDS[level]
    if not gdf["geo_id"].is_unique or panel.duplicated(["geo_id", "year"]).any():
        raise ValueError("Map343 identifiers must be unique by geometry and panel year")
    fine = units.loc[units["scale"].eq("map343"), ["geo_id", parent_col]]
    coarse = units.loc[units["scale"].eq(scope), ["geo_id", "name"]]
    fine_ids = set(fine["geo_id"])
    if (
        not fine["geo_id"].is_unique
        or fine[parent_col].isna().any()
        or not coarse["geo_id"].is_unique
        or coarse["name"].isna().any()
        or fine_ids != set(gdf["geo_id"])
        or fine_ids != set(panel["geo_id"])
        or set(fine[parent_col]) != set(coarse["geo_id"])
    ):
        raise ValueError(
            "Observatory hierarchy and map343 panel or boundary IDs disagree"
        )
    if panel.groupby("year")["geo_id"].count().ne(len(fine)).any():
        raise ValueError(
            "Map343 panel must retain keys even when estimates are missing"
        )
    if panel["gdp_pwt"].isna().ne(panel["pop_pwt"].isna()).any():
        raise ValueError("Map343 GDP and population must share geographic support")

    members = fine.set_index("geo_id")[parent_col]
    geo = gdf[["geo_id", "geometry"]].copy()
    geo["geo_id"] = geo["geo_id"].map(members)
    # Union the original source geometry, not its analytical coverage mask.
    dissolved = geo.dissolve(by="geo_id", as_index=False)
    if not dissolved.geometry.is_valid.all() or dissolved.geometry.is_empty.any():
        raise ValueError("Map343 rollup produced invalid or empty parent geometry")

    x = panel[["geo_id", "year", *_LEVELS]].copy()
    x["population_split"] = panel["pop_ine_method"].eq("census_share_interpolated")
    x["geo_id"] = x["geo_id"].map(members)
    groups = x.groupby(["geo_id", "year"], sort=True)
    out = groups[_LEVELS].sum(min_count=1).reset_index()
    out["n_municipal_units"] = groups.size().to_numpy()
    out["n_supported_municipal_units"] = groups["gdp_pwt"].count().to_numpy()
    out["n_population_split_units"] = groups["population_split"].sum().to_numpy()
    out["name"] = out["geo_id"].map(coarse.set_index("geo_id")["name"])
    complete = out["n_supported_municipal_units"].eq(out["n_municipal_units"])
    gdp_dollars = out["gdp_pwt"].where(complete) * 1_000_000
    out["gdppc"] = gdp_dollars / out["pop_ine"].where(out["pop_ine"].gt(0))
    out["ln_gdppc"] = np.log(out["gdppc"].where(out["gdppc"].gt(0)))
    pop_modeled = out["pop_pwt"].where(out["pop_pwt"].gt(0))
    out["gdppc_pwt"] = out["gdp_pwt"] / pop_modeled
    out["ln_gdppc_pwt"] = np.log(out["gdppc_pwt"].where(out["gdppc_pwt"].gt(0)))
    census = out["pop_census_2012"].astype(float)
    out["gdppc_census_2012"] = gdp_dollars / census.where(census.gt(0))
    np.testing.assert_allclose(
        out.groupby("year")[_LEVELS]
        .sum(min_count=1)
        .to_numpy(dtype=float, na_value=np.nan),
        panel.groupby("year")[_LEVELS]
        .sum(min_count=1)
        .to_numpy(dtype=float, na_value=np.nan),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    cols = [
        "geo_id",
        "name",
        "year",
        "gdp_pwt",
        "pop_ine",
        "gdppc",
        "ln_gdppc",
        "pop_census_2012",
        "gdppc_census_2012",
        "pop_pwt",
        "gdppc_pwt",
        "ln_gdppc_pwt",
        "n_municipal_units",
        "n_supported_municipal_units",
        "n_population_split_units",
    ]
    return dissolved[["geo_id", "geometry"]], out[cols]
