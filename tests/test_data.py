"""Tests for the geometrics.data case-study loaders."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

import geometrics.data as gdata
from geometrics.data import (
    GeometricsDataError,
    _registry,
    load_india,
    load_india_raw,
    load_india_states,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

YEARS = [1996, 1999, 2000, 2004, 2005, 2010]

PANEL_COLUMNS = [
    "statedist",
    "state",
    "district",
    "year",
    "ntl_rural",
    "ntl_urban",
    "ntl_total",
    "ntl_pc_1996",
    "log_ntl_pc_1996",
    "growth_ntl_pc_9610",
    "pop_1996",
    "pop_2001",
    "agri_suitability",
    "rainfall",
    "malaria",
    "temperature",
    "ruggedness",
    "dist_coast",
    "latitude",
    "rural_share",
    "log_pop_density",
    "sc_share",
    "st_share",
    "work_share",
    "literacy_share",
    "higher_edu_share",
    "electricity_share",
    "log_paved_roads",
]

CONTROL_COLUMNS = PANEL_COLUMNS[12:]

STATES_COLUMNS = ["region", "year", "ntl_sum", "pop", "ntl_pc", "log_ntl_pc"]

DICT_HEADER = ["var_name", "var_def", "label", "type", "role", "can_be_na"]
DICT_TYPES = {"entity", "time", "factor", "logical", "numeric"}
DICT_ROLES = {"", "outcome", "covariate", "entity_name"}


# ---------------------------------------------------------------------------
# Offline tests (fixtures served through the _fetch seam)
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_fetch(monkeypatch):
    """Serve the miniature fixtures instead of downloading the real files."""
    mapping = {
        "india520.dta": FIXTURES / "mini520.dta",
        "india520.geojson": FIXTURES / "mini520.geojson",
    }

    def fake_fetch(name: str) -> Path:
        if name not in mapping:
            raise AssertionError(f"unexpected fetch in offline test: {name!r}")
        return mapping[name]

    monkeypatch.setattr(_registry, "_fetch", fake_fetch)


def test_load_india_offline_shapes_and_columns(fixture_fetch):
    gdf, df, df_dict = load_india()

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_dict, pd.DataFrame)

    assert list(gdf.columns) == ["statedist", "geometry"]
    assert len(gdf) == 6
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326

    assert df.shape == (36, len(PANEL_COLUMNS))
    assert list(df.columns) == PANEL_COLUMNS
    assert pd.api.types.is_integer_dtype(df["year"])
    assert sorted(df["year"].unique()) == YEARS
    assert df["statedist"].nunique() == 6
    assert set(gdf["statedist"]) == set(df["statedist"])

    # sorted by statedist then year
    assert df[["statedist", "year"]].equals(
        df[["statedist", "year"]].sort_values(["statedist", "year"], ignore_index=True)
    )


def test_load_india_offline_dict_matches_df(fixture_fetch):
    _, df, df_dict = load_india()
    assert list(df_dict["var_name"]) == list(df.columns)


def test_load_india_offline_values_match_source(fixture_fetch):
    _, df, _ = load_india()
    raw = pd.read_stata(FIXTURES / "mini520.dta")
    raw = raw.sort_values("statedist", ignore_index=True)

    # paper-replication columns are carried verbatim (exact float equality)
    verbatim = {
        "ntl_pc_1996": "light96_rcr_cap",
        "log_ntl_pc_1996": "log_light96_rcr_cap",
        "growth_ntl_pc_9610": "light_growth96_10rcr_cap",
    }
    for year in YEARS:
        sub = df.loc[df["year"] == year].sort_values("statedist", ignore_index=True)
        for new, src in verbatim.items():
            assert (sub[new].to_numpy() == raw[src].to_numpy()).all()
        # melted NTL columns come from the year-specific source columns
        for new, prefix in [
            ("ntl_rural", "r"),
            ("ntl_urban", "u"),
            ("ntl_total", "t"),
        ]:
            src = f"{prefix}{year}_1996_rcr_snd"
            assert (sub[new].to_numpy() == raw[src].to_numpy()).all()


def test_load_india_offline_controls_constant_within_district(fixture_fetch):
    _, df, _ = load_india()
    static = [*CONTROL_COLUMNS, "ntl_pc_1996", "pop_1996", "pop_2001"]
    nunique = df.groupby("statedist")[static].nunique()
    assert nunique.eq(1).all().all()


# ---------------------------------------------------------------------------
# Bundled data dictionary integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "n_rows"), [("india520_dict.csv", 28), ("india32_dict.csv", 6)]
)
def test_bundled_dict_integrity(name, n_rows):
    ref = resources.files("geometrics.data").joinpath(name)
    with resources.as_file(ref) as path:
        dd = pd.read_csv(path, dtype=str, keep_default_na=False)

    assert list(dd.columns) == DICT_HEADER
    assert len(dd) == n_rows
    assert dd["var_name"].is_unique
    assert set(dd["type"]) <= DICT_TYPES
    assert set(dd["role"]) <= DICT_ROLES
    assert set(dd["can_be_na"]) <= {"True", "False"}
    assert (dd["label"] != "").all()
    assert (dd["var_def"] != "").all()


def test_india520_dict_roles():
    ref = resources.files("geometrics.data").joinpath("india520_dict.csv")
    with resources.as_file(ref) as path:
        dd = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("var_name")

    assert dd.loc["statedist", "type"] == "entity"
    assert dd.loc["year", "type"] == "time"
    assert dd.loc["district", "role"] == "entity_name"
    assert dd.loc["growth_ntl_pc_9610", "role"] == "outcome"
    covariates = dd.index[dd["role"] == "covariate"]
    assert list(covariates) == ["log_ntl_pc_1996", *CONTROL_COLUMNS]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_fetch_failure_raises_geometrics_data_error(monkeypatch):
    def boom(name):
        raise OSError("simulated network failure")

    monkeypatch.setattr(_registry._POOCH, "fetch", boom)

    with pytest.raises(GeometricsDataError) as excinfo:
        _registry._fetch("india520.dta")

    message = str(excinfo.value)
    assert _registry.BASE_URL + "india520.dta" in message
    assert str(_registry._POOCH.abspath) in message
    assert "GEOMETRICS_DATA_DIR" in message
    assert isinstance(excinfo.value, RuntimeError)


# ---------------------------------------------------------------------------
# Network tests (real downloads, pinned commit)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_load_india_network():
    gdf, df, df_dict = load_india()

    assert list(gdf.columns) == ["statedist", "geometry"]
    assert len(gdf) == 520
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326

    assert df.shape == (3120, len(PANEL_COLUMNS))
    assert list(df.columns) == PANEL_COLUMNS
    assert pd.api.types.is_integer_dtype(df["year"])
    assert df["statedist"].nunique() == 520

    assert set(gdf["statedist"]) == set(df["statedist"])

    assert list(df_dict["var_name"]) == list(df.columns)
    must_not_be_na = df_dict.loc[~df_dict["can_be_na"], "var_name"]
    for col in must_not_be_na:
        assert df[col].notna().all(), f"unexpected NaN in {col}"


@pytest.mark.network
def test_load_india_states_network():
    gdf, df, df_dict = load_india_states()

    assert list(gdf.columns) == ["region", "geometry"]
    assert len(gdf) == 32
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326

    assert df.shape == (32, len(STATES_COLUMNS))
    assert list(df.columns) == STATES_COLUMNS
    assert (df["year"] == 1992).all()
    assert set(gdf["region"]) == set(df["region"])

    assert list(df_dict["var_name"]) == list(df.columns)
    must_not_be_na = df_dict.loc[~df_dict["can_be_na"], "var_name"]
    for col in must_not_be_na:
        assert df[col].notna().all(), f"unexpected NaN in {col}"


@pytest.mark.network
def test_load_india_raw_network():
    gdf, df = load_india_raw()
    assert gdf.shape == (520, 25)
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326
    assert df.shape == (520, 341)
    assert "statedist" in gdf.columns
    assert "statedist" in df.columns


@pytest.mark.network
def test_clear_cache_roundtrip():
    # populate at least one file, clear, then re-fetch
    _registry._fetch("ntl/india32_ntl_percapita_1992.csv")
    cache_dir = Path(_registry._POOCH.abspath)
    assert cache_dir.exists()
    gdata.clear_cache()
    assert not cache_dir.exists()
    path = _registry._fetch("ntl/india32_ntl_percapita_1992.csv")
    assert path.exists()
