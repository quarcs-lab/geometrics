"""Tests for the Bolivia (BOL-005popAdj-PWTscaled) loaders.

The offline tests serve the loaders from the data files committed under
``datasets/`` in this repository (the same bytes the pinned raw URLs serve),
by monkeypatching the ``_fetch_bolivia`` seam. The ``network`` tests exercise
the real pinned-URL download path.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from geometrics.data import (
    load_bolivia,
    load_bolivia_departments,
    load_bolivia_grid,
    load_bolivia_raw,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "BOL-005popAdj-PWTscaled"

#: Provinces whose grid cells are all censored at the 0_05 threshold: they
#: have boundary polygons but no panel rows.
CENSORED_ADM2 = {"BOL.2.1_2", "BOL.2.8_2", "BOL.2.11_2", "BOL.2.13_2", "BOL.5.16_2"}

DICT_COLUMNS = ["var_name", "var_def", "label", "type", "role", "can_be_na"]
YEARS = list(range(2012, 2023))


@pytest.fixture(autouse=True)
def _offline_fetch(request, monkeypatch):
    """Serve the in-repo datasets/ files instead of downloading (offline tests)."""
    if "network" in request.keywords:
        return
    from geometrics.data import _registry

    def fake_fetch(name: str) -> Path:
        path = DATA_DIR / name
        assert path.exists(), f"fixture file missing: {path}"
        return path

    monkeypatch.setattr(_registry, "_fetch_bolivia", fake_fetch)


def _assert_trio_contract(gdf, df, df_dict, *, entity: str) -> None:
    assert list(gdf.columns) == [entity, "geometry"]
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326
    assert gdf[entity].is_unique
    assert list(df_dict.columns) == DICT_COLUMNS
    assert list(df_dict["var_name"]) == list(df.columns)
    entities = df_dict.loc[df_dict["type"] == "entity", "var_name"].tolist()
    assert entities == [entity]
    assert df_dict.loc[df_dict["type"] == "time", "var_name"].tolist() == ["year"]
    assert df["year"].dtype == "int64"
    assert sorted(df["year"].unique()) == YEARS


def test_load_bolivia_shapes_and_contract():
    gdf, df, df_dict = load_bolivia()
    _assert_trio_contract(gdf, df, df_dict, entity="gid")
    assert len(gdf) == 112
    assert df.shape == (1177, 21)
    counts = df.groupby("gid")["year"].count()
    assert len(counts) == 107 and (counts == 11).all()


def test_load_bolivia_censored_provinces_documented():
    gdf, df, _ = load_bolivia()
    missing = set(gdf["gid"]) - set(df["gid"])
    assert missing == CENSORED_ADM2


def test_load_bolivia_departments_shapes():
    gdf, df, df_dict = load_bolivia_departments()
    _assert_trio_contract(gdf, df, df_dict, entity="gid")
    assert len(gdf) == 9
    assert df.shape == (99, 19)
    assert set(gdf["gid"]) == set(df["gid"])


def test_load_bolivia_grid_shapes_and_key():
    gdf, df, df_dict = load_bolivia_grid()
    _assert_trio_contract(gdf, df, df_dict, entity="cell")
    assert len(gdf) == 1603
    assert df.shape == (17633, 31)
    assert df.columns[0] == "cell"
    assert not df.duplicated(["cell", "year"]).any()
    assert set(gdf["cell"]) == set(df["cell"])
    counts = df.groupby("cell")["year"].count()
    assert (counts == 11).all()


def test_bolivia_values_are_pwt_dollars():
    _, df, _ = load_bolivia_departments()
    assert (df["gdppc"] > 0).all()
    import numpy as np

    assert np.allclose(df["ln_gdppc"], np.log(df["gdppc"]), atol=1e-9)


def test_bolivia_dict_vocabulary():
    for loader in (load_bolivia, load_bolivia_departments, load_bolivia_grid):
        _, _, df_dict = loader()
        assert set(df_dict["type"]) <= {
            "entity",
            "time",
            "factor",
            "logical",
            "numeric",
        }
        roles = {str(r) for r in df_dict["role"].fillna("")}
        assert roles <= {"", "outcome", "covariate", "entity_name", "nan"}


def test_load_bolivia_raw_levels():
    df0, gdf0 = load_bolivia_raw("adm0")
    assert df0.shape[0] == 11 and len(gdf0) == 1
    dfg, gdfg = load_bolivia_raw("grid")
    assert len(gdfg) == 1603 and dfg.shape[0] == 17633
    with pytest.raises(ValueError, match="unknown level"):
        load_bolivia_raw("adm3")


def test_bolivia_set_labels_wires_panel():
    import geometrics as gm
    from geometrics._panel import resolve_panel

    gdf, df, df_dict = load_bolivia_departments()
    df = gm.set_labels(df, df_dict, set_panel=True)
    assert resolve_panel(df, None, None, require_entity=True, require_time=True) == (
        "gid",
        "year",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gm.explore_choropleth_map(df, "ln_gdppc", gdf=gdf, period=2022, k=3)
    assert res.gdf_plotted.shape[0] == 9


@pytest.mark.network
def test_load_bolivia_network_roundtrip():
    gdf, df, df_dict = load_bolivia()
    assert len(gdf) == 112 and df.shape == (1177, 21) and df_dict.shape == (21, 6)


@pytest.mark.network
def test_load_bolivia_grid_network_roundtrip():
    gdf, df, _ = load_bolivia_grid()
    assert len(gdf) == 1603 and df.shape == (17633, 31)
