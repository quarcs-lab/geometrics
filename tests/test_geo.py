"""Tests for the private geometry infrastructure (geometrics._geo).

Known-answer contiguity assertions use a locally built lattice whose cell edges are
exactly representable in binary floating point (step 0.125), so shared boundaries are
exact and queen contiguity is well defined.
"""

from __future__ import annotations

import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

from geometrics._geo import (
    _align_cross_section,
    _align_panel_wide,
    ensure_metric_crs,
    read_gdf,
    resolve_gdf_entity,
)
from geometrics._validation import GeometricsWarning


def _clean_lattice(side: int = 8, step: float = 0.125) -> gpd.GeoDataFrame:
    """A lattice with exactly representable edges (queen contiguity is exact)."""
    cells, ids = [], []
    for row in range(side):
        for col in range(side):
            x0 = 78.0 + col * step
            y0 = 20.0 + row * step
            cells.append(box(x0, y0, x0 + step, y0 + step))
            ids.append(f"u{row * side + col:02d}")
    return gpd.GeoDataFrame({"unit": ids}, geometry=cells, crs="EPSG:4326")


def _queen_w(gdf: gpd.GeoDataFrame):
    from libpysal.weights import Queen

    w = Queen.from_dataframe(
        gdf, ids=list(gdf["unit"]), use_index=False, silence_warnings=True
    )
    w.transform = "r"
    return w


# --- read_gdf --------------------------------------------------------------------


def test_read_gdf_geodataframe_passthrough(grid_gdf):
    out = read_gdf(grid_gdf)
    assert out is not grid_gdf
    assert out.attrs["geometrics_geo"] == {"entity": "unit", "entity_name": None}
    assert "geometrics_geo" not in grid_gdf.attrs  # the source is never mutated
    assert list(out["unit"]) == list(grid_gdf["unit"])


def test_read_gdf_roundtrip_geojson(grid_gdf, tmp_path):
    path = tmp_path / "grid.geojson"
    grid_gdf.to_file(path)
    out = read_gdf(path)
    assert len(out) == len(grid_gdf)
    assert out.crs is not None
    assert out.attrs["geometrics_geo"]["entity"] == "unit"


def test_read_gdf_roundtrip_gpkg(grid_gdf, tmp_path):
    path = tmp_path / "grid.gpkg"
    grid_gdf.to_file(path, driver="GPKG")
    out = read_gdf(path)
    assert len(out) == len(grid_gdf)
    assert out.attrs["geometrics_geo"]["entity"] == "unit"


def test_read_gdf_roundtrip_zipped_shapefile(grid_gdf, tmp_path):
    shp_dir = tmp_path / "shp"
    shp_dir.mkdir()
    grid_gdf.to_file(shp_dir / "grid.shp")
    archive = shutil.make_archive(str(tmp_path / "grid"), "zip", root_dir=shp_dir)
    out = read_gdf(archive)
    assert len(out) == len(grid_gdf)
    assert set(out["unit"]) == set(grid_gdf["unit"])


def test_read_gdf_missing_crs_errors_and_declares(grid_gdf):
    naked = gpd.GeoDataFrame(
        {"unit": list(grid_gdf["unit"])}, geometry=list(grid_gdf.geometry)
    )
    assert naked.crs is None
    with pytest.raises(ValueError, match="crs"):
        read_gdf(naked)
    out = read_gdf(naked, crs="EPSG:4326")
    assert out.crs is not None
    assert out.crs.to_epsg() == 4326


def test_read_gdf_duplicate_ids_error():
    gdf = gpd.GeoDataFrame(
        {"unit": ["a", "a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_gdf(gdf)


def test_read_gdf_entity_resolution_sole_column():
    gdf = gpd.GeoDataFrame(
        {"whatever": ["x", "y"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    out = read_gdf(gdf)
    assert out.attrs["geometrics_geo"]["entity"] == "whatever"


def test_read_gdf_entity_resolution_by_name_hint():
    gdf = gpd.GeoDataFrame(
        {"region": ["x", "y"], "value": [1.0, 2.0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    out = read_gdf(gdf)
    assert out.attrs["geometrics_geo"]["entity"] == "region"


def test_read_gdf_entity_unresolvable_lists_columns():
    gdf = gpd.GeoDataFrame(
        {"foo": ["x", "y"], "bar": [1.0, 2.0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="foo"):
        read_gdf(gdf)


def test_read_gdf_explicit_entity_wins_and_validates():
    gdf = gpd.GeoDataFrame(
        {"region": ["x", "y"], "code": ["1", "2"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    out = read_gdf(gdf, entity="code", entity_name="region")
    assert out.attrs["geometrics_geo"] == {"entity": "code", "entity_name": "region"}
    with pytest.raises(KeyError, match="nope"):
        read_gdf(gdf, entity="nope")
    with pytest.raises(KeyError, match="nope"):
        read_gdf(gdf, entity="code", entity_name="nope")


def test_read_gdf_empty_geometry_names_ids():
    gdf = gpd.GeoDataFrame(
        {"unit": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), Polygon()],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="b"):
        read_gdf(gdf)


def test_read_gdf_repairs_invalid_geometry_with_warning():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])  # self-intersecting
    gdf = gpd.GeoDataFrame(
        {"unit": ["a", "b"]}, geometry=[bowtie, box(2, 0, 3, 1)], crs="EPSG:4326"
    )
    assert not gdf.geometry.is_valid.all()
    with pytest.warns(GeometricsWarning, match="repaired"):
        out = read_gdf(gdf)
    assert out.geometry.is_valid.all()


def test_read_gdf_rejects_bad_inputs(tmp_path):
    with pytest.raises(TypeError, match="GeoDataFrame"):
        read_gdf(42)
    with pytest.raises(ValueError, match="format"):
        read_gdf(tmp_path / "data.csv")
    with pytest.raises(ValueError, match="not found"):
        read_gdf(tmp_path / "missing.geojson")


# --- resolve_gdf_entity ------------------------------------------------------------


def test_resolve_gdf_entity_explicit_attrs_sole(grid_gdf):
    assert resolve_gdf_entity(grid_gdf, "unit") == "unit"
    with pytest.raises(KeyError, match="nope"):
        resolve_gdf_entity(grid_gdf, "nope")
    assert resolve_gdf_entity(grid_gdf) == "unit"  # sole non-geometry column
    tagged = read_gdf(grid_gdf)
    assert resolve_gdf_entity(tagged) == "unit"  # stored attrs


def test_resolve_gdf_entity_unresolvable():
    gdf = gpd.GeoDataFrame(
        {"a": [1], "b": [2]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="entity"):
        resolve_gdf_entity(gdf)


# --- ensure_metric_crs --------------------------------------------------------------


def test_ensure_metric_crs_auto_projects_with_one_time_warning(grid_gdf, monkeypatch):
    monkeypatch.setattr("geometrics._geo._UTM_WARNED", False)
    with pytest.warns(GeometricsWarning, match="UTM"):
        out = ensure_metric_crs(grid_gdf, "auto", func="test")
    assert out.crs.is_projected
    # second call: the advisory is one-time
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", GeometricsWarning)
        out2 = ensure_metric_crs(grid_gdf, "auto", func="test")
    assert out2.crs.is_projected


def test_ensure_metric_crs_none_and_explicit(grid_gdf):
    assert ensure_metric_crs(grid_gdf, None) is grid_gdf
    out = ensure_metric_crs(grid_gdf, "EPSG:7755")
    assert out.crs.to_epsg() == 7755
    projected = ensure_metric_crs(out, "auto")  # already metric: pass through
    assert projected.crs.to_epsg() == 7755


def test_ensure_metric_crs_requires_declared_crs(grid_gdf):
    naked = gpd.GeoDataFrame({"unit": ["a"]}, geometry=[box(0, 0, 1, 1)])
    with pytest.raises(ValueError, match="CRS"):
        ensure_metric_crs(naked, "auto")


# --- _align_cross_section ------------------------------------------------------------


def _two_period_frame(ids: list[str]) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame(
        {
            "unit": ids * 2,
            "year": [2000] * n + [2005] * n,
            "v": np.arange(2 * n, dtype=float),
        }
    )


def test_align_cross_section_gdf_order_and_latest_period(grid_gdf):
    ids = list(grid_gdf["unit"])
    df = _two_period_frame(ids)
    shuffled = df.sample(frac=1.0, random_state=7)  # order must not matter
    cs, w_out, info = _align_cross_section(
        shuffled, grid_gdf, ["v"], entity="unit", time="year", func="test"
    )
    assert list(cs["unit"]) == ids  # rows in gdf order
    assert info["period"] == 2005
    assert any("latest period" in note for note in info["notes"])
    assert info["n"] == 64
    assert info["dropped"] == 0
    assert w_out is None
    # the values are the 2005 values, aligned per unit
    expected = {u: 64.0 + i for i, u in enumerate(ids)}
    assert list(cs["v"]) == [expected[u] for u in ids]


def test_align_cross_section_explicit_and_bad_period(grid_gdf):
    df = _two_period_frame(list(grid_gdf["unit"]))
    _cs, _, info = _align_cross_section(
        df, grid_gdf, ["v"], entity="unit", time="year", period=2000, func="test"
    )
    assert info["period"] == 2000
    assert not any("latest" in n for n in info["notes"])
    with pytest.raises(ValueError, match="2000"):
        _align_cross_section(
            df, grid_gdf, ["v"], entity="unit", time="year", period=1999, func="test"
        )


def test_align_cross_section_duplicates_keep_first(grid_gdf):
    df = _two_period_frame(list(grid_gdf["unit"]))
    dup = pd.concat([df, df.iloc[[64]]], ignore_index=True)  # duplicate u00/2005
    cs, _, info = _align_cross_section(
        dup, grid_gdf, ["v"], entity="unit", time="year", func="test"
    )
    assert len(cs) == 64
    assert any("duplicate" in note for note in info["notes"])


def test_align_cross_section_zero_overlap_raises_with_samples(grid_gdf):
    df = _two_period_frame([f"x{i:02d}" for i in range(64)])
    with pytest.raises(ValueError, match="x00"):
        _align_cross_section(
            df, grid_gdf, ["v"], entity="unit", time="year", func="test"
        )


def test_align_cross_section_string_normalization_retry(grid_gdf):
    ids = [f"{u} " for u in grid_gdf["unit"]]  # e.g. 'u01 ' with trailing space
    df = _two_period_frame(ids)
    with pytest.warns(GeometricsWarning, match="normalization"):
        cs, _, info = _align_cross_section(
            df, grid_gdf, ["v"], entity="unit", time="year", func="test"
        )
    assert info["n"] == 64
    assert list(cs["unit"]) == list(grid_gdf["unit"])


def test_align_cross_section_match_accounting_warns(grid_gdf):
    ids = list(grid_gdf["unit"])
    df = _two_period_frame([*ids[:60], "z1", "z2", "z3", "z4"])
    with pytest.warns(GeometricsWarning, match="unmatched"):
        cs, _, info = _align_cross_section(
            df, grid_gdf, ["v"], entity="unit", time="year", func="test"
        )
    assert info["n"] == 60
    assert list(cs["unit"]) == ids[:60]


def test_align_cross_section_nan_subsets_and_restandardizes_w():
    gdf = _clean_lattice()
    w = _queen_w(gdf)
    df = _two_period_frame(list(gdf["unit"]))
    df.loc[(df["unit"] == "u10") & (df["year"] == 2005), "v"] = np.nan
    with pytest.warns(GeometricsWarning, match="missing"):
        cs, w_out, info = _align_cross_section(
            df, gdf, ["v"], entity="unit", time="year", w=w, func="test"
        )
    assert len(cs) == 63
    assert w_out.n == 63  # w shrank with the dropped row
    assert "u10" not in w_out.id_order
    assert list(w_out.id_order) == list(cs["unit"])
    row_sums = [sum(w_out.weights[i]) for i in w_out.id_order]
    assert np.allclose(row_sums, 1.0)  # still row-standardized
    assert any("restricted" in note for note in info["notes"])
    assert info["dropped"] == 1
    assert w.n == 64  # the original W is untouched


def test_align_cross_section_min_obs_and_missing_column(grid_gdf):
    df = _two_period_frame(list(grid_gdf["unit"]))
    with pytest.raises(KeyError, match="nope"):
        _align_cross_section(
            df, grid_gdf, ["nope"], entity="unit", time="year", func="test"
        )
    df["v"] = np.nan
    with (
        pytest.warns(GeometricsWarning, match="missing"),
        pytest.raises(ValueError, match="at least"),
    ):
        _align_cross_section(
            df, grid_gdf, ["v"], entity="unit", time="year", min_obs=5, func="test"
        )


def test_align_cross_section_no_time_is_pure_cross_section(grid_gdf):
    ids = list(grid_gdf["unit"])
    df = pd.DataFrame({"unit": ids, "v": np.arange(64.0)})
    _cs, _, info = _align_cross_section(df, grid_gdf, ["v"], entity="unit", func="test")
    assert info["period"] is None
    assert info["n"] == 64


# --- _align_panel_wide ---------------------------------------------------------------


def test_align_panel_wide_shape_and_order(convergence_panel, grid_gdf, grid_w):
    values, ids, periods, info = _align_panel_wide(
        convergence_panel,
        grid_gdf,
        "gdppc",
        w=grid_w,
        entity="unit",
        time="year",
        func="test",
    )
    assert values.shape == (64, 6)
    assert ids == list(grid_w.id_order)
    assert periods == [2000, 2001, 2002, 2003, 2004, 2005]
    assert info["n"] == 64
    assert info["n_periods"] == 6
    # spot-check a value against the long panel
    mask = (convergence_panel["unit"] == ids[0]) & (
        convergence_panel["year"] == periods[0]
    )
    expected = convergence_panel.loc[mask, "gdppc"].iloc[0]
    assert values[0, 0] == pytest.approx(expected)


def test_align_panel_wide_unbalanced_raises(convergence_panel, grid_gdf, grid_w):
    broken = convergence_panel.drop(convergence_panel.index[3])
    victim = convergence_panel.iloc[3]["unit"]
    with pytest.raises(ValueError, match="balanced"):
        try:
            _align_panel_wide(
                broken,
                grid_gdf,
                "gdppc",
                w=grid_w,
                entity="unit",
                time="year",
                func="test",
            )
        except ValueError as err:
            assert str(victim) in str(err)
            raise


def test_align_panel_wide_validates_var(convergence_panel, grid_gdf, grid_w):
    with pytest.raises(KeyError, match="nope"):
        _align_panel_wide(
            convergence_panel,
            grid_gdf,
            "nope",
            w=grid_w,
            entity="unit",
            time="year",
            func="test",
        )
    df = convergence_panel.assign(label=lambda d: d["unit"].astype(str))
    with pytest.raises(TypeError, match="numeric"):
        _align_panel_wide(
            df, grid_gdf, "label", w=grid_w, entity="unit", time="year", func="test"
        )


def test_align_panel_wide_duplicates_keep_first(convergence_panel, grid_gdf, grid_w):
    dup = pd.concat([convergence_panel, convergence_panel.iloc[[0]]], ignore_index=True)
    values, _, _, info = _align_panel_wide(
        dup, grid_gdf, "gdppc", w=grid_w, entity="unit", time="year", func="test"
    )
    assert values.shape == (64, 6)
    assert any("duplicate" in note for note in info["notes"])
