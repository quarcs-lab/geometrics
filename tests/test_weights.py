"""Tests for geometrics.weights: construction, defaults, and the connectivity map.

Known-answer contiguity assertions use a locally built lattice whose cell edges are
exactly representable in binary floating point (step 0.125), so shared boundaries are
exact: on an 8x8 queen lattice corners have 3 neighbors, edges 5, interior cells 8,
giving mean (4*3 + 24*5 + 36*8) / 64 = 6.5625.
"""

from __future__ import annotations

import dataclasses

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from geometrics._types import ConnectivityMapResult
from geometrics._validation import GeometricsWarning
from geometrics.weights import _default_weights, explore_connectivity_map, make_weights

SIDE = 8
STEP = 0.125  # exactly representable: lattice edges match to the last bit


@pytest.fixture(scope="module")
def lattice():
    """8x8 lattice with exact shared boundaries, entity ids u00..u63."""
    cells, ids = [], []
    for row in range(SIDE):
        for col in range(SIDE):
            x0 = 78.0 + col * STEP
            y0 = 20.0 + row * STEP
            cells.append(box(x0, y0, x0 + STEP, y0 + STEP))
            ids.append(f"u{row * SIDE + col:02d}")
    return gpd.GeoDataFrame({"unit": ids}, geometry=cells, crs="EPSG:4326")


def _corner_edge_interior(side: int = SIDE):
    corners, edges, interior = [], [], []
    for row in range(side):
        for col in range(side):
            uid = f"u{row * side + col:02d}"
            on_row = row in (0, side - 1)
            on_col = col in (0, side - 1)
            if on_row and on_col:
                corners.append(uid)
            elif on_row or on_col:
                edges.append(uid)
            else:
                interior.append(uid)
    return corners, edges, interior


# --- make_weights: contiguity ---------------------------------------------------------


def test_queen_cardinalities_known_answers(lattice):
    w = make_weights(lattice, method="queen")
    corners, edges, interior = _corner_edge_interior()
    assert all(w.cardinalities[u] == 3 for u in corners)
    assert all(w.cardinalities[u] == 5 for u in edges)
    assert all(w.cardinalities[u] == 8 for u in interior)
    assert w.mean_neighbors == pytest.approx((4 * 3 + 24 * 5 + 36 * 8) / 64)  # 6.5625
    assert w.islands == []
    assert list(w.id_order) == list(lattice["unit"])


def test_rook_cardinalities_known_answers(lattice):
    w = make_weights(lattice, method="rook")
    corners, edges, interior = _corner_edge_interior()
    assert all(w.cardinalities[u] == 2 for u in corners)
    assert all(w.cardinalities[u] == 3 for u in edges)
    assert all(w.cardinalities[u] == 4 for u in interior)
    assert w.mean_neighbors == pytest.approx((4 * 2 + 24 * 3 + 36 * 4) / 64)  # 3.5


def test_row_standardization_rows_sum_to_one(lattice):
    w = make_weights(lattice, method="queen")
    assert str(w.transform).upper() == "R"
    row_sums = [sum(w.weights[i]) for i in w.id_order]
    assert np.allclose(row_sums, 1.0)
    raw = make_weights(lattice, method="queen", row_standardize=False)
    assert set(v for vals in raw.weights.values() for v in vals) == {1.0}
    assert raw.geometrics_meta["row_standardized"] is False


# --- make_weights: knn and distance families ------------------------------------------


def test_knn_exact_cardinality(lattice):
    w = make_weights(lattice, method="knn", k=4)
    assert set(w.cardinalities.values()) == {4}
    assert w.geometrics_meta["k"] == 4
    assert "4-nearest-neighbor" in w.geometrics_meta["spec"]
    with pytest.raises(ValueError, match="k="):
        make_weights(lattice, method="knn", k=64)


def test_distance_band_auto_threshold_no_islands(lattice):
    w = make_weights(lattice, method="distance_band", row_standardize=False)
    assert w.islands == []
    assert w.geometrics_meta["threshold"] is not None
    assert w.geometrics_meta["threshold"] > 0
    assert set(v for vals in w.weights.values() for v in vals) == {1.0}  # binary


def test_inverse_distance_decays_with_power(lattice):
    # A band wide enough (25 km) to reach both the adjacent cell u01 (~13 km) and the
    # diagonal cell u09 (~19 km): 1/d^p weights must decay, and squaring the power
    # must square the weight ratio.
    kwargs = dict(method="inverse_distance", threshold=25_000, row_standardize=False)
    w1 = make_weights(lattice, power=1.0, **kwargs)
    w2 = make_weights(lattice, power=2.0, **kwargs)
    nb1 = dict(zip(w1.neighbors["u00"], w1.weights["u00"], strict=True))
    nb2 = dict(zip(w2.neighbors["u00"], w2.weights["u00"], strict=True))
    ratio1 = nb1["u01"] / nb1["u09"]
    assert ratio1 > 1.0  # nearer neighbors weigh more
    assert nb2["u01"] / nb2["u09"] == pytest.approx(ratio1**2, rel=1e-6)
    assert w2.geometrics_meta["power"] == 2.0
    assert w2.geometrics_meta["threshold"] == 25_000
    assert "inverse distance" in w2.geometrics_meta["spec"]


def test_make_weights_validates_inputs(lattice):
    with pytest.raises(ValueError, match="method"):
        make_weights(lattice, method="hexagon")
    dup = pd.concat([lattice, lattice.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        make_weights(dup)
    with pytest.raises(TypeError, match="GeoDataFrame"):
        make_weights(pd.DataFrame({"unit": ["a"]}))


# --- islands ---------------------------------------------------------------------------


@pytest.fixture()
def lattice_with_island(lattice):
    far = gpd.GeoDataFrame(
        {"unit": ["far"]},
        geometry=[box(80.0, 24.0, 80.0 + STEP, 24.0 + STEP)],
        crs="EPSG:4326",
    )
    return gpd.GeoDataFrame(
        pd.concat([lattice, far], ignore_index=True), crs="EPSG:4326"
    )


def test_queen_leaves_far_polygon_an_island(lattice_with_island):
    with pytest.warns(GeometricsWarning, match="island"):
        w = make_weights(lattice_with_island, method="queen", attach_islands=False)
    assert w.islands == ["far"]
    assert w.geometrics_meta["islands_attached"] == []


def test_make_weights_attaches_island_and_records_meta(lattice_with_island):
    with pytest.warns(GeometricsWarning, match="nearest neighbor"):
        w = make_weights(lattice_with_island, method="queen")
    assert w.islands == []  # attached
    assert w.cardinalities["far"] >= 1
    assert w.geometrics_meta["islands_attached"] == ["far"]
    assert "island" in w.geometrics_meta["spec"]
    assert sum(w.weights["far"]) == pytest.approx(1.0)  # standardized after attach


# --- meta ------------------------------------------------------------------------------


def test_geometrics_meta_contents(lattice):
    w = make_weights(lattice, method="queen")
    meta = w.geometrics_meta
    assert meta["method"] == "queen"
    assert meta["n"] == 64
    assert meta["row_standardized"] is True
    assert meta["k"] is None and meta["threshold"] is None and meta["power"] is None
    assert meta["spec"] == "queen contiguity, row-standardized, n=64"


# --- _default_weights -------------------------------------------------------------------


def test_default_weights_polygons_get_queen(lattice):
    with pytest.warns(GeometricsWarning, match="defaulting"):
        w = _default_weights(lattice, func="somefunc")
    assert w.geometrics_meta["method"] == "queen"


def test_default_weights_points_get_knn(lattice):
    pts = lattice.copy()
    pts = pts.set_geometry(
        gpd.GeoSeries(
            [Point(78.0 + i * 0.01, 20.0 + i * 0.01) for i in range(len(pts))],
            crs="EPSG:4326",
        )
    )
    with pytest.warns(GeometricsWarning, match="defaulting"):
        w = _default_weights(pts, func="somefunc")
    assert w.geometrics_meta["method"] == "knn"
    assert w.geometrics_meta["k"] == 6


# --- explore_connectivity_map -------------------------------------------------------------


def test_connectivity_map_result_fields(lattice):
    w = make_weights(lattice, method="queen")
    res = explore_connectivity_map(lattice, w=w)
    assert isinstance(res, ConnectivityMapResult)
    assert res.n_units == 64
    assert res.mean_neighbors == pytest.approx((4 * 3 + 24 * 5 + 36 * 8) / 64)
    assert res.min_neighbors == 3
    assert res.max_neighbors == 8
    assert 0 < res.pct_nonzero < 100
    assert res.n_components == 1
    assert res.islands == ()
    assert res.w_spec == "queen contiguity, row-standardized, n=64"
    assert list(res.df.columns) == ["unit", "n_neighbors"]
    assert list(res.df["unit"]) == list(lattice["unit"])
    assert res.df["n_neighbors"].sum() == 4 * 3 + 24 * 5 + 36 * 8
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.n_units = 0


def test_connectivity_map_figures(lattice):
    w = make_weights(lattice, method="queen")
    res = explore_connectivity_map(lattice, w=w)
    assert len(res.fig.data) >= 2  # polygons + edges + nodes
    assert {t.type for t in res.fig.data} == {"choroplethmap", "scattermap"}
    node_trace = res.fig.data[-1]
    assert node_trace.hovertemplate.endswith("<extra></extra>")
    assert len(res.fig_hist.data) == 1
    assert res.fig_hist.data[0].type == "bar"


def test_connectivity_map_vector_variant(lattice):
    w = make_weights(lattice, method="queen")
    res = explore_connectivity_map(lattice, w=w, tiles=None, title="Custom title")
    assert {t.type for t in res.fig.data} == {"choropleth", "scattergeo"}
    assert res.fig.layout.geo.visible is False
    assert res.fig.layout.title.text == "Custom title"


def test_connectivity_map_defaults_weights_with_note(lattice):
    with pytest.warns(GeometricsWarning, match="no spatial weights"):
        res = explore_connectivity_map(lattice)
    assert any("defaulted" in note for note in res.notes)
    assert res.mean_neighbors == pytest.approx(6.5625)


def test_connectivity_map_reports_pre_attachment_islands(lattice_with_island):
    with pytest.warns(GeometricsWarning, match="nearest neighbor"):
        w = make_weights(lattice_with_island, method="queen")
    res = explore_connectivity_map(lattice_with_island, w=w)
    assert res.islands == ("far",)  # pre-attachment, read from geometrics_meta
    assert res.min_neighbors >= 1  # but the delivered W has no empty rows


def test_connectivity_map_rejects_mismatched_w(lattice):
    w = make_weights(lattice.iloc[:32], method="knn", k=3)
    with pytest.raises(ValueError, match="match"):
        explore_connectivity_map(lattice, w=w)


def test_connectivity_map_interpret(lattice):
    w = make_weights(lattice, method="queen")
    res = explore_connectivity_map(lattice, w=w)
    text = res.interpret()
    assert "neighbors" in text
    assert "associations, not causal" in text
    assert "causes" not in text
    assert "effect of" not in text
    assert "single connected component" in text
