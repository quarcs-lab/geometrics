"""Tests for the GWR / MGWR vertical (``geometrics.gwr``).

The known-answer fixture plants a spatially varying coefficient on the 8x8
lattice: ``b1(u) = 1 + col_index / 7`` so the ``x1`` association doubles from the
western edge (col 0) to the eastern edge (col 7). GWR/MGWR should recover the
west-to-east gradient in the local ``x1`` coefficients.
"""

from __future__ import annotations

import dataclasses
import time as _time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from geometrics._types import GWRResult, MGWRResult
from geometrics.gwr import analyze_gwr, analyze_mgwr
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE

GRID_SIDE = 8
GRID_N = GRID_SIDE * GRID_SIDE

# Wall-clock seconds of the shared MGWR fit, written by the mgwr_result fixture.
_MGWR_ELAPSED: dict[str, float] = {}


@pytest.fixture(scope="module")
def svc_field(grid_gdf):
    """Cross-section with a planted spatially varying coefficient (SVC) on x1.

    ``y = b0 + b1(u) * x1 + eps`` with ``b1(u) = 1 + col_index / 7``: the x1
    association doubles west (col 0, b1 = 1) to east (col 7, b1 = 2).
    """
    rng = np.random.default_rng(20260702)
    ids = list(grid_gdf["unit"])
    col_index = np.array([i % GRID_SIDE for i in range(GRID_N)], dtype=float)
    x1 = rng.normal(0.0, 1.0, GRID_N)
    b1 = 1.0 + col_index / 7.0
    y = 0.5 + b1 * x1 + rng.normal(0.0, 0.1, GRID_N)
    return pd.DataFrame(
        {"unit": ids, "year": 2020, "col_index": col_index, "x1": x1, "y": y}
    )


@pytest.fixture(scope="module")
def gwr_result(svc_field, grid_gdf):
    """One shared searched-bandwidth GWR fit (bandwidth search is the slow part)."""
    return analyze_gwr(
        svc_field, "y", ["x1"], gdf=grid_gdf, entity="unit", time="year", tiles=None
    )


@pytest.fixture(scope="module")
def mgwr_result(svc_field, grid_gdf):
    """One shared MGWR fit on the same planted-SVC field."""
    start = _time.monotonic()
    res = analyze_mgwr(
        svc_field, "y", ["x1"], gdf=grid_gdf, entity="unit", time="year", tiles=None
    )
    _MGWR_ELAPSED["seconds"] = _time.monotonic() - start
    return res


# ---------------------------------------------------------------------------
# analyze_gwr
# ---------------------------------------------------------------------------


def test_gwr_selected_bandwidth_is_adaptive_and_interior(gwr_result):
    assert isinstance(gwr_result, GWRResult)
    assert gwr_result.fixed is False
    assert 2 < gwr_result.bw < 64
    assert any("bandwidth" in note.lower() for note in gwr_result.notes)


def test_gwr_recovers_west_to_east_gradient(gwr_result, svc_field):
    merged = gwr_result.df.merge(svc_field[["unit", "col_index"]], on="unit")
    west = merged.loc[merged["col_index"] <= 1, "x1_coef"].mean()
    east = merged.loc[merged["col_index"] >= 6, "x1_coef"].mean()
    assert east > west  # the planted b1 gradient doubles west -> east
    assert merged["x1_coef"].min() > 0.0  # b1 is positive everywhere


def test_gwr_result_frame_terms_and_flags(gwr_result):
    df = gwr_result.df
    assert len(df) == 64
    assert df.columns[0] == "unit"
    for term in ("const", "x1"):
        for suffix in ("coef", "se", "t", "significant"):
            assert f"{term}_{suffix}" in df.columns
    assert df["const_significant"].dtype == bool
    assert df["x1_significant"].dtype == bool
    assert df["x1_significant"].all()  # the planted signal is strong everywhere
    assert gwr_result.covariates == ("x1",)
    assert gwr_result.outcome == "y"


def test_gwr_local_r2_bounded(gwr_result):
    assert gwr_result.df["local_r2"].between(0.0, 1.0).all()
    assert 0.0 <= gwr_result.r2 <= 1.0


def test_gwr_figs_one_map_per_term(gwr_result):
    assert list(gwr_result.figs) == ["const", "x1"]
    for fig in gwr_result.figs.values():
        assert isinstance(fig, go.Figure)
    # x1 is significant everywhere: a single diverging trace, no grey mask layer.
    x1_fig = gwr_result.figs["x1"]
    assert len(x1_fig.data) == 1
    assert x1_fig.data[0].zmid == 0.0
    assert isinstance(gwr_result.fig, go.Figure)  # the local-R2 map


def test_gwr_tiles_none_uses_vector_choropleth(gwr_result):
    traces = list(gwr_result.fig.data)
    for fig in gwr_result.figs.values():
        traces.extend(fig.data)
    assert traces
    assert all(isinstance(trace, go.Choropleth) for trace in traces)
    assert not any(isinstance(trace, go.Choroplethmap) for trace in traces)
    assert gwr_result.fig.layout.geo.visible is False


def test_gwr_hover_and_period_semantics(gwr_result):
    trace = gwr_result.figs["x1"].data[0]
    assert trace.hovertemplate.endswith("<extra></extra>")
    assert {row[0] for row in trace.customdata} == {f"u{i:02d}" for i in range(64)}
    assert gwr_result.period == 2020  # latest (sole) period, with a note
    assert any("period" in note.lower() for note in gwr_result.notes)


def test_gwr_correction_and_surfaces(gwr_result):
    # The corrected alpha is stricter than the nominal 5% level.
    assert 0.0 < gwr_result.adj_alpha < 0.05
    assert gwr_result.critical_t > 1.96
    assert np.isfinite(gwr_result.aicc)
    glance = gwr_result.glance()
    assert len(glance) == 1
    assert glance.loc[0, "bw"] == gwr_result.bw
    assert type(gwr_result.gt).__name__ == "GT"
    assert gwr_result.tidy() is gwr_result.df
    with pytest.raises(dataclasses.FrozenInstanceError):
        gwr_result.bw = 10.0  # type: ignore[misc]


def test_gwr_explicit_bw_skips_search(svc_field, grid_gdf, monkeypatch):
    import mgwr.sel_bw

    def _boom(*args, **kwargs):  # pragma: no cover - would mean search ran
        raise AssertionError("Sel_BW should not be constructed when bw is given")

    monkeypatch.setattr(mgwr.sel_bw, "Sel_BW", _boom)
    res = analyze_gwr(
        svc_field,
        "y",
        ["x1"],
        gdf=grid_gdf,
        entity="unit",
        time="year",
        bw=20,
        tiles=None,
    )
    assert res.bw == 20.0
    assert any("user-specified" in note for note in res.notes)


def test_gwr_standardize_adds_note(svc_field, grid_gdf):
    res = analyze_gwr(
        svc_field,
        "y",
        ["x1"],
        gdf=grid_gdf,
        entity="unit",
        time="year",
        bw=20,
        standardize=True,
        tiles=None,
    )
    assert any("standardized" in note for note in res.notes)
    # Standardized outcome/covariate: local coefficients change scale.
    assert res.df["x1_coef"].abs().max() < 5.0


def test_gwr_validation_errors(svc_field, grid_gdf):
    with pytest.raises(KeyError, match="nope"):
        analyze_gwr(svc_field, "y", ["nope"], gdf=grid_gdf, entity="unit", time="year")
    bad = svc_field.assign(txt="x")
    with pytest.raises(TypeError, match="numeric"):
        analyze_gwr(bad, "txt", ["x1"], gdf=grid_gdf, entity="unit", time="year")
    with pytest.raises(TypeError, match="GeoDataFrame"):
        analyze_gwr(svc_field, "y", ["x1"], gdf=pd.DataFrame(), entity="unit")
    with pytest.raises(ValueError, match="outcome"):
        analyze_gwr(svc_field, None, ["x1"], gdf=grid_gdf, entity="unit", time="year")
    with pytest.raises(ValueError, match="covariate"):
        analyze_gwr(svc_field, "y", [], gdf=grid_gdf, entity="unit", time="year")
    with pytest.raises(ValueError, match="alpha"):
        analyze_gwr(
            svc_field, "y", ["x1"], gdf=grid_gdf, entity="unit", time="year", alpha=1.5
        )
    with pytest.raises(ValueError, match="kernel"):
        analyze_gwr(
            svc_field,
            "y",
            ["x1"],
            gdf=grid_gdf,
            entity="unit",
            time="year",
            kernel="triangle",
        )
    with pytest.raises(ValueError, match="bw"):
        analyze_gwr(
            svc_field, "y", ["x1"], gdf=grid_gdf, entity="unit", time="year", bw=-3
        )


def test_gwr_roles_default_from_attrs(svc_field, grid_gdf):
    from geometrics._roles import set_roles

    df = set_roles(svc_field.copy(), outcome="y", covariates=["x1"])
    res = analyze_gwr(df, gdf=grid_gdf, entity="unit", time="year", bw=20, tiles=None)
    assert res.outcome == "y"
    assert res.covariates == ("x1",)


# ---------------------------------------------------------------------------
# analyze_mgwr
# ---------------------------------------------------------------------------


def test_mgwr_bandwidth_dict_and_runtime(mgwr_result):
    assert isinstance(mgwr_result, MGWRResult)
    assert list(mgwr_result.bw) == ["const", "x1"]
    for bw in mgwr_result.bw.values():
        assert 0 < bw <= 64
    assert _MGWR_ELAPSED["seconds"] < 60.0
    assert list(mgwr_result.adj_alpha) == ["const", "x1"]
    assert list(mgwr_result.critical_t) == ["const", "x1"]
    assert all(t > 0 for t in mgwr_result.critical_t.values())


def test_mgwr_notes_mention_standardization(mgwr_result):
    assert any("standardized" in note for note in mgwr_result.notes)


def test_mgwr_recovers_gradient_on_standardized_scale(mgwr_result, svc_field):
    merged = mgwr_result.df.merge(svc_field[["unit", "col_index"]], on="unit")
    west = merged.loc[merged["col_index"] <= 1, "x1_coef"].mean()
    east = merged.loc[merged["col_index"] >= 6, "x1_coef"].mean()
    assert east > west


def test_mgwr_result_surfaces(mgwr_result):
    df = mgwr_result.df
    for term in ("const", "x1"):
        for suffix in ("coef", "se", "t", "significant"):
            assert f"{term}_{suffix}" in df.columns
    assert "residual" in df.columns
    assert list(mgwr_result.figs) == ["const", "x1"]
    assert isinstance(mgwr_result.fig, go.Figure)
    traces = [t for f in (mgwr_result.fig, *mgwr_result.figs.values()) for t in f.data]
    assert all(isinstance(trace, go.Choropleth) for trace in traces)  # tiles=None
    assert type(mgwr_result.gt).__name__ == "GT"
    assert type(mgwr_result.gt_bw).__name__ == "GT"
    assert 0.0 <= mgwr_result.r2 <= 1.0


def test_mgwr_validation_errors(svc_field, grid_gdf):
    with pytest.raises(KeyError, match="nope"):
        analyze_mgwr(svc_field, "y", ["nope"], gdf=grid_gdf, entity="unit", time="year")
    with pytest.raises(ValueError, match="max_iter"):
        analyze_mgwr(
            svc_field,
            "y",
            ["x1"],
            gdf=grid_gdf,
            entity="unit",
            time="year",
            max_iter=0,
        )
    with pytest.raises(ValueError, match="criterion"):
        analyze_mgwr(
            svc_field,
            "y",
            ["x1"],
            gdf=grid_gdf,
            entity="unit",
            time="year",
            criterion="R2",
        )


# ---------------------------------------------------------------------------
# interpret
# ---------------------------------------------------------------------------


def test_interpret_gwr_names_strongest_location(gwr_result):
    text = gwr_result.interpret()
    assert "strongest" in text
    # The named unit is the largest-|coef| significant unit for x1.
    df = gwr_result.df
    pool = df.loc[df["x1_significant"]]
    expected = str(pool.loc[pool["x1_coef"].abs().idxmax(), "unit"])
    assert expected in text
    assert "x1" in text
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)


def test_interpret_mgwr_mentions_scales_and_standardization(mgwr_result):
    text = mgwr_result.interpret()
    assert "strongest" in text
    assert "bandwidth" in text.lower()
    assert "standardized" in text
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)
