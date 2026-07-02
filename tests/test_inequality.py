"""Tests for the regional-inequality vertical (``regional_inequality``)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from great_tables import GT

from geometrics._types import InequalityOverTimeResult, TheilDecompositionResult
from geometrics._validation import GeometricsWarning
from geometrics.pedagogy._interpret._shared import _ASSOC_NOTE
from geometrics.regional_inequality import (
    analyze_inequality_over_time,
    analyze_theil_decomposition,
)


def _panel(values_by_year: dict[int, list[float]], ids: list[str]) -> pd.DataFrame:
    """Build a long panel from a {year: values} mapping."""
    frames = [
        pd.DataFrame({"unit": ids, "year": year, "y": values})
        for year, values in values_by_year.items()
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def two_point_panel():
    """Two periods of the two-point distribution [1, 1, 1, 3]."""
    ids = ["u1", "u2", "u3", "u4"]
    return _panel({2000: [1.0, 1.0, 1.0, 3.0], 2001: [1.0, 1.0, 1.0, 3.0]}, ids)


@pytest.fixture(scope="module")
def state_panel():
    """Two states with two districts each over three years (positive values)."""
    rows = []
    values = {
        ("d1", "north"): [10.0, 11.0, 12.0],
        ("d2", "north"): [12.0, 13.0, 14.0],
        ("d3", "south"): [30.0, 33.0, 36.0],
        ("d4", "south"): [36.0, 40.0, 44.0],
    }
    for (district, state), series in values.items():
        for year, v in zip((2000, 2001, 2002), series, strict=True):
            rows.append(
                {"district": district, "state": state, "year": year, "income": v}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# analyze_inequality_over_time — closed-form values
# ---------------------------------------------------------------------------


def test_equal_values_give_zero_inequality():
    ids = [f"u{i}" for i in range(4)]
    df = _panel({y: [5.0] * 4 for y in (2000, 2001, 2002)}, ids)
    res = analyze_inequality_over_time(
        df, "y", entity="unit", time="year", measures=("gini", "theil", "cv")
    )
    assert np.allclose(res.df["gini"], 0.0, atol=1e-12)
    assert np.allclose(res.df["theil"], 0.0, atol=1e-12)
    assert np.allclose(res.df["cv"], 0.0, atol=1e-12)
    # log-trends need positive dispersion: none here, so the slopes are NaN.
    assert res.summary["slope"].isna().all()
    assert not res.summary["converging"].any()
    assert any("trend was not estimated" in n for n in res.notes)


def test_gini_matches_hand_computed_value(two_point_panel):
    res = analyze_inequality_over_time(
        two_point_panel, "y", entity="unit", time="year", measures=("gini",)
    )
    # Gini of [1, 1, 1, 3] by the standard mean-absolute-difference formula:
    # sum_ij |xi - xj| / (2 n^2 mean) = 12 / (2 * 16 * 1.5) = 0.25 — and
    # inequality.gini.Gini(y).g returns exactly this value (verified v1.1.2).
    assert res.df["gini"].to_numpy() == pytest.approx([0.25, 0.25], abs=1e-12)


def test_theil_requires_strictly_positive_values():
    ids = ["a1", "a2", "a3"]
    df = _panel({2000: [1.0, 2.0, 3.0], 2001: [1.0, 0.0, 3.0]}, ids)
    with pytest.raises(ValueError, match="a2"):
        analyze_inequality_over_time(
            df, "y", entity="unit", time="year", measures=("gini", "theil")
        )


def test_gini_tolerates_zero_values():
    ids = ["a1", "a2", "a3"]
    df = _panel({2000: [1.0, 2.0, 3.0], 2001: [1.0, 0.0, 3.0]}, ids)
    res = analyze_inequality_over_time(
        df, "y", entity="unit", time="year", measures=("gini",)
    )
    assert np.isfinite(res.df["gini"]).all()


# ---------------------------------------------------------------------------
# analyze_inequality_over_time — validation and panel semantics
# ---------------------------------------------------------------------------


def test_missing_column_raises_keyerror(two_point_panel):
    with pytest.raises(KeyError, match="nope"):
        analyze_inequality_over_time(
            two_point_panel, "nope", entity="unit", time="year"
        )


def test_non_numeric_var_raises_typeerror(two_point_panel):
    df = two_point_panel.assign(label=lambda d: d["unit"])
    with pytest.raises(TypeError, match="label"):
        analyze_inequality_over_time(df, "label", entity="unit", time="year")


def test_unknown_measure_raises(two_point_panel):
    with pytest.raises(ValueError, match="bogus"):
        analyze_inequality_over_time(
            two_point_panel, "y", entity="unit", time="year", measures=("gini", "bogus")
        )


def test_w_without_gdf_raises(two_point_panel, grid_w):
    with pytest.raises(ValueError, match="without gdf"):
        analyze_inequality_over_time(
            two_point_panel, "y", entity="unit", time="year", w=grid_w
        )


def test_single_period_raises():
    df = _panel({2000: [1.0, 2.0, 3.0]}, ["a", "b", "c"])
    with pytest.raises(ValueError, match="at least 2 periods"):
        analyze_inequality_over_time(df, "y", entity="unit", time="year")


def test_unbalanced_panel_noted(convergence_panel):
    df = convergence_panel.iloc[1:]  # drop one row: one unit missing in 2000
    with pytest.warns(GeometricsWarning, match="unbalanced"):
        res = analyze_inequality_over_time(
            df, "gdppc", entity="unit", time="year", measures=("gini",)
        )
    assert any("unbalanced" in n for n in res.notes)
    assert res.df["n_units"].iloc[0] == 63
    assert res.df["n_units"].iloc[-1] == 64


def test_start_end_window(convergence_panel):
    res = analyze_inequality_over_time(
        convergence_panel,
        "gdppc",
        entity="unit",
        time="year",
        measures=("gini",),
        start=2001,
        end=2004,
    )
    assert res.n_periods == 4
    assert res.df["time"].tolist() == [2001, 2002, 2003, 2004]


# ---------------------------------------------------------------------------
# analyze_inequality_over_time — planted convergence and result surface
# ---------------------------------------------------------------------------


def test_convergence_panel_gini_converges(convergence_panel):
    res = analyze_inequality_over_time(
        convergence_panel,
        "gdppc",
        entity="unit",
        time="year",
        measures=("gini", "theil", "cv"),
    )
    gini_row = res.summary.set_index("measure").loc["gini"]
    assert gini_row["slope"] < 0
    assert bool(gini_row["converging"])
    assert res.n_periods == 6
    assert res.n_units == 64


def test_inequality_result_surface(convergence_panel):
    res = analyze_inequality_over_time(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    assert isinstance(res, InequalityOverTimeResult)
    assert isinstance(res.fig, go.Figure)
    assert isinstance(res.gt, GT)
    assert list(res.summary.columns) == [
        "measure",
        "slope",
        "se",
        "pvalue",
        "r2",
        "converging",
    ]
    assert {"time", "n_units", "gini", "theil"} <= set(res.df.columns)
    assert res.w_spec is None
    assert len(res.models) == 2
    assert res.tidy().equals(res.summary)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.var = "other"  # type: ignore[misc]
    # each measure draws a line plus a dashed fitted trend
    assert len(res.fig.data) == 4


def test_inequality_interpret_is_association_only(convergence_panel):
    res = analyze_inequality_over_time(
        convergence_panel, "gdppc", entity="unit", time="year"
    )
    text = res.interpret()
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)
    assert "narrowing" in text


# ---------------------------------------------------------------------------
# analyze_inequality_over_time — spatial Gini decomposition
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def positive_field_panel(sar_field):
    """Three periods of strictly positive values (exp-transformed SAR field)."""
    base = np.exp(sar_field["y"].to_numpy() / 2.0)
    frames = [
        pd.DataFrame(
            {"unit": sar_field["unit"], "year": year, "gdp": base * (1.0 + 0.02 * i)}
        )
        for i, year in enumerate((2020, 2021, 2022))
    ]
    return pd.concat(frames, ignore_index=True)


def test_spatial_gini_columns_present(grid_gdf, grid_w, positive_field_panel):
    res = analyze_inequality_over_time(
        positive_field_panel,
        "gdp",
        entity="unit",
        time="year",
        gdf=grid_gdf,
        w=grid_w,
        permutations=49,
    )
    assert "gini_spatial" in res.df.columns
    assert "gini_spatial_p" in res.df.columns
    p = res.df["gini_spatial_p"].to_numpy(dtype=float)
    assert ((p >= 0.0) & (p <= 1.0)).all()
    # the neighbor component is a positive slice of the overall Gini
    assert (res.df["gini_spatial"] > 0).all()
    assert (res.df["gini_spatial"] <= res.df["gini"] + 1e-12).all()
    assert res.w_spec is not None and "n=64" in res.w_spec


def test_spatial_gini_p_reproducible(grid_gdf, grid_w, positive_field_panel):
    kwargs = dict(entity="unit", time="year", gdf=grid_gdf, w=grid_w, permutations=49)
    res1 = analyze_inequality_over_time(positive_field_panel, "gdp", **kwargs)
    res2 = analyze_inequality_over_time(positive_field_panel, "gdp", **kwargs)
    assert res1.df["gini_spatial_p"].tolist() == res2.df["gini_spatial_p"].tolist()


def test_spatial_gini_default_weights_warn(grid_gdf, positive_field_panel):
    with pytest.warns(GeometricsWarning, match="no spatial weights supplied"):
        res = analyze_inequality_over_time(
            positive_field_panel,
            "gdp",
            entity="unit",
            time="year",
            gdf=grid_gdf,
            permutations=9,
        )
    assert "gini_spatial" in res.df.columns
    assert res.w_spec is not None and "queen" in res.w_spec


def test_spatial_gini_no_permutations_gives_nan_p(
    grid_gdf, grid_w, positive_field_panel
):
    res = analyze_inequality_over_time(
        positive_field_panel,
        "gdp",
        entity="unit",
        time="year",
        gdf=grid_gdf,
        w=grid_w,
        permutations=0,
    )
    assert res.df["gini_spatial_p"].isna().all()
    assert any("permutations=0" in n for n in res.notes)


# ---------------------------------------------------------------------------
# analyze_theil_decomposition
# ---------------------------------------------------------------------------


def test_theil_decomposition_identity(state_panel):
    res = analyze_theil_decomposition(
        state_panel, "income", "state", entity="district", time="year"
    )
    gap = (res.df["between"] + res.df["within"] - res.df["theil"]).abs()
    assert (gap < 1e-10).all()
    assert res.df["between_share"].between(0.0, 1.0).all()
    assert res.n_groups == 2
    assert "p_between" not in res.df.columns


def test_theil_partition_by_entity_gives_zero_within(state_panel):
    res = analyze_theil_decomposition(
        state_panel, "income", "district", entity="district", time="year"
    )
    assert np.allclose(res.df["within"], 0.0, atol=1e-12)
    assert np.allclose(res.df["between"], res.df["theil"], atol=1e-12)


def test_theil_group_must_be_constant_within_entity(state_panel):
    df = state_panel.copy()
    df.loc[(df["district"] == "d1") & (df["year"] == 2001), "state"] = "south"
    with pytest.raises(ValueError, match="d1"):
        analyze_theil_decomposition(
            df, "income", "state", entity="district", time="year"
        )


def test_theil_decomposition_zero_value_raises(state_panel):
    df = state_panel.copy()
    df.loc[(df["district"] == "d3") & (df["year"] == 2001), "income"] = 0.0
    with pytest.raises(ValueError, match="d3"):
        analyze_theil_decomposition(
            df, "income", "state", entity="district", time="year"
        )


def test_theil_single_group_raises(state_panel):
    df = state_panel.assign(state="all")
    with pytest.raises(ValueError, match="at least 2"):
        analyze_theil_decomposition(
            df, "income", "state", entity="district", time="year"
        )


def test_theil_permutation_inference(state_panel):
    res = analyze_theil_decomposition(
        state_panel,
        "income",
        "state",
        entity="district",
        time="year",
        permutations=50,
    )
    assert "p_between" in res.df.columns
    p = res.df["p_between"].to_numpy(dtype=float)
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert res.permutations == 50
    # seeded: a second run reproduces the p-values exactly
    res2 = analyze_theil_decomposition(
        state_panel,
        "income",
        "state",
        entity="district",
        time="year",
        permutations=50,
    )
    assert res.df["p_between"].tolist() == res2.df["p_between"].tolist()


def test_theil_result_surface_and_interpret(state_panel):
    res = analyze_theil_decomposition(
        state_panel, "income", "state", entity="district", time="year"
    )
    assert isinstance(res, TheilDecompositionResult)
    assert isinstance(res.fig, go.Figure)
    assert isinstance(res.gt, GT)
    assert list(res.df.columns) == [
        "time",
        "theil",
        "between",
        "within",
        "between_share",
    ]
    assert len(res.fig.data) == 3  # between + within areas, between-share line
    assert res.tidy().equals(res.df)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.group = "other"  # type: ignore[misc]
    text = res.interpret()
    assert "causes" not in text
    assert "effect of" not in text
    assert text.endswith(_ASSOC_NOTE)
    assert "between" in text
