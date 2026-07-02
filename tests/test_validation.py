"""Tests for the shared validation helpers: NaN-drop reporting + column checks."""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geometrics._validation import (
    GeometricsWarning,
    drop_missing,
    ensure_dataframe,
    ensure_geodataframe,
    require_columns,
)


def test_geometrics_warning_is_userwarning():
    # Subclassing UserWarning keeps existing ``pytest.warns(UserWarning)`` callers matching.
    assert issubclass(GeometricsWarning, UserWarning)


def test_ensure_dataframe_passes_and_raises():
    df = pd.DataFrame({"a": [1]})
    assert ensure_dataframe(df) is df
    with pytest.raises(TypeError, match="pandas DataFrame"):
        ensure_dataframe([1, 2, 3])


def test_ensure_geodataframe_passes_and_raises():
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0.0, 0.0)])
    assert ensure_geodataframe(gdf) is gdf
    # a plain DataFrame is not enough
    with pytest.raises(TypeError, match=r"gdf needs to be a geopandas GeoDataFrame"):
        ensure_geodataframe(pd.DataFrame({"a": [1]}))
    with pytest.raises(
        TypeError, match=r"choropleth: frame needs to be a geopandas GeoDataFrame"
    ):
        ensure_geodataframe([1, 2], arg="frame", func="choropleth")


def test_drop_missing_warns_with_count():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, np.nan], "b": [1, 2, 3, 4, 5]})
    with pytest.warns(
        GeometricsWarning,
        match=r"myfunc: dropped 2 of 5 row\(s\) \(40%\) with missing values in \['a'\]",
    ):
        out = drop_missing(df, ["a"], func="myfunc")
    assert len(out) == 3
    assert not out["a"].isna().any()


def test_drop_missing_silent_when_clean():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1, 2, 3]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise here
        out = drop_missing(df, ["a", "b"], func="f")
    assert out.equals(df)


def test_drop_missing_empty_frame_no_warning_no_zerodiv():
    df = pd.DataFrame({"a": []})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = drop_missing(df, ["a"], func="f")
    assert len(out) == 0


def test_drop_missing_stacklevel_points_at_caller():
    df = pd.DataFrame({"a": [1.0, np.nan]})

    def public_like(frame):  # simulates the public function that calls the helper
        return drop_missing(frame, ["a"], func="f")  # default stacklevel=3

    with pytest.warns(GeometricsWarning) as record:
        public_like(df)
    # stacklevel=3: warn -> drop_missing -> public_like -> this test => the test's frame.
    assert record[0].filename.endswith("test_validation.py")


def test_require_columns_names_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(
        ValueError, match=r"xtsum: column\(s\) not found in df: \['nope'\]"
    ):
        require_columns(df, ["a", "nope"], where="xtsum")


def test_require_columns_passes_when_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    require_columns(df, ["a", "b"], where="x")  # no raise
