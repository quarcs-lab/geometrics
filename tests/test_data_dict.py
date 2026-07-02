"""Tests for :func:`geometrics.build_data_dict` — data-dictionary inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geometrics._data_dict import build_data_dict
from geometrics._labels import set_labels
from geometrics._panel import resolve_panel

_DICT_COLUMNS = ["var_name", "var_def", "label", "type", "role", "can_be_na"]
_TYPES = {"entity", "time", "factor", "logical", "numeric"}


def test_returns_column_contract():
    df = pd.DataFrame({"country": ["A", "B"], "year": [2000, 2001], "gdp": [1.0, 2.0]})
    ddict = build_data_dict(df)
    assert list(ddict.columns) == _DICT_COLUMNS
    assert ddict["can_be_na"].dtype == bool
    assert len(ddict) == df.shape[1]
    assert list(ddict["var_name"]) == list(df.columns)  # one row per column, in order
    assert set(ddict["type"]).issubset(_TYPES)
    # No covariate/outcome is ever auto-assigned; role is blank unless an entity_name is found.
    assert set(ddict["role"]).issubset({"", "entity_name"})


def test_type_inference_table():
    n = 12
    df = pd.DataFrame(
        {
            "country": (["A"] * 6) + (["B"] * 6),  # object name hint -> entity
            "year": list(range(2000, 2006)) * 2,  # int year -> time
            "continent": (["x", "y", "z"] * 4),  # >2 object -> factor
            "flag": [True, False] * 6,  # bool -> logical
            "binary": [0, 1] * 6,  # 2-valued -> logical
            "grade": ([1, 2, 3] * 4),  # low-card numeric -> factor
            "gdp": np.linspace(1.0, 50.0, n),  # high-card continuous -> numeric
        }
    )
    ddict = (
        build_data_dict(df, factor_cutoff=10).set_index("var_name")["type"].to_dict()
    )
    assert ddict["country"] == "entity"
    assert ddict["year"] == "time"
    assert ddict["continent"] == "factor"
    assert ddict["flag"] == "logical"
    assert ddict["binary"] == "logical"
    assert ddict["grade"] == "factor"
    assert ddict["gdp"] == "numeric"


def test_explicit_entity_time_win_over_detection():
    df = pd.DataFrame(
        {"country": ["A", "B"], "year": [2000, 2001], "wave": [1, 2], "x": [1.0, 2.0]}
    )
    ddict = build_data_dict(df, entity="x", time="wave").set_index("var_name")["type"]
    assert ddict["x"] == "entity"
    assert ddict["wave"] == "time"
    # the auto-detected country/year are demoted because the user pinned the ids
    assert ddict["country"] != "entity"
    assert ddict["year"] != "time"


def test_explicit_id_must_exist():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="not in df"):
        build_data_dict(df, entity="missing")


def test_all_numeric_frame_has_no_false_panel():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(20, 3)), columns=["m1", "m2", "m3"])
    ddict = build_data_dict(df)
    assert "entity" not in set(ddict["type"])
    assert "time" not in set(ddict["type"])


def test_year_detection_without_name_hint():
    df = pd.DataFrame(
        {"unit": ["A", "B", "A", "B"], "yr_col": [1998, 1998, 1999, 1999]}
    )
    # "yr_col" tokenizes to {yr, col}; "yr" is a time hint, so it is detected as time.
    ddict = build_data_dict(df).set_index("var_name")["type"]
    assert ddict["yr_col"] == "time"
    assert ddict["unit"] == "entity"


def test_humanized_labels():
    df = pd.DataFrame({"gini_regional": [0.1, 0.2], "log_gdp_pc": [1.0, 2.0]})
    ddict = build_data_dict(df).set_index("var_name")
    assert ddict.loc["gini_regional", "label"] == "Gini Regional"
    assert ddict.loc["log_gdp_pc", "label"] == "Log Gdp Pc"
    assert (
        ddict.loc["gini_regional", "var_def"] == "Gini Regional"
    )  # var_def defaults to label


def test_roundtrip_recovers_panel():
    """build_data_dict -> set_labels(set_panel=True) declares the right panel."""
    df = pd.DataFrame(
        {
            "region": ["A", "A", "B", "B"],
            "year": [2000, 2001, 2000, 2001],
            "gdp_pc": [1.0, 1.1, 2.0, 2.1],
        }
    )
    ddict = build_data_dict(df)
    df = set_labels(df, ddict, set_panel=True)
    assert resolve_panel(df) == ("region", "year")
    # the inferred dictionary satisfies the column contract
    assert set(ddict["type"]).issubset(_TYPES)
    assert ddict["can_be_na"].dtype == bool


def test_empty_frame_is_safe():
    ddict = build_data_dict(pd.DataFrame())
    assert list(ddict.columns) == _DICT_COLUMNS
    assert len(ddict) == 0


def test_non_dataframe_raises():
    with pytest.raises(TypeError):
        build_data_dict([1, 2, 3])
