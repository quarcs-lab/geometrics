"""Tests for the df_dict-driven metadata: analytical roles and entity names.

Covers the storage helpers (``set_roles`` / ``set_panel(entity_name=)``), their resolution from
a data dictionary (``set_labels(df_dict, set_panel=True)``), the entity-name auto-detection in
``build_data_dict``, the shared ``Name (id)`` display helper, and the two ``df_dict`` fields
enforced downstream (``type`` authority and ``can_be_na`` completeness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geometrics._common import entity_display_map, lead_columns
from geometrics._data_dict import build_data_dict
from geometrics._labels import set_labels
from geometrics._panel import resolve_entity_name, set_panel, stored_entity_name
from geometrics._roles import resolve_roles, set_roles, stored_roles
from geometrics._validation import drop_required, required_columns


def _toy() -> pd.DataFrame:
    """A 2-unit panel with an id, a readable name, a code and two numeric variables."""
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "name": ["Foo", "Foo", "Bar", "Bar"],
            "code": ["F", "F", "B", "B"],
            "year": [2000, 2001, 2000, 2001],
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.5, 1.0, 1.5, 2.0],
        }
    )


# --------------------------------------------------------------------------- roles storage ---
def test_set_roles_roundtrip():
    df = set_roles(_toy(), outcome="y", covariates=["x"])
    assert stored_roles(df) == ("y", ["x"])
    assert resolve_roles(df) == ("y", ["x"])


def test_set_roles_single_covariate_string():
    df = set_roles(_toy(), outcome="y", covariates="x")
    assert stored_roles(df) == ("y", ["x"])


def test_set_roles_partial_update():
    df = set_roles(_toy(), outcome="y", covariates=["x"])
    set_roles(df, covariates=["x", "year"])  # leave outcome untouched
    assert stored_roles(df) == ("y", ["x", "year"])


def test_set_roles_validates_columns():
    with pytest.raises(ValueError):
        set_roles(_toy(), outcome="nope")
    with pytest.raises(ValueError):
        set_roles(_toy(), covariates=["x", "nope"])


def test_resolve_roles_explicit_wins():
    df = set_roles(_toy(), outcome="y", covariates=["x"])
    assert resolve_roles(df, outcome="x") == ("x", ["x"])
    assert resolve_roles(df, covariates=["year"]) == ("y", ["year"])


def test_stored_roles_empty_by_default():
    assert stored_roles(_toy()) == (None, [])


# ----------------------------------------------------------------------- entity name storage ---
def test_set_panel_entity_name_roundtrip():
    df = set_panel(_toy(), entity="id", time="year", entity_name="name")
    assert stored_entity_name(df) == "name"
    assert resolve_entity_name(df) == "name"


def test_resolve_entity_name_absent_returns_none():
    df = _toy()
    assert resolve_entity_name(df) is None  # nothing declared
    df = set_panel(df, entity="id", time="year", entity_name="name")
    assert resolve_entity_name(df.drop(columns=["name"])) is None  # column gone


# ------------------------------------------------------------------------- entity display map ---
def test_entity_display_map_name_id():
    df = _toy()
    disp = entity_display_map(df, "id", "name")
    assert disp == {"1": "Foo (1)", "2": "Bar (2)"}


def test_entity_display_map_identity_when_no_name():
    df = _toy()
    assert entity_display_map(df, "id", None) == {"1": "1", "2": "2"}
    assert entity_display_map(df, "id", "id") == {"1": "1", "2": "2"}  # no "X (X)"
    assert entity_display_map(df, "id", "missing") == {"1": "1", "2": "2"}


def test_entity_display_map_blank_name_falls_back():
    df = _toy()
    df.loc[df["id"] == 2, "name"] = np.nan
    disp = entity_display_map(df, "id", "name")
    assert disp == {"1": "Foo (1)", "2": "2"}


# ---------------------------------------------------------------------------- df_dict loading ---
def test_set_labels_loads_roles_and_entity_name():
    df0 = _toy()
    ddict = build_data_dict(df0)
    ddict.loc[ddict["var_name"] == "y", "role"] = "outcome"
    ddict.loc[ddict["var_name"] == "x", "role"] = "covariate"
    df = set_labels(df0, ddict, set_panel=True)
    assert stored_roles(df) == ("y", ["x"])
    # ``name`` is auto-detected as the entity-name column.
    assert stored_entity_name(df) == "name"


def test_set_labels_roleless_df_dict_tolerated():
    df0 = _toy()
    ddict = build_data_dict(df0).drop(columns=["role"])
    df = set_labels(df0, ddict, set_panel=True)
    assert stored_roles(df) == (None, [])
    assert stored_entity_name(df) is None


# -------------------------------------------------------------- build_data_dict auto-detection ---
def test_build_data_dict_emits_role_and_detects_name():
    ddict = build_data_dict(_toy())
    assert "role" in ddict.columns
    # ``name`` wins over ``code`` (name-like, longer) and over ``id`` (the entity itself).
    names = list(ddict.loc[ddict["role"] == "entity_name", "var_name"])
    assert names == ["name"]
    assert set(ddict.loc[ddict["role"] != "entity_name", "role"]) == {""}


def test_build_data_dict_no_name_when_id_already_readable():
    # A panel keyed on ``country`` (a readable name) paired with a code — no backwards label.
    df = pd.DataFrame(
        {
            "country": ["Alpha", "Alpha", "Beta", "Beta"],
            "iso": ["AL", "AL", "BE", "BE"],
            "year": [2000, 2001, 2000, 2001],
            "gdp": [1.0, 1.1, 2.0, 2.1],
        }
    )
    ddict = build_data_dict(df)
    assert list(ddict.loc[ddict["role"] == "entity_name", "var_name"]) == []


def test_lead_columns_orders_keys_first():
    assert lead_columns(["a", "x", "y", "b"], ["y", "x"]) == ["y", "x", "a", "b"]
    assert lead_columns(["a", "b"], [None, "missing"]) == ["a", "b"]  # no-op


# ------------------------------------------------------------ the inert df_dict fields ---
def test_required_columns_and_drop_required():
    df0 = _toy()
    # Mark only ``x`` required (entity/time default to required too, but keep the toy explicit).
    ddict = build_data_dict(df0)
    ddict["can_be_na"] = True
    ddict.loc[ddict["var_name"] == "x", "can_be_na"] = False
    assert required_columns(ddict) == ["x"]
    df_missing = df0.copy()
    df_missing.loc[0, "x"] = np.nan
    kept = drop_required(df_missing, ddict)
    assert len(kept) == len(df0) - 1
    # No required columns -> a no-op.
    ddict["can_be_na"] = True
    assert len(drop_required(df_missing, ddict)) == len(df_missing)
