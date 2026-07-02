"""Tests for the variable-label helper (set_labels / resolve_label / resolve_labels)."""

from __future__ import annotations

import pandas as pd
import pytest

from geometrics._labels import resolve_label, resolve_labels, set_labels
from geometrics._panel import resolve_panel


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small deterministic panel: 2 firms x 2 years and three numeric columns."""
    return pd.DataFrame(
        {
            "firm": [1, 1, 2, 2],
            "year": [2000, 2001, 2000, 2001],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [0.5, 1.0, 1.5, 2.0],
            "x3": [2.0, 1.0, 4.0, 3.0],
        }
    )


def _toy_dict() -> pd.DataFrame:
    """A small df_dict exercising the label -> var_def -> var_name fallback chain."""
    return pd.DataFrame(
        {
            "var_name": ["firm", "year", "x1", "x2", "x3"],
            "var_def": ["Firm id", "Year", "First X", "", "Third X"],
            "label": ["Firm", "Year", "Variable one", "", "   "],
            "type": ["entity", "time", "numeric", "numeric", "numeric"],
        }
    )


# ------------------------------------------------------------------ set_labels / resolve ---
def test_dict_mapping_and_precedence(sample_df):
    df = set_labels(sample_df.copy(), {"x1": "Alpha"})
    assert resolve_label(df, "x1") == "Alpha"  # from attrs
    assert resolve_label(df, "x2") == "x2"  # no label -> bare name
    assert resolve_label(df, "x1", label="Override") == "Override"  # explicit wins


def test_unknown_name_never_raises(sample_df):
    df = set_labels(sample_df.copy(), {"x1": "Alpha"})
    # regression terms are not columns; resolve must fall back, not raise
    assert resolve_label(df, "log_gdp_pc_sq") == "log_gdp_pc_sq"


def test_resolve_labels_vectorized_with_override(sample_df):
    df = set_labels(sample_df.copy(), {"x1": "Alpha", "x2": "Beta"})
    out = resolve_labels(df, ["x1", "x2", "x3"], labels={"x1": "Over"})
    assert out == ["Over", "Beta", "x3"]


def test_df_dict_label_then_var_def_then_name(sample_df):
    df = set_labels(sample_df.copy(), _toy_dict())
    assert resolve_label(df, "x1") == "Variable one"  # label column
    assert resolve_label(df, "x2") == "x2"  # blank label + blank var_def -> name
    assert resolve_label(df, "x3") == "Third X"  # whitespace label -> var_def


def test_df_dict_without_label_column_uses_var_def(sample_df):
    dd = pd.DataFrame({"var_name": ["x1"], "var_def": ["First X"], "type": ["numeric"]})
    df = set_labels(sample_df.copy(), dd)
    assert resolve_label(df, "x1") == "First X"


def test_set_panel_flag_declares_panel(sample_df):
    df = set_labels(sample_df.copy(), _toy_dict(), set_panel=True)
    assert resolve_panel(df) == ("firm", "year")


def test_set_labels_merges_partial_updates(sample_df):
    df = set_labels(sample_df.copy(), {"x1": "Alpha"})
    set_labels(df, {"x2": "Beta"})
    assert resolve_label(df, "x1") == "Alpha"  # first mapping survives
    assert resolve_label(df, "x2") == "Beta"
