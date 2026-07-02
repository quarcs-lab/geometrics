"""Tests for the panel declaration helper (set_panel / resolve_panel)."""

from __future__ import annotations

import pandas as pd
import pytest

from geometrics._panel import resolve_panel, set_panel


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small deterministic panel: 3 firms x 4 years, a group and a numeric column."""
    firms = [1, 2, 3]
    years = [2000, 2001, 2002, 2003]
    return pd.DataFrame(
        {
            "firm": [f for f in firms for _ in years],
            "year": years * len(firms),
            "grp": ["a", "a", "b", "b"] * len(firms),
            "x1": [float(i) for i in range(len(firms) * len(years))],
        }
    )


# ----------------------------------------------------------------- set_panel / resolve_panel ---
def test_set_panel_stores_and_resolves(sample_df):
    df = set_panel(sample_df.copy(), entity="firm", time="year")
    assert resolve_panel(df) == ("firm", "year")


def test_explicit_args_win_over_attrs(sample_df):
    df = set_panel(sample_df.copy(), entity="firm", time="year")
    # an explicit override beats the stored default
    assert resolve_panel(df, entity="grp") == ("grp", "year")


def test_resolve_without_declaration_returns_none(sample_df):
    assert resolve_panel(sample_df.copy()) == (None, None)


def test_require_raises_when_unresolved(sample_df):
    with pytest.raises(ValueError):
        resolve_panel(sample_df.copy(), require_time=True)
    with pytest.raises(ValueError):
        resolve_panel(sample_df.copy(), require_entity=True)


def test_unknown_column_raises(sample_df):
    with pytest.raises(ValueError):
        set_panel(sample_df.copy(), entity="not_a_column")
    with pytest.raises(ValueError):
        resolve_panel(sample_df.copy(), time="not_a_column")


def test_set_panel_partial_update(sample_df):
    df = set_panel(sample_df.copy(), entity="firm", time="year")
    set_panel(df, entity="grp")  # only entity changes; time is preserved
    assert resolve_panel(df) == ("grp", "year")
