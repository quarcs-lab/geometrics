"""Shared input-validation helpers used across the analytical functions."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import geopandas as gpd
import pandas as pd
from pandas.api import types as pdt

__all__ = [
    "ensure_dataframe",
    "ensure_geodataframe",
    "is_numeric_or_logical",
    "numeric_logical_columns",
    "GeometricsWarning",
    "drop_missing",
    "drop_required",
    "required_columns",
    "require_columns",
]


class GeometricsWarning(UserWarning):
    """Advisory warning raised by geometrics (e.g. dropped rows, sampling).

    A subclass of :class:`UserWarning`, so existing ``pytest.warns(UserWarning)``
    callers keep matching while users can silence *only* geometrics' advisory notices
    with ``warnings.filterwarnings("ignore", category=geometrics.GeometricsWarning)``.
    """


def ensure_dataframe(df: object) -> pd.DataFrame:
    """Return ``df`` as a DataFrame or raise ``TypeError``.

    Parameters
    ----------
    df
        Object expected to be a :class:`pandas.DataFrame`.

    Returns
    -------
    pandas.DataFrame
        The validated data frame (a shallow copy is *not* made).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df needs to be a pandas DataFrame")
    return df


def ensure_geodataframe(
    gdf: object, arg: str = "gdf", func: str = ""
) -> gpd.GeoDataFrame:
    """Return ``gdf`` as a GeoDataFrame or raise ``TypeError``.

    Parameters
    ----------
    gdf
        Object expected to be a :class:`geopandas.GeoDataFrame`.
    arg
        Name of the argument being validated (used in the error message).
    func
        Name of the public function reporting the error (prefixed in the message
        when non-empty).

    Returns
    -------
    geopandas.GeoDataFrame
        The validated geo data frame (a shallow copy is *not* made).
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        prefix = f"{func}: " if func else ""
        raise TypeError(f"{prefix}{arg} needs to be a geopandas GeoDataFrame")
    return gdf


def is_numeric_or_logical(series: pd.Series) -> bool:
    """Return ``True`` if ``series`` is numeric or boolean (R's numeric-or-logical)."""
    return bool(pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series))


def numeric_logical_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of columns that are numeric or boolean.

    Mirrors R's ``df[sapply(df, is.logical) | sapply(df, is.numeric)]``.
    """
    return [c for c in df.columns if is_numeric_or_logical(df[c])]


def drop_missing(
    df: pd.DataFrame,
    subset: Sequence[str],
    *,
    func: str,
    stacklevel: int = 3,
) -> pd.DataFrame:
    """Drop rows with missing values in ``subset`` and warn if any were dropped.

    A consistent, advisory replacement for a silent ``df.dropna(subset=...)``: the
    complete-case frame is returned, and when rows are lost a :class:`GeometricsWarning`
    naming the calling function, the count/percentage and the offending columns is
    emitted — in the same style as the library's sampling notices.

    Parameters
    ----------
    df
        Data frame to filter.
    subset
        Column names that must be non-missing for a row to be kept.
    func
        Name of the public function reporting the drop (prefixed in the message).
    stacklevel
        Stack level passed to :func:`warnings.warn` so the warning points at the
        user's call. The default ``3`` is correct when ``drop_missing`` is called
        directly inside a public function; pass ``4`` from a helper one frame deeper.

    Returns
    -------
    pandas.DataFrame
        The complete-case frame (rows with no missing value in ``subset``).
    """
    cols = list(subset)
    n_before = len(df)
    out = df.dropna(subset=cols)
    n_dropped = n_before - len(out)
    if n_dropped:
        pct = n_dropped / n_before if n_before else 0.0
        warnings.warn(
            f"{func}: dropped {n_dropped} of {n_before} row(s) "
            f"({pct:.0%}) with missing values in {cols}",
            GeometricsWarning,
            stacklevel=stacklevel,
        )
    return out


def required_columns(df_dict: pd.DataFrame | None) -> list[str]:
    """Return the variables a data dictionary marks as required (``can_be_na`` is ``False``).

    A required variable is one whose ``can_be_na`` flag is falsey; rows missing it are dropped
    from the analysis sample (see :func:`drop_required`). Returns ``[]`` for a ``df_dict`` that
    is ``None`` or lacks the ``can_be_na`` / ``var_name`` columns.
    """
    cols = getattr(df_dict, "columns", [])
    if df_dict is None or "can_be_na" not in cols or "var_name" not in cols:
        return []
    required = df_dict["can_be_na"].apply(
        lambda v: not bool(v) if pd.notna(v) else False
    )
    return [str(v) for v in df_dict.loc[required, "var_name"]]


def drop_required(df: pd.DataFrame, df_dict: pd.DataFrame | None) -> pd.DataFrame:
    """Drop rows missing any variable the ``df_dict`` marks required (``can_be_na`` ``False``).

    Quietly applies the dictionary's completeness contract: only variables flagged
    ``can_be_na == False`` and present in ``df`` are enforced. Returns ``df`` unchanged when no
    such variables exist (the common case — by default only the entity/time ids are required).

    Parameters
    ----------
    df
        The data frame to filter.
    df_dict
        The data dictionary describing ``df`` (or ``None``).

    Returns
    -------
    pandas.DataFrame
        The frame restricted to rows that have every required variable present.
    """
    df = ensure_dataframe(df)
    cols = [c for c in required_columns(df_dict) if c in df.columns]
    return df.dropna(subset=cols) if cols else df


def require_columns(df: pd.DataFrame, cols: Sequence[str], *, where: str) -> None:
    """Raise ``ValueError`` naming any of ``cols`` not present in ``df``.

    Parameters
    ----------
    df
        Data frame whose columns are checked.
    cols
        Column names that must all be present.
    where
        Name of the calling function, prefixed in the error message.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: column(s) not found in df: {missing}")
