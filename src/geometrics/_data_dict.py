"""Infer a data dictionary (``df_dict``) from a raw DataFrame.

The bundled datasets ship a *data dictionary* — a ``df_dict`` frame describing each column's
human-readable label and its role in the panel (``entity`` / ``time`` / ``factor`` /
``logical`` / ``numeric``). :func:`build_data_dict` produces a best-guess dictionary for *any*
DataFrame, so a user who brings only a data file still gets labelled figures and panel-aware
views — and, in the ``geometrics`` apps, an editable starting point.

The result is a plain frame with the same columns the loaders return
(``var_name`` / ``var_def`` / ``label`` / ``type`` / ``role`` / ``can_be_na``), consumable
directly by :func:`~geometrics.set_labels`::

    df = gm.set_labels(df, gm.build_data_dict(df), set_panel=True)

The inference is deliberately conservative: column-name hints and dtypes pick the most likely
roles, but it is only a *guess* — pass ``entity=`` / ``time=`` to pin the panel ids, or edit
the returned frame.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd
from pandas.api import types as pdt

from geometrics._validation import ensure_dataframe

__all__ = ["build_data_dict"]

#: The six columns of a data-dictionary frame, in order.
_COLUMNS = ["var_name", "var_def", "label", "type", "role", "can_be_na"]

#: Lower-cased name tokens that hint a column is a cross-sectional (unit) identifier.
_ENTITY_HINTS = {
    "id",
    "ids",
    "code",
    "iso",
    "iso2",
    "iso3",
    "country",
    "countries",
    "nation",
    "firm",
    "company",
    "unit",
    "entity",
    "region",
    "state",
    "province",
    "prov",
    "municipality",
    "muni",
    "department",
    "dept",
    "district",
    "ticker",
    "gvkey",
    "permno",
    "cusip",
    "individual",
    "person",
    "household",
}

#: Lower-cased name tokens that hint a column holds a human-readable *entity name* (a label
#: for the unit, e.g. a country/province name), used to pick the entity-name column.
_NAME_TOKENS = {
    "name",
    "names",
    "country",
    "countries",
    "nation",
    "province",
    "prov",
    "region",
    "state",
    "district",
    "municipality",
    "muni",
    "department",
    "dept",
    "firm",
    "company",
    "person",
    "household",
    "individual",
    "label",
    "title",
}

#: Lower-cased name tokens that hint a column is a *code/id* (the opposite of a readable name).
_CODE_TOKENS = {
    "id",
    "ids",
    "code",
    "iso",
    "iso2",
    "iso3",
    "ticker",
    "gvkey",
    "permno",
    "cusip",
    "key",
    "num",
    "no",
    "gid",
}

#: Lower-cased name tokens that hint a column is the time identifier.
_TIME_HINTS = {
    "year",
    "yr",
    "date",
    "time",
    "period",
    "quarter",
    "qtr",
    "month",
    "week",
    "wave",
    "fyear",
    "datadate",
}


def _tokens(name: object) -> set[str]:
    """Split a column name into lower-cased word tokens."""
    return set(re.split(r"[\s_\-./]+", str(name).strip().lower())) - {""}


def _name_matches(name: object, hints: set[str]) -> bool:
    """Return ``True`` when any word token of ``name`` is one of ``hints``."""
    return bool(_tokens(name) & hints)


def _humanize(name: object) -> str:
    """Turn a column name into a title-cased display label (``gdp_pc`` -> ``Gdp Pc``)."""
    return re.sub(r"[\s_\-./]+", " ", str(name)).strip().title() or str(name)


def _looks_like_year(s: pd.Series) -> bool:
    """Return ``True`` for an integer-valued column whose values look like calendar years."""
    vals = s.dropna()
    if vals.empty or vals.nunique() < 2:
        return False
    integral = pdt.is_integer_dtype(s) or (
        pdt.is_float_dtype(s) and bool((vals % 1 == 0).all())
    )
    if not integral:
        return False
    return bool((vals >= 1500).all() and (vals <= 2200).all())


def _value_type(s: pd.Series, factor_cutoff: int) -> str:
    """Classify a non-id column as ``logical`` / ``factor`` / ``numeric``."""
    n = int(s.dropna().nunique())
    if pdt.is_bool_dtype(s) or n == 2:
        return "logical"
    if isinstance(s.dtype, pd.CategoricalDtype) or pdt.is_object_dtype(s):
        return "factor"
    if pdt.is_numeric_dtype(s):
        return "factor" if 1 < n <= factor_cutoff else "numeric"
    return "factor"


def _detect_time(df: pd.DataFrame, cols: list[str]) -> str | None:
    """Detect the most likely time column, or ``None``."""
    hinted = [c for c in cols if _name_matches(c, _TIME_HINTS)]
    datetimes = [c for c in cols if pdt.is_datetime64_any_dtype(df[c])]
    yearish = [c for c in cols if _looks_like_year(df[c])]
    for group in (
        [c for c in hinted if c in yearish or c in datetimes],
        hinted,
        datetimes,
        yearish,
    ):
        if group:
            return group[0]
    return None


def _detect_entities(
    df: pd.DataFrame, cols: list[str], time_col: str | None
) -> list[str]:
    """Detect the cross-sectional identifier column(s), in column order, or ``[]``."""
    candidates = [c for c in cols if c != time_col]
    hinted = [
        c
        for c in candidates
        if _name_matches(c, _ENTITY_HINTS) and not pdt.is_float_dtype(df[c])
    ]
    if hinted:
        return hinted
    if time_col is not None:  # fall back: a column that forms a key with the time id
        for c in candidates:
            keyable = (
                pdt.is_object_dtype(df[c])
                or isinstance(df[c].dtype, pd.CategoricalDtype)
                or pdt.is_integer_dtype(df[c])
            )
            if (
                keyable
                and df[c].notna().all()
                and not df.duplicated([c, time_col]).any()
            ):
                return [c]
    return []


def _avg_len(s: pd.Series) -> float:
    """Average string length of a column's non-missing values (a name-likeness tiebreak)."""
    vals = s.dropna().astype(str)
    return float(vals.str.len().mean()) if len(vals) else 0.0


def _name_likeness(name: object, s: pd.Series) -> int:
    """Score how *name-like* (vs *code-like*) a column is — higher means more readable."""
    toks = _tokens(name)
    score = 2 * bool(toks & _NAME_TOKENS) - 2 * bool(toks & _CODE_TOKENS)
    if pdt.is_integer_dtype(s):
        score -= 1
    return score


def _detect_entity_name(
    df: pd.DataFrame, entities: list[str], time_col: str | None
) -> str | None:
    """Detect a human-readable entity-name column, or ``None``.

    A candidate is a text/categorical column that is **constant within** the primary entity id
    and **~1:1** with it (one label per unit). The most name-like candidate wins, but only when
    it is strictly more name-like than the entity id itself (so an id that is already a name —
    e.g. ``country`` paired with ``iso`` — yields ``None`` rather than a backwards label).
    """
    if not entities:
        return None
    primary = entities[0]
    n_ent = int(df[primary].dropna().nunique())
    if n_ent == 0:
        return None
    candidates: list[str] = []
    for c in df.columns:
        if c in (primary, time_col):
            continue
        s = df[c]
        if pdt.is_numeric_dtype(s) or pdt.is_bool_dtype(s):
            continue
        if not (
            pdt.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pdt.is_string_dtype(s)
        ):
            continue
        g = df[[primary, c]].dropna()
        if g.empty or (g.groupby(primary)[c].nunique() > 1).any():
            continue  # not constant within the entity
        if g[c].nunique() < 0.95 * n_ent:
            continue  # not ~1:1 with the entities
        candidates.append(c)
    if not candidates:
        return None
    key = lambda c: (_name_likeness(c, df[c]), _avg_len(df[c]))  # noqa: E731
    best = max(candidates, key=key)
    if key(best) <= (_name_likeness(primary, df[primary]), _avg_len(df[primary])):
        return None
    return best


def build_data_dict(
    df: pd.DataFrame,
    *,
    entity: str | Sequence[str] | None = None,
    time: str | None = None,
    factor_cutoff: int = 10,
) -> pd.DataFrame:
    """Infer a best-guess data dictionary (``df_dict``) for ``df``.

    Produces one row per column with an inferred ``type`` and a humanized ``label``, ready to
    pass to :func:`~geometrics.set_labels`. Column-name hints and dtypes drive the guess: a
    column is typed ``entity`` (name hints like ``country`` / ``iso`` / ``id``, or — failing
    that — the column that uniquely keys the rows together with the time id), ``time`` (name
    hints like ``year`` / ``date``, a datetime dtype, or an integer column in the calendar-year
    range), ``logical`` (boolean or two-valued), ``factor`` (categorical/object, or numeric
    with at most ``factor_cutoff`` distinct values), else ``numeric``.

    A best-guess ``role`` is also filled: a text column that is constant within the entity and
    ~1:1 with it (a readable label for the unit, e.g. a country name beside an ISO code) is
    tagged ``entity_name``; all other rows are left blank. The analytical roles ``outcome`` /
    ``covariate`` are never guessed — mark them yourself (in the dictionary or via
    :func:`~geometrics.set_roles`).

    Parameters
    ----------
    df
        The data frame to describe.
    entity
        Explicit entity (unit) identifier column name(s); when given, these win over
        detection (and are validated against ``df``).
    time
        Explicit time identifier column name; when given, it wins over detection.
    factor_cutoff
        Numeric columns with at most this many distinct values are typed ``factor``.

    Returns
    -------
    pandas.DataFrame
        A dictionary frame with columns ``var_name``, ``var_def``, ``label``, ``type``,
        ``role`` and ``can_be_na`` (one row per column of ``df``, in column order).

    Examples
    --------
    Build a dictionary for any frame, then attach labels + declare the panel in one step:

    ```python
    import pandas as pd

    import geometrics as gm

    df = pd.DataFrame(
        {
            "region": ["A", "A", "B", "B"],
            "year": [2000, 2001, 2000, 2001],
            "gdp_pc": [1.0, 1.1, 2.0, 2.1],
        }
    )
    ddict = gm.build_data_dict(df)
    df = gm.set_labels(df, ddict, set_panel=True)
    ddict.head()
    ```
    """
    df = ensure_dataframe(df)
    cols = list(df.columns)

    explicit_entities = [entity] if isinstance(entity, str) else list(entity or [])
    for col in (*explicit_entities, *([time] if time is not None else [])):
        if col not in cols:
            raise ValueError(f"column {col!r} is not in df")

    time_col = time if time is not None else _detect_time(df, cols)
    entities = explicit_entities or _detect_entities(df, cols, time_col)
    entity_set = set(entities)
    name_col = _detect_entity_name(df, entities, time_col)

    rows = []
    for col in cols:
        if col in entity_set:
            typ = "entity"
        elif col == time_col:
            typ = "time"
        else:
            typ = _value_type(df[col], factor_cutoff)
        label = _humanize(col)
        rows.append(
            {
                "var_name": col,
                "var_def": label,
                "label": label,
                "type": typ,
                "role": "entity_name" if col == name_col else "",
                "can_be_na": typ not in ("entity", "time"),
            }
        )

    out = pd.DataFrame(rows, columns=_COLUMNS)
    out["can_be_na"] = out["can_be_na"].astype(bool)
    return out
