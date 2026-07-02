"""Attach human-readable variable labels once, then reuse them in figures and tables.

Plots and tables read better when an axis says "Regional inequality (Gini)" rather than the
bare column name ``gini_regional``. Rather than pass a label on every call, :func:`set_labels`
stashes a ``{name: label}`` mapping on the frame's :attr:`pandas.DataFrame.attrs` and
:func:`resolve_label` reads it back, falling back to the bare name when no label is known. An
explicit ``label=`` argument always wins over the stored default, so the helper is a
convenience, never a constraint.

The mapping can be supplied directly, or extracted from a data-dictionary frame (``df_dict``):
for each variable the concise ``label`` column is used, falling back to the longer ``var_def``
description, and finally to the bare ``var_name``. Because a ``df_dict`` also tags each
column's ``type`` (entity / time / …), :func:`set_labels` can declare the panel in the same
call via ``set_panel=True``.

Note that pandas does not always propagate ``attrs`` through operations (e.g. some merges or
column selections drop it). Call :func:`set_labels` again after such steps, or simply pass the
labels explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import pandas as pd

from geometrics._panel import set_panel as _set_panel
from geometrics._roles import set_roles as _set_roles
from geometrics._validation import ensure_dataframe

__all__ = ["label_map", "resolve_label", "resolve_labels", "set_labels"]

_LABELS_KEY = "geometrics_labels"


def _is_blank(value: object) -> bool:
    """Return ``True`` for ``None``/NaN/NA/empty-or-whitespace strings."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not str(value).strip()


def _mapping_from_dict(df_dict: pd.DataFrame) -> dict[str, str]:
    """Build a ``{var_name: label}`` mapping from a data-dictionary frame.

    Per row the concise ``label`` is preferred, then the longer ``var_def`` description; rows
    where both are blank are skipped so they fall through to the bare name at resolve time.
    """
    if "var_name" not in df_dict.columns:
        raise ValueError("df_dict must have a 'var_name' column")
    has_label = "label" in df_dict.columns
    has_var_def = "var_def" in df_dict.columns
    mapping: dict[str, str] = {}
    for _, row in df_dict.iterrows():
        candidates = []
        if has_label:
            candidates.append(row["label"])
        if has_var_def:
            candidates.append(row["var_def"])
        label = next((c for c in candidates if not _is_blank(c)), None)
        if label is not None:
            mapping[str(row["var_name"])] = str(label).strip()
    return mapping


def _panel_ids_from_dict(df_dict: pd.DataFrame) -> tuple[str | None, str | None]:
    """Extract the first ``entity`` and ``time`` column names from a ``df_dict``."""
    if "type" not in df_dict.columns:
        return None, None
    entities = list(df_dict.loc[df_dict["type"] == "entity", "var_name"])
    times = list(df_dict.loc[df_dict["type"] == "time", "var_name"])
    return (entities[0] if entities else None), (times[0] if times else None)


def _entity_name_from_dict(df_dict: pd.DataFrame) -> str | None:
    """Return the first ``var_name`` a ``df_dict`` marks ``role == "entity_name"`` (else None)."""
    if "role" not in df_dict.columns:
        return None
    names = list(df_dict.loc[df_dict["role"] == "entity_name", "var_name"])
    return str(names[0]) if names else None


def _roles_from_dict(df_dict: pd.DataFrame) -> tuple[str | None, list[str]]:
    """Return ``(outcome, covariates)`` from a ``df_dict``'s ``role`` column (else ``(None, [])``)."""
    if "role" not in df_dict.columns:
        return None, []
    outcomes = list(df_dict.loc[df_dict["role"] == "outcome", "var_name"])
    covariates = list(df_dict.loc[df_dict["role"] == "covariate", "var_name"])
    return (str(outcomes[0]) if outcomes else None), [str(c) for c in covariates]


def set_labels(
    df: pd.DataFrame,
    labels: Mapping[str, str] | pd.DataFrame | None = None,
    *,
    set_panel: bool = False,
) -> pd.DataFrame:
    """Declare human-readable variable labels on ``df`` and return it.

    The labels are stored under ``df.attrs["geometrics_labels"]`` so that subsequent
    ``explore_*`` / ``analyze_*`` calls can title axes, legends and table headers with them.
    Explicit ``label=`` arguments to those functions still take precedence.

    Parameters
    ----------
    df
        The data frame (modified in place — its ``attrs`` are updated and the same object is
        returned).
    labels
        Either a ``{column_name: label}`` mapping, or a data-dictionary frame (``df_dict``)
        whose ``label`` / ``var_def`` columns supply the labels. ``None`` leaves the stored
        mapping unchanged.
    set_panel
        When ``True`` and ``labels`` is a ``df_dict``, also declare the structural metadata it
        carries: the panel (``entity`` / ``time``, plus an ``entity_name`` column tagged
        ``role == "entity_name"``) via :func:`~geometrics.set_panel`, and the analytical roles
        (``role`` of ``outcome`` / ``covariate``) via :func:`~geometrics.set_roles`.

    Returns
    -------
    pandas.DataFrame
        The same ``df``, with ``df.attrs["geometrics_labels"]`` updated.

    Examples
    --------
    Declare labels once, then explore with readable titles:

    ```python
    import pandas as pd

    import geometrics as gm

    df = pd.DataFrame({"region": ["A", "B"], "gini": [0.42, 0.35]})
    df = gm.set_labels(df, {"gini": "Regional inequality (Gini)"})
    ```
    """
    df = ensure_dataframe(df)
    if labels is not None:
        if isinstance(labels, pd.DataFrame):
            mapping = _mapping_from_dict(labels)
            if set_panel:
                entity, time = _panel_ids_from_dict(labels)
                entity_name = _entity_name_from_dict(labels)
                _set_panel(df, entity=entity, time=time, entity_name=entity_name)
                outcome, covariates = _roles_from_dict(labels)
                if outcome is not None or covariates:
                    _set_roles(df, outcome=outcome, covariates=covariates)
        else:
            mapping = {str(k): str(v) for k, v in labels.items()}
        current = dict(df.attrs.get(_LABELS_KEY, {}))
        current.update(mapping)
        df.attrs[_LABELS_KEY] = current
    return df


def label_map(df: pd.DataFrame) -> dict[str, str]:
    """Return a copy of the ``{name: label}`` mapping stored on ``df`` (empty if none).

    Useful for handing the whole label dictionary to a renderer that relabels several terms
    at once (e.g. a regression table's coefficient rows), without changing the underlying
    raw names.
    """
    return dict(df.attrs.get(_LABELS_KEY, {}))


def resolve_label(df: pd.DataFrame, name: str, *, label: str | None = None) -> str:
    """Resolve the display label for ``name``: explicit ``label`` wins, else ``attrs``, else name.

    Never raises on an unknown ``name`` (regression terms such as ``log_gdp_pc_sq`` are not
    columns), so it is safe to call on any axis variable or model term.

    Parameters
    ----------
    df
        The data frame whose ``attrs`` may hold the stored labels.
    name
        The column or term name to label.
    label
        An explicit override; when given it is returned unchanged.

    Returns
    -------
    str
        The resolved label.
    """
    if label is not None:
        return label
    stored = df.attrs.get(_LABELS_KEY, {})
    return stored.get(name, name)


def resolve_labels(
    df: pd.DataFrame,
    names: Iterable[str],
    *,
    labels: Mapping[str, str] | None = None,
) -> list[str]:
    """Resolve display labels for several ``names`` at once (see :func:`resolve_label`).

    Parameters
    ----------
    df
        The data frame whose ``attrs`` may hold the stored labels.
    names
        The column or term names to label.
    labels
        Optional per-call ``{name: label}`` overrides, taking precedence over ``df.attrs``.

    Returns
    -------
    list of str
        The resolved labels, in the order of ``names``.
    """
    overrides = labels or {}
    stored = df.attrs.get(_LABELS_KEY, {})
    return [overrides[n] if n in overrides else stored.get(n, n) for n in names]
