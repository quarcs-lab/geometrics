"""Declare a panel's entity (unit) and time identifiers once, then reuse them.

Most Explore functions need both the cross-sectional **entity** id (the unit) and the
**time** id. Rather than repeat them on every call, :func:`set_panel` stashes the pair on the
frame's :attr:`pandas.DataFrame.attrs` and :func:`resolve_panel` reads them back. An explicit
argument passed to a function always wins over the stored default, so the helper is a
convenience, never a constraint.

Note that pandas does not always propagate ``attrs`` through operations (e.g. some merges or
column selections drop it). Call :func:`set_panel` again after such steps, or simply pass the
ids explicitly.
"""

from __future__ import annotations

import pandas as pd

from geometrics._validation import ensure_dataframe

__all__ = [
    "resolve_entity_name",
    "resolve_panel",
    "set_panel",
    "stored_entity_name",
    "stored_panel",
]

_PANEL_KEY = "geometrics_panel"


def stored_panel(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return the declared ``(entity, time)`` ids stored on ``df``, without validation.

    Unlike :func:`resolve_panel`, this never raises if a stored id column is absent from
    ``df`` — useful when a frame may be a column subset that dropped the ids (e.g. a
    descriptive table that should fall back to a non-panel layout rather than error).
    """
    stored = df.attrs.get(_PANEL_KEY, {})
    return stored.get("entity"), stored.get("time")


def stored_entity_name(df: pd.DataFrame) -> str | None:
    """Return the declared human-readable ``entity_name`` column on ``df``, or ``None``.

    The entity-name column holds a readable label for each unit (e.g. a country name when the
    entity id is an ISO code); figures render units as ``Name (id)`` when it is declared.
    Never raises if the column is absent from ``df``.
    """
    return df.attrs.get(_PANEL_KEY, {}).get("entity_name")


def resolve_entity_name(df: pd.DataFrame, entity_name: str | None = None) -> str | None:
    """Resolve the ``entity_name`` column for ``df``: explicit arg wins, else ``df.attrs``.

    Returns ``None`` (so callers fall back to the raw entity id) when no entity-name column is
    declared or the resolved column is absent from ``df``. Never raises.
    """
    name = entity_name if entity_name is not None else stored_entity_name(df)
    return name if (name is not None and name in df.columns) else None


def set_panel(
    df: pd.DataFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    entity_name: str | None = None,
) -> pd.DataFrame:
    """Declare the panel's ``entity``, ``time`` (and optional ``entity_name``) columns on ``df``.

    The ids are stored under ``df.attrs["geometrics_panel"]`` so that subsequent
    ``explore_*`` / ``analyze_*`` calls can omit them. Explicit arguments to those
    functions still take precedence.

    Parameters
    ----------
    df
        The panel data frame (modified in place — its ``attrs`` are updated and the same
        object is returned).
    entity
        Name of the cross-sectional (unit) identifier column, or ``None`` to leave it unset.
    time
        Name of the time identifier column, or ``None`` to leave it unset.
    entity_name
        Name of a column holding a human-readable label for each unit (e.g. ``"country"`` when
        ``entity`` is an ISO code). When declared, figures render units as ``Name (id)``.
        ``None`` leaves it unset.

    Returns
    -------
    pandas.DataFrame
        The same ``df``, with ``df.attrs["geometrics_panel"]`` updated.

    Examples
    --------
    Declare the panel once, then explore without repeating the ids:

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
    df = gm.set_panel(df, entity="region", time="year")
    ```
    """
    df = ensure_dataframe(df)
    for label, col in (
        ("entity", entity),
        ("time", time),
        ("entity_name", entity_name),
    ):
        if col is not None and col not in df.columns:
            raise ValueError(f"{label} column {col!r} is not in df")
    current = dict(df.attrs.get(_PANEL_KEY, {}))
    if entity is not None:
        current["entity"] = entity
    if time is not None:
        current["time"] = time
    if entity_name is not None:
        current["entity_name"] = entity_name
    df.attrs[_PANEL_KEY] = current
    return df


def resolve_panel(
    df: pd.DataFrame,
    entity: str | None = None,
    time: str | None = None,
    *,
    require_entity: bool = False,
    require_time: bool = False,
) -> tuple[str | None, str | None]:
    """Resolve the ``(entity, time)`` ids for ``df``: explicit args win, else ``df.attrs``.

    Parameters
    ----------
    df
        The panel data frame.
    entity, time
        Explicit identifiers. When ``None``, fall back to the values stored by
        :func:`set_panel` (if any).
    require_entity, require_time
        When ``True``, raise :class:`ValueError` if the corresponding id cannot be resolved.

    Returns
    -------
    tuple of (str or None, str or None)
        The resolved ``(entity, time)`` column names.

    Raises
    ------
    ValueError
        If a resolved column is not present in ``df``, or a required id is unresolved.
    """
    df = ensure_dataframe(df)
    stored = df.attrs.get(_PANEL_KEY, {})
    entity = entity if entity is not None else stored.get("entity")
    time = time if time is not None else stored.get("time")

    for label, col in (("entity", entity), ("time", time)):
        if col is not None and col not in df.columns:
            raise ValueError(f"{label} column {col!r} is not in df")
    if require_entity and entity is None:
        raise ValueError(
            "an entity (unit) id is required — pass entity=... or call set_panel(df, "
            "entity=...)"
        )
    if require_time and time is None:
        raise ValueError(
            "a time id is required — pass time=... or call set_panel(df, time=...)"
        )
    return entity, time
