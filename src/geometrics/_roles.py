"""Declare a dataset's *analytical roles* — the main outcome and covariate(s) — once.

Beyond the panel ids (:mod:`geometrics._panel`) and display labels (:mod:`geometrics._labels`),
a dataset usually has a *focus*: one variable is the **outcome** of interest and one or more are
the **covariates** (regressors / explanatory variables) you keep coming back to. Declaring that
focus once lets the figures, tables and the no-code apps **default to the key variables** instead
of the first numeric column — e.g. a scatter plots the covariate against the outcome, a
regression uses the outcome as the dependent variable and the covariates as the regressors.

Like :func:`~geometrics.set_panel` / :func:`~geometrics.set_labels`, the roles are stashed on the
frame's :attr:`pandas.DataFrame.attrs` and an explicit argument passed to a function always wins
over the stored default, so the helper is a convenience, never a constraint. The roles can also
be declared from a data dictionary (``df_dict``) via ``set_labels(df, df_dict, set_panel=True)``
when the dictionary carries a ``role`` column.

Note that pandas does not always propagate ``attrs`` through operations (e.g. some merges or
column selections drop it). Call :func:`set_roles` again after such steps, or simply pass the
variables explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from geometrics._validation import ensure_dataframe

__all__ = ["resolve_roles", "set_roles", "stored_roles"]

_ROLES_KEY = "geometrics_roles"


def stored_roles(df: pd.DataFrame) -> tuple[str | None, list[str]]:
    """Return the declared ``(outcome, covariates)`` stored on ``df``, without validation.

    Never raises if a stored role column is absent from ``df`` (a column subset may have dropped
    it) — useful for callers that should fall back to their own default rather than error.

    Returns
    -------
    tuple of (str or None, list of str)
        The stored ``(outcome, covariates)``; ``(None, [])`` when no roles are declared.
    """
    stored = df.attrs.get(_ROLES_KEY, {})
    return stored.get("outcome"), list(stored.get("covariates", []))


def set_roles(
    df: pd.DataFrame,
    *,
    outcome: str | None = None,
    covariates: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Declare the main ``outcome`` and ``covariates`` on ``df`` and return it.

    The roles are stored under ``df.attrs["geometrics_roles"]`` so that subsequent
    ``explore_*`` / ``analyze_*`` calls (and the no-code apps) can default to them when their
    primary variable argument is omitted. Explicit arguments to those functions still take
    precedence.

    Parameters
    ----------
    df
        The data frame (modified in place — its ``attrs`` are updated and the same object is
        returned).
    outcome
        Name of the main outcome (dependent) variable, or ``None`` to leave it unset.
    covariates
        Name(s) of the main covariate(s) — a single column or a sequence — or ``None`` to leave
        them unset.

    Returns
    -------
    pandas.DataFrame
        The same ``df``, with ``df.attrs["geometrics_roles"]`` updated.

    Examples
    --------
    Declare the key variables once, then explore/analyze without repeating them:

    ```python
    import pandas as pd

    import geometrics as gm

    df = pd.DataFrame(
        {
            "region": ["A", "A", "B", "B"],
            "year": [2000, 2001, 2000, 2001],
            "gini": [0.42, 0.41, 0.35, 0.34],
            "log_gdp_pc": [8.1, 8.2, 9.0, 9.1],
        }
    )
    df = gm.set_panel(df, entity="region", time="year")
    df = gm.set_roles(df, outcome="gini", covariates=["log_gdp_pc"])
    ```
    """
    df = ensure_dataframe(df)
    if outcome is not None and outcome not in df.columns:
        raise ValueError(f"outcome column {outcome!r} is not in df")
    covs = [covariates] if isinstance(covariates, str) else covariates
    if covs is not None:
        missing = [c for c in covs if c not in df.columns]
        if missing:
            raise ValueError(f"covariate column(s) not in df: {missing}")
    current = dict(df.attrs.get(_ROLES_KEY, {}))
    if outcome is not None:
        current["outcome"] = outcome
    if covs is not None:
        current["covariates"] = list(covs)
    df.attrs[_ROLES_KEY] = current
    return df


def resolve_roles(
    df: pd.DataFrame,
    outcome: str | None = None,
    covariates: str | Sequence[str] | None = None,
) -> tuple[str | None, list[str]]:
    """Resolve the ``(outcome, covariates)`` for ``df``: explicit args win, else ``df.attrs``.

    Parameters
    ----------
    df
        The data frame.
    outcome, covariates
        Explicit roles. When ``None``, fall back to the values stored by :func:`set_roles`
        (if any).

    Returns
    -------
    tuple of (str or None, list of str)
        The resolved ``(outcome, covariates)``.
    """
    df = ensure_dataframe(df)
    stored = df.attrs.get(_ROLES_KEY, {})
    outcome = outcome if outcome is not None else stored.get("outcome")
    if covariates is None:
        covariates = list(stored.get("covariates", []))
    elif isinstance(covariates, str):
        covariates = [covariates]
    return outcome, list(covariates)
