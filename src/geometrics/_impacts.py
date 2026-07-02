"""spreg impact machinery: LeSage-Pace direct/indirect/total impact decompositions.

Internal helpers shared by the spatial-model feature modules. They distill the working
replication code of the source paper (its ``run_sdm`` / ``_adi`` / ``full_rank_lag_mask``
/ ``stars`` helpers) and generalize the single-regressor impact computation to every
regressor of a fitted :mod:`spreg` model.

Everything is computed from ``model.betas`` and ``model.vm`` with terms located by their
``model.name_x`` labels — impacts are never scraped from printed summaries and indices
are never derived by positional arithmetic.

Parameter layout of the installed spreg (1.9.0), verified at runtime — the paper's
notebooks were pinned to spreg 1.8.5 and located terms positionally instead:

- ``ML_Lag(slx_lags=1)`` (SDM): ``name_x`` is ``['CONSTANT', <x names>, 'W_<x name>'``
  for each spatially lagged column, ``'W_<name_y>']``. ``betas`` stacks the same order
  with rho (labelled ``'W_<name_y>'``) last, and ``vm`` is
  ``(len(betas), len(betas))`` **including** the rho row/column.
- ``OLS(slx_lags=1)`` (SLX): ``name_x`` is ``['CONSTANT', <x>, 'W_<x>']`` — no spatial
  parameter.
- ``ML_Error(slx_lags=1)`` (Durbin error): ``name_x`` ends with ``'lambda'``;
  ``betas`` / ``vm`` include it.
- ``GM_Lag(slx_lags=1)``: ``name_x`` omits the lagged-dependent label
  (``len(name_x) == len(betas) - 1``) but rho is still the last row of ``betas`` and
  ``vm`` includes it.

The spatially-lagged-regressor prefix is ``'W_'`` when ``slx_lags=1`` (higher lag
orders, which this package does not use, switch to ``'W2_'`` etc.).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "full_rank_lag_mask",
    "adi",
    "locate_terms",
    "mc_impacts",
    "analytic_slx_impacts",
    "stars",
]

#: Prefix spreg 1.9 puts on spatially lagged regressor names when ``slx_lags=1``.
_WX_PREFIX = "W_"

#: Truncation order of the 'power' series approximation (the notebooks' ``_P_POW``).
_POWER_ORDER = 30

_IMPACT_COLUMNS = [
    "term",
    "direct",
    "se_direct",
    "indirect",
    "se_indirect",
    "total",
    "se_total",
]


def full_rank_lag_mask(x: np.ndarray, w_dense: np.ndarray) -> list[bool]:
    """Return a boolean mask over the columns of ``x`` indicating which to spatially lag.

    Greedily keeps only those ``W @ x`` columns that add rank to the stacked design
    ``[1, x, W@x]``, dropping redundant spatial lags. This reproduces Stata's automatic
    omission of collinear lagged terms (e.g. the lag of a small-state fixed-effect dummy
    whose neighbour structure makes its spatial lag collinear), so an SLX/SDM design
    built from the masked columns is full rank.

    Parameters
    ----------
    x
        Regressor matrix of shape ``(n, k)`` (no constant column).
    w_dense
        Dense ``(n, n)`` spatial-weights matrix.

    Returns
    -------
    list of bool
        ``k`` flags, ``True`` where the column's spatial lag can enter the design.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> w = np.eye(6)[list(range(1, 6)) + [0]]  # directed ring, row-standardized
    >>> x = rng.normal(size=(6, 2))
    >>> full_rank_lag_mask(x, w)
    [True, True]
    >>> dup = np.column_stack([x[:, 0], x[:, 0]])  # duplicated column
    >>> full_rank_lag_mask(dup, w)
    [True, False]
    """
    x = np.asarray(x, dtype=float)
    w_dense = np.asarray(w_dense, dtype=float)
    if x.ndim != 2:
        raise ValueError("full_rank_lag_mask: x must be a 2-D array")
    n, k = x.shape
    if w_dense.shape != (n, n):
        raise ValueError(
            f"full_rank_lag_mask: w_dense must be ({n}, {n}) to match x, "
            f"got {w_dense.shape}"
        )
    wx = w_dense @ x
    design = np.hstack([np.ones((n, 1)), x])  # constant + X always kept
    rank = np.linalg.matrix_rank(design)
    mask = [True] * k
    for j in range(k):
        test = np.hstack([design, wx[:, [j]]])
        if np.linalg.matrix_rank(test) > rank:
            design, rank = test, rank + 1
        else:
            mask[j] = False  # redundant lag -> do not lag
    return mask


def _adi_array(
    method: str,
    rhos: np.ndarray,
    eigenvalues: np.ndarray,
    n: int,
    w_dense: np.ndarray | None,
) -> np.ndarray:
    """Vectorized average direct-impact multiplier over an array of rho values."""
    rhos = np.atleast_1d(np.asarray(rhos, dtype=float))
    if method == "simple":
        return np.ones_like(rhos)
    if method == "full":
        eig = np.asarray(eigenvalues)
        if eig.shape != (n,):
            raise ValueError(
                f"adi: eigenvalues must have shape ({n},), got {eig.shape}"
            )
        # mean(diag((I - rho W)^-1)) == mean_i 1 / (1 - rho * lambda_i(W))
        return (1.0 / (1.0 - np.outer(rhos, eig))).real.mean(axis=1)
    if method == "power":
        if w_dense is None:
            raise ValueError("adi: method 'power' requires w_dense")
        w = np.asarray(w_dense, dtype=float)
        if w.shape != (n, n):
            raise ValueError(f"adi: w_dense must be ({n}, {n}), got {w.shape}")
        traces = np.array(
            [np.trace(np.linalg.matrix_power(w, p)) for p in range(1, _POWER_ORDER + 1)]
        )
        powers = rhos[:, None] ** np.arange(1, _POWER_ORDER + 1)
        return 1.0 + (powers * traces).sum(axis=1) / n
    raise ValueError(f"adi: unknown impact method {method!r}")


def adi(
    method: str,
    rho: float,
    eigenvalues: np.ndarray,
    n: int,
    w_dense: np.ndarray | None = None,
) -> float:
    """Average direct-impact multiplier of a spatial-lag model at parameter ``rho``.

    The average direct impact of regressor ``k`` in a model with spatial multiplier
    ``(I - rho W)^-1`` is ``adi * beta_k`` where ``adi = mean(diag((I - rho W)^-1))``.
    Three computations are supported, mirroring spreg's ``spat_impacts`` options:

    - ``'simple'`` — the Kim-Phipps-Anselin scalar approximation: the multiplier is
      fixed at ``1.0``.
    - ``'full'`` — the exact LeSage-Pace value via the eigenvalues of ``W``:
      ``mean_i 1 / (1 - rho * lambda_i)``.
    - ``'power'`` — the truncated power series
      ``1 + sum_p rho^p tr(W^p) / n`` (``p = 1..30``), requiring ``w_dense``.

    Parameters
    ----------
    method
        ``'simple'``, ``'full'`` or ``'power'``.
    rho
        Spatial autoregressive parameter.
    eigenvalues
        The ``n`` eigenvalues of the dense weights matrix (used by ``'full'``; may be
        complex for non-symmetric row-standardized weights).
    n
        Number of observations (rows of ``W``).
    w_dense
        Dense ``(n, n)`` weights matrix; required only for ``'power'``.

    Returns
    -------
    float
        The average direct-impact multiplier.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.eye(4)[[1, 2, 3, 0]]  # directed ring, row-standardized
    >>> eig = np.linalg.eigvals(w)
    >>> adi("simple", 0.5, eig, 4)
    1.0
    >>> expected = np.diag(np.linalg.inv(np.eye(4) - 0.5 * w)).mean()
    >>> bool(np.isclose(adi("full", 0.5, eig, 4), expected))
    True
    """
    return float(_adi_array(method, np.array([rho]), eigenvalues, int(n), w_dense)[0])


def locate_terms(model: Any, names: list[str]) -> dict[str, Any]:
    """Map regressor names to their row indices in ``model.betas`` via ``model.name_x``.

    Terms are located exclusively by their labels (never positional arithmetic), so the
    mapping is robust to spreg's per-estimator layout differences. The spatially lagged
    regressors carry the ``'W_'`` prefix (spreg 1.9, ``slx_lags=1``); the spatial-lag
    parameter rho is labelled ``'W_<name_y>'`` by ``ML_Lag`` (its last row — for
    ``GM_Lag`` the label is absent from ``name_x`` but rho is still the last beta row);
    the spatial-error parameter is labelled ``'lambda'`` by ``ML_Error``.

    For the installed spreg 1.9, ``model.vm`` covers **all** rows of ``model.betas`` —
    for ``ML_Lag`` this includes the rho row/column — which this function verifies.

    Parameters
    ----------
    model
        A fitted spreg model exposing ``betas``, ``vm``, ``name_x`` (and ``name_y``).
    names
        The regressor names, exactly as passed to the estimator's ``name_x``.

    Returns
    -------
    dict
        ``{'const': int | None, 'x': {name: int}, 'wx': {name: int}, 'rho': int |
        None, 'lam': int | None}`` where ``'wx'`` holds only the regressors whose
        spatial lag entered the model, keyed by the *unprefixed* name.

    Raises
    ------
    KeyError
        If any of ``names`` is not found in ``model.name_x``.
    ValueError
        If ``model.vm`` does not cover every row of ``model.betas``.

    Examples
    --------
    >>> import contextlib, io
    >>> import numpy as np, spreg
    >>> from libpysal.weights import lat2W
    >>> w = lat2W(4, 4, rook=True)
    >>> w.transform = "r"
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(16, 2))
    >>> y = (x @ [1.0, -0.5] + rng.normal(size=16)).reshape(-1, 1)
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     m = spreg.ML_Lag(y=y, x=x, w=w, slx_lags=1, name_x=["a", "b"], name_y="y")
    >>> loc = locate_terms(m, ["a", "b"])
    >>> loc["x"], loc["wx"], loc["rho"]
    ({'a': 1, 'b': 2}, {'a': 3, 'b': 4}, 5)
    """
    name_x = [str(nm) for nm in model.name_x]
    betas = np.asarray(model.betas).flatten()
    n_params = len(betas)
    vm = np.asarray(model.vm)
    if vm.shape != (n_params, n_params):
        raise ValueError(
            f"locate_terms: model.vm has shape {vm.shape} but model.betas has "
            f"{n_params} rows; expected a ({n_params}, {n_params}) covariance "
            "(spreg 1.9 includes the spatial parameter in vm)"
        )

    x_idx: dict[str, int] = {}
    for name in names:
        if name not in name_x:
            raise KeyError(
                f"locate_terms: regressor {name!r} not found in model.name_x {name_x}"
            )
        x_idx[name] = name_x.index(name)

    wx_idx = {
        name: name_x.index(_WX_PREFIX + name)
        for name in names
        if _WX_PREFIX + name in name_x
    }

    const = name_x.index("CONSTANT") if "CONSTANT" in name_x else None
    lam = name_x.index("lambda") if "lambda" in name_x else None

    rho: int | None = None
    rho_label = _WX_PREFIX + str(getattr(model, "name_y", ""))
    if rho_label in name_x:
        rho = name_x.index(rho_label)  # ML_Lag: rho labelled 'W_<name_y>', last
    elif n_params == len(name_x) + 1 and hasattr(model, "rho"):
        rho = n_params - 1  # GM_Lag: rho unlabelled but last in betas/vm

    return {"const": const, "x": x_idx, "wx": wx_idx, "rho": rho, "lam": lam}


def _check_slx_mask(
    x_names: Sequence[str],
    slx_mask: Sequence[bool] | None,
    wx_idx: dict[str, int],
    *,
    where: str,
) -> None:
    """Validate that ``slx_mask`` is consistent with the lags the model contains."""
    if slx_mask is None:
        return
    if len(slx_mask) != len(x_names):
        raise ValueError(
            f"{where}: slx_mask has {len(slx_mask)} entries for "
            f"{len(x_names)} regressors"
        )
    for name, keep in zip(x_names, slx_mask, strict=True):
        if keep and name not in wx_idx:
            raise ValueError(
                f"{where}: slx_mask marks {name!r} as lagged but the model has no "
                f"{_WX_PREFIX}{name} term"
            )
        if not keep and name in wx_idx:
            raise ValueError(
                f"{where}: slx_mask marks {name!r} as not lagged but the model "
                f"contains {_WX_PREFIX}{name}"
            )


def mc_impacts(
    model: Any,
    w_dense: np.ndarray,
    x_names: Sequence[str],
    *,
    has_slx: bool,
    slx_mask: Sequence[bool] | None = None,
    n_draws: int = 10_000,
    seed: int | None = None,
    method: str = "full",
) -> pd.DataFrame:
    """LeSage-Pace impact decomposition of a spatial-lag/Durbin model, per regressor.

    For each regressor ``k`` with own coefficient ``b_k``, spatial-lag coefficient
    ``g_k`` (``0`` when its lag was masked or the model has no SLX terms) and spatial
    parameter ``rho``::

        total_k    = (b_k + g_k) / (1 - rho)
        direct_k   = adi(method, rho) * b_k
        indirect_k = total_k - direct_k

    which is the paper notebooks' decomposition (matching Stata's ``estat impact``).
    Because the weights are row-standardized the total is invariant to ``method``
    (``1 / (1 - rho)``); only the direct/indirect split depends on the average
    direct-impact multiplier.

    Standard errors come from ``n_draws`` Monte-Carlo draws of the full parameter
    vector from ``N(betas, vm)``, recomputing the three impacts — including the
    multiplier at each drawn rho — per draw and reporting the standard deviations.

    Parameters
    ----------
    model
        A fitted spreg spatial-lag model (e.g. ``ML_Lag``, with or without
        ``slx_lags=1``) exposing ``betas``, ``vm``, ``name_x``.
    w_dense
        Dense ``(n, n)`` spatial-weights matrix the model was estimated with.
    x_names
        Regressor names (as passed to ``name_x``) to decompose.
    has_slx
        Whether the model includes spatially lagged regressors (SDM vs pure lag).
    slx_mask
        Optional per-regressor flags (aligned with ``x_names``) recording which lags
        entered the design (from :func:`full_rank_lag_mask`); validated against the
        model's actual ``W_`` terms.
    n_draws
        Number of Monte-Carlo parameter draws.
    seed
        Seed for the :class:`numpy.random.Generator` (draws are reproducible).
    method
        Average direct-impact multiplier method: ``'full'`` (default), ``'simple'``
        or ``'power'`` (see :func:`adi`).

    Returns
    -------
    pandas.DataFrame
        One row per regressor with columns ``term``, ``direct``, ``se_direct``,
        ``indirect``, ``se_indirect``, ``total``, ``se_total``.

    Examples
    --------
    >>> import contextlib, io
    >>> import numpy as np, spreg
    >>> from libpysal.weights import lat2W
    >>> w = lat2W(4, 4, rook=True)
    >>> w.transform = "r"
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(16, 2))
    >>> y = (x @ [1.0, -0.5] + rng.normal(size=16)).reshape(-1, 1)
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     m = spreg.ML_Lag(y=y, x=x, w=w, slx_lags=1, name_x=["a", "b"], name_y="y")
    >>> out = mc_impacts(m, w.full()[0], ["a", "b"], has_slx=True, n_draws=500, seed=1)
    >>> list(out.columns)
    ['term', 'direct', 'se_direct', 'indirect', 'se_indirect', 'total', 'se_total']
    """
    w_dense = np.asarray(w_dense, dtype=float)
    if w_dense.ndim != 2 or w_dense.shape[0] != w_dense.shape[1]:
        raise ValueError(f"mc_impacts: w_dense must be square, got {w_dense.shape}")
    if n_draws < 2:
        raise ValueError(f"mc_impacts: n_draws must be at least 2, got {n_draws}")

    names = list(x_names)
    terms = locate_terms(model, names)
    if terms["rho"] is None:
        raise ValueError(
            "mc_impacts: model has no spatial-lag parameter (rho); use "
            "analytic_slx_impacts for SLX / Durbin-error models"
        )
    wx_idx: dict[str, int] = terms["wx"] if has_slx else {}
    if not has_slx and terms["wx"]:
        raise ValueError(
            "mc_impacts: has_slx=False but the model contains spatially lagged "
            f"regressors {sorted(terms['wx'])}"
        )
    if has_slx:
        _check_slx_mask(names, slx_mask, wx_idx, where="mc_impacts")

    betas = np.asarray(model.betas).flatten().astype(float)
    vm = np.asarray(model.vm, dtype=float)
    i_rho: int = terms["rho"]
    rho = betas[i_rho]

    n = w_dense.shape[0]
    eigenvalues = np.linalg.eigvals(w_dense)
    adi_point = adi(method, float(rho), eigenvalues, n, w_dense=w_dense)

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(betas, vm, size=n_draws)
    r_draws = draws[:, i_rho]
    adi_draws = _adi_array(method, r_draws, eigenvalues, n, w_dense)

    rows: list[dict[str, Any]] = []
    for name in names:
        i_b = terms["x"][name]
        j_g = wx_idx.get(name)
        b = betas[i_b]
        g = betas[j_g] if j_g is not None else 0.0
        total = (b + g) / (1.0 - rho)
        direct = adi_point * b
        indirect = total - direct

        d_draws = draws[:, i_b]
        g_draws = draws[:, j_g] if j_g is not None else np.zeros(n_draws)
        t_draws = (d_draws + g_draws) / (1.0 - r_draws)
        deff_draws = adi_draws * d_draws
        ieff_draws = t_draws - deff_draws
        rows.append(
            {
                "term": name,
                "direct": float(direct),
                "se_direct": float(deff_draws.std()),
                "indirect": float(indirect),
                "se_indirect": float(ieff_draws.std()),
                "total": float(total),
                "se_total": float(t_draws.std()),
            }
        )
    return pd.DataFrame(rows, columns=_IMPACT_COLUMNS)


def analytic_slx_impacts(
    model: Any,
    x_names: Sequence[str],
    slx_mask: Sequence[bool] | None = None,
) -> pd.DataFrame:
    """Analytic impact table for SLX / spatial-Durbin-error models.

    Without a spatially lagged dependent variable there is no multiplier feedback, so
    impacts are read directly off the coefficients: for regressor ``k`` with own
    coefficient ``b_k`` and spatial-lag coefficient ``g_k``, ``direct = b_k`` (with its
    standard error), ``indirect = g_k`` (with its standard error) and
    ``total = b_k + g_k`` with ``var(total) = var(b_k) + var(g_k) + 2 cov(b_k, g_k)``
    from ``model.vm``. Regressors whose lag was masked (or never lagged) get
    ``indirect = NaN`` and ``total = direct``.

    Parameters
    ----------
    model
        A fitted spreg SLX-type model (``OLS`` or ``ML_Error`` with ``slx_lags=1``)
        exposing ``betas``, ``vm``, ``name_x``.
    x_names
        Regressor names (as passed to ``name_x``) to decompose.
    slx_mask
        Optional per-regressor flags (aligned with ``x_names``) recording which lags
        entered the design; validated against the model's actual ``W_`` terms.

    Returns
    -------
    pandas.DataFrame
        One row per regressor with columns ``term``, ``direct``, ``se_direct``,
        ``indirect``, ``se_indirect``, ``total``, ``se_total``.

    Examples
    --------
    >>> import contextlib, io
    >>> import numpy as np, spreg
    >>> from libpysal.weights import lat2W
    >>> w = lat2W(4, 4, rook=True)
    >>> w.transform = "r"
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(16, 2))
    >>> y = (x @ [1.0, -0.5] + rng.normal(size=16)).reshape(-1, 1)
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     m = spreg.OLS(y=y, x=x, w=w, slx_lags=1, name_x=["a", "b"], name_y="y")
    >>> out = analytic_slx_impacts(m, ["a", "b"])
    >>> list(out["term"])
    ['a', 'b']
    """
    names = list(x_names)
    terms = locate_terms(model, names)
    _check_slx_mask(names, slx_mask, terms["wx"], where="analytic_slx_impacts")

    betas = np.asarray(model.betas).flatten().astype(float)
    vm = np.asarray(model.vm, dtype=float)

    rows: list[dict[str, Any]] = []
    for name in names:
        i_b = terms["x"][name]
        j_g = terms["wx"].get(name)
        direct = betas[i_b]
        se_direct = math.sqrt(vm[i_b, i_b])
        if j_g is None:
            indirect, se_indirect = float("nan"), float("nan")
            total, se_total = direct, se_direct
        else:
            indirect = betas[j_g]
            se_indirect = math.sqrt(vm[j_g, j_g])
            total = direct + indirect
            var_total = vm[i_b, i_b] + vm[j_g, j_g] + 2.0 * vm[i_b, j_g]
            se_total = math.sqrt(max(var_total, 0.0))
        rows.append(
            {
                "term": name,
                "direct": float(direct),
                "se_direct": float(se_direct),
                "indirect": float(indirect),
                "se_indirect": float(se_indirect),
                "total": float(total),
                "se_total": float(se_total),
            }
        )
    return pd.DataFrame(rows, columns=_IMPACT_COLUMNS)


def stars(est: float, se: float) -> str:
    """Significance stars for an estimate at the 10/5/1 percent levels (z-based).

    Computes the two-sided normal p-value of ``est / se`` via :func:`math.erf` and
    returns ``'***'`` for ``p < 0.01``, ``'**'`` for ``p < 0.05``, ``'*'`` for
    ``p < 0.10`` and ``''`` otherwise (also for a non-positive or non-finite ``se``).

    Parameters
    ----------
    est
        Point estimate.
    se
        Standard error of the estimate.

    Returns
    -------
    str
        ``''``, ``'*'``, ``'**'`` or ``'***'``.

    Examples
    --------
    >>> stars(2.0, 1.0)
    '**'
    >>> stars(1.0, 1.0)
    ''
    >>> stars(1.0, 0.0)
    ''
    """
    if se <= 0 or not np.isfinite(se):
        return ""
    z = abs(est / se)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
