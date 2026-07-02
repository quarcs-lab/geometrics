"""Private geometry infrastructure every spatial function funnels through.

Four building blocks keep the geometry handling uniform across the library:

* :func:`read_gdf` — the single entry point for user geometry (files or GeoDataFrames),
  enforcing the geometry contract (a CRS, valid non-empty geometries, a unique entity id).
* :func:`resolve_gdf_entity` — resolve the entity-id column of a GeoDataFrame the same way
  everywhere (explicit argument, stored ``attrs``, sole non-geometry column).
* :func:`ensure_metric_crs` — a projected copy for metric operations (centroid distances),
  estimating a UTM CRS by default.
* :func:`_align_cross_section` / :func:`_align_panel_wide` — the df/gdf/W alignment
  helpers. All spatial functions go through them so data rows, geometry rows and weights
  ids are always in the same order (killing the classic PySAL misalignment bug), with
  weights restricted and re-standardized whenever rows drop.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pandas.api import types as pdt

from geometrics._data_dict import _ENTITY_HINTS, _name_matches
from geometrics._panel import resolve_panel
from geometrics._validation import (
    GeometricsWarning,
    ensure_dataframe,
    ensure_geodataframe,
)

__all__ = ["ensure_metric_crs", "read_gdf", "resolve_gdf_entity"]

#: Key under which the geometry metadata is stored on ``gdf.attrs``.
_GEO_KEY = "geometrics_geo"

#: File suffixes :func:`read_gdf` accepts (pyogrio reads zipped shapefiles natively).
_READABLE_SUFFIXES = (".shp", ".zip", ".geojson", ".json", ".gpkg")

# One-time flag so the "projecting to estimated UTM" advisory fires once per process.
_UTM_WARNED = False


def _non_geometry_columns(gdf: gpd.GeoDataFrame) -> list[str]:
    """Return the column names of ``gdf`` that are not geometry columns."""
    return [
        c for c in gdf.columns if not isinstance(gdf[c].dtype, gpd.array.GeometryDtype)
    ]


def _resolve_entity_column(gdf: gpd.GeoDataFrame, entity: str | None) -> str:
    """Resolve the entity-id column for :func:`read_gdf` (explicit > sole > name hints)."""
    if entity is not None:
        if entity not in gdf.columns:
            raise KeyError(
                f"read_gdf: entity column {entity!r} not found in the geometry"
            )
        return entity
    candidates = _non_geometry_columns(gdf)
    if len(candidates) == 1:
        return candidates[0]
    hinted = [c for c in candidates if _name_matches(c, _ENTITY_HINTS)]
    if (
        len(hinted) > 1
    ):  # narrow ambiguity: prefer the candidates that uniquely key rows
        unique = [c for c in hinted if gdf[c].notna().all() and gdf[c].is_unique]
        if len(unique) == 1:
            hinted = unique
    if len(hinted) == 1:
        return hinted[0]
    raise ValueError(
        "read_gdf: could not resolve the entity id column — pass entity=... "
        f"(available columns: {candidates})"
    )


def read_gdf(
    source: gpd.GeoDataFrame | str | Path,
    *,
    entity: str | None = None,
    entity_name: str | None = None,
    layer: str | None = None,
    crs: Any | None = None,
    make_valid: bool = True,
) -> gpd.GeoDataFrame:
    """Read user geometry into a validated GeoDataFrame (the geometry entry point).

    Accepts a GeoDataFrame or a path to a shapefile (plain or zipped), GeoJSON, or
    GeoPackage, and enforces the geometry contract every spatial function relies on:
    a declared CRS, no empty/missing geometries (invalid ones repaired), and a unique
    entity id. The resolved ids are stored on ``gdf.attrs["geometrics_geo"]`` so later
    calls can omit ``entity=``.

    Parameters
    ----------
    source
        A :class:`geopandas.GeoDataFrame` (copied, never mutated) or a path to a
        ``.shp`` / ``.zip`` / ``.geojson`` / ``.json`` / ``.gpkg`` file.
    entity
        Name of the entity (unit) id column. When ``None`` it is resolved automatically:
        the sole non-geometry column if there is exactly one, else a column whose name
        matches the entity-id hints (``id`` / ``code`` / ``region`` / ...).
    entity_name
        Optional column holding a human-readable label for each unit (e.g. a district
        name next to a census code).
    layer
        Layer name for multi-layer sources (GeoPackage), forwarded to
        :func:`geopandas.read_file`.
    crs
        Coordinate reference system to *declare* (``set_crs``) when the source carries
        none. ``read_gdf`` never reprojects — use :func:`ensure_metric_crs` for that.
    make_valid
        Repair invalid geometries with :func:`shapely.make_valid` (a
        :class:`~geometrics.GeometricsWarning` reports how many were repaired).

    Returns
    -------
    geopandas.GeoDataFrame
        The validated geometry, with ``attrs["geometrics_geo"]`` recording the resolved
        ``entity`` (and ``entity_name``).

    Raises
    ------
    TypeError
        If ``source`` is neither a GeoDataFrame nor a path.
    KeyError
        If an explicit ``entity`` / ``entity_name`` column is absent.
    ValueError
        If the format is unsupported, no CRS is declared and ``crs`` is ``None``, the
        entity cannot be resolved, ids are duplicated, or geometries are empty/missing.

    Examples
    --------
    Validate an in-memory GeoDataFrame (the entity id is auto-resolved):

    ```python
    import geopandas as gpd
    from shapely.geometry import box

    from geometrics._geo import read_gdf

    gdf = gpd.GeoDataFrame(
        {"region": ["A", "B"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    validated = read_gdf(gdf)
    validated.attrs["geometrics_geo"]["entity"]
    ```
    """
    if isinstance(source, gpd.GeoDataFrame):
        gdf = source.copy()
    elif isinstance(source, str | Path):
        path = Path(source)
        if path.suffix.lower() not in _READABLE_SUFFIXES:
            raise ValueError(
                f"read_gdf: unsupported file format {path.suffix!r} — expected one of "
                f"{list(_READABLE_SUFFIXES)}"
            )
        if not path.exists():
            raise ValueError(f"read_gdf: file not found: {path}")
        gdf = gpd.read_file(path, layer=layer)
    else:
        raise TypeError(
            "read_gdf: source needs to be a geopandas GeoDataFrame or a path to a "
            "geometry file"
        )

    # CRS: declare (never reproject). A file with no CRS must be told what it is in.
    if gdf.crs is None:
        if crs is None:
            raise ValueError(
                "read_gdf: the geometry has no coordinate reference system — pass "
                "crs=... (e.g. crs='EPSG:4326') to declare it"
            )
        gdf = gdf.set_crs(crs)
    elif crs is not None:
        gdf = gdf.set_crs(crs, allow_override=True)

    # Entity resolution (before geometry hygiene so offending rows can be named by id).
    entity = _resolve_entity_column(gdf, entity)
    if entity_name is not None and entity_name not in gdf.columns:
        raise KeyError(
            f"read_gdf: entity_name column {entity_name!r} not found in the geometry"
        )

    # Geometry hygiene: no empty/missing geometries; repair invalid ones.
    geom = gdf.geometry
    bad = geom.isna() | geom.is_empty
    if bool(bad.any()):
        offending = list(gdf.loc[bad, entity].astype(str).head(5))
        raise ValueError(
            f"read_gdf: {int(bad.sum())} row(s) have empty or missing geometry "
            f"(e.g. {entity} = {offending})"
        )
    invalid = ~geom.is_valid
    if bool(invalid.any()) and make_valid:
        gdf = gdf.set_geometry(geom.make_valid())
        warnings.warn(
            f"read_gdf: repaired {int(invalid.sum())} invalid geometrie(s) with "
            "shapely make_valid",
            GeometricsWarning,
            stacklevel=2,
        )

    # Unique ids: the entity column is the join key to data and weights.
    dups = gdf[entity].duplicated()
    if bool(dups.any()):
        offending = list(gdf.loc[dups, entity].astype(str).drop_duplicates().head(5))
        raise ValueError(
            f"read_gdf: entity column {entity!r} has {int(dups.sum())} duplicate "
            f"id(s) (e.g. {offending}) — ids must be unique"
        )

    gdf.attrs[_GEO_KEY] = {"entity": entity, "entity_name": entity_name}
    return gdf


def resolve_gdf_entity(gdf: gpd.GeoDataFrame, entity: str | None = None) -> str:
    """Resolve the entity-id column of ``gdf``: explicit arg, else ``attrs``, else sole column.

    Parameters
    ----------
    gdf
        The geometry frame.
    entity
        Explicit entity column name (wins when given).

    Returns
    -------
    str
        The resolved entity column name.

    Raises
    ------
    KeyError
        If an explicit ``entity`` is not a column of ``gdf``.
    ValueError
        If no entity can be resolved.

    Examples
    --------
    Resolution falls back to the sole non-geometry column:

    ```python
    import geopandas as gpd
    from shapely.geometry import Point

    from geometrics._geo import resolve_gdf_entity

    gdf = gpd.GeoDataFrame(
        {"unit": ["a", "b"]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"
    )
    resolve_gdf_entity(gdf)
    ```
    """
    gdf = ensure_geodataframe(gdf)
    if entity is not None:
        if entity not in gdf.columns:
            raise KeyError(f"entity column {entity!r} not found in gdf")
        return entity
    stored = gdf.attrs.get(_GEO_KEY, {}).get("entity")
    if stored is not None and stored in gdf.columns:
        return str(stored)
    candidates = _non_geometry_columns(gdf)
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "could not resolve the gdf entity id — pass entity=... or load the geometry "
        f"through read_gdf() (available columns: {candidates})"
    )


def ensure_metric_crs(
    gdf: gpd.GeoDataFrame, crs: Any = "auto", *, func: str = ""
) -> gpd.GeoDataFrame:
    """Return a copy of ``gdf`` in a metric (projected) CRS for distance operations.

    Parameters
    ----------
    gdf
        The geometry frame.
    crs
        ``"auto"`` (default) estimates a suitable UTM CRS via
        :meth:`~geopandas.GeoDataFrame.estimate_utm_crs` (a one-time
        :class:`~geometrics.GeometricsWarning` announces the projection; already
        projected frames pass through unchanged). ``None`` returns ``gdf`` as-is —
        reproducing analyses that measured distances on geographic coordinates. Any
        other value is passed to :meth:`~geopandas.GeoDataFrame.to_crs`.
    func
        Name of the calling function, prefixed in messages.

    Returns
    -------
    geopandas.GeoDataFrame
        A projected copy (or ``gdf`` itself when ``crs`` is ``None``).

    Examples
    --------
    Project a geographic lattice to its estimated UTM zone:

    ```python
    import geopandas as gpd
    from shapely.geometry import box

    from geometrics._geo import ensure_metric_crs

    gdf = gpd.GeoDataFrame(
        {"unit": ["a"]}, geometry=[box(78, 20, 78.1, 20.1)], crs="EPSG:4326"
    )
    ensure_metric_crs(gdf).crs.is_projected
    ```
    """
    global _UTM_WARNED
    if crs is None:
        return gdf
    gdf = ensure_geodataframe(gdf, func=func)
    prefix = f"{func}: " if func else ""
    if isinstance(crs, str) and crs == "auto":
        if gdf.crs is None:
            raise ValueError(
                f"{prefix}the geometry has no CRS — declare one (read_gdf(..., "
                "crs=...)) before metric operations"
            )
        if gdf.crs.is_projected:
            return gdf.copy()
        utm = gdf.estimate_utm_crs()
        if not _UTM_WARNED:
            warnings.warn(
                f"{prefix}projecting to the estimated UTM CRS {utm.to_string()} for "
                "metric distances (pass crs=... to choose a projection; this notice "
                "appears once per session)",
                GeometricsWarning,
                stacklevel=2,
            )
            _UTM_WARNED = True
        return gdf.to_crs(utm)
    return gdf.to_crs(crs)


def _first_ids(values: Any, k: int = 5) -> list[str]:
    """Return up to ``k`` distinct ids of ``values`` as strings (for error messages)."""
    seen = list(dict.fromkeys(str(v) for v in values))
    return seen[:k]


def _align_cross_section(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    cols: list[str],
    *,
    entity: str | None = None,
    time: str | None = None,
    period: Any = None,
    w: Any = None,
    min_obs: int = 1,
    func: str = "",
) -> tuple[gpd.GeoDataFrame, Any, dict]:
    """Align a data cross-section to ``gdf`` (and ``w``) in gdf row order.

    The single alignment path for every cross-sectional spatial function. In order:
    resolve the panel ids, slice the requested ``period`` (defaulting to the latest,
    with a note), harmonize the join keys (exact first, then a string-normalized retry
    on zero overlap), account for unmatched ids, order the output rows by ``gdf``,
    drop incomplete cases on ``cols``, and — when rows dropped and a ``w`` is given —
    restrict the weights with ``w_subset`` and re-apply their transform so
    ``w.n`` always equals the number of output rows.

    Parameters
    ----------
    df
        Long-form data with the entity (and optionally time) ids.
    gdf
        Geometry frame carrying the entity ids (see :func:`read_gdf`).
    cols
        Data columns to attach to the geometry (complete-case enforced).
    entity, time
        Panel ids; resolved via :func:`~geometrics.resolve_panel` when ``None``.
    period
        Period to slice when the panel has a time dimension; ``None`` uses the latest
        period and records a note.
    w
        Optional ``libpysal`` weights aligned to the gdf ids; restricted and
        re-standardized when rows drop.
    min_obs
        Minimum number of complete rows required (``ValueError`` below it).
    func
        Name of the calling public function (prefixed in messages).

    Returns
    -------
    tuple
        ``(cross_section, w, info)`` where ``cross_section`` is a GeoDataFrame in gdf
        row order with ``cols`` attached, ``w`` the (possibly restricted) weights or
        ``None``, and ``info`` a dict with keys ``period``, ``notes`` (tuple of
        strings), ``n`` and ``dropped``.
    """
    df = ensure_dataframe(df)
    gdf = ensure_geodataframe(gdf, func=func)
    entity, time = resolve_panel(df, entity, time, require_entity=True)
    gdf_entity = resolve_gdf_entity(gdf)
    cols = list(dict.fromkeys(cols))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{func}: column(s) not found in df: {missing}")

    notes: list[str] = []
    sub = df

    # (b) period semantics: default to the latest period; unknown periods are an error.
    if time is not None:
        periods = sorted(pd.unique(sub[time].dropna()).tolist())
        if not periods:
            raise ValueError(f"{func}: time column {time!r} has no non-missing periods")
        if period is None:
            period = periods[-1]
            notes.append(
                f"{func}: period not specified — using the latest period ({period})"
            )
        elif period not in periods:
            raise ValueError(
                f"{func}: period {period!r} not found in {time!r}; available "
                f"periods: {periods}"
            )
        sub = sub.loc[sub[time] == period]
    dup = sub.duplicated(subset=[entity])
    if bool(dup.any()):
        sub = sub.drop_duplicates(subset=[entity], keep="first")
        notes.append(
            f"{func}: kept the first of {int(dup.sum())} duplicate row(s) per "
            f"{entity!r}" + (f" in period {period}" if time is not None else "")
        )

    # (c) key harmonization: exact join first; on zero overlap retry once with
    # string-normalized keys (str.strip) on both sides.
    df_keys = sub[entity]
    gdf_keys = gdf[gdf_entity]
    if not set(df_keys.dropna()) & set(gdf_keys.dropna()):
        norm_df = df_keys.astype(str).str.strip()
        norm_gdf = gdf_keys.astype(str).str.strip()
        if set(norm_df) & set(norm_gdf):
            df_keys, gdf_keys = norm_df, norm_gdf
            msg = (
                f"{func}: no exact overlap between df.{entity} and "
                f"gdf.{gdf_entity} ids — matched after string normalization "
                "(str.strip on both sides)"
            )
            warnings.warn(msg, GeometricsWarning, stacklevel=3)
            notes.append(msg)
        else:
            raise ValueError(
                f"{func}: df.{entity} and gdf.{gdf_entity} share no ids — sample df "
                f"ids: {_first_ids(df_keys.dropna())}; sample gdf ids: "
                f"{_first_ids(gdf_keys.dropna())}"
            )

    # (d) match accounting: name what fails to join on either side.
    df_only = set(df_keys.dropna()) - set(gdf_keys.dropna())
    gdf_only = set(gdf_keys.dropna()) - set(df_keys.dropna())
    if df_only or gdf_only:
        parts = []
        if df_only:
            parts.append(
                f"{len(df_only)} df id(s) not in gdf (e.g. {_first_ids(sorted(df_only, key=str))})"
            )
        if gdf_only:
            parts.append(
                f"{len(gdf_only)} gdf id(s) not in df (e.g. {_first_ids(sorted(gdf_only, key=str))})"
            )
        msg = f"{func}: unmatched ids — " + "; ".join(parts)
        warnings.warn(msg, GeometricsWarning, stacklevel=3)
        notes.append(msg)

    # (e) output rows in gdf order, complete-case on cols.
    values = sub[cols].copy()
    values.index = pd.Index(df_keys.to_numpy())
    keep_gdf_cols = [c for c in gdf.columns if c not in cols]
    out = gdf[keep_gdf_cols].copy()
    out["_gm_key"] = gdf_keys.to_numpy()
    out = out.loc[out["_gm_key"].isin(values.index)]
    joined = values.loc[out["_gm_key"]]
    for c in cols:
        out[c] = joined[c].to_numpy()
    out = out.drop(columns="_gm_key")

    n_matched = len(out)
    complete = out.dropna(subset=cols)
    dropped = n_matched - len(complete)
    if dropped:
        msg = (
            f"{func}: dropped {dropped} of {n_matched} matched unit(s) with missing "
            f"values in {cols}"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=3)
        notes.append(msg)

    # (f) enough complete observations to estimate anything.
    if len(complete) < min_obs:
        raise ValueError(
            f"{func}: only {len(complete)} complete observation(s) after alignment; "
            f"need at least {min_obs}"
        )

    # Restrict the weights to the kept rows and re-apply their transform.
    w_out = w
    if w is not None:
        from libpysal.weights.util import w_subset

        kept = list(complete[gdf_entity])
        if list(w.id_order) != kept:
            not_in_w = [i for i in kept if i not in set(w.id_order)]
            if not_in_w:
                raise ValueError(
                    f"{func}: w does not cover all aligned units (e.g. "
                    f"{_first_ids(not_in_w)}) — build w from the same gdf"
                )
            transform = w.transform
            w_out = w_subset(w, kept, silence_warnings=True)
            w_out.transform = transform
            meta = dict(getattr(w, "geometrics_meta", {}) or {})
            if meta:
                meta["n"] = w_out.n
                w_out.geometrics_meta = meta
            notes.append(
                f"{func}: restricted the spatial weights to the {len(kept)} aligned "
                f"unit(s) and re-applied transform {transform!r}"
            )
        assert w_out.n == len(complete)  # the alignment contract

    info = {
        "period": period,
        "notes": tuple(notes),
        "n": len(complete),
        "dropped": dropped,
    }
    return complete, w_out, info


def _align_panel_wide(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    var: str,
    *,
    w: Any,
    entity: str | None = None,
    time: str | None = None,
    func: str = "",
) -> tuple[Any, list, list, dict]:
    """Reshape a long panel to a wide ``(n, t)`` array in W row order.

    The alignment path for the space-time functions (Markov chains, Moran over time):
    the long panel is pivoted to one row per unit and one column per period, rows are
    ordered by the weights ``id_order`` (falling back to gdf order without ``w``),
    columns by sorted periods. The panel must be balanced.

    Parameters
    ----------
    df
        Long-form panel with entity, time and ``var`` columns.
    gdf
        Geometry frame carrying the entity ids.
    var
        The (numeric) variable to reshape.
    w
        ``libpysal`` weights whose ids match the gdf entities; the output rows follow
        ``w.id_order`` and ``w.n`` must equal the number of rows.
    entity, time
        Panel ids; resolved via :func:`~geometrics.resolve_panel` when ``None``.
    func
        Name of the calling public function (prefixed in messages).

    Returns
    -------
    tuple
        ``(values, ids, periods, info)`` — the ``(n, t)`` float array, the row ids,
        the sorted periods, and an info dict with keys ``notes`` (tuple), ``n`` and
        ``n_periods``.
    """
    df = ensure_dataframe(df)
    gdf = ensure_geodataframe(gdf, func=func)
    entity, time = resolve_panel(
        df, entity, time, require_entity=True, require_time=True
    )
    gdf_entity = resolve_gdf_entity(gdf)
    if var not in df.columns:
        raise KeyError(f"{func}: column {var!r} not found in df")
    if not pdt.is_numeric_dtype(df[var]):
        raise TypeError(f"{func}: {var!r} needs to be numeric")

    notes: list[str] = []
    sub = df
    dup = sub.duplicated(subset=[entity, time])
    if bool(dup.any()):
        sub = sub.drop_duplicates(subset=[entity, time], keep="first")
        notes.append(
            f"{func}: kept the first of {int(dup.sum())} duplicate "
            f"({entity!r}, {time!r}) row(s)"
        )

    gdf_ids = list(gdf[gdf_entity])
    if w is not None:
        if set(w.id_order) != set(gdf_ids):
            raise ValueError(
                f"{func}: w ids do not match gdf.{gdf_entity} ids — build w from the "
                "same gdf (e.g. make_weights(gdf))"
            )
        ids = list(w.id_order)
    else:
        ids = gdf_ids

    extra = set(sub[entity].dropna()) - set(ids)
    if extra:
        msg = (
            f"{func}: {len(extra)} df entit(ies) not in the geometry were ignored "
            f"(e.g. {_first_ids(sorted(extra, key=str))})"
        )
        warnings.warn(msg, GeometricsWarning, stacklevel=3)
        notes.append(msg)

    wide = sub.pivot(index=entity, columns=time, values=var)
    periods = sorted(wide.columns.tolist())
    wide = wide.reindex(index=ids, columns=periods)
    incomplete = [str(i) for i in wide.index[wide.isna().any(axis=1)]]
    if incomplete:
        raise ValueError(
            f"{func}: needs a balanced panel of {var!r} over periods {periods} — "
            f"{len(incomplete)} entit(ies) are incomplete (e.g. {incomplete[:5]})"
        )
    values = wide.to_numpy(dtype=float)
    if w is not None:
        assert w.n == values.shape[0]  # the alignment contract

    info = {
        "notes": tuple(notes),
        "n": values.shape[0],
        "n_periods": len(periods),
    }
    return values, ids, periods, info
