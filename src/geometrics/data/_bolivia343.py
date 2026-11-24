"""Pure area interpolation from Bolivia GRID to fixed SDSN map343 boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

_EQUAL_AREA = "EPSG:6933"
_AREA_EPS_M2 = 1.0
_AREA_REL_TOL = 1e-8
_VALUES = ("gdp_pwt", "pop_pwt", "is_cell_censored")


@dataclass
class Bolivia343Build:
    """Constructed municipal sample and tabular diagnostics."""

    gdf: gpd.GeoDataFrame
    panel: pd.DataFrame
    ledger: pd.DataFrame
    weights: pd.DataFrame
    sensitivity: pd.DataFrame
    diagnostics: dict


def _require(df, cols, label):
    absent = set(cols).difference(df.columns)
    if absent:
        raise ValueError(f"{label}: missing columns {sorted(absent)}")


def prepare_grid(raw_geometry: gpd.GeoDataFrame, raw_panel: pd.DataFrame):
    """Synthesize the compound cell IDs used by load_bolivia_grid()."""
    parts = ["cell_id", "subcell_id", "subcell_id_0_25"]
    xy = ["longitude", "latitude"]
    _require(raw_panel, parts + xy, "GRID panel")
    _require(raw_geometry, [*xy, "geometry"], "GRID geometry")
    panel = raw_panel.copy()
    if panel[parts + xy].isna().any().any():
        raise ValueError("GRID panel has missing cell key or centers")
    panel["cell"] = panel[parts].astype(str).agg(".".join, axis=1)
    centers = panel[["cell", *xy]].drop_duplicates()
    geometry = raw_geometry[[*xy, "geometry"]].merge(
        centers, on=xy, validate="one_to_one"
    )
    if (
        len(geometry) != len(raw_geometry)
        or len(geometry) != len(centers)
        or not centers.cell.is_unique
    ):
        raise ValueError("GRID panel and geometry centers must match one to one")
    return geometry[["cell", "geometry"]], panel


def canonical_grid_footprints(
    legacy_geometry: gpd.GeoDataFrame, panel: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Recover the source's regular quarter-degree envelopes by southwest corner.

    The distributed legacy GeoPackage treats longitude/latitude as centers.
    We use it only to verify the cell roster. From the authors' definition,
    each full cell spans [longitude, longitude+.25] by
    [latitude, latitude+.25]. This does not recover their irregular
    country-clipped polygons.
    """
    _require(legacy_geometry, ("cell", "geometry"), "GRID geometry")
    _require(panel, ("cell", "longitude", "latitude"), "GRID panel")
    coords = panel[["cell", "longitude", "latitude"]].drop_duplicates()
    if (
        coords["cell"].isna().any()
        or not coords["cell"].is_unique
        or coords.duplicated(["longitude", "latitude"]).any()
        or not legacy_geometry["cell"].is_unique
        or set(coords["cell"]) != set(legacy_geometry["cell"])
    ):
        raise ValueError(
            "GRID southwest coordinates must match the geometry cell roster"
        )
    x = coords["longitude"].to_numpy(dtype=float)
    y = coords["latitude"].to_numpy(dtype=float)
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any((x < -180) | (x > 179.75) | (y < -90) | (y > 89.75))
        or np.any(x * 4 != np.round(x * 4))
        or np.any(y * 4 != np.round(y * 4))
    ):
        raise ValueError(
            "GRID coordinates must be southwest corners on the quarter-degree lattice"
        )
    return gpd.GeoDataFrame(
        {"cell": coords["cell"].to_numpy()},
        geometry=shapely.box(x, y, x + 0.25, y + 0.25),
        crs="EPSG:4326",
    ).sort_values("cell", ignore_index=True)


def _project(gdf, label):
    if gdf.crs is None:
        raise ValueError(f"{label} requires a known coordinate reference system")
    result = gdf.to_crs(_EQUAL_AREA)
    geom = np.asarray(result.geometry)
    if (
        not np.all(shapely.is_valid(geom))
        or not result.geom_type.isin(["Polygon", "MultiPolygon"]).all()
        or shapely.is_empty(geom).any()
    ):
        raise ValueError(f"{label} requires valid nonempty polygons")
    area = shapely.area(geom)
    if not np.isfinite(area).all() or not np.all(area > _AREA_EPS_M2):
        raise ValueError(f"{label} has a nonpositive or tiny area")
    return geom, area


def _aggregate(a, grid_values, support):
    output = a @ grid_values
    output[~support, :] = np.nan
    return output


def _derive_pc(gdp, pop):
    """Compute per-capita from aggregate totals; undefined values remain missing."""
    gdppc = np.divide(gdp, pop, out=np.full_like(gdp, np.nan), where=pop > 0)
    logged = np.log(gdppc, where=gdppc > 0, out=np.full_like(gdppc, np.nan))
    return gdppc, logged


def _validate_inputs(grid, values, map343):
    _require(grid, ("cell", "geometry"), "GRID geometry")
    _require(values, ("cell", "year", *_VALUES), "GRID panel")
    _require(map343, ("geo_id", "name", "geometry"), "map343")
    if any(
        not len(k) or k.isna().any() or not k.is_unique
        for k in (grid.cell, map343.geo_id)
    ):
        raise ValueError(
            "Grid cell and municipal geometry identifiers must be unique and present"
        )
    if map343.name.isna().any():
        raise ValueError("Municipal labels must be present")
    if set(values.cell) != set(grid.cell) or values.duplicated(["cell", "year"]).any():
        raise ValueError("GRID must have one row per geometry cell and year")
    if values.year.isna().any() or not len(values):
        raise ValueError("GRID years must be present")
    years = np.array(sorted(values.year.unique()))
    if len(values) != len(grid) * len(years):
        raise ValueError("GRID panel is unbalanced")
    gdp, pop, censored = (values[k].to_numpy(dtype=float) for k in _VALUES)
    if (
        not np.isfinite(gdp).all()
        or not np.isfinite(pop).all()
        or np.any(gdp < 0)
        or np.any(pop < 0)
    ):
        raise ValueError("Scaled GDP and population must be finite and nonnegative")
    if not np.isin(censored, [0, 1]).all() or np.any((censored == 1) & (gdp != 0)):
        raise ValueError(
            "Invalid source censor flags or positive GDP in censored cells"
        )
    return years


def build_bolivia343(
    grid_geometry: gpd.GeoDataFrame, grid_panel: pd.DataFrame, map343: gpd.GeoDataFrame
) -> Bolivia343Build:
    """Build the panel without reading files or making network requests.

    Both GRID arguments must carry the common compound cell key. Each annual
    Bolivia-specific cell budget is split across mapped municipal overlaps
    proportional to their areas. A cell with no municipal overlap remains
    unassigned. Because the budget is already estimated for Bolivia, uncovered
    parts of partly covered rectangles do not get a fraction of the main
    budget. The full rectangle denominator is a separate sensitivity. The
    coverage indicator uses unnormalized area: it is not a population share
    or a correction for what the upstream GDP model misses.
    """
    years = _validate_inputs(grid_geometry, grid_panel, map343)
    grid = grid_geometry[["cell", "geometry"]].sort_values("cell", ignore_index=True)
    mun = map343[["geo_id", "name", "geometry"]].sort_values(
        "geo_id", ignore_index=True
    )
    grid_geoms, grid_area = _project(grid, "GRID geometry")
    mun_geoms, mun_area = _project(mun, "map343")
    tree = shapely.STRtree(mun_geoms)
    gi, mi = tree.query(grid_geoms, predicate="intersects")
    area = shapely.area(shapely.intersection(grid_geoms[gi], mun_geoms[mi]))
    mask = area > _AREA_EPS_M2
    n_small = int(((area > 0) & ~mask).sum())
    gi, mi, area = gi[mask], mi[mask], area[mask]
    n_grid, n_mun = len(grid), len(mun)
    src_area = np.bincount(gi, weights=area, minlength=n_grid)
    dst_area = np.bincount(mi, weights=area, minlength=n_mun)
    if np.any(src_area > grid_area * (1 + _AREA_REL_TOL)):
        raise ValueError("Municipal footprints overlap within a GRID cell")
    if np.any(dst_area > mun_area * (1 + _AREA_REL_TOL)):
        raise ValueError("GRID cell footprints overlap within a municipality")
    m_union = float(shapely.area(shapely.union_all(mun_geoms)))
    if abs(mun_area.sum() - m_union) > m_union * _AREA_REL_TOL:
        raise ValueError("SDSN municipal geometries are not mutually disjoint in area")
    covered, support = src_area > 0, dst_area > 0
    w = np.zeros((n_mun, n_grid))
    strict = np.zeros_like(w)
    w[mi, gi] = area / src_area[gi]
    strict[mi, gi] = area / grid_area[gi]
    # At complete cells, remove roundoff above unity from the sensitivity ledger.
    over = strict.sum(axis=0) > 1
    strict[:, over] /= strict[:, over].sum(axis=0)
    np.testing.assert_allclose(w.sum(axis=0), covered, rtol=0, atol=1e-12)
    center = gpd.GeoSeries(shapely.centroid(np.asarray(grid.geometry)), crs=grid.crs)
    cg, cm = tree.query(np.asarray(center.to_crs(_EQUAL_AREA)), predicate="covered_by")
    strict_centroid = np.zeros_like(w)
    centroid_count = np.bincount(cg, minlength=n_grid)
    # Boundary ties go to lexicographically first sorted geo_id.
    order = np.lexsort((cm, cg))
    u = order[np.r_[True, np.diff(cg[order]) != 0]] if len(order) else order
    strict_centroid[cm[u], cg[u]] = 1.0
    centroid_support = strict_centroid.any(axis=1)
    matrices = (w, strict, strict_centroid)
    methods = ("covered_area", "full_footprint", "centroid")
    gdp, pop, censored = (
        grid_panel.pivot(index="cell", columns="year", values=k)
        .loc[grid.cell, years]
        .to_numpy(dtype=float)
        for k in _VALUES
    )
    n_links = (w > 0).sum(axis=1)
    n_censored = (w > 0).astype(int) @ censored.astype(int)
    fraction = np.minimum(dst_area / mun_area, 1.0)
    status = np.select(
        [~support, fraction >= 1 - _AREA_REL_TOL], ["no_support", "complete"], "partial"
    )
    gdp_agg, pop_agg = (_aggregate(w, x, support) for x in (gdp, pop))
    gdppc, ln = _derive_pc(gdp_agg, pop_agg)
    censored_area = np.zeros((n_mun, len(years)))
    np.add.at(censored_area, mi, area[:, None] * censored[gi])
    area_fraction = np.divide(
        censored_area,
        dst_area[:, None],
        out=np.full_like(censored_area, np.nan),
        where=dst_area[:, None] > 0,
    )
    pop_fraction = np.divide(
        w @ (pop * censored),
        pop_agg,
        out=np.full_like(pop_agg, np.nan),
        where=pop_agg > 0,
    )
    censor_status = np.select(
        [
            np.broadcast_to(~support[:, None], n_censored.shape),
            n_censored == 0,
            n_censored == n_links[:, None],
        ],
        ["no_support", "none", "all"],
        "mixed",
    )
    ny = len(years)
    key = {
        "geo_id": np.repeat(mun.geo_id.to_numpy(), ny),
        "year": np.tile(years, n_mun),
    }
    panel = pd.DataFrame(
        {
            "geo_id": key["geo_id"],
            "name": np.repeat(mun.name.to_numpy(), ny),
            "year": key["year"],
            "gdp_pwt": gdp_agg.ravel(),
            "pop_pwt": pop_agg.ravel(),
            "gdppc": gdppc.ravel(),
            "ln_gdppc": ln.ravel(),
            "coverage_fraction": np.repeat(fraction, ny),
            "coverage_status": np.repeat(status, ny),
            "n_source_cells": np.repeat(n_links, ny),
            "n_censored_cells": n_censored.ravel(),
            "censored_area_fraction": area_fraction.ravel(),
            "censored_pop_fraction": pop_fraction.ravel(),
            "censor_status": censor_status.ravel(),
        }
    )
    ledger, alternatives = [], []
    for a, method, spatial_support in zip(
        matrices, methods, (support, support, centroid_support), strict=False
    ):
        if method == "covered_area":
            mass_unassigned = (~covered).astype(float)
        else:
            mass_unassigned = np.maximum(1.0 - a.sum(axis=0), 0.0)
        alloc_gdp, alloc_pop = (a @ gdp).sum(axis=0), (a @ pop).sum(axis=0)
        unalloc_gdp, unalloc_pop = mass_unassigned @ gdp, mass_unassigned @ pop
        err_gdp = gdp.sum(axis=0) - alloc_gdp - unalloc_gdp
        err_pop = pop.sum(axis=0) - alloc_pop - unalloc_pop
        for residual, total in ((err_gdp, gdp.sum(axis=0)), (err_pop, pop.sum(axis=0))):
            np.testing.assert_array_less(
                np.abs(residual), np.maximum(total, 1.0) * 1e-11
            )
        ledger.append(
            pd.DataFrame(
                {
                    "year": years,
                    "method": method,
                    "source_gdp_pwt": gdp.sum(axis=0),
                    "assigned_gdp_pwt": alloc_gdp,
                    "unassigned_gdp_pwt": unalloc_gdp,
                    "gdp_residual": err_gdp,
                    "source_pop_pwt": pop.sum(axis=0),
                    "assigned_pop_pwt": alloc_pop,
                    "unassigned_pop_pwt": unalloc_pop,
                    "pop_residual": err_pop,
                }
            )
        )
        ag, ap = (_aggregate(a, x, spatial_support) for x in (gdp, pop))
        pc, _ = _derive_pc(ag, ap)
        alternatives.append(
            pd.DataFrame(
                {
                    **key,
                    "method": method,
                    "gdp_pwt": ag.ravel(),
                    "pop_pwt": ap.ravel(),
                    "gdppc": pc.ravel(),
                }
            )
        )
    weights = pd.DataFrame(
        {
            "cell": grid.cell.to_numpy()[gi],
            "geo_id": mun.geo_id.to_numpy()[mi],
            "overlap_m2": area,
            "cell_area_m2": grid_area[gi],
            "covered_area_fraction": np.minimum(src_area[gi] / grid_area[gi], 1.0),
            "weight_covered_area": w[mi, gi],
            "weight_full_footprint": strict[mi, gi],
        }
    )
    if (~covered).any():
        holes = pd.DataFrame(
            {
                "cell": grid.loc[~covered, "cell"].to_numpy(),
                "geo_id": None,
                "overlap_m2": 0.0,
                "cell_area_m2": grid_area[~covered],
                "covered_area_fraction": 0.0,
                "weight_covered_area": 0.0,
                "weight_full_footprint": 0.0,
            }
        )
        weights = pd.concat([weights, holes], ignore_index=True)
    weights = weights.sort_values(["cell", "geo_id"], ignore_index=True)
    small_coverage = covered & (src_area / grid_area < 0.01)
    diagnostics = {
        "equal_area_crs": _EQUAL_AREA,
        "overlap_cutoff_m2": _AREA_EPS_M2,
        "geometry_completion_relative_tolerance": _AREA_REL_TOL,
        "grid_cells": n_grid,
        "municipalities_and_gaioc": n_mun,
        "years": [int(y) for y in years],
        "positive_overlap_links": len(area),
        "positive_overlaps_below_cutoff": n_small,
        "no_supported_cells": int((~covered).sum()),
        "partially_covered_cells": int(
            (covered & (src_area < grid_area * (1 - _AREA_REL_TOL))).sum()
        ),
        "no_supported_municipalities": int((~support).sum()),
        "municipalities_centroid_unsupported": int((~centroid_support).sum()),
        "centroid_unassigned_cells": int((centroid_count == 0).sum()),
        "centroid_boundary_ties": int((centroid_count > 1).sum()),
        "minimum_municipal_coverage": float(fraction.min()),
        "median_municipal_coverage": float(np.median(fraction)),
        "maximum_municipal_coverage": float(fraction.max()),
        "area_coverage_whole_map": float(dst_area.sum() / mun_area.sum()),
        "cells_with_under_1pct_supported_area": int(small_coverage.sum()),
        "small_coverage_gdp_share_by_year": {
            str(year): float(share)
            for year, share in zip(
                years,
                np.divide(
                    gdp[small_coverage].sum(axis=0),
                    gdp.sum(axis=0),
                    out=np.zeros(len(years)),
                    where=gdp.sum(axis=0) > 0,
                ),
                strict=False,
            )
        },
        "small_coverage_pop_share_by_year": {
            str(year): float(share)
            for year, share in zip(
                years,
                np.divide(
                    pop[small_coverage].sum(axis=0),
                    pop.sum(axis=0),
                    out=np.zeros(len(years)),
                    where=pop.sum(axis=0) > 0,
                ),
                strict=False,
            )
        },
        "fully_censored_municipality_years": int((censor_status == "all").sum()),
    }
    return Bolivia343Build(
        gdf=mun[["geo_id", "geometry"]],
        panel=panel,
        ledger=pd.concat(ledger, ignore_index=True),
        weights=weights,
        sensitivity=pd.concat(alternatives, ignore_index=True),
        diagnostics=diagnostics,
    )
