#!/usr/bin/env python3
"""Independently reconcile pinned Bolivia GRID, legacy GADM, and geo4bolivia geographies.

Run after installing geometrics from the checkout. Pass the three pinned GeoJSON
files in --boundaries-dir from gdo4bolivia commit 9a2e2b53's geo/boundaries/.
Outputs contain derived data; use a local output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from build_bolivia_adm3 import (
    GRID_CSV_SHA256,
    GRID_GPKG_SHA256,
    MAP343_SHA256,
    SDSN_REPO_COMMIT,
    verify,
)

from geometrics.data._bolivia343 import (
    build_bolivia343,
    canonical_grid_footprints,
    prepare_grid,
)

TOTALS = ["gdp_pwt", "pop_pwt"]
PINS = {
    "map343": MAP343_SHA256,
    "department": "41cee8a68d5cda704f14d6a975a8516d449abc07883f7ed8e3bc952d0e6b46ed",
    "province": "6711a56dee131db01fecf2551a1888ceb0605510508589774bf54cccf2509565",
}
EPSG = 6933
SIMILAR_IOU = 0.95
EXACT_IOU = 1 - 1e-8


def legacy(grid, data_root, level):
    """Reconstruct published legacy totals and assignment counts from stored IDs."""
    base = data_root / f"ADM{level}" / f"bolivia_adm{level}"
    official = pd.read_csv(base.with_suffix(".csv"))
    assignments = grid[f"adm{level}_assign"]
    if assignments.isna().any() or not set(assignments).issubset({"within", "nearest"}):
        raise ValueError(f"Unknown legacy ADM{level} assignment status")
    if grid[f"gid_{level}"].isna().any():
        raise ValueError(f"Missing legacy ADM{level} grid assignment ID")
    grouped = grid.assign(
        gid=grid[f"gid_{level}"], n_cells=1, n_cells_nearest=assignments.eq("nearest")
    ).groupby(["gid", "year"])
    cols = [*TOTALS, "n_cells", "n_cells_nearest"]
    derived = grouped[cols].sum().reset_index()
    derived["gdppc"] = derived.gdp_pwt / derived.pop_pwt.replace(0, np.nan)
    derived["ln_gdppc"] = np.log(derived.gdppc.where(derived.gdppc.gt(0)))
    cols += ["gdppc", "ln_gdppc"]
    joined = derived.merge(
        official,
        on=["gid", "year"],
        suffixes=("_rebuilt", "_original"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"Legacy ADM{level} key mismatch")
    for col in cols:
        x, y = joined[f"{col}_rebuilt"], joined[f"{col}_original"]
        if not np.allclose(x, y, atol=1e-9, rtol=1e-12, equal_nan=True):
            raise ValueError(f"Legacy ADM{level} cell reproduction failed for {col}")
        joined[f"{col}_error"] = x - y
    boundaries = base.with_name(base.name + "_boundaries.gpkg")
    geom = gpd.read_file(boundaries)[["gid", "name", "geometry"]]
    return (
        derived,
        joined,
        geom,
        {
            "units_panel": int(derived.gid.nunique()),
            "units_geometry": len(geom),
            "geometry_without_grid_assignment": sorted(
                set(geom.gid) - set(derived.gid)
            ),
            "unit_years": len(derived),
            "nearest_per_year": derived.groupby("year").n_cells_nearest.sum().to_dict(),
            "max_abs_error": {
                col: float(joined[f"{col}_error"].abs().max()) for col in cols
            },
            "legacy_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (base.with_suffix(".csv"), boundaries)
            },
        },
    )


def source_lattice_check(data):
    """Check independently that every SW coordinate decodes to its original global cell IDs."""
    index = data[
        ["cell_id", "subcell_id", "subcell_id_0_25", "longitude", "latitude"]
    ].drop_duplicates()
    lon, lat = index.longitude, index.latitude
    parent_x = (index.cell_id - 1) % 360 - 180
    parent_y = 89 - (index.cell_id - 1) // 360
    quadrant_half = (
        np.floor((lon - parent_x) * 2) * 2 + 2 - np.floor((lat - parent_y) * 2)
    )
    quadrant_quarter = (lon * 4 % 2) * 2 + 2 - (lat * 4 % 2)
    agreement = (
        (lon % 0.25).eq(0)
        & (lat % 0.25).eq(0)
        & parent_x.eq(np.floor(lon))
        & parent_y.eq(np.floor(lat))
        & index.subcell_id.eq(quadrant_half)
        & index.subcell_id_0_25.eq(quadrant_quarter)
    )
    if not agreement.all():
        raise ValueError(
            "Provider southwest coordinates do not decode to source lattice keys"
        )
    return int(agreement.sum())


def geometry(gdf):
    """Transform to equal area. Repair invalid polygons (counts are reported)."""
    gdf = gdf.to_crs(EPSG).copy()
    invalid = int((~gdf.geometry.is_valid).sum())
    if invalid:
        gdf.geometry = gdf.geometry.make_valid()
    return gdf, invalid


def geographical_comparison(legacy_gdf, geo4_gdf):
    """Return all geo4 candidate matches, without forcing a match or bijection."""
    left = geo4_gdf.rename(columns={"geo_id": "geo4_id", "name": "geo4_name"})
    right = legacy_gdf.rename(columns={"name": "gadm_name"})
    left, invalid_l = geometry(left)
    right, invalid_r = geometry(right)
    i, j = right.sindex.query(left.geometry, predicate="intersects")
    x, y = left.iloc[i].reset_index(drop=True), right.iloc[j].reset_index(drop=True)
    inter = shapely.area(shapely.intersection(x.geometry.array, y.geometry.array))
    tab = pd.DataFrame(
        {
            "geo4_id": x.geo4_id,
            "geo4_name": x.geo4_name,
            "gid": y.gid,
            "gadm_name": y.gadm_name,
            "intersection_m2": inter,
            "geo4_area_m2": x.area,
            "gadm_area_m2": y.area,
        }
    )
    tab = tab.loc[tab.intersection_m2.gt(1)].copy()
    tab["iou"] = tab.intersection_m2 / (
        tab.geo4_area_m2 + tab.gadm_area_m2 - tab.intersection_m2
    )
    tab["fraction_geo4"] = tab.intersection_m2 / tab.geo4_area_m2
    tab["fraction_gadm"] = tab.intersection_m2 / tab.gadm_area_m2
    ranked = tab.sort_values(["iou", "geo4_id", "gid"], ascending=[False, True, True])
    one = ranked.drop_duplicates("geo4_id").merge(
        ranked.drop_duplicates("gid")[["geo4_id", "gid"]].assign(mutual_best=True),
        on=["geo4_id", "gid"],
        how="left",
        validate="one_to_one",
    )
    one["mutual_best"] = one.mutual_best.eq(True)
    one["classification"] = np.select(
        [
            one.mutual_best & one.iou.ge(EXACT_IOU),
            one.mutual_best & one.iou.ge(SIMILAR_IOU),
        ],
        ["geometrically_equivalent_1e-8", "near_geometry_not_exact"],
        default="no_high_overlap_match",
    )
    one = left[["geo4_id", "geo4_name"]].merge(
        one.drop(columns="geo4_name"), on="geo4_id", how="left", validate="one_to_one"
    )
    one["classification"] = one.classification.fillna("no_intersection")
    return (
        tab,
        one,
        {
            "geo4_invalid_repaired": invalid_l,
            "gadm_invalid_repaired": invalid_r,
            "classifications": one.classification.value_counts().to_dict(),
            "min_best_iou_among_intersected": float(one.iou.min()),
            "gadm_without_selected_geo4": sorted(set(right.gid) - set(one.gid)),
        },
    )


def parent_geometry(municipal, official_parent, n):
    """Compare unions on exact hierarchical parent IDs to published geo4 polygons."""
    municipal, _ = geometry(municipal)
    official_parent, _ = geometry(official_parent)
    municipal = municipal.assign(parent_id=municipal.geo_id.str.slice(0, n))
    if set(municipal.parent_id) != set(official_parent.geo_id):
        raise ValueError("map343 prefix IDs do not exactly span the parent layer")
    union = municipal[["parent_id", "geometry"]].dissolve("parent_id").sort_index()
    pub = official_parent.set_index("geo_id").sort_index()
    tab = pd.DataFrame({"name": pub.name})
    tab["child_count"] = municipal.groupby("parent_id").size()
    difference = shapely.area(
        shapely.symmetric_difference(union.geometry.array, pub.geometry.array)
    )
    intersection = shapely.area(
        shapely.intersection(union.geometry.array, pub.geometry.array)
    )
    tab["union_area_m2"] = union.area
    tab["published_area_m2"] = pub.area
    tab["symmetric_difference_m2"] = difference
    tab["iou"] = intersection / (difference + intersection)
    return tab.reset_index(names="geo4_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets-root", type=Path, default=Path("datasets/BOL-005popAdj-PWTscaled")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, digest in PINS.items():
        path = args.boundaries_dir / f"{name}.geojson"
        verify(path, digest)
        sources[name] = gpd.read_file(path)
    csv = args.datasets_root / "GRID/bolivia_grid_cells.csv"
    gpkg = args.datasets_root / "GRID/bolivia_grid_cells.gpkg"
    verify(csv, GRID_CSV_SHA256)
    verify(gpkg, GRID_GPKG_SHA256)
    original = pd.read_csv(csv)
    grid, panel = prepare_grid(gpd.read_file(gpkg), original)
    historical_center = build_bolivia343(grid, panel, sources["map343"])
    corrected = canonical_grid_footprints(grid, panel)
    construction = build_bolivia343(corrected, panel, sources["map343"])
    paired = historical_center.panel.merge(
        construction.panel,
        on=["geo_id", "name", "year"],
        how="outer",
        suffixes=("_incorrect_center", "_corrected_sw"),
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("ADM3 historical/corrected comparison keys differ")
    for col in [*TOTALS, "gdppc"]:
        paired[f"{col}_absolute_change"] = (
            paired[f"{col}_corrected_sw"] - paired[f"{col}_incorrect_center"]
        ).abs()
        paired[f"{col}_absolute_relative_change"] = (
            paired[f"{col}_corrected_sw"]
            / paired[f"{col}_incorrect_center"].replace(0, np.nan)
            - 1
        ).abs()
    medians = paired.groupby("year")[
        [f"{col}_absolute_relative_change" for col in [*TOTALS, "gdppc"]]
    ].median()
    tables = {
        "area_ledger": construction.ledger,
        "historic_incorrect_center_area_ledger": historical_center.ledger,
        "adm3_incorrect_center_vs_corrected_sw": paired.drop(columns="_merge"),
        "adm3_corrected_sw": construction.panel,
    }
    summary = {
        "geo4_commit": SDSN_REPO_COMMIT,
        "geo4_pin_sha256": PINS,
        "area_crs_epsg": EPSG,
        "grid_coordinate_audit": {
            "source": "https://github.com/jialingzhang0730/Local_GDP_Estimates_Around_the_World/blob/329eb0af15029b9548a7046369157fa91c183906/step3_obtain_cell_level_GDP_and_predictors_data/4_get_true_cell_GDP.R#L161-L171",
            "provider_lattice_cells_verified": source_lattice_check(original),
            "legacy_geometry": "historical GPKG square wrongly centered on source coordinates",
            "corrected_geometry": "exact 0.25 degree full squares reconstructed from southwest corners; area weights by pinned map343 intersection",
            "incorrect_center": historical_center.diagnostics,
            "corrected_sw": construction.diagnostics,
            "median_absolute_relative_change_by_year": medians.to_dict(orient="index"),
        },
        "gadm_comparison_iou_near_threshold": SIMILAR_IOU,
        "gadm_comparison_iou_equivalent_tolerance": 1 - EXACT_IOU,
        "legacy": {},
        "parent_geo4": {},
        "legacy_vs_geo4_geometry": {},
        "geo4_numeric_truth_validation": "not available; reference validates geography only",
    }
    national = original.groupby("year")[TOTALS].sum()
    adm0 = pd.read_csv(args.datasets_root / "ADM0/bolivia_adm0.csv").set_index("year")[
        TOTALS
    ]
    if not np.allclose(national, adm0.reindex(national.index), rtol=1e-12, atol=1e-10):
        raise ValueError("GRID and original ADM0 do not reconcile")
    summary["grid_vs_adm0_max_absolute_error"] = (national - adm0).abs().max().to_dict()

    old_provinces = pd.read_csv(args.datasets_root / "ADM2/bolivia_adm2.csv")
    gadm_parents = old_provinces[["gid", "gid1"]].drop_duplicates()
    nested_legacy_assignments = original.merge(
        gadm_parents,
        left_on="gid_2",
        right_on="gid",
        how="left",
        validate="many_to_one",
    )
    bad_parents = int(
        nested_legacy_assignments.gid1.ne(nested_legacy_assignments.gid_1).sum()
    )
    if bad_parents:
        raise ValueError(
            f"{bad_parents} legacy ADM2 assignments contradict their ADM1 parents"
        )
    summary["legacy_hierarchy_disagreement_cell_years"] = bad_parents
    rollups = {}
    for level, name, prefix in ((1, "department", 5), (2, "province", 8)):
        label = f"adm{level}"
        old, check, gadm, validation = legacy(original, args.datasets_root, level)
        tables[f"{label}_legacy_reproduction"] = check
        summary["legacy"][label] = validation
        nested = parent_geometry(sources["map343"], sources[name], prefix)
        tables[f"{label}_map343_parent_geometry"] = nested
        summary["parent_geo4"][label] = {
            "units": len(nested),
            "min_iou": float(nested.iou.min()),
            "median_iou": float(nested.iou.median()),
            "n_equivalent_to_dissolve": int(nested.iou.ge(EXACT_IOU).sum()),
            "symmetric_difference_area_km2": float(
                nested.symmetric_difference_m2.sum() / 1e6
            ),
        }
        m = construction.panel
        agg = (
            m.assign(geo4_id=m.geo_id.str.slice(0, prefix))
            .groupby(["geo4_id", "year"], as_index=False)[TOTALS]
            .sum(min_count=1)
        )
        agg = agg.merge(
            nested[["geo4_id", "name"]], on="geo4_id", validate="many_to_one"
        )
        agg["gdppc"] = agg.gdp_pwt / agg.pop_pwt.replace(0, np.nan)
        agg["ln_gdppc"] = np.log(agg.gdppc.where(agg.gdppc.gt(0)))
        tables[f"{label}_map343_additive_rollup"] = agg
        rollups[level] = agg
        validation["legacy_vs_grid_national_max_absolute_error"] = (
            (old.groupby("year")[TOTALS].sum() - national).abs().max().to_dict()
        )
        rollup = agg.groupby("year")[TOTALS].sum()
        assigned = (
            construction.ledger.query("method == 'covered_area'")
            .set_index("year")[[f"assigned_{v}" for v in TOTALS]]
            .set_axis(TOTALS, axis=1)
        )
        if not np.allclose(
            rollup, assigned.reindex(rollup.index), atol=1e-10, rtol=1e-12
        ):
            raise ValueError("geo4 rollup failed to match assigned municipal totals")
        summary["parent_geo4"][label]["rollup_error"] = (
            (rollup - assigned).abs().max().to_dict()
        )

        overlaps, matches, comparison = geographical_comparison(gadm, sources[name])
        tables[f"{label}_geo4_vs_gadm_intersections"] = overlaps
        tables[f"{label}_geo4_vs_gadm_candidate_matches"] = matches
        summary["legacy_vs_geo4_geometry"][label] = comparison
        # Numeric diagnostics only for high-overlap candidates, never a claim of identical geography.
        eligible = matches[
            matches.classification.isin(
                ["near_geometry_not_exact", "geometrically_equivalent_1e-8"]
            )
        ]
        numeric = agg.merge(
            eligible[["geo4_id", "gid", "iou", "classification"]], on="geo4_id"
        ).merge(
            old,
            on=["gid", "year"],
            suffixes=("_geo4_area", "_gadm_legacy"),
            validate="one_to_one",
        )
        for col in [*TOTALS, "gdppc"]:
            numeric[f"{col}_relative_difference"] = (
                numeric[f"{col}_geo4_area"]
                / numeric[f"{col}_gadm_legacy"].replace(0, np.nan)
                - 1
            )
        tables[f"{label}_near_geometry_numeric_sensitivity"] = numeric
        comparison["numeric_sensitivity_matches"] = int(numeric.geo4_id.nunique())
        comparison["numeric_sensitivity_median_abs_relative_difference"] = {
            col: float(numeric[f"{col}_relative_difference"].abs().median())
            if len(numeric)
            else None
            for col in [*TOTALS, "gdppc"]
        }
    province = rollups[2]
    via_province = (
        province.assign(geo4_id=province.geo4_id.str.slice(0, 5))
        .groupby(["geo4_id", "year"])[TOTALS]
        .sum(min_count=1)
    )
    direct_department = rollups[1].set_index(["geo4_id", "year"])[TOTALS]
    if not via_province.index.equals(direct_department.index) or not np.allclose(
        via_province,
        direct_department,
        atol=1e-10,
        rtol=1e-12,
    ):
        raise ValueError("ADM3→ADM2→ADM1 and ADM3→ADM1 disagree")
    summary["geo4_hierarchy_via_adm2_max_absolute_error"] = (
        (via_province - direct_department).abs().max().to_dict()
    )
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
