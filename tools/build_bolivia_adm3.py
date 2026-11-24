#!/usr/bin/env python3
"""Build map343 GDP panels locally from verified upstream geometry and GRID."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import pandas as pd

from geometrics.data import _with_bolivia343_population
from geometrics.data._bolivia343 import (
    build_bolivia343,
    canonical_grid_footprints,
    prepare_grid,
)

GRID_COMMIT = "c23a78de67db2a72c3243c43887927675b685c7c"
SDSN_REPO_COMMIT = "9a2e2b536b743700a69ca8b682ba57feb9bf7e26"
MAP343_URL = (
    "https://raw.githubusercontent.com/quarcs-lab/gdo4bolivia/"
    f"{SDSN_REPO_COMMIT}/geo/boundaries/map343.geojson"
)
MAP343_SHA256 = "19c2f1a4470b0f5642dd78850079d996593a5af64022db764f85a49391a959e3"
GRID_CSV_SHA256 = "3af614177f84322211ead2366822f1baa6ad074773ae774de1515ce26e82f2e2"
GRID_GPKG_SHA256 = "26df0f506929cc041969311fcbd69f369d24a1f99d890a05a564cf78e8971040"


def verify(path, expected):
    """Fail before construction if a source differs from the pinned bytes."""
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != expected:
        raise ValueError(f"{path.name}: expected SHA256 {expected}, found {got}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("datasets/BOL-005popAdj-PWTscaled"),
        help="Collection directory containing GRID/",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Local output directory"
    )
    parser.add_argument(
        "--map343", type=Path, help="Use local copy, still checksum verified"
    )
    args = parser.parse_args()
    gpkg = args.datasets_root / "GRID" / "bolivia_grid_cells.gpkg"
    csv = args.datasets_root / "GRID" / "bolivia_grid_cells.csv"
    verify(gpkg, GRID_GPKG_SHA256)
    verify(csv, GRID_CSV_SHA256)
    with TemporaryDirectory(prefix="geometrics-bolivia343-") as temp:
        map_path = args.map343
        if map_path is None:
            map_path = Path(temp) / "map343.geojson"
            with urllib.request.urlopen(MAP343_URL, timeout=60) as response:
                map_path.write_bytes(response.read())
        verify(map_path, MAP343_SHA256)
        map343 = gpd.read_file(map_path)
    grid, panel = prepare_grid(gpd.read_file(gpkg), pd.read_csv(csv))
    if len(grid) != 1603 or len(panel) != 1603 * 11 or len(map343) != 343:
        raise ValueError("Pinned dataset dimensions differ from their contract")
    out = build_bolivia343(canonical_grid_footprints(grid, panel), panel, map343)
    if (
        out.diagnostics["years"] != list(range(2012, 2023))
        or len(out.panel) != 343 * 11
    ):
        raise ValueError("Unexpected municipal panel dimensions or date range")
    args.output.mkdir(parents=True, exist_ok=True)
    out.gdf.to_file(args.output / "bolivia_adm3_boundaries.gpkg", index=False)
    ine_panel, dictionary = _with_bolivia343_population(out.panel)
    tables = {
        "bolivia_adm3.csv": out.panel,
        "bolivia_adm3_ine.csv": ine_panel,
        "bolivia_adm3_ine_dictionary.csv": dictionary,
        "allocation_ledger.csv": out.ledger,
        "allocation_cell_weights.csv": out.weights,
        "sensitivity.csv": out.sensitivity,
    }
    for name, table in tables.items():
        table.to_csv(args.output / name, float_format="%.16g", index=False)
    validation = {
        "sources": {
            "GRID_commit": GRID_COMMIT,
            "GRID_csv_sha256": GRID_CSV_SHA256,
            "GRID_gpkg_sha256": GRID_GPKG_SHA256,
            "map343_commit": SDSN_REPO_COMMIT,
            "map343_url": MAP343_URL,
            "map343_sha256": MAP343_SHA256,
        },
        "grid_geometry": "full quarter-degree envelope from documented southwest corner",
        "upstream_clipped_border_polygons_used": False,
        **out.diagnostics,
    }
    (args.output / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
