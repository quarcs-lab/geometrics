"""Generate the miniature India fixtures used by the offline data tests.

Reads the source files of quarcs-lab/project2025s-py and writes:

- ``mini520.dta``: the first 6 districts (by sorted ``statedist``) of
  india520.dta, keeping only the source columns the loader touches.
- ``mini520.geojson``: the matching 6 features of india520.geojson, keeping
  ``statedist``, three other properties, and the geometry.

Usage
-----
    uv run python tests/fixtures/make_fixtures.py [--source PATH]

``--source`` may point at a clone of the paper repo (files under ``data/``)
or at a flat directory such as the pooch cache. Without it, the files are
resolved through the geometrics pooch registry (cache hit or download).

The script is deterministic: the same source files always produce
byte-identical fixtures.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

from geometrics.data import _CONTROL_COLS, _PAPER_COLS, _POP_COLS, _YEARS
from geometrics.data._registry import _fetch

HERE = Path(__file__).resolve().parent

N_DISTRICTS = 6

#: Source columns of india520.dta that the loader touches (42 columns).
SOURCE_COLUMNS = [
    "statedist",
    "state",
    "district",
    *(f"{prefix}{year}_1996_rcr_snd" for prefix in "rut" for year in _YEARS),
    *_PAPER_COLS,
    *_POP_COLS,
    *_CONTROL_COLS,
]

#: Properties of india520.geojson kept in the fixture (statedist + 3 others).
GEOJSON_COLUMNS = ["statedist", "state", "district", "Census_no", "geometry"]


def _resolve(source: Path | None, name: str) -> Path:
    """Locate a source file in a clone/cache directory or via the registry."""
    if source is None:
        return _fetch(name)
    for candidate in (source / name, source / "data" / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name!r} not found under {source} (or {source}/data)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Directory holding the paper-repo data (clone root or pooch "
        "cache). Default: fetch through the geometrics pooch registry.",
    )
    args = parser.parse_args()

    raw = pd.read_stata(_resolve(args.source, "india520.dta"))
    keep_ids = sorted(raw["statedist"])[:N_DISTRICTS]
    mini = (
        raw.loc[raw["statedist"].isin(keep_ids), SOURCE_COLUMNS]
        .sort_values("statedist")
        .reset_index(drop=True)
    )
    dta_path = HERE / "mini520.dta"
    # a fixed header timestamp keeps the fixture byte-identical across runs
    mini.to_stata(dta_path, write_index=False, time_stamp=datetime.datetime(2025, 1, 1))
    print(f"wrote {dta_path} ({mini.shape[0]} rows x {mini.shape[1]} cols)")

    gdf = gpd.read_file(_resolve(args.source, "india520.geojson"))
    mini_gdf = (
        gdf.loc[gdf["statedist"].isin(keep_ids), GEOJSON_COLUMNS]
        .sort_values("statedist")
        .reset_index(drop=True)
    )
    geojson_path = HERE / "mini520.geojson"
    mini_gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"wrote {geojson_path} ({mini_gdf.shape[0]} features)")


if __name__ == "__main__":
    main()
