#!/usr/bin/env python3
"""Rebuild the bundled INE-based map343 denominator from pinned public inputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from fractions import Fraction
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from validate_bolivia_census import (
    GDO_BASE,
    GDO_FILES,
    INE_HASH,
    INE_URL,
    normalized,
    require,
    source_tables,
    verified_bytes,
)

YEARBOOK_PROJECTION = "ine_yearbook_projection"
CENSUS_SHARE_INTERPOLATED = "census_share_interpolated"
POPULATION_YEARS = tuple(range(2012, 2023))
INE_PROJECTED_TOTALS = (
    10356978,
    10521247,
    10685994,
    10851103,
    11016438,
    11181861,
    11347241,
    11512468,
    11677406,
    11841955,
    12006031,
)
INE_CENSUS_2012_TOTAL = 10059856

PROJECTIONS_URL = "https://anuario.ine.gob.bo/2023/documentos/201/20112.xlsx"
PROJECTIONS_HASH = "8cd56bb4ea1c4e4d7dab83e5f1f51acc5949513751c7bebac4a771fbf8eebaa4"
BRIDGE_PATH = "geo/crosswalks/map339__map343.csv"
BRIDGE_HASH = "6288e75cd221a7266261c065a4dd06576d0da0fcf72d7de247d4bb449a40ebe4"

# Each tuple is (source map339 IDs, added map343 successor ID).
SPLITS = (
    (("BO-02-12-02",), "BO-02-12-04"),  # Mizque and Raqaypampa
    (("BO-03-06-01", "BO-03-08-02"), "BO-03-09-01"),  # San Ignacio, Santa Ana, TIM
    (("BO-07-02-01", "BO-07-02-02"), "BO-07-02-04"),  # Porco, Tomave and Yura
    (("BO-07-04-01",), "BO-07-04-05"),  # Colquechaca and Macha
)

# Explicit aliases replace differing map339 names in the yearbook, per department.
PROJECTION_ALIASES = {
    "BO-03-02-01": "Rurrenabaque",
    "BO-03-08-02": "Santa Ana del Yacuma",
    "BO-01-08-01": "Villa Alcalá",
    "BO-01-04-01": "Huacaya",
    "BO-01-04-03": "Muyupampa",
    "BO-02-03-01": "Independencia",
    "BO-02-11-02": "Toko",
    "BO-02-13-04": "Cuchumuela",
    "BO-04-10-02": "Jesús de Machaca",
    "BO-04-10-03": "San Andrés de Machaca",
    "BO-04-17-04": "La Paz",
    "BO-04-05-06": "Puerto Carabuco",
    "BO-04-10-05": "Guaqui",
    "BO-04-20-07": "Callapa",
    "BO-04-19-07": "Ancoraimes",
    "BO-05-15-01": "Santiago de Andamarca",
    "BO-05-03-04": "Soracachi",
    "BO-05-04-02": "Salinas de Garci Mendoza",
    "BO-05-14-01": "Huari",
    "BO-05-07-01": "Huayllamarca",
    "BO-05-01-02": "Quillacas",
    "BO-05-12-01": "San Pedro de Totora",
    "BO-05-10-01": "Chipaya",
    "BO-05-05-05": "Yunguyo de Litoral",
    "BO-06-02-03": "Villa Nueva - Loma Alta",
    "BO-07-13-02": "Chuquihuta",
    "BO-07-01-02": "Villa de Sacaca",
    "BO-07-15-03": "San Pablo de Lípez",
    "BO-07-03-01": "San Pedro de Buena Vista",
    "BO-08-07-01": "Ascensión de Guarayos",
    "BO-08-04-04": "Charagua",
    "BO-08-06-01": "Carmen Rivero Torrez",
    "BO-08-11-02": "Gral. Saavedra",
    "BO-08-04-06": "Gutiérrez",
    "BO-08-08-03": "San Juan de Yapacaní",
    "BO-09-05-02": "Villa San Lorenzo",
}


def yearbook(data: bytes, units: pd.DataFrame) -> tuple[dict, dict]:
    """Read exact total-person projections on the historical municipal roster."""
    with io.BytesIO(data) as stream:
        workbook = load_workbook(stream, read_only=True, data_only=True)
        rows = list(workbook["20112"].values)
        workbook.close()
    ix = {int(rows[10][j]): j for j in range(2, 35, 3) if rows[11][j] == "Total"}
    require(tuple(ix) == POPULATION_YEARS, "Yearbook projection years changed")
    dept_ids = {
        normalized(n): g
        for n, g in units.loc[
            units.scale.eq("department"), ["name", "geo_id"]
        ].itertuples(index=False, name=None)
    }
    old = units.loc[units.scale.eq("map339")]
    lookup = {
        (dept, normalized(PROJECTION_ALIASES.get(g, name))): g
        for g, dept, name in old[["geo_id", "department_id", "name"]].itertuples(
            index=False, name=None
        )
    }
    projections, totals = {}, {}
    department = None
    for row in rows[15:]:
        name = str(row[1])
        key = normalized(name)
        values = tuple(row[j] for j in ix.values())
        if name.isupper() and key in dept_ids:
            department = dept_ids[key]
            require(department not in totals, "Duplicate yearbook department")
            totals[department] = values
        elif isinstance(values[0], int):
            g = lookup[(department, key)]
            require(g not in projections, f"Duplicate yearbook municipality {g}")
            require(
                all(isinstance(v, int) and v > 0 for v in values),
                f"Invalid yearbook projections for {g}",
            )
            projections[g] = values
    require(len(projections) == len(lookup) == 339, "Yearbook roster did not match")
    require(len(totals) == 9, "Yearbook department roster did not match")
    require(
        tuple(rows[13][j] for j in ix.values()) == INE_PROJECTED_TOTALS
        and tuple(map(sum, zip(*projections.values(), strict=True)))
        == INE_PROJECTED_TOTALS,
        "Historical municipal counts do not reproduce the yearbook national totals",
    )
    for dept, wanted in totals.items():
        actual = tuple(
            map(
                sum,
                zip(*(v for k, v in projections.items() if k[:5] == dept), strict=True),
            )
        )
        require(actual == wanted, f"Historical municipal counts miss {dept} total")
    return projections, totals


def apportion(total: int, shares: dict[str, Fraction]) -> dict[str, int]:
    """Round to integer persons, assigning remaining people by residual then ID."""
    require(sum(shares.values()) == 1, "Within-group census shares do not sum to 1")
    raw = {g: total * s for g, s in shares.items()}
    result = {g: int(v) for g, v in raw.items()}
    order = sorted(result, key=lambda g: (result[g] - raw[g], g))
    remainders = total - sum(result.values())
    require(0 <= remainders < len(shares), "Invalid largest-remainder computation")
    for g in order[:remainders]:
        result[g] += 1
    require(
        sum(result.values()) == total, "Group allocation did not preserve population"
    )
    return result


def main() -> None:
    """Verify source snapshots and write derived CSV and provenance only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gdo-local", type=Path, help="Pinned public GDO snapshot root"
    )
    parser.add_argument(
        "--ine-projections", type=Path, help="Pinned INE yearbook workbook"
    )
    parser.add_argument("--ine-census", type=Path, help="Pinned INE census workbook")
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination for the derived CSV"
    )
    parser.add_argument(
        "--metadata", type=Path, required=True, help="Destination for provenance JSON"
    )
    args = parser.parse_args()

    def gdo_file(key: str, digest: str) -> pd.DataFrame:
        return pd.read_csv(
            io.BytesIO(
                verified_bytes(
                    GDO_BASE + key,
                    digest,
                    args.gdo_local / key if args.gdo_local else None,
                )
            ),
            dtype={"ine_code": "string"},
        )

    units = gdo_file("geo/units.csv", GDO_FILES["geo/units.csv"])
    bridge = gdo_file(BRIDGE_PATH, BRIDGE_HASH)
    counts = source_tables(args.gdo_local, args.ine_census).set_index("geo_id")
    targets = set(units.loc[units.scale.eq("map343"), "geo_id"])
    projections, department_totals = yearbook(
        verified_bytes(PROJECTIONS_URL, PROJECTIONS_HASH, args.ine_projections),
        units,
    )
    require(
        set(projections) < targets
        and targets - set(projections) == {new for _, new in SPLITS}
        and set(counts.index) == targets,
        "Unexpected historical/census geography roster",
    )
    group_labels: dict = {}
    panel: dict = {}
    audit = []
    for historical, successor in SPLITS:
        observed_parents = set(
            bridge.loc[
                bridge.geo_id_to.eq(successor) & bridge.weight_area.gt(0), "geo_id_from"
            ]
        )
        require(
            set(historical) == observed_parents, f"Crosswalk changed for {successor}"
        )
        keys = (*historical, successor)
        require(
            len({g[:5] for g in keys}) == 1
            and not set(keys).intersection(group_labels),
            "Group crosses a department or overlaps another group",
        )
        label = "+".join(historical)
        group_labels.update(dict.fromkeys(keys, label))
        by_anchor = {
            year: {g: int(counts.loc[g, f"ine{year}"]) for g in keys}
            for year in (2012, 2024)
        }
        for values in by_anchor.values():
            require(all(v > 0 for v in values.values()), "Missing group census shares")
        group_totals = tuple(
            map(sum, zip(*(projections[g] for g in historical), strict=True))
        )
        for yr, total in zip(POPULATION_YEARS, group_totals, strict=True):
            shares = {
                g: Fraction(2024 - yr, 12)
                * Fraction(by_anchor[2012][g], sum(by_anchor[2012].values()))
                + Fraction(yr - 2012, 12)
                * Fraction(by_anchor[2024][g], sum(by_anchor[2024].values()))
                for g in keys
            }
            panel.update({(g, yr): v for g, v in apportion(total, shares).items()})
        audit.append(
            {
                "historical_geo_ids": list(historical),
                "successor_geo_id": successor,
                "census_anchor_people_by_geo_id": by_anchor,
                "annual_projection_total_people": list(group_totals),
            }
        )
    require(
        len(group_labels) == 10 and len(panel) == 10 * len(POPULATION_YEARS),
        "Expected 10 split units",
    )
    for g, n in projections.items():
        if g not in group_labels:
            panel.update({(g, y): v for y, v in zip(POPULATION_YEARS, n, strict=True)})
    require(
        set(panel) == {(g, y) for g in targets for y in POPULATION_YEARS},
        "Harmonized output is not balanced",
    )
    people_2012 = {g: int(counts.loc[g, "ine2012"]) for g in targets}
    require(
        sum(people_2012.values()) == INE_CENSUS_2012_TOTAL,
        "2012 census accounting differs",
    )
    tbl = pd.DataFrame(
        [
            {
                "geo_id": g,
                "year": y,
                "pop_ine_harmonized": people,
                "pop_ine_census2012": people_2012[g] if y == 2012 else pd.NA,
                "pop_ine_method": CENSUS_SHARE_INTERPOLATED
                if g in group_labels
                else YEARBOOK_PROJECTION,
                "pop_ine_group": group_labels.get(g, g),
            }
            for (g, y), people in sorted(panel.items())
        ]
    )
    require(
        tuple(tbl.groupby("year").pop_ine_harmonized.sum()) == INE_PROJECTED_TOTALS,
        "Harmonization changed the national projections",
    )
    for dept, wanted in department_totals.items():
        actual = tuple(
            tbl[tbl.geo_id.str.startswith(dept + "-")]
            .groupby("year")
            .pop_ine_harmonized.sum()
        )
        require(actual == wanted, f"Harmonization changed {dept} projections")
    tbl.to_csv(args.output, index=False, lineterminator="\n")
    metadata = {
        "geography": "SDSN Bolivia 2025 map343, using pinned GDO IDs",
        "units": "people",
        "years": list(POPULATION_YEARS),
        "source_yearbook": {
            "url": PROJECTIONS_URL,
            "table": "2.01.12",
            "sha256": PROJECTIONS_HASH,
        },
        "source_census": {"url": INE_URL, "table": "3.09.02.02", "sha256": INE_HASH},
        "gdo_pinned_base": GDO_BASE,
        "gdo_inputs_sha256": {**GDO_FILES, BRIDGE_PATH: BRIDGE_HASH},
        "recipe": "tools/build_bolivia_population.py",
        "projection_method": {
            YEARBOOK_PROJECTION: "333 roster matches; INE annual population unchanged, no polygon equivalence asserted",
            CENSUS_SHARE_INTERPOLATED: "10 units in four groups; allocate summed historical projections by linear 2012-2024 group census shares, integer largest remainder with geo_id tie-break",
        },
        "census_2012_separate": "Official tabulation on matched map343 roster; present in year 2012 only",
        "annual_projected_total_people": list(INE_PROJECTED_TOTALS),
        "census_2012_total_people": INE_CENSUS_2012_TOTAL,
        "method_unit_counts": {YEARBOOK_PROJECTION: 333, CENSUS_SHARE_INTERPOLATED: 10},
        "split_groups": audit,
        "csv_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{len(tbl)} rows; projected national and department totals preserved exactly; SHA-256 {metadata['csv_sha256']}"
    )


if __name__ == "__main__":
    main()
