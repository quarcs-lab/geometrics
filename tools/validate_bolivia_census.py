#!/usr/bin/env python
"""Audit the modeled Bolivia municipal population against same-year INE counts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

GDO_COMMIT = "9a2e2b536b743700a69ca8b682ba57feb9bf7e26"
GDO_BASE = f"https://raw.githubusercontent.com/quarcs-lab/gdo4bolivia/{GDO_COMMIT}/"
INE_URL = "https://nimbus.ine.gob.bo/index.php/s/EgN8PPYD8Y4Rdig/download"
INE_HASH = "fb0fd5af8614bf0ce25ed962d2b8048b39b58839e06a631bf6e7e2b1c2a470f7"
GDO_FILES = {
    "data/map343/sdg.csv": "675179c3770f89429eb4df07b931434a869ad41b5810f3f786836dc05cc8b99c",
    "geo/units.csv": "345e64bb9ebc34a3ff070b8eab58c934b8583abf7fd933800083f43c9b8ab004",
    "geo/crosswalks/map343__department.csv": "a9ea4d233a86a21d1af8eb85cf4907181ed38276fe9d2dbe11ec3546864df52a",
    "geo/crosswalks/map343__province.csv": "02efa7fc4359329115126161fe62ad8ad55795cdad208ab108ac1dbc9fff09d4",
}
PROVINCE_ALIASES = {
    ("chuquisaca", "sudcinti"): "surcinti",
    ("cochabamba", "estebanarze"): "estebanarce",
    ("lapaz", "suryungas"): "sudyungas",
    ("potosi", "generalbernardinobilbao"): "generalbilbao",
    ("santacruz", "obisposantisteban"): "obisposantistevan",
    ("tarija", "avilez"): "aviles",
}
MUNICIPAL_ALIASES = {
    ("santacruz", "chiquitos", "sanjose"): "sanjosedechiquitos",
    ("santacruz", "sara", "santarosa"): "santarosadelsara",
    ("santacruz", "velasco", "sanignacio"): "sanignaciodevelasco",
    ("santacruz", "velasco", "sanmiguel"): "sanmigueldevelasco",
}


def verified_bytes(url: str, sha256: str, local: Path | None = None) -> bytes:
    """Read a publisher snapshot by digest without caching in the repository."""
    if local is None:
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read()
    else:
        data = local.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{local or url}: does not match pinned SHA-256 {sha256}")
    return data


def normalized(label: str) -> str:
    """Make an accent-, case-, and punctuation-insensitive key."""
    folded = (
        unicodedata.normalize("NFKD", label.casefold())
        .encode("ascii", "ignore")
        .decode()
    )
    return re.sub(r"[^a-z0-9]", "", folded)


def require(condition: bool, message: str) -> None:
    """Keep integrity checks active even under python -O."""
    if not condition:
        raise ValueError(message)


def benchmark(table: pd.DataFrame) -> dict:
    """Summarize sign and error metrics for equally weighted geographic units."""
    errors = table.model_people - table.census_people
    ape = errors.abs() / table.census_people
    return {
        "units": len(table),
        "model_people": float(table.model_people.sum()),
        "census_people": int(table.census_people.sum()),
        "national_difference_pct": float(
            100 * errors.sum() / table.census_people.sum()
        ),
        "mae_people": float(errors.abs().mean()),
        "rmse_people": float((errors.pow(2).mean()) ** 0.5),
        "median_absolute_pct_error": float(100 * ape.median()),
        "sum_absolute_error_over_census_total_pct": float(
            100 * errors.abs().sum() / table.census_people.sum()
        ),
        "spearman": float(
            table.model_people.corr(table.census_people, method="spearman")
        ),
        "units_over_25pct_absolute_error": int((ape > 0.25).sum()),
    }


def source_tables(gdo_local: Path | None, ine_local: Path | None):
    """Read and reconcile GDO's keyed Atlas data and independent INE release."""
    files = {
        key: pd.read_csv(
            io.BytesIO(
                verified_bytes(
                    GDO_BASE + key, sha, gdo_local / key if gdo_local else None
                )
            ),
            dtype={"ine_code": "string"},
        )
        for key, sha in GDO_FILES.items()
    }
    official = pd.read_excel(
        io.BytesIO(verified_bytes(INE_URL, INE_HASH, ine_local)),
        sheet_name="3090202",
        header=None,
        engine="openpyxl",
    )
    rows = official.loc[
        official[2].isin(["País", "Departamento", "Provincia", "Municipio/TIOC"])
    ]
    rows = rows.loc[~rows[6].isin(["Rural", "Urbana"])]
    require(
        rows[2].value_counts().to_dict()
        == {"Municipio/TIOC": 343, "Provincia": 113, "Departamento": 9, "País": 1},
        "Unexpected INE census hierarchy",
    )
    direct_municipal = rows.loc[rows[2].eq("Municipio/TIOC"), [3, 4, 5, 10, 13]].copy()
    direct_municipal.columns = [
        "department_name",
        "province_name",
        "ine_name",
        "ine2012",
        "ine2024",
    ]
    direct_municipal[["dk", "pk", "mk"]] = direct_municipal.iloc[:, :3].apply(
        lambda col: col.map(normalized)
    )
    direct_municipal["pk"] = [
        PROVINCE_ALIASES.get((d, p), p)
        for d, p in direct_municipal[["dk", "pk"]].itertuples(index=False, name=None)
    ]
    direct_municipal["mk"] = [
        MUNICIPAL_ALIASES.get((d, p, m), m)
        for d, p, m in direct_municipal[["dk", "pk", "mk"]].itertuples(
            index=False, name=None
        )
    ]
    units = files["geo/units.csv"]
    department_names = units.loc[units.scale.eq("department")].set_index("geo_id").name
    province_names = units.loc[units.scale.eq("province")].set_index("geo_id").name
    mun = units.loc[
        units.scale.eq("map343"),
        ["geo_id", "name", "department_id", "province_id", "mun_id", "ine_code"],
    ].copy()
    require(len(mun) == 343 and mun.geo_id.is_unique, "Unexpected GDO municipal keys")
    for scale, names in (
        ("department", department_names),
        ("province", province_names),
    ):
        xwalk = files[f"geo/crosswalks/map343__{scale}.csv"].set_index("geo_id_from")
        require(
            xwalk.index.is_unique and set(xwalk.index) == set(mun.geo_id),
            f"Bad {scale} crosswalk",
        )
        require(xwalk.weight_area.eq(1).all(), f"Nonunit {scale} crosswalk weights")
        require(
            mun.geo_id.map(xwalk.geo_id_to).eq(mun[f"{scale}_id"]).all(),
            f"Parent ID mismatch: {scale}",
        )
        mun[f"{scale}_name"] = mun[f"{scale}_id"].map(names)
    mun[["dk", "pk", "mk"]] = mun[["department_name", "province_name", "name"]].apply(
        lambda col: col.map(normalized)
    )
    direct_municipal = direct_municipal.merge(
        mun,
        on=["dk", "pk", "mk"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_ine", ""),
    )
    require(
        direct_municipal._merge.eq("both").all(),
        "INE and GDO municipalities did not match",
    )
    sdg = files["data/map343/sdg.csv"]
    require(not sdg.duplicated(["geo_id", "year"]).any(), "Duplicate GDO geo_id/year")
    require(set(sdg.year) == {2012, 2024}, "GDO ref years changed")
    require(
        sdg.groupby("year").geo_id.nunique().eq(343).all(), "Missing GDO municipality"
    )
    require(
        sdg.geo_id.map(mun.set_index("geo_id").mun_id).eq(sdg.mun_id).all(),
        "GDO INE code mismatch",
    )
    census = sdg.pivot(index="geo_id", columns="year", values="population")
    census.columns = ["gdo2012", "gdo2024"]
    cmp = direct_municipal.merge(census, on="geo_id", validate="one_to_one")
    require(
        len(cmp) == 343 and cmp.gdo2012.eq(cmp.ine2012).all(), "INE/GDO 2012 mismatch"
    )
    national = rows.loc[rows[2].eq("País"), [10, 13]].iloc[0].astype(int)
    require(
        (cmp.ine2012.sum(), cmp.ine2024.sum()) == tuple(national),
        "INE census accounting mismatch",
    )
    direct_department = rows.loc[rows[2].eq("Departamento"), [3, 10, 13]].copy()
    direct_department.columns = ["department_name", "ine2012", "ine2024"]
    direct_department = direct_department.set_index("department_name")
    dm = cmp.groupby("department_name")[["ine2012", "ine2024"]].sum().sort_index()
    require(
        dm.eq(direct_department.sort_index()).all().all(),
        "INE department accounting mismatch",
    )
    direct_province = rows.loc[rows[2].eq("Provincia"), [3, 4, 10, 13]].copy()
    direct_province.columns = ["department_name", "province_name", "ine2012", "ine2024"]
    direct_province["dept_key"] = direct_province.department_name.map(normalized)
    direct_province["province_key"] = [
        PROVINCE_ALIASES.get((d, p), p)
        for d, p in zip(
            direct_province.dept_key,
            direct_province.province_name.map(normalized),
            strict=True,
        )
    ]
    pm = cmp.groupby(["dk", "pk"])[["ine2012", "ine2024"]].sum().sort_index()
    direct_province = direct_province.set_index(
        ["dept_key", "province_key"]
    ).sort_index()[["ine2012", "ine2024"]]
    require(
        pm.index.equals(direct_province.index) and pm.eq(direct_province).all().all(),
        "INE province accounting mismatch",
    )
    return cmp


def main() -> None:
    """Run one same-year audit; save derived diagnostics without geometry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adm3",
        required=True,
        type=Path,
        help="Output bolivia_adm3.csv from the builder",
    )
    parser.add_argument(
        "--gdo-local",
        type=Path,
        help="Optional exact local copy of the pinned GDO data tree",
    )
    parser.add_argument(
        "--ine-workbook",
        type=Path,
        help="Optional exact local copy of the INE workbook",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Local/private output directory, not the repository",
    )
    args = parser.parse_args()
    cmp = source_tables(args.gdo_local, args.ine_workbook)
    est = pd.read_csv(args.adm3)
    est = est.loc[est.year.eq(2012)].copy()
    require(
        len(est) == 343 and set(est.geo_id) == set(cmp.geo_id),
        "Unmatched 2012 model records",
    )
    est["model_people"] = est.pop_pwt * 1_000_000
    table = est.merge(
        cmp[
            [
                "geo_id",
                "department_id",
                "province_id",
                "department_name",
                "province_name",
                "ine2012",
            ]
        ],
        on="geo_id",
        validate="one_to_one",
    ).rename(columns={"ine2012": "census_people"})
    require(
        table.model_people.notna().all() and table.census_people.gt(0).all(),
        "Missing/nonpositive comparison",
    )
    parents = {}
    for scale in ("province", "department"):
        keys = [f"{scale}_id", f"{scale}_name"]
        parents[scale] = table.groupby(keys, as_index=False)[
            ["model_people", "census_people"]
        ].sum()
    baseline = table.census_people.sum()
    estimates = table.model_people.sum()
    shares_abs = (
        (table.model_people / estimates - table.census_people / baseline).abs().sum()
    )
    out = {
        "geography": "SDSN 2025 map343 and GDO nesting, same year 2012",
        "population_benchmark_year": 2012,
        "adm3_source_sha256": hashlib.sha256(args.adm3.read_bytes()).hexdigest(),
        "gdo_commit": GDO_COMMIT,
        "gdo_input_hashes": GDO_FILES,
        "ine_url": INE_URL,
        "ine_workbook_sha256": INE_HASH,
        "same_year_level_metrics": {
            "municipality": benchmark(table),
            **{scale: benchmark(parent) for scale, parent in parents.items()},
        },
        "municipal_share_total_variation_pct": float(50 * shares_abs),
        "municipal_2024_gdo_minus_ine": cmp.loc[
            cmp.gdo2024.ne(cmp.ine2024), ["geo_id", "name", "gdo2024", "ine2024"]
        ].to_dict(orient="records"),
        "caveat": "Census counts are not used in this allocation. They are not GDP, and a separate release does not rule out shared upstream census information in the population raster. No same-year comparison for 2022.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for scale, frame in {"municipality": table, **parents}.items():
        id_fields = (
            ["geo_id", "name", "coverage_fraction"]
            if scale == "municipality"
            else [f"{scale}_id", f"{scale}_name"]
        )
        result = frame[[*id_fields, "model_people", "census_people"]].copy()
        result["error_people"] = result.model_people - result.census_people
        result["error_pct"] = 100 * result.error_people / result.census_people
        result.to_csv(args.out / f"{scale}_2012.csv", index=False)
    (args.out / "summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
