"""Case-study data loaders for geometrics.

geometrics analyses take three inputs: ``gdf`` (an ID-only geometry table),
``df`` (a long panel of observations), and ``df_dict`` (a data dictionary
describing every ``df`` column). This subpackage ships two case studies in
exactly that shape — the source files are downloaded from GitHub raw URLs
pinned to a commit, verified against SHA-256 hashes, cached locally, and
reshaped:

- the **India** nighttime lights study of `quarcs-lab/project2025s-py
  <https://github.com/quarcs-lab/project2025s-py>`_, and
- the **Bolivia** PWT-anchored subnational GDP collection
  (``BOL-005popAdj-PWTscaled``, committed under ``datasets/`` in the
  geometrics repository): the 0.25-degree gridded GDP of Rossi-Hansberg &
  Zhang (2026) rescaled so national totals equal Penn World Table 11.0, at
  department (ADM1), province (ADM2), and grid-cell level, 2012--2022.

Functions
---------
load_india
    520 Indian districts, radiance-calibrated DMSP-OLS nighttime lights panel
    (1996--2010) with 16 conditional controls.
load_india_states
    32 Indian states/union territories, corrected DMSP-OLS nighttime lights
    per capita cross-section for 1992.
load_india_raw
    The untouched source files of the district case study.
load_bolivia
    112 Bolivian provinces (ADM2), PWT-anchored GDP per capita panel
    (2012--2022).
load_bolivia_departments
    9 Bolivian departments (ADM1), same panel.
load_bolivia_grid
    1603 0.25-degree grid cells, same panel.
load_bolivia_raw
    The untouched files of any Bolivia level (including ADM0).
clear_cache
    Remove the local download cache.
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd

from geometrics.data import _registry
from geometrics.data._registry import GeometricsDataError

__all__ = [
    "load_india",
    "load_india_states",
    "load_india_raw",
    "load_bolivia",
    "load_bolivia_departments",
    "load_bolivia_grid",
    "load_bolivia_raw",
    "clear_cache",
    "GeometricsDataError",
]

#: Panel years available in the radiance-calibrated DMSP-OLS series.
_YEARS = (1996, 1999, 2000, 2004, 2005, 2010)

#: Paper-replication columns, carried verbatim from the source (never recomputed).
_PAPER_COLS = {
    "light96_rcr_cap": "ntl_pc_1996",
    "log_light96_rcr_cap": "log_ntl_pc_1996",
    "light_growth96_10rcr_cap": "growth_ntl_pc_9610",
}

#: Population columns.
_POP_COLS = {
    "pop_96": "pop_1996",
    "pop01": "pop_2001",
}

#: The 16 conditional controls of the paper, renamed to geometrics conventions.
_CONTROL_COLS = {
    "suit_mean_snd": "agri_suitability",
    "rain_mean_snd": "rainfall",
    "mala_mean_snd": "malaria",
    "temp_mean_snd": "temperature",
    "rug_mean_snd": "ruggedness",
    "distance": "dist_coast",
    "latitude": "latitude",
    "rur_percent96_rcr": "rural_share",
    "log_tot_density_rcr": "log_pop_density",
    "sc_percent96": "sc_share",
    "st_percent96": "st_share",
    "workp_percent96": "work_share",
    "lit_percent96": "literacy_share",
    "higheredu_percent96": "higher_edu_share",
    "elechh_percent96": "electricity_share",
    "log_puccaroads": "log_paved_roads",
}

#: Column order of the district panel returned by :func:`load_india`.
_PANEL_COLUMNS = [
    "statedist",
    "state",
    "district",
    "year",
    "ntl_rural",
    "ntl_urban",
    "ntl_total",
    *_PAPER_COLS.values(),
    *_POP_COLS.values(),
    *_CONTROL_COLS.values(),
]

#: Column order of the states cross-section returned by :func:`load_india_states`.
_STATES_COLUMNS = ["region", "year", "ntl_sum", "pop", "ntl_pc", "log_ntl_pc"]


def _read_bundled_dict(name: str) -> pd.DataFrame:
    """Read a data dictionary CSV bundled with the package."""
    ref = resources.files("geometrics.data").joinpath(name)
    with resources.as_file(ref) as path:
        return pd.read_csv(path)


def _ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return ``gdf`` with its CRS set to EPSG:4326 if it is missing."""
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf


def _build_india_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide india520.dta table into the long geometrics panel."""
    static = raw[
        ["statedist", "state", "district", *_PAPER_COLS, *_POP_COLS, *_CONTROL_COLS]
    ].rename(columns={**_PAPER_COLS, **_POP_COLS, **_CONTROL_COLS})
    for col in ("statedist", "state", "district"):
        static[col] = static[col].astype(str)

    frames = []
    for year in _YEARS:
        frame = static.copy()
        frame["year"] = year
        frame["ntl_rural"] = raw[f"r{year}_1996_rcr_snd"].to_numpy()
        frame["ntl_urban"] = raw[f"u{year}_1996_rcr_snd"].to_numpy()
        frame["ntl_total"] = raw[f"t{year}_1996_rcr_snd"].to_numpy()
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)[_PANEL_COLUMNS]
    df["year"] = df["year"].astype("int64")
    return df.sort_values(["statedist", "year"], ignore_index=True)


def load_india() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the India district (n=520) nighttime lights case study.

    Downloads (or reads from the local cache) the source files of
    quarcs-lab/project2025s-py, pinned to a commit, and reshapes them into
    the three geometrics inputs.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        520 district geometries with columns ``["statedist", "geometry"]``,
        CRS EPSG:4326.
    df : pandas.DataFrame
        Long panel of 3120 rows (520 districts x 6 years: 1996, 1999, 2000,
        2004, 2005, 2010). Year-varying nighttime luminosity
        (``ntl_rural``, ``ntl_urban``, ``ntl_total``), the paper-replication
        columns (``ntl_pc_1996``, ``log_ntl_pc_1996``,
        ``growth_ntl_pc_9610``; carried verbatim from the source), two
        population columns, and the 16 conditional controls (repeated per
        year). Sorted by ``statedist`` then ``year``.
    df_dict : pandas.DataFrame
        Data dictionary with one row per ``df`` column, in ``df`` column
        order, with columns
        ``var_name, var_def, label, type, role, can_be_na``.

    Raises
    ------
    GeometricsDataError
        If a source file cannot be downloaded or fails hash verification.

    See Also
    --------
    load_india_raw : The same source files without any reshaping.
    load_india_states : State-level (n=32) companion dataset.

    Examples
    --------
    >>> from geometrics.data import load_india
    >>> gdf, df, df_dict = load_india()  # doctest: +SKIP
    >>> df.shape  # doctest: +SKIP
    (3120, 28)
    """
    gdf = gpd.read_file(_registry._fetch("india520.geojson"))
    gdf = _ensure_wgs84(gdf[["statedist", "geometry"]])
    raw = pd.read_stata(_registry._fetch("india520.dta"))
    df = _build_india_panel(raw)
    df_dict = _read_bundled_dict("india520_dict.csv")
    return gdf, df, df_dict


def load_india_states() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the India states (n=32) nighttime lights cross-section for 1992.

    Regional sums of corrected DMSP-OLS nighttime lights (CCNL v1) over
    GlobPOP gridded population, computed in Google Earth Engine by the
    authors of quarcs-lab/project2025s-py.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        32 state/union territory geometries with columns
        ``["region", "geometry"]``, CRS EPSG:4326.
    df : pandas.DataFrame
        32 rows with columns
        ``["region", "year", "ntl_sum", "pop", "ntl_pc", "log_ntl_pc"]``
        (``year`` is always 1992).
    df_dict : pandas.DataFrame
        Data dictionary with one row per ``df`` column, in ``df`` column
        order, with columns
        ``var_name, var_def, label, type, role, can_be_na``.

    Raises
    ------
    GeometricsDataError
        If a source file cannot be downloaded or fails hash verification.

    See Also
    --------
    load_india : District-level (n=520) panel case study.
    """
    gdf = gpd.read_file(_registry._fetch("maps/india32.geojson"))
    gdf = _ensure_wgs84(gdf[["region", "geometry"]])
    df = pd.read_csv(_registry._fetch("ntl/india32_ntl_percapita_1992.csv"))
    df = df.rename(
        columns={"sum_ntl": "ntl_sum", "sum_pop": "pop", "ln_ntl_pc": "log_ntl_pc"}
    )
    df["year"] = 1992
    df = df[_STATES_COLUMNS]
    df_dict = _read_bundled_dict("india32_dict.csv")
    return gdf, df, df_dict


def load_india_raw() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Load the untouched source files of the India district case study.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        india520.geojson with all of its properties (25 columns including
        ``geometry``), CRS EPSG:4326.
    df : pandas.DataFrame
        india520.dta read as-is: 520 rows, 341 columns.

    Raises
    ------
    GeometricsDataError
        If a source file cannot be downloaded or fails hash verification.

    See Also
    --------
    load_india : The reshaped ``(gdf, df, df_dict)`` version of this data.
    """
    gdf = _ensure_wgs84(gpd.read_file(_registry._fetch("india520.geojson")))
    df = pd.read_stata(_registry._fetch("india520.dta"))
    return gdf, df


# ---------------------------------------------------------------------------
# Bolivia — BOL-005popAdj-PWTscaled (PWT-anchored subnational GDP, 2012-2022)
# ---------------------------------------------------------------------------

#: Registry file stems per Bolivia level: (directory, file stem).
_BOL_LEVELS = {
    "adm0": ("ADM0", "bolivia_adm0"),
    "adm1": ("ADM1", "bolivia_adm1"),
    "adm2": ("ADM2", "bolivia_adm2"),
    "grid": ("GRID", "bolivia_grid_cells"),
}

_BOL_CITATION = """    The product is derived data; cite the underlying GDP estimates, the
    national benchmark, and the boundaries:

    - Rossi-Hansberg, E., & Zhang, J. (2026). Local GDP estimates around the
      world. *Journal of Urban Economics*, 154, 103871.
    - Feenstra, R. C., Inklaar, R., & Timmer, M. P. (2015). The Next
      Generation of the Penn World Table. *American Economic Review*,
      105(10), 3150-3182. Data: Penn World Table 11.0.
    - GADM (2022). Database of Global Administrative Areas, version 4.10.

    Full methodological documentation (the proportional rescaling to PWT
    national totals, the 0_05 low-density censoring, and per-level
    dictionaries) lives in ``datasets/BOL-005popAdj-PWTscaled/README.md`` of
    the geometrics repository."""


def _normalize_bolivia_dict(
    df_dict: pd.DataFrame, entity: str, demote: tuple[str, ...]
) -> pd.DataFrame:
    """Return the shipped dictionary with exactly one entity-typed row.

    The source dictionaries mark compound keys (e.g. ``gid`` + ``iso``) as
    ``type="entity"``. Within this single-country collection the ``entity``
    column alone is unique, so every other key component in ``demote`` is
    retyped ``factor`` with a provenance note appended to its definition.
    """
    out = df_dict.copy()
    for name in demote:
        row = out["var_name"] == name
        out.loc[row, "type"] = "factor"
        out.loc[row, "var_def"] = (
            out.loc[row, "var_def"].astype(str)
            + f" In this single-country collection the entity id is {entity!r};"
            " this column is kept as a factor."
        )
    return out


def _load_bolivia_level(
    level: str, entity: str
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Load one admin level of the Bolivia collection as the geometrics trio."""
    folder, stem = _BOL_LEVELS[level]
    gdf = gpd.read_file(_registry._fetch_bolivia(f"{folder}/{stem}_boundaries.gpkg"))
    gdf = _ensure_wgs84(gdf[[entity, "geometry"]])
    df = pd.read_csv(_registry._fetch_bolivia(f"{folder}/{stem}.csv"))
    df["year"] = df["year"].astype("int64")
    df = df.sort_values([entity, "year"], ignore_index=True)
    df_dict = pd.read_csv(_registry._fetch_bolivia(f"{folder}/{stem}_data_def.csv"))
    df_dict = _normalize_bolivia_dict(df_dict, entity, demote=("iso",))
    return gdf, df, df_dict


def load_bolivia() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Bolivia province (ADM2, n=112) PWT-anchored GDP panel.

    Subnational GDP for 2012--2022 derived from the 0.25-degree gridded
    estimates of Rossi-Hansberg & Zhang (2026) under their most aggressive
    low-population-density censoring (``0_05``), proportionally rescaled so
    Bolivian national totals equal Penn World Table 11.0 (``rgdpo`` and
    ``pop``), and aggregated to GADM 4.10 provinces. GDP and population are
    therefore in interpretable 2021 PPP US$ units and the relative spatial
    pattern of the underlying model is preserved exactly.

    {citation}

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        112 province geometries with columns ``["gid", "geometry"]``, CRS
        EPSG:4326. Five provinces (``BOL.2.1_2``, ``BOL.2.8_2``,
        ``BOL.2.11_2``, ``BOL.2.13_2``, ``BOL.5.16_2``) have **no panel
        rows**: all of their grid cells are censored at the 0_05 threshold.
        geometrics' alignment warns about them, which is expected.
    df : pandas.DataFrame
        Balanced panel of 1177 rows (107 provinces x 11 years, 2012--2022).
        Key variables: ``gdp_pwt`` (millions of 2021 PPP US$), ``pop_pwt``
        (millions of persons), ``gdppc`` (2021 PPP US$ per person) and
        ``ln_gdppc``, plus provenance/scaling columns documented in the
        dictionary. Sorted by ``gid`` then ``year``.
    df_dict : pandas.DataFrame
        Data dictionary with one row per ``df`` column, in ``df`` column
        order (``gid`` is the entity, ``name`` the entity name, ``year`` the
        time id).

    Raises
    ------
    GeometricsDataError
        If a source file cannot be downloaded or fails hash verification.

    See Also
    --------
    load_bolivia_departments : Department-level (ADM1, n=9) version.
    load_bolivia_grid : The underlying 0.25-degree grid cells (n=1603).
    load_bolivia_raw : Untouched files of any level, including ADM0.

    Examples
    --------
    >>> from geometrics.data import load_bolivia
    >>> gdf, df, df_dict = load_bolivia()  # doctest: +SKIP
    >>> df.shape  # doctest: +SKIP
    (1177, 21)
    """
    return _load_bolivia_level("adm2", "gid")


def load_bolivia_departments() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Bolivia department (ADM1, n=9) PWT-anchored GDP panel.

    The department-level aggregation of the same product as
    :func:`load_bolivia`: Rossi-Hansberg & Zhang (2026) gridded GDP under
    0_05 censoring, rescaled to Penn World Table 11.0 national totals, on
    GADM 4.10 boundaries, 2012--2022, in 2021 PPP US$.

    {citation}

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        9 department geometries with columns ``["gid", "geometry"]``, CRS
        EPSG:4326.
    df : pandas.DataFrame
        Balanced panel of 99 rows (9 departments x 11 years). Key variables
        as in :func:`load_bolivia`.
    df_dict : pandas.DataFrame
        Data dictionary with one row per ``df`` column, in ``df`` column
        order.

    Raises
    ------
    GeometricsDataError
        If a source file cannot be downloaded or fails hash verification.

    See Also
    --------
    load_bolivia : Province-level (ADM2, n=112) version.
    load_bolivia_grid : The underlying 0.25-degree grid cells (n=1603).
    """
    return _load_bolivia_level("adm1", "gid")


def load_bolivia_grid() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Bolivia 0.25-degree grid cells (n=1603) PWT-anchored GDP panel.

    The raw cells of the same product as :func:`load_bolivia` before any
    administrative aggregation: Rossi-Hansberg & Zhang (2026) gridded GDP
    under 0_05 censoring, rescaled to Penn World Table 11.0 national totals,
    2012--2022, in 2021 PPP US$.

    The source keys cells by the compound (``cell_id``, ``subcell_id``,
    ``subcell_id_0_25``); geometrics needs a single entity id shared between
    ``gdf`` and ``df``, so the loader synthesizes ``cell`` as
    ``cell_id.subcell_id.subcell_id_0_25`` and joins the geometry on the
    cells' unique (longitude, latitude) centers.

    {citation}

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        1603 grid-cell polygons with columns ``["cell", "geometry"]``, CRS
        EPSG:4326.
    df : pandas.DataFrame
        Balanced panel of 17633 rows (1603 cells x 11 years) with ``cell``
        first, then every source column. Sorted by ``cell`` then ``year``.
    df_dict : pandas.DataFrame
        Data dictionary with one row per ``df`` column, in ``df`` column
        order (``cell`` is the sole entity-typed row; the three source key
        components are kept as factors).

    Raises
    ------
    GeometricsDataError
        If a source file cannot be downloaded or fails hash verification.

    See Also
    --------
    load_bolivia : Province-level (ADM2, n=112) aggregation.
    load_bolivia_departments : Department-level (ADM1, n=9) aggregation.
    """
    folder, stem = _BOL_LEVELS["grid"]
    df = pd.read_csv(_registry._fetch_bolivia(f"{folder}/{stem}.csv"))
    df["year"] = df["year"].astype("int64")
    cell = (
        df["cell_id"].astype(str)
        + "."
        + df["subcell_id"].astype(str)
        + "."
        + df["subcell_id_0_25"].astype(str)
    )
    df.insert(0, "cell", cell)
    df = df.sort_values(["cell", "year"], ignore_index=True)

    # Geometry: the gpkg lacks the subcell columns but shares the cells'
    # unique (longitude, latitude) centers — 0.25-degree centers are exact
    # binary fractions, so the equality join is lossless.
    gdf = gpd.read_file(_registry._fetch_bolivia(f"{folder}/{stem}.gpkg"))
    lookup = df[["cell", "longitude", "latitude"]].drop_duplicates()
    gdf = _ensure_wgs84(
        gdf.merge(lookup, on=["longitude", "latitude"], validate="1:1")[
            ["cell", "geometry"]
        ]
    )

    df_dict = pd.read_csv(_registry._fetch_bolivia(f"{folder}/{stem}_data_def.csv"))
    df_dict = _normalize_bolivia_dict(
        df_dict, "cell", demote=("cell_id", "subcell_id", "subcell_id_0_25")
    )
    cell_row = pd.DataFrame(
        [
            {
                "var_name": "cell",
                "var_def": (
                    "Single cell id synthesized by geometrics as "
                    "cell_id.subcell_id.subcell_id_0_25 (the source's compound "
                    "key); unique per 0.25-degree cell."
                ),
                "label": "Grid cell",
                "type": "entity",
                "role": "",
                "can_be_na": False,
            }
        ]
    )
    df_dict = pd.concat([cell_row, df_dict], ignore_index=True)
    return gdf, df, df_dict


def load_bolivia_raw(
    level: Literal["adm0", "adm1", "adm2", "grid"] = "adm2",
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Load the untouched files of one Bolivia level (including ADM0).

    Parameters
    ----------
    level : {"adm0", "adm1", "adm2", "grid"}
        Which level of the collection to load. ``"adm0"`` is the national
        aggregate (one unit; useful for checking the PWT anchoring).

    Returns
    -------
    df : pandas.DataFrame
        The level's long panel CSV read as-is.
    gdf : geopandas.GeoDataFrame
        The level's geometry with all of its attribute columns (the
        boundaries GeoPackage for admin levels; the cells GeoPackage for
        ``"grid"``), CRS EPSG:4326.

    Raises
    ------
    GeometricsDataError
        If a source file cannot be downloaded or fails hash verification.

    See Also
    --------
    load_bolivia : The reshaped ``(gdf, df, df_dict)`` province version.
    """
    if level not in _BOL_LEVELS:
        raise ValueError(
            f"load_bolivia_raw: unknown level {level!r} — expected one of "
            f"{sorted(_BOL_LEVELS)}"
        )
    folder, stem = _BOL_LEVELS[level]
    df = pd.read_csv(_registry._fetch_bolivia(f"{folder}/{stem}.csv"))
    geo_name = (
        f"{folder}/{stem}.gpkg"
        if level == "grid"
        else (f"{folder}/{stem}_boundaries.gpkg")
    )
    gdf = _ensure_wgs84(gpd.read_file(_registry._fetch_bolivia(geo_name)))
    return df, gdf


for _fn in (load_bolivia, load_bolivia_departments, load_bolivia_grid):
    # The placeholder sits at the docstring's 4-space body indent; the citation
    # block carries its own indentation, so the indent is consumed with it.
    _fn.__doc__ = (_fn.__doc__ or "").replace("    {citation}", _BOL_CITATION)
del _fn


def clear_cache() -> None:
    """Remove the local cache of downloaded case-study files.

    Deletes the pooch cache directory used by the loaders (the OS cache
    directory for ``geometrics``, or the directory named by the
    ``GEOMETRICS_DATA_DIR`` environment variable). The next loader call
    downloads the files again.
    """
    cache_dir = Path(_registry._POOCH.abspath)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
