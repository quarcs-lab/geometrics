"""Case-study data loaders for geometrics.

geometrics analyses take three inputs: ``gdf`` (an ID-only geometry table),
``df`` (a long panel of observations), and ``df_dict`` (a data dictionary
describing every ``df`` column). This subpackage ships the India nighttime
lights case study of `quarcs-lab/project2025s-py
<https://github.com/quarcs-lab/project2025s-py>`_ in exactly that shape:
the source files are downloaded from GitHub raw URLs pinned to a commit,
verified against SHA-256 hashes, cached locally, and reshaped.

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
clear_cache
    Remove the local download cache.
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

import geopandas as gpd
import pandas as pd

from geometrics.data import _registry
from geometrics.data._registry import GeometricsDataError

__all__ = [
    "load_india",
    "load_india_states",
    "load_india_raw",
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
