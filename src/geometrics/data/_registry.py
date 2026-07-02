"""Remote registry and cached fetching for the geometrics case-study data.

The India case study ships as loaders that download the source files of the
paper repository `quarcs-lab/project2025s-py <https://github.com/quarcs-lab/project2025s-py>`_
from GitHub raw URLs pinned to a specific commit, verify them against known
SHA-256 hashes, and cache them locally with :mod:`pooch`.

The Bolivia collection (BOL-005popAdj-PWTscaled) lives in the geometrics
repository itself under ``datasets/`` and is fetched the same way — from raw
URLs pinned to the commit that added the data, hash-verified, and cached.

The cache location defaults to the OS cache directory for ``geometrics`` and
can be overridden with the ``GEOMETRICS_DATA_DIR`` environment variable.
"""

from __future__ import annotations

from pathlib import Path

import pooch

#: Commit of quarcs-lab/project2025s-py that the registry hashes are pinned to.
COMMIT = "b5688fe367af536da06880d97aacaebb3c09d29f"

#: Base URL of the paper repository's ``data/`` directory at :data:`COMMIT`.
BASE_URL = (
    f"https://raw.githubusercontent.com/quarcs-lab/project2025s-py/{COMMIT}/data/"
)

#: File names (relative to :data:`BASE_URL`) mapped to their SHA-256 hashes.
REGISTRY = {
    "india520.dta": "sha256:586a5c797f9dfb94fe593f219dc5eeeb46a0ec27eec3adc0282626b5b58b3c05",
    "india520.geojson": "sha256:af717cfde06d6eb51711ccca1be7c956b74ca32131a3071ec9ab7036b05fff5e",
    "maps/india32.geojson": "sha256:70a541f735b934099c3cb25807dacc1df5f0a57676ea74723c33159e2e38e772",
    "ntl/india32_ntl_percapita_1992.csv": "sha256:1e3c5e83316ee77e273e9d7e5b577bafc929b6f0760545c761982a7f61417234",
}

_POOCH = pooch.create(
    path=pooch.os_cache("geometrics"),
    base_url=BASE_URL,
    registry=REGISTRY,
    env="GEOMETRICS_DATA_DIR",
)


class GeometricsDataError(RuntimeError):
    """Raised when a geometrics case-study data file cannot be retrieved.

    The message names the remote URL that failed, the local cache directory,
    and the ``GEOMETRICS_DATA_DIR`` environment variable that overrides it.
    """


def _fetch(name: str) -> Path:
    """Return the local path of a registry file, downloading it if needed.

    This is the single choke point through which every loader retrieves data;
    tests monkeypatch this function to serve local fixture files instead.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``"india520.dta"`` or ``"maps/india32.geojson"``.

    Returns
    -------
    pathlib.Path
        Path of the verified file inside the local cache.

    Raises
    ------
    GeometricsDataError
        If the download or hash verification fails.
    """
    try:
        return Path(_POOCH.fetch(name))
    except Exception as exc:
        raise GeometricsDataError(
            f"Could not retrieve the geometrics case-study file {name!r} "
            f"from {BASE_URL + name}. "
            f"(local cache: {_POOCH.abspath}). "
            "Check your internet connection and retry; if the file exists but "
            "is corrupted, delete it (or call geometrics.data.clear_cache()) "
            "and retry. To use a custom download location, set the "
            "GEOMETRICS_DATA_DIR environment variable."
        ) from exc


#: Commit of quarcs-lab/geometrics that added ``datasets/BOL-005popAdj-PWTscaled``.
BOL_COMMIT = "c23a78de67db2a72c3243c43887927675b685c7c"

#: Base URL of the Bolivia collection at :data:`BOL_COMMIT`.
BOL_BASE_URL = (
    "https://raw.githubusercontent.com/quarcs-lab/geometrics/"
    f"{BOL_COMMIT}/datasets/BOL-005popAdj-PWTscaled/"
)

#: File names (relative to :data:`BOL_BASE_URL`) mapped to their SHA-256 hashes.
BOL_REGISTRY = {
    "ADM0/bolivia_adm0.csv": "sha256:d760a9bdacd6401496dda41829d080c976faf4b9b5e7d27a90eb360c8a48098c",
    "ADM0/bolivia_adm0_data_def.csv": "sha256:eb80910c0a0ddbce4f57daf58640d01e8db25f56e110cfab09bba35b0363fef0",
    "ADM0/bolivia_adm0_boundaries.gpkg": "sha256:7d57cb318cea6fed2c6e03e3d7f21022f8abde824692439b680a76970ce6c9e7",
    "ADM1/bolivia_adm1.csv": "sha256:1f79230a9248380d51cf61995cafa7264930a33be17e83b04cba981de744429f",
    "ADM1/bolivia_adm1_boundaries.gpkg": "sha256:1084533470c41dc13ce273045f643430ac4874c9a346d562f42da6391cfe20b6",
    "ADM1/bolivia_adm1_data_def.csv": "sha256:0f76ec6955cb9d7341b73a82ce059939ea66d8f42a13327d2244e965b552e4a4",
    "ADM2/bolivia_adm2.csv": "sha256:e3993f592090fe13761114976225b0fbf0b1a1b3b7b041041f47f3ad0e546a3d",
    "ADM2/bolivia_adm2_boundaries.gpkg": "sha256:a3463d2acbd7650bbb43e87a2f45df0cf236d82b4f93c5bbac4393f67c0f030f",
    "ADM2/bolivia_adm2_data_def.csv": "sha256:011c0e2020911f6ae24d1433792574739f7dd723788f1102cb10b5662cee3196",
    "GRID/bolivia_grid_cells.csv": "sha256:3af614177f84322211ead2366822f1baa6ad074773ae774de1515ce26e82f2e2",
    "GRID/bolivia_grid_cells.gpkg": "sha256:26df0f506929cc041969311fcbd69f369d24a1f99d890a05a564cf78e8971040",
    "GRID/bolivia_grid_cells_data_def.csv": "sha256:7c0096e975c90a27c204db8224f5abecc05299ec6f412adf020d5f2cf32ce998",
}

_POOCH_BOL = pooch.create(
    path=pooch.os_cache("geometrics"),
    base_url=BOL_BASE_URL,
    registry=BOL_REGISTRY,
    env="GEOMETRICS_DATA_DIR",
)


def _fetch_bolivia(name: str) -> Path:
    """Return the local path of a Bolivia-collection file, downloading if needed.

    The Bolivia analogue of :func:`_fetch` — the single choke point through
    which the ``load_bolivia*`` loaders retrieve data; tests monkeypatch this
    function to serve the in-repository ``datasets/`` files instead.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``"ADM2/bolivia_adm2.csv"``.

    Returns
    -------
    pathlib.Path
        Path of the verified file inside the local cache.

    Raises
    ------
    GeometricsDataError
        If the download or hash verification fails.
    """
    try:
        return Path(_POOCH_BOL.fetch(name))
    except Exception as exc:
        raise GeometricsDataError(
            f"Could not retrieve the geometrics case-study file {name!r} "
            f"from {BOL_BASE_URL + name}. "
            f"(local cache: {_POOCH_BOL.abspath}). "
            "Check your internet connection and retry; if the file exists but "
            "is corrupted, delete it (or call geometrics.data.clear_cache()) "
            "and retry. To use a custom download location, set the "
            "GEOMETRICS_DATA_DIR environment variable."
        ) from exc
