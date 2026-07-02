"""Remote registry and cached fetching for the geometrics case-study data.

The India case study ships as loaders that download the source files of the
paper repository `quarcs-lab/project2025s-py <https://github.com/quarcs-lab/project2025s-py>`_
from GitHub raw URLs pinned to a specific commit, verify them against known
SHA-256 hashes, and cache them locally with :mod:`pooch`.

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
