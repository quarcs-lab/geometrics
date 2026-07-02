"""The package import surface stays lazy (PEP 562) and API-complete.

``import geometrics`` must stay cheap: it should pull in *no* estimator submodules and
*no* heavy third-party libraries — each public name imports only its own submodule on
first access. The ``sys.modules`` guards run in a fresh subprocess because pytest shares
one interpreter across test modules (others import the heavy submodules directly, which
would poison an in-process check).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import geometrics as gm

# Submodules that used to be imported eagerly by __init__ and must now stay dormant until
# their public name is actually touched.
_DORMANT_SUBMODULES = [
    "geometrics.maps",
    "geometrics.weights",
    "geometrics.dependence",
    "geometrics.spacetime",
    "geometrics.convergence",
    "geometrics.clubs",
    "geometrics.spatial_models",
    "geometrics.distribution_dynamics",
    "geometrics.regional_inequality",
    "geometrics.gwr",
    "geometrics.sandbox",
    "geometrics.pedagogy",
    "geometrics.data",
]

# Heavy third-party libraries that no bare `import geometrics` should trigger.
_HEAVY_LIBS = ["statsmodels", "geopandas", "giddy", "mgwr", "spreg", "great_tables"]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a fresh interpreter so sys.modules starts clean."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


# --- laziness (fresh-process guards) --------------------------------------------------


def test_bare_import_loads_nothing_heavy():
    proc = _run(
        f"""
        import sys
        import geometrics  # noqa: F401
        loaded = set(sys.modules)
        dormant = [m for m in {_DORMANT_SUBMODULES!r} if m in loaded]
        assert not dormant, f"eagerly imported submodules: {{dormant}}"
        heavy = [
            h for h in {_HEAVY_LIBS!r}
            if any(k == h or k.startswith(h + ".") for k in loaded)
        ]
        assert not heavy, f"eagerly imported heavy libs: {{heavy}}"
        """
    )
    assert proc.returncode == 0, proc.stderr


def test_touching_a_name_imports_only_its_submodule():
    proc = _run(
        """
        import sys
        import geometrics as gm
        _ = gm.explore_choropleth_map
        assert "geometrics.maps" in sys.modules
        # Unrelated estimators / their heavy deps stay dormant.
        assert "geometrics.spatial_models" not in sys.modules
        assert "geometrics.distribution_dynamics" not in sys.modules
        assert "statsmodels" not in sys.modules
        """
    )
    assert proc.returncode == 0, proc.stderr


def test_learn_path_stays_light():
    # A Learn sandbox must not drag in statsmodels / geopandas / great_tables on access.
    proc = _run(
        f"""
        import sys
        import geometrics as gm
        _ = gm.learn_spatial_autocorrelation
        heavy = [
            h for h in {_HEAVY_LIBS!r}
            if any(k == h or k.startswith(h + ".") for k in sys.modules)
        ]
        assert not heavy, f"learn path pulled heavy libs: {{heavy}}"
        """
    )
    assert proc.returncode == 0, proc.stderr


# --- public API completeness (in-process) ---------------------------------------------


def test_all_public_names_resolve():
    for name in gm.__all__:
        assert getattr(gm, name) is not None


def test_data_subpackage_accessible():
    from geometrics import data

    assert gm.data is data
    assert hasattr(data, "load_india")


def test_dir_matches_all():
    assert dir(gm) == sorted(gm.__all__)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        _ = gm.no_such_public_name


def test_lazy_map_covers_public_api():
    # Same invariant the module-level assert enforces at import time, surfaced as a test.
    assert set(gm._ATTR_TO_MODULE) | {"data"} == set(gm.__all__)


def test_version_is_nonempty_string():
    assert isinstance(gm.__version__, str) and gm.__version__
