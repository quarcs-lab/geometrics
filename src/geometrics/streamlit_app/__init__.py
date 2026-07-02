"""The geometrics interactive apps, built with Streamlit.

geometrics ships **three** no-code apps — one per module — that build a multipage UI
on top of the library's ``explore_*`` / ``analyze_*`` / ``learn_*`` functions:

* :func:`ExploreApp` — maps, weights, Moran/LISA, space-time views,
* :func:`AnalyzeApp` — convergence, spatial models with impacts, dynamics, inequality,
* :func:`LearnApp` — the teaching layer (concept sandboxes and explainers).

All three share one lean shell (a bundled-dataset picker and the spatial-weights
controls) and differ only in which pages they expose. Because Streamlit runs as its
own process, each launcher starts ``streamlit run`` in a subprocess tagged with the
module via the ``GEOMETRICS_MODULE`` environment variable; the app body is
:func:`geometrics.streamlit_app._entry.run_app`. The apps need the ``streamlit``
extra: ``pip install "geometrics[streamlit]"``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

__all__ = [
    "ExploreApp",
    "AnalyzeApp",
    "LearnApp",
    "main_explore",
    "main_analyze",
    "main_learn",
    "build_command",
]

# launch kwarg -> Streamlit CLI flag.
_FLAG_MAP = {
    "port": "--server.port",
    "host": "--server.address",
    "address": "--server.address",
    "base_url_path": "--server.baseUrlPath",
}


def _entry_script() -> str:
    """Filesystem path to the packaged runnable script Streamlit should execute."""
    from importlib import resources

    return str(resources.files("geometrics.streamlit_app") / "_run.py")


def build_command(entry: str, run_kwargs: dict[str, Any]) -> list[str]:
    """Build the ``streamlit run`` command line from the launch kwargs."""
    cmd = [sys.executable, "-m", "streamlit", "run", entry]
    headless = run_kwargs.get("headless")
    if headless is None and run_kwargs.get("launch_browser") is False:
        headless = True
    if headless is not None:
        cmd += ["--server.headless", "true" if headless else "false"]
    for key, flag in _FLAG_MAP.items():
        value = run_kwargs.get(key)
        if value is not None:
            cmd += [flag, str(value)]
    return cmd


def _launch(module: str, *, run: bool = True, **run_kwargs: Any):
    """Start (or, with ``run=False``, describe) one module's app subprocess.

    A subprocess (``python -m streamlit run``) is used rather than Streamlit's
    in-process bootstrap because the latter installs signal handlers and calls
    ``sys.exit``, which would hijack the caller's Python session.
    """
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            'the apps require the streamlit extra: pip install "geometrics[streamlit]"'
        ) from exc
    cmd = build_command(_entry_script(), run_kwargs)
    env = {**os.environ, "GEOMETRICS_MODULE": module}
    if not run:
        return cmd
    return subprocess.Popen(cmd, env=env)


def ExploreApp(*, run: bool = True, **run_kwargs: Any):
    """Launch the Explore app (maps, weights, Moran/LISA, space-time views).

    Parameters
    ----------
    run
        Start the subprocess (default). ``False`` returns the command line instead.
    **run_kwargs
        Server options forwarded to ``streamlit run`` — ``port``, ``host``,
        ``headless``, ``base_url_path``.

    Examples
    --------
    ```python
    import geometrics.streamlit_app as apps

    apps.ExploreApp(port=8601)  # doctest: +SKIP
    ```
    """
    return _launch("explore", run=run, **run_kwargs)


def AnalyzeApp(*, run: bool = True, **run_kwargs: Any):
    """Launch the Analyze app (convergence, spatial models, dynamics, inequality)."""
    return _launch("analyze", run=run, **run_kwargs)


def LearnApp(*, run: bool = True, **run_kwargs: Any):
    """Launch the Learn app (concept sandboxes and explainers)."""
    return _launch("learn", run=run, **run_kwargs)


def main_explore() -> None:
    """Console entry point: ``geometrics-explore``."""
    ExploreApp().wait()


def main_analyze() -> None:
    """Console entry point: ``geometrics-analyze``."""
    AnalyzeApp().wait()


def main_learn() -> None:
    """Console entry point: ``geometrics-learn``."""
    LearnApp().wait()
