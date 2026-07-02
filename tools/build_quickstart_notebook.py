#!/usr/bin/env python
"""Build the Google Colab notebooks from the docs pages.

The docs pages (``docs/quickstart.qmd`` and ``docs/articles/india-case-study.qmd``) are
the single source of truth for the code-along walkthroughs. This script regenerates one
notebook per page under ``notebooks/`` so the two never drift — each can be opened
straight from GitHub in Google Colab.

The conversion is ``quarto convert`` (the canonical ``.qmd`` -> ``.ipynb`` mapping:
prose -> markdown cells, ``{python}`` blocks -> code cells) followed by light
post-processing with ``nbformat``: the YAML front-matter is stripped and three cells
are prepended — a title cell, a GitHub ``pip install`` so the notebook is runnable from
a cold Colab runtime, and a setup cell that forces Plotly's ``colab`` renderer.

Run it through uv (quarto must be on PATH)::

    uv run python tools/build_quickstart_notebook.py

or simply ``make notebooks``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

REPO = Path(__file__).resolve().parents[1]

PAGES = [
    {
        "qmd": "docs/quickstart.qmd",
        "slug": "quickstart",
        "title_md": (
            "# geometrics — quickstart\n"
            "\n"
            "_Notebook version: built {BUILD_STAMP} — re-open this notebook from GitHub "
            "if yours is older, to get the latest version._\n"
            "\n"
            "A cloud-runnable walkthrough of "
            "[geometrics](https://github.com/quarcs-lab/geometrics): regional growth, "
            "convergence, and inequality analysis on the PySAL stack, illustrated with "
            "the bundled Indian district case study. Run the install cell below first, "
            "then run the rest top to bottom.\n"
            "\n"
            "> The first cell installs everything and then **restarts the Colab runtime "
            "once** so upgraded packages load cleanly. When it reconnects, run the cells "
            "again (Runtime > Run all) — the install cell skips the restart the second "
            "time.\n"
            "\n"
            "This notebook mirrors the [quickstart page]"
            "(https://quarcs-lab.github.io/geometrics/quickstart.html) of the docs."
        ),
    },
    {
        "qmd": "docs/articles/india-case-study.qmd",
        "slug": "india_case_study",
        "title_md": (
            "# The India case study — regional growth, convergence, and inequality "
            "from outer space\n"
            "\n"
            "_Notebook version: built {BUILD_STAMP} — re-open this notebook from GitHub "
            "if yours is older, to get the latest version._\n"
            "\n"
            "An end-to-end replication arc with "
            "[geometrics](https://github.com/quarcs-lab/geometrics): 520 Indian "
            "districts observed by satellite nighttime lights (1996-2010), following "
            "the analysis of [Mendez, Kabiraj & Li]"
            "(https://github.com/quarcs-lab/project2025s-py) — maps, spatial "
            "dependence, beta/sigma/club convergence, spatial spillovers (SDM), "
            "distribution dynamics, and inequality decomposition. Run the install cell "
            "below first, then run the rest top to bottom.\n"
            "\n"
            "> The first cell installs everything and then **restarts the Colab runtime "
            "once** so upgraded packages load cleanly. When it reconnects, run the cells "
            "again (Runtime > Run all) — the install cell skips the restart the second "
            "time.\n"
            "\n"
            "This notebook mirrors the [India case study article]"
            "(https://quarcs-lab.github.io/geometrics/articles/india-case-study.html) "
            "of the docs."
        ),
    },
]

# The [all] extra brings giddy (distribution dynamics) and kaleido (PNG export). The
# second pip line force-refreshes only the geometrics code to the latest main commit
# (pip skips a git reinstall when the version string is unchanged, so a warm runtime
# would otherwise keep stale code). The cell then restarts the Colab runtime ONCE
# (guarded by a /tmp flag so "Run all" does not loop): Colab pre-imports plotly/numpy at
# startup, so without a restart the kernel can keep stale modules in sys.modules.
INSTALL_CELL = """import importlib.util
import os

!pip install -q "geometrics[all] @ git+https://github.com/quarcs-lab/geometrics.git"
!pip install -q --force-reinstall --no-deps "geometrics @ git+https://github.com/quarcs-lab/geometrics.git"

_RESTART_FLAG = "/tmp/.geometrics_runtime_restarted"
_ON_COLAB = importlib.util.find_spec("google.colab") is not None
if _ON_COLAB and not os.path.exists(_RESTART_FLAG):
    with open(_RESTART_FLAG, "w"):
        pass
    print("Install complete - restarting the runtime once so packages load cleanly.")
    print("After it reconnects, run the cells again (Runtime > Run all).")
    os.kill(os.getpid(), 9)"""

# Colab does not always pick a Plotly renderer that draws figures returned as the last
# cell expression, so force the dedicated "colab" renderer there. A no-op in Jupyter.
SETUP_CELL = (
    "# Ensure Plotly figures render in Google Colab (a no-op elsewhere).\n"
    "import plotly.io as pio\n"
    "\n"
    "try:\n"
    "    import google.colab  # noqa: F401  (present only on Colab)\n"
    "\n"
    '    pio.renderers.default = "colab"\n'
    "except ImportError:\n"
    "    pass"
)

KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}


def _strip_front_matter(source: str) -> str:
    """Remove a leading ``---\\n...\\n---`` YAML block, keeping any prose after it."""
    match = re.match(r"\s*---\n.*?\n---\n?", source, flags=re.DOTALL)
    return source[match.end() :].lstrip("\n") if match else source


def _strip_raw_html(source: str) -> str:
    """Remove ```` ```{=html} ... ``` ```` raw blocks (site-only)."""
    return re.sub(r"```\{=html\}\n.*?\n```\n?", "", source, flags=re.DOTALL).lstrip(
        "\n"
    )


def convert_with_quarto(qmd: Path, dest: Path) -> None:
    """Run ``quarto convert`` to turn ``qmd`` into the notebook ``dest``."""
    quarto = shutil.which("quarto")
    if quarto is None:
        sys.exit("quarto not found on PATH — install it or run inside CI's docs job")
    subprocess.run([quarto, "convert", str(qmd), "--output", str(dest)], check=True)


def build_one(page: dict, build_stamp: str) -> Path:
    """Generate ``notebooks/<slug>.ipynb`` from the page's qmd source."""
    src_qmd = REPO / page["qmd"]
    out_ipynb = REPO / "notebooks" / f"{page['slug']}.ipynb"
    with tempfile.TemporaryDirectory() as tmp:
        converted = Path(tmp) / "converted.ipynb"
        convert_with_quarto(src_qmd, converted)
        nb = nbformat.read(converted, as_version=4)

    cells = list(nb.cells)
    if cells and cells[0].source.lstrip().startswith("---"):
        cells[0]["cell_type"] = "markdown"
        cells[0]["source"] = _strip_front_matter(cells[0].source)
    for cell in cells:
        if cell.cell_type == "markdown":
            cell["source"] = _strip_raw_html(cell.source)
    cells = [c for c in cells if c.cell_type != "markdown" or c.source.strip()]

    rendered_title = page["title_md"].replace("{BUILD_STAMP}", build_stamp)
    title_cell = new_markdown_cell(rendered_title)
    install = new_code_cell(INSTALL_CELL)
    setup = new_code_cell(SETUP_CELL)
    nb.cells = [title_cell, install, setup, *cells]

    if nb.nbformat_minor >= 5:
        title_cell["id"], install["id"], setup["id"] = "title", "install", "setup"
    else:
        for cell in nb.cells:
            cell.pop("id", None)

    nb.metadata["kernelspec"] = KERNELSPEC
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nbformat.validate(nb)
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_ipynb)
    print(f"wrote {out_ipynb.relative_to(REPO)}  ({len(nb.cells)} cells)")
    return out_ipynb


def build() -> None:
    """Generate one Colab notebook per docs page."""
    build_stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    for page in PAGES:
        build_one(page, build_stamp)


if __name__ == "__main__":
    build()
