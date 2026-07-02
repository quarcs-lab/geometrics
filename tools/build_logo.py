#!/usr/bin/env python
"""Rasterise the geometrics logo/icon assets from the canonical SVGs.

The three hand-authored SVGs under ``src/geometrics/_assets`` are the single source of
truth for the "classified lattice" mark — a 3x3 choropleth whose values rise toward a
high-high cluster in the lower-right corner, with the connectivity graph drawn over the
cluster's centroids:

- ``logo.svg``        — the mark in the cosmo-blue ramp on a transparent background
                        (README, light backgrounds).
- ``logo-navbar.svg`` — the mark in white with opacity-stepped cells, legible on the
                        solid blue Quarto navbar.
- ``favicon.svg``     — the white mark on a solid rounded blue tile, which stays legible
                        at browser-tab sizes (favicon / Streamlit page icon).

This script renders the PNG copies with ``rsvg-convert`` and copies the web-facing files
into ``docs/images/`` so the docs site, the README, and the apps all share one design.
Re-run it whenever a source SVG changes and commit the generated outputs — there is no
build-time dependency on ``rsvg-convert``::

    python tools/build_logo.py

Fallbacks if librsvg is unavailable: ``qlmanage -t -s 256 -o . favicon.svg`` renders a
PNG thumbnail on macOS, and the motif is simple enough to redraw with Pillow's
``ImageDraw.rounded_rectangle`` if it ever comes to that.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "src" / "geometrics" / "_assets"
DOCS_IMAGES = REPO / "docs" / "images"

LOGO_SVG = ASSETS / "logo.svg"
NAVBAR_LOGO_SVG = ASSETS / "logo-navbar.svg"
FAVICON_SVG = ASSETS / "favicon.svg"


def _rsvg() -> str:
    """Locate ``rsvg-convert`` or exit with an install hint."""
    exe = shutil.which("rsvg-convert")
    if exe is None:
        sys.exit(
            "rsvg-convert not found on PATH — install librsvg "
            "(e.g. `brew install librsvg` or `apt install librsvg2-bin`)."
        )
    return exe


def render(svg: Path, out: Path, size: int) -> None:
    """Rasterise ``svg`` to a square ``size``x``size`` PNG at ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_rsvg(), "-w", str(size), "-h", str(size), str(svg), "-o", str(out)],
        check=True,
    )
    print(f"  rendered {out.relative_to(REPO)}  ({size}px)")


def copy(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest``, creating parent directories as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    print(f"  copied   {dest.relative_to(REPO)}")


def build() -> None:
    """Generate every derived logo asset from the canonical SVGs."""
    print("Packaged rasters (src/geometrics/_assets):")
    render(LOGO_SVG, ASSETS / "logo.png", 512)  # README / general use
    render(FAVICON_SVG, ASSETS / "favicon.png", 256)  # Streamlit page icon (tab tile)

    print("Docs site copies (docs/images):")
    copy(LOGO_SVG, DOCS_IMAGES / "logo.svg")  # ramp mark, for light backgrounds
    copy(
        NAVBAR_LOGO_SVG, DOCS_IMAGES / "logo-navbar.svg"
    )  # Quarto navbar (white, on blue)
    copy(
        ASSETS / "logo.png", DOCS_IMAGES / "logo.png"
    )  # README raw.githubusercontent URL
    render(FAVICON_SVG, DOCS_IMAGES / "favicon.png", 64)  # Quarto favicon


if __name__ == "__main__":
    build()
