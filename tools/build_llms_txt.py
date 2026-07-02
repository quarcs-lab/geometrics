#!/usr/bin/env python
"""Build llms.txt (committed, drift-checked) and llms-full.txt (site-only).

``docs/llms.txt`` is the curated, stable index that LLMs and agents fetch first: what
the package is, the three-input contract, and where everything lives. It is committed
and drift-checked in CI, so regenerate it whenever the public API changes::

    uv run python tools/build_llms_txt.py

``llms-full.txt`` concatenates the docs sources (qmd prose + code) plus every public
signature and docstring; it is emitted into ``docs/_site`` at docs-build time only
(pass ``--full`` after the site exists). ``--canonical-only`` (CI drift check) writes
just ``docs/llms.txt``.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SITE = DOCS / "_site"
BASE = "https://quarcs-lab.github.io/geometrics"

PAGES = [
    ("Quickstart", "quickstart", "load the India case study and run the core flow"),
    ("The data model", "articles/data-model", "the (gdf, df, df_dict) contract"),
    ("Convergence", "articles/convergence", "beta/sigma convergence and clubs"),
    ("Spatial dependence", "articles/spatial-dependence", "weights, Moran, LISA"),
    ("Spatial spillovers", "articles/spillovers", "the spreg suite and impacts"),
    ("Regional inequality", "articles/inequality", "Gini/Theil and decompositions"),
    ("Distribution dynamics", "articles/dynamics", "Markov and spatial Markov"),
    ("The India case study", "articles/india-case-study", "the full replication arc"),
    ("Changelog", "changelog", "release notes"),
]


def _api_groups() -> list[tuple[str, list[str]]]:
    import geometrics as gm

    names = [n for n in gm.__all__ if n[0].islower() and n != "data"]
    return [
        ("explore_*", [n for n in names if n.startswith("explore_")]),
        ("analyze_*", [n for n in names if n.startswith("analyze_")]),
        (
            "utilities",
            [n for n in names if not n.startswith(("explore_", "analyze_"))],
        ),
        (
            "geometrics.data",
            ["load_india", "load_india_states", "load_india_raw", "clear_cache"],
        ),
    ]


def build_canonical() -> str:
    lines = [
        "# geometrics",
        "",
        "> Regional growth, convergence, and inequality analysis on the PySAL stack",
        "> (libpysal, esda, giddy, inequality, mapclassify, spreg, mgwr) with Plotly",
        "> figures, Great Tables, and plain-language interpretation on every result.",
        "",
        "Three inputs: gdf (geometry with ONLY the entity ID; shapefile / zipped",
        "shapefile / GeoJSON / GeoPackage via read_gdf), df (long-form panel declared",
        "with set_panel / set_labels), df_dict (6-column data dictionary: var_name,",
        "var_def, label, type, role, can_be_na). Every public function returns a frozen",
        "result dataclass with .df, .fig and/or .gt, .interpret() and .explain().",
        "",
        "Install: pip install geometrics  (extras: [dynamics] for Markov via giddy,",
        "[png] for static export, [all]).",
        "",
        "## Docs",
        "",
    ]
    lines += [f"- [{t}]({BASE}/{slug}.html): {desc}" for t, slug, desc in PAGES]
    lines += ["", "## API", ""]
    for group, names in _api_groups():
        lines.append(f"- {group}: " + ", ".join(names))
    lines += [
        "",
        "## Source",
        "",
        "- [Repository](https://github.com/quarcs-lab/geometrics)",
        f"- [API reference]({BASE}/reference/index.html)",
        f"- [llms-full.txt]({BASE}/llms-full.txt): full docs text + signatures",
        "",
    ]
    return "\n".join(lines)


def build_full() -> str:
    import geometrics as gm

    parts = [build_canonical(), "\n\n# ===== Docs pages (source) =====\n"]
    for _, slug, _ in PAGES:
        qmd = DOCS / f"{slug}.qmd"
        if qmd.exists():
            parts.append(f"\n\n## ----- {slug}.qmd -----\n\n{qmd.read_text()}")
    parts.append("\n\n# ===== Public API signatures =====\n")
    for group, names in _api_groups():
        parts.append(f"\n\n## {group}\n")
        for name in names:
            obj = (
                getattr(gm.data, name, None)
                if group == "geometrics.data"
                else getattr(gm, name, None)
            )
            if obj is None or not callable(obj):
                continue
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = "(...)"
            doc = inspect.getdoc(obj) or ""
            parts.append(f"\n### {name}{sig}\n\n{doc}\n")
    return "".join(parts)


def main() -> None:
    canonical = build_canonical()
    (DOCS / "llms.txt").write_text(canonical)
    print(f"wrote docs/llms.txt ({len(canonical.splitlines())} lines)")
    if "--canonical-only" in sys.argv:
        return
    if SITE.exists():
        full = build_full()
        (SITE / "llms.txt").write_text(canonical)
        (SITE / "llms-full.txt").write_text(full)
        print(f"wrote docs/_site/llms-full.txt ({len(full.splitlines())} lines)")


if __name__ == "__main__":
    main()
