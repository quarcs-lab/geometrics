# Project status

_Last updated: 2026-07-02._

**geometrics** is a Python library for **regional growth, convergence, and inequality**
analysis with explicit spatial methods, built on the PySAL stack (libpysal, esda, giddy,
inequality, mapclassify, spreg, mgwr) — organized as three modules (**Explore**,
**Analyze**, **Learn**), three no-code Streamlit apps, and a Quarto + quartodoc
documentation site (<https://quarcs-lab.github.io/geometrics/>). It follows the design
language of [expdpy](https://github.com/cmg777/expdpy).

Current version: **0.1.3** — on `main` and **released to PyPI**
(`pip install geometrics`; extras `[dynamics]`, `[streamlit]`, `[png]`, `[all]`).

## Release history

- **v0.1.0** — first public release: the three-input data contract
  (`gdf` / `df` / `df_dict`), 7 `explore_*` + 13 `analyze_*` functions, frozen result
  objects with `.interpret()` / `.explain()`, the India case study (520 districts by
  nighttime lights, 1996–2010, paper-parity tested), Quarto docs, Colab notebooks,
  OIDC trusted publishing.
- **v0.1.1** — the Bolivia dataset (PWT-anchored local GDP, 2012–2022, at province /
  department / 0.25° grid scales); data committed in-repo under `datasets/` and served
  from pinned, hash-verified raw URLs.
- **v0.1.2** — the **Learn module**: 11 `learn_*` concept sandboxes
  (`geometrics.sandbox`) that simulate from a known DGP and let the real estimator
  recover the planted parameter; `SandboxResult`; the test suite's synthetic DGPs
  factored into `sandbox/_dgp.py`. Also the "classified lattice" visual identity and
  the **For AI / LLMs** docs page.
- **v0.1.3** — the three **Streamlit apps** (`geometrics.streamlit_app`): a lean
  shared shell (bundled case-study picker + weights controls) with self-gating pages,
  `.interpret()` under every figure, and sliders on every sandbox knob. Deployed on
  Streamlit Community Cloud:
  [Explore](https://geometrics-explore.streamlit.app/) ·
  [Analyze](https://geometrics-analyze.streamlit.app/) ·
  [Learn](https://geometrics-learn.streamlit.app/).

## Recently shipped — the three-module presentation

The site now mirrors expdpy's architecture: navbar **Home | Explore | Analyze | Learn |
For AI / LLMs | Articles | Reference | Changelog**; a landing page with the hero image
(composed from the package's real India LISA map), CTA row, and three module cards; and
one pedagogical, executable walkthrough per module, each with a try-banner linking its
app and Colab notebook:

- **Explore** (India) — three inputs → choropleth → weights → Moran → LISA →
  distribution dynamics; inherits the retired quickstart's URL via an alias redirect.
- **Analyze** (Bolivia provinces) — growth cross-section → β OLS vs SDM with
  LeSage-Pace impacts → LM diagnostics → weights robustness → σ → clubs → Markov →
  inequality/Theil → GWR.
- **Learn** — `.interpret()` / `.explain()` on real results, the 30-topic explainer
  index, and the sandbox catalog.

Two notable fixes along the way: CI workflows now run `uv run --no-sync` (plain
`uv run` rebuilt the venv without extras and without the matrix Python, so the
dynamics tests had never actually run in CI), and the `llms.txt` alternate links in
every page head had 404'd since v0.1.0 (Quarto rewrites root-relative hrefs) — a
site-wide link audit now comes back fully clean.

## Status of checks

- **Tests** — `make test` green (457 passed: known-answer, result-surface, sandbox
  recovery, and 17 offline AppTest smoke tests); `-m network` green (real-data
  roundtrips incl. Table-1 paper parity).
- **Lint / types** — `ruff check`, `ruff format --check`, and `mypy src` clean.
- **Docs** — `make docs` renders the full site with every example executing; the
  module notebooks and `llms.txt` are drift-checked in CI and fresh.
- **Live** — all site links (63 unique across the key pages) resolve; the three
  Community Cloud apps are up.

## Operations notes

- Version lives in **three** places: `pyproject.toml`, `src/geometrics/__init__.py`,
  `CITATION.cff`. Releases are small and frequent (patch bumps); tag `v*` publishes to
  PyPI via OIDC trusted publishing.
- After each PyPI release, **Reboot** the three Community Cloud apps (their
  `requirements.txt` tracks the latest release, but a warm container can hold the old
  wheel).
- The Intel-mac numba constraints in `[tool.uv] constraint-dependencies` and the
  registered `geometrics` Jupyter kernelspec for docs builds must stay (see
  `CLAUDE.md`).

## Open items / next steps

- **MGWR in the Analyze app** — deferred (backfitting is too slow for a 1-core
  container); documented as "run locally".
- **Bolivia grid in the app picker** — excluded (1,603 polygons strain the free tier);
  available via `load_bolivia_grid()`.
- **AI layer** — `llms.txt` / `llms-full.txt` and the For AI / LLMs page ship today;
  function-calling tool schemas and an MCP server (expdpy-style) are natural later
  patch releases.
- **More localGDP countries** — the `datasets/` layout and loader pattern generalize;
  a future `load_<country>()` per committed collection.
