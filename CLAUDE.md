# geometrics — development conventions

geometrics studies regional growth, convergence, and inequality with explicit spatial
methods, built on the PySAL stack (libpysal, esda, giddy, inequality, mapclassify,
spreg, mgwr) plus geopandas. It mirrors the design language of
[expdpy](https://github.com/cmg777/expdpy) by the same author.

## Toolchain

- **uv** manages the environment (`uv sync --locked --all-extras --group dev --group docs`).
  The lockfile `uv.lock` is committed. Dev tasks run through the `Makefile`
  (`make test`, `make lint`, `make typecheck`, `make docs`).
- Python floor is **3.11** (current PySAL line). `.python-version` pins local dev to 3.12.
- On Intel macs, giddy's numba chain needs `numba<0.61` + `numpy<2.1`; this is handled by
  `[tool.uv] constraint-dependencies` — do not remove it.

## The three-input data contract

Users provide:
1. `gdf` — geometry (shapefile / zipped shapefile / GeoJSON / GeoPackage or a
   GeoDataFrame) with ONLY the entity ID (+ optional name) and geometry → `read_gdf()`
2. `df` — a long-form panel (entity, time, variables) → declared via `set_panel()` /
   `set_labels(df, df_dict, set_panel=True)`
3. `df_dict` — a 6-column data dictionary: `var_name, var_def, label, type, role, can_be_na`
   (`type` ∈ entity/time/factor/logical/numeric; `role` ∈ ""/outcome/covariate/entity_name)

## Function conventions (the expdpy rulebook)

- Public functions are module-prefixed: `explore_*` (ESDA, maps, space-time descriptives)
  and `analyze_*` (estimation/inference). Cross-cutting helpers are unprefixed utilities.
- Signatures: DataFrame first, focal variable(s) next, everything else keyword-only after
  `*`. `entity`/`time` default to `None` and resolve from `df.attrs` (explicit arg wins).
  `gdf` and `w` are always explicit keyword arguments — geometry is data, never attrs.
- Every public function returns a **frozen dataclass** from `_types.py` exposing `.df`
  (tidy frame), `.fig` (Plotly) and/or `.gt` (Great Tables), named scalars, `notes`,
  and `w_spec` (human-readable weights description) where spatial. Most mix in
  `Interpretable` → `.interpret()` (plain-language, association-only: never "causes" or
  "effect of") and `.explain()` (concept explainer).
- Figures: Plotly only, themed via `_theme.apply_default_layout`; entity hover through
  `customdata`; never call `.show()`. Maps: `go.Choroplethmap` (MapLibre tiles) or
  `tiles=None` → `go.Choropleth` vector (deterministic PNG export). No matplotlib/splot.
- Validation order in every function body: `ensure_dataframe`/`ensure_geodataframe` →
  `resolve_panel(...)` → missing column `KeyError` → non-numeric `TypeError` → too few
  rows / zero variance `ValueError` → estimate → build themed figure → frozen result.
  Advisory degradation uses `GeometricsWarning` only.
- All df/gdf/W alignment goes through `_geo._align_cross_section` / `_align_panel_wide`
  (gdf/W row order; `w_subset` + re-standardize when rows drop). Never hand-align.
- spreg impacts are computed in-package from `betas` + `vm` (term lookup by `name_x`
  labels), never scraped from printed summaries.

## Tests

- `make test` = offline suite (`-m "not network"`). Markers: `network` (real data
  download), `dynamics` (giddy extra), `slow`.
- Known-answer style: synthetic lattice DGPs with planted parameters (SAR field, planted
  β-convergence, planted Markov chain); result-surface tests (frozen, fig traces,
  interpret() has no causal words); paper-parity tests against the India Table 1.

## Version sync

The version lives in TWO places: `pyproject.toml` and `src/geometrics/__init__.py`
(`__version__`). Keep them in sync when bumping.

## Case-study data

`geometrics.data.load_india()` fetches the paper's files from GitHub raw URLs pinned to
a commit SHA (see `data/_registry.py`), cached via pooch. Never point loaders at
branches, never bundle the heavy files in the wheel. The authored dictionary CSVs in
`src/geometrics/data/` ARE bundled and are user-facing documentation — keep them precise.
