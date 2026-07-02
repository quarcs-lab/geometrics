# geometrics

[![CI](https://github.com/quarcs-lab/geometrics/actions/workflows/ci.yml/badge.svg)](https://github.com/quarcs-lab/geometrics/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-quarcs--lab.github.io%2Fgeometrics-blue)](https://quarcs-lab.github.io/geometrics/)
[![PyPI](https://img.shields.io/pypi/v/geometrics.svg)](https://pypi.org/project/geometrics/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/geometrics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quarcs-lab/geometrics/blob/main/notebooks/quickstart.ipynb)

**geometrics** studies **regional growth, convergence, and inequality** with explicit
spatial methods. It builds on the excellent [PySAL](https://pysal.org) family —
[libpysal](https://pysal.org/libpysal/), [esda](https://pysal.org/esda/),
[giddy](https://pysal.org/giddy/), [inequality](https://pysal.org/inequality/),
[mapclassify](https://pysal.org/mapclassify/), [spreg](https://pysal.org/spreg/), and
[mgwr](https://mgwr.readthedocs.io/) — and wraps the standard analyses of the regional
convergence literature into illustrative, easy-to-apply functions that return
interactive [Plotly](https://plotly.com/python/) figures,
[Great Tables](https://posit-dev.github.io/great-tables/), and tidy DataFrames.

It follows the design language of [expdpy](https://github.com/cmg777/expdpy): every
function returns a typed result object with `.df`, `.fig`, plain-language
`.interpret()`, and concept `.explain()`.

## The data model: three inputs

| Input | What it is | How it enters |
|---|---|---|
| `gdf` | Geometry with **only the entity ID** — shapefile, zipped shapefile, GeoJSON, or GeoPackage (or a GeoDataFrame) | `gm.read_gdf("districts.gpkg", entity="district_id")` |
| `df` | A **long-form panel** — one row per (entity, time) | `gm.set_panel(df, entity="district_id", time="year")` |
| `df_dict` | A **data dictionary** — `var_name, var_def, label, type, role, can_be_na` | `gm.set_labels(df, df_dict, set_panel=True)` |

## Installation

```bash
pip install geometrics                 # core
pip install "geometrics[dynamics]"     # + Markov / spatial Markov (giddy)
pip install "geometrics[all]"          # everything, incl. PNG export
```

## Bundled case studies

- **India** — 520 districts observed by satellite nighttime lights (1996-2010), from
  [Mendez, Kabiraj & Li (quarcs-lab/project2025s-py)](https://github.com/quarcs-lab/project2025s-py):
  `gm.data.load_india()`, `load_india_states()`
- **Bolivia** — PWT-anchored local GDP (2021 PPP US$, 2012-2022) at three scales,
  derived from [Rossi-Hansberg & Zhang (2026)](https://bfidatastudio.org/gdp) and
  Penn World Table 11.0: `gm.data.load_bolivia()` (112 provinces),
  `load_bolivia_departments()` (9 departments), `load_bolivia_grid()` (1,603 cells) —
  see [`datasets/`](datasets/BOL-005popAdj-PWTscaled/) for the citation-grade documentation

## Quickstart: the Indian case study

```python
import geometrics as gm

gdf, df, df_dict = gm.data.load_india()      # ID-only geometry, long panel, dictionary
df = gm.set_labels(df, df_dict, set_panel=True)

gm.explore_choropleth_map(df, "ntl_total", gdf=gdf, period=2010).fig
w = gm.make_weights(gdf, method="knn", k=6)
gm.explore_lisa_cluster_map(df, "log_ntl_pc_1996", gdf=gdf, w=w).fig

res = gm.analyze_beta_convergence(
    df, "ntl_total", model="sdm", gdf=gdf, w=w
)
print(res.interpret())                       # plain-language reading
res.fig                                      # convergence scatter
```

## Features

- **Maps & ESDA** — classified/animated choropleths (`explore_choropleth_map`), weights
  connectivity (`explore_connectivity_map`), Moran scatterplots, LISA cluster maps,
  Moran over time
- **Space-time dynamics** — cross-sectional distribution evolution
  (`explore_distribution_over_time`), entity-by-time heatmaps
- **Convergence** — β-convergence with OLS or spatial (SAR/SEM/SLX/SDM) estimators and
  LeSage-Pace impact decomposition, σ-convergence, Phillips-Sul convergence clubs with
  club maps
- **Spatial econometrics** — the spreg suite (`analyze_spatial_model`), LM diagnostics
  with a model recommendation (`analyze_spatial_diagnostics`), alternative-weights
  robustness (`analyze_spatial_model_by_weights`)
- **Distribution dynamics** — Markov and spatial Markov transition analysis
  (`analyze_markov_transitions`, `analyze_spatial_markov`)
- **Inequality** — Gini/Theil trends with spatial decomposition
  (`analyze_inequality_over_time`), Theil between/within decomposition
  (`analyze_theil_decomposition`)
- **Local models** — GWR and multiscale GWR with mapped local coefficients
  (`analyze_gwr`, `analyze_mgwr`)

## Documentation

- Website: https://quarcs-lab.github.io/geometrics/
- The India case study article and Colab notebooks: see `docs/` and `notebooks/`

## Development

```bash
git clone https://github.com/quarcs-lab/geometrics
cd geometrics
uv sync --locked --all-extras --group dev --group docs
make test && make lint && make typecheck
```

## Citation

If you use geometrics in your research, please cite the repository (see
`CITATION.cff`) and the underlying PySAL packages.

## Acknowledgments

Developed at the [QuaRCS Lab](https://quarcs-lab.org) (Quantitative Regional and
Computational Science). geometrics stands on the shoulders of the
[PySAL](https://pysal.org) project, [geopandas](https://geopandas.org),
[Plotly](https://plotly.com/python/), and
[Great Tables](https://posit-dev.github.io/great-tables/).

## License

MIT
