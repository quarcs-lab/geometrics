# BOL-005popAdj-PWTscaled / ADM0 — ADM0 (Bolivia)

> **Self-contained dataset.** This folder can be used on its own; everything below describes the files
> in this directory. It is one of four levels (ADM0/ADM1/ADM2/GRID) of the `BOL-005popAdj-PWTscaled` collection.

PWT-anchored local GDP for **Bolivia** at the **ADM0** level, 2012–2022 — the
0.25-degree gridded estimates **rescaled so national totals equal Penn World Table 11.0**, in
interpretable 2021 PPP US$. This dataset holds **11 years (2012–2022) for one country**. National GDP per capita is about **$7,082** (2021 PPP US$) in 2012.

## Method
The published 0.25-degree cells (constant 2021 PPP, `0_05` low-pop-density censoring) are **rescaled (raked) so their national totals match Penn World Table 11.0**. For each year, two national factors are applied uniformly to every cell:

```
gdp_scale[y] = PWT rgdpo[y] / SUM(model GCP)    gdp_pwt = model_GCP * gdp_scale  (mil 2021 US$)
pop_scale[y] = PWT pop[y]   / SUM(pop_cell)     pop_pwt = pop_cell  * pop_scale  (millions)
gdppc = gdp_pwt / pop_pwt   (2021 US$/person)   ln_gdppc = ln(gdppc)
```

Region totals are **sums of the rescaled cells**; `gdppc` is population-weighted (`Σ gdp_pwt / Σ pop_pwt`), never a mean of cell ratios.

## Files in this folder
- `bolivia_adm0.csv` — the panel as text (one row per observation)
- `bolivia_adm0.dta` — the panel as a labelled Stata 118 file
- `bolivia_adm0.gpkg` — boundaries + per-year `gdppc_<year>` columns (for choropleths)
- `bolivia_adm0.parquet` — the panel (fast columnar load)
- `bolivia_adm0_boundaries.gpkg` — the region boundary polygons (EPSG:4326)
- `bolivia_adm0_data_def.csv` — the machine-readable column dictionary (expdpy `df_def`)

## Data sources & how to cite
* **Gridded GDP** -- Rossi-Hansberg & Zhang, *Local GDP Estimates Around the World* (0.25-degree
  Gross Cell Product; the `0_05` variant zeroes the GDP, not the population, of cells below
  0.05 persons/km2).
* **National accounts** -- Feenstra, Inklaar & Timmer, *Penn World Table 11.0* (`rgdpo` = output-side
  real GDP at chained PPPs, mil. 2021 US$; `pop` in millions).
* **Administrative boundaries** -- GADM 4.10 (Database of Global Administrative Areas).

Please cite all three primary sources when using these data.

## Units & interpretation
Interpretable **PWT levels**: GDP in **millions of 2021 PPP US$**, population in **millions of persons**, GDP per capita (`gdppc`) in **2021 PPP US$ per person**, `ln_gdppc` its natural log. CRS **EPSG:4326**.

## Column dictionary
| Column | Type | Role | Description |
|---|---|---|---|
| `level` | factor | · | Aggregation level of the row (adm0=country, adm1=GADM L1, adm2=GADM L2). |
| `gid` | factor | · | GADM region code at this level (GID_1 for adm1, GID_2 for adm2). Unique together with iso; a few disputed regions (e.g. Z0x codes) appear under more than one country. |
| `name` | factor | · | GADM region name at this level (NAME_1 for adm1, NAME_2 for adm2). |
| `engtype` | factor | · | GADM region type in English (ENGTYPE_n; e.g. State, Province, Department, County). |
| `iso` | entity | · | ISO3 country code (GADM GID_0); HKG/MAC/XKX restored to the cell-data codes. For region panels it is the parent country and part of the compound entity key. |
| `country` | factor | entity_name | Country name (GADM COUNTRY / NAME_0). |
| `year` | time | · | Calendar year of the estimate (2012-2022). |
| `threshold` | factor | · | Low-population-density adjustment variant. Constant in each export (no_extra_adjust for model-scale panels, 0_05 for the PWT-scaled panels). |
| `gcp_model` | numeric | · | Sum of model GCP over the region's cells, model-scale const 2021 PPP (provenance). |
| `pop_persons` | numeric | · | Sum of cell population over the region, persons (provenance). |
| `gdp_scale` | numeric | · | Per-year national GDP raking factor = PWT rgdpo / sum(model GCP). |
| `pop_scale` | numeric | · | Per-year national population raking factor = PWT pop / sum(pop_cell). |
| `gdp_pwt` | numeric | · | GDP rescaled to PWT; national sum per year equals PWT rgdpo exactly. |
| `pop_pwt` | numeric | · | Population rescaled to PWT; national sum per year equals PWT pop exactly. |
| `gdppc` | numeric | · | gdp_pwt / pop_pwt, 2021 US$/person; 0 for censored cells, missing if zero population. |
| `ln_gdppc` | numeric | outcome | Natural log of gdppc; missing where gdppc <= 0 (censored / zero-pop). |
| `n_cells` | numeric | · | Number of 0.25-degree cells aggregated into this region-year. |
| `n_cells_nearest` | numeric | · | Of n_cells, the number placed by the nearest-region fallback (centroid outside all polygons; lower confidence). 0 at adm0. |
| `pwt_rgdpo` | numeric | · | Penn World Table 11.0 rgdpo for the country-year (national benchmark). |
| `pwt_pop` | numeric | · | Penn World Table 11.0 pop for the country-year. |
| `gdppc_pwt` | numeric | · | pwt_rgdpo / pwt_pop (the PWT benchmark per-capita). |
| `ln_gdppc_pwt` | numeric | · | Natural log of gdppc_pwt; undefined where gdppc_pwt <= 0. |
| `gdp_resid` | numeric | · | gdp_pwt - pwt_rgdpo (national raking residual; ~0). |
| `gdppc_resid` | numeric | · | gdppc - gdppc_pwt (raking residual; ~0). |

The same dictionary is in `bolivia_adm0_data_def.csv` (expdpy `df_def`: `var_name, var_def, label, type, role, can_be_na`).

## Nearest-region fallback — what it is and how to read it
**Not applicable at the country level.** Every cell of the country is aggregated here, so `n_cells_nearest = 0`. The fallback matters at ADM1 / ADM2 / GRID — see those folders.

## Merits & limitations
**Merits**

- **Interpretable levels.** GDP in millions of 2021 PPP US$, population in millions, GDP per capita in 2021 PPP US$ per person — directly comparable to national accounts.
- **Anchored to an authoritative benchmark.** National totals **equal Penn World Table 11.0 exactly** every year (`Σ cells = rgdpo`, `Σ pop = pop`).
- **Spatial pattern preserved.** Rescaling is a single per-year factor, so every cell's share and the **rank ordering are unchanged** (Spearman = 1.000); only the level is re-based.
- **Consistent across scales.** `Σ ADM1 = Σ ADM2 = ADM0 = PWT` each year; GRID cells aggregate up.
- **Reproducible & checked.** Built deterministically; a hard-assert battery runs before writing (see the collection's `VERIFICATION.md`).

**Limitations**

- **Re-levels, does not re-allocate.** PWT national totals are treated as authoritative and the model's *relative* geography as given — any spatial bias in the underlying estimates is carried through.
- **Adopts PWT's concept & vintage** (output-side real GDP at chained PPPs, `rgdpo`, 2021 base) — a different national concept/year would shift all levels.
- **`0_05` censoring.** Cells below 0.05 persons/km2 have GDP zeroed (population kept), so their `gdppc` is 0 or missing.
- **Cell-level uncertainty is NOT additive.** `gcp_pwt_q05/q95/tree_sd` (GRID) are per-cell predictive bands — never sum them across cells; no uncertainty is attached to the raking factor (PWT is treated as exact).
- **Approximate cell→region placement** for some cells (see *Nearest-region fallback*).
- **No independent subnational benchmark** — only the national level is externally anchored.

**This level (ADM0).** A single national series — exact and simple, but with no subnational variation; it carries the PWT block (`pwt_rgdpo`, `pwt_pop`, `gdppc_pwt`) and near-zero raking residuals, so the PWT match is self-evident in the file.

## How the levels relate
This is the **ADM0** level of a four-level collection — **ADM0** (country), **ADM1** (level-1 regions), **ADM2** (level-2 regions) and **GRID** (0.25-degree cells). By construction `Σ ADM1 = Σ ADM2 = ADM0 = PWT national total` (and likewise for population) **every year**; the GRID cells aggregate up to all three administrative levels. Pick the level that matches your question: ADM0 for the national series, ADM1 for robust regional comparisons, ADM2 for the finest detail (with the caveats above), GRID for the raw spatial field.

## How to load
**pandas**
```python
import pandas as pd
df = pd.read_parquet("bolivia_adm0.parquet")     # or pd.read_csv("bolivia_adm0.csv")
```

**Stata**
```stata
use "bolivia_adm0.dta", clear
```

**expdpy** ([docs](https://cmg777.github.io/expdpy))
```python
import pandas as pd, expdpy as ex
df = pd.read_csv("bolivia_adm0.csv")
ddef = pd.read_csv("bolivia_adm0_data_def.csv")
df   = ex.set_labels(df, ddef, set_panel=True)   # entity=iso, time=year, outcome=ln_gdppc, entity_name="country"
ex.explore_panel_structure(df)
```

## Reproduce
```bash
python code/build_bolivia_pwtscaled.py
```
