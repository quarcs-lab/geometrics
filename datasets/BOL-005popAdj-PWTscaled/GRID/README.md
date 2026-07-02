# BOL-005popAdj-PWTscaled / GRID — GRID (Bolivia)

> **Self-contained dataset.** This folder can be used on its own; everything below describes the files
> in this directory. It is one of four levels (ADM0/ADM1/ADM2/GRID) of the `BOL-005popAdj-PWTscaled` collection.

PWT-anchored local GDP for **Bolivia** at the **GRID** level, 2012–2022 — the
0.25-degree gridded estimates **rescaled so national totals equal Penn World Table 11.0**, in
interpretable 2021 PPP US$. This dataset holds **1,603 cells × 11 years = 17,633 rows**.

## Method
The published 0.25-degree cells (constant 2021 PPP, `0_05` low-pop-density censoring) are **rescaled (raked) so their national totals match Penn World Table 11.0**. For each year, two national factors are applied uniformly to every cell:

```
gdp_scale[y] = PWT rgdpo[y] / SUM(model GCP)    gdp_pwt = model_GCP * gdp_scale  (mil 2021 US$)
pop_scale[y] = PWT pop[y]   / SUM(pop_cell)     pop_pwt = pop_cell  * pop_scale  (millions)
gdppc = gdp_pwt / pop_pwt   (2021 US$/person)   ln_gdppc = ln(gdppc)
```

Region totals are **sums of the rescaled cells**; `gdppc` is population-weighted (`Σ gdp_pwt / Σ pop_pwt`), never a mean of cell ratios.

## Files in this folder
- `bolivia_grid_cells.csv` — the panel as text (one row per observation)
- `bolivia_grid_cells.dta` — the panel as a labelled Stata 118 file
- `bolivia_grid_cells.gpkg` — boundaries + per-year `gdppc_<year>` columns (for choropleths)
- `bolivia_grid_cells.parquet` — the panel (fast columnar load)
- `bolivia_grid_cells_data_def.csv` — the machine-readable column dictionary (expdpy `df_def`)

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
| `cell_id` | entity | · | Source grid-cell identifier (1-degree parent; not unique alone). Part of the compound cell entity key. |
| `subcell_id` | entity | · | Source subcell identifier. Part of the compound cell entity key. |
| `subcell_id_0_25` | entity | · | Source 0.25-degree subcell identifier. Part of the compound cell entity key. |
| `iso` | factor | · | ISO3 country code (GADM GID_0); HKG/MAC/XKX restored to the cell-data codes. For region panels it is the parent country and part of the compound entity key. |
| `year` | time | · | Calendar year of the estimate (2012-2022). |
| `longitude` | numeric | · | Cell centroid longitude (degrees, EPSG:4326). |
| `latitude` | numeric | · | Cell centroid latitude (degrees, EPSG:4326). |
| `predicted_GCP_const_2021_PPP` | numeric | · | Predicted cell GDP total (Gross Cell Product), const 2021 PPP. |
| `pop_cell` | numeric | · | Population of the cell (persons). |
| `cell_GDPC_const_2021_PPP` | numeric | · | Cell GDP per capita, const 2021 PPP (= GCP / pop_cell). |
| `is_cell_censored` | logical | · | 1 if the cell GDP was zeroed by the low-pop-density adjustment, else 0. |
| `gdp_scale` | numeric | · | Per-year national GDP raking factor = PWT rgdpo / sum(model GCP). |
| `pop_scale` | numeric | · | Per-year national population raking factor = PWT pop / sum(pop_cell). |
| `gdp_pwt` | numeric | · | GDP rescaled to PWT; national sum per year equals PWT rgdpo exactly. |
| `pop_pwt` | numeric | · | Population rescaled to PWT; national sum per year equals PWT pop exactly. |
| `gdppc` | numeric | · | gdp_pwt / pop_pwt, 2021 US$/person; 0 for censored cells, missing if zero population. |
| `ln_gdppc` | numeric | outcome | Natural log of gdppc; missing where gdppc <= 0 (censored / zero-pop). |
| `gcp_pwt_q05` | numeric | · | GCP 5th pct rescaled to PWT. Cell-level; do NOT sum across cells. |
| `gcp_pwt_q95` | numeric | · | GCP 95th pct rescaled to PWT. Cell-level; do NOT sum across cells. |
| `gcp_pwt_tree_sd` | numeric | · | Across-tree SD of GCP rescaled to PWT. Cell-level; not additive. |
| `national_population` | numeric | · | National population from the source dataset. |
| `country` | factor | · | Country name (GADM COUNTRY / NAME_0). |
| `gid_1` | factor | · | GADM admin1 code (GID_1) of the region containing the cell. |
| `name_1` | factor | · | GADM admin1 name (NAME_1). |
| `engtype_1` | factor | · | GADM admin1 type (ENGTYPE_1; e.g. Department). |
| `adm1_assign` | factor | · | How the cell was assigned to its admin1: 'within' (centroid in polygon) or 'nearest' (snapped fallback). |
| `gid_2` | factor | · | GADM admin2 code (GID_2) of the region containing the cell. |
| `name_2` | factor | · | GADM admin2 name (NAME_2). |
| `engtype_2` | factor | · | GADM admin2 type (ENGTYPE_2; e.g. Province). |
| `adm2_assign` | factor | · | How the cell was assigned to its admin2: 'within' or 'nearest' (snapped fallback). |

The same dictionary is in `bolivia_grid_cells_data_def.csv` (expdpy `df_def`: `var_name, var_def, label, type, role, can_be_na`).

## Nearest-region fallback — what it is and how to read it
Each 0.25-degree cell is assigned to the GADM region whose polygon **contains its centroid** (flagged `within`). A cell whose centroid falls outside every region of the country — typically a **coastal, island, or border** cell, or one over water near the coast — is instead **snapped to the nearest region of the same country**, measured in a local equidistant (metre) projection, and flagged `nearest`. The `adm1_assign` / `adm2_assign` flags ('within' vs 'nearest') records this; about **8%** of this dataset's cells were placed by the fallback.

**How to interpret it.** Read it as a *confidence signal*: a region (or cell) with a high `nearest` share rests partly on cells whose location-to-region match is approximate, so its estimate is less certain — be cautious with small, coastal, or island regions that sit on few cells. **National totals are unaffected** — every cell is counted exactly once and belongs to the country by source attribution; only the *within-country allocation across regions* carries this uncertainty.

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

**This level (GRID).** Full spatial resolution with raw provenance columns, but the 0.25-degree cell is **not an economic unit**; cell-level uncertainty is non-additive and censored / zero-population cells carry `gdppc = 0` or missing.

## How the levels relate
This is the **GRID** level of a four-level collection — **ADM0** (country), **ADM1** (level-1 regions), **ADM2** (level-2 regions) and **GRID** (0.25-degree cells). By construction `Σ ADM1 = Σ ADM2 = ADM0 = PWT national total` (and likewise for population) **every year**; the GRID cells aggregate up to all three administrative levels. Pick the level that matches your question: ADM0 for the national series, ADM1 for robust regional comparisons, ADM2 for the finest detail (with the caveats above), GRID for the raw spatial field.

## How to load
**pandas**
```python
import pandas as pd
df = pd.read_parquet("bolivia_grid_cells.parquet")     # or pd.read_csv("bolivia_grid_cells.csv")
```

**Stata**
```stata
use "bolivia_grid_cells.dta", clear
```

**expdpy** ([docs](https://cmg777.github.io/expdpy))
```python
import pandas as pd, expdpy as ex
df = pd.read_csv("bolivia_grid_cells.csv")
ddef = pd.read_csv("bolivia_grid_cells_data_def.csv")
# NOTE: GRID is a cell-level spatial layer, not a clean panel. set_panel would resolve
# entity="cell_id" (the 1-degree parent id, NOT a unique cell). For cell-level work use
# the .gpkg geometry or the (cell_id, subcell_id, subcell_id_0_25) key.
```

## Reproduce
```bash
python code/build_bolivia_pwtscaled.py
```
