# BOL-005popAdj-PWTscaled / ADM2 — ADM2 (Bolivia)

> **Self-contained dataset.** This folder can be used on its own; everything below describes the files
> in this directory. It is one of four levels (ADM0/ADM1/ADM2/GRID) of the `BOL-005popAdj-PWTscaled` collection.

PWT-anchored local GDP for **Bolivia** at the **ADM2** level, 2012–2022 — the
0.25-degree gridded estimates **rescaled so national totals equal Penn World Table 11.0**, in
interpretable 2021 PPP US$. This dataset holds **107 level-2 regions × 11 years = 1,177 rows**.

## Method
The published 0.25-degree cells (constant 2021 PPP, `0_05` low-pop-density censoring) are **rescaled (raked) so their national totals match Penn World Table 11.0**. For each year, two national factors are applied uniformly to every cell:

```
gdp_scale[y] = PWT rgdpo[y] / SUM(model GCP)    gdp_pwt = model_GCP * gdp_scale  (mil 2021 US$)
pop_scale[y] = PWT pop[y]   / SUM(pop_cell)     pop_pwt = pop_cell  * pop_scale  (millions)
gdppc = gdp_pwt / pop_pwt   (2021 US$/person)   ln_gdppc = ln(gdppc)
```

Region totals are **sums of the rescaled cells**; `gdppc` is population-weighted (`Σ gdp_pwt / Σ pop_pwt`), never a mean of cell ratios.

## Files in this folder
- `bolivia_adm2.csv` — the panel as text (one row per observation)
- `bolivia_adm2.dta` — the panel as a labelled Stata 118 file
- `bolivia_adm2.gpkg` — boundaries + per-year `gdppc_<year>` columns (for choropleths)
- `bolivia_adm2.parquet` — the panel (fast columnar load)
- `bolivia_adm2_boundaries.gpkg` — the region boundary polygons (EPSG:4326)
- `bolivia_adm2_data_def.csv` — the machine-readable column dictionary (expdpy `df_def`)

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
| `gid` | entity | · | GADM region code at this level (GID_1 for adm1, GID_2 for adm2). Unique together with iso; a few disputed regions (e.g. Z0x codes) appear under more than one country. |
| `name` | factor | entity_name | GADM region name at this level (NAME_1 for adm1, NAME_2 for adm2). |
| `engtype` | factor | · | GADM region type in English (ENGTYPE_n; e.g. State, Province, Department, County). |
| `gid1` | factor | · | GADM GID_1 of the admin1 region this admin2 unit belongs to. |
| `name1` | factor | · | GADM NAME_1 of the parent admin1 region. |
| `iso` | entity | · | ISO3 country code (GADM GID_0); HKG/MAC/XKX restored to the cell-data codes. For region panels it is the parent country and part of the compound entity key. |
| `country` | factor | · | Country name (GADM COUNTRY / NAME_0). |
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
| `pop_nat` | numeric | · | MAX(national_population) for the country-year; reference only. At adm1/adm2 it is identical across a country's regions and is NOT a valid sub-national denominator. |

The same dictionary is in `bolivia_adm2_data_def.csv` (expdpy `df_def`: `var_name, var_def, label, type, role, can_be_na`).

## Nearest-region fallback — what it is and how to read it
Each 0.25-degree cell is assigned to the GADM region whose polygon **contains its centroid** (flagged `within`). A cell whose centroid falls outside every region of the country — typically a **coastal, island, or border** cell, or one over water near the coast — is instead **snapped to the nearest region of the same country**, measured in a local equidistant (metre) projection, and flagged `nearest`. The `n_cells_nearest` count (out of `n_cells`) records this; about **8%** of this dataset's cells were placed by the fallback.

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

**This level (ADM2).** The finest administrative detail, but **many regions rest on very few 0.25-degree cells** (some on one), so small-region estimates are coarse and noisier; the nearest-fallback share is higher than at ADM1, and GADM regions with no cell centroid are absent from the panel (they appear with missing GDPpc in the `.gpkg`).

## How the levels relate
This is the **ADM2** level of a four-level collection — **ADM0** (country), **ADM1** (level-1 regions), **ADM2** (level-2 regions) and **GRID** (0.25-degree cells). By construction `Σ ADM1 = Σ ADM2 = ADM0 = PWT national total` (and likewise for population) **every year**; the GRID cells aggregate up to all three administrative levels. Pick the level that matches your question: ADM0 for the national series, ADM1 for robust regional comparisons, ADM2 for the finest detail (with the caveats above), GRID for the raw spatial field.

## How to load
**pandas**
```python
import pandas as pd
df = pd.read_parquet("bolivia_adm2.parquet")     # or pd.read_csv("bolivia_adm2.csv")
```

**Stata**
```stata
use "bolivia_adm2.dta", clear
```

**expdpy** ([docs](https://cmg777.github.io/expdpy))
```python
import pandas as pd, expdpy as ex
df = pd.read_csv("bolivia_adm2.csv")
ddef = pd.read_csv("bolivia_adm2_data_def.csv")
df   = ex.set_labels(df, ddef, set_panel=True)   # entity=gid, time=year, outcome=ln_gdppc, entity_name="name"
ex.explore_panel_structure(df)
```

## Reproduce
```bash
python code/build_bolivia_pwtscaled.py
```
