# BOL-005popAdj-PWTscaled — PWT-anchored Bolivia gridded GDP (2012–2022)

A Bolivia subnational GDP product in which the **0.25° gridded GDP estimates** of
Rossi-Hansberg & Zhang (2026) — using their **most aggressive low-population-density censoring
(`0_05`)** and the **constant-2021-PPP** measure — are **rescaled so that their national totals
exactly equal Penn World Table 11.0** (`rgdpo`, output-side real GDP at chained PPPs, and `pop`),
and GDP per capita and its natural logarithm are then derived. The collection is delivered at four
levels of analysis — country (**ADM0**), department (**ADM1**), province (**ADM2**), and the raw
**GRID** cells — for **2012–2022**, in **EPSG:4326**. Rescaling re-bases the otherwise
arbitrary model scale into **interpretable 2021 PPP US dollars** (national GDP per capita reads as
**\$7,082 per person in 2012**) while preserving the underlying model's relative spatial
distribution of GDP and population exactly.

> **This README is self-contained and citation-grade.** It documents (1) the data sources, (2) the
> method by which the underlying cell-level GDP was originally estimated, and (3) — in full
> mathematical and computational detail — **how the rescaling implemented here works**. It is
> written to be quoted/paraphrased directly in the *Data* and *Methods* sections of academic
> papers. Per-level details and machine-readable column dictionaries live in each subfolder's
> `README.md` and `<level>_data_def.csv` (an [expdpy](https://cmg777.github.io/expdpy) `df_def`
> dictionary, paired with a long-form `<level>.csv` panel); numerical validation lives in
> [`VERIFICATION.md`](./VERIFICATION.md).

---

## 1. How to cite

This product is a derived dataset. If you use it, please cite the **underlying GDP estimates**, the
**national benchmark (PWT)**, and the **administrative boundaries (GADM)**:

- **Rossi-Hansberg, E., & Zhang, J. (2026).** Local GDP estimates around the world. *Journal of
  Urban Economics*, 154, 103871. https://doi.org/10.1016/j.jue.2026.103871 — data portal:
  Rossi-Hansberg, E., & Zhang, J. *Local Economies, Global Perspective: Illuminating Subnational
  GDP Worldwide* [data set], BFI Data Studio, University of Chicago, https://bfidatastudio.org/gdp.
- **Feenstra, R. C., Inklaar, R., & Timmer, M. P. (2015).** The Next Generation of the Penn World
  Table. *American Economic Review*, 105(10), 3150–3182. https://doi.org/10.1257/aer.20130954 —
  data: **Penn World Table version 11.0**, Groningen Growth and Development Centre, University of
  Groningen, https://www.rug.nl/ggdc/productivity/pwt/.
- **GADM (2022).** *Database of Global Administrative Areas*, version 4.10. https://gadm.org.

A suggested *Data* sentence: *"Subnational GDP is derived from the 0.25° gridded estimates of
Rossi-Hansberg & Zhang (2026), rescaled so that the Bolivian national totals match output-side real
GDP at chained PPPs (`rgdpo`) and population from Penn World Table 11.0 (Feenstra, Inklaar & Timmer
2015), and aggregated to GADM 4.10 departments and provinces."*

---

## 2. Overview and purpose

The published cell-level GDP values are reported on the model's **internally-consistent but
arbitrary scale**: summed over Bolivia they total only ≈ 85–117 (in the `const_2021_PPP` column),
so the absolute numbers cannot be read as money and per-capita values are not directly
interpretable. This product solves that by **anchoring the cells to an authoritative national
benchmark** — Penn World Table 11.0 — using a transparent, conservative **proportional rescaling
(raking)**. After rescaling:

- the **sum of all grid cells' GDP** in each year equals PWT `rgdpo` (millions of 2021 PPP US\$);
- the **sum of all grid cells' population** equals PWT `pop` (millions of persons);
- **GDP per capita** = GDP / population is therefore in **2021 PPP US\$ per person**, and at the
  national level it equals PWT `rgdpo/pop` exactly;
- the **relative spatial pattern** of GDP and of GDP-per-capita is unchanged (the rescaling only
  re-bases the level, never re-allocates across space — see §5).

The result is a four-level (grid → province → department → country) Bolivia panel whose numbers are
interpretable in 2021 PPP dollars and whose national aggregates are, by construction, consistent
with the Penn World Table.

---

## 3. Data sources

### 3.1 Underlying gridded GDP — Rossi-Hansberg & Zhang (2026)

The cell-level GDP is taken from *"Local GDP estimates around the world"* (Rossi-Hansberg & Zhang,
2026, *Journal of Urban Economics* 154:103871), which produces **gridded (cell-level) GDP for the
entire world, 2012–2022**, at three spatial resolutions (1°, 0.5°, 0.25°). This product uses the
**0.25° resolution** (cells ≈ 27.8 km across at the equator; they narrow toward the poles).

**Estimation method (summary; see the paper and `../BOL/README.md` for the authoritative text).**
The estimator does **not** predict GDP levels directly; it predicts, for each cell, its **share of
the GDP of its "parent" area** (the country, or — for the nine countries with state/province
training data: Australia, Brazil, Canada, China, India, Kazakhstan, Mexico, Russia, USA — the
state), then converts shares back to levels:

1. **Predictors → shares.** Satellite/environmental rasters are aggregated to each cell and
   expressed as the cell's **share** of its parent area's total. Working in shares removes
   country-specific level effects so the model generalizes to countries absent from training.
   Predictor families: **nighttime lights, population, CO₂ emissions, net primary productivity,
   land-cover shares, terrain ruggedness, and national GDP per capita**, plus **one-year lagged
   shares** to help predict growth.
2. **Training target.** Cell-level GDP "truth" for training regions is built by assuming uniform
   GDP per capita within each region with observed subnational GDP and distributing it by
   population: `y_i = Σ_r (Y_r / P_r) · p_{ir}`, where `Y_r/P_r` is region *r*'s GDP per capita and
   `p_{ir}` is the population of the intersection of region *r* and cell *i*.
3. **Model.** A **random forest** (Breiman 2001; `ranger` in R) maps predictor shares to GDP
   shares.
4. **Cross-validation.** **Group 5-fold cross-validation by country** — whole countries are held
   out, so reported skill reflects prediction for *unseen* countries, not merely unseen cells.
   Hyperparameters are tuned to maximize the weighted R² of year-over-year log changes.
5. **Prediction & rescaling to national accounts.** Predicted shares are **rescaled to sum to one
   within each parent area and multiplied by that area's GDP from national accounts** (national GDP
   from IMF World Economic Outlook, with World Bank / UN as backups). Reported out-of-sample fit at
   0.25° is high (R² ≈ 0.95 for levels and ≈ 0.82 for annual log changes; Table 1 of the paper).

**Currency measures.** The published cells carry four currency variants
(`const_2021_USD`, `current_USD`, `const_2021_PPP`, `current_PPP`), all with **base year 2021**; the
PPP conversion uses IMF WEO PPP factors. **This product uses `predicted_GCP_const_2021_PPP`** —
constant-2021 purchasing-power-parity "international dollars" (USA 2021 = 1) — chosen to be
conceptually consistent with the PPP benchmark (PWT `rgdpo`, also a 2021-based PPP measure).

**Uncertainty.** Each cell additionally carries cross-tree predictive uncertainty added in April
2026: 5th/95th prediction quantiles (`q05`, `q95`), an across-tree standard deviation (`tree_sd`),
and a currency-invariant SD of log GDP (`sd_log_gdp`). These are **cell-level** quantities computed
across the forest's trees; they are **not additive across cells** (see §5.6, §10).

### 3.2 The `0_05` low-population-density censoring (this product's threshold)

Because GDP per capita is poorly identified in nearly-unpopulated cells, the authors publish each
table under **four low-population-density censoring rules**, with thresholds of **none, 0.01, 0.02,
and 0.05 persons/km²**. **This product uses the most aggressive variant, `0_05`** (source file
`final_GDPC_0_25deg_postadjust_pop_dens_0_05_adjust.csv`; the cell `method` field reads *"post-adjust
zero GDP for pop density <= 0.05 (population per cell land area in km2)"*). Under `0_05`:

- the predicted GDP of every cell with **population density ≤ 0.05 persons/km² is set to zero**;
- the remaining cells' shares are **re-normalized to sum to one within each country × year** and
  multiplied by national GDP.

Consequences (all verified for Bolivia, see [`VERIFICATION.md`](./VERIFICATION.md) and `../BOL/`):
**(i)** the censored quantity is GDP (per capita), **not population — cell population is unchanged
and threshold-invariant**; **(ii)** national GDP is **conserved** (censoring redistributes GDP to
populated cells, it does not destroy it); **(iii)** in Bolivia, 125–266 cells per year are zeroed
under `0_05` (out of 1,603).

### 3.3 National benchmark — Penn World Table 11.0

National control totals are from **Penn World Table 11.0** (`pwt110.dta`; 185 economies, 1950–2023;
base year **2021**). Two variables are used (labels quoted verbatim from the file):

| PWT variable | Label (verbatim) | Unit | Role here |
|---|---|---|---|
| `rgdpo` | *Output-side real GDP at chained PPPs (in mil. 2021US\$)* | millions of 2021 PPP US\$ | GDP control total |
| `pop` | *Population (in millions)* | millions of persons | population control total |

`rgdpo` is the **output-side** (production-approach) real GDP at **chained PPPs**, i.e. converted
across countries at purchasing-power parity with a chained index and expressed in constant 2021 US
dollars; it is the standard PWT measure for comparing **real output levels** across countries and
over time. It is chosen here over alternatives — `rgdpe` (expenditure-side), `cgdpo` (current PPPs,
not inflation-adjusted across years), and `rgdpna` (national prices, not PPP) — because it is the
real, chained-PPP, output-side concept that matches the cells' constant-2021-PPP measure. PWT 11.0
is the natural benchmark because its **2021 base year** coincides with the cells' 2021-PPP base.

Bolivia (ISO `BOL`) values used (per year, 2012–2022); endpoints:

| Year | `rgdpo` (mil 2021 PPP US\$) | `pop` (millions) | `rgdpo/pop` (US\$/person) |
|---|---|---|---|
| 2012 | 74,472.97 | 10.5158 | 7,082.00 |
| 2022 | 114,239.13 | 12.0772 | 9,459.11 |

### 3.4 Administrative boundaries — GADM 4.10

Subnational units and geometries are from **GADM version 4.10** (`gadm_410.gpkg`; EPSG:4326). The
hierarchy used: `GID_0` = country (**Bolivia**), `GID_1` = **department** (9: La Paz, Santa Cruz,
Cochabamba, Potosí, Chuquisaca, Tarija, Oruro, Beni, Pando), `GID_2` = **province** (112). GADM is
distributed for academic/non-commercial use; see https://gadm.org for terms.

### 3.5 Upstream inputs to the original estimates (provenance only)

These feed the *original* Rossi-Hansberg & Zhang model and are listed for completeness; they are not
read by this product. (Sources as documented in the paper's online appendix and `../BOL/README.md`.)

| Input | Product / source | Role |
|---|---|---|
| Population | LandScan Global (Oak Ridge National Laboratory), ~1 km | predictor + cell population |
| Nighttime lights | VIIRS Black Marble VNP46A4 (NASA; Román et al. 2018), 500 m | predictor |
| Net primary productivity | MODIS MOD17A3HGF v6.1, 500 m | predictor |
| Land cover | MODIS MCD12Q1 v6.1, 500 m | predictor (urban/cropland/forest/other) |
| CO₂ emissions | EDGAR v8.0, 0.1°, six sectors | predictor |
| Terrain ruggedness | Terrain Roughness Index (Nunn & Puga 2012) | predictor |
| Water bodies | Global Lakes and Wetlands Database (GLWD-1) | masking |
| National GDP & population | IMF World Economic Outlook (WB/UN backup) | level anchor for the cells |
| Subnational GDP (training) | OECD Regional Statistics; DOSE v2.11 (Wenz et al. 2023); national statistical offices | training target |

---

## 4. Inputs actually used to build *this* product

This product is built **entirely from already-derived local files** — it does **not** re-read the
5 GB source CSVs, dissolve GADM, or perform a spatial join:

- **`../BOL/GRID/bolivia_grid_cells.parquet`**, filtered to `threshold == '0_05'`: the **1,603
  Bolivian 0.25° cells × 11 years**, already carrying the `const_2021_PPP` model GDP, `pop_cell`,
  uncertainty columns, and **GADM admin1/admin2 membership**. That membership was assigned in the
  upstream Bolivia build (`code/build_bolivia_cells.py`) by **centroid-in-polygon** containment
  against full-resolution Bolivian GADM, with a **nearest-region fallback** (local
  equidistant projection) for the 132 border/lake cells whose centroid falls outside all polygons
  (flagged `adm1_assign` / `adm2_assign` = `nearest`).
- **`pwt110.dta`**: Bolivia `rgdpo` and `pop`, 2012–2022.
- **`../BOL/ADM0|ADM1|ADM2/bolivia_adm{0,1,2}_boundaries.gpkg`**: the dissolved Bolivian country,
  department, and province polygons, copied unchanged into this collection.

---

## 5. Rescaling methodology — how the rescaling was implemented

This is the **novel step** of this product. It is a **per-year, two-margin proportional rescaling
(statistical "raking" / benchmarking)** of the grid cells to the PWT national control totals,
followed by aggregation. Implemented in **`code/build_bolivia_pwtscaled.py`**.

### 5.1 The problem it solves

The cells' `predicted_GCP_const_2021_PPP` values are internally consistent (relative magnitudes are
meaningful) but on an **arbitrary absolute scale** — Bolivia's national sum is ≈ 85–117 rather than
the ≈ 75,000–114,000 (millions of 2021 PPP US\$) that the economy actually produces. We want to
re-base the cells to a recognized national figure **without disturbing their relative spatial
distribution**. Proportional rescaling to a control total is exactly that operation.

### 5.2 Formal definition

Fix a year *y*. Index Bolivian cells by *i*. Let

- `g_i` = `predicted_GCP_const_2021_PPP` (model-scale cell GDP), with national sum `G_y = Σ_i g_i`;
- `n_i` = `pop_cell` (cell population, persons), with national sum `N_y = Σ_i n_i`;
- `R_y` = PWT `rgdpo` (millions of 2021 PPP US\$); `P_y` = PWT `pop` (millions of persons).

Define two **per-year scale factors** (one for GDP, one for population):

```
gdp_scale[y]  =  R_y / G_y          (≈ 873 … 981 for Bolivia)
pop_scale[y]  =  P_y / N_y          (≈ 1.010e-6 … 1.018e-6 for Bolivia)
```

and apply them **uniformly to every cell**:

```
gdp_pwt_i  =  g_i · gdp_scale[y]          # rescaled cell GDP, millions of 2021 PPP US$
pop_pwt_i  =  n_i · pop_scale[y]          # rescaled cell population, millions of persons
gdppc_i    =  gdp_pwt_i / pop_pwt_i       # GDP per capita, 2021 PPP US$ per person
ln_gdppc_i =  ln(gdppc_i)
```

The same `gdp_scale[y]` is also applied to the three predictive-uncertainty columns
(`q05`, `q95`, `tree_sd` → `gcp_pwt_q05`, `gcp_pwt_q95`, `gcp_pwt_tree_sd`), which is exact because
those are GDP-valued (quantiles are scale-equivariant; a standard deviation scales by the factor).

**Magnitudes (Bolivia).** Because `pop_scale` converts *persons* to *millions of persons*, it is
≈ 1.0×10⁻⁶; `gdp_scale` ≈ 873–981 absorbs both the unit change and the gap between the cells'
arbitrary scale and PWT's level. Their ratio `k_y = gdp_scale/pop_scale ≈ 8.6×10⁸ … 9.7×10⁸` is the
single constant by which every cell's GDP-per-capita is multiplied (see §5.4). Per-year values are
tabulated in [`VERIFICATION.md`](./VERIFICATION.md).

### 5.3 Implementation, step by step

The script `code/build_bolivia_pwtscaled.py` executes:

1. **`load_cells()`** — read `../BOL/GRID/bolivia_grid_cells.parquet`; keep `threshold == '0_05'`
   (1,603 cells × 11 years), with provenance and GADM-membership columns.
2. **`load_pwt()`** — read `pwt110.dta`; subset `countrycode == 'BOL'`, 2012–2022; rename
   `rgdpo → pwt_rgdpo`, `pop → pwt_pop`; compute `gdppc_pwt = pwt_rgdpo / pwt_pop`.
3. **`compute_scales()`** — group the cells by `year` to form `G_y = Σ g_i` and `N_y = Σ n_i`;
   merge the PWT totals on `year`; compute `gdp_scale = pwt_rgdpo / G_y` and
   `pop_scale = pwt_pop / N_y`. (Eleven rows, one per year.)
4. **`rescale_cells()`** — merge the two factors back onto the cells by `year`; form `gdp_pwt`,
   `pop_pwt`, and the rescaled uncertainty columns; derive `gdppc` with a **guarded division**
   (`gdp_pwt / pop_pwt` where `pop_pwt > 0`, else missing) and `ln_gdppc = ln(gdppc)` where
   `gdppc > 0` (else missing). See §5.6.
5. **`aggregate_level(level)`** — build the country/department/province panels. Region totals are
   the **sums of the already-rescaled cells**: `gdp_pwt = Σ_region gdp_pwt_i`,
   `pop_pwt = Σ_region pop_pwt_i`; region **GDP per capita is population-weighted**,
   `gdppc = gdp_pwt / pop_pwt` (never a mean of cell GDP-per-capita). The country (ADM0) panel also
   merges the PWT benchmark columns (`pwt_rgdpo`, `pwt_pop`, `gdppc_pwt`) and reports the residuals
   `gdp_resid = gdp_pwt − pwt_rgdpo` and `gdppc_resid = gdppc − gdppc_pwt` (both ≈ 0 by
   construction).

Because the factors are uniform within a year, **aggregating-then-rescaling and
rescaling-then-aggregating coincide**: a region's rescaled GDP equals its share of the national
total times `R_y`, and the regions sum back to `R_y` exactly.

### 5.4 Why this is statistically sound

Matching a set of disaggregated estimates to an authoritative aggregate by a common multiplicative
factor is **iterative proportional fitting (IPF / RAS) with a single marginal constraint**, which
has a **closed-form, one-step** solution (there is nothing to iterate when there is only one
margin). Among all non-negative reweightings of the cells that hit the control total, the uniform
factor is the **minimum-discrimination-information (minimum cross-entropy / Kullback–Leibler
I-projection)** solution: it satisfies the constraint while perturbing the model's relative
allocation **as little as possible — in fact, not at all.** This yields four exact invariants
(all checked at build time):

1. **National totals match exactly:** `Σ_i gdp_pwt_i = gdp_scale·G_y = R_y` and
   `Σ_i pop_pwt_i = P_y`. *(measured GDP residual 0; population residual 1.8×10⁻¹⁵.)*
2. **National GDP per capita equals PWT exactly:** `(Σ gdp_pwt)/(Σ pop_pwt) = R_y/P_y = gdppc_pwt`.
   *(measured |diff| ≤ 1.8×10⁻¹²; 2012 = \$7,082.00.)*
3. **Spatial shares are preserved:** `gdp_pwt_i / Σ gdp_pwt = g_i / G_y` — the rescaled cell GDP
   distribution is identical to the model's. *(measured max share drift 5.6×10⁻¹⁷.)*
4. **GDP-per-capita is re-based, not re-shaped:** `gdppc_i = (g_i/n_i)·k_y` with `k_y` a per-year
   constant. Every cell's GDP-per-capita is the model's cell GDP-per-capita multiplied by the same
   positive constant, so **rank order is preserved (Spearman = 1.000)** and **all scale-free
   dispersion measures — coefficient of variation, Gini, Theil, variance of logs — are unchanged**;
   in logs the transformation is a pure level shift `ln(k_y)`. *(measured: within-year SD of
   `ln_gdppc − ln(cell_GDPC_model)` ≤ 1.6×10⁻¹⁴.)*

A fifth, structural invariant follows from cells nesting in exactly one region at each level:
**`Σ ADM1 = Σ ADM2 = ADM0 = PWT`** for both GDP and population, every year.

### 5.5 What the rescaling does and does not do

- **Does:** re-base the *level* of the cells from the arbitrary model scale into PWT's units
  (millions of 2021 PPP US\$ for GDP; millions of persons for population) and re-anchor the national
  total to PWT `rgdpo`/`pop`. The per-year factor simultaneously absorbs the unit convention and the
  difference between the cells' original national-accounts anchor (IMF WEO) and PWT.
- **Does not:** **re-allocate** GDP or population across space. Raking treats PWT as the
  authoritative *total* and the model as the authoritative *relative geography*; any spatial bias in
  the underlying model is carried through unchanged. It also adopts **PWT's `rgdpo` concept and
  vintage**, which can differ from the IMF-WEO series the cells were originally anchored to — this
  re-anchoring is intentional (the goal is PWT consistency).

### 5.6 Edge cases (and why no value can blow up)

- **Censored cells** (`is_cell_censored == 1`, so `g_i = 0`) with positive population →
  `gdp_pwt = 0`, `gdppc = 0`, `ln_gdppc` **missing**. They retain their population, so they remain
  in the national denominator — which is exactly why the national GDP-per-capita still matches PWT.
- **Zero-population cells** (`n_i = 0`) → `pop_pwt = 0`; `gdppc` is set **missing** via the guarded
  division. In this dataset **every zero-population cell also has zero GDP**, so the expression is
  `0/0` → missing, **never `+∞`** — there are no infinities anywhere.
- Net effect: `gdppc` is missing exactly for zero-population cells; `ln_gdppc` is missing for every
  zero-GDP cell (the union of censored-with-population and zero-population). Counts per year are
  reported in [`VERIFICATION.md`](./VERIFICATION.md).
- **Uncertainty columns** are rescaled by `gdp_scale` and kept **only on the GRID**; they are
  cell-level predictive bands and **must never be summed across cells**. No uncertainty is added for
  the raking factor itself (PWT is treated as exact).

---

## 6. Units and interpretation

| Quantity | Column(s) | Unit |
|---|---|---|
| Rescaled GDP (cell / region) | `gdp_pwt` | **millions of 2021 PPP US\$**; national sum/yr = PWT `rgdpo` |
| Rescaled population | `pop_pwt` | **millions of persons**; national sum/yr = PWT `pop` |
| GDP per capita | `gdppc` | **2021 PPP US\$ per person**; national value = PWT `rgdpo/pop` |
| Log GDP per capita | `ln_gdppc` | natural log of `gdppc` |
| Scale factors | `gdp_scale`, `pop_scale` | per-year national raking factors (see §5.2) |
| Model provenance | `predicted_GCP_const_2021_PPP` / `gcp_model`, `pop_cell` / `pop_persons`, `cell_GDPC_const_2021_PPP` | original model-scale values (unitless model scale / persons) |

**Worked example (2012):** national `gdp_pwt` = 74,472.97 (= PWT `rgdpo`); `pop_pwt` = 10.5158
(= PWT `pop`); therefore `gdppc` = 74,472.97 / 10.5158 = **\$7,082.00 per person** (= PWT
`rgdpo/pop`). Note that, because population is expressed in *millions*, a single cell's `pop_pwt`
is a small number (e.g. a 6,600-person cell ≈ 0.0066), while its `gdp_pwt` is in millions of US\$;
their ratio is the interpretable per-capita figure.

---

## 7. Collection structure and contents

```
BOL-005popAdj-PWTscaled/
├── README.md                        ← this file (methods + data, paper-ready)
├── VERIFICATION.md                  ← build-time validation battery + per-year scale table
├── ADM0/   bolivia_adm0.{parquet,csv,dta,gpkg} + bolivia_adm0_boundaries.gpkg
│           bolivia_adm0_data_def.csv + README.md                  (11 rows: 1 country × 11 yr)
├── ADM1/   bolivia_adm1.{parquet,csv,dta,gpkg} + bolivia_adm1_boundaries.gpkg
│           bolivia_adm1_data_def.csv + README.md                  (99 rows: 9 depts × 11 yr)
├── ADM2/   bolivia_adm2.{parquet,csv,dta,gpkg} + bolivia_adm2_boundaries.gpkg
│           bolivia_adm2_data_def.csv + README.md                  (1,177 rows: 107 provinces × 11 yr)
└── GRID/   bolivia_grid_cells.{parquet,csv,dta,gpkg}
            bolivia_grid_cells_data_def.csv + README.md            (17,633 rows: 1,603 cells × 11 yr)
```

- **Formats per level:** `.parquet` (analysis), `.csv` (portable, long-form panel for
  [expdpy](https://cmg777.github.io/expdpy)), `.dta` (Stata 118, every column labeled), `.gpkg`
  (geometry, EPSG:4326). Each subfolder also has an expdpy `df_def` dictionary
  `<level>_data_def.csv` (`var_name, var_def, label, type, role, can_be_na`) and its own `README.md`.
- **Years:** 2012–2022 (annual; no gaps).
- **GeoPackages:** the `bolivia_<level>.gpkg` value files carry the boundary polygons plus per-year
  wide columns `gdppc_<year>` (and, for the grid, also `gdp_<year>` and `pop_<year>`). The ADM2
  value gpkg includes **all 112 GADM provinces**; the **5 provinces too small to contain any 0.25°
  cell centroid** (see §10) carry missing GDP-per-capita — present in the geometry but absent from
  the 1,177-row panel.

---

## 8. Variable dictionaries

Full machine-readable dictionaries are in each subfolder's `<level>_data_def.csv` (expdpy `df_def`). Key columns:

**GRID — `GRID/bolivia_grid_cells.*`** (primary key `cell_id, subcell_id, subcell_id_0_25, year`;
`cell_id` alone is **not** unique — it is the 1° parent id):

| Column | Meaning |
|---|---|
| `cell_id, subcell_id, subcell_id_0_25` | source cell identifiers (the latter two complete the key) |
| `iso, year, longitude, latitude` | country (BOL), year, cell centroid (EPSG:4326) |
| `predicted_GCP_const_2021_PPP` | **provenance:** original model GCP (model scale, const 2021 PPP) |
| `pop_cell` | **provenance:** original cell population (persons) |
| `cell_GDPC_const_2021_PPP` | **provenance:** original model GDP per capita (model scale) |
| `is_cell_censored` | source flag: 1 if GDP zeroed (pop density ≤ 0.05/km²) |
| `gdp_scale, pop_scale` | per-year raking factors (§5.2) |
| **`gdp_pwt`** | **rescaled GDP** (mil 2021 PPP US\$) |
| **`pop_pwt`** | **rescaled population** (millions) |
| **`gdppc`** | **GDP per capita** (2021 PPP US\$/person); 0 for censored, missing if zero-pop |
| **`ln_gdppc`** | ln(`gdppc`); missing where `gdppc ≤ 0` |
| `gcp_pwt_q05, gcp_pwt_q95, gcp_pwt_tree_sd` | rescaled cell-level uncertainty (mil 2021 PPP US\$); **not additive** |
| `national_population` | source national population (reference) |
| `country, gid_1, name_1, engtype_1, adm1_assign, gid_2, name_2, engtype_2, adm2_assign` | GADM membership + assignment method (`within` / `nearest`) |

**ADM0 / ADM1 / ADM2 panels** (`<level>/bolivia_<level>.*`; key `year` for ADM0, `gid, year`
otherwise) share: `level, gid, name, engtype, iso, country, year, threshold(=0_05), gcp_model`
(Σ model GCP), `pop_persons` (Σ pop_cell), `gdp_scale, pop_scale,` **`gdp_pwt`, `pop_pwt`,
`gdppc`, `ln_gdppc`,** `n_cells, n_cells_nearest`. **ADM2** adds parent `gid1, name1`. **ADM1/ADM2**
add `pop_nat` (parent national population, reference only). **ADM0** additionally carries the PWT
benchmark block `pwt_rgdpo, pwt_pop, gdppc_pwt, ln_gdppc_pwt` and the residuals `gdp_resid,
gdppc_resid` (≈ 0) so the exact match is self-evident in the file.

---

## 9. Verification and quality control

Every build runs an automated battery of **26 hard assertions** (the build aborts on any failure)
and writes the results, with the per-year scale-factor table, to
[`VERIFICATION.md`](./VERIFICATION.md). Headline results:

- **National match:** Σ grid `gdp_pwt` = PWT `rgdpo` (max abs diff 0.0); Σ grid `pop_pwt` = PWT
  `pop` (1.8×10⁻¹⁵).
- **Cross-level conservation:** Σ ADM1 = Σ ADM2 = ADM0 = PWT, for both GDP and population.
- **National GDP per capita:** ADM0 `gdppc` = PWT `gdppc` (|diff| ≤ 1.8×10⁻¹²; 2012 = \$7,082.00).
- **Share preservation:** max |rescaled share − model share| = 5.6×10⁻¹⁷.
- **Rank/shape preservation:** Spearman(`gdppc`, model GDP-per-capita) = 1.000 each year;
  within-year SD of the log difference ≤ 1.6×10⁻¹⁴.
- **Counts & keys:** 1,603 cells (132 nearest-fallback) every year; primary keys unique and
  complete (GRID 17,633; ADM0 11; ADM1 99; ADM2 1,177); each `gid_2` nests under its `gid_1`.
- **No infinities/negatives;** missing-value accounting matches the censored/zero-pop cell counts;
  the 2020 COVID dip is reproduced (`gdp_pwt[2020] < gdp_pwt[2019]`).

---

## 10. Caveats and limitations

- **Re-based, not re-estimated.** Absolute levels are now interpretable, but the *relative*
  geography is entirely the underlying random-forest model's; rescaling re-levels and cannot correct
  any spatial misallocation in the source.
- **PWT concept/vintage.** National totals adopt PWT 11.0 `rgdpo` (output-side, chained PPP, 2021
  base), which may differ from other GDP series (e.g. the IMF-WEO figures the cells were originally
  anchored to, or expenditure-side / current-PPP concepts). Choose the benchmark deliberately when
  comparing to other sources.
- **`0_05` censoring artifacts.** Under the most aggressive threshold, 125–266 cells/year are zeroed;
  these read as `gdppc = 0` (or missing where also zero-population). This concentrates GDP in
  populated cells — appropriate for per-capita interpretation, but note it when mapping sparse areas.
- **Uncertainty is cell-level and not aggregated.** Region-level GDP carries point estimates only;
  proper region-level uncertainty would require per-tree predictions that are not in the published
  cell files. The grid's `gcp_pwt_q05/q95/tree_sd` must not be summed across cells.
- **Centroid membership.** A ~28 km cell straddling a border is assigned wholly to one region; 132
  border/lake cells are assigned by nearest-region fallback (`adm1_assign`/`adm2_assign = nearest`).
- **Five small provinces carry no cell (ADM2 only).** Because assignment is **centroid-in-polygon**
  — a 0.25° cell (~27 km, ~740 km²) goes entirely to whichever province contains its centroid — the
  five *smallest* Bolivian provinces (189–555 km², each smaller than one cell and ≤ ~37 km across)
  contain **no grid centroid** and so receive no cell: they are absent from the 1,177-row panel and
  appear with missing values in the boundary/value GeoPackage. They are all interior provinces —
  **Cercado, Germán Jordán, Punata, Arani** (Cochabamba) and **Tomás Barrón** (Oruro) — so the
  nearest-region fallback (which only catches cells outside *all* Bolivian polygons) does not reach
  them; their economic mass is absorbed by the neighbouring province whose centroid claims the
  overlapping cell, **within the same department**, leaving national and department (ADM1) totals
  unaffected. This includes **Cercado, the urban province containing the city of Cochabamba**, so
  the artifact can affect a populous small province, not only sparsely-populated land (the
  Cochabamba-city cell is centroid-assigned to the adjacent Capinota province). It is a discretization
  effect of the 0.25° grid, not missing data; a finer-resolution input would resolve these provinces.
- **No independent subnational benchmark.** PWT is national only, so the *level* anchoring is
  validated at the country level; department/province levels inherit the national anchor and the
  model's relative shares — there is no province-level ground truth to validate against.

---

## 11. Reproducibility

Deterministic (no randomness, fixed inputs); rebuild in ≈ 3 seconds:

```bash
conda activate localgdp-val          # project env, Python 3.11
python code/build_bolivia_pwtscaled.py
```

Inputs: `data/BOL/GRID/bolivia_grid_cells.parquet` (threshold `0_05`), `pwt110.dta`, and
`data/BOL/{ADM0,ADM1,ADM2}/bolivia_adm{0,1,2}_boundaries.gpkg`. The script verifies all invariants
(§9) before writing and regenerates this collection (data, geometry, data dictionaries, per-level
READMEs, and `VERIFICATION.md`).

---

## 12. References

*Original estimates & method*
- Rossi-Hansberg, E., & Zhang, J. (2026). Local GDP estimates around the world. *Journal of Urban
  Economics*, 154, 103871. https://doi.org/10.1016/j.jue.2026.103871
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

*National benchmark*
- Feenstra, R. C., Inklaar, R., & Timmer, M. P. (2015). The Next Generation of the Penn World Table.
  *American Economic Review*, 105(10), 3150–3182. https://doi.org/10.1257/aer.20130954 — Penn World
  Table 11.0, Groningen Growth and Development Centre.

*Boundaries*
- GADM (2022). *Database of Global Administrative Areas*, version 4.10. https://gadm.org

*Selected upstream inputs to the original estimates*
- Román, M. O., et al. (2018). NASA's Black Marble nighttime lights product suite. *Remote Sensing
  of Environment*, 210, 113–143.
- Nunn, N., & Puga, D. (2012). Ruggedness: The blessing of bad geography in Africa. *Review of
  Economics and Statistics*, 94(1), 20–36.
- Wenz, L., et al. (2023). DOSE — Global data set of reported sub-national economic output.
  *Scientific Data*, 10, 425.

*Method context (rescaling)*
- Deming, W. E., & Stephan, F. F. (1940). On a least squares adjustment of a sampled frequency table
  when the expected marginal totals are known. *Annals of Mathematical Statistics*, 11(4), 427–444.
  (iterative proportional fitting / raking)

---

## 13. Attribution & terms

The GDP estimates are © the authors and distributed via **BFI Data Studio**
(https://bfidatastudio.org/gdp); consult that portal for terms of use. The article is © 2026
Elsevier Inc. **Penn World Table 11.0** is provided by the Groningen Growth and Development Centre
(cite Feenstra, Inklaar & Timmer 2015). **GADM 4.10** is for academic/non-commercial use
(https://gadm.org). Third-party remote-sensing inputs (NASA MODIS/VIIRS, EDGAR, LandScan, etc.) are
subject to their own licenses. This derived product is provided for research use; cite the three
primary sources in §1.
