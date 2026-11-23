# Bolivia GDP at the municipal level: ADM3 (map343)

This is the construction record for the **estimated municipal GDP and
GDP-per-capita panel for 2012–2022**. The default GDP numerator uses corrected
grid-area allocation; its population denominator uses INE Yearbook projections
harmonized to map343. These estimates are not official municipal GDP. The
target is the 343 municipalities and Indigenous autonomies in SDSN Bolivia's 2025 map, held fixed for all years.
Older observations are estimates for these footprints, not claims about
when any unit legally existed. The parent collection's GADM-based ADM1 and
ADM2 files are unchanged. The municipal panel is built directly from GRID,
**not** by dividing the ADM2 totals or interpolating GDP per capita.

- [Validation record](VERIFICATION.md)
- [Construction notebook (source)](../../../notebooks/bolivia343_construction.ipynb)
- [Colab after the implementation is on main](https://colab.research.google.com/github/quarcs-lab/geometrics/blob/main/notebooks/bolivia343_construction.ipynb)
- [Teaching guide](../../../docs/articles/bolivia343-construction.qmd)
- [Machine-readable data dictionary](../../../src/geometrics/data/bolivia343_dict.csv)

## Source versions

| Input | Role and frozen version |
|---|---|
| [Original GRID](../GRID/README.md) | The 1,603 full 0.25° cell footprints and 17,633 annual records. Rossi-Hansberg and Zhang (2026) modeled GDP using the `0_05` low-density censoring variant. The geometrics release already scales national GDP and population separately to PWT 11.0 `rgdpo` and `pop`. GRID source commit: `c23a78de67db2a72c3243c43887927675b685c7c`. |
| [INE Statistical Yearbook 2023, table 2.01.12](https://anuario.ine.gob.bo/2023/documentos/201/20112.xlsx) | Annual municipal population projections for both sexes (people), 2012–2022, on the source's 339-unit roster. The published projections and the splits below are distinct from historical census counts. |
| [INE census table 3.09.02.02](https://nimbus.ine.gob.bo/index.php/s/EgN8PPYD8Y4Rdig) | Population counts for the fixed 343 municipality/TIOC geography, both sexes. The 2012 counts provide the separate census denominator; 2012 and 2024 counts supply endpoint shares for the limited projection crosswalk below. |
| [SDSN Bolivia map343](https://quarcs-lab.github.io/geo4bolivia/articles/data-model.html#units-boundaries-and-crosswalks) | Municipal polygons and persistent `geo_id` from the GeoDevelopment Observatory of Bolivia distribution, commit `9a2e2b536b743700a69ca8b682ba57feb9bf7e26`. Do not substitute the older 339-unit map or assume GADM identifiers match these units. |

Exact source SHA-256 checksums:

| File | SHA-256 |
|---|---|
| `GRID/bolivia_grid_cells.csv` | `3af614177f84322211ead2366822f1baa6ad074773ae774de1515ce26e82f2e2` |
| `GRID/bolivia_grid_cells.gpkg` | `26df0f506929cc041969311fcbd69f369d24a1f99d890a05a564cf78e8971040` |
| `geo/boundaries/map343.geojson` | `19c2f1a4470b0f5642dd78850079d996593a5af64022db764f85a49391a959e3` |
| INE Yearbook `20112.xlsx` | `8cd56bb4ea1c4e4d7dab83e5f1f51acc5949513751c7bebac4a771fbf8eebaa4` |
| INE census `3090202.xlsx` | `fb0fd5af8614bf0ce25ed962d2b8048b39b58839e06a631bf6e7e2b1c2a470f7` |

## Construction

1. Form a unique grid key from the three source cell ID fields. Match
   the archived GRID geometry's roster to annual values by longitude and
   latitude. Do not use its polygons: it treated the source coordinates as
   centers. The [provider's 2026 definitions, p. 3](https://bfidatastudio.org/wp-content/uploads/2026/05/Data-Definition.pdf)
   and [replication code](https://github.com/jialingzhang0730/Local_GDP_Estimates_Around_the_World/blob/329eb0af15029b9548a7046369157fa91c183906/step3_obtain_cell_level_GDP_and_predictors_data/4_get_true_cell_GDP.R#L161-L171)
   identify the coordinates as southwest corners. For corner `(x,y)`,
   use the full square `[x,x+0.25] × [y,y+0.25]`. This recovers the
   documented regular grid, not the provider's original irregular
   border polygons. The municipal shapes use Observatory `geo_id`, not
   names.
2. Project both sets of polygons to the equal-area CRS **EPSG:6933**.
   Intersect them and retain overlaps larger than **1 m²**. Check that the
   inputs are valid polygons and that overlapping geometries do not double
   count source mass. Report coverage relative to each full source cell and
   each target polygon.
3. In the main specification, let `a[i,m]` be the retained overlap area for
   cell `i` and municipal unit `m`. For each cell with support, use
   `w[i,m] = a[i,m] / sum_m(a[i,m])`. Apply these **same fixed weights**
   separately to the existing Bolivia-specific `gdp_pwt` and `pop_pwt`
   cell totals, then sum over contributing cells by `geo_id` and year.
   This keeps the full modeled total of a matched cell in Bolivia even when
   the full rectangle crosses the boundary. A cell with no retained
   overlap stays **unassigned**; there is no nearest-place fallback and
   no second national scaling. The [ledger](VERIFICATION.md) accounts for
   municipal and unassigned totals separately.
4. Join `pop_ine` and the separate 2012 census counts to the finished GDP
   panel by `geo_id` and year. This changes no GDP allocation, original
   grid population or ledger totals. Calculate the default `gdppc` using
   INE projections, not the population allocated from GRID (see below).
5. Keep geography coverage distinct from inherited source GDP censoring.
   The source sets censored GDP to zero, not population. A missing
   geographic match is not a measured GDP zero. Report the number of
   contributing and censored cells, the geometric coverage fraction, and
   the covered area and GRID-allocated population fractions associated with
   censored source cells. Those flags describe the original spatial allocation,
   not the INE denominator.

Area weights assume GDP and the retained GRID population are distributed
proportionally to area within each cell's matched footprint. They do not
identify where within the cell production occurs. An INE denominator adds
municipal population information, not information about within-cell GDP. A
third of an area-allocated cell's GDP may not imply a third of its economic activity. Border cells
with only a small domestic overlap are a particular concern; the audit
shows where this occurs and how results change under different assumptions.

## The INE projection denominator

Use the all-sex totals from the frozen Yearbook, which cover the municipal
window of [INE Revisión 2020](https://www.ine.gob.bo/index.php/censos-y-proyecciones-de-poblacion-sociales/).
The public Revisión 2025 municipality/TIOC series starts in 2024; its
2000–2050 national series is not a municipal backcast. Of 343 target units,
**333** map one-to-one to a source municipality. This is exact
correspondence of the crosswalk keys and copies the source series
unchanged; it does not establish identical historical polygon edges.
The other **10** target units belong to four groups with **six** old
source municipalities in total:

| Linked group | Source projection totals pooled | Target units receiving the pool |
|---|---|---|
| Cochabamba | Mizque | Mizque; TIOC-Raqaypampa |
| Beni | San Ignacio; Santa Ana | San Ignacio; Santa Ana; TIOC-Territorio Indígena Multiétnico |
| Potosí (Antonio Quijarro) | Porco; Tomave | Porco; Tomave; TIOC-Jatun Ayllu Yura |
| Potosí (Chayanta) | Colquechaca | Colquechaca; San Pedro de Macha |

For target unit `m` in linked group `g`, calculate its share of the group
census population at each endpoint `s[m,2012]` and `s[m,2024]`. The
allocation weight in year `t` is
`(1 − u) × s[m,2012] + u × s[m,2024]`, where `u = (t − 2012)/12`.
Multiply by the sum of the published INE projections for the old members
of that group and year. The **shares** are interpolated, not the group
population totals. Group, department, and national projected totals are
preserved when converting to people by largest-remainder rounding, with
municipal IDs (`geo_id`) breaking ties in ascending order. These
are retrospective estimates, informed by the 2024 census, not data that
were known as of each historical year. The method flag `pop_ine_method`
records `ine_yearbook_projection` or `census_share_interpolated`. A split value is not an individually published INE projection, even for
an old-name unit in the group.

The 2012 population comparison is **census agreement**, not independent
validation: [INE's methodology, §7(d)](https://siip.produccion.gob.bo/noticias/files/2021-934cb-1estimaINE.pdf#page=44)
uses the 2012 census to adjust its 2012 municipal shares, and the split
adds a second use of 2012 census data for ten units. No census population
is used to alter the GDP numerator. PWT 11.0's `rgdpo` anchor measures
output-side real GDP at chained PPPs; it is not `rgdpna` (real GDP using
national-accounts growth rates) or INE territorial accounts.

| Field | Units and formula |
|---|---|
| `gdp_pwt` | Modeled GDP in million 2021 PPP US dollars, corrected area allocation |
| `pop_ine` | People, INE projection matched to the fixed municipal footprint or allocated within a linked group |
| `pop_ine_group` | Historical map339 identifier or plus-separated group of source identifiers used for this projection |
| `gdppc`, `ln_gdppc` | `1e6 × gdp_pwt / pop_ine`, 2021 PPP US dollars per person; natural log of a positive ratio |
| `pop_pwt` | Original GRID-modeled population allocation in million people, unchanged |
| `gdppc_pwt`, `ln_gdppc_pwt` | `gdp_pwt / pop_pwt`, the former main ratio and its natural log |
| `pop_census_2012`, `gdppc_census_2012` | 2012 census people and `1e6 × gdp_pwt / pop_census_2012`, only in 2012; missing in other years |

The projected denominator in 2012 is not replaced with the census count.
Use the separate census ratio when a comparison specifically needs the
2012 population counted on this modern geography. Neither denominator
makes the GDP numerator observed. Building-surface weights and the
Santa Cruz natural-gas exercise remain separate, exploratory numerator
sensitivities and are not used in the loader's default allocation.

### Reproduce the population asset

The package bundles the derived [population table](../../../src/geometrics/data/bolivia343_population.csv),
not the raw INE workbooks. Its SHA-256 is
`e8f25cb572bbec0f7699a55580b37171aecf6edcc7583265d65a10440c4c6158`.
The [provenance JSON](../../../src/geometrics/data/bolivia343_population_metadata.json)
records the input pins, allocation anchors and exact totals of every split
group. Rebuild to a scratch location and compare bytes:

```bash
uv run --locked --no-dev --with openpyxl python tools/build_bolivia_population.py \
  --output /tmp/bolivia343_population.csv \
  --metadata /tmp/bolivia343_population_metadata.json
cmp /tmp/bolivia343_population.csv src/geometrics/data/bolivia343_population.csv
```

The build uses pinned sources by default. For an offline rebuild, provide
`--ine-projections`, `--ine-census` and `--gdo-local` paths to those exact
source bytes. When the loader runs, it joins the bundled denominator,
so it does not need to download the INE workbooks again.

## Rebuild and use

Run this from the root of a repository checkout containing ADM3:

```bash
git rev-parse HEAD
uv run --locked --no-dev python tools/build_bolivia_adm3.py \
  --datasets-root datasets/BOL-005popAdj-PWTscaled \
  --output /tmp/geometrics-bolivia343
```

The script checks the source file hashes and fetches the pinned map from its
original distribution. For an offline rebuild with local inputs, supply
`--map343 /path/to/map343.geojson`; it must match the pinned hash. Record
the checkout commit and input hashes for reruns.

| Local output | What it contains |
|---|---|
| `bolivia_adm3_ine.csv` | 3,773 rows: one per `geo_id` and year, with the annual INE denominator and the same per-capita fields as the default loader. |
| `bolivia_adm3.csv` | Grid-only intermediate for allocation and census diagnostics. Its legacy `gdppc` and `ln_gdppc` use `pop_pwt` (corresponding to the public loader's `gdppc_pwt` and `ln_gdppc_pwt`). |
| `bolivia_adm3_boundaries.gpkg` | 343 ID-only polygons for joining to the panel; CRS EPSG:4326. |
| `allocation_cell_weights.csv` | Source cell and municipal overlap pairs, overlap and full-cell area, main and full-footprint weights; unmatched cells have blank municipal IDs. |
| `allocation_ledger.csv` | Annual source, assigned, unassigned and precision-residual totals for each algorithm. |
| `sensitivity.csv` | Allocation-only diagnostic: municipal GDP, GRID population and their ratio under the main, full-footprint and strict centroid methods. The denominator is not `pop_ine`. |
| `validation.json` | Input pins and hashes, build settings and coverage and censoring summaries. |

For analysis in Python:

```python
from geometrics.data import load_bolivia343

gdf, panel, data_dictionary = load_bolivia343()
```

The Python loader constructs the enriched `bolivia_adm3_ine.csv` panel from
cached, checksum-verified grid and map inputs and the bundled population. The notebook additionally exposes the diagnostics.
Municipal geometry and prebuilt estimates are not duplicated in this
directory. Cite Rossi-Hansberg and Zhang, PWT 11.0, the INE Yearbook and
censuses, SDSN Bolivia (2025), and the Observatory distribution. The software
license does not grant
redistribution rights for the source map or its derivatives.


## Consistent ADM1 and ADM2 for comparison

```python
from geometrics.data import load_bolivia_harmonized

gdf1, adm1, d1 = load_bolivia_harmonized("adm1")  # 9 departments
gdf2, adm2, d2 = load_bolivia_harmonized("adm2")  # 113 provinces
gdf3, adm3, d3 = load_bolivia_harmonized("adm3")  # 343 map343 units
```

The coarser geometries are unions of map343 child units. The names and
parent keys come from the pinned [Observatory units registry](https://raw.githubusercontent.com/quarcs-lab/gdo4bolivia/9a2e2b536b743700a69ca8b682ba57feb9bf7e26/geo/units.csv)
(SHA-256 `345e64bb9ebc34a3ff070b8eab58c934b8583abf7fd933800083f43c9b8ab004`).
Sum GDP and each population denominator first, then divide; do not average
child GDP per capita. All three levels have the same assigned GDP and the same
total for each additive denominator. `n_population_split_units` counts
municipal children allocated using census shares, not the number of provincial
children or a measure of population uncertainty. The Beni split spans
province boundaries, so the harmonized ADM2 populations are not an
independently published provincial projection. Department totals match the
Yearbook projection by construction; the harmonized series is still a rollup.
The historical GADM tables remain separate and must not be joined by name or treated as the same geography.
