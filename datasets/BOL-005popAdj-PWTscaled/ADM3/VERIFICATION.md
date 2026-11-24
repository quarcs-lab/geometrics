# Bolivia ADM1–ADM3 audit and ADM3 verification

[Construction and inputs](README.md) ·
[Executable notebook](../../../notebooks/bolivia343_construction.ipynb)

## 1. Correcting the archived geometry

The [2026 provider definitions](https://bfidatastudio.org/wp-content/uploads/2026/05/Data-Definition.pdf)
and [replication code](https://github.com/jialingzhang0730/Local_GDP_Estimates_Around_the_World/blob/329eb0af15029b9548a7046369157fa91c183906/step3_obtain_cell_level_GDP_and_predictors_data/4_get_true_cell_GDP.R#L161-L171)
confirm that longitude and latitude name the southwest corner of the
full quarter-degree rectangle. All 1,603 distinct coordinate keys
independently matched the upstream 1° cell and nested quadrant IDs.
The archived geometrics GPKG built squares centered on the source
coordinate, shifting each square 0.125° west and south. This error
propagated into the archival GADM ADM1/ADM2 assignments and the initial
ADM3 draft. The corrected builders construct the exact full rectangles
from the southwest coordinates. They do not claim to reproduce the
provider's irregular country-clipped polygons. The [source's clipping
code](https://github.com/jialingzhang0730/Local_GDP_Estimates_Around_the_World/blob/329eb0af15029b9548a7046369157fa91c183906/step5_predict_and_post_adjustments/3_obtain_country_xdeg_intersect_geom.R#L15-L37)
subtracts large waters before intersecting cells. Access to the release
archive timed out, so equivalence of source and SDSN support at borders
or Lake Titicaca has not been verified. The original grid GDP,
population, and PWT anchors are unchanged.

| Check | Superseded quarter-degree geometry | Corrected quarter-degree geometry |
|---|---:|---:|
| Source cells with no positive municipal overlap | 48 | 6 |
| Retained cell-to-municipal links (>1 m²) | 3,448 | 3,464 |
| Area of the full municipal map covered | 99.4580% | 99.9988% |
| Lowest single-unit geometric coverage | 58.71% | 99.90% |
| Median 2022 absolute GDP change from correction, relative to the old build | – | 37.4% |

The corrected map supports all 343 units and has 3,773 unique annual keys.
342 units have effectively full coverage and one is partial. Coverage
measures land area, not economic accuracy. The border audit records 19
source cells with positive coverage smaller than 1% of the full cell.
A matched cell's entire Bolivia-specific total goes to its matched
municipal areas in the main method. Estimates and flags are carried at
the municipal level; the parent loaders do not average child coverage or
censoring fractions.

## 2. Grid accounting and allocation sensitivity

This section uses the original `pop_pwt` denominators and the corresponding
`gdppc_pwt` ratios. It does not compare INE projections or census counts.

Annual municipal allocation plus unassigned mass reconciles to the
original GRID totals. Across 33 year-by-method rows, the maximum
absolute residual was `3.12e-10` million 2021 PPP US dollars for GDP and
`3.24e-14` million persons for population. Original GRID annual sums
match the existing national ADM0 PWT totals to numerical precision.
There is no second rebasing.

| Year | GDP unassigned, main method | Population unassigned, main method |
|---:|---:|---:|
| 2012 | 0.000149% | 0.000068% |
| 2013 | 0.000168% | 0.000133% |
| 2014 | 0.000143% | 0.000131% |
| 2015 | 0.000165% | 0.000129% |
| 2016 | 0.000140% | 0.000127% |
| 2017 | 0.000145% | 0.000135% |
| 2018 | 0.000130% | 0.000124% |
| 2019 | 0.000144% | 0.000131% |
| 2020 | 0.000156% | 0.000120% |
| 2021 | 0.000142% | 0.000127% |
| 2022 | 0.000157% | 0.000117% |

| Placement rule | 2022 GDP unassigned | 2022 population unassigned | Units with GDP per capita |
|---|---:|---:|---:|
| Covered municipal area (main) | 0.000157% | 0.000117% | 343 |
| Full rectangle denominator | 1.561% | 2.026% | 343 |
| Strict centroid, no nearest fallback | 1.365% | 1.748% | 277 |

Across common 2022 units, GDP-per-capita rank correlation with the main
method was **0.996** for the full rectangle and **0.886** for strict
centroid assignment. Under the centroid rule, **18 of 277** comparable
units differed by more than 10%, and 66 lacked coverage. A different
allocation rule tests sensitivity; it does not produce confidence limits.
The strict centroid uses corrected footprints and is not the archival
GADM method, which uses offset locations and a nearest fallback.

## 3. Population source and crosswalk checks

- **One-to-one:** 333 municipal units inherit the published annual INE
  projection unchanged; all 11 projection years are present.
- **Split:** 10 municipal units use 2012/2024 census population shares
  interpolated within the four documented groups. Check that every
  annual group's shares total one before rounding and its target integer
  projections sum to its pooled historical municipalities after rounding.
  Check the department and national sums independently against the source
  tables. These are consistency checks, not empirical validation of each
  disaggregated unit.
- **Census:** The separate 2012 denominator sums to **10,059,856** people.
  It is the census count on the fixed footprint, not the 2012 projected
  total. It is present for 343 2012 rows only. Using the 2012 and 2024 counts
  to compute the 10 split series is calibration, not a held-out
  comparison. The 2024 census is not a matched 2022 observation.
- **Accounting and units:** Preserve the original GDP and GRID-population
  ledger. The three ratio checks are `gdppc = 1e6 × gdp_pwt / pop_ine`,
  `gdppc_pwt = gdp_pwt / pop_pwt`, and (2012 only)
  `gdppc_census_2012 = 1e6 × gdp_pwt / pop_census_2012`. These are checks
  of formulas, not evidence of municipal GDP accuracy. Rollups sum the
  corresponding denominators separately and recompute each ratio.

| Year | Harmonized INE projected population (people) |
|---:|---:|
| 2012 | 10,356,978 |
| 2013 | 10,521,247 |
| 2014 | 10,685,994 |
| 2015 | 10,851,103 |
| 2016 | 11,016,438 |
| 2017 | 11,181,861 |
| 2018 | 11,347,241 |
| 2019 | 11,512,468 |
| 2020 | 11,677,406 |
| 2021 | 11,841,955 |
| 2022 | 12,006,031 |

## 4. Validation from ADM1 through ADM3

- **Geography:** Pinned Observatory map343 dissolves match the published
  [geo4bolivia](https://quarcs-lab.github.io/geo4bolivia/articles/data-model.html)
  boundaries exactly for all **9 departments and 113 provinces** in
  an equal-area comparison, with zero symmetric-difference area.
  GDP and population match across direct and sequential ADM3 rollups.
  In an independent direct overlay, reconstructed parent amounts agree
  up to the overlap threshold and numerical rounding.
- **Legacy reproduction:** The stored archived GRID assignment keys
  reproduce the GADM ADM1 and ADM2 panels (largest GDP difference
  below `7.3e-12` million PPP US dollars). This validates historical
  implementation, not the old coordinates. Only 107 of the 112
  legacy province geometries have estimates. The new geography's 113
  provinces are not interchangeable with GADM; one has no GADM ID,
  and shared identifiers do not guarantee equivalent polygon footprints.
- **Audit of the old GRID population denominator:** The 2012 geo4bolivia census
  benchmark matches all 343 units on the fixed map and was checked
  against the official 2012 INE workbook. Median absolute percentage
  error for modeled population dropped from **54.05% to 38.24%** at
  ADM3, **46.22% to 19.54%** at ADM2, and **9.29% to 4.94%** at ADM1.
  Municipal rank correlation improved **0.649 to 0.822**;
  **227 of 343** municipalities still differ by more than 25%.
  La Guardia remains at roughly 428,031 modeled people versus 89,284
  in the census. Those are different population concepts and totals:
  PWT 2012 provides a larger national denominator than the census.
  The difference cannot explain the local discrepancies. This motivated
  retaining the GRID denominator for comparison but using the INE projection
  as the default, without claiming the GDP numerator improved.
- **INE denominator diagnostic:** Comparing the new 2012 annual projections
  against separate 2012 census counts gives a **7.91%** median absolute
  difference across the 343 units (versus **38.24%** for corrected GRID
  populations). Among the 333 one-to-one matches it is **8.00%**. The
  [INE Revisión 2020 methodology, §7(d)](https://siip.produccion.gob.bo/noticias/files/2021-934cb-1estimaINE.pdf#page=44)
  blends estimated 2012 municipal shares with the 2012 census, weighting
  the census more. The 343-unit and 333-unit comparisons are therefore
  measures of census agreement, not independent holdouts. The ten split
  units additionally use 2012 and 2024 census shares. These statistics
  compare denominators, not the accuracy of municipal GDP.
- **External 2022 departmental GDP:** With
  [INE's nominal GDP shares](https://www.ine.gob.bo/referencia2017/pib_departamental.html)
  as a broad reference, the corrected 9-department rank correlation
  is **0.967**, with a mean absolute gap of **2.64 percentage points**.
  Santa Cruz remains **24.39%** in the modeled PPP rollup versus
  **32.23%** in the nominal data. The 2022 official values are
  preliminary. This is not a like-for-like price concept or a
  municipal GDP validation.

- **Departmental real GDP growth:** Across 45 official year-to-year changes
  (9 departments, 2018–2022), the unchanged corrected-area GDP numerator's
  mean absolute growth gap was **4.72 percentage points**. For 2017–2022,
  cumulative growth ranks correlated only **0.383**. Differences in the PWT
  and INE national series matter, but differences remain when each national
  growth series is removed: annual absolute error **3.96 log percentage
  points**. See [INE volume table D.10.2.1](https://www.ine.gob.bo/referencia2017/pib_departamental.html).
- **Municipal triangulation:** In the 2019 comparison with
  [FAM–AR-LAT](https://sistemaseconomicoslocales.fam.org.bo/sistemas-economicos-locales.html),
  both models used the same INE projected population after coarsening to
  common geography. Rank correlation was **0.214** across 337
  municipal groups and **0.136** across 111 provincial groups. FAM is a
  separate model with possible shared source inputs, not official
  municipal accounts. This is a check of disagreement in numerators,
  not an accuracy ranking between models.
- **Alternatives:** Nonresidential building weighting improved agreement
  with a 2012 resident-employment proxy on some metrics but not the
  official departmental GDP-growth benchmark. A Santa Cruz gas
  reallocation was sensitive to the assumed gas share and is not
  municipal GDP validation. Neither was adopted as a default.

Use the harmonized series for comparisons across scales within the same
product. The INE denominator is preferable for population accounting on
this geography, but municipal GDP and its per capita ratio remain exploratory.

## Reproduce

Run the notebook or the [command-line build](README.md#rebuild-and-use).
Inspect the ledger, crosswalk, and sensitivity files it generates.
Use `bolivia_adm3.csv` for grid-only diagnostics and the old-denominator
census comparison. Use `bolivia_adm3_ine.csv` to check parity with the public
loader and verify the new ratios. Use [the geography and accounting audit](../../../tools/audit_bolivia_scales.py)
and [the independent census comparison](../../../tools/validate_bolivia_census.py)
for the cross-scale checks. The census script identifies the exact INE/
geo4bolivia inputs and benchmarks the original `pop_pwt` allocation. It
should not be interpreted as validation of projections or split counts. The 2024 source
counts are audited for consistency but do not validate the 2022 model;
2024 is outside the modeled panel.


From a checkout containing the rebuilt panel, the census check can be
run independently (the optional Excel reader is installed only for
this run):

```bash
uv run --locked --no-dev --with openpyxl python tools/validate_bolivia_census.py \
  --adm3 /tmp/geometrics-bolivia343/bolivia_adm3.csv \
  --out /tmp/geometrics-bolivia343-census
```

For the geometry audit, pass `--boundaries-dir` pointing to the
pinned Observatory checkout's `geo/boundaries` folder and
`--output` pointing to a writable audit directory. The script checks
the hashes of `map343.geojson`, `province.geojson`, and
`department.geojson` before running.
