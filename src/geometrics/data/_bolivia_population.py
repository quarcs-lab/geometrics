"""Load the bundled, census-supported Bolivia map343 population denominator."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd

YEARBOOK_PROJECTION = "ine_yearbook_projection"
CENSUS_SHARE_INTERPOLATED = "census_share_interpolated"
POPULATION_YEARS = tuple(range(2012, 2023))
INE_PROJECTED_TOTALS = (
    10356978,
    10521247,
    10685994,
    10851103,
    11016438,
    11181861,
    11347241,
    11512468,
    11677406,
    11841955,
    12006031,
)
INE_CENSUS_2012_TOTAL = 10059856
POPULATION_ASSET = "bolivia343_population.csv"
_KEY = ["geo_id", "year"]
_FIELDS = [
    "pop_ine_harmonized",
    "pop_ine_census2012",
    "pop_ine_method",
    "pop_ine_group",
]


def load_bolivia343_population() -> pd.DataFrame:
    """Return population in persons on map343, with method and group identifiers.

    The yearbook projection is unchanged for 333 matched units and allocated
    across 10 units in four boundary-change groups. The separate official 2012
    census tabulation is present in 2012 only. No network requests are made.
    """
    resource = files(__package__).joinpath(POPULATION_ASSET)
    with resource.open("rb") as stream:
        population = pd.read_csv(
            stream,
            dtype={
                "year": "int64",
                "geo_id": "str",
                "pop_ine_harmonized": "int64",
                "pop_ine_census2012": "Int64",
                "pop_ine_method": "str",
                "pop_ine_group": "str",
            },
        )
    if (
        list(population.columns) != [*_KEY, *_FIELDS]
        or len(population) != 343 * len(POPULATION_YEARS)
        or population.duplicated(_KEY).any()
        or population.geo_id.nunique() != 343
        or not population.groupby("geo_id")
        .year.nunique()
        .eq(len(POPULATION_YEARS))
        .all()
        or tuple(sorted(population.year.unique())) != POPULATION_YEARS
        or tuple(population.groupby("year").pop_ine_harmonized.sum())
        != INE_PROJECTED_TOTALS
        or population.pop_ine_harmonized.le(0).any()
        or population.pop_ine_census2012.notna().ne(population.year.eq(2012)).any()
        or population.pop_ine_census2012.sum() != INE_CENSUS_2012_TOTAL
        or not population.pop_ine_census2012.dropna().gt(0).all()
        or population.pop_ine_group.isna().any()
        or population.loc[population.year.eq(2012), "pop_ine_method"]
        .value_counts()
        .to_dict()
        != {YEARBOOK_PROJECTION: 333, CENSUS_SHARE_INTERPOLATED: 10}
    ):
        raise ValueError("Invalid bundled Bolivia map343 population panel")
    return population


def attach_bolivia343_population(panel: pd.DataFrame) -> pd.DataFrame:
    """Append checked population fields to a keyed municipal panel, in input order."""
    if not set(_KEY).issubset(panel.columns) or set(_FIELDS).intersection(
        panel.columns
    ):
        raise ValueError("Expected map343 geo_id and year without population columns")
    population = load_bolivia343_population()
    merged = panel.merge(
        population, on=_KEY, how="left", validate="one_to_one", sort=False
    )
    if merged.pop_ine_harmonized.isna().any():
        raise ValueError(
            "Some municipality-year keys are outside the bundled INE population panel"
        )
    merged.index = panel.index
    return merged
