"""Explainers for spatial foundations: weights, autocorrelation, LISA, maps, CRS.

These topics cover the shared machinery of exploratory spatial data analysis — how
neighborhoods are encoded (``spatial_weights``, ``row_standardization``), how spatial
association is measured globally (``spatial_autocorrelation``) and locally
(``local_moran``), what a spatial lag is, how choropleth classes are formed, and why
coordinate reference systems matter.
"""

from __future__ import annotations

from geometrics.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="spatial_weights",
        title="Spatial weights (W)",
        what=(
            "A spatial weights matrix **W** encodes who counts as whose *neighbor*: cell "
            "(i, j) is nonzero when unit j is a neighbor of unit i. Contiguity criteria "
            "(**queen**: shared border or corner; **rook**: shared border only) suit "
            "administrative polygons; **k-nearest-neighbors** guarantees every unit exactly "
            "k neighbors; **distance bands** and **inverse-distance** weights encode decay "
            "with separation. Almost every spatial statistic — Moran's I, LISA, spatial "
            "regression, the spatial Markov chain — is defined *relative to a chosen W*: "
            "the weights are part of the hypothesis, not a nuisance detail."
        ),
        when_to_use=(
            "Build one W per analysis with `make_weights` and reuse it everywhere so "
            "results are comparable. Queen contiguity is the parameter-free default for "
            "polygon maps; KNN when polygons vary wildly in size or the data are points; "
            "distance decay when interaction plausibly weakens smoothly with distance. "
            "Check the graph with `explore_connectivity_map` before modelling: islands "
            "(zero-neighbor units) and disconnected components break estimators and "
            "conditioning. Test substantive conclusions against alternative W choices "
            "with `analyze_spatial_model_by_weights`."
        ),
        caveats=(
            "Results are conditional on W; report the construction rule (and geometrics "
            "records it as `w_spec` on every spatial result).",
            "Contiguity on real maps produces islands (literally, or via topology gaps); "
            "geometrics attaches them to their nearest neighbor by default and notes it.",
            "KNN weights are asymmetric (j can be i's neighbor without the converse), "
            "which some estimators handle differently than symmetric contiguity.",
            "Distance calculations need a metric (projected) CRS — degrees are not "
            "kilometres; geometrics projects automatically and warns.",
        ),
        see_also=(
            "row_standardization",
            "spatial_autocorrelation",
            "weights_robustness",
        ),
        references=(
            "Anselin (1988), Spatial Econometrics: Methods and Models, ch. 3",
            "Rey, Arribas-Bel & Wolf (2023), Geographic Data Science, ch. 4",
        ),
    ),
    aliases=("w", "weights"),
)

register_topic(
    Explainer(
        topic="row_standardization",
        title="Row standardization",
        what=(
            "Row standardization rescales each row of W to sum to one, so the spatial lag "
            "**Wy** becomes the *average* of unit i's neighbors' values rather than their "
            "sum. This makes lags comparable across units with different neighbor counts, "
            "bounds the spatial autoregressive parameter into an interpretable range, and "
            "is assumed by the textbook reading of the Moran scatterplot (slope = Moran's "
            "I) and by the impact formulas of spatial-lag models."
        ),
        when_to_use=(
            "Use it (the geometrics default) whenever the spatial lag should mean 'what "
            "my average neighbor looks like' — ESDA, convergence spillovers, Markov "
            "conditioning. Keep weights unstandardized only when totals matter, e.g. "
            "exposure or mass-interaction models."
        ),
        caveats=(
            "Standardization makes W asymmetric even when contiguity was symmetric.",
            "Units whose islands were attached to a single neighbor get a lag equal to "
            "that one neighbor's value — interpret their local statistics cautiously.",
        ),
        see_also=("spatial_weights", "spatial_lag"),
        references=("Anselin (1988), Spatial Econometrics, ch. 3",),
    ),
)

register_topic(
    Explainer(
        topic="spatial_autocorrelation",
        title="Spatial autocorrelation (Moran's I)",
        what=(
            "Spatial autocorrelation is the tendency of nearby units to resemble each "
            "other. **Moran's I** measures it globally as the cross-product between each "
            "unit's (standardized) value and its spatial lag: positive I means high values "
            "cluster near high values and low near low; negative I means checkerboard-like "
            "alternation; I near its expectation E[I] = -1/(n-1) means spatial randomness. "
            "Inference uses **conditional permutations**: values are reshuffled across the "
            "map many times to build a reference distribution, giving a pseudo p-value "
            "(`p_sim`). The Moran scatterplot draws value against lag; under a "
            "row-standardized W its regression slope *is* Moran's I."
        ),
        when_to_use=(
            "Run it first on any regional variable — levels and growth rates alike. "
            "Strong autocorrelation in a regression's residuals signals that OLS standard "
            "errors and possibly coefficients are unreliable and a spatial model is worth "
            "considering (`analyze_spatial_diagnostics`). Track it over time with "
            "`explore_moran_over_time` to see whether spatial structure is strengthening "
            "or dissolving."
        ),
        caveats=(
            "I is one number for the whole map; it can hide offsetting local pockets — "
            "pair it with the local version (LISA).",
            "The value and its significance depend on the chosen W.",
            "A significant I says values cluster; it does not say *why* — common shocks, "
            "spillovers, and omitted spatially-smooth covariates all produce it.",
        ),
        see_also=("local_moran", "spatial_weights", "lm_diagnostics"),
        references=(
            "Moran (1950), 'Notes on Continuous Stochastic Phenomena', Biometrika",
            "Anselin (1995), 'Local Indicators of Spatial Association — LISA', Geog. Anal.",
        ),
    ),
    aliases=("moran", "morans_i"),
)

register_topic(
    Explainer(
        topic="local_moran",
        title="Local Moran statistics (LISA)",
        what=(
            "Local Indicators of Spatial Association decompose global Moran's I into one "
            "statistic per unit, asking *where* the clustering lives. Each unit lands in a "
            "quadrant of the Moran scatterplot: **High-High** (a high value among high "
            "neighbors — a hot spot) and **Low-Low** (cold spot) are clusters; "
            "**High-Low** and **Low-High** are spatial outliers. Conditional permutations "
            "give each unit a pseudo p-value, and the LISA cluster map colors only the "
            "units significant at the chosen level, using the conventional red/blue "
            "palette shared with GeoDa and splot."
        ),
        when_to_use=(
            "Use it after a significant global Moran's I to localize the structure: which "
            "regions form the persistent high-income core, where are the poverty traps, "
            "which units buck their surroundings (outliers worth case studies). In "
            "convergence work, LISA maps of initial levels and of growth rates reveal "
            "whether catch-up is spatially organized."
        ),
        caveats=(
            "Many tests run at once (one per unit) — at α = 0.05 several 'significant' "
            "units are expected by chance; treat the map as exploratory.",
            "Pseudo p-values come from conditional permutations, not analytic nulls.",
            "Cluster labels depend on W and on the significance cutoff; report both.",
        ),
        see_also=("spatial_autocorrelation", "spatial_weights"),
        references=(
            "Anselin (1995), 'Local Indicators of Spatial Association — LISA', Geog. Anal.",
        ),
    ),
    aliases=("lisa",),
)

register_topic(
    Explainer(
        topic="spatial_lag",
        title="The spatial lag (Wy)",
        what=(
            "The spatial lag of a variable is its weighted average over each unit's "
            "neighbors, **Wy**. With row-standardized weights it answers 'what does my "
            "average neighbor look like?'. It is the workhorse of spatial analysis: the "
            "x-axis-vs-y-axis pair of the Moran scatterplot, the endogenous regressor of "
            "the spatial-lag (SAR) model, the lagged covariates of SLX/Durbin models, and "
            "the conditioning variable of the spatial Markov chain."
        ),
        when_to_use=(
            "Construct it whenever a hypothesis involves neighbors' outcomes or "
            "characteristics — spillovers, demonstration effects, shared shocks. In "
            "geometrics it is built internally by the ESDA and modelling functions; "
            "``libpysal.weights.lag_spatial(w, y)`` is the underlying call."
        ),
        caveats=(
            "A lag mixes neighbors' values with the map's edge effects: border units "
            "average over fewer, possibly unrepresentative neighbors.",
            "Lagging an already-noisy variable smooths noise too — lags always look "
            "smoother than the raw map.",
        ),
        see_also=("spatial_weights", "row_standardization", "spatial_lag_model"),
        references=("Anselin (1988), Spatial Econometrics, ch. 3",),
    ),
)

register_topic(
    Explainer(
        topic="choropleth_classification",
        title="Choropleth classification",
        what=(
            "A classified choropleth assigns each unit to one of k value classes and "
            "colors classes, not raw values — the map's message is largely decided by the "
            "classifier. **Fisher-Jenks** minimizes within-class variance (natural "
            "breaks); **quantiles** put equal counts in each class (good for skewed "
            "distributions, but can split near-identical values); **equal intervals** "
            "keep class widths constant (comparable across maps, but can leave classes "
            "empty); **user-defined** breaks pin classes to meaningful thresholds. "
            "geometrics classifies with mapclassify and draws one legend entry per class "
            "so classes can be toggled."
        ),
        when_to_use=(
            "Fisher-Jenks (the default) for a single map that should respect the data's "
            "own gaps; quantiles for rank-flavored comparisons; equal intervals or fixed "
            "user breaks when several maps (periods, variables) must share a scale — "
            "geometrics's animated maps pool the classification across periods for "
            "exactly this reason."
        ),
        caveats=(
            "Different classifiers can tell visually different stories from the same "
            "data — report the scheme and k.",
            "Class counts near 5-7 read best; more classes exceed color discrimination.",
            "Continuous (unclassified) color scales avoid the classification choice but "
            "make between-unit comparisons harder.",
        ),
        see_also=("local_moran",),
        references=(
            "Slocum et al. (2009), Thematic Cartography and Geovisualization",
            "Rey et al. (2023), Geographic Data Science, ch. 5",
        ),
    ),
    aliases=("mapclassify", "fisher_jenks"),
)

register_topic(
    Explainer(
        topic="crs_projections",
        title="Coordinate reference systems (CRS)",
        what=(
            "A CRS ties coordinates to the Earth. Geographic CRSs (like EPSG:4326, plain "
            "longitude/latitude) measure in **degrees**, whose ground length varies with "
            "latitude — so distances, areas, and centroids computed in degrees are "
            "distorted. Projected (metric) CRSs measure in metres and are the right frame "
            "for k-nearest-neighbor searches, distance-band weights, geometry "
            "simplification, and GWR bandwidths. geometrics's rule: analysis values are "
            "CRS-free, distance computations auto-project to a suitable UTM zone (with a "
            "warning), and web maps reproject to EPSG:4326 for rendering."
        ),
        when_to_use=(
            "Declare the CRS when reading geometry (`read_gdf(..., crs=...)` if the file "
            "lacks one). Accept the automatic UTM projection for distance-based weights "
            "unless your study area spans multiple zones — then pass an explicit "
            "equal-distance CRS suited to the region."
        ),
        caveats=(
            "A missing CRS is an error, not a guess — geometrics refuses to invent one.",
            "Auto-UTM suits regional extents; continental-scale maps deserve a "
            "purpose-chosen projection.",
            "Centroids of degree-based geometries are only approximately where you think "
            "they are; geometrics computes them after projecting.",
        ),
        see_also=("spatial_weights",),
        references=(
            "Lovelace, Nowosad & Muenchow (2019), Geocomputation with R, ch. 6",
        ),
    ),
    aliases=("crs", "projections"),
)
