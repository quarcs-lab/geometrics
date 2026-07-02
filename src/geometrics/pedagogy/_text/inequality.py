"""Explainers for inequality measurement: Gini, Theil, and decompositions.

Covers the workhorse inequality indices, their evolution over time as the
inequality-narrative complement of σ-convergence, and the additive between/within
decomposition of the Theil index (including its spatial reading).
"""

from __future__ import annotations

from geometrics.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="gini",
        title="The Gini index",
        what=(
            "The Gini index measures relative inequality as the average absolute "
            "difference between all pairs of units, scaled by twice the mean — "
            "equivalently, twice the area between the Lorenz curve and the equality "
            "diagonal. It runs from 0 (everyone equal) to values approaching 1 (one unit "
            "holds everything), is scale-invariant, and is most sensitive to differences "
            "around the middle of the distribution. Rey's **spatial Gini** splits the "
            "pairwise differences into *neighbor* and *non-neighbor* components under a "
            "weights matrix W, with permutation inference: when inequality lives mostly "
            "between non-neighbors, the map is polarized into rich and poor zones."
        ),
        when_to_use=(
            "The default headline measure for regional inequality over time "
            "(`analyze_inequality_over_time`). Add the spatial decomposition (pass a "
            "gdf/W) to ask whether inequality is spatially organized — the between-zones "
            "vs within-zones question at the pair level."
        ),
        caveats=(
            "One number cannot describe a distribution; pair it with densities "
            "(`explore_distribution_over_time`).",
            "Insensitive to where in the distribution a transfer happens compared with "
            "entropy measures; middle-sensitive by construction.",
            "Comparisons across differently-sized systems of regions need care — the "
            "index is population-blind when computed over unweighted regional means.",
        ),
        see_also=("theil_index", "sigma_convergence", "distribution_dynamics"),
        references=(
            "Rey & Smith (2013), 'A spatial decomposition of the Gini coefficient', "
            "Letters in Spatial and Resource Sciences",
        ),
    ),
    aliases=("gini_index", "spatial_gini"),
)

register_topic(
    Explainer(
        topic="theil_index",
        title="The Theil index",
        what=(
            "The Theil index is an entropy-based inequality measure: T = (1/n) Σ "
            "(y_i/μ) ln(y_i/μ). It is 0 under perfect equality, unbounded above, "
            "requires strictly positive values (the log), and is more sensitive to the "
            "top of the distribution than the Gini. Its defining virtue is **additive "
            "decomposability**: for any partition of units into groups, total Theil "
            "splits exactly into between-group plus within-group components — something "
            "the Gini cannot do cleanly."
        ),
        when_to_use=(
            "Whenever a decomposition question is on the table — how much of national "
            "inequality is between states vs within them? Report it alongside the Gini: "
            "agreement is robustness, divergence flags where in the distribution the "
            "action is."
        ),
        caveats=(
            "Zero or negative values are undefined — geometrics raises and names the "
            "offending units.",
            "Top-sensitivity means outliers move it more than the Gini.",
            "Less familiar to general audiences; label it clearly.",
        ),
        see_also=("theil_decomposition", "gini"),
        references=(
            "Theil (1967), Economics and Information Theory",
            "Shorrocks (1980), 'The class of additively decomposable inequality "
            "measures', Econometrica",
        ),
    ),
    aliases=("theil",),
)

register_topic(
    Explainer(
        topic="theil_decomposition",
        title="Theil between/within decomposition",
        what=(
            "For any grouping of units (states, macro-regions, coastal/interior), the "
            "Theil index decomposes *exactly* into a **between-group** term — the "
            "inequality that would remain if every unit were replaced by its group mean "
            "— and a **within-group** term, the group-size-weighted average of "
            "inequality inside each group. The **between share** (between/total) is the "
            "headline: a high share says the story is group membership (geography, in "
            "regional applications); a low share says most inequality lives inside "
            "groups. Tracked over time, the shares reveal whether national convergence "
            "is really between-region convergence, within-region convergence, or "
            "neither. Permutation inference (shuffling group labels) tests whether the "
            "between component exceeds chance."
        ),
        when_to_use=(
            "The first decomposition to run on any regional hierarchy — districts "
            "within states in the bundled India study. It turns 'inequality rose' into "
            "'inequality rose *between* states while falling within them', a far more "
            "actionable sentence."
        ),
        caveats=(
            "The split depends on the chosen partition; different hierarchies give "
            "different shares.",
            "Between/within is a *statistical* decomposition of a static partition — "
            "group composition changes over time muddy long comparisons.",
            "Small groups make the within terms noisy.",
        ),
        see_also=("theil_index", "gini", "spatial_weights"),
        references=(
            "Shorrocks & Wan (2005), 'Spatial decomposition of inequality', J. of "
            "Economic Geography",
            "Akita (2003), 'Decomposing regional income inequality in China and "
            "Indonesia', Annals of Regional Science",
        ),
    ),
    aliases=("between_within",),
)
