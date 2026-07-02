"""Explainers for distribution dynamics: Markov chains, spatial Markov, mobility.

These topics cover the Quah-style view of regional growth — following the *entire
cross-sectional distribution* over time rather than a single regression slope — and its
Markov-chain operationalization, including Rey's spatially conditioned variant.
"""

from __future__ import annotations

from geometrics.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="distribution_dynamics",
        title="Distribution dynamics",
        what=(
            "Distribution dynamics studies how the whole cross-sectional distribution of "
            "a variable (typically income per capita relative to the average) evolves: "
            "does it narrow (convergence), widen (divergence), or split into humps "
            "('twin peaks' — convergence clubs)? The toolkit follows the density of the "
            "*relative* variable over time (kernel densities per period) and summarizes "
            "movement within the distribution with transition matrices. It answers "
            "questions a single β cannot: β-convergence is compatible with a widening or "
            "polarizing distribution (Quah's critique)."
        ),
        when_to_use=(
            "Alongside β- and σ-convergence, whenever the *shape* of the regional "
            "distribution matters — emergence of clubs, polarization, hollowing middles. "
            "`explore_distribution_over_time` draws the density evolution; "
            "`analyze_markov_transitions` quantifies movement between distribution "
            "positions."
        ),
        caveats=(
            "Kernel density shapes depend on bandwidth; read humps cautiously.",
            "Relative (mean-normalized) values confound own growth with the average's "
            "growth.",
            "Distributional statements are descriptive; mechanisms need models.",
        ),
        see_also=(
            "markov_chains",
            "beta_convergence",
            "sigma_convergence",
            "convergence_clubs",
        ),
        references=(
            "Quah (1993), 'Empirical cross-section dynamics in economic growth', EER",
            "Quah (1997), 'Empirics for growth and distribution', J. of Economic Growth",
        ),
    ),
)

register_topic(
    Explainer(
        topic="markov_chains",
        title="Markov transition analysis",
        what=(
            "The Markov approach discretizes the cross-sectional distribution into k "
            "classes (e.g. quintiles of relative income) and counts how units move "
            "between classes from one period to the next. The estimated **transition "
            "matrix** P gives the probability of moving from class i to class j; heavy "
            "diagonals mean persistence, off-diagonal mass means mobility. Its "
            "**ergodic (steady-state) distribution** is the long-run class mix implied "
            "if current dynamics continued forever, and **sojourn times** say how long a "
            "unit typically stays in a class."
        ),
        when_to_use=(
            "To quantify persistence and long-run implications of regional dynamics: is "
            "the poorest quintile a trap (diagonal near 1)? Does the steady state pile "
            "mass at the extremes (polarization) or the middle (convergence)? Choose k "
            "and the discretization scheme to keep classes populated — quintiles per "
            "period is the literature default."
        ),
        caveats=(
            "Results depend on k and the class breaks; per-period quantiles measure "
            "movement *relative to the current distribution*, fixed breaks measure "
            "absolute movement.",
            "First-order Markov and time-homogeneity are assumptions, not findings.",
            "Sparse classes make transition estimates noisy; check the count matrix.",
            "The steady state is a thought experiment — 'if these dynamics persisted' — "
            "not a forecast.",
        ),
        see_also=("spatial_markov", "mobility_measures", "distribution_dynamics"),
        references=(
            "Quah (1993), 'Empirical cross-section dynamics in economic growth', EER",
            "Rey (2001), 'Spatial empirics for economic growth and convergence', "
            "Geographical Analysis",
        ),
    ),
    aliases=("markov", "transition_matrix"),
)

register_topic(
    Explainer(
        topic="spatial_markov",
        title="Spatial Markov chains",
        what=(
            "Rey's spatial Markov chain asks whether transition dynamics depend on the "
            "*neighborhood*: it estimates a separate k-by-k transition matrix for each "
            "class of the spatial lag — one matrix for units surrounded by poor "
            "neighbors, another for units surrounded by rich neighbors, and so on. "
            "Comparing the conditional matrices (and their steady states) reveals "
            "geography's grip on mobility: upward moves that are likely next to rich "
            "neighbors and rare next to poor ones are the signature of spatial poverty "
            "traps. Likelihood-ratio and Q homogeneity tests ask whether the "
            "conditional matrices differ significantly from the pooled one."
        ),
        when_to_use=(
            "After a classic Markov analysis, whenever LISA maps show clustering: it "
            "connects the distribution-dynamics and spatial-dependence views. Rejecting "
            "homogeneity means regional fortunes are not independent of neighbors — "
            "context for spillover modelling."
        ),
        caveats=(
            "Conditioning splits the data k ways; small samples produce sparse, "
            "degenerate conditional matrices (geometrics degrades the tests to NaN with "
            "a note when that happens).",
            "The neighbor class uses the same W as everything else — conclusions are "
            "conditional on it.",
            "Homogeneity tests are asymptotic; treat borderline p-values gently.",
        ),
        see_also=("markov_chains", "local_moran", "spatial_weights"),
        references=(
            "Rey (2001), 'Spatial empirics for economic growth and convergence', "
            "Geographical Analysis",
            "Rey, Kang & Wolf (2016), 'The properties of tests for spatial effects in "
            "discrete Markov chain models', J. of Geographical Systems",
        ),
    ),
)

register_topic(
    Explainer(
        topic="mobility_measures",
        title="Mobility indices",
        what=(
            "Mobility indices compress a transition matrix into one number. The "
            "**Shorrocks** index, (k - trace P)/(k - 1), is 0 when nothing ever moves "
            "(identity matrix) and rises toward k/(k-1) with mobility; the **Prais** "
            "(determinant-based) and **Bartholomew** (distance-weighted) variants weight "
            "movement differently — Bartholomew rewards *long* jumps across classes. "
            "Together they let you compare mobility across periods, regions, or "
            "variables on a common scale."
        ),
        when_to_use=(
            "To rank or track mobility: is the regional income ladder more fluid this "
            "decade than last? Report at least two indices — agreement across them is "
            "the robust finding."
        ),
        caveats=(
            "Each index embodies a value judgement about which moves count most.",
            "All inherit the discretization (k, scheme) of the underlying matrix.",
        ),
        see_also=("markov_chains",),
        references=("Shorrocks (1978), 'The measurement of mobility', Econometrica",),
    ),
    aliases=("shorrocks",),
)
