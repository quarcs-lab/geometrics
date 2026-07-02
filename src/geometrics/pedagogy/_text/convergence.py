"""Explainers for convergence analysis: β-convergence, σ-convergence and convergence clubs.

``beta_convergence`` covers the growth-vs-initial-level regression (unconditional, conditional,
speed and half-life); ``sigma_convergence`` covers the dispersion-over-time view;
``convergence_clubs`` covers the Phillips-Sul log(t) test and data-driven club clustering.
"""

from __future__ import annotations

from geometrics.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="beta_convergence",
        title="Beta convergence",
        what=(
            "β-convergence asks whether units that start *behind* grow *faster* and so catch "
            "up. The test regresses each unit's average growth rate over a horizon on its "
            "**initial level** — canonically the growth of GDP per capita on initial log GDP "
            "per capita. A **negative** slope β is convergence: lower starting points are "
            "associated with faster growth. The slope maps to a structural **speed of "
            "convergence** λ = -ln(1 + β·T) / T (per period) and a **half-life** ln 2 / λ, the "
            "time to close half of an initial gap. **Unconditional** (absolute) convergence "
            "uses the initial level alone; **conditional** convergence adds controls for each "
            "unit's steady-state determinants and, by the Frisch-Waugh-Lovell theorem, reads "
            "the convergence slope from a partial-regression scatter that holds those controls "
            "fixed. The same machinery works for any variable — income, schooling, health."
        ),
        when_to_use=(
            "Use it to summarise catch-up dynamics in a panel: are poorer economies (or "
            "lower-scoring regions/firms) closing the gap, and how fast? Reach for "
            "**unconditional** convergence to describe raw catch-up, and **conditional** "
            "convergence when units have different steady states (different savings, human "
            "capital, institutions) so that catch-up is only expected *relative to* each "
            "unit's own steady state. A rolling-window version shows whether the convergence "
            "speed has itself changed over time."
        ),
        caveats=(
            "β-convergence is a *descriptive association* between growth and an initial level, "
            "not a causal mechanism; regression to the mean and measurement error in the "
            "initial level can both produce a negative slope (Galton's fallacy / Quah's "
            "critique).",
            "The estimate depends on the chosen start and end years and the horizon T — report "
            "them, and prefer a common window across units when comparing.",
            "Conditional convergence is conditional on the controls you include; a different "
            "control set implies a different steady state and can change the slope.",
            "Speed and half-life are only well defined when 1 + β·T > 0; a non-negative slope "
            "(divergence) has no finite positive half-life.",
        ),
        see_also=("fwl", "fixed_effects", "correlation_vs_causation"),
        references=(
            "Barro & Sala-i-Martin, Economic Growth (2nd ed.), ch. 11-12",
            "Sala-i-Martin (1996), 'The Classical Approach to Convergence Analysis', EJ",
        ),
    ),
    aliases=("convergence", "conditional_convergence"),
)

register_topic(
    Explainer(
        topic="sigma_convergence",
        title="Sigma convergence",
        what=(
            "σ-convergence asks whether the *cross-sectional dispersion* of a variable shrinks "
            "over time — whether units become more alike. At each period the dispersion is "
            "measured across units (the **standard deviation**, the **Gini index**, the "
            "**coefficient of variation**), and the test regresses the **log dispersion** on "
            "time: a **negative** slope means dispersion falls by a roughly constant proportion "
            "each period, the hallmark of σ-convergence. It is the distributional complement to "
            "β-convergence: β-convergence (poorer units growing faster) is *necessary but not "
            "sufficient* for σ-convergence, because new shocks can re-spread the distribution "
            "even while laggards catch up (Quah's critique)."
        ),
        when_to_use=(
            "Use it to describe whether a cross-section is compressing or fanning out over time "
            "— income or productivity across regions, test scores across schools, health across "
            "countries. Pair it with β-convergence: β answers 'do laggards grow faster?' while "
            "σ answers 'is the whole distribution narrowing?'. Report several dispersion "
            "measures, since the standard deviation is scale-dependent while the Gini and the "
            "coefficient of variation are scale-free."
        ),
        caveats=(
            "σ-convergence is a *descriptive* statement about the distribution, not a causal "
            "mechanism; a narrowing spread does not say why units converged.",
            "The standard deviation is in the variable's own units and grows with its level; "
            "the Gini and the coefficient of variation are scale-free and often tell a clearer "
            "story — compare them.",
            "Dispersion is only comparable across periods when the set of units is fixed, so a "
            "balanced panel is required; a changing composition can masquerade as convergence "
            "or divergence.",
            "The Gini index is only defined for non-negative values, and the coefficient of "
            "variation is unstable when the mean is near zero.",
        ),
        see_also=("beta_convergence", "fwl", "correlation_vs_causation"),
        references=(
            "Barro & Sala-i-Martin, Economic Growth (2nd ed.), ch. 11",
            "Sala-i-Martin (1996), 'The Classical Approach to Convergence Analysis', EJ",
            "Quah (1993), 'Galton's Fallacy and Tests of the Convergence Hypothesis', SJE",
        ),
    ),
    aliases=("dispersion_convergence", "sigma"),
)

register_topic(
    Explainer(
        topic="convergence_clubs",
        title="Convergence clubs (Phillips-Sul log t)",
        what=(
            "Club convergence asks whether a panel forms **one** converging group, **several** "
            "catch-up clubs, or none. Phillips & Sul (2007) model each unit as "
            "``X_it = delta_it * mu_t`` — a common trend ``mu_t`` scaled by a time-varying, "
            "unit-specific loading ``delta_it`` — and remove the common trend with the "
            "**relative transition path** ``h_it = X_it / mean_i(X_it)`` (its cross-sectional "
            "mean is 1 by construction). If the units converge, the cross-sectional variance "
            "``H_t = mean_i (h_it - 1)^2`` tends to zero, and the **log(t) regression** "
            "``log(H_1/H_t) - 2 log(log t) = a + b log t`` has a non-negative slope "
            "``b = 2*alpha``; a one-sided ``t_b > -1.65`` fails to reject convergence. When the "
            "whole panel rejects, a **data-driven clustering algorithm** sorts units by their "
            "final level, forms a core group by maximising ``t_b``, sieves in the remaining "
            "units that keep the group converging, recurses on the residual, and finally "
            "**merges** adjacent clubs that jointly converge. The series is usually smoothed "
            "first with the **Hodrick-Prescott filter** (lambda = 400 for annual data) so the "
            "test runs on the long-run trend rather than the business cycle."
        ),
        when_to_use=(
            "Use it when β- and σ-convergence give a muddy verdict — when the panel is plausibly "
            "*not* one homogeneous group but several. It is the standard tool for 'multiple "
            "equilibria' / poverty-trap questions: convergence clubs in income, labor "
            "productivity, carbon intensity, house prices or health, where a subset of units "
            "catches up to a high path while others settle on a lower one. It is data-driven "
            "(no ex-ante grouping by region or income) and robust to whether the series is "
            "trend- or difference-stationary."
        ),
        caveats=(
            "Club membership is a *descriptive* clustering of transition paths, not a causal "
            "account of why a unit lands in a given club.",
            "Results depend on the trimming fraction r (use 0.3 for moderate T, 0.2 for large "
            "T), the HP smoothing parameter, and the sorting/sieve options — report them, and "
            "check that nearby clubs are not an artefact of the merge rule.",
            "The log(t) t-statistic is asymptotic; with few periods (small T) the test has low "
            "power and clubs can be unstable, so prefer longer panels.",
            "Rejecting whole-panel convergence does not by itself prove distinct clubs exist; "
            "the algorithm can also return a single divergent group.",
        ),
        see_also=("beta_convergence", "sigma_convergence", "correlation_vs_causation"),
        references=(
            "Phillips & Sul (2007), 'Transition Modeling and Econometric Convergence Tests', "
            "Econometrica 75(6): 1771-1855",
            "Phillips & Sul (2009), 'Economic Transition and Growth', JAE 24(7): 1153-1185",
            "Schnurbus, Haupt & Meier (2016), 'Economic Transition and Growth: A Replication', "
            "JAE",
            "Du (2017), 'Econometric Convergence Test and Club Clustering Using Stata', "
            "Stata Journal 17(4)",
        ),
    ),
    aliases=("club_convergence", "log_t", "phillips_sul"),
)

register_topic(
    Explainer(
        topic="spatial_convergence",
        title="Spatial convergence (convergence with spillovers)",
        what=(
            "Classic β-convergence treats regions as isolated islands: each unit's growth "
            "depends only on its own initial level. Spatial convergence models drop that "
            "fiction. In the Ertur-Koch tradition, knowledge and capital externalities "
            "spill across borders, so a region's growth also responds to its *neighbors'* "
            "growth (ρWy) and initial conditions (WX) — exactly the spatial Durbin "
            "structure. The convergence parameter then becomes an impact: the **total** "
            "impact of initial income, (β + γ)/(1 - ρ), replaces raw β, and the implied "
            "speed of convergence is computed from it. Ignoring significant spillovers "
            "typically *understates* convergence: part of each region's catch-up arrives "
            "through its neighborhood (the headline finding of the bundled India study, "
            "where the SDM raises the implied speed from about 3 to 5 percent a year)."
        ),
        when_to_use=(
            "Whenever ESDA shows the convergence variables are spatially clustered "
            "(significant Moran's I on initial levels or growth) — run "
            "`analyze_spatial_diagnostics` on the convergence regression, and estimate "
            "`analyze_beta_convergence(model='sdm')` alongside OLS. Compare the OLS β "
            "with the SDM total impact and its direct/indirect split: the indirect share "
            "is the neighborhood contribution to catch-up."
        ),
        caveats=(
            "Impacts, speeds and half-lives are conditional on the weights matrix W; "
            "re-check under alternatives (`analyze_spatial_model_by_weights`).",
            "The direct/indirect split is model-based; it summarizes associations "
            "propagated through W, not measured flows of goods or ideas.",
            "Spatial and aspatial βs are not directly comparable — compare OLS β with "
            "the SDM *total* impact.",
        ),
        see_also=("beta_convergence", "spatial_durbin_model", "spatial_impacts"),
        references=(
            "Ertur & Koch (2007), 'Growth, technological interdependence and spatial "
            "externalities', JAE 22(6)",
            "Rey & Montouri (1999), 'US regional income convergence: a spatial "
            "econometric perspective', Regional Studies 33(2)",
        ),
    ),
    aliases=("ertur_koch",),
)
