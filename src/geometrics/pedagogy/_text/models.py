"""Explainers for spatial econometric models and local regression.

Covers the specification family (SAR, SEM, SLX, SDM), the LM specification tests, the
LeSage-Pace impact decomposition, weights robustness, and geographically weighted
regression (GWR / MGWR).
"""

from __future__ import annotations

from geometrics.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="spatial_lag_model",
        title="Spatial lag model (SAR)",
        what=(
            "The spatial-lag (spatial autoregressive) model adds the *outcome's own "
            "spatial lag* as a regressor: y = ρWy + Xβ + ε. A positive ρ means each "
            "unit's outcome moves with its neighbors' outcomes, generating global "
            "feedback: a shock to one unit ripples to its neighbors, their neighbors, and "
            "back. Because y appears on both sides, OLS is inconsistent — maximum "
            "likelihood (or GMM/IV) is used, and coefficients can no longer be read as "
            "marginal associations: the impact decomposition (direct/indirect/total) is "
            "the honest summary."
        ),
        when_to_use=(
            "When theory says outcomes themselves interact across space — growth "
            "spillovers, strategic interaction among governments, diffusion. If the LM "
            "diagnostics point to the lag structure, start here; if lagged covariates "
            "also matter, the Durbin model nests both."
        ),
        caveats=(
            "ρ is identified by functional form and W; a mis-specified W biases it.",
            "Never read raw β as an effect on y — use the impact decomposition.",
            "Global feedback assumes simultaneity; for temporal diffusion a dynamic "
            "panel model may fit the story better.",
        ),
        see_also=("spatial_durbin_model", "spatial_impacts", "lm_diagnostics"),
        references=(
            "Anselin (1988), Spatial Econometrics, ch. 6",
            "LeSage & Pace (2009), Introduction to Spatial Econometrics, ch. 2",
        ),
    ),
    aliases=("sar", "spatial_lag_regression"),
)

register_topic(
    Explainer(
        topic="spatial_error_model",
        title="Spatial error model (SEM)",
        what=(
            "The spatial-error model keeps the mean equation aspatial but lets the "
            "*disturbances* be spatially autoregressive: y = Xβ + u with u = λWu + ε. "
            "Spatial correlation is treated as a nuisance — omitted spatially-smooth "
            "factors, common shocks, boundary mismatch — that invalidates OLS standard "
            "errors without changing what β means. Estimation is again ML or GMM; β "
            "retains its ordinary marginal-association reading and there are no "
            "spillover impacts by construction."
        ),
        when_to_use=(
            "When diagnostics show residual spatial correlation but you have no "
            "substantive story of outcome-on-outcome interaction — the goal is efficient "
            "estimates and honest inference, not measuring spillovers."
        ),
        caveats=(
            "SEM cannot *measure* spillovers; if indirect effects are the question, use "
            "SAR/SDM.",
            "A significant λ often proxies for missing spatially-structured covariates — "
            "adding them can be better than modelling the error.",
        ),
        see_also=("spatial_lag_model", "lm_diagnostics"),
        references=("Anselin (1988), Spatial Econometrics, ch. 6",),
    ),
    aliases=("sem", "spatial_error_regression"),
)

register_topic(
    Explainer(
        topic="slx_model",
        title="SLX model (spatially lagged X)",
        what=(
            "The SLX model adds the *covariates'* spatial lags: y = Xβ + WXγ + ε. Each γ "
            "measures how a unit's outcome co-moves with its neighbors' characteristics — "
            "local spillovers with no global feedback loop. It estimates by OLS, keeps "
            "the familiar coefficient reading (β = direct, γ = indirect), and is the "
            "simplest specification that produces spillover estimates."
        ),
        when_to_use=(
            "As the first spillover model when you want interpretable, OLS-estimable "
            "local spillovers — e.g. does my growth associate with my neighbors' initial "
            "income, infrastructure, human capital? It is also the building block the "
            "Durbin model adds to SAR."
        ),
        caveats=(
            "Lagged covariates are often highly collinear with the covariates "
            "themselves; geometrics drops collinear lags automatically (and notes it).",
            "Spillovers are local only (one ring of neighbors under contiguity); global "
            "diffusion needs ρWy.",
        ),
        see_also=("spatial_durbin_model", "spatial_impacts"),
        references=(
            "Halleck Vega & Elhorst (2015), 'The SLX Model', J. of Regional Science",
        ),
    ),
    aliases=("slx",),
)

register_topic(
    Explainer(
        topic="spatial_durbin_model",
        title="Spatial Durbin model (SDM)",
        what=(
            "The Durbin model combines both channels: y = ρWy + Xβ + WXγ + ε — outcomes "
            "respond to neighbors' outcomes (global feedback through ρ) *and* to "
            "neighbors' characteristics (local spillovers through γ). It nests SAR "
            "(γ = 0), SLX (ρ = 0) and, under a parameter restriction, SEM — which makes "
            "it the natural robust choice when LM diagnostics are ambiguous. Estimated "
            "by ML as a lag model with lagged covariates; interpretation runs entirely "
            "through the LeSage-Pace impact decomposition, where the total impact of a "
            "covariate is (β + γ)/(1 - ρ)."
        ),
        when_to_use=(
            "The workhorse for regional convergence with spillovers (the specification "
            "of the bundled India study): it lets initial income matter directly and "
            "through neighbors, and lets growth diffuse. Prefer it when both LM robust "
            "tests fire, or when omitted spatially-correlated covariates are plausible "
            "(the Durbin terms absorb them better than SAR alone)."
        ),
        caveats=(
            "Impacts, not coefficients, are the reportable quantities.",
            "The WX block can be collinear; geometrics's full-rank mask drops redundant "
            "lags exactly as Stata does, so estimates match across tools.",
            "ρ and γ separate only through W's structure; alternative-W robustness "
            "checks are essential.",
        ),
        see_also=(
            "spatial_impacts",
            "spatial_lag_model",
            "slx_model",
            "weights_robustness",
        ),
        references=(
            "LeSage & Pace (2009), Introduction to Spatial Econometrics, ch. 2-3",
            "Elhorst (2014), Spatial Econometrics: From Cross-Sectional Data to Panels",
        ),
    ),
    aliases=("sdm", "durbin"),
)

register_topic(
    Explainer(
        topic="lm_diagnostics",
        title="LM specification tests (Anselin-Florax)",
        what=(
            "The Lagrange-multiplier tests examine OLS residuals for two rival spatial "
            "structures: LM-lag (missing ρWy) and LM-error (missing λWu). Because each "
            "test also reacts to the *other* misspecification, their **robust** versions "
            "correct for that cross-contamination. The Anselin-Florax decision rule: if "
            "neither LM test is significant keep OLS; if exactly one robust test fires, "
            "estimate that model; if both fire, take the larger robust statistic — and "
            "consider the Durbin model, which nests both structures."
        ),
        when_to_use=(
            "Right after a baseline OLS on spatial data, before committing to a spatial "
            "specification. geometrics's `analyze_spatial_diagnostics` runs the battery "
            "and returns the rule's recommendation with its reasoning."
        ),
        caveats=(
            "The tests assume the mean equation is otherwise well specified; omitted "
            "aspatial covariates can masquerade as spatial structure.",
            "All statistics are conditional on W.",
            "With very small n the asymptotic χ² reference is unreliable.",
        ),
        see_also=("spatial_lag_model", "spatial_error_model", "spatial_durbin_model"),
        references=(
            "Anselin, Bera, Florax & Yoon (1996), 'Simple diagnostic tests for spatial "
            "dependence', Regional Science and Urban Economics",
            "Florax, Folmer & Rey (2003), 'Specification searches in spatial "
            "econometrics', Regional Science and Urban Economics",
        ),
    ),
    aliases=("lm_tests", "specification_tests"),
)

register_topic(
    Explainer(
        topic="spatial_impacts",
        title="Direct, indirect, and total impacts",
        what=(
            "In models with ρWy, a change in one unit's covariate propagates to its "
            "neighbors and feeds back, so no single coefficient describes the "
            "association. The LeSage-Pace decomposition summarizes the full matrix of "
            "cross-unit derivatives: the **direct** impact is the average own-unit "
            "association (including feedback through neighbors and back); the "
            "**indirect** impact is the average spillover to all other units; their sum "
            "is the **total** impact, which for the Durbin model equals (β + γ)/(1 - ρ). "
            "geometrics computes these from the estimated parameters and simulates their "
            "standard errors by Monte-Carlo draws from the ML covariance matrix — the "
            "same quantities Stata's estat impact reports."
        ),
        when_to_use=(
            "Always, when reporting SAR/SDM results: read and compare direct vs indirect "
            "impacts, not raw coefficients. In convergence work the *total* impact of "
            "initial income is what maps to the speed of convergence."
        ),
        caveats=(
            "Impacts are averages over all units; unit-specific impacts vary with map "
            "position.",
            "Monte-Carlo standard errors depend on the number of draws — geometrics "
            "records it on the result.",
            "The word 'impact' is the literature's term of art; in geometrics's reading "
            "these remain associations, not causal quantities.",
        ),
        see_also=(
            "spatial_durbin_model",
            "spatial_lag_model",
            "correlation_vs_causation",
        ),
        references=(
            "LeSage & Pace (2009), Introduction to Spatial Econometrics, ch. 2",
        ),
    ),
    aliases=("direct_indirect_total", "impacts"),
)

register_topic(
    Explainer(
        topic="weights_robustness",
        title="Robustness to the weights choice",
        what=(
            "Every spatial estimate is conditional on W, and W is chosen, not observed. "
            "The standard check re-estimates the preferred model under a battery of "
            "alternative weights — different k for KNN, queen vs rook contiguity, "
            "inverse-distance decay — and compares the impact estimates. Conclusions "
            "that survive across the battery are findings; conclusions that appear under "
            "exactly one W are artifacts of that W."
        ),
        when_to_use=(
            "Before reporting any headline spatial result. "
            "`analyze_spatial_model_by_weights` runs the battery and draws the "
            "dot-whisker comparison against your baseline specification."
        ),
        caveats=(
            "The battery spans reasonable codifications, not all possible ones; W "
            "choices should still be theory-led.",
            "Some variation across W is expected — look for sign and significance "
            "stability, not identical point estimates.",
        ),
        see_also=("spatial_weights", "spatial_impacts"),
        references=(
            "LeSage & Pace (2014), 'The Biggest Myth in Spatial Econometrics', "
            "Econometrics",
        ),
    ),
)

register_topic(
    Explainer(
        topic="gwr",
        title="Geographically weighted regression (GWR)",
        what=(
            "GWR drops the assumption that one coefficient vector describes the whole "
            "map: it fits a weighted regression *at every location*, borrowing nearby "
            "observations with a distance-decay kernel, and returns a coefficient "
            "**surface** per covariate. The **bandwidth** — how far borrowing reaches — "
            "is chosen by criterion search (AICc by default) and is the model's key "
            "parameter: small bandwidths mean highly local relationships, bandwidths "
            "near n mean the relationship is effectively global. Because thousands of "
            "local tests run at once, significance uses a multiple-testing-corrected "
            "threshold; geometrics greys out entities below it on the coefficient maps."
        ),
        when_to_use=(
            "When you suspect the strength (or sign) of an association varies across the "
            "map — convergence that is faster in one macro-region, an amenity valued "
            "only in cities. Use it as an exploratory lens on heterogeneity, alongside — "
            "not instead of — a global model."
        ),
        caveats=(
            "Local coefficients are strongly smoothed by the kernel; sharp boundaries "
            "blur.",
            "Collinearity among covariates is amplified locally.",
            "GWR describes spatial heterogeneity in associations; it does not test a "
            "spillover mechanism (that is the spatial-lag family's job).",
        ),
        see_also=("mgwr", "spatial_durbin_model"),
        references=(
            "Fotheringham, Brunsdon & Charlton (2002), Geographically Weighted "
            "Regression",
        ),
    ),
    aliases=("geographically_weighted_regression",),
)

register_topic(
    Explainer(
        topic="mgwr",
        title="Multiscale GWR (MGWR)",
        what=(
            "Classic GWR forces every covariate to operate at one shared spatial scale — "
            "the single bandwidth. MGWR relaxes that: each covariate gets its **own** "
            "bandwidth, estimated by backfitting, so slowly-varying processes (climate, "
            "institutions) and hyper-local ones (neighborhood effects) coexist in one "
            "model. The per-covariate bandwidths are themselves interpretable — they "
            "estimate the *scale* at which each association operates. MGWR requires all "
            "variables standardized, so coefficients are in standard-deviation units."
        ),
        when_to_use=(
            "When GWR's single bandwidth is a straitjacket — typically whenever "
            "covariates plausibly differ in operating scale. Compare each MGWR "
            "bandwidth with n: bandwidths near n say 'this association is global'."
        ),
        caveats=(
            "Backfitting is iterative and slower than GWR; large maps take time.",
            "Coefficients are on the standardized scale; translate before reporting "
            "magnitudes.",
            "Bandwidth uncertainty is real; treat scale estimates as indicative.",
        ),
        see_also=("gwr",),
        references=(
            "Fotheringham, Yang & Kang (2017), 'Multiscale Geographically Weighted "
            "Regression', Annals of the AAG",
            "Oshan et al. (2019), 'mgwr: A Python Implementation', IJGI",
        ),
    ),
    aliases=("multiscale_gwr",),
)
