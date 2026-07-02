"""Explainers for correlation analysis (Pearson, Spearman, correlation vs causation)."""

from __future__ import annotations

from geometrics.pedagogy._registry import Explainer, register_topic

register_topic(
    Explainer(
        topic="pearson",
        title="Pearson correlation",
        what=(
            "Pearson's r measures the strength of the *linear* relationship between two "
            "numeric variables, ranging from -1 (perfect negative line) through 0 (no linear "
            "relationship) to +1 (perfect positive line)."
        ),
        when_to_use=(
            "For roughly linear relationships between continuous variables with no extreme "
            "outliers — a quick first read of how two variables move together."
        ),
        caveats=(
            "Only captures *linear* association: a strong curved (e.g. U-shaped) relationship "
            "can have r near zero.",
            "Highly sensitive to outliers, which can manufacture or hide a correlation.",
            "A correlation is not a causal effect.",
        ),
        see_also=("spearman", "correlation_vs_causation"),
        references=("Wooldridge, Introductory Econometrics, ch. 2",),
    )
)

register_topic(
    Explainer(
        topic="spearman",
        title="Spearman rank correlation",
        what=(
            "Spearman's rho is Pearson's correlation computed on the *ranks* of the data. It "
            "measures whether two variables move together monotonically, regardless of whether "
            "the relationship is a straight line."
        ),
        when_to_use=(
            "When the relationship is monotonic but not linear, when variables are skewed, or "
            "when outliers would distort Pearson's r. Comparing the two is itself informative."
        ),
        caveats=(
            "Captures monotonic (always-up or always-down) association only — a non-monotonic "
            "relationship can still show a small rho.",
            "A large gap between Pearson and Spearman is a clue: non-linearity or outliers.",
        ),
        see_also=("pearson", "correlation_vs_causation"),
        references=("Wooldridge, Introductory Econometrics, ch. 2",),
    )
)

register_topic(
    Explainer(
        topic="correlation_vs_causation",
        title="Correlation is not causation",
        what=(
            "Two variables can move together because one drives the other, because a third "
            "factor drives both (confounding), because of reverse causality, or by chance. A "
            "correlation alone cannot tell these apart."
        ),
        when_to_use=(
            "Always keep this in mind when reading a correlation or a regression coefficient: "
            "describe what you find as an *association* unless a research design supports a "
            "causal claim."
        ),
        caveats=(
            "A credible causal interpretation needs a design: a randomized experiment, an "
            "instrument, a difference-in-differences setup, or a regression-discontinuity.",
            "Confounding, reverse causality and selection are the usual reasons an association "
            "is not the causal effect.",
        ),
        see_also=("pearson", "spearman", "ols"),
        references=("Angrist & Pischke, Mastering 'Metrics, ch. 1",),
    )
)
