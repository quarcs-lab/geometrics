"""The Learn app's pages — concept sandboxes with sliders, and the explainer browser."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

import streamlit as st

from geometrics.streamlit_app._common import show_sandbox

__all__ = ["page_sandboxes", "page_explainers"]


def _sliders_spatial_autocorrelation() -> None:
    import geometrics as gm

    rho = st.slider("Planted spatial dependence ρ", -0.9, 0.9, 0.6, 0.1, key="sa_rho")
    side = st.slider("Lattice side (n = side²)", 6, 16, 12, key="sa_side")
    show_sandbox(gm.learn_spatial_autocorrelation(rho=rho, side=int(side)))


def _sliders_spatial_weights() -> None:
    import geometrics as gm

    rho = st.slider("Planted ρ (under queen)", 0.0, 0.9, 0.6, 0.1, key="sw_rho")
    k = st.slider("k nearest neighbors", 2, 12, 4, key="sw_k")
    show_sandbox(gm.learn_spatial_weights(rho=rho, k=int(k)))


def _sliders_lisa() -> None:
    import geometrics as gm

    shift = st.slider(
        "Planted shift (in noise SDs)", 0.0, 4.0, 2.0, 0.5, key="li_shift"
    )
    alpha = st.select_slider("Significance α", (0.01, 0.05, 0.10), 0.05, key="li_alpha")
    show_sandbox(gm.learn_lisa_clusters(shift=shift, alpha=float(alpha)))


def _sliders_spillovers() -> None:
    import geometrics as gm

    rho = st.slider("Planted ρ", 0.0, 0.8, 0.5, 0.1, key="sp_rho")
    gamma = st.slider("Planted γ (coefficient on Wx)", -1.0, 1.0, 0.5, 0.25, key="sp_g")
    show_sandbox(gm.learn_spatial_spillovers(rho=rho, gamma=gamma, n_draws=1000))


def _sliders_omitted_lag() -> None:
    import geometrics as gm

    rho = st.slider("Planted ρ (drives the OLS bias)", 0.0, 0.9, 0.7, 0.1, key="ol_rho")
    show_sandbox(gm.learn_omitted_spatial_lag(rho=rho))


def _sliders_beta() -> None:
    import geometrics as gm

    rate = st.slider("Planted convergence rate b", 0.005, 0.05, 0.02, 0.005, key="bc_b")
    noise = st.slider("Noise SD", 0.0, 0.02, 0.005, 0.005, key="bc_noise")
    show_sandbox(gm.learn_beta_convergence(convergence_rate=rate, noise=noise))


def _sliders_sigma() -> None:
    import geometrics as gm

    rho = st.slider("Contraction ρ (per period)", 0.80, 0.99, 0.93, 0.01, key="sc_rho")
    show_sandbox(gm.learn_sigma_convergence(rho=rho))


def _sliders_clubs() -> None:
    import geometrics as gm

    gap = st.slider("Gap between the two club levels", 0.2, 2.0, 1.0, 0.2, key="cc_gap")
    spread = st.slider("Within-club initial spread", 0.1, 0.8, 0.4, 0.1, key="cc_sp")
    show_sandbox(gm.learn_convergence_clubs(levels=(10.0, 10.0 - gap), spread=spread))


def _sliders_markov() -> None:
    import geometrics as gm

    stay = st.slider("Planted persistence (diagonal)", 0.5, 0.95, 0.8, 0.05, key="mk_p")
    move = (1.0 - stay) / 2.0
    p = (
        (stay, 2 * move, 0.0),
        (move, stay, move),
        (0.0, 2 * move, stay),
    )
    show_sandbox(gm.learn_markov_chains(p=p))


def _sliders_spatial_markov() -> None:
    import geometrics as gm

    contextual = st.slider(
        "Planted context effect (0 = homogeneous)", 0.0, 0.5, 0.25, 0.05, key="sm_ctx"
    )
    show_sandbox(gm.learn_spatial_markov(contextual=contextual))


def _sliders_theil() -> None:
    import geometrics as gm

    within = st.slider("Within-group log SD", 0.1, 1.5, 0.5, 0.1, key="th_within")
    groups = st.slider("Groups", 2, 6, 3, key="th_groups")
    show_sandbox(gm.learn_theil_decomposition(within_sd=within, n_groups=int(groups)))


_SANDBOXES: dict[str, tuple[str, Callable[[], None], bool]] = {
    # label -> (description, renderer, needs_giddy)
    "Spatial autocorrelation — see ρ": (
        "A SAR field and Moran's I across planted ρ.",
        _sliders_spatial_autocorrelation,
        False,
    ),
    "Spatial weights — W is a choice": (
        "One field read through queen / rook / knn graphs.",
        _sliders_spatial_weights,
        False,
    ),
    "LISA — planted clusters, recovered": (
        "Hot and cold blocks, hit rates and false positives.",
        _sliders_lisa,
        False,
    ),
    "Spillovers — impacts vs a known truth": (
        "Direct/indirect/total against the closed form.",
        _sliders_spillovers,
        False,
    ),
    "Omitted spatial lag — why SAR exists": (
        "OLS absorbs the multiplier; SAR recovers β and ρ.",
        _sliders_omitted_lag,
        False,
    ),
    "β-convergence at a planted rate": (
        "Growth-on-initial recovers -b, speed and half-life.",
        _sliders_beta,
        False,
    ),
    "σ-convergence on a planted path": (
        "Dispersion contracts by ρ; the trend is ln ρ.",
        _sliders_sigma,
        False,
    ),
    "Convergence clubs — two planted clubs": (
        "Phillips-Sul finds the clubs and scores the match.",
        _sliders_clubs,
        False,
    ),
    "Markov chains — a planted matrix": (
        "Estimated transition probabilities vs the truth.",
        _sliders_markov,
        True,
    ),
    "Spatial Markov — mobility depends on neighbors": (
        "A planted context effect, detected by the LR test.",
        _sliders_spatial_markov,
        True,
    ),
    "Theil — a planted between/within split": (
        "The between share rises with the planted gap.",
        _sliders_theil,
        False,
    ),
}


def page_sandboxes() -> None:
    """Interactive teaching demos that simulate their own data."""
    st.header("Concept sandboxes")
    st.caption(
        "Simulated demonstrations — pick one and turn the knobs to see the concept "
        "in action. The data are generated from a known truth; only the chosen "
        "sandbox runs."
    )
    has_giddy = importlib.util.find_spec("giddy") is not None
    options = [
        name
        for name, (_, _, needs_giddy) in _SANDBOXES.items()
        if has_giddy or not needs_giddy
    ]
    choice = st.selectbox("Sandbox", options, key="sandbox_choice")
    description, renderer, _ = _SANDBOXES[choice]
    st.caption(description)
    renderer()
    if not has_giddy:
        st.caption(
            "The two Markov sandboxes need the dynamics extra: "
            '`pip install "geometrics[dynamics]"`.'
        )


def page_explainers() -> None:
    """Browse the concept explainers — the searchable topic index."""
    from geometrics import explain, list_topics

    st.header("Concept explainers")
    st.caption(
        "Plain-language explainers for every method and idea in geometrics — what it "
        "is, when to use it, and the caveats. These need no dataset."
    )
    topics = list_topics()
    query = (
        st.text_input(
            "Search topics",
            key="explainer_search",
            placeholder="e.g. moran, durbin, theil",
        )
        .strip()
        .lower()
    )
    if query:

        def _matches(name: str) -> bool:
            exp = explain(name)
            return query in f"{name} {exp.title} {exp.what}".lower()

        topics = [t for t in topics if _matches(t)]
    if not topics:
        st.info("No topics match your search.")
        return
    topic = st.selectbox("Topic", topics, key="explainer_topic")
    if topic:
        st.markdown(explain(topic).to_markdown())
