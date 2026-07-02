"""The Analyze app's pages — convergence, spatial models, dynamics, inequality, GWR."""

from __future__ import annotations

import streamlit as st

from geometrics.streamlit_app._common import compute, show_result
from geometrics.streamlit_app._data import Active, run_cached
from geometrics.streamlit_app._pages_explore import _pick_var

__all__ = [
    "page_convergence",
    "page_clubs",
    "page_spatial_model",
    "page_by_weights",
    "page_markov",
    "page_inequality",
    "page_gwr",
]

_BETA_MODELS = ("ols", "sar", "sem", "slx", "sdm")
_SPREG_MODELS = ("ols", "lag", "error", "slx", "durbin", "durbin_error")


def page_convergence(active: Active) -> None:
    """β-convergence (OLS or spatial) and σ-convergence."""
    st.header("Convergence")
    c1, c2 = st.columns([2, 1])
    with c1:
        var = _pick_var(active, key="conv_var")
    with c2:
        model = st.selectbox(
            "β model", _BETA_MODELS, key="conv_model", help="Spatial variants add W."
        )

    st.subheader("β-convergence — do poorer regions grow faster?")
    kwargs: dict = {"var": var, "model": model}
    needs: tuple[str, ...] = ("df",)
    if model != "ols":
        needs = ("df", "gdf", "w")
        kwargs["n_draws"] = 1000
    beta = compute(
        lambda: run_cached(
            "analyze_beta_convergence",
            active.name,
            active.w_method,
            active.w_k,
            needs=needs,
            **kwargs,
        )
    )
    if beta is not None:
        show_result(beta, show_gt=True)

    st.subheader("σ-convergence — is the gap actually narrowing?")
    import geometrics as gm

    sigma = compute(lambda: gm.analyze_sigma_convergence(active.df, var))
    if sigma is not None:
        show_result(sigma, show_gt=True)


def page_clubs(active: Active) -> None:
    """Phillips-Sul convergence clubs."""
    st.header("Convergence clubs (Phillips-Sul)")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        var = _pick_var(active, key="clubs_var")
    with c2:
        hp = st.checkbox("HP-filter the paths", value=True, key="clubs_hp")
    with c3:
        trim = st.slider("log(t) trim", 0.2, 0.4, 0.3, 0.05, key="clubs_trim")
    res = compute(
        lambda: run_cached(
            "analyze_convergence_clubs",
            active.name,
            active.w_method,
            active.w_k,
            needs=("df", "gdf"),
            var=var,
            hp_filter=bool(hp),
            trim=float(trim),
        )
    )
    if res is not None:
        show_result(res, show_gt=True)
        if res.fig_map is not None:
            st.subheader("The club map")
            st.plotly_chart(res.fig_map, width="stretch")


def page_spatial_model(active: Active) -> None:
    """Estimate the spreg suite with impacts, plus the LM diagnostics."""
    st.header("Spatial econometric model")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        outcome = _pick_var(active, key="sm_outcome", label="Outcome")
    with c2:
        covariates = st.multiselect(
            "Covariates",
            [v for v in active.numeric_vars if v != outcome],
            default=[v for v in active.numeric_vars if v != outcome][:1],
            key="sm_covs",
            format_func=active.label,
        )
    with c3:
        model = st.selectbox("Model", _SPREG_MODELS, index=4, key="sm_model")
    period = st.selectbox(
        f"Period ({active.time})",
        list(active.periods),
        index=len(active.periods) - 1,
        key="sm_period",
    )
    if not covariates:
        st.info("Pick at least one covariate.")
        return

    res = compute(
        lambda: run_cached(
            "analyze_spatial_model",
            active.name,
            active.w_method,
            active.w_k,
            needs=("df", "gdf", "w"),
            outcome=outcome,
            covariates=tuple(covariates),
            model=model,
            period=period,
            n_draws=1000,
        )
    )
    if res is not None:
        show_result(res, fig=None, show_gt=True, show_df=True)
        if res.impacts is not None:
            st.subheader("LeSage-Pace impacts")
            st.dataframe(res.impacts, width="stretch", hide_index=True)

    with st.expander("🧭 Which model do the LM diagnostics prefer?"):
        diag = compute(
            lambda: run_cached(
                "analyze_spatial_diagnostics",
                active.name,
                active.w_method,
                active.w_k,
                needs=("df", "gdf", "w"),
                outcome=outcome,
                covariates=tuple(covariates),
                period=period,
            )
        )
        if diag is not None:
            st.markdown(f"**Recommendation: `{diag.recommendation}`**")
            st.text(diag.reasoning)
            st.dataframe(diag.df, width="stretch", hide_index=True)


def page_by_weights(active: Active) -> None:
    """Re-estimate the same model under alternative weights."""
    st.header("Robustness to the weights choice")
    c1, c2 = st.columns([2, 2])
    with c1:
        outcome = _pick_var(active, key="byw_outcome", label="Outcome")
    with c2:
        covariates = st.multiselect(
            "Covariates",
            [v for v in active.numeric_vars if v != outcome],
            default=[v for v in active.numeric_vars if v != outcome][:1],
            key="byw_covs",
            format_func=active.label,
        )
    period = st.selectbox(
        f"Period ({active.time})",
        list(active.periods),
        index=len(active.periods) - 1,
        key="byw_period",
    )
    if not covariates:
        st.info("Pick at least one covariate.")
        return
    res = compute(
        lambda: run_cached(
            "analyze_spatial_model_by_weights",
            active.name,
            active.w_method,
            active.w_k,
            needs=("df", "gdf"),
            outcome=outcome,
            covariates=tuple(covariates),
            period=period,
            n_draws=1000,
        )
    )
    if res is not None:
        show_result(res, show_gt=True, show_df=True)


def page_markov(active: Active) -> None:
    """Markov and spatial Markov transition analysis (dynamics extra)."""
    import geometrics as gm

    st.header("Distribution dynamics (Markov)")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        var = _pick_var(active, key="mkv_var")
    with c2:
        k = st.slider("Classes", 3, 7, 4, key="mkv_k")
    with c3:
        relative = st.checkbox("Relative to the period mean", key="mkv_rel")

    st.subheader("The transition matrix")
    mkv = compute(
        lambda: gm.analyze_markov_transitions(
            active.df, var, k=int(k), relative=relative
        )
    )
    if mkv is not None:
        show_result(mkv, show_gt=True)

    st.subheader("Conditioned on the neighbors (spatial Markov)")
    spm = compute(
        lambda: run_cached(
            "analyze_spatial_markov",
            active.name,
            active.w_method,
            active.w_k,
            needs=("df", "gdf", "w"),
            var=var,
            k=int(k),
            m=int(k),
        )
    )
    if spm is not None:
        show_result(spm, show_gt=True)


def page_inequality(active: Active) -> None:
    """Inequality trends (incl. spatial Gini) and the Theil decomposition."""
    import geometrics as gm

    st.header("Regional inequality")
    c1, c2 = st.columns([2, 1])
    with c1:
        var = _pick_var(active, key="ineq_var")
    with c2:
        permutations = st.selectbox(
            "Spatial-Gini permutations", (0, 99), key="ineq_perm"
        )

    res = compute(
        lambda: run_cached(
            "analyze_inequality_over_time",
            active.name,
            active.w_method,
            active.w_k,
            needs=("df", "gdf", "w"),
            var=var,
            permutations=int(permutations),
        )
    )
    if res is not None:
        show_result(res, show_gt=True)

    if active.factor_vars:
        st.subheader("Between or within? (Theil decomposition)")
        group = st.selectbox(
            "Grouping",
            list(active.factor_vars),
            key="ineq_group",
            format_func=active.label,
        )
        theil = compute(lambda: gm.analyze_theil_decomposition(active.df, var, group))
        if theil is not None:
            show_result(theil, show_gt=True)


def page_gwr(active: Active) -> None:
    """Geographically weighted regression, behind an explicit run button."""
    st.header("Local models — GWR")
    st.caption(
        "The bandwidth search re-estimates the model many times — press Run when the "
        "selections are ready (≈ a minute on the 520-district panel; cached after)."
    )
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        outcome = _pick_var(active, key="gwr_outcome", label="Outcome")
    with c2:
        covariates = st.multiselect(
            "Covariates",
            [v for v in active.numeric_vars if v != outcome],
            default=[v for v in active.numeric_vars if v != outcome][:1],
            key="gwr_covs",
            format_func=active.label,
        )
    with c3:
        kernel = st.selectbox("Kernel", ("bisquare", "gaussian"), key="gwr_kernel")
    period = st.selectbox(
        f"Period ({active.time})",
        list(active.periods),
        index=len(active.periods) - 1,
        key="gwr_period",
    )
    if not covariates:
        st.info("Pick at least one covariate.")
        return
    if not st.button("Run GWR", type="primary", key="gwr_run"):
        st.info("Multiscale GWR (`analyze_mgwr`) is heavier still — run it locally.")
        return
    res = compute(
        lambda: run_cached(
            "analyze_gwr",
            active.name,
            active.w_method,
            active.w_k,
            needs=("df", "gdf"),
            outcome=outcome,
            covariates=tuple(covariates),
            period=period,
            kernel=kernel,
        )
    )
    if res is not None:
        for term, fig in res.figs.items():
            st.subheader(f"Local coefficient: {active.label(term)}")
            st.plotly_chart(fig, width="stretch")
        show_result(res, fig=None, show_gt=True)
