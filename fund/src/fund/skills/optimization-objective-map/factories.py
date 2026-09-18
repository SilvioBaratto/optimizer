"""Illustrative optimizer factory skeletons for `optimization-objective-map`.

READ-ONLY reference for the allocator agent: it shows the exact skfolio/optimizer
call shape per objective so the LLM emits a matching *config*, never weights. This
file is a text resource — its directory name has a hyphen, so it is NOT importable;
the real optimizer wiring lives behind the `optimize_portfolio` @tool (Phase 3).

Load-bearing rule: the LLM chooses the objective / measure / knobs; skfolio computes
the weights. Feed LINEAR returns (`prices_to_returns`); `shuffle=False` in any CV.
"""

# --- min risk (variance / CVaR / CDaR) -> QP / LP ---------------------------
# from skfolio import RiskMeasure
# from skfolio.optimization import MeanRisk, ObjectiveFunction
# MeanRisk(
#     objective_function=ObjectiveFunction.MINIMIZE_RISK,
#     risk_measure=RiskMeasure.CVAR,        # or VARIANCE / CDAR
#     cvar_beta=0.95,                        # from ConstraintSet.beta
#     min_weights=0.0, max_weights=cap,      # long-only + cardinality cap
# )

# --- max utility (risk aversion A) -> QP ------------------------------------
# MeanRisk(
#     objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
#     risk_aversion=a_gamma,                 # from ConstraintSet.a_gamma
#     risk_measure=RiskMeasure.VARIANCE,
# )

# --- max ratio (Sharpe / Sortino / CVaR-ratio) -> SOCP ----------------------
# MeanRisk(
#     objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
#     risk_measure=RiskMeasure.VARIANCE,
# )

# --- risk budgeting / ERC -> log-barrier ------------------------------------
# from skfolio.optimization import RiskBudgeting
# RiskBudgeting(risk_measure=RiskMeasure.VARIANCE)   # equal risk contribution default

# --- robust (prudent profiles): uncertainty set on mu -----------------------
# from skfolio.uncertainty_set import EmpiricalMuUncertaintySet
# MeanRisk(
#     # 1.0 uncertainty set: radius / geometry / norm
#     mu_uncertainty_set_estimator=EmpiricalMuUncertaintySet(),
# )
