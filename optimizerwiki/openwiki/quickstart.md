---
type: guide
title: "Quickstart & Task-Routing Map"
description: Entry point for the portfolio-selection theory wiki — what the monograph covers, the three-step logical structure it follows (measure uncertainty, define an efficiency criterion, optimize), how the sections fit together, and which page to open for a given portfolio-construction question, from foundations through estimation, optimization, risk, regimes, signals, validation, and execution workflows.
tags: [quickstart, index, task-routing, asset-allocation, portfolio-selection, mean-variance, monograph-map]
sources:
  - id: openwiki-source-6c9d6efc01319baa1619bff0
    resource: repo://docs/00_introduction_and_scope.md
  - id: openwiki-source-37acb20b2ff9ad3b3748f1ff
    resource: repo://docs/theory.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Quickstart & Task-Routing Map

This is the entry point for the theory wiki. The corpus is a monograph on **portfolio-selection theory**,
running from choice under uncertainty to the revision of the optimal portfolio. It places portfolio
selection inside the investment decision process, formally defines a financial portfolio as a transfer of
wealth under uncertainty, and builds up the quantitative tools — single-period return measurement, the
mean/variance pair, and the mean-variance dominance criterion — before extending to advanced estimation,
optimization, risk, regime, signal, validation, and execution topics. For a broader narrative see the
[overview](./overview.md); for the theoretical starting point see
[mean-variance selection](./foundations/mean-variance-selection.md).

## The three-step structure of the whole monograph

Portfolio selection under uncertainty is articulated in three steps, and the whole monograph follows
them: **(1)** identify a tool to *measure* the uncertainty of an investment choice; **(2)** define an
*efficiency criterion* that splits all possible choices into two mutually exclusive sets — efficient and
inefficient; **(3)** specify an appropriate *optimization* approach to find, among the efficient choices,
the optimal one. Optimization can take several forms: maximize return subject to a risk ceiling, minimize
risk subject to a return floor, optimize a summary index such as $\text{return}-\lambda\cdot\text{risk}$
(with $\lambda>0$ a risk-aversion measure), or maximize a von Neumann-Morgenstern utility.

The starting object is the consumer's choice problem, extended to wealth transfer across time: a portfolio
is the technical instrument that transfers wealth from one period to the next. Formally, given wealth $W$
and $N$ investment choices, a portfolio is an $N$-vector $\mathbf x'=(x_1,\dots,x_N)$ with $x_i$ the
fraction of $W$ invested in choice $i$ and $\sum_i x_i=1$. The monograph works in a single-period economy,
where the consumption-savings decision has limited impact and portfolio selection can be formalized
directly.

## Foundations — measuring, then choosing

- **[Choice under uncertainty](./foundations/choice-under-uncertainty.md)** (ch. 1) — formalizes investor
  preferences under uncertainty and von Neumann-Morgenstern utility, the basis of the optimization
  criterion. Start here for *why* a rational investor prefers more to less and is risk-averse.
- **[Mean-variance selection](./foundations/mean-variance-selection.md)** (ch. 2) — from the mean-variance
  dominance criterion to Markowitz's efficient frontier and its construction. The theoretical core.
- **[Black-Litterman](./foundations/black-litterman.md)** (ch. 3) — the sensitivity of the optimal
  composition to input values, and the remedies based on views and robust estimation. Open this when the
  optimizer's weights look unstable or extreme.

Single-period performance is measured either by the **net percentage return**
$R_{\%}=(P_t+D-P_{t-\Delta t})/P_{t-\Delta t}$ or the **net log return**
$R_{\ln}=\ln((P_t+D)/P_{t-\Delta t})$; the two are asymptotically equivalent for $R_{\%}\in(-100\%,100\%)$,
but only the log return is additive over time. The mean of the return measures profitability and its
variance measures risk.

## Risk measures — beyond variance

- **[Coherent risk measures](./risk-measures/coherent-risk-measures.md)** (ch. 4) — the properties a good
  risk measure should satisfy, motivated by the limits of variance (which penalizes upside deviations
  exactly like downside), semi-variance and mean-absolute deviation.
- **[Drawdown risk measures](./risk-measures/drawdown-risk-measures.md)** (ch. 20) — path-dependent risk
  from peak-to-trough losses.

## Estimation — inputs and their errors

- **[Estimation error, outliers and shrinkage](./estimation/estimation-error-and-shrinkage.md)** (ch. 8) —
  why sample moments are noisy and how shrinkage tames them.
- **[Dynamic covariance estimation](./estimation/dynamic-covariance.md)** (ch. 18) — EWMA and multivariate
  GARCH for time-varying second moments.

## Factor models

- **[Factor models](./factor-models/factor-models.md)** (ch. 5) — describing returns through common
  factors.
- **[Factor timing and rotation](./factor-models/factor-timing-and-rotation.md)** (ch. 27) — varying factor
  exposures over time.

## Optimization — from convex theory to specialized objectives

- **[Convex optimization](./optimization/convex-optimization.md)** (ch. 14) — cones, duality and KKT: the
  machinery underlying every constrained portfolio problem.
- **[Constraints and metaheuristics](./optimization/constraints-and-metaheuristics.md)** (ch. 6) — selection
  under operational constraints and the corresponding algorithms.
- **[Robust optimization](./optimization/robust-optimization.md)** (ch. 15) — uncertainty sets and
  worst-case objectives.
- **[CVaR optimization](./optimization/cvar-optimization.md)** (ch. 16) — the Rockafellar-Uryasev linear
  formulation.
- **[Risk budgeting and risk parity](./optimization/risk-budgeting-and-risk-parity.md)** (ch. 17) —
  allocating risk rather than capital.
- **[Maximum diversification](./optimization/maximum-diversification.md)** (ch. 19) — the diversification
  ratio objective.

## Signals — from fundamentals and data to alpha

- **[Fundamental stock selection](./signals/fundamental-stock-selection.md)** (ch. 9)
- **[Signal decay, horizon and turnover](./signals/signal-decay-horizon-turnover.md)** (ch. 11) — how a
  signal's decay and horizon generate turnover.
- **[From signal to alpha: the fundamental law](./signals/signal-to-alpha-fundamental-law.md)** (ch. 12) —
  information coefficient, breadth and the fundamental law of active management.
- **[Quantitative selection and construction](./signals/quantitative-selection-and-construction.md)** (ch. 13)
- **[Alternative data and sentiment](./signals/alternative-data-and-sentiment.md)** (ch. 25)

## Regimes — conditioning on the state of the world

- **[Market regimes and views](./regimes/market-regimes-and-views.md)** (ch. 21)
- **[Market states and the cross-section](./regimes/market-states-and-the-cross-section.md)** (ch. 26)
- **[Turbulence and systemic risk](./regimes/turbulence-and-systemic-risk.md)** (ch. 28) — turbulence and
  the absorption ratio.
- **[Macro signals and risk premia](./regimes/macro-signals-and-risk-premia.md)** (ch. 29)

## Risk management

- **[Risk attribution, budgets and limits](./risk-management/risk-attribution-budget-limits.md)** (ch. 23)
- **[Stress testing and scenarios](./risk-management/stress-testing-and-scenarios.md)** (ch. 24)

## Validation

- **[Validation, data snooping and overfitting](./validation/validation-data-snooping-overfitting.md)**
  (ch. 10) — read before trusting any backtested strategy.

## Workflows — turning theory into traded portfolios

- **[From conditional forecasts to weights](./workflows/from-conditional-forecasts-to-weights.md)** (ch. 30)
  — the plug-in, decision-theoretic and direct routes from a conditional return forecast to portfolio
  weights.
- **[Portfolio revision](./workflows/portfolio-revision.md)** (ch. 7) — when and whether to rebalance,
  given estimate variability, transaction costs and capital-gains taxes.
- **[Optimal execution and market impact](./workflows/optimal-execution-and-market-impact.md)** (ch. 22) —
  executing the resulting turnover at minimum cost.

## How to route a question

- *"Which portfolio is optimal at one instant?"* → [mean-variance selection](./foundations/mean-variance-selection.md).
- *"My optimizer gives extreme, unstable weights."* → [Black-Litterman](./foundations/black-litterman.md)
  and [estimation error and shrinkage](./estimation/estimation-error-and-shrinkage.md).
- *"Is variance the right risk measure?"* → [coherent risk measures](./risk-measures/coherent-risk-measures.md).
- *"How do I know my backtest isn't overfit?"* → [validation, data snooping and overfitting](./validation/validation-data-snooping-overfitting.md).
- *"How often should I rebalance, and is it worth it?"* → [portfolio revision](./workflows/portfolio-revision.md).
- *"How do I trade the target without moving the price against me?"* → [optimal execution and market impact](./workflows/optimal-execution-and-market-impact.md).
- *"How do I turn a macro forecast into weights?"* → [from conditional forecasts to weights](./workflows/from-conditional-forecasts-to-weights.md).
