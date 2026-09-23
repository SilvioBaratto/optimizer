---
type: overview
title: "Overview: Scope of the Portfolio-Selection Monograph"
description: The purpose and arc of the monograph — portfolio selection as the transfer of wealth under uncertainty, the single-period measurement of performance by percentage and log returns, mean and variance as the reward/risk pair, the intrinsic limits of variance (semi-variance, mean absolute deviation), the mean-variance dominance criterion, and the three-step structure (measure uncertainty, define an efficiency criterion, optimize) that organizes the whole body of theory.
tags: [overview, portfolio-selection, asset-allocation, mean-variance, log-returns, dominance-criterion, semi-variance, scope]
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

# Overview: Scope of the Portfolio-Selection Monograph

This monograph develops the theory of portfolio selection from
[choice under uncertainty](./foundations/choice-under-uncertainty.md) through
[mean-variance selection](./foundations/mean-variance-selection.md) and onward to the
advanced optimization, risk, regime, and signal topics, closing with the operational
bridge to weights in
[from conditional forecasts to weights](./workflows/from-conditional-forecasts-to-weights.md).
It places portfolio selection inside the investment decision process, defines the
financial portfolio formally as a transfer of wealth under uncertainty, and introduces the
basic quantitative tools.

## The investment decision process

**Asset allocation** (portfolio management) is the set of procedures and decisions an
investor uses to build and manage a portfolio with desired characteristics from assets of
differing characteristics. It is an ordered decision process: selecting asset classes,
forming asset assumptions, strategic allocation, tactical allocation, short- and long-term
portfolio revision, and ex-post analysis. Upstream sits the evaluation of asset
characteristics, drawing on three information categories — economic/financial
fundamentals, intrinsic valuation elements, and psychological/technical factors — to
estimate three key quantities: **return**, **risk**, and **dependence**. Each operational
phase corresponds to a class of decision problems and specific quantitative tools, so the
theory can be ordered by the sequence of problems the manager faces: the mean-variance
model, the CAPM, and multifactor models precede the more advanced topics of active
management, tracking-error control, risk management, strategic/tactical/dynamic allocation,
top-down and bottom-up strategies, and performance attribution.

Allocation is classified along three criteria: by **information set** (quantitative,
qualitative, mixed), by **market posture** (conservative, moderate, aggressive), and by
**time horizon** (strategic, tactical, mixed). Its advantage over other approaches rests on
the investor, the market, and the asset classes, and produces real operational benefit only
under six critical conditions: portfolio performance (real, total, risk-adjusted return),
stability of asset relationships (structural breaks limit the use of past information),
dependence between assets (high dependence nullifies diversification and rises long-term
with globalization and short-term in panics), sensitivity of results (remedied by
Black-Litterman, robust estimation, and sensitivity analysis), rebalancing (frequency and
conditions considering estimate variability, transaction costs, and stop-loss rules), and
errors and fraud. These six factors direct the topics of the later chapters.

## Investing as the transfer of wealth under uncertainty

The theory starts from the consumer's choice problem
$\max_q u(q_1,\dots,q_N)$ subject to $\sum_i q_ip_i=M$, $q_i\ge0$, whose implicit
assumption — the economy is born, lives, and dies at instant $t$ — is unrealistic, since
many consumers transfer part of current income $M$ to future instants. A consumer faces
two correlated decisions: the consumption-saving decision and the portfolio-selection
decision. The monograph's scope is the second — how to invest across assets to transfer
wealth from the current to the future period. Because the future is largely unknown (the
riskiness is roughly of a *random walk* type), investing is risky, and financial economics
supplies the theory, methods, and tools for transferring wealth while managing uncertainty.
The two decisions cannot in general be taken independently, but many important results
follow more easily in a **single-period** setting where the consumption-saving allocation
has limited substantive impact; the monograph therefore formalizes portfolio selection
directly in a single-period economy. Investment choices are random variables: a set
$\mathbb X=\{X_1,\dots,X_N\}$ of stochastic investments, each with outcomes and
probabilities. The central object: given wealth $W$ and investments $\mathbb X$, a
**financial portfolio** is an $N$-vector $\mathbf x'=(x_1,\dots,x_N)$ where $x_i$ is the
percentage of $W$ invested in $X_i$, with $\sum_i x_i=1$ — the technical instrument that
transfers wealth from one period to the next.

## Measuring single-period performance

How single-period performance is measured depends on the price-dynamics law chosen.
Financial-mathematics convention adopts the **net percentage return**: from
$P_t=P_{t-\Delta t}(1+R_{\%,\Delta t})-D$ one gets
$R_{\%,\Delta t}=(P_t+D-P_{t-\Delta t})/P_{t-\Delta t}$, but this dynamics admits an
inadmissible negative price when the return is sufficiently negative. Mathematical-finance
convention instead adopts the **net logarithmic return**: from
$P_t=P_{t-\Delta t}e^{R_{\ln,\Delta t}}-D$ one gets
$R_{\ln,\Delta t}=\ln((P_t+D)/P_{t-\Delta t})$, which keeps $P_t>0$ for any return when
$D=0$. The two are **asymptotically equivalent**: $R_{\ln,\Delta t}=\ln(1+R_{\%,\Delta t})=R_{\%,\Delta t}-R_{\%,\Delta t}^2/2+\dots$
for $R_{\%,\Delta t}\in(-100\%,100\%)$, which holds when $\Delta t$ is small enough (daily,
weekly, monthly). Despite this, percentage returns are **not additive over time** while log
returns are: over prices $P_0,P_1,P_2$, $R_{\ln,(0,1]}+R_{\ln,(1,2]}=R_{\ln,(0,2]}$ but the
percentage sum does not equal $R_{\%,(0,2]}$. Numerically, prices 100/125/100 give
one-period percentage returns of 25.00% and −20.00% summing to 5.00% (not the 0.00%
two-period figure), while log returns 22.31% and −22.31% sum to exactly 0.00%.

## Mean and variance as reward and risk measures

Portfolio selection under uncertainty unfolds in three steps: (1) identify a tool to
*measure* the uncertainty of an investment; (2) define an efficiency criterion splitting all
investments into a mutually exclusive efficient set and inefficient set; (3) specify an
optimization approach to find, among efficient choices, the optimal one — as maximizing
return under a risk cap, minimizing risk under a return floor, optimizing a summary index
(e.g. maximizing $\text{return}-\lambda\cdot\text{risk}$), or maximizing a von
Neumann-Morgenstern utility. For step one the tool is a pair of statistical indices of the
single-period return random variable: its **mean** and **variance**, following the rule that
the investor treats expected return as desirable and return variance as undesirable
(Markowitz). Markowitz's innovation was to measure portfolio risk via the joint
(multivariate) distribution of all asset returns — marginal properties through the first
two moments, dependence through the pairwise Pearson linear correlation. The **mean**
$\mathbb E(R)=r$ measures profitability (any odd moment can serve as a reward measure), and
the **variance** $\mathrm{Var}(R)=\sigma^2$ measures risk (any even moment can serve as a
risk measure); in general mean and variance do not fully characterize a random variable.

## A limit of variance: semi-variance and mean absolute deviation

Variance is not, in general, a "good" risk measure. For
$R=\{(1\%,\tfrac14),(3\%,\tfrac15),(9\%,\tfrac14),(10\%,\tfrac15),(12\%,\tfrac1{10})\}$ the
mean is 6.3% and the variance $(4.124318\%)^2=17.01\%^2$; the deviations from the mean are
−5.3%, −3.3%, 2.7%, 3.7%, 5.7%, and variance penalizes the positive deviations (returns
*above* expectation, not risky) exactly like the negative ones by squaring all of them.
The **semi-variance** considers only negative deviations,
$\sum_i(\min\{0,x_i-\mathbb E(R)\})^2 p_i$, giving $(3.03323\%)^2=9.2005\%^2<17.01\%^2$;
Markowitz proposed it because only downside risk matters to the investor. The **mean
absolute deviation (MAD)** $\sum_i|x_i-\mathbb E(R)|p_i=3.97\%$ has $\mathrm{MAD}^2=15.76\%^2<17.01\%^2$;
its use as a portfolio risk function is due to Konno and Yamazaki. Still, neither semi-variance
nor MAD is a better risk measure than variance, which has analytical advantages other
variability measures lack — the properties a risk measure should satisfy are taken up in
[coherent risk measures](./risk-measures/coherent-risk-measures.md).

## The mean-variance dominance criterion

For step two, the efficiency criterion rests on mean and variance. $X_1$ **dominates**
$X_2$ ($X_1\succ_{MV}X_2$) if $\mathbb E(X_1)\ge\mathbb E(X_2)$ and
$\mathrm{Var}(X_1)\le\mathrm{Var}(X_2)$ with at least one strict inequality. This induces
only a **partial ordering**: with means/variances $(4,3),(2,7),(6,5)$, $X_1\succ_{MV}X_2$
and $X_3\succ_{MV}X_2$, but $X_1$ and $X_3$ are incomparable ($X_3$ has both higher mean and
higher variance). In the $(\mathrm{Var},\mathbb E)$ plane, dominance is determined in the
higher-mean/lower-variance (and lower-mean/higher-variance) quadrants and indeterminate in
the other two. Applied financially, fixing an expected-return level $\bar r$ the criterion
selects the minimum-variance choice among all with that mean; iterating over all return
levels constructs, point by point, the efficient set — anticipating the efficient frontier
developed in [mean-variance selection](./foundations/mean-variance-selection.md).

## Map of the corpus

The monograph follows the three-step structure — measure uncertainty, define an efficiency
criterion, optimize — and extends it to later theory and operational applications. The
foundational chapters run: choice under uncertainty (formalizing preferences and vNM
utility, the basis of the optimization criterion); mean-variance selection (from the
dominance criterion to Markowitz's efficient frontier); the limits of MPT and the
Black-Litterman model (sensitivity of the optimum to inputs, remedied by views and robust
estimation); coherent risk measures (properties beyond variance/semi-variance/MAD); factor
models; constraints and metaheuristics; and portfolio revision (rebalancing over time with
estimate variability, transaction costs, and intervention rules). Subsequent chapters cover
estimation, quantitative selection and construction, validation and overfitting, signal
decay and turnover, the fundamental law of active management, convex and robust and CVaR
optimization, risk budgeting, dynamic covariance, maximum diversification, drawdown risk,
market regimes and view formation, optimal execution, risk attribution, stress testing,
alternative-data signals, market states, factor timing, systemic risk, macro signals, and
the mapping from conditional forecasts to weights.
