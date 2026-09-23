---
type: concept
title: "Dynamic Covariance Estimation: EWMA and Multivariate GARCH"
description: Time-varying conditional covariance for portfolio risk — the RiskMetrics EWMA and Engle's Dynamic Conditional Correlation (DCC) multivariate GARCH, their update rules, two-step estimation, and their role as an alternative input to the static sample covariance in optimization.
tags: [covariance, ewma, garch, dcc, riskmetrics, volatility, conditional-correlation, portfolio-risk]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
sources:
  - id: openwiki-source-815e4be3ff588fc86e976460
    resource: repo://docs/18_dynamic_covariance_estimation_ewma_and_multivariate_garch.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
---

# Dynamic Covariance Estimation: EWMA and Multivariate GARCH

The covariance matrix of asset returns is **not constant over time**. This chapter
derives two families of *conditional* covariance estimators that update with the
most recent information — the exponentially weighted moving average (EWMA) of
RiskMetrics and the Dynamic Conditional Correlation (DCC) model of Engle — and
shows that a time-varying covariance is a direct, alternative input to the static
sample covariance used in [mean-variance selection](../foundations/mean-variance-selection.md)
and in [shrinkage/regularization methods](./estimation-error-and-shrinkage.md).

## Why the covariance is time-varying

Correlations and volatilities are critical inputs for hedging, asset allocation,
and risk measurement, and allocation/risk tasks typically require a large number of
them at once: building a constrained optimal portfolio requires a forecast of the
return covariance matrix, and computing today's portfolio standard deviation
requires the covariance of all its constituents. The empirical evidence motivating
a dynamic treatment is **conditional heteroskedasticity** — return variance depends
on time — together with **volatility clustering**: volatility reacts to market
shocks (it rises after a large-magnitude return) and then decays gradually, so
high- and low-volatility episodes cluster. Correlations are equally non-stationary,
with pronounced structural breaks (e.g. the breakdown of the European currency
links in August 1992). These regularities make a static sample covariance
inadequate and call for conditional estimators.

## RiskMetrics EWMA

EWMA weights recent observations more heavily via a decay factor $\lambda$ with
$0<\lambda<1$. Assuming infinite data and zero mean, the one-step variance forecast
is the recursion

$$\sigma^2_{t+1|t}=\lambda\,\sigma^2_{t|t-1}+(1-\lambda)\,r^2_{t},$$

an exponentially weighted sum of past squared returns. Relative to a simple moving
average, volatility reacts faster to shocks and then decays exponentially rather
than dropping abruptly when the outlier leaves a fixed window. The **covariance**
is built the same way, replacing the square of one series with the product of two:

$$\sigma^2_{12,t+1|t}=\lambda\,\sigma^2_{12,t|t-1}+(1-\lambda)\,r_{1,t}\,r_{2,t},
\qquad \rho_{12,t+1|t}=\frac{\sigma^2_{12,t+1|t}}{\sigma_{1,t+1|t}\,\sigma_{2,t+1|t}}.$$

### Choosing the decay factor

The decay factor governs both the weights and the *effective* number of
observations, $K=\ln\Upsilon_L/\ln\lambda$ for a tolerance $\Upsilon_L$ (e.g.
$\lambda=0.97$ at 1% tolerance uses ≈151 days). The optimal $\lambda^\*$ minimizes
the RMSE of variance forecasts. Crucially, in the multivariate case the individual
decay factors are *not* independent — a matrix built from element-specific factors
is subject to substantial distortion — so RiskMetrics applies a **single** decay
factor to the whole covariance matrix, obtained as an accuracy-weighted average of
individual optima over 450+ series: **0.94 for daily** data and **0.97 for
monthly** data.

### Multi-step forecasts and the IGARCH link

Under an IID-normal increment model the EWMA multi-step variance obeys the
square-root-of-time rule $\sigma_{t+T|t}=\sqrt{T}\,\sigma_{t+1|t}$, and the forecast
correlation is horizon-invariant. This implicitly assumes the variance process is
**non-stationary**: EWMA is essentially an **IGARCH (Integrated GARCH) without
intercept**, motivated bottom-up (simple to implement) rather than estimated by
maximum likelihood.

## From univariate GARCH to the multivariate problem

Univariate GARCH generalizes EWMA:
$h_{it}=\omega_i+\sum_p\alpha_{ip}r^2_{i,t-p}+\sum_q\beta_{iq}h_{i,t-q}$, with
non-negativity and stationarity $\sum_p\alpha_{ip}+\sum_q\beta_{iq}<1$. The direct
multivariate extension (the `vec` model, and its positive-definite BEKK
restriction) suffers a **parameter explosion** — BEKK needs $O(k^4)$ parameters in
general, $O(k^2)$ in diagonal/scalar form — so few studies handle more than a
handful of assets. *Variance targeting* fixes the long-run covariance to the sample
covariance $S=\tfrac1T\sum_t r_t r_t'$. DCC is the response to this scale problem.

## The Dynamic Conditional Correlation (DCC) model

DCC decomposes the conditional covariance into standard deviations and correlations:

$$H_t=D_t\,R_t\,D_t,\qquad D_t=\mathrm{diag}\{\sqrt{h_{i,t}}\},$$

where $D_t$ collects univariate-GARCH conditional standard deviations and $R_t$ is
the conditional correlation matrix. With standardized residuals
$\varepsilon_t=D_t^{-1}r_t$, one has $E_{t-1}(\varepsilon_t\varepsilon_t')=R_t$, so
$R_t$ is both the return correlation and the covariance of the standardized
residuals. Whereas Bollerslev's constant-correlation model fixes $R_t=R$, DCC lets
$R_t$ vary through an auxiliary matrix $Q_t$, updated either by an integrated
exponential smoother or by a mean-reverting GARCH(1,1) form:

$$Q_t=S(1-\alpha-\beta)+\alpha\,(\varepsilon_{t-1}\varepsilon_{t-1}')+\beta\,Q_{t-1},$$

mean-reverting while $\alpha+\beta<1$ and collapsing to the integrated smoother when
the sum equals one. Since $q_{ij,t}$ is not itself a correlation, $Q_t$ is
**rescaled** to a valid correlation matrix
$R_t=Q_t^{*-1}Q_tQ_t^{*-1}$ with $Q_t^{*}=\mathrm{diag}\{\sqrt{q_{ii,t}}\}$.
Positive definiteness follows from the same conditions as a univariate GARCH
process ($\alpha_m,\beta_n\ge0$, $\sum\alpha_m+\sum\beta_n<1$, …) and is *sufficient,
not necessary*.

## Two-step estimation and its properties

DCC's decisive advantage: the number of parameters in the correlation process is
**independent of the number of series**. Assuming $r_t\mid\mathcal F_{t-1}\sim
N(0,H_t)$, the log-likelihood splits into a volatility part and a correlation part,
$L(\theta,\phi)=L_V(\theta)+L_C(\theta,\phi)$, where $L_V$ is the sum of the
individual univariate-GARCH likelihoods. This yields a two-step scheme:
first $\hat\theta=\arg\max_\theta L_V(\theta)$ (univariate GARCH variances), then
$\max_\phi L_C(\hat\theta,\phi)$ (correlation dynamics from standardized residuals).
By two-stage GMM results, first-step consistency implies second-step consistency;
the estimator is consistent and asymptotically normal with a quasi-maximum-
likelihood (QML) interpretation, first-stage standard errors coinciding with the
Bollerslev–Wooldridge robust estimator. A constant-correlation test (a restricted
VAR on outer products of standardized residuals) **rejects constant correlation in
every model considered**, in favor of dynamic structure.

## Empirical properties

In Monte Carlo experiments with a known bivariate correlation process (constant,
sinusoidal, step, ramp), the mean-reverting ML-estimated DCC has the smallest mean
absolute error in four of six cases and the best summed error overall. On real data
the estimator recovers hard-to-quantify time variation — the Dow/NASDAQ correlation
swinging between 0.6–0.9 with dips below 0.4, and the European-currency breaks of
1992 and the pre-euro convergence toward unity. A particularly revealing check uses
the **minimum-variance portfolio**, whose weights $w_t=H_t^{-1}\iota/(\iota'H_t^{-1}\iota)$
depend entirely on the estimated covariance and thus amplify any misspecification:
DCC keeps standardized-residual variances inside the confidence band where
RiskMetrics does not, and is competitive with far more complex multivariate GARCH
specifications while being much simpler to estimate.

## Role as an optimizer input

The tie to portfolio selection is direct: both constrained optimal-portfolio
construction and portfolio standard-deviation computation require a covariance
forecast. Substituting a dynamic conditional forecast $H_t$ for the static sample
covariance makes the optimal weights themselves time-varying and responsive to
volatility and correlation shifts. The quality of the input covariance propagates to
the quality of the portfolio — the reason the minimum-variance portfolio serves as
a specification test bed. Dynamic covariance estimation thus sits upstream of the
optimization problems treated in [convex optimization](../optimization/convex-optimization.md)
and [risk budgeting and risk parity](../optimization/risk-budgeting-and-risk-parity.md),
supplying the risk input on which they depend. It is also the estimation counterpart
to the systemic-stress measures in
[turbulence and systemic risk](../regimes/turbulence-and-systemic-risk.md).
