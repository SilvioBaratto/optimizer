---
type: "Reference"
title: "Coherent Risk Measures"
openwiki_generated: true
sources:
  - id: openwiki-source-8a9a17ee9b377d6e6c6512c3
    resource: repo://docs/04_coherent_risk_measures.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---


# Coherent Risk Measures

This chapter defines the Value-at-Risk, establishes its properties and limits — chiefly that it
is not in general subadditive and hence not coherent — and presents the tail-based coherent
measures that replace it. The Conditional Value-at-Risk introduced here is optimized directly in
[CVaR optimization](../optimization/cvar-optimization.md), and the tail perspective extends to
the peak-to-trough measures of
[drawdown risk measures](./drawdown-risk-measures.md). Of the coherence axioms, only positive
homogeneity is needed for the Euler allocation of
[risk attribution, budget and limits](../risk-management/risk-attribution-budget-limits.md),
which applies to any differentiable risk measure homogeneous of degree one, VaR included.

## The Value-at-Risk: definition and interpretation

The **Value-at-Risk (VaR)**, widely used since the 1990s and approved mid-decade by the Basel
Committee for market-risk capital reserves, is qualitatively the *minimum* loss at a given
confidence level over a predefined holding period (the Basel Committee requires a one-tailed 99%
level over ten days; 95% over one day is the RiskMetrics convention). A
5% daily VaR of €50,000 means a 5% probability that the portfolio loses at least €50,000
overnight — informally, one day in twenty. Formally, for a risky payoff $`X`$ at confidence
$`(1-\epsilon)`$,

$$VaR_\epsilon(X) = -\inf_x \{x \mid P(X \le x) \ge \epsilon\},$$

that is, writing the tail probability as $`\alpha\in(0,1)`$, minus the quantile leaving
probability $`\alpha`$ to its left, $`VaR_\alpha(X)=-\inf\{x\mid P(X\le x)\ge\alpha\}`$.
Graphically $`-VaR_\alpha`$ is the P&L-distribution quantile beyond
which (to the left) the tail probability $`\alpha`$ of worst losses concentrates. VaR has two
notable properties failed by variance: for a riskless payoff $`C`$,
$`VaR_\epsilon(X+C)=VaR_\epsilon(X)-C`$, and for a positive constant $`\lambda`$,
$`VaR_\epsilon(\lambda X)=\lambda\,VaR_\epsilon(X)`$.

## The probabilistic definition and its critique

Probabilistically, for a continuous loss $`L=-X`$ the VaR at tail probability $`\alpha`$ satisfies
$`P(L>VaR_\alpha)=\alpha`$ (in general $`P(L>VaR_\alpha)\le\alpha\le P(L\ge VaR_\alpha)`$). This
reveals its main limit: VaR is *indifferent to how large the losses beyond the threshold actually
are*, saying nothing about the magnitude exceeding the level, so the
actual loss can be far worse. Portfolios with the same $`VaR_\alpha`$ can have dramatically
different loss levels in the worst $`100\alpha\%`$ of cases — VaR identifies a threshold, not the
severity beyond it. A structural defect compounds this: for two risky payoffs it can happen that

$$VaR_\epsilon(X+Y) > VaR_\epsilon(X) + VaR_\epsilon(Y),$$

so the combined risk exceeds the sum of individual risks, contradicting the diversification
effect. In general VaR is **not subadditive**, and since subadditivity is required of a
*coherent* risk measure, VaR is not coherent; one of the few cases where subadditivity holds is
jointly elliptical returns with finite variance (Gaussian included), linear portfolios and a
confidence level of at least 50%. This motivates the tail-based measures below.

## Computing VaR: the RiskMetrics Group approach

VaR of a single asset or portfolio can be computed analytically, by structured Monte Carlo or by
historical simulation, the three methods of the 1996 RiskMetrics document of J.P. Morgan and
Reuters; two routes are considered here, the analytical RiskMetrics approach and the historical
method. The RiskMetrics approach assumes conditionally multivariate-normal returns, with
exponentially weighted moving-average volatilities and a zero one-day mean; using the
standard-normal quantiles $`z_{0.01}=-2.3263`$ and $`z_{0.05}=-1.6449`$,

$$VaR_\epsilon = -(z_\epsilon\cdot\sigma_{return} + \mu_{return}) = |z_\epsilon|\,\sigma_{return} - \mu_{return},$$

with $`\sigma_{return},\mu_{return}`$ the standard deviation and mean of the return over the
holding period and VaR per unit of position value — the same formula applied to the portfolio
return for the VaR of a portfolio linear in the asset returns. Alongside this parametric approach, the
historical method derives VaR directly from the empirical return realizations without assuming a
distributional form.

## Average VaR, Conditional VaR and Expected Shortfall

The **Average Value-at-Risk (AVaR)** — also **Conditional Value-at-Risk (CVaR)** or **Expected
Shortfall (ES)** — is a coherent measure without VaR's shortcomings (it needs a finite expected
loss, $`E[X^-]<\infty`$, which a stable law with $`\alpha>1`$ has), defined as the average of
the VaRs at all tail levels below the chosen one:

$$AVaR_\epsilon(X) := \frac{1}{\epsilon}\int_0^\epsilon VaR_p(X)\,dp.$$

Where VaR answers "what loss could my portfolio exceed only with a small given probability?" (the
threshold breached with frequency $`\alpha`$), CVaR answers "when it loses more than that, how
much could it lose?" (the average tail loss beyond the VaR threshold), sitting deeper in the tail — between the VaR
threshold and the maximum observed loss. From the cumulative distribution $`F(x)=P(X\le x)`$ and
its inverse $`F^{-1}(\alpha)=\inf\{x\mid F(x)\ge\alpha\}`$, the Expected Shortfall at significance
(tail probability) $`\alpha\in(0,1)`$ is

$$ES_\alpha(X) := -\frac{1}{\alpha}\Big(E\big[X\,\mathbb 1_{\{X\le x^{(\alpha)}\}}\big] - x^{(\alpha)}\big(P[X\le x^{(\alpha)}]-\alpha\big)\Big),$$

with $`x^{(\alpha)}`$ any $`\alpha`$-quantile of $`X`$, for instance $`x^{(\alpha)}=-VaR_\alpha(X)`$,
and it admits the equivalent quantile-average form
$`ES_\alpha(X)=-\tfrac1\alpha\int_0^\alpha F^{-1}(p)\,dp`$. The ES coincides with the CVaR (and the
AVaR) for every distribution with $`E[X^-]<\infty`$, continuous or not. Empirically, ordering the
$`n`$ realizations and averaging the worst $`100\alpha\%`$ of them,

$$ES_\alpha(X) = -\frac{\sum_{i=1}^{w} X_{i:n}}{w},\qquad w=\max\{m\mid m\le n\alpha,\ m\in\mathbb N\}.$$

The ES is *universal* (applicable to any instrument and risk source), *simple and complete* (one
number even for multi-risk portfolios), and *continuous in the confidence level*: its value
changes only slightly when the confidence level shifts by a few basis points, whereas VaR, TCE,
and WCE can jump on discontinuous distributions.

## Tail and Worst Conditional Expectation

Two further tail measures on the left tail are the Tail and Worst Conditional Expectations. The
**Tail Conditional Expectation** (TailVaR) is
$`TCE_\alpha(X):=-E[X\mid X\le -VaR_\alpha(X)]`$, and the **Worst Conditional Expectation** is
$`WCE_\alpha(X):=-\inf\{E[X\mid A]\mid P[A]>\alpha\}`$ over unfavorable events $`A`$. Both address
"how bad is bad," taking the conditional mean of the left tail where losses reside. For
continuous variables the TCE coincides with the Conditional Value-at-Risk
$`\tfrac1\alpha\int_0^\alpha VaR_\gamma(X)\,d\gamma`$, recovering the same tail-quantile average.
They obey $`TCE_\alpha\le WCE_\alpha\le ES_\alpha`$, with equality throughout for continuous
distributions, but the coherence status of TCE and WCE is asymmetric: WCE fully satisfies the
coherence axioms but is almost purely theoretical (it depends on the underlying probability
space, not only on the distribution of $`X`$), while TCE is more tractable in practice but is
not always subadditive, hence not fully coherent.

## Two-sided coherent risk measures

The measures above act on one side (the loss tail); a different construction combines both sides
via a convex combination of the 1-norm of the *upside* and the $`p`$-norm of the *downside*:

$$\rho_{a,p}(X) := a\,\|(X-E[X])^+\|_1 + (1-a)\,\|(X-E[X])^-\|_p - E[X],$$

with $`\|Y\|_p=(E|Y|^p)^{1/p}`$ the norm under the probability measure (for $`n`$ equiprobable
scenarios, $`(\tfrac1n\sum_i|y_i|^p)^{1/p}`$) — a coherent risk measure written as a combination of
positive and negative distribution moments with weights $`a`$ and $`1-a`$. Since
$`E[(X-E[X])^+]=E[(X-E[X])^-]`$, the upside 1-norm equals the downside 1-norm, so the measure
depends only on deviations below the mean. The two parameters model risk aversion:
$`a\in[0,1]`$ is a *global* factor shifting weight between the first-order and the $`p`$-th-order
downside deviation, and $`p\in[1,+\infty)`$ is a *local* factor growing with risk aversion (the
$`p`$-norm is non-decreasing in $`p`$) and incorporating distributional
features like skewness and kurtosis (its degree of penetration into the distribution).

## Lévy-Pareto stable distributions: motivation

The distributional question underlying the choice of risk measure concerns the law of the log
return $`r_{t,1}=\ln(P_t/P_{t-1})`$. The empirical starting point — a histogram of daily returns
sharply concentrated at zero, with a high central peak and heavy tails reaching extreme values —
motivates comparing the normal against the Lévy-Pareto stable distribution:

| Normal distribution | Lévy-Pareto stable distribution |
|---|---|
| Finite variance | Infinite variance |
| "Normal" tails | Fat tails |
| Lower risk | Higher risk |

The theory runs from Pareto's income law $`Pr\{Y>y\}=y^{-\alpha}`$ above a threshold, with
$`\alpha`$ of about 1.5 (his estimates range from about 1.1 to 1.9) and infinite variance, through
Bachelier's normality hypothesis for price changes (later restated for $`r_{t,1}`$), the
introduction of stable random variables ($`\alpha\in(0,2]`$, infinite variance for every
$`\alpha<2`$), to the use of the log return as a stable
variable with infinite variance and fat tails (Mandelbrot, Fama).

## Stable random variables: definitions by convolution

Four equivalent definitions characterize the stable class; the first three use convolution
(summation). **Definition 1**: $`X`$ is *stable* if for every $`a,b>0`$ there exist $`c>0`$, $`d\in\mathbb R`$
with $`aX_1+bX_2=_d cX+d`$ for i.i.d. $`X_1,X_2,X`$ — the sum of two independent stable variables
with the same $`\alpha`$ is stable.
**Definition 2**: for every integer $`n\ge 2`$ there exist $`a_n>0`$, $`b_n`$ with
$`X_1+\cdots+X_n=_d a_nX+b_n`$ — the sum of $`n`$ independent copies of a stable variable is
stable. **Definition 3**: $`X`$ has
a domain of attraction, i.e. i.i.d. $`X_i`$ and constants $`c_n>0`$, $`d_n`$ exist with
$`(X_1+\cdots+X_n)/c_n+d_n\to_d X`$ — whenever the normalized sum of a sufficiently large number of
i.i.d. variables (even non-stable) converges, its limit is stable. Definitions 1 and 2 are
"sisters," 3 a "cousin," and
normal variables satisfy all three. Financially, the weekly return telescopes into a sum of daily
returns $`r_{t,5}=r_{t,1}+\cdots+r_{t-4,1}`$ (so i.i.d. stable daily returns give a stable weekly
return, Def. 1–2), and the annual return sums ~250 daily returns (so even non-stable i.i.d. daily
returns in the domain of attraction of a stable law give an approximately stable annual return,
Def. 3, normal when the daily variance is finite).

## Stable random variables: characteristic function and parameters

**Definition 4** characterizes the stable class via the characteristic function: $`X`$ is stable if
there exist $`\alpha\in(0,2]`$, $`\beta\in[-1,1]`$, $`\mu\in\mathbb R`$, $`\sigma\in[0,+\infty)`$ such that
$`E(e^{i\vartheta X})`$ takes the stable form (with a distinct $`\alpha=1`$ case), written
$`X\sim S(\alpha,\beta,\mu,\sigma)`$. The density is rarely available in closed form; two notable
cases are the **Cauchy** distribution $`S(1,0,\mu,\sigma)`$ with
$`f(x)=\sigma/\{\pi[\sigma^2+(x-\mu)^2]\}`$, and the **normal** $`N(\mu,2\sigma^2)`$ at $`\alpha=2`$. The
four parameters:
- **$`\alpha\in(0,2]`$ — characteristic exponent (stability index)**: tied to kurtosis and the tail
  probability. Smaller $`\alpha`$ means fatter tails; if $`\alpha<2`$ the variance is infinite (so the
  normal $`\alpha=2`$ is the only finite-variance stable law), and if $`\alpha\le1`$ the mean does
  not exist either — meaning not that risk is infinite but that the variance is not a "good" risk
  measure.
- **$`\mu\in\mathbb R`$ — location**: coincides with the mean when $`\alpha\in(1,2]`$; adding a
  constant translates the distribution.
- **$`\beta\in[-1,1]`$ — skewness**: $`\beta=0`$ symmetric, $`\beta<0`$/$`\beta>0`$ left/right skew, with
  $`\pm1`$ complete one-sided skew.
- **$`\sigma\in[0,+\infty)`$ — scale**: tied to dispersion, behaving like a standard deviation, and
  usable as a risk measure.

## The normality hypothesis: a historical digression and the empirical evidence

The idea that returns are normal arises not merely from the bell-shaped histograms but from
Bachelier's theoretical framework: the speculator has zero conditional expectation (a martingale
market) and the price is a continuous, time- and space-homogeneous Markov process whose
one-dimensional densities satisfy the Chapman-Kolmogorov equation, solved by the Gaussian density
with linearly growing variance. Bachelier's Gaussian law concerns price changes (an arithmetic
Brownian motion, negative prices allowed); normal log returns $`r_{t,1}`$ came later with Osborne
(1959) and Samuelson (1965). The continuity of the price process already rules out the
Lévy-Pareto stable laws with $`\alpha<2`$, whose processes jump; those laws were in any case
introduced only in 1925 and applied to finance from 1963. The
non-Gaussian approach asserts that empirical variances behave as if infinite and that empirical
distributions fit non-Gaussian stable members better; if the population variance of first
differences is infinite, the sample variance is a meaningless dispersion measure. This threatens
Markowitz-style selection, Sharpe-ratio performance measures, and Black-Scholes/Merton option
pricing. The distinction between the stable Paretian *distribution* and *hypothesis* is that both
the Gaussian and stable-Paretian hypotheses assume a stable underlying distribution; the conflict
is over $`\alpha`$, set to 2 by the Gaussian and strictly below 2 by the stable-Paretian.

Empirically, an iterative regression-type estimator applied to daily returns of six COMIT
sectoral indices (1984-1992, ~2000 observations each) and four Italian stocks yields $`\alpha`$
estimates such as 1.729 (Banking), 1.656 (Financial), 1.631 (Communications) for the indices and
1.504 (Ansaldo), 1.562 (Benetton), 1.719 (FIAT) for the stocks — all with positive $`\beta`$
(from 0.037 to 0.368). These were taken as representative of stock-return distributions: returns
are generally stable, rarely normal ($`\alpha\ne 2`$), and typically $`\alpha\in(1,2)`$, giving under
the stable model a finite mean but infinite variance, so the non-Gaussian stable distribution
reproduces both the higher central peak and the heavier tails better than the normal. A stable
fit to leptokurtic data always returns $`\alpha<2`$, however, and later tail-index studies of
daily index returns find a tail exponent of about 3, a finite variance: heavy tails are well
documented, infinite variance is not. Stable distributions are used in asset-liability
management, credit-risk models, risk management, portfolio selection, term-structure models, and
option pricing, with active links to the fractal structure of returns.
