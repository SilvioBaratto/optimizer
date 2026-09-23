---
type: concept
title: "Risk Budgeting and Risk Parity"
description: Allocating by risk contribution rather than capital. The Euler decomposition of volatility into marginal and total risk contributions, the Equal-Risk-Contribution (ERC) portfolio and its existence and uniqueness as the solution of a convex program with a logarithmic constraint, its placement between minimum variance and equal weighting, and Qian's risk-parity perspective across asset classes that corrects the hidden concentration of a classic 60/40.
tags: [risk-budgeting, risk-parity, equal-risk-contribution, marginal-risk-contribution, euler-decomposition, convex-optimization, diversification, leverage]
sources:
  - id: openwiki-source-959295574384ca4b5b14e0d3
    resource: repo://docs/17_risk_budgeting_and_risk_parity.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Risk Budgeting and Risk Parity

Risk budgeting shifts the object of allocation from *capital* to *risk*. Instead of
assigning weights to assets, the allocator assigns a budget of risk to each component and
seeks the portfolio that realizes it. The simplest case — an equal budget for every
component — is the Equal-Risk-Contribution (ERC) portfolio, a risk-based criterion
alongside [maximum diversification](../optimization/maximum-diversification.md) and, on
the measurement side, [risk attribution and budget limits](../risk-management/risk-attribution-budget-limits.md).

## From capital to risk

[Mean-variance selection](../foundations/mean-variance-selection.md) suffers two known
practical defects: optimal portfolios concentrate in a small subset of assets, and the
solution is excessively sensitive to inputs — especially expected returns, where small
changes reshape the portfolio. Alternatives (resampling, robust allocation) add
computational burden, so many investors prefer simple heuristics believed robust because
they ignore expected returns. Two well-known examples are the minimum-variance (MV)
portfolio — the only efficient-frontier portfolio using no expected-return information,
robust but concentrated — and the equal-weighted ($1/n$) portfolio, widely used and
efficient out of sample but prone to very limited *risk* diversification when individual
risks differ substantially. The idea here is an intermediate heuristic: **equalize the
risk contributions of the portfolio's components** — allocate risk rather than capital.
The risk contribution of a component is the share of total portfolio risk attributable to
it, computed as its weight times its marginal risk contribution (the change in total risk
from an infinitesimal increase in that position). Managing portfolios in terms of risk
contributions is standard institutional practice under the label *risk budgeting*; Qian
showed that risk contributions are not a mere ex-ante mathematical decomposition but carry
financial meaning as good predictors of the ex-post contribution to losses, especially
large ones.

## Risk decomposition: marginal and total contributions

For a portfolio $x=(x_1,\dots,x_n)$ with covariance matrix $\Sigma$, portfolio volatility
is $\sigma(x)=\sqrt{x^\top\Sigma x}$. The **marginal risk contributions** are the partial
derivatives

$$\partial_{x_i}\sigma(x)=\frac{x_i\sigma_i^2+\sum_{j\neq i}x_j\sigma_{ij}}{\sigma(x)},$$

which in vector form are $\Sigma x/\sqrt{x^\top\Sigma x}$ — the change in portfolio
volatility from a small increase in a component's weight. Defining the **total risk
contribution** of asset $i$ as $\sigma_i(x)=x_i\,\partial_{x_i}\sigma(x)$ gives the
decomposition

$$\sigma(x)=\sum_{i=1}^n\sigma_i(x).$$

The additivity is justified by a property of volatility: $\sigma$ is homogeneous of degree
1, $\sigma(\lambda x)=\lambda\sigma(x)$, so by Euler's theorem it equals the sum of its
arguments times their first partial derivatives, i.e. $\sigma(x)=\sum_i x_i\partial_{x_i}\sigma(x)=\sum_i\sigma_i(x)$;
in vector form $x^\top\Sigma x/\sqrt{x^\top\Sigma x}=\sqrt{x^\top\Sigma x}=\sigma(x)$. The
principle applies to any risk measure that is linearly homogeneous in the weights — under
suitable assumptions this holds for Value-at-Risk too (see
[coherent risk measures](../risk-measures/coherent-risk-measures.md)).

## The Equal-Risk-Contribution (ERC) portfolio

The ERC strategy seeks a risk-balanced portfolio with the same risk contribution for
every asset, restricted (for fair comparison with other heuristics, and because most
investors cannot short) to $0\le x\le1$:

$$x^\star=\Big\{x\in[0,1]^n:\ \textstyle\sum_i x_i=1,\ \ x_i(\Sigma x)_i=x_j(\Sigma x)_j\ \forall i,j\Big\},$$

using $\partial_{x_i}\sigma(x)\propto(\Sigma x)_i$. The budget constraint $\sum_i x_i=1$
acts only as normalization: any $y\ge0$ satisfying the equal-contribution condition can be
rescaled to $x_i=y_i/\sum_j y_j$.

**Two-asset case.** With correlation $\rho$ and $x=(w,1-w)$, equal contributions require
$w^2\sigma_1^2=(1-w)^2\sigma_2^2$, whose only solution in $[0,1]$ is
$x^\star=(\sigma_1^{-1},\sigma_2^{-1})/(\sigma_1^{-1}+\sigma_2^{-1})$ — **independent of the
correlation** $\rho$.

**General case.** For $n>2$ the problem generally has no closed form, but special cases do.
Under **constant correlation** $\rho_{ij}=\rho$, the ERC condition reduces to
$x_i\sigma_i=x_j\sigma_j$, giving $x_i=\sigma_i^{-1}/\sum_j\sigma_j^{-1}$: each weight is
the inverse of its volatility over the harmonic mean of volatilities — higher volatility,
lower weight. Under **equal volatilities** with differing correlations, the weight is the
inverse of $i$'s weighted average correlation over the same quantity summed across
assets, an endogenous solution since $x_i$ depends on itself. In the fully general case,
introducing the beta of component $i$ to the portfolio $\beta_i=\sigma_{ix}/\sigma^2(x)$
gives $\sigma_i(x)=x_i\beta_i\sigma(x)$, and equalizing contributions to $\sigma(x)/n$
yields

$$x_i=\frac{\beta_i^{-1}}{\sum_j\beta_j^{-1}}=\frac{\beta_i^{-1}}{n},$$

weight inversely proportional to beta — assets with high volatility or high correlation
with the rest are penalized. This too is endogenous.

## Existence, uniqueness, and numerical solution

Because of the endogeneity, the ERC portfolio generally requires a numerical algorithm.
One approach minimizes the variance of the (rescaled) risk contributions via SQP,
$f(x)=\sum_{i,j}(x_i(\Sigma x)_i-x_j(\Sigma x)_j)^2$ subject to $\mathbf1^\top x=1$,
$0\le x\le1$; the ERC portfolio exists only when $f(x^\star)=0$. A cleaner alternative is

$$y^\star=\arg\min\sqrt{y^\top\Sigma y}\quad\text{s.t.}\quad \sum_i\ln y_i\ge c,\ y\ge0,$$

with $c$ an arbitrary constant and $x_i^\star=y_i^\star/\sum_j y_j^\star$. This is a
convex function minimized subject to a convex constraint (see
[convex optimization](../optimization/convex-optimization.md)), which shows the ERC
solution is **unique whenever $\Sigma$ is positive definite**; relaxing the long-only
constraint can instead yield several solutions satisfying the ERC condition.

## Placement between minimum variance and equal weighting

The ERC sits naturally between $1/n$ and MV, appearing as a good substitute for both. The
three strategies are distinguished by (using the fact that MV portfolios equalize marginal
risk contributions):

$$x_i=x_j\ (1/n);\qquad \partial_{x_i}\sigma(x)=\partial_{x_j}\sigma(x)\ (\text{mv});\qquad x_i\partial_{x_i}\sigma(x)=x_j\partial_{x_j}\sigma(x)\ (\text{erc}).$$

$1/n$ equalizes weights; MV equalizes marginal contributions (a small increase in any
asset raises total risk equally, but total contributions are far from equal, so the
investor concentrates risk); ERC equalizes total contributions. Making the placement
explicit, the log-constrained program $\min\sqrt{x^\top\Sigma x}$ subject to
$\sum_i\ln x_i\ge c$, $\mathbf1^\top x=1$, $x\ge0$ recovers MV at $c=-\infty$ and $1/n$ at
$c=-n\ln n$ (since $\sum_i\ln x_i$ under $\sum_i x_i=1$ is maximized at $x_i=1/n$; the
quantity $-\sum_i x_i\ln x_i$ is entropy). ERC is thus a minimum-variance portfolio subject
to a sufficient-diversification constraint, giving the volatility ordering

$$\sigma_{\mathrm{mv}}\le\sigma_{\mathrm{erc}}\le\sigma_{1/n}.$$

Numerically, with four assets of volatility 10%, 20%, 30%, 40% and constant correlation,
$1/n$ assigns 25% each, ERC gives 48%/24%/16%/12% (inverse-volatility proportional), while
MV at zero correlation gives (70.2%, 17.6%, 7.8%, 4.4%) — far more concentrated. Out of
sample, equal-weighted portfolios are inferior on performance and every risk measure;
minimum-variance portfolios can reach higher Sharpe ratios via lower volatility but suffer
larger short-term drawdowns, are always far more concentrated, and are much less
turnover-efficient.

## When ERC is optimal

The ERC coincides with the maximum-Sharpe-ratio (MSR, tangency) portfolio
$\Sigma^{-1}(\mu-r)/\mathbf1^\top\Sigma^{-1}(\mu-r)$ under a specific assumption. The MSR
portfolio equalizes the ratio of marginal excess return to marginal risk across assets,
requiring $\mu-r=(\tfrac{\mu(x)-r}{\sigma(x)})\Sigma x/\sigma(x)$. The ERC is MSR-optimal
if one assumes a **constant correlation matrix and identical individual Sharpe ratios**
$s_i=(\mu_i-r)/\sigma_i$: under constant correlation the total contribution is
$(\Sigma x)_i/\sigma(x)$, equal across assets by construction, so equal Sharpe ratios
satisfy the MSR condition. When correlations differ or Sharpe ratios differ, ERC departs
from MSR.

## Risk budgeting: assigning risk budgets

ERC is the simplest instance of the general risk-budgeting principle: the allocator
assigns each component a risk budget and finds the portfolio realizing it; ERC distributes
an equal budget so no component contributes more than others (ex ante) — a "$1/n$ filter"
in risk space, maximizing the dispersion of risks rather than of weights. What makes this
operational is that the budget constraint $\sum_i x_i=1$ is mere normalization: the
allocation condition is stated on the total contributions $x_i\partial_{x_i}\sigma(x)$, and
the weight constraint only rescales afterward. Its financial relevance: Qian shows risk
contributions predict the contribution to losses (especially large ones), so allocating
risk budgets approximates allocating expected-loss budgets. ERC shares a
diversification philosophy with the Most-Diversified Portfolio (see
[maximum diversification](../optimization/maximum-diversification.md)), but the two are
generally distinct, coinciding only when the correlation coefficient among components is
unique.

## Qian's perspective: risk parity across asset classes

Qian's Risk Parity Portfolios allocate market risk equally across asset classes —
equities, bonds, commodities — aiming at true diversification that limits any single
component's loss impact. **All eggs in one basket.** A "balanced" 60/40 portfolio places
over 90% of its eggs in one basket, because sizes matter: with equity and bond annual
volatilities of 15% and 5%, equities are nine times riskier in variance terms (six equity
eggs of size 9 plus four bond eggs of size 1 give $6\times9+4=58$, of which 54 — about 93%
— are equity). Empirically, 1983-2004, the Russell 1000 excess return had 15.1%
annualized volatility, the Lehman Aggregate 4.6%, correlation 0.2; in a 60/40 equities
contributed 93% of risk and bonds 7%, and the 60/40 return correlated above 0.98 with the
Russell 1000. **From risk contribution to loss contribution.** For the 60/40, on losses
above 2% equities averaged 95.6% of the loss and bonds 4.4% — close to the 93% risk
contribution — giving empirical support to the economic interpretation of risk
contributions computed from variances and covariances.

**The risk-parity portfolios.** So the 60/40 is poorly diversified: in any appreciable
loss, over 90% traces to equities, making the bond diversification insignificant. Equating
the expected loss contribution across components — 23% Russell 1000 and 77% Lehman
Aggregate in the example — yields equal risk contribution and a near-parity loss split
(48.4% equities, 51.6% bonds on losses above 2%). On results (excess over 3-month T-bills,
1983-2004): Russell 1000 return 8.3%, sd 15.1%, Sharpe 0.55; Lehman 3.7%, 4.6%, 0.80;
60/40 6.4%, 9.6%, 0.67; parity 4.7%, 5.4%, 0.87. The 60/40 Sharpe (0.67) is *below* the
bond Sharpe — a sign of poor diversification — while the parity Sharpe (0.87) exceeds both
components, representing the benefits of true diversification.

**Optimality of risk parity.** Unlike mean-variance optimization, risk-parity portfolios
rest purely on risk diversification, yet are **mean-variance optimal if the components have
equal Sharpe ratios and uncorrelated returns**: equal Sharpe ratios mean expected return
proportional to each class's risk (assets priced by risk), and the actual equity-bond
correlation, though nonzero, is fairly low. Further benefits: every asset gets a nonzero
weight, and weights respond desirably to correlations — assets more correlated with the
rest receive less, so commodities would receive significant weight from their low
correlations. **Risk level and leverage.** The unlevered parity portfolio underperforms
the 60/40 in return because of much lower risk, so leverage is applied — typically to the
low-risk bonds so they match equities' risk contribution. A 1.8:1 levered parity matched
the 60/40's risk at 8.4% return vs 6.4% (about 2% p.a. more); a 2.8:1 version at equity
risk beat the Russell 1000 by nearly 5% p.a., with backtested Sharpe of 1.1 over
1983-2004. Risk-parity portfolios serve as standalone beta products or bases for alpha
strategies, in unlevered (4-5% risk), levered (~2:1, 8-10%), and global-macro (16-20%,
4:1) versions.
