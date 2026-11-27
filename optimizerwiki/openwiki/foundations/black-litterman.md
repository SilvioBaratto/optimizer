---
type: concept
title: "Limits of MPT and the Black-Litterman Model"
description: Where the six assumptions of base mean-variance theory break, the operational problem of choosing a target return, and the Black-Litterman model as a Bayesian answer that updates the market-equilibrium implied returns with investor views to stabilize allocations.
tags: [black-litterman, mean-variance, bayesian, equilibrium-returns, reverse-optimization, views, estimation-error, capm]
sources:
  - id: openwiki-source-5ef1eda81135683d7afe90a4
    resource: repo://docs/03_mpt_limits_and_black_litterman_model.md
generated: { by: "claude-code", at: "2026-09-23T08:34:43.263Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Limits of MPT and the Black-Litterman Model

Base Modern Portfolio Theory rests on six assumptions that are individually easy to
understand but not all realistic. Separately from those assumptions, mean-variance
raises practical problems of its own: sensitivity to estimation error in the inputs and
the choice of a target return. The **Black-Litterman
model** answers as a Bayesian procedure: it updates the returns implied by market
equilibrium with the investor's views, weighting each by confidence. This page
extends [mean-variance selection](./mean-variance-selection.md) and feeds the
[views workflow from conditional forecasts to weights](../workflows/from-conditional-forecasts-to-weights.md).

## The six assumptions and where they break

The short-sale-admitting base MPT model rests on six assumptions about markets and
information, besides its own premises of a single period and of risk-averse,
non-satiated investors who rank portfolios by mean and variance alone:

1. **No transaction costs** on buying/selling assets;
2. **No taxation** on trading gains;
3. **Perfect divisibility** of every risky asset;
4. **Short sales allowed** on every risky asset;
5. Agents **know the first and second moments** of every asset's returns;
6. Agents' trades **do not influence** the return probability distributions.

Assumptions 1–3 jointly constitute the **frictionless-market** hypothesis.
Internet intermediation has pushed trading costs toward zero, but divisibility fails
structurally — assets trade in integer **minimum lots** (e.g. one share minimum on
the Italian equity market), and treating structural features as negligible frictions
can dangerously understate their impact. Assumption 4 (**no institutional
restrictions**) is often realistic but national regulators can suspend short-selling
in periods of stress. Assumption 5 is less innocent than it looks: it implicitly
requires that portfolios can be ranked by the first two moments of returns alone (true
under quadratic utility, or for any utility when returns are jointly elliptical — the
Gaussian is the natural candidate, but the fat-tailed Student-t also qualifies) and
knowledge of all pairwise covariances, whereas in reality agents can only estimate the
moments, which takes resources and brings estimation error; it
even presupposes those moments *exist*, which fails for Pareto-Lévy stable
distributions whose second moment is infinite. Assumption 6 (the **price-taker
investor**) treats every investor as small, whereas real markets contain medium and
large **price-makers** who move prices by trading in size.

## Choosing the target return operationally

Practitioners rarely use mean-variance intensively. The obstacle most often cited is
the sensitivity of the weights to estimation error in the inputs, above all the
expected returns (Michaud, 1989); another practical problem is the **operational choice
of the target return** $`r_P=\pi`$, because a coherent $`\pi`$
is hard to pin down in unstable markets — a high $`\pi`$ carries excessive variance, a
low $`\pi`$ leaves expected return "on the table." Three procedures dodge specifying
$`\pi`$:

- **Tangency portfolio.** Under the [fund separation theorem](./mean-variance-selection.md),
  the investor holds a mix of the risk-free asset (return $`r_c`$) and the tangency
  portfolio $`x=V^{-1}(r-r_c e)/[e'V^{-1}(r-r_c e)]`$, with $`r`$ the expected
  returns, $`V`$ the positive definite covariance matrix and $`e=(1,\dots,1)'`$; it is
  the efficient tangency portfolio when $`r_c<r_{GMV}=e'V^{-1}r/e'V^{-1}e`$. It fixes
  the risky mix only: the split with the risk-free asset still needs a risk aversion
  or a risk target. Its practical limit: sample tangency portfolios perform
  poorly out of sample.
- **Direct expected-utility maximization.** Solve $`\max_x x'r-\tfrac{\lambda}{2}x'Vx`$
  s.t. $`x'e=1`$ with $`\lambda>0`$ (at $`\lambda=0`$ the problem is unbounded, and the
  scale of $`\lambda`$ depends on the units and horizon of $`r`$ and $`V`$); the limit
  is choosing the risk-aversion $`\lambda`$, estimated by
  various authors to lie in $`[0,5]`$.
- **The maximum of $`r_{1/N}`$ and $`r_{GMV}`$.** Set the target to $`\max(r_{1/N},r_{GMV})`$,
  where $`r_{GMV}`$ is the expected return of the global minimum-variance portfolio
  $`x=V^{-1}e/(e'V^{-1}e)`$, the solution of $`\min_x x'Vx`$ s.t. $`x'e=1`$. Taking the
  larger of the two returns means the chosen frontier portfolio is never dominated by
  1/N (investors care about GMV only when it dominates 1/N).

### 1/N versus the global minimum-variance portfolio

The **equal-weighted 1/N portfolio** is easy to implement, delivers appreciable
performance, avoids concentration, always invests in the best performers, never
performs worse than the worst asset, and — under large estimation error — is expected
to *beat* mean-variance optimization itself. The **global minimum-variance (GMV)
portfolio** is efficient and, decisively, is unaffected by estimation error in the
expected returns: under i.i.d. normal returns the confidence interval for the mean
is about 40% wider in absolute terms than that for the standard deviation, and relative
to the size of the parameter the mean's estimation error is about an order of magnitude
larger and does not shrink with higher sampling frequency, so routing around expected
returns is a genuine robustness gain. GMV still needs a well-conditioned covariance
estimate, which is singular when there are no more observations than assets. After
the 2007 crisis investors shifted toward these less risky portfolios.

## Black-Litterman as a Bayesian answer

Black-Litterman starts from the CAPM equilibrium, in which the **market portfolio**
$`M`$ (asset weights proportional to capitalization) is held by the representative
investor, and, under the CAPM assumptions (including homogeneous expectations), all
investors hold combinations of the risk-free asset and
$`M`$ along the capital market line — so in theory everyone holds the *same* risky
portfolio. In practice investors hold different opinions, expressed as **views** on
an asset's absolute expected return or on the return of one asset relative to
another. The model integrates these views with the market's, balancing them by the
investor's confidence; it is commonly derived and presented as an application of
**Bayes' theorem**: the prior distribution of expected returns implicit in market
equilibrium is updated by the new information in the views to yield a
posterior, $`\Pr(M\mid W)=\Pr(M)\Pr(W\mid M)/\Pr(W)`$.

## The prior and the equilibrium returns Π*

The prior rests on two distributional assumptions: returns are normal
$`\mathbf R\sim\mathcal N(\mathbf r,\mathbf V)`$, and the expected-return vector is
itself normal $`\mathbf r\sim\mathcal N(\mathbf\Pi,\tau\mathbf V)`$ with an uncertainty
scalar $`\tau\in(0,\infty)`$, typically small (0.01 to 0.05), and $`\mathbf V`$
positive definite. The mean $`\mathbf\Pi`$ is obtained by **reverse
optimization** (the inverse process): rather than mapping returns to optimal weights,
it starts from the observed market-capitalization weights $`\mathbf x_M`$ (which the
CAPM justifies as the equilibrium portfolio) and recovers the
equilibrium expected returns that justify them. Maximizing the mean-variance
criterion $`\max_x x'\Pi-\tfrac{a}{2}x'Vx`$ without a budget constraint, with
$`\Pi`$ the expected returns in excess of the risk-free rate, the first-order
condition at the market-clearing weights $`x=x_M`$, $`\Pi-aVx_M=0`$, gives

$$\mathbf\Pi^*=a\,\mathbf V\,\mathbf x_M,$$

the returns implied by market equilibrium and the mean of the prior.

## Views and the posterior distribution

Views are specified as $`\mathbf P\mathbf r\sim(\mathbf Q,\mathbf\Omega)`$, where
$`\mathbf P`$ is the $`(K,N)`$ **view-mapping matrix**, $`\mathbf Q`$ the $`(K,1)`$ vector of
view returns, and $`\mathbf\Omega`$ the $`(K,K)`$ matrix of view variances, usually
taken **diagonal** (independent view errors); $`K`$ is the number of views (rows of
$`\mathbf P`$), typically but not necessarily no larger than $`N`$; views can be
absolute (a single asset's expected return) or relative (one asset versus another).
If the views are Gaussian,
$`\mathbf Q\mid\mathbf r\sim\mathcal N(\mathbf P\mathbf r,\mathbf\Omega)`$ with
$`\mathbf\Omega`$ positive definite, combining the prior $`\mathbf r\sim\mathcal
N(\mathbf\Pi^*,\tau\mathbf V)`$ with the views by Bayes' theorem yields the posterior of
the expected returns $`\mathbf r\mid\mathbf Q\sim\mathcal N(\mathbf r_{BL},\mathbf V_{BL})`$ with

$$
\mathbf r_{BL}=\big[(\tau\mathbf V)^{-1}+\mathbf P'\mathbf\Omega^{-1}\mathbf P\big]^{-1}\big[(\tau\mathbf V)^{-1}\mathbf\Pi^*+\mathbf P'\mathbf\Omega^{-1}\mathbf Q\big],\qquad
\mathbf V_{BL}=\big[(\tau\mathbf V)^{-1}+\mathbf P'\mathbf\Omega^{-1}\mathbf P\big]^{-1}.
$$

$`\mathbf V_{BL}`$ is the uncertainty left about the mean; the returns themselves have
covariance $`\mathbf V+\mathbf V_{BL}`$, $`\mathbf R\mid\mathbf Q\sim\mathcal N(\mathbf r_{BL},\mathbf V+\mathbf V_{BL})`$.
The posterior mean $`\mathbf r_{BL}`$ is a **precision-weighted** (matrix-weighted)
combination of the equilibrium returns $`\mathbf\Pi^*`$ and the views $`\mathbf Q`$, and
depends on $`\tau`$ and $`\mathbf\Omega`$ only through $`\mathbf\Omega/\tau`$: the smaller
$`\mathbf\Omega`$ (the more confident the investor), the closer the view portfolios'
returns $`\mathbf P\mathbf r_{BL}`$ move to $`\mathbf Q`$; the larger $`\mathbf\Omega`$,
the closer $`\mathbf r_{BL}`$ stays to $`\mathbf\Pi^*`$.

## A complete numerical example

For $`N=3`$ assets with market weights $`x_M=(0.50,0.30,0.20)'`$, risk aversion $`a=2.5`$,
uncertainty scalar $`\tau=0.05`$ and covariance
$`V=\begin{psmallmatrix}0.0225&0.0060&0.0027\\0.0060&0.0400&0.0090\\0.0027&0.0090&0.0324\end{psmallmatrix}`$,
reverse optimization gives the equilibrium excess returns
$`\Pi^*=(0.0340,0.0420,0.0263)'`$, and the inverse map $`x_M=\tfrac1a V^{-1}\Pi^*`$
recovers the starting weights exactly. The investor states $`K=2`$ views — an absolute
$`E(r_1)=0.06`$ and a relative $`E(r_2-r_3)=0.02`$ — encoded as $`Q=(0.06,0.02)`$,
$`\Omega=\mathrm{diag}(0.0004,0.0009)`$, and $`P=\begin{psmallmatrix}1&0&0\\0&1&-1\end{psmallmatrix}`$.
The posterior mean is $`r_{BL}=(0.0532,0.0478,0.0281)`$. With the covariance of returns
$`V+V_{BL}`$, the optimal weights $`x=\tfrac1a (V+V_{BL})^{-1}r_{BL}=(0.833,0.297,0.179)`$
sum to 1.309, the extra 0.309 being borrowed at the risk-free rate; normalized to a fully
invested portfolio they are $`(0.636,0.227,0.137)`$. The absolute view raises Asset 1's
weight well above its starting 50%, and the relative view moves weight from Asset 3 to
Asset 2 (the ratio of their weights rises from 1.50 to 1.66); Assets 2 and 3 both fall
after normalization because of the rescaling.

## Relation to the wider machinery

The following readings go beyond what the chapter derives. Black-Litterman can be seen
as an equilibrium-prior counterpart to the shrinkage remedies for estimation error,
since the posterior pulls the expected returns toward $`\Pi^*`$. Because a view is any
linear statement $`\mathbf P\mathbf r`$ about expected returns, views can in principle
refer to portfolios built on [factor models](../factor-models/factor-models.md) or be
informed by the regimes in
[market regimes and views](../regimes/market-regimes-and-views.md), which also discusses
views on covariances that this model's views on expected returns do not cover. The
pipeline in
[from conditional forecasts to weights](../workflows/from-conditional-forecasts-to-weights.md)
reaches weights by other routes (plug-in moments, Bayesian predictive distributions,
direct weight policies) and
connects to Black-Litterman through its conditioning views, not through reverse
optimization.
