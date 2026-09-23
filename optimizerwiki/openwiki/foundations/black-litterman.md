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
understand but not all realistic. Once they break, the practical problem becomes how
to pick a target return without producing fragile allocations. The **Black-Litterman
model** answers as a Bayesian procedure: it updates the returns implied by market
equilibrium with the investor's views, weighting each by confidence. This page
extends [mean-variance selection](./mean-variance-selection.md) and feeds the
[views workflow from conditional forecasts to weights](../workflows/from-conditional-forecasts-to-weights.md).

## The six assumptions and where they break

The short-sale-admitting base MPT model rests on six assumptions:

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
requires the return distributions to be fully described by their first two moments
(symmetric, mesokurtic — the Gaussian is the natural candidate), knowledge of all
pairwise covariances, and that agents estimate rather than know the true moments; it
even presupposes those moments *exist*, which fails for Pareto-Lévy stable
distributions whose second moment is infinite. Assumption 6 (the **price-taker
investor**) treats every investor as small, whereas real markets contain medium and
large **price-makers** who move prices by trading in size.

## Choosing the target return operationally

Practitioners rarely use mean-variance intensively; the chief practical obstacle is
the **operational choice of the target return** $r_P=\pi$, because a coherent $\pi$
is hard to pin down in unstable markets — a high $\pi$ carries excessive variance, a
low $\pi$ leaves expected return "on the table." Three procedures dodge specifying
$\pi$:

- **Tangency portfolio.** Under the [fund separation theorem](./mean-variance-selection.md),
  the investor holds a mix of the risk-free asset and the tangency portfolio
  $x=V^{-1}(r-r_c e)/[e'V^{-1}(r-r_c e)]$, so the mix implicitly fixes the point on
  the efficient frontier. Its practical limit: sample tangency portfolios perform
  poorly out of sample.
- **Direct expected-utility maximization.** Solve $\max_x x'r-\tfrac{\lambda}{2}x'Vx$
  s.t. $x'e=1$; the limit is choosing the risk-aversion $\lambda$, estimated by
  various authors to lie in $[0,5]$.
- **The maximum of $r_{1/N}$ and $r_{GMV}$.** Set the target to $\max(r_{1/N},r_{GMV})$,
  where $r_{GMV}$ solves $\min_x x'Vx$ s.t. $x'e=1$ with solution
  $x=V^{-1}e/(e'V^{-1}e)$.

### 1/N versus the global minimum-variance portfolio

The **equal-weighted 1/N portfolio** is easy to implement, delivers appreciable
performance, avoids concentration, always invests in the best performers, never
performs worse than the worst asset, and — under large estimation error — is expected
to *beat* mean-variance optimization itself. The **global minimum-variance (GMV)
portfolio** is efficient and, decisively, is unaffected by estimation error in the
expected returns: under i.i.d. normal returns the confidence interval for the means
is about 40% wider than that for the standard deviation, so routing around expected
returns is a genuine robustness gain. After the 2007 crisis investors shifted toward
these less risky portfolios.

## Black-Litterman as a Bayesian answer

Black-Litterman starts from the CAPM equilibrium, in which the **market portfolio**
$M$ (asset weights proportional to capitalization) is held by the representative
investor, and all rational investors hold combinations of the risk-free asset and
$M$ along the capital market line — so in theory everyone holds the *same* risky
portfolio. In practice investors hold different opinions, expressed as **views** on
an asset's absolute expected return or on the return of one asset relative to
another. The model integrates these views with the market's, balancing them by the
investor's confidence, via **Bayes' theorem**: the prior return distribution implicit
in market equilibrium is updated by the new information in the views to yield a
posterior, $\Pr(M\mid W)=\Pr(M)\Pr(W\mid M)/\Pr(W)$.

## The prior and the equilibrium returns Π*

The prior rests on two distributional assumptions: returns are normal
$\mathbf R\sim\mathcal N(\mathbf r,\mathbf V)$, and the expected-return vector is
itself normal $\mathbf r\sim\mathcal N(\mathbf\Pi,\tau\mathbf V)$ with an uncertainty
scalar $\tau\in[0,\infty)$. The mean $\mathbf\Pi$ is obtained by **reverse
optimization** (the inverse process): rather than mapping returns to optimal weights,
it starts from the known market weights $\mathbf x_M$ (from the CAPM) and recovers the
equilibrium expected returns that justify them. Maximizing the quadratic utility
$\max_x x_M'\Pi-\tfrac{a}{2}x_M'Vx_M$, the first-order condition $\Pi-aVx_M=0$ gives

$$\mathbf\Pi^*=a\,\mathbf V\,\mathbf x_M,$$

the returns implied by market equilibrium and the mean of the prior.

## Views and the posterior distribution

Views are specified as $\mathbf P\mathbf r\sim(\mathbf Q,\mathbf\Omega)$, where
$\mathbf P$ is the $(K,N)$ **view-mapping matrix**, $\mathbf Q$ the $(K,1)$ vector of
view returns, and $\mathbf\Omega$ the **diagonal** $(K,K)$ matrix of view variances,
with $K\le N$; views can be absolute (a single asset's expected return) or relative
(one asset versus another). Combining the prior $\mathbf r\sim\mathcal
N(\mathbf\Pi^*,\tau\mathbf V)$ with the views by Bayes' theorem yields the posterior
$\mathbf R_{BL}\sim(\mathbf r_{BL},\mathbf V_{BL})$ with

$$\mathbf r_{BL}=\big[(\tau\mathbf V)^{-1}+\mathbf P'\mathbf\Omega^{-1}\mathbf P\big]^{-1}\big[(\tau\mathbf V)^{-1}\mathbf\Pi^*+\mathbf P'\mathbf\Omega^{-1}\mathbf Q\big],\qquad
\mathbf V_{BL}=\big[(\tau\mathbf V)^{-1}+\mathbf P'\mathbf\Omega^{-1}\mathbf P\big]^{-1}.$$

The posterior mean $\mathbf r_{BL}$ is a **precision-weighted average** of the
equilibrium returns $\mathbf\Pi^*$ and the views $\mathbf Q$: the smaller
$\mathbf\Omega$ (the more confident the investor), the closer $\mathbf r_{BL}$ moves
to $\mathbf Q$; the larger $\mathbf\Omega$, the closer it stays to $\mathbf\Pi^*$.

## A complete numerical example

For $N=3$ assets with market weights $x_M=(0.50,0.30,0.20)'$, risk aversion $a=2.5$,
and the given covariance $V$, reverse optimization gives the equilibrium returns
$\Pi^*=(0.0340,0.0420,0.0263)'$, and the inverse map $x_M=\tfrac1a V^{-1}\Pi^*$
recovers the starting weights exactly. The investor states $K=2$ views — an absolute
$E(r_1)=0.06$ and a relative $E(r_2-r_3)=0.02$ — encoded as $Q=(0.06,0.02)$,
$\Omega=\mathrm{diag}(0.0004,0.0009)$, and $P=\begin{psmallmatrix}1&0&0\\0&1&-1\end{psmallmatrix}$.
The posterior mean is $r_{BL}=(0.0532,0.0478,0.0281)$, and the optimal weights
$x=\tfrac1a V^{-1}r_{BL}=(0.627,0.230,0.143)$ show the view raises Asset 1's weight
well above its starting 50% at the expense of the other two.

## Relation to the wider machinery

Black-Litterman is the equilibrium-prior counterpart to the shrinkage remedies for
estimation error, and its view mechanism generalizes to factor-referenced views built
on [factor models](../factor-models/factor-models.md) and to regime-conditioned views in
[market regimes and views](../regimes/market-regimes-and-views.md). Its reverse
optimization and posterior-return blending are the theoretical basis for the
practical pipeline in
[from conditional forecasts to weights](../workflows/from-conditional-forecasts-to-weights.md).
