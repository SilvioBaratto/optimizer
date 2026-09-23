---
type: concept
title: "Maximum Diversification"
description: The diversification ratio — the weighted average of asset volatilities divided by portfolio volatility — and the Most-Diversified Portfolio that maximizes it, treated as an allocation criterion distinct from minimum variance or risk parity. Its equivalent characterization as the minimum-variance portfolio on the correlation matrix, its core property (every held asset shares the same correlation with the portfolio), its three invariances, and the condition under which it coincides with the tangency portfolio.
tags: [maximum-diversification, diversification-ratio, most-diversified-portfolio, risk-based-allocation, correlation-matrix, tangency-portfolio, invariance, choueifaty]
sources:
  - id: openwiki-source-7a06bd8416f6deddb94f4ffc
    resource: repo://docs/19_maximum_diversification.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Maximum Diversification

Maximum diversification treats diversification not as a side effect of utility
optimization but as the *explicit* objective of portfolio construction. It belongs to
the family of risk-based allocation criteria alongside the
[risk budgeting and risk parity](../optimization/risk-budgeting-and-risk-parity.md)
approaches and the minimum-variance idea of
[mean-variance selection](../foundations/mean-variance-selection.md): like them it needs
no estimate of expected returns, building only on the covariance matrix.

## Diversification as an allocation criterion

Mean-variance selection needs two inputs: the covariance matrix and the vector of
expected returns. The first can be estimated with reasonable reliability; the second is
far harder, so much so that the most widely used models — the CAPM, or the
[Black-Litterman model](../foundations/black-litterman.md) — end up partly or wholly
setting expected returns aside. In parallel, the belief spread that
capitalization-weighted indices are inefficient, prompting alternatives such as
fundamental indexation or equal weighting. Haugen and Baker had already shown, for
1972-1989, that a minimum-variance portfolio of US stocks earned returns matching or
exceeding a broad cap index at systematically lower volatility, evidencing the *ex post*
inefficiency of the cap-weighted index. Against this backdrop the maximum-diversification
criterion recasts Markowitz's "only free lunch" of finance as the goal itself, favoring
diversification and avoiding bets founded on return forecasts or on the implicit bets of
cap-weighted benchmarks.

The setting: an universe of $N$ risky assets with volatility vector $\sigma=(\sigma_i)$,
correlation matrix $C=(\rho_{i,j})$, and covariance matrix
$V=\Sigma=(\rho_{i,j}\sigma_i\sigma_j)$. A portfolio is a weight vector $w$ with
$\sum_i w_i=1$; unless stated otherwise all portfolios are long-only (non-negative
weights).

## The diversification ratio

Writing $\langle w\mid\sigma\rangle=\sum_i w_i\sigma_i$ for the weighted average of asset
volatilities and $\sigma(w)=\sqrt{w'Vw}$ for portfolio volatility, the **diversification
ratio** of a portfolio $P$ is

$$D(w)=\frac{\langle w\mid\sigma\rangle}{\sigma(w)}=\frac{w'\sigma}{\sqrt{w'Vw}}.$$

For a long-only portfolio the overall volatility is at most the weighted sum of the
individual volatilities, so $D\ge1$ always, equalling $1$ exactly for a single-asset
portfolio (no diversification) and being strictly greater than $1$ unless the portfolio
is equivalent to a mono-asset one. Two elementary cases fix intuition: an equal-weight
portfolio of two independent equal-volatility assets has $D=\sqrt2$, and $N$ independent
equal-volatility assets give $D=\sqrt N$. An operational example: with two assets of
volatility 15% and 30% and correlation below 1, wanting both to contribute equally to
portfolio volatility makes the maximizing weights inversely proportional to
volatilities — 66.6% on $A$, 33.3% on $B$. With three assets — two strongly correlated
banks ($\rho=0.9$) and a weakly correlated pharma ($\rho=0.1$), all of equal volatility —
the diversification-maximizing weights are 25.7% each bank and 48.6% pharma:
diversification rewards the asset bringing more independent risk.

## Decomposition and degrees of freedom

The diversification ratio decomposes into two levers. With $\bar w=w\odot\sigma$,

$$\frac{1}{D(w)^2}=(1-\rho(w))\,\mathrm{CR}(w)+\rho(w),$$

where $\rho(w)$ is the volatility-weighted average correlation and $\mathrm{CR}(w)=\sum_i(w_i\sigma_i)^2/(\sum_i w_i\sigma_i)^2$
is the **volatility-weighted concentration ratio**. A fully concentrated long-only
portfolio has $\mathrm{CR}=1$ (mono-asset), while a volatility-equal-weighted portfolio
has the minimum $\mathrm{CR}=1/N$; $\mathrm{CR}$ generalizes the Herfindahl-Hirschman
index by weighting assets by volatility, so it measures concentration of *risk*, not just
of weights. The decomposition shows $D$ grows as average correlation and/or concentration
fall; if correlations tend to 1, $D=1$ regardless of $\mathrm{CR}$, and when pairwise
correlations are equal, maximizing $D$ reduces to minimizing $\mathrm{CR}$. The square
$D^2$ reads as the number of independent risk factors (degrees of freedom) represented in
the portfolio: the MSCI World had $D=1.7$ at end-2010, so a passive investor was exposed
to $1.7^2\approx3$ independent factors, whereas maximizing the ratio would have reached
$D=2.6$, i.e. $\approx7$ effective degrees of freedom.

## The Most-Diversified Portfolio: definition and optimality condition

Under a set $\Gamma$ of linear weight constraints, the portfolio maximizing $D$ is the
**Most-Diversified Portfolio (MDP)**, $w^{\mathrm{MDP}}=\arg\max_{w\in\Pi^+}D(w)$ in the
long-only case. Because $D$ is invariant to scalar rescaling of the weights, the
maximization is equivalent to the quadratic program

$$\min_w \tfrac12\,w'Vw \quad\text{s.t.}\quad w_i\ge0,\ \sum_i w_i\sigma_i=1,$$

rescaling weights to sum to 1 afterward. This is a QP over a convex set (see
[convex optimization](../optimization/convex-optimization.md)); existence follows, and
uniqueness follows too if $V$ is definite — for the objective to be finite there must be
no zero-volatility long-only portfolio carrying a positive premium. Applying the KKT
theorem to $f(w)=\ln D(w)$, the optimum satisfies

$$V\,w^{\mathrm{MDP}}=\frac{\sigma(w^{\mathrm{MDP}})}{D(w^{\mathrm{MDP}})}\,\sigma+\lambda,\qquad \min(\lambda,w^{\mathrm{MDP}})=0,$$

with non-negative dual variables $\lambda$ satisfying complementarity; the stationarity
condition is independent of the sum-to-one constraint, consistent with scale invariance.

## Geometric characterization: minimum variance on the correlation matrix

The MDP has a transparent characterization: it is the minimum-variance portfolio computed
on the **correlation matrix** rather than the covariance matrix, obtained by transferring
the problem to a synthetic universe where all assets share the same volatility. Assuming
lending/borrowing at a common rate, define synthetic assets
$Y_i=X_i/\sigma_i+(1-1/\sigma_i)\$$ (with $\$$ the risk-free asset); each $Y_i$ has unit
volatility, so the synthetic covariance matrix $V_S$ equals the original correlation
matrix $C$ (correlation is leverage-invariant). Imposing $S'\Sigma_S=1$, maximizing the
synthetic diversification ratio is minimizing $S'CS$ — minimizing variance in a universe
of equal-volatility assets, exactly the benefit expected from diversification. If $C$ is
invertible and $\Gamma=\varnothing$ the solution is $S\propto C^{-1}\mathbf1$, and
reconstructing the real assets (dividing each synthetic weight by its volatility, then
rescaling to 100% invested) gives

$$M\propto \sigma^{-1}C^{-1}\mathbf1,$$

with $\sigma$ the diagonal matrix of volatilities. The MDP is thus the minimum-variance
portfolio on the correlation matrix, rescaled by inverse volatilities.

## The core property: same correlation with the portfolio

From $M=\kappa\,\sigma^{-1}C^{-1}\mathbf1$ follows a set of correlation properties. Since
$VM=\kappa\,\sigma$, the correlation of any portfolio $P$ with $M$ is
$\rho_{P,M}=D(P)\,\kappa/\sigma_M$ — proportional to $P$'s own diversification ratio.
Applied to a single asset ($D=1$), $\rho_{i,M}=\kappa/\sigma_M$ is identical for every
asset: the MDP is the portfolio in which all assets share the same positive correlation
with it. Identifying the constant gives $\rho_{P,M}=D(P)/D(M)$ and $\rho_{i,M}=1/D(M)$,
yielding a single-(diversification-)factor model resembling the CAPM but identifying
correlation with the ratio of diversification levels.

In the constrained long-only case the equivalent **core property** states: every asset
*not* held by the MDP is more correlated with it than any asset that *is* held, and all
held assets share the same correlation with it. So all universe assets are effectively
represented even if not physically held — an MDP on a 500-stock index may hold about 50,
the other 450 being more correlated with the portfolio than the 50 held, consistent with
the MDP being the *non-diversifiable* portfolio. Equivalently, for any long-only $w$,
$\rho_{w,w^{\mathrm{MDP}}}\ge D(w)/D(w^{\mathrm{MDP}})$: the more diversified a long-only
portfolio, the higher its correlation with the MDP.

## The three invariances

An unbiased, agnostic construction process should treat an equivalent universe
identically. Choueifaty, Froidure and Reynier formalize three **invariance properties**,
all satisfied by the MDP:

1. **Duplication invariance** — duplicating an asset (e.g. multiple listings) leaves the
   weights on original assets unchanged, since a redundant asset yields a redundant
   first-order equation.
2. **Leverage invariance** — if a company changes its leverage, all else equal the
   weights allocated to its underlying business do not change (cash exposure is handled
   separately).
3. **Polico invariance** — adding a positive linear combination ("polico", e.g. a
   leveraged long-only ETF on a subset) does not alter the weights on original assets;
   this follows from the core property, since any unselected asset has correlation above
   $1/D(M)$ and a polico's diversification ratio exceeds 1, so it is never selected.

In a two-asset comparison ($\sigma_A=20\%$, $\sigma_B=10\%$, $\rho=50\%$) only the MDP
and the Equal-Risk-Contribution portfolio (ERC — see
[risk budgeting and risk parity](../optimization/risk-budgeting-and-risk-parity.md))
deliver a genuinely diversified risk allocation; equal weighting concentrates risk in the
more volatile asset and minimum variance invests 100% in the low-risk asset. On
invariances: the MDP satisfies all three; minimum variance is duplication-invariant only;
ERC is leverage-invariant only; equal weighting satisfies none.

## Relation to global minimum variance and the tangency portfolio

The MDP sits in precise relation to two notable frontier portfolios. **Global minimum
variance**: if all assets share the same volatility, the MDP coincides with the global
minimum-variance portfolio, because minimizing $S'CS$ on the correlation matrix then
equals minimizing $w'Vw$ on the covariance matrix up to scale. **Tangency and
mean-variance optimality**: the MDP is mean-variance optimal exactly when assets' excess
expected returns are proportional to their volatilities — "risk is homogeneously
rewarded." In a homogeneous universe where all assets have identical *ex ante* Sharpe
ratios, $E(r_i)-r_f=k\sigma_i$, so for any portfolio
$E(r_w)-r_f=k\langle w\mid\sigma\rangle=k\,\sigma(w)D(w)$; dividing by $\sigma(w)$ shows
maximizing $D$ equals maximizing the Sharpe ratio, making the MDP the tangency portfolio.
Under CAPM assumptions a security-market-line relation reappears with the MDP playing the
market portfolio's role,
$E(r_i)-r_f=\rho_{\mathrm{MDP}}(\sigma_i/\sigma_{\mathrm{MDP}})(E(r_{\mathrm{MDP}})-r_f)$;
by contrast the cap-weighted benchmark is optimal when excess returns are proportional to
total risk and correlation with the benchmark, and minimum variance is optimal when all
excess returns are equal.

## Empirical evidence and implementation

Operationally the MDP is built by solving the long-only QP $\min_w\tfrac12 w'Vw$ subject
to $w_i\ge0$ and $\sum_i w_i\sigma_i=1$ (rescaling afterward), or equivalently by
maximizing $D(w)$ directly under the desired constraints $\Gamma$. In practice one adds
per-asset weight caps, sector/region limits and turnover penalties, and estimates the
covariance with the methods of
[estimation error and shrinkage](../estimation/estimation-error-and-shrinkage.md) and
[dynamic covariance](../estimation/dynamic-covariance.md); the long-only constraint alone
acts like a robust estimation technique. Empirically, MDPs built monthly on US and
Eurozone equity universes (1992-2008) with covariance from 250 daily returns and a 4%
per-asset risk-contribution cap earned higher risk-adjusted returns than the cap
benchmark, minimum variance and equal weighting, at lower volatility than the index
(13.9% vs 17.9% Eurozone; 12.7% vs 13.4% US). On MSCI World (1999-2010) the MDP had the
highest diversification ratio — its primary goal — and the highest Sharpe ratio, closest
to tangency, while minimum variance realized the lowest *ex post* volatility; each keeps
its promise. Fama-French three-factor regressions confirm the MDP produces the highest
alpha among tested strategies. When the correlation matrix is singular the solution may be
non-unique, but since all such portfolios deliver maximum diversification and are
perfectly correlated, the choice is immaterial — the MDP stands as a solid candidate for
the non-diversifiable portfolio that classical theory identifies with the equity risk
premium.
