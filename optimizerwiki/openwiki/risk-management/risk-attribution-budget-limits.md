---
type: concept
title: "Risk Attribution, Budget and Limits"
description: How the risk of an already-allocated portfolio decomposes into additive contributions imputable to its components — the Euler allocation principle and its RORAC compatibility, closed-form contributions for standard deviation, VaR and Expected Shortfall, their financial interpretation as expected contributions to loss, the axiomatic justification of coherent capital allocation (the Aumann-Shapley value), and the risk-budgeting framework with its existence and uniqueness conditions.
tags: [risk-attribution, euler-allocation, rorac, risk-contribution, value-at-risk, expected-shortfall, risk-budgeting, capital-allocation, aumann-shapley, diversification-index]
sources:
  - id: openwiki-source-93d6b2b16aae80fd170cd927
    resource: repo://docs/23_risk_attribution_budget_and_limits.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Risk Attribution, Budget and Limits

This chapter establishes how the risk of an already-allocated portfolio decomposes into
additive contributions imputable to its components — securities, factors, sub-portfolios — and
how those contributions become the instrument for monitoring and enforcing limits. Attribution
is distinct from construction: building equally-weighted-risk-contribution and risk-parity
portfolios is the subject of [risk budgeting and risk parity](../optimization/risk-budgeting-and-risk-parity.md);
here the focus is ex-ante attribution on a given book and the limit control that follows. It rests
on the [factor models](../factor-models/factor-models.md) used to isolate systematic risk drivers
and connects to the loss-distribution reasoning of
[stress testing and scenarios](./stress-testing-and-scenarios.md).

## Attribution versus construction: the problem

Take an already-allocated portfolio. Let $X_1,\dots,X_n$ be the profit/loss random variables of
the individual securities, factors, or sub-portfolios, and $X=\sum_i X_i$ the profit/loss of the
whole book. The economic capital required to cover heavy losses is set by a risk measure $\rho$,
$\mathrm{EC}=\rho(X)$, typically tied to the variance or a quantile of the loss distribution.
Measuring aggregate risk is only the first step: identifying concentrations, risk-sensitive
pricing, and book diagnostics all require decomposing economic capital into a sum of
contributions imputable to the sub-portfolios or individual exposures. This is the attribution
question — *how much does component $i$ contribute to $\mathrm{EC}=\rho(X)$?*

The problem is non-trivial because of the diversification effect: the sum of the components'
stand-alone risks is normally larger than the risk of their sum. The group's risk capital falls
below the sum of stand-alone capitals, and this diversification benefit must be shared fairly
among the components. The exercise serves comparison: knowing both the profit generated and the
risk taken by each component allows a far more sensible comparison than profit alone, and it
underlies risk-adjusted performance measures and the return on risk-adjusted capital (RORAC).

## The Euler principle and RORAC compatibility

To define contributions, introduce weights $u=(u_1,\dots,u_n)$ and consider
$X(u)=\sum_i u_i X_i$ with $X=X(1,\dots,1)$, so $u_i$ is the amount invested in $X_i$; fixing the
joint distribution, set $f_\rho(u)=\rho(X(u))$. The Euler allocation principle applies to any
risk measure homogeneous of degree one, i.e. $\rho(hX)=h\,\rho(X)$ for $h\ge 0$. With
$\mu_i=\mathrm{E}[X_i]$, the portfolio and per-component RORAC are $\mathrm{RORAC}(X)=\mathrm{E}[X]/\rho(X)$
and $\mathrm{RORAC}(X_i\mid X)=\mu_i/\rho(X_i\mid X)$. Contributions satisfy the **full allocation
property** if $\sum_i\rho(X_i\mid X)=\rho(X)$, and are **RORAC compatible** if raising a
higher-RORAC component's weight marginally improves portfolio RORAC — the formal statement that
contributions correctly steer reallocation decisions.

**Uniqueness.** For a smooth risk measure, RORAC compatibility fully determines the
contributions. If $f_\rho$ is continuously differentiable and contributions are RORAC compatible
for arbitrary expected values, they are uniquely given by the directional derivative
$$ \rho_{\text{Euler}}(X_i\mid X) = \left.\frac{d\rho}{dh}(X+hX_i)\right|_{h=0} = \frac{\partial f_\rho}{\partial u_i}(1,\dots,1). $$
By Euler's theorem on homogeneous functions, $f_\rho(u)=\sum_i u_i\,\partial f_\rho/\partial u_i$
holds *if and only if* $f_\rho$ is homogeneous of degree one, so full allocation and RORAC
compatibility are obtained simultaneously exactly when $\rho$ is degree-one homogeneous. These
are the **Euler contributions**, and the procedure is the **Euler allocation**; it has been
justified from practical, portfolio-diagnostic, and game-theoretic angles.

**Marginal contributions.** The Euler contribution differs from the *marginal* (with-without)
contribution $\rho_{\text{marg}}(X_i\mid X)=\rho(X)-\rho(X-X_i)$. For sub-additive, continuously
differentiable, degree-one-homogeneous measures the marginal contributions are always below the
Euler contributions, so their sum *understates* total risk and fails the full allocation
property; forcing it by rescaling generally destroys RORAC compatibility — the technical reason
ex-ante attribution uses Euler and not marginal contributions.

## Closed-form contributions: standard deviation, VaR and Expected Shortfall

For the most-used families the defining partial derivative has closed form. For the
standard-deviation family $\sigma_c(X)=c\sqrt{\mathrm{var}[X]}$ (homogeneous and sub-additive) the
Euler contribution is proportional to the covariance between component and portfolio,
$$ \sigma_c(X_i\mid X) = c\,\frac{\mathrm{cov}[X_i,X]}{\sqrt{\mathrm{var}[X]}}. $$
For the **Value-at-Risk** $\mathrm{VaR}_\alpha(X)=q_\alpha(-X)$ — homogeneous and co-monotonic
additive but not in general sub-additive — under regularity conditions (a density for $X$) the
Euler contributions are conditional expectations,
$\mathrm{VaR}_\alpha(X_i\mid X)=-\mathrm{E}[X_i\mid X=-\mathrm{VaR}_\alpha(X)]$. For the
**Expected Shortfall** $\mathrm{ES}_\alpha(X)=(1-\alpha)^{-1}\int_\alpha^1\mathrm{VaR}_u(X)\,du$ —
homogeneous, co-monotonic additive *and* sub-additive — the contribution is the elementary
conditional expectation
$$ \mathrm{ES}_\alpha(X_i\mid X) = -\mathrm{E}[X_i\mid X\le -\mathrm{VaR}_\alpha(X)] = -(1-\alpha)^{-1}\,\mathrm{E}\!\left[X_i\,\mathbf{1}_{\{X\le -\mathrm{VaR}_\alpha(X)\}}\right], $$
elementary because the conditioning event has positive probability — unlike the VaR case.

## Estimating contributions from sample data

When the loss distribution is not analytic it is estimated from a simulated or historical
sample. For standard-deviation and ES contributions, substituting the empirical variables into
the contribution formulas yields consistent estimators, and ES estimation is direct thanks to
the elementary conditional-expectation form. The naïve approach fails for VaR contributions when
$X$ is continuous, since the conditioning event $\{X=-\mathrm{VaR}_\alpha(X)\}$ has zero
probability; smoothing the empirical measure (kernel estimation with a bandwidth $b$) yields an
approximation whose right-hand side is exactly the Nadaraya-Watson kernel estimator of
$\mathrm{E}[X_i\mid X=-\mathrm{VaR}_\alpha(X)]$, linking Euler VaR contributions to nonparametric
conditional-expectation estimation. Bandwidth choice is critical (e.g. Silverman's rule
$b=0.9\min(\sigma,R/1.34)N^{-1/5}$), and VaR/ES contribution estimates are highly volatile, so
importance sampling has been proposed to dampen the problem.

A closed-form alternative uses the **Cornish-Fisher** expansion, keeping the "mean plus z-score
times standard deviation" form but correcting the z-score for higher moments,
$\tilde z_\alpha\approx z_\alpha+\tfrac16(z_\alpha^2-1)s+\tfrac1{24}(z_\alpha^3-3z_\alpha)k-\tfrac1{36}(2z_\alpha^3-5z_\alpha)s^2$,
with $s,k$ skewness and excess kurtosis. Since mean, standard deviation, skewness and kurtosis
are explicit functions of the weights, the Cornish-Fisher VaR is a differentiable function of
the weights whose partial derivatives give the contributions — especially relevant for
portfolios holding hedge funds, whose returns can have significant skewness and kurtosis that a
moment-ignoring risk budget would badly understate.

## Marginal contributions, concentration and diversification

Euler allocation is well suited to spotting risk concentrations — single exposures or groups
large enough to threaten a bank's soundness, likely the single most important cause of severe
problems. If $\rho$ is homogeneous of degree one *and* sub-additive, Euler contributions never
exceed the components' stand-alone risks, $\rho_{\text{Euler}}(X_i\mid X)\le\rho(X_i)$ (for
credit assets, never above face value); for degree-one-homogeneous, continuously differentiable
measures this property and sub-additivity are essentially equivalent. The **diversification
index** $\mathrm{DI}_\rho(X)=\rho(X)/\sum_i\rho(X_i)$ satisfies $\mathrm{DI}_\rho(X)\le 1$ under
homogeneity, sub-additivity and co-monotonic additivity; a value near 100% flags nearly
co-monotone (highly dependent) components — a high-concentration portfolio — while a low value
signals good diversification. The **risk impact** of a systematic factor set $S$ applies Euler
allocation to the loss's decomposition into its factor-conditional expectation and orthogonal
residual; for the standard-deviation measure it reduces to
$\mathrm{RI}_\sigma(L\mid S)=\mathrm{var}[\mathrm{E}[L\mid S]]/\mathrm{var}[L]\in[0,1]$, a
generalization of the regression $R^2$ that ranks factors by their impact.

## The financial interpretation of contributions (Qian)

A recurring objection is that the contribution — a partial derivative of a non-additive risk —
is a mere mathematical decomposition without economic content, casting doubt on whether risk
budgets "add up." The answer is that they do, via interpreting contributions as expected
*contributions to loss*. For a two-asset portfolio the **percentage contribution to risk**
$p_i=(w_i\,\partial\sigma/\partial w_i)/\sigma$ equals the ratio of the component-portfolio
covariance to total variance — the component's **beta** to the portfolio — and the betas sum to
one. Assuming jointly normal returns, the expected percentage contribution to a loss of size $L$
is
$$ c_i = \frac{w_i\mu_i}{L} + \frac{\mathrm{cov}(w_i r_i, R)}{\mathrm{var}(R)}\left(1-\frac{\mu_R}{L}\right) = p_i + \frac{D_i}{L}, $$
and since $\sum_i D_i=0$ the expected loss contributions sum to 100%, exactly like the percentage
contributions to risk. This is the central relation: **risk budgeting is loss budgeting**.

The two coincide, $c_i=p_i$, in three cases: when expected returns are null ($\mu_i=0$, a
reasonable short-horizon assumption), independent of $L$; the trivial zero-weight case; and, most
interestingly, when $w_1\mu_1/p_1=w_2\mu_2/p_2$ — the first-order condition for a
mean-variance-optimal portfolio, treated in
[mean-variance selection](../foundations/mean-variance-selection.md) — where the risk budget
becomes an expected-return budget. The $D_i$ terms measure the portfolio's sub-optimality: when
loss $L$ far exceeds them (as in crises) $c_i\approx p_i$, so the loss contribution is well
captured by the risk contribution; in quiet periods with tiny losses the ex-post loss attribution
may bear no relation to the risk contribution, but these small events do not undermine the
concept. The interpretation extends to VaR: the percentage VaR contribution is exactly the
expected contribution to a loss equal to the VaR, though more restrictive — it applies only to the
loss that exactly equals the VaR, and changing the VaR changes the contribution.

## The coherence of risk-capital allocation (Denault)

Denault's approach is axiomatic — argue the necessary properties of an allocation principle,
then find the principles that satisfy them — paralleling how coherent *measures* of risk were
defined, and assuming all risk measures are coherent (sub-additive, monotone, positively
homogeneous, translation invariant), as discussed in
[coherent risk measures](../risk-measures/coherent-risk-measures.md). An allocation principle
$\Pi$ maps each problem $(N,\rho)$ to an allocation with $\sum_i K_i=\rho(X)$ (full allocation),
and is **coherent** if it satisfies: *no undercut* — $\sum_{i\in M}K_i\le\rho(\sum_{i\in M}X_i)$
for every coalition $M$; *symmetry* — components with identical marginal contributions receive
equal capital; *riskless allocation* — a riskless portfolio $\alpha r_f$ is allocated exactly
$-\alpha$.

Modeled as a cooperative game with cost function $c(S)=\rho(\sum_{i\in S}X_i)$, the no-undercut
allocations are exactly those in the game's *core*, and the core of a game whose cost function
comes from a coherent risk measure is non-empty (via the Bondareva-Shapley theorem, using both
sub-additivity and homogeneity). The Shapley value gives a coherent principle but needs $c$ on
all $2^n$ coalitions; extending to *fractional players* with cost $r(\lambda)=\rho(\sum_i(\lambda_i/\Lambda_i)X_i)$,
the **Aumann-Shapley value** — cost per unit $k_i^{AS}=\int_0^1\partial r/\partial\lambda_i(\gamma\Lambda)\,d\gamma$
— collapses, since $r$ is degree-one homogeneous, to the **gradient** $k^{AS}=\nabla r(\Lambda)$.
Aubin's result identifies the fuzzy core with the sub-differential $\partial r(\Lambda)$,
reducing to the single vector $\nabla r(\Lambda)$ when $r$ is differentiable, so for a coherent,
differentiable cost function the Aumann-Shapley value is a coherent fuzzy value — the unique
coherent *linear* allocation principle, and the reason $\nabla r(\Lambda)$ is called the **Euler
principle**. This converges fully with Tasche's RORAC characterization; when the measure is
Expected Shortfall the coherent contribution is itself of shortfall type,
$K_i=\mathrm{E}[-X_i\mid\sum_i X_i\le q_\alpha]$. Non-negativity remains unresolved: a component
may legitimately have negative allocated capital, problematic only when used in a
return/allocated-capital RAPM quotient.

## Risk budgets and limits: existence and uniqueness (Bruder-Roncalli)

Assigning budgets to components and realizing them is the inverse of attribution: from given
contributions to the portfolio that produces them. For a coherent convex measure the Euler
decomposition holds, so the risk contribution is
$\mathrm{RC}_i(x)=x_i\,\partial\mathcal R(x)/\partial x_i$. The **risk-budgeting (RB) portfolio**
for budgets $\{b_1,\dots,b_n\}$ is defined by $\mathrm{RC}_i(x)=b_i$ — a *nonlinear* system,
unlike the weight-budgeting portfolio $x_i=b_i$. With volatility as the measure,
$\mathrm{RC}_i(x)=x_i(\Sigma x)_i/\sqrt{x^\top\Sigma x}$ summing to $\sigma(x)$; in a Gaussian
world VaR and ES differ only by a multiplicative constant, so relative contributions coincide.
The managerial specification imposes relative, long-only, normalized weights and budgets:
$$ x_i\cdot(\Sigma x)_i = b_i\cdot(x^\top\Sigma x),\quad b_i\ge 0,\ x_i\ge 0,\quad \sum_i b_i=1,\ \sum_i x_i=1. $$
Null budgets are pathological — $\mathrm{RC}_k=0$ admits $x_k=0$ but also an $x_k>0$ solution
requiring negative correlations — so one imposes $b_i>0$ and pre-reduces the universe; with null
budgets up to $2^m$ solutions arise, the acceptable one being the perturbation limit
$b_i=\varepsilon_i\to 0$.

**Existence and uniqueness.** Reformulating as the convex program
$$ y^\star = \arg\min\sqrt{y^\top\Sigma y}\quad\text{s.t.}\quad\sum_i b_i\ln y_i\ge c,\ y\ge 0, $$
then normalizing $x_i^\star=y_i^\star/\sum_i y_i^\star$, the log constraint and objective are both
convex, so the RB portfolio **exists and is unique** provided $\Sigma$ is positive definite and
$b_i>0$; the KKT conditions return $y_i\,\partial\sigma/\partial y_i=\lambda_c b_i$
(contributions proportional to budgets), with the multiplier strictly positive because the
constraint is necessarily active — convex-optimization machinery covered in
[convex optimization](../optimization/convex-optimization.md). Parametrically, the RB volatility
lies between the two polar cases $c=-\infty$ (minimum variance) and
$c=\sum_i b_i\ln b_i$ (weight budgeting):
$$ \sigma_{\mathrm{mv}} \le \sigma_{\mathrm{rb}} \le \sigma_{\mathrm{wb}}. $$

**Optimality and beta interpretation.** Like the ERC portfolio, risk budgeting is heuristic but
connects to optimality: for the quadratic utility $x^\top\mu-\phi\,x^\top\Sigma x$, the
performance contribution $\mathrm{PC}_i=x_i\tilde\mu_i=(2/\phi)x_i(\Sigma x)_i\propto b_i$, so
specifying risk budgets also fixes attributed expected performance — welding Bruder-Roncalli's
analysis to Qian's optimal case. With $\beta_i=(\Sigma x)_i/(x^\top\Sigma x)$ and
$\mathrm{RC}_i=x_i\beta_i\sigma(x)$, the condition $\mathrm{RC}_i=b_i$ gives
$x_i=b_i\beta_i^{-1}/\sum_j b_j\beta_j^{-1}$: the weight is inversely proportional to beta
(endogenously, since $\beta_i$ depends on the portfolio). In ex-ante monitoring, controlling
$\mathrm{RC}_i=x_i\beta_i\sigma(x)$ jointly controls each component's weight, beta and volatility
relative to total risk.
