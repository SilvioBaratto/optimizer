---
type: concept
title: "Mean-Variance Selection"
description: Markowitz mean-variance optimization — the two-asset diversification cases, the general N-asset quadratic program with its closed-form frontier (parabola in variance-mean, hyperbola in sd-mean), the global minimum-variance portfolio, the capital allocation line and Sharpe-maximizing tangency portfolio, Tobin's two-fund separation, and the marginal-risk-contribution decomposition underpinning risk budgeting.
tags: [mean-variance, efficient-frontier, diversification, tangency-portfolio, sharpe-ratio, two-fund-theorem, gmvp, marginal-risk-contribution, covariance]
sources:
  - id: openwiki-source-07522f265f1b6a22eb0cf8c0
    resource: repo://docs/02_mean_variance_selection.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Mean-Variance Selection

Modern portfolio theory begins with Markowitz's mean-variance formulation (developed
independently the same year by Roy and anticipated by de Finetti). Treating each
return as a random variable, one measures profitability by its **expected value** and
risk by its **variance**, and — for a risk-averse investor whose von
Neumann-Morgenstern utility reduces to a function of mean and variance (see
[choice under uncertainty](./choice-under-uncertainty.md)) — utility increases in
expected return and decreases in variance. A portfolio is **mean-variance efficient**
if it has maximum expected return for its variance and minimum variance for its
return; portfolio $i$ dominates $j$ when $E(r_i)\ge E(r_j)$ and $\sigma_i\le\sigma_j$
with one strict, so the preferred direction in the mean-sd plane is "north-west."

The full path has three stages: the risky-only efficient frontier, the frontier with
a risk-free asset and its tangency portfolio, and the choice of the optimal portfolio
by risk aversion — where only the third depends on preferences.

## Returns, covariance, and correlation

Portfolio weights $x_i$ (fractions of total value, negative for short positions) sum
to one. Arithmetic returns are additive across assets, so the expected portfolio
return is linear, $E(R_P)=\sum_i x_i r_i=\mathbf x'\mathbf r$. Variance is **not**
linear:

$$\sigma_P^2=\sum_i x_i^2\sigma_i^2+2\sum_i\sum_{j>i}x_ix_j\sigma_{ij}=\mathbf x'\mathbf V\mathbf x,$$

with covariance $\sigma_{ij}=E(R_iR_j)-E(R_i)E(R_j)$ measuring co-movement. Since
covariance magnitude conflates volatility with the strength of the link, one
normalizes to the **correlation** $\rho_{ij}=\sigma_{ij}/(\sigma_i\sigma_j)\in[-1,1]$.
Same-sector stocks typically correlate more than cross-sector ones. Because true
moments are unobservable, they are estimated from realized returns treated as
equiprobable scenarios, with the $T-1$ degrees-of-freedom correction making variance
and covariance unbiased; data frequency matters for (co)variance accuracy but not for
mean returns, and variances add over time so the standard deviation grows as
$\sqrt{T}$. Arithmetic returns are used for portfolio analysis (additive across
assets) whereas log returns are additive over time.

## The two-asset case and the role of correlation

With two risky assets ($r_1<r_2$, $\sigma_1^2<\sigma_2^2$) and $x_2=1-x_1$, portfolio
volatility falls below the weighted average of individual volatilities whenever
$\rho_{1,2}<1$ — this is the **benefit of diversification**. The three subcases:

- **$\rho_{1,2}=+1$:** $\sigma_P=x_1\sigma_1+(1-x_1)\sigma_2$; the frontier is a
  straight segment with no diversification, and a risk-free mix requires shorting the
  more volatile asset.
- **$\rho_{1,2}=-1$:** $\sigma_P=|x_1\sigma_1-(1-x_1)\sigma_2|$ vanishes at
  $x=(\sigma_2,\sigma_1)/(\sigma_1+\sigma_2)$, so a **perfect hedge** (zero-risk
  portfolio) is achievable with long-only weights; the frontier is two half-lines
  meeting at $\sigma=0$, the upper efficient and the lower dominated.
- **$\rho_{1,2}\in(-1,1)$:** portfolios with $\sigma_P^2<\min\{\sigma_1^2,\sigma_2^2\}$
  exist; as $\rho$ falls the curve bows further left. Correlation has no effect on the
  expected return, which stays linear in the weights.

## The general N-asset problem: convexity and closed form

The base problem minimizes variance for a target return $\pi$ subject to a budget
constraint: $\min_x \mathbf x'\mathbf V\mathbf x$ s.t. $\mathbf x'\mathbf r=\pi$,
$\mathbf x'\mathbf e=1$. Since $\mathbf V$ is positive definite, $\mathbf x'\mathbf
V\mathbf x$ is convex and the two linear constraints define a convex set, so the
problem has a **unique** solution obtained from the first-order conditions. With
$\alpha=\mathbf r'\mathbf V^{-1}\mathbf r$, $\beta=\mathbf r'\mathbf V^{-1}\mathbf e$,
$\gamma=\mathbf e'\mathbf V^{-1}\mathbf e$, the optimum is

$$\mathbf x^*=\frac{(\gamma\mathbf V^{-1}\mathbf r-\beta\mathbf V^{-1}\mathbf e)\pi+(\alpha\mathbf V^{-1}\mathbf e-\beta\mathbf V^{-1}\mathbf r)}{\alpha\gamma-\beta^2},$$

derived via the Lagrangian $\mathfrak L=\mathbf x'\mathbf V\mathbf x-\lambda_1(\mathbf
x'\mathbf r-\pi)-\lambda_2(\mathbf x'\mathbf e-1)$. The optimal variance
$\sigma_{P^*}^2=(\gamma\pi^2-2\beta\pi+\alpha)/(\alpha\gamma-\beta^2)$ traces a
**parabola** in the variance-mean plane and a **hyperbola** in the sd-mean plane. Its
vertex, at $(\sigma^2=1/\gamma, r=\beta/\gamma)$, is the **global minimum-variance
portfolio (GMVP)** — the lowest-variance portfolio overall, whose weights do not
depend on expected returns and which separates the inefficient lower branch from the
efficient upper one. Adding investment opportunities can only improve the frontier
(the old frontier is contained in the new), and individually dominated assets can
still receive positive weight through low or negative correlation — it is an asset's
covariance with the portfolio, not its standalone risk, that sets its incremental
risk. With non-negativity or other constraints the closed form is lost and the
problem is solved numerically.

## Diversification: market risk versus specific risk

For the equal-weighted portfolio ($x_i=1/N$), with average variance $\bar\sigma^2$
and average covariance $\overline{\mathrm{Cov}}$,

$$\sigma_P^2=\frac1N\bar\sigma^2+\frac{N-1}{N}\overline{\mathrm{Cov}}\ \xrightarrow[N\to\infty]{}\ \overline{\mathrm{Cov}}.$$

Thus risk splits into **firm-specific (diversifiable, idiosyncratic)** risk, removable
by diversification, and **market (systematic, non-diversifiable)** risk, which
persists. If risks were independent the equal-weighted volatility would be
$\sigma/\sqrt n\to0$, but real equity returns are positively correlated: with a
typical 40% volatility and 25% correlation the equal-weighted volatility converges to
$\sqrt{0.25\times0.40\times0.40}=20\%$, not zero, and most of the benefit is achieved
with about thirty stocks (the reduction from one to two assets far exceeds that from
100 to 101). This is why diversification is a "free lunch" — it cuts risk without
sacrificing expected return.

## The risk-free asset: CAL, Sharpe ratio, and tangency

The risk-free asset has zero return variance and is uncorrelated with all others.
Mixing a fraction $x$ into a risky portfolio $P$ gives $E(R_{xP})=r_f+x(E(R_P)-r_f)$
and $SD(R_{xP})=x\,SD(R_P)$, so combinations lie on a straight **capital allocation
line (CAL)** $E(R_C)=r_f+\frac{E(R_P)-r_f}{\sigma_P}\sigma_C$; $x>1$ means borrowing at
$r_f$ (a levered portfolio). The CAL slope is the **Sharpe ratio**
$SR_P=(E(R_P)-r_f)/\sigma_P$, reward per unit of volatility. The **tangency portfolio**
generates the steepest CAL tangent to the risky frontier and has the maximum Sharpe
ratio in the economy; once the risk-free asset is introduced, the efficient frontier
is no longer the Markowitz hyperbola's upper branch but the half-line from $r_f$
through the tangency portfolio, which dominates the hyperbola at every risk level.

## The two-fund separation theorem

The **two mutual fund theorem** states that any efficient-frontier portfolio is a
linear combination of any two efficient portfolios on that frontier: given two
minimum-variance portfolios with different expected returns, every minimum-variance
portfolio is their linear combination and vice versa, with combination weight
$\alpha=(E(r_c)-E(r_b))/(E(r_a)-E(r_b))$. The sharpest consequence: taking the two
funds to be the risk-free asset and the tangency portfolio, **all investors hold the
same risky tangency portfolio regardless of risk aversion**, and the only subjective
choice is the split between the risk-free asset and tangency — the more risk-averse
allocate less to tangency, the less risk-averse possibly lever up. Selection thus
separates into a purely technical stage (find tangency) and a preference-dependent
stage (allocate the complete portfolio), a result due to Tobin extending Markowitz to
the risk-free case; substituting $y=\sigma_C/\sigma_T$ recovers the efficient-frontier
line $E(R_C)=r_f+SR_T\,\sigma_C$.

## Different lending and borrowing rates

The CAL assumes lending and borrowing at the same rate. In practice the borrowing
rate exceeds the lending rate — e.g. lend at $r_f=7\%$ but borrow at $r_f^B=9\%$ —
producing two CAL segments of different slope that break at the fully-invested risky
point ($y=1$). There is then no single tangency portfolio: one tangency for the
lending segment (built against $r_f$) and one for the borrowing segment (against
$r_f^B$), so the efficient frontier has three parts — a lending half-line, a stretch
of the risky-only frontier between the two tangency portfolios, and a borrowing
half-line.

## Non-negativity constraints: the frontier as a bounded arc

Short sales extend the risky-only frontier beyond the extreme assets. Imposing
non-negativity ($\mathbf w\ge0$) makes the problem
$\min_w\sigma_p^2$ s.t. $\sum w_i=1$, $\mathbf w\ge0$, $\sum w_iE(r_i)=\bar\mu_p$,
which generally has no closed form. Under non-negativity the efficient frontier does
not extend indefinitely but becomes a **bounded arc** reaching the individual assets
as endpoints, unable to exceed the maximum-expected-return asset above or drop below
the feasible-set GMVP.

## Volatility decomposition and marginal risk contribution

Portfolio variance is a weighted average of each asset's covariance with the
portfolio, $\mathrm{Var}(R_P)=\sum_i x_i\,\mathrm{Cov}(R_i,R_P)$, so $\mathrm{Cov}(R_i,R_P)$
is the **marginal risk contribution** — the marginal variance increase from a small
increment in asset $i$ financed by shorting the risk-free asset. Dividing by
portfolio sd gives the volatility decomposition

$$SD(R_P)=\sum_i x_i\,SD(R_i)\,\mathrm{Corr}(R_i,R_P),$$

which is strictly less than $\sum_i x_i SD(R_i)$ unless all correlations equal 1 —
part of the volatility is diversified away. Increasing an asset's weight (financed at
$r_f$) improves the Sharpe ratio iff $\frac{E(R_i)-r_f}{SD(R_i)}>\mathrm{Corr}(R_i,R_P)\frac{E(R_P)-r_f}{SD(R_P)}$;
defining the portfolio-beta $\beta_i^P=SD(R_i)\mathrm{Corr}(R_i,R_P)/SD(R_P)$, this is
$E(R_i)>r_f+\beta_i^P(E(R_P)-r_f)$, and a portfolio is efficient iff
$E(R_i)=r_f+\beta_i^{\text{eff}}(E(R_{\text{eff}})-r_f)$ holds for every asset. Since
risk contributions split as $x_i\beta_i^P$, this decomposition underpins **risk
budgeting** — apportioning a risk budget across assets by marginal contribution.

## Connections

Mean-variance selection feeds the estimation-error treatment in
[estimation error and shrinkage](../estimation/estimation-error-and-shrinkage.md),
its equilibrium/tangency logic is the prior for
[Black-Litterman](./black-litterman.md), the risk-contribution decomposition is the
seed of [risk budgeting and risk parity](../optimization/risk-budgeting-and-risk-parity.md),
and the constrained numerical form is treated under
[convex optimization](../optimization/convex-optimization.md).
