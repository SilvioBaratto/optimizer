---
type: concept
title: "Factor Models"
description: How linear factor structure reduces the O(N²) mean-variance input problem to O(N) or O(NK) by routing co-movement through a few common factors — Sharpe's diagonal/single-index model, K-factor covariance decomposition, OLS estimation, and the equilibrium (CAPM) and no-arbitrage (APT) return-beta relations, with empirical tests and regularization.
tags: [factor-models, single-index, capm, apt, fama-french, covariance, beta, pca, smart-beta, ridge-lasso]
sources:
  - id: openwiki-source-e9b1def8cfbf39ffa874119e
    resource: repo://docs/05_factor_models.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Factor Models

Markowitz [mean-variance selection](../foundations/mean-variance-selection.md)
requires an $O(N^2)$ set of inputs — a mean and variance per asset plus a
correlation for every pair — which is costly to estimate and often
**ill-conditioned**. Factor models cut this to $O(N)$ or $O(NK)$ by not modeling
correlations directly but the *sources* of co-movement: assets move together
because a few **common factors** drive them.

## The cost of the full covariance and ill-conditioning

The number of parameters is $N(N+3)/2$, of order $O(N^2)$: 150 assets need 11,475
inputs, 350 assets exceed 60,000. Beyond estimation burden, the covariance can be
**ill-conditioned**. This is not abstract — three assets with pairwise correlations
$\rho_{12}=0.9,\rho_{13}=0.9,\rho_{23}=-0.9$ give a correlation matrix with negative
determinant (not positive semidefinite), and weights $w=(-1,1,1)$ produce a
*negative* portfolio variance $-0.1405$, economic nonsense. Correlations estimated
one pair at a time do not guarantee a jointly valid covariance. Factor models
sidestep this by imposing structure.

## Sharpe's diagonal model and the single-index model

Sharpe's **diagonal model** writes $R_i=A_i+B_iI+C_i$ with a single common factor
$I$ and uncorrelated idiosyncratic errors, giving the key result that the
covariance between any two assets flows *only* through their factor sensitivities:
$\mathrm{Cov}(R_i,R_j)=B_iB_jQ_{N+1}$. This needs $3N+2$ parameters versus
Markowitz's $(N^2+3N)/2$ — already under half at $N=10$, and just 1.7% at $N=350$.

The **single-index model** takes the factor to be the market:
$r_i^e=\alpha_i+\beta_i r_m^e+e_i$, a *statistical* (not equilibrium)
representation. Its moments give the compact covariance

$$\Sigma=\sigma_m^2\,\beta\beta'+\mathrm{diag}(\sigma_{e_1}^2,\dots,\sigma_{e_N}^2),$$

a rank-1 market term plus diagonal specific risk — $3N+1$ parameters, $O(N)$. It
also explains **diversification**: for an equal-weighted portfolio the
idiosyncratic variance $\tfrac1N\bar\sigma^2(e)\to0$ as $N\to\infty$, leaving only
the non-diversifiable **systematic risk** $\beta_p^2\sigma_m^2$.

### Centered factors

A factor is **centered** if $E[f]=0$. Centering changes the intercept but not
$\beta_i$ (covariance is translation-invariant), and isolates the *unexpected*
component (a shock) that risk premia compensate. Traded factors like $r_m-r_f$ are
often left *uncentered* so the return relation $E[r_i^e]=\alpha_i+\beta_iE[r_m-r_f]$
exposes the market risk premium directly instead of absorbing it into the intercept.

## K-factor models and covariance decomposition

With $K$ factors, $\mathbf r^e=\alpha+B\mathbf f+\mathbf e$ and the covariance
decomposes as

$$\Sigma=B\Sigma_f B'+\Sigma_e,\qquad \Sigma_e=\mathrm{diag}(\sigma_{e_i}^2).$$

The systematic term $B\Sigma_f B'$ has rank at most $K$ — systematic risk lives in a
$K$-dimensional subspace — while $\Sigma_e$ is diagonal because idiosyncratic risk
generates no cross-asset covariance. Portfolio betas are weighted averages of asset
betas ($\beta_{pk}=\sum_i w_i\beta_{ik}$). At $N=500,K=5$ the model needs ~3,500
parameters versus 125,750 for the full covariance.

## Specifying and extracting factors

Factors can be **economic** — macroeconomic (industrial production, expected/
unexpected inflation, term and default spreads) or **characteristic-based** (the
Fama-French $SMB$ size and $HML$ value factors) — or **statistical**, extracted by
**PCA**: diagonalizing the sample covariance $\Sigma=V\Lambda V'$, keeping the top
$K$ orthogonal directions of maximal explained variance. The first component
resembles a broad market factor; on an S&P 500 sample it alone explains ~30% of
total variance and the first five ~50% cumulatively. The trade-off is statistical
efficiency (PCA) versus economic interpretability (economic factors).

## OLS estimation of betas

Betas are estimated by OLS, $\hat\theta_i=(X'X)^{-1}X'\mathbf r_i^e$. The classical
assumptions are linearity, **exogeneity** $E[\epsilon\mid X]=0$, i.i.d. sampling,
finite fourth moments, homoskedasticity, and no perfect collinearity. Exogeneity is
critical: an omitted correlated variable or a mismeasured regressor induces
endogeneity — the latter causing **attenuation bias** toward zero. Under the
assumptions the **Gauss-Markov theorem** makes OLS the Best Linear Unbiased
Estimator (BLUE). The **Frisch-Waugh-Lovell theorem** shows a multifactor $\hat
\beta_1$ measures the $Y$–$X_1$ relation *net of* the other factors. Empirically
betas contain error and regress toward 1 over time; the **Blume correction**
$\beta_{i,2}=0.67\,\beta_{i,1}+0.33$ is a shrinkage toward the mean beta.

## The CAPM: equilibrium, CML, SML

The **CAPM** imposes general equilibrium: under homogeneous expectations every
investor holds the same tangency portfolio, which must equal the **market
portfolio** (cap-weighted) to clear markets. The **capital market line** is the
efficient frontier through the market and has the maximum Sharpe ratio — the
theoretical basis for passive indexing — with market price of risk $E(r_m)-r_f=\bar
A\sigma_m^2$. Equating each asset's premium-to-variance contribution yields the
**fundamental CAPM equation**

$$E(r_s)-r_f=\beta_s\,[E(r_m)-r_f],\qquad \beta_s=\frac{\sigma_{sm}}{\sigma_m^2},$$

so **only systematic risk is priced**, not total volatility. Graphically this is the
**security market line (SML)**, which unlike the CML applies to every asset and
relates return to $\beta$ alone. Deviations ($\alpha\neq0$) open the door to active
management, though mutual-fund alphas are on average slightly negative and
insignificant net of costs. Extensions include the **zero-beta CAPM** (no risk-free
asset, flatter SML) and the **ICAPM** (extra hedging factors).

## The APT: no-arbitrage, well-diversified portfolios

The **APT** needs neither mean-variance nor equilibrium — only **no arbitrage**, a
linear factor structure, and enough assets to diversify idiosyncratic risk. For a
**well-diversified** portfolio the specific risk vanishes, so two such portfolios
with the same beta cannot carry different premia without creating a riskless,
zero-net-investment profit. Imposing $\alpha_p=0$ recovers the *same* SML as the
CAPM but **without its restrictive assumptions and without the true market
portfolio**. Generalizing to $K$ factors, a **replicating portfolio** forces the
**multifactor SML**

$$E(r_i)=r_f+\sum_{k=1}^K\beta_{ki}\lambda_k,$$

where the $\lambda_k$ are factor **risk prices** and a **factor portfolio** has unit
exposure to one factor and zero to the rest ($\lambda_k=E(r_k^e)$). The APT applies
to well-diversified portfolios, so isolated single-asset mispricings are permitted.

## Empirical tests: from CAPM to the factor zoo

Linear factor models are tested with **time-series** regressions (traded factor:
the restriction is simply $\alpha_i=0$, premium $\hat\lambda=E(f)$) and
**cross-sectional** regressions, notably **Fama-MacBeth** (a cross-sectional
regression each period, averaged over time with standard errors from the temporal
variation). Portfolios sorted on a characteristic serve as test assets to reduce
errors-in-variables attenuation. The first failure was the **small-firm effect**
(small caps beat their market beta). The **Roll critique** notes the unobservable
market portfolio makes tests joint tests of model and proxy. The **Fama-French
three-factor model** adds $SMB$ and $HML$ to capture the **value premium** the CAPM
misses, reaching $R^2$ of 90–95% and readable as an APT implementation. Momentum,
liquidity risk, and macro factors followed, producing the **"factor zoo"** with its
data-mining and multiple-testing concerns and post-publication anomaly decay.

## Efficient frontier with factor models and smart beta

Imposing factor structure barely moves the frontier: on five assets, the full
Markowitz, single-index, and three-factor tangency Sharpe ratios are ~35.30%,
34.46%, 34.73% — nearly identical, because factor structure captures most systematic
co-movement while adding robustness; accumulated estimation error in the full
covariance can even make it *worse* than the factor model. **Smart beta** strategies
exploit this by deliberately, transparently tilting $\beta_p=w'B$ toward priced
factors (value, size, momentum), e.g. $\max_w \beta_{p,k}$ subject to $\mathbf1'w=1$,
or a factor-structured mean-variance program.

## Appendix: ridge and LASSO regularization

With many correlated candidate factors, **regularization** trades a little bias for
a large variance reduction (bias²+variance+irreducible error decomposition), and
requires **standardizing** the data. **Ridge** ($L_2$ penalty
$\lambda\sum\beta_k^2$) shrinks all coefficients proportionally toward zero without
zeroing them — ideal when many correlated factors all matter. **LASSO** ($L_1$
penalty $\lambda\sum|\beta_k|$) drives negligible coefficients exactly to zero,
performing **variable selection** — ideal when only a few of many factors truly
count. The Blume beta correction is a special case of this shrinkage principle,
which is the estimation-side counterpart to
[estimation error and shrinkage](../estimation/estimation-error-and-shrinkage.md).
Factor models also underpin [fundamental stock selection](../signals/fundamental-stock-selection.md),
feed [factor timing and rotation](./factor-timing-and-rotation.md), and supply the
prior structure for [Black-Litterman](../foundations/black-litterman.md).
