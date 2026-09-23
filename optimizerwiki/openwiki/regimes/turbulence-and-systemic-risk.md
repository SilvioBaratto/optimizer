---
type: concept
title: "Turbulence, Absorption Ratio and Systemic Risk"
description: Observable market-state indicators built from the statistical structure of returns — Mahalanobis financial turbulence and its magnitude/correlation-surprise decomposition, the absorption ratio and effective rank from covariance eigenvalues as measures of market compactness and fragility, and rising correlations as a signal of systemic risk.
tags: [turbulence, mahalanobis, absorption-ratio, effective-rank, systemic-risk, principal-components, correlation-surprise, market-fragility]
sources:
  - id: openwiki-source-5844574d459d5a6aa5aa4642
    resource: repo://docs/28_turbulence_absorption_ratio_and_systemic_risk.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Turbulence, Absorption Ratio and Systemic Risk

Mean-variance selection requires estimates of expected returns, standard deviations, and
correlations, typically drawn from equally weighted historical data. This approach, while
giving reasonable risk estimates over the whole investment horizon, likely understates
variances and correlations during periods of stress or financial crisis: in such periods asset
returns tend to become more volatile and more strongly correlated, so the diversification that
characterizes the sample on average vanishes exactly when it is most needed. This chapter
measures how unusual or fragile the market is from the statistical structure of returns alone —
a problem sharpened after 2008, when securitization, private transactions, complexity, and
flexible accounting made the explicit links between institutions unobservable, forcing reliance
on implicit measures inferred from prices. Two ideas organize it: a return vector can be judged
usual or unusual by its distance from typical behavior (the Mahalanobis distance, giving
financial turbulence), and market fragility can be read in the compactness of the covariance
structure (principal-component analysis, giving the absorption ratio). These provide an
observable measure of the state that [market regimes and views](./market-regimes-and-views.md)
models as a latent regime, the same quadratic form that bounds
[stress-testing scenarios](../risk-management/stress-testing-and-scenarios.md), and rest on the
PCA also used in [factor models](../factor-models/factor-models.md).

## The generalized Mahalanobis distance

Mahalanobis measured the distance between two normal statistical populations in $P$ variates.
If they share the same dispersion, differ only in means, and the $P$ variates are independent,
the distance is $P\cdot\Delta^2 = \sum_{i=1}^{P}(\alpha_i-\alpha'_i)^2/\alpha_{ii}$, weighting
each mean gap by the inverse of the corresponding variance so that gaps along highly dispersed
directions count less. It generalizes to $P$ correlated variates: with dispersion matrix
$\alpha_{ij}=\sigma_i\sigma_j\rho_{ij}$ and $\alpha^{ij}$ the inverse-matrix element,
$$ P\cdot\Delta^2 = \sum_{i,j=1}^{P}\alpha^{ij}(\alpha_i-\alpha'_i)(\alpha_j-\alpha'_j), $$
a quadratic form that is the exact statistical analog of the line element
$ds^2=g_{\mu\nu}dx^\mu dx^\nu$: the inverse dispersion plays the role of the metric tensor.
Defined in population parameters, the distance is free of sampling fluctuations and is made a
statistic by substituting sample estimates for the parameters.

## Financial turbulence and period classification

Applied to finance, the distance measures how unusual a vector of contemporaneous returns is
relative to its historical distribution. A univariate outlier is a return outside a confidence
interval; a multivariate outlier is a collectively unusual set of returns — either a single
return far from its mean or a normally correlated pair diverging enough to make the period
unusual — so it can arise from one asset's unusual performance or from an unusual interaction
among assets none of which is unusual alone. The distance of an observed vector from the
multivariate mean is
$$ d_t = (y_t - \mu)\,\Sigma^{-1}\,(y_t - \mu)', $$
which for two uncorrelated series reduces to the equation of an ellipse (a circle when
variances are equal). For $n$ normal series $d_t$ is distributed $\chi^2$ with $n$ degrees of
freedom; defining an outlier as an observation in the outer 25% of the distribution, for 12
series the tolerance boundary is a $\chi^2$ score of 14.84. Such episodes need not coincide
with low-return periods: stress is an *unusual* period, not merely a loss period. As a measure
of financial turbulence — atypical price behavior, decoupling of correlated assets and
convergence of uncorrelated ones — it spikes in recognizable stress periods of elevated
volatility and broken correlations, is linked to performance (a wide variety of risk-premium
returns are significantly lower in turbulent periods), and is persistent (turbulent episodes
cluster and do not dissolve immediately).

## Portfolios with distinct covariance matrices for the two states

Because the covariance matrix estimated in unusual periods differs substantially from the calm
one, portfolios can be built accounting separately for the two sets. On monthly returns of
eight asset classes (Jan 1988–Sep 1998), the outer-25% criterion flags 27 of 129 months as
outliers: for the outlier sample the average standard deviation rises to 18.27% vs. 11.67% for
the full sample (a 57% increase) and average correlation to 13.72% vs. 11.94%. Excluding
commodities, average correlation rises 36% (16.86% to 22.96%), while commodity average
correlation falls fivefold ($-2.81\%$ to $-13.99\%$): full-sample correlations mask both the
weaker diversification of financial assets in stress and the stronger diversification of
commodities in turmoil. Focusing only on stress would give overly conservative portfolios, so
returns are modeled as a mixture of an inner and an outlier distribution, the covariance being
$$ \Sigma = p\,\Sigma_i + (1-p)\,\Sigma_o. $$
Substituting into expected utility gives
$EU(R_p) = w'\mu - \lambda(p\,w'\Sigma_i w + (1-p)\,w'\Sigma_o w)$, and specifying separate
inner and outlier risk aversions $\lambda_i,\lambda_o$ rescaled to sum to two lets a blended
covariance $\Sigma^* = \lambda'_i p\,\Sigma_i + \lambda'_o(1-p)\Sigma_o$ recover the standard
form $EU(R_p)=w'\mu-\lambda(w'\Sigma^* w)$. This keeps two distinct pieces of information
separate — a *forecast* of each distribution's likelihood ($p$) and a *behavioral* risk-
aversion attitude ($\lambda$) — and nests the original mean-variance model. Empirically, the
full-sample optimal portfolio's standard deviation rises nearly 70% (7.27% to 12.32%) when
subjected to the outlier covariance, so the preferred solution is a compromise from the blended
covariances.

## Decomposition: magnitude surprise and correlation surprise

Turbulence extends by separating two components. Dividing the distance by the (constant) number
of assets $n$, $d_t = (y_t-\mu)\Sigma^{-1}(y_t-\mu)'/n$, gives a multivariate z-score capturing
both how far risk-adjusted magnitudes differ from historical means and how incoherent their
interaction is with the historical correlation matrix. The *magnitude surprise* is the
correlation-blind turbulence, computed by zeroing all off-diagonal covariance elements; the
*correlation surprise* is the ratio
$$ \text{correlation surprise} = \frac{\text{turbulence}}{\text{magnitude surprise}}. $$
For one asset turbulence is just the squared z-score $(x/\sigma_x)^2 = z_x^2$ and there is no
correlation surprise. For two assets, using the 2×2 inverse identity,
$$ \text{CS} = \frac{1}{1-\rho^2}\left(1 - \frac{\rho\,z_x z_y}{\tfrac{1}{2}(z_x^2 + z_y^2)}\right), $$
with all magnitude units cancelling: correlation surprise carries only the multivariate
*direction* of co-movement, like a compass. For given $\rho$ it ranges from
$(1-|\rho|)/(1-\rho^2)$ to $(1+|\rho|)/(1-\rho^2)$, both equal to one at $\rho=0$ (no structural
relation makes any co-movement pattern more unusual). A value above one signals correlation
breaks (correlated assets diverging, negatively correlated ones converging); below one, typical
outcomes. Correlation surprise is orthogonal to volatility and carries incremental forward-
looking information: controlling for volatility, high-correlation-surprise periods average
higher risk and lower returns. Across US stocks, European stocks, and currencies, contemporaneous
correlation and magnitude surprise are negatively correlated — the most volatile days show more
typical correlations — but the next day the pattern reverses: elevated volatility with atypical
correlations foreshadows higher next-day volatility than elevated volatility with typical ones.

## The absorption ratio as a fragility measure

The second family starts from PCA of the covariance matrix: the first eigenvector explains the
largest fraction of total asset variance, the second (orthogonal) the largest fraction of the
residual, and so on. The interest is not in interpreting the risk sources but in measuring how
compact they are. The **absorption ratio** is the fraction of total variance absorbed by a
fixed number of eigenvectors,
$$ \text{AR} = \frac{\sum_{i=1}^{n}\sigma_{E_i}^2}{\sum_{j=1}^{N}\sigma_{A_j}^2}, $$
with $N$ assets and $n$ eigenvectors. A high AR corresponds to high systemic risk because risk
sources are more unified; a low AR indicates more disparate sources. A high AR does not
necessarily cause depreciation or turbulence — it is an indication of *fragility*, since a shock
is more likely to propagate quickly and widely when risk sources are tightly coupled. AR differs
from average correlation, which ignores assets' relative contributions: in one example, raising
the correlation of two high-volatility assets and lowering that of two low-volatility assets
drops average correlation slightly (0.36 to 0.32) while AR rises sharply (0.55 to 0.80).

Empirically, a 500-day rolling window is used, the eigenvector count fixed at about 1/5 of the
assets, and variances exponentially weighted with a 250-day half-life. On 51 US industries (10
eigenvectors), AR is inversely associated with the equity index level and reaches its all-time
high in the 2008 crisis. Standardizing the shift as
$$ \Delta\text{AR} = \frac{\text{AR}_{15\text{d}} - \text{AR}_{1\text{yr}}}{\sigma}, $$
all of the worst 1% of monthly drawdowns were preceded by a one-standard-deviation AR spike, and
a high fraction of other significant drawdowns followed such spikes — a spike is a nearly
necessary but not sufficient condition for a significant drawdown. Equity returns are much lower
after AR spikes than after marked drops, and a strategy reducing equity exposure after a rise and
increasing it after a drop improves risk-adjusted return. The global AR over 42 countries
oscillates in a 65–85% range and rises in major crises (the October 1997 Hong Kong attack, the
August 1998 Russian and LTCM collapses); the median standardized shift begins rising about 40
days before turbulent episodes, suggesting use as a precursor. AR summarizes much of what more
complex structural contagion models produce — correlating 81% with a structural model's
covariance-change measure.

## Effective dimensionality: the effective rank

A complementary compactness measure is the effective dimensionality — how many independent
directions are actually active. Given a matrix $A$ with singular values
$\sigma_1\ge\cdots\ge\sigma_Q\ge 0$, define the normalized distribution
$p_k = \sigma_k/\lVert\sigma\rVert_1$ and the **effective rank** as the exponential of its
Shannon entropy,
$$ \operatorname{erank}(A) = \exp\{H(p_1,\dots,p_Q)\}, \qquad H = -\sum_{k=1}^{Q}p_k\log p_k, $$
with $0\log 0 = 0$. It satisfies $1 \le \operatorname{erank}(A) \le \operatorname{rank}(A) \le Q$,
equals 1 iff one singular value is nonzero, and equals the rank iff the singular values are
uniform. Being real-valued, it quantifies an "effective dimension": a strongly correlated 2-D
Gaussian has rank two but concentrates energy on one direction, so its spectral entropy nears
zero and effective rank is just above one; for a 4×4 Hermitian circulant it is maximal at
$\rho=0$ and decreases with $|\rho|$ while the rank is unchanged. Since any $A$ associates with
$\sqrt{AA^*}$ sharing its singular values, the effective rank applies directly to a covariance
matrix from its normalized eigenvalues, giving the average number of significant dimensions in
the universe.

## Rising correlations as a systemic-risk indicator

Correlations intensify in stress and can serve as a systemic-risk indicator. Studying 10 US and
10 European sector indices with rolling-window PCA, the rate of increase of the first principal
component over short 12-month windows is proposed as the indicator: systemic risk rises when the
largest eigenvalue explains most of the data variation. Defining the change
$\Delta PC1(t) = PC1(t) - PC1(t-m)$ on monthly data, for the ten US sectors the variance explained
by the first eigenvalue rises almost monotonically from about the start of 2007, with PC1
capturing about 60% of variability (41% in early 2007 to 86% in 2011). With 12-month windows and
$m=1$, the largest PC1 increase falls in August 2007 — when the interbank market froze, before the
global recession that began December 2007 and before markets declined. Longer windows push the
peak forward (saturating around 20 months); the distribution of past PC1 changes is asymmetric,
its right tail associated with the largest systemic-risk increases. After the August 2007 peak a
rapid return decline follows. For US financial-sector indices the largest increase falls in July
2007; for European and ex-US developed sectors it falls in February 2008, months later, reflecting
the time for the crisis to propagate from the US to Europe.

## Temporal evolution of market correlations

Combining PCA and random-matrix theory on $N=98$ financial products (1999–2010), the empirical
correlation matrix $\mathbf{R} = \frac{1}{T}\hat{\mathbf{Z}}\hat{\mathbf{Z}}^T$ is rolled over
$T=100$-observation windows (475 matrices). Comparing the eigenvalue distribution with the
random-matrix (Wishart) prediction with edges
$\gamma_\pm = \sigma^2(1 + 1/Q \pm 2\sqrt{1/Q})$, $Q=T/N$, many market eigenvalues exceed the upper
edge $\gamma_+$, so the matrices carry structure incompatible with random price changes. The
variance explained by the $k$-th component equals the normalized eigenvalue $\beta_k/N$. The first
component's explained variance rises from 2001 to 2010, jumping when the Lehman-collapse week
(September 2008) enters the window, and in the NBER recession-declaration week the first component
explained nearly 40%. In 2001 the first twelve components explained about 65% of variance; by 2010
the first five sufficed — few components characterize market correlations, far fewer than $N$. To
distinguish few-asset from market-wide correlation increases, the inverse participation ratio
$I_k = \sum_i [\omega_{ki}]^4$ (participation ratio $1/I_k$) is used: identical contributions give
$I_k=1/N$, a single nonzero component gives $I_k=1$. The first component's participation ratio rises
from 2001 to 2010, jumping after Lehman as it moves from a localized state (mainly bonds
contributing) to an extended one (nearly all assets strongly correlated). A market-wide correlation
increase makes cross-class diversification much harder, while within-class increases matter less;
the number of significant components (by the Kaiser-Guttman $\beta > 1/N$ criterion and versus
random data) decreases from 2001 to 2010, confirming markets became more correlated and describable
by far fewer than $N$ components.
