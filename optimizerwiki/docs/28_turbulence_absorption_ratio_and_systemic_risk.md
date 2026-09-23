---
title: "Turbulence, Absorption Ratio, and Systemic Risk"
chapter: 28
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-24
---

> [!abstract] Summary
> The chapter builds a set of observable indicators of market state from the statistical structure of returns. From the Mahalanobis distance a measure of financial turbulence is derived, which classifies periods as usual or unusual and decomposes into a magnitude component and a correlation-surprise component; from the eigenvalues of the covariance matrix are derived the absorption ratio, a measure of compactness and hence of market fragility, and the effective rank, a measure of its dimensionality. It is further established, on empirical grounds, that correlations among assets intensify during stress phases and that their temporal evolution signals rising systemic risk.

## Market State as an Object of Measurement

The mean-variance selection process introduced by [Markowitz1952] requires estimates of expected returns, standard deviations, and correlations; with this information the assets are combined so that, for a given level of expected return, the portfolio offers the lowest possible risk, measured as standard deviation or variance. Analysts typically estimate variances and correlations from equally weighted historical data, treated as representative of a sample. As [ChowEtAl1999] observe, this approach, while providing reasonable risk estimates over the whole investment horizon, likely understates variances and correlations during periods of stress or financial crisis: in such periods asset returns tend to become more volatile and more strongly correlated, so that the diversification that characterizes the sample on average vanishes precisely when it is needed most.

This chapter addresses the problem of measuring how unusual or fragile the market's state is, based solely on the statistical structure of returns. This is recognized as a central problem: the failure of oversight and the excessive risk-taking by some financial institutions pushed the system to the brink of systemic collapse in 2008, and both regulators and investors have since become keenly interested in developing tools to monitor systemic risk [KritzmanLiPageRigobon2011]. Securitization, private transactions, complexity, and "flexible" accounting, however, prevent the many explicit links among financial institutions from being observed directly; one must therefore fall back on implicit measures, inferred from price behavior.

Two ideas organize the chapter. The first is that a vector of returns can be judged usual or unusual based on its distance from typical behavior, measured taking into account the dispersion and correlation among the variables: this is the Mahalanobis distance, from which the notion of financial turbulence and its decomposition follow. The second is that market fragility can be read in the compactness of its covariance structure — how few components suffice to explain its variance — through principal component analysis. These indicators provide an observable measure of the state that the chapter [[21 Regimi di mercato e formazione delle view]] models as a latent regime; the Mahalanobis distance is the same quadratic form that delimits the plausibility regions of the stress-testing scenarios of the chapter [[24 Stress testing e scenari]]; and the absorption ratio rests on the principal component analysis of the chapter [[05 Modelli fattoriali]].

## The Generalized Mahalanobis Distance

[Mahalanobis1936] addresses the problem of measuring the distance between two statistical normal populations in $P$ variates. If the two populations share the same dispersion and differ only in mean values, and if the $P$ variates are independent, the distance is given by the statistic

$$ P\cdot\Delta^2 = \sum_{i=1}^{P} \frac{(\alpha_i - \alpha'_i)^2}{\alpha_{ii}}, $$

where $\alpha_i$ and $\alpha'_i$ are the mean values of the two populations and $\alpha_{ii} = \sigma_i^2$ their respective variances. Each discrepancy between the means is thus weighted by the inverse of the variance of the corresponding variate, so that differences along highly dispersed directions count less than those along less dispersed directions.

The formula generalizes to $P$ correlated variates. Defining the fundamental dispersion matrix $\alpha$ with elements $\alpha_{ij} = \sigma_i\sigma_j\rho_{ij}$, where $\rho_{ij}$ is the correlation coefficient between the $i$-th and $j$-th variates, and denoting by $\alpha^{ij}$ the cofactor of $\alpha_{ij}$ divided by the determinant of $\alpha$ — that is, the element of the inverse matrix — the generalized distance is written

$$ P\cdot\Delta^2 = \sum_{i,j=1}^{P} \alpha^{ij}\,(\alpha_i - \alpha'_i)(\alpha_j - \alpha'_j). $$

Adopting the summation convention over repeated indices and setting $d\alpha_\mu = \alpha_\mu - \alpha'_\mu$, the same quantity becomes the quadratic form $P\cdot\Delta^2 = \alpha^{\mu\nu}\,d\alpha_\mu\,d\alpha_\nu$. [Mahalanobis1936] notes that this expression is the exact statistical analogue of the line element $ds^2 = g_{\mu\nu}\,dx^\mu\,dx^\nu$: the inverse dispersion matrix plays in statistics the same role as the metric tensor. The distance is defined in terms of population parameters and is therefore not subject to sampling fluctuations; in practice, it is converted into a statistical equation by substituting sample statistics for the parameters.

## Financial Turbulence and the Classification of Periods

The financial application of the Mahalanobis distance consists in measuring how unusual a vector of contemporaneous returns is relative to its historical distribution. An outlier for a single return series is simple to identify: it is a return that falls outside a chosen confidence interval around the expected value. A multivariate outlier is harder to spot, because it represents a set of returns that is collectively unusual for one or more reasons: it may be that one of the returns is sufficiently far from its own mean, or that a pair of returns normally strongly correlated exhibits a difference marked enough to make the period unusual. A multivariate outlier can thus result as much from the unusual performance of a single asset as from the unusual interaction of a set of assets, none of which need be unusual in isolation [ChowEtAl1999].

Graphically, for two uncorrelated series of equal variance the boundary between usual observations and outliers is a circle centered on the mean; with different variances it becomes an ellipse with horizontal and vertical axes; with non-zero correlation the ellipse rotates, with the tilt reflecting the sign of the correlation. Beyond three series visualization is no longer possible and matrix algebra is required. The distance of an observed vector from the multivariate mean is

$$ d_t = (y_t - \mu)\,\Sigma^{-1}\,(y_t - \mu)', $$

where $y_t$ is the vector of returns for period $t$, $\mu$ the vector of means, and $\Sigma$ the covariance matrix of the series [ChowEtAl1999]. For two uncorrelated series the formula reduces to

$$ d_t = \frac{(y - \mu_y)^2}{\sigma_y^2} + \frac{(x - \mu_x)^2}{\sigma_x^2}, $$

the equation of an ellipse, which degenerates into a circle when the variances are equal. In the general case of $n$ normal series, $d_t$ is distributed as a $\chi^2$ with $n$ degrees of freedom; defining an outlier as an observation that falls beyond the outer 25% of the distribution, for $12$ series the tolerance boundary corresponds to a $\chi^2$ score of $14{,}84$, and a vector is classified as an outlier if its score exceeds it.

The episodes so identified do not necessarily coincide with periods of low or negative returns: stress is defined as an unusual period, not as a period of losses only. [KritzmanLi2010] adopt this statistic as a measure of financial turbulence, a condition in which asset prices behave atypically relative to their historical pattern — extreme price movements, decoupling of correlated assets, and convergence of uncorrelated assets. Turbulence tends to spike during recognizable periods of market stress characterized by heightened volatility and correlation breaks; it is linked to performance, since on average returns of a wide variety of risk premia are significantly lower during turbulent periods; and it is persistent, since turbulent episodes tend to cluster in time and do not dissolve immediately after appearing [KinlawTurkington2014].

## Portfolios with Distinct Covariance Matrices for the Two States

Because the covariance matrix estimated during unusual periods differs substantially from the one estimated during calm periods, [ChowEtAl1999] propose constructing portfolios that separately account for the two sets of observations. The empirical evidence is obtained from monthly returns of eight asset classes (domestic, foreign, and emerging equities; domestic, foreign, and high-yield bonds; commodities and cash) over the period January 1988 – September 1998. Applying the outer-25% criterion, $27$ of the $129$ months turn out to be outliers. For the outlier sample the average standard deviation rises to $18{,}27\%$ against $11{,}67\%$ for the full sample, an increase of $57\%$; the average correlation rises to $13{,}72\%$ against $11{,}94\%$. This aggregate figure, however, hides important detail: excluding commodities, the average correlation increases by $36\%$, from $16{,}86\%$ to $22{,}96\%$, while the average correlation of commodities falls by a factor of $5$, from $-2{,}81\%$ to $-13{,}99\%$. The correlations of the full sample thus mask the weaker diversification properties of financial assets during stress periods, while at the same time understating the diversification benefits of commodities when markets are in turmoil.

It would, however, be shortsighted to focus only on stress periods, since that would lead to unduly conservative portfolios. Investors care simultaneously about risk in calm periods and risk in stress periods. [ChowEtAl1999] model returns as a mixture of two distributions, an "inner" one and an "outlier" one, the former with probability $p$, so that the covariance of returns is written

$$ \Sigma = p\,\Sigma_i + (1-p)\,\Sigma_o. $$

Substituting this mixed covariance into the expected utility function of a portfolio with weight vector $w$ gives

$$ EU(R_p) = w'\mu - \lambda\big(\,p\,w'\Sigma_i w + (1-p)\,w'\Sigma_o w\,\big), $$

where $\lambda$ is overall risk aversion. To let the investor vary risk aversion between the two states, an inner risk aversion $\lambda_i$ and an outlier risk aversion $\lambda_o$ are specified and rescaled so that they sum to two, $\lambda'_i = 2\lambda_i/(\lambda_i+\lambda_o)$ and $\lambda'_o = 2\lambda_o/(\lambda_i+\lambda_o)$, giving

$$ EU(R_p) = w'\mu - \lambda\big(\lambda'_i\,p\,w'\Sigma_i w + \lambda'_o\,(1-p)\,w'\Sigma_o w\big). $$

Defining an overall covariance $\Sigma^* = \lambda'_i\,p\,\Sigma_i + \lambda'_o\,(1-p)\,\Sigma_o$ the objective function is rewritten in the standard form $EU(R_p) = w'\mu - \lambda\,(w'\Sigma^* w)$. The substitutions transparently incorporate two distinct pieces of information: a forecast of the likelihood of each distribution over the coming investment period, through $p$, and a behavioral parameter of risk aversion, through the $\lambda$'s. It is important to keep the two elements separate, since one is a forecast and the other an attitude. The framework nests the original mean-variance model: the full-sample objective function is recovered by setting $p$ equal to the empirical frequency of inner observations and keeping risk aversion equal across the two states. Empirically, the portfolio optimized on the full sample sees its standard deviation rise by nearly $70\%$, from $7{,}27\%$ to $12{,}32\%$, when subjected to the outlier covariance; re-optimizing for the outlier environment reduces volatility in that environment but lowers the expected return, so that the preferable solution is a compromise obtained from the mixed covariances.

## Decomposition: Magnitude Surprise and Correlation Surprise

[KinlawTurkington2014] extend the turbulence measure by separating it into two components. They adopt the definition of [KritzmanLi2010], dividing it by the number of assets $n$ (constant over time) to aid interpretation:

$$ d_t = (y_t - \mu)\,\Sigma^{-1}\,(y_t - \mu)'/n. $$

This statistic can be thought of as a multivariate z-score: it measures the statistical unusualness of a contemporaneous cross-section of returns relative to their historical distribution, capturing both the extent to which the risk-adjusted magnitudes of returns differ from their historical means and the extent to which their interaction is inconsistent with the historical correlation matrix.

To isolate the two components, one first computes the magnitude surprise: the "correlation-blind" turbulence score, obtained by setting all off-diagonal elements of the covariance matrix to zero. This measure captures surprises in magnitude but ignores whether the co-movement is typical or atypical. The correlation surprise is the ratio of full turbulence to magnitude surprise,

$$ \text{correlation surprise} = \frac{\text{turbulence}}{\text{magnitude surprise}}. $$

For a single asset, turbulence is simply the squared z-score, $x(\sigma_x^2)^{-1}x = (x/\sigma_x)^2 = z_x^2$; by definition a single asset cannot exhibit any correlation surprise. For two assets, turbulence normalizes by the whole covariance matrix,

$$ (x\ \ y)\begin{pmatrix}\sigma_x^2 & \rho\sigma_x\sigma_y \\ \rho\sigma_x\sigma_y & \sigma_y^2\end{pmatrix}^{-1}\begin{pmatrix}x \\ y\end{pmatrix}, $$

assuming zero means for simplicity. Using the identity $\left(\begin{smallmatrix}a&b\\c&d\end{smallmatrix}\right)^{-1} = \frac{1}{ad-bc}\left(\begin{smallmatrix}d&-b\\-c&a\end{smallmatrix}\right)$ and carrying out the products, one finds, as shown in the appendix to [KinlawTurkington2014],

$$ \text{CS} = \frac{\sigma_x^2\sigma_y^2}{(1-\rho^2)\sigma_x^2\sigma_y^2}\cdot\frac{x^2\sigma_y^2 + y^2\sigma_x^2 - 2xy\rho\sigma_x\sigma_y}{x^2\sigma_y^2 + y^2\sigma_x^2} = \frac{1}{1-\rho^2}\left(1 - \frac{\rho\,z_x z_y}{\tfrac{1}{2}(z_x^2 + z_y^2)}\right), $$

where $z_x$ and $z_y$ are the z-scores normalized by volatility and $\rho$ the historical correlation. All magnitude units cancel out: the correlation surprise contains only information about the multivariate direction of co-movement, analogous to a compass or a radial coordinate. For a given $\rho$ the correlation surprise has minimum $(1-|\rho|)/(1-\rho^2)$ and maximum $(1+|\rho|)/(1-\rho^2)$; for $\rho = 0$ both minimum and maximum equal one, since without a structural expected relation no co-movement pattern is more or less unusual than another.

A correlation surprise value greater than one is associated with correlation breaks — previously correlated assets diverging, or previously negatively correlated assets converging; a value less than one is associated with relatively typical correlation outcomes. Correlation surprise is orthogonal to volatility and contains incremental forward-looking information: controlling for volatility, periods characterized by correlation surprise tend on average to higher risk and lower returns. Across three universes (U.S. equities, European equities, and currencies), on a contemporaneous basis correlation surprise and magnitude surprise are negatively correlated — the most volatile days tend to exhibit more typical correlations — but on the following day the pattern reverses: heightened volatility accompanied by atypical correlations foreshadows higher volatility the next day than heightened volatility accompanied by typical correlations. The decomposition of the Mahalanobis distance thus makes it possible to analyze the intertemporal relation between correlation surprise and magnitude surprise.

## The Absorption Ratio as a Measure of Fragility

The second family of indicators starts from the principal component analysis of the covariance matrix. Given a covariance matrix of returns estimated over a given period, the first eigenvector is the linear combination of weights that explains the largest fraction of the total variance of the assets; the second eigenvector is orthogonal to the first and explains the largest fraction of residual variance, that is, the variance not explained or absorbed by the first eigenvector; subsequent eigenvectors are identified in the same way. These eigenvectors may or may not be associated with observable economic variables: the interest here is not to interpret the sources of risk, but to measure how compact they are becoming [KritzmanLiPageRigobon2011].

The measure used as an indicator of systemic risk is the absorption ratio, defined as the fraction of the total variance of a set of assets explained or absorbed by a fixed number of eigenvectors:

$$ \text{AR} = \frac{\sum_{i=1}^{n} \sigma_{E_i}^2}{\sum_{j=1}^{N} \sigma_{A_j}^2}, $$

where $N$ is the number of assets, $n$ the number of eigenvectors in the numerator, $\sigma_{E_i}^2$ the variance of the $i$-th eigenvector, and $\sigma_{A_j}^2$ the variance of the $j$-th asset. A high absorption ratio value corresponds to a high level of systemic risk, because it implies that the sources of risk are more unified; a low value indicates more disparate sources of risk. A high absorption ratio does not necessarily lead to asset depreciation or turbulence: it is simply an indication of market fragility, in the sense that a shock is more likely to propagate quickly and widely when the sources of risk are tightly coupled.

The absorption ratio differs from average correlation, which might appear an equivalent measure of market compactness but is not, because it does not account for the relative importance of individual assets' contributions to systemic risk. [KritzmanLiPageRigobon2011] illustrate the point with an example in which the correlation of two relatively high-volatility assets increases and that of two relatively low-volatility assets decreases: average correlation falls slightly from $0{,}36$ to $0{,}32$, while the absorption ratio rises sharply from $0{,}55$ to $0{,}80$.

In the empirical estimation a moving window of $500$ days is used for the covariance matrix and eigenvectors, the number of eigenvectors is fixed at about $1/5$ of the number of assets, and the variances $\sigma_{E_i}^2$ and $\sigma_{A_j}^2$ are computed with exponential weighting with a half-life equal to half the window, $250$ days, so that the market's memory of past events fades gradually. Applied to the returns of $51$ U.S. industries — hence $10$ eigenvectors — the absorption ratio shows a clear inverse association with the level of the equity index and rises to its all-time highest value during the 2008 financial crisis, coinciding with a steep decline in prices.

To assess the relation with drawdowns, a standardized version of the index deviation is constructed, defined as

$$ \Delta\text{AR} = \frac{\text{AR}_{15\text{d}} - \text{AR}_{1\text{yr}}}{\sigma}, $$

where $\text{AR}_{15\text{d}}$ is the 15-day moving average, $\text{AR}_{1\text{yr}}$ the one-year moving average, and $\sigma$ the one-year standard deviation of the absorption ratio. All of the worst monthly drawdowns (the top $1\%$) were preceded by a one-standard-deviation spike in the absorption ratio, and a high percentage of other significant drawdowns occurred after such spikes. A spike in the absorption ratio is thus an almost necessary, but not sufficient, condition for a significant drawdown. Moreover, on average, equity returns are much lower after spikes in the absorption ratio than after its marked declines; a strategy that reduces equity exposure after an increase and raises it after a decline in the index improves both return and risk-adjusted return. The global absorption ratio, computed on equity returns from $42$ countries, oscillates in a range of $65$–$85\%$ and rises in coincidence with major crises, such as the speculative attack on Hong Kong in October 1997 and the Russian and LTCM crashes of August 1998. Before turbulent episodes, the median standardized deviation of the absorption ratio begins to rise about $40$ days in advance and continues to rise throughout the episode, which suggests its use as a precursor. The absorption ratio summarizes a large fraction of the information produced by more complex and computationally intensive structural models of financial contagion: its value turns out to be $81\%$ correlated with the moving average of the average covariance change derived from such a structural model.

## Effective Dimensionality: The Effective Rank

The absorption ratio quantifies how much variance is absorbed by a fixed number of components. A complementary measure of compactness is the effective dimensionality of an asset universe: how many independent directions are in fact active in the covariance structure. [RoyVetterli2007] propose the effective rank for this purpose. Given a matrix $A$ of dimension $M\times N$ with singular value decomposition $A = UDV$ and singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_Q \ge 0$ with $Q = \min\{M,N\}$, one defines the distribution of normalized singular values

$$ p_k = \frac{\sigma_k}{\lVert\sigma\rVert_1}, \qquad k = 1,2,\dots,Q, $$

with $\lVert\sigma\rVert_1 = \sum_{k=1}^{Q}|\sigma_k|$. The effective rank is the exponential of the Shannon entropy of this distribution,

$$ \operatorname{erank}(A) = \exp\{H(p_1,p_2,\dots,p_Q)\}, \qquad H(p_1,\dots,p_Q) = -\sum_{k=1}^{Q} p_k \log p_k, $$

with the convention $0\log 0 = 0$. The effective rank satisfies $1 \le \operatorname{erank}(A) \le \operatorname{rank}(A) \le Q$; $\operatorname{erank}(A) = 1$ holds if and only if only one singular value is non-zero, and $\operatorname{erank}(A) = \operatorname{rank}(A)$ if and only if the distribution of singular values is uniform over its support. While rank is an integer quantity, the effective rank can take real values in the interval $[1,Q]$, and for this reason it quantifies an "effective dimension." The typical example is that of a two-dimensional Gaussian vector with strongly correlated components: its covariance matrix has rank two, but the distribution concentrates its energy along the direction of a single singular value, so that the spectral entropy approaches zero and the effective rank turns out to be only slightly above one. Similarly, for a $4\times 4$ Hermitian circulant matrix defined by a correlation parameter $\rho$, the effective rank is maximal for $\rho = 0$ and decreases as $|\rho|$ increases, while the rank remains unchanged. Since every matrix $A$ is associated with the positive semi-definite Hermitian matrix $\sqrt{AA^*}$ having the same singular values, and these singular values correspond to eigenvalues when the matrix is positive semi-definite, the effective rank applies directly to a covariance matrix from its normalized eigenvalues. Its operational meaning, borrowed from Campbell's coefficient rate, is that of the average number of significant dimensions in the representation of the universe under consideration.

## Rising Correlations as an Indicator of Systemic Risk

The evidence that correlations among securities intensify during stress phases and can serve as an indicator of systemic risk is documented by [ZhengEtAl2012]. Studying $10$ sector indices of the U.S. economy and $10$ of the European economy, the authors apply principal component analysis over rolling windows and propose the rate of increase of the first principal component, computed over short $12$-month time windows, as a systemic risk indicator. The underlying idea is that, applying principal component analysis, systemic risk grows when the largest eigenvalue explains a large share of the variation in the data: the more the first component $PC1$ explains the variance, the greater the systemic risk.

The temporal dynamics are captured by defining the change in the first component

$$ \Delta PC1(t) = PC1(t) - PC1(t-m), $$

where, since the data are monthly, $m$ is the number of months over which the change is measured. For the returns of the ten U.S. sectors, the proportion of variance explained by the first eigenvalue begins to increase in an almost monotonic way around the beginning of 2007, and $PC1$ captures about $60\%$ of the variability, ranging from $41\%$ at the start of 2007 to $86\%$ in 2011. With $12$-month windows and $m = 1$, the largest increase in $PC1$ falls in August 2007, the month in which the interbank market froze, before the global recession that began in December 2007 and before markets started to decline. The choice of window size $n$ affects the time coordinate at which the largest increase in cross-correlations is expected: with longer windows large shocks get overwhelmed by other signals, and the peak shifts forward until it saturates for $n$ around $20$ months. The distribution of past changes in $PC1$ has an asymmetric functional form, with the right tail of rare values associated with the largest increase in systemic risk. Comparing monthly changes in $PC1$ with future annual returns of the index, after the largest $PC1$ spike of August 2007 a rapid drop in return is observed, consistent with the idea that a higher level of systemic risk makes a crisis more likely in the near future. The same authors validate the approach on multiple data sets. On U.S. financial sector indices the largest increase in $PC1$ falls in July 2007, ahead of the recession and the global financial crisis. The check is then extended to sectors of the European economy and to developed-market sectors excluding the United States, for which the largest increase in $PC1$ falls in February 2008: both dates turn out to be a few months later than the U.S. market result, reflecting the time needed for the crisis to spread from the United States to Europe.

## Temporal Evolution of Market Correlations

[FennEtAl2011] investigate the temporal evolution of market correlations by combining principal component analysis and random matrix theory on $N = 98$ financial products — equity indices of developed and emerging markets, corporate and government bonds, currencies, metals, fuels, and other commodities — over the period 1999–2010. From standardized weekly log returns $\hat z_i(t)$ the empirical correlation matrix $\mathbf{R} = \frac{1}{T}\hat{\mathbf{Z}}\hat{\mathbf{Z}}^T$ is constructed, and it is rolled over windows of $T = 100$ observations shifted one point at a time, yielding $475$ correlation matrices. Comparing the distribution of eigenvalues with that predicted by random matrix theory for uncorrelated series — the Wishart distribution with density

$$ \rho(\gamma) = \frac{Q}{2\pi\sigma^2(\hat{\mathbf{Z}})}\frac{\sqrt{(\gamma_+ - \gamma)(\gamma - \gamma_-)}}{\gamma}, \qquad \gamma_{\pm} = \sigma^2(\hat{\mathbf{Z}})\left(1 + \frac{1}{Q} \pm 2\sqrt{\frac{1}{Q}}\right), $$

where $Q = T/N$ — it is found that many market eigenvalues exceed the upper bound $\gamma_+$, which implies that the correlation matrices contain structure incompatible with random price variations.

The proportion of total variance explained by the $k$-th principal component equals the normalized eigenvalue,

$$ \frac{\sigma^2(\mathbf{y}_k)}{\sum_{i=1}^{N}\sigma^2(\hat z_i)} = \frac{\beta_k}{\beta_1 + \dots + \beta_N} = \frac{\beta_k}{N}. $$

The fraction of variance explained by the first component increases from 2001 to 2010, with a marked jump when the week of the Lehman Brothers bankruptcy of September 2008 enters the window; in the week the National Bureau of Economic Research officially declared the recession, the first component came to explain nearly $40\%$ of the variance. The large variance explained by a single component implies substantial common variation across markets and signals their close ties; in 2001 the first twelve components explained about $65\%$ of the variance, while in 2010 the first five sufficed for the same proportion. Few components thus suffice to characterize market correlations, and their number is much smaller than $N$.

To distinguish whether the increase in variance explained by a component derives from rising correlations among a few assets or from an effect extending to the whole market, the inverse participation ratio of the $k$-th component is used,

$$ I_k = \sum_{i=1}^{N} [\omega_{ki}]^4, $$

where $\omega_{ki}$ are the eigenvector coefficients, and the participation ratio is defined as $1/I_k$. An eigenvector with identical contributions $\omega_{ki} = 1/\sqrt{N}$ from all $N$ assets has $I_k = 1/N$; an eigenvector with only one non-zero component has $I_k = 1$. A high participation ratio indicates that many assets contribute to the component. The participation ratio of the first component grows from 2001 to 2010, with a marked increase coinciding with the turmoil following the Lehman Brothers collapse; after the bankruptcy, the first component moves from a localized state in which only bonds contributed significantly to an extended state in which almost all assets — equities, currencies, metals, fuels, other commodities, and some government bonds — are strongly correlated with it. The distinction has relevant financial implications: an increase in correlations extending across the whole market makes it much harder to reduce risk by diversifying across different asset classes, while increases in correlation internal to a single class have a smaller impact on diversification. The number of significant components, estimated both with the Kaiser-Guttman criterion (a component is significant if its eigenvalue satisfies $\beta > 1/N$) and by comparing eigenvalue profiles with those of random data, decreases from 2001 to 2010, confirming that markets have become more correlated and can be described by far fewer than $N$ components.

## References

- **[ChowEtAl1999]** Chow, G., Jacquier, E., Kritzman, M., & Lowry, K. (1999). Optimal Portfolios in Good Times and Bad. Financial Analysts Journal, 55(3), 65–73 (Revere Street Group Working Paper No. 272-1).
- **[FennEtAl2011]** Fenn, D. J., Porter, M. A., Williams, S., McDonald, M., Johnson, N. F., & Jones, N. S. (2011). Temporal Evolution of Financial Market Correlations. arXiv:1011.3225v2 [q-fin.ST].
- **[KinlawTurkington2014]** Kinlaw, W., & Turkington, D. (2014). Correlation Surprise. Journal of Asset Management, 14(6), 385–399.
- **[KritzmanLi2010]** Kritzman, M., & Li, Y. (2010). Skulls, Financial Turbulence, and Risk Management. Financial Analysts Journal, 66(5), 30–41.
- **[KritzmanLiPageRigobon2011]** Kritzman, M., Li, Y., Page, S., & Rigobon, R. (2011). Principal Components as a Measure of Systemic Risk. The Journal of Portfolio Management, 37(4), 112–126.
- **[Mahalanobis1936]** Mahalanobis, P. C. (1936). On the Generalised Distance in Statistics. Proceedings of the National Institute of Sciences of India, 2(1), 49–55.
- **[Markowitz1952]** Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77–91.
- **[RoyVetterli2007]** Roy, O., & Vetterli, M. (2007). The Effective Rank: A Measure of Effective Dimensionality. Proceedings of the 15th European Signal Processing Conference (EUSIPCO), Poznań, 606–610.
- **[ZhengEtAl2012]** Zheng, Z., Podobnik, B., Feng, L., & Li, B. (2012). Changes in Cross-Correlations as an Indicator for Systemic Risk. Scientific Reports, 2, 888.
