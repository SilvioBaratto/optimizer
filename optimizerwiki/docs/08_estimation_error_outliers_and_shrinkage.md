---
title: "Estimation Error, Outliers, and Shrinkage"
chapter: 8
tags:
  - optimizer
  - tipo/capitolo
grounded: false
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter shows that expected returns estimated from historical data are affected by an estimation error so large that they remain unusable even with decades of observations, and that mean-variance optimization, having to invert the covariance matrix, amplifies this imprecision. It then presents three remedies with a common statistical root — winsorization of outliers, Bayesian shrinkage of beta, and linear and nonlinear shrinkage of the covariance matrix — united by the principle that shrinking a noisy estimate toward a structured target reduces its mean squared error.

## The Unreliability of Historical Expected Returns

The starting point of portfolio selection is the distribution of future returns, of which, however, we possess only a single sample: the sequence of realized returns. If the distribution of past returns and that of future returns coincide, and if investors are on average neither too optimistic nor too pessimistic, the realized average return converges to the expected return, and we can use the historical average to estimate it [BerkDeMarzo2019]. The arithmetic mean of annual returns for years $1$ through $T$ is

$$\overline{R} = \frac{1}{T}\sum_{t=1}^{T} R_t,$$

and the variance is estimated as the average of the squared deviations from the sample mean,

$$\widehat{\mathrm{Var}}(R) = \frac{1}{T-1}\sum_{t=1}^{T} (R_t - \overline{R})^2 .$$

But the sample mean is only an *estimate* of the true expected return, and as such it is subject to estimation error [BerkDeMarzo2019]. The measure of this error is the **standard error**, that is, the standard deviation of the sample mean around the true value. If the return distribution is identical every year and returns are independent over time, the standard error is obtained from the standard deviation of the individual return divided by the square root of the number of observations:

$$\mathrm{SE}\big(\text{mean}\big) = \frac{\mathrm{SD}\big(\text{individual risk}\big)}{\sqrt{\text{number of observations}}} .$$

Since the realized average falls within two standard errors of the true expected value about $95\%$ of the time, the $95\%$ confidence interval for the expected return is

$$\overline{R} \pm \big(2 \times \mathrm{SE}\big) .$$

Applied to the S&P 500 index between $1926$ and $2014$ — average return $12{,}0\%$, volatility $20{,}1\%$, eighty-nine annual observations — this gives

$$12{,}0\% \pm 2\left(\frac{20{,}1\%}{\sqrt{89}}\right) = 12{,}0\% \pm 4{,}3\%,$$

that is, an interval ranging from $7{,}7\%$ to $16{,}3\%$: even with eighty-nine years of data the expected return is not estimated precisely [BerkDeMarzo2019]. With shorter windows the situation worsens sharply. Over just the years $2002$–$2014$ of the S&P 500 (mean $8{,}7\%$, volatility $19{,}5\%$, thirteen observations) the standard error is $19{,}5\%/\sqrt{13} = 5{,}4\%$ and the $95\%$ interval runs from $-2{,}1\%$ to $19{,}5\%$: with only a few years of data one cannot even establish the sign of the expected return [BerkDeMarzo2019].

Individual securities tend to be even more volatile than large portfolios, and many have existed for only a few years, offering little basis for estimation [BerkDeMarzo2019]. For Cisco stock, with an annualized volatility of $37\%$ and fifteen years of monthly data, the standard error of the expected return is $37\%/\sqrt{15} = 9{,}6\%$, giving a $95\%$ confidence interval of $3{,}3\% \pm 19\%$; even with a hundred years of data the bounds would still be $\pm 7{,}4\%$ [BerkDeMarzo2019].

The standard error formula clarifies how much data would be needed for useful precision. Setting the half-width of the interval $h = 2\sigma/\sqrt{T}$ and inverting, the number of years required is $T = (2\sigma/h)^2$. With $\sigma = 20{,}1\%$, achieving a half-width of $2\%$ would require about $(40{,}2/2)^2 \approx 404$ years, and a half-width of $1\%$ about $1616$ years. This is why the realized average of individual securities is not a reliable estimate of their expected return, and a different method is needed, one that relies on more robust statistical estimates [BerkDeMarzo2019].

## The Consequence for Mean-Variance Optimization

Mean-variance selection, described in [[02 Selezione media-varianza]], requires two inputs: the vector of expected returns and the covariance matrix of returns [Markowitz1952]. The first, as we have seen, is estimated with an error so large that it does not even allow distinguishing positive from negative expected returns over short windows [BerkDeMarzo2019]. The second enters the problem through its own inverse.

Selecting an efficient mean-variance portfolio from a large universe of securities is, among the applications requiring estimation of a covariance matrix and its inverse, the one with matrix dimension large relative to sample size [LedoitWolf2004]. The usual estimator, the sample covariance matrix, performs poorly in this regime: when the dimension $p$ exceeds the number of observations $n$ it is not even invertible, and when the ratio $p/n$ is below one but not negligible the sample matrix is invertible but numerically **ill-conditioned**, meaning that inverting it drastically amplifies the estimation error [LedoitWolf2004]. For large $p$ it is difficult to collect enough observations to make $p/n$ negligible, and it is therefore important to have a well-conditioned estimator for large-dimensional covariance matrices [LedoitWolf2004].

This outlines the reason why optimal weights inherit and magnify the imprecision of the inputs: on one hand they depend on expected returns that the historical average estimates with great uncertainty, and on the other on the inverse of an ill-conditioned covariance matrix, which amplifies the error contained in the sample estimate. It is partly to bypass the first difficulty that, instead of directly estimating the expected returns of individual securities, one prefers to go through the risk-return relationship: a security's beta can be inferred from historical data with reasonable accuracy even with just two years of data, whereas the historical average return remains unreliable [BerkDeMarzo2019]. The following sections instead address the second difficulty, and more generally the problem of how to make the estimates that feed optimization less noisy.

## Outliers and Winsorization

The empirical estimates that enter optimization are sensitive to **outliers**, that is, to observations of unusually large magnitude. In beta estimation via linear regression this sensitivity is pronounced [BerkDeMarzo2019]. The example of Genentech stock, with monthly returns for $2002$–$2004$, is instructive: two observations show extreme movements — a drop of nearly $30\%$ in April $2002$ and a rise of nearly $65\%$ in May $2003$ — both a reaction to specific announcements about the outcome of clinical trials, and hence firm-specific risk rather than market risk. Since they occurred in months when the market also moved in the same direction, they bias the beta estimate upward: the standard regression returns $1{,}21$, but replacing the returns of those two months with the average return of similar biotechnology firms gives $0{,}60$, likely a much more accurate assessment of the true market risk over that period [BerkDeMarzo2019].

This treatment — replacing the extreme value rather than discarding it — is the logic of **winsorization**. Its theoretical basis is estimation from censored normal samples [Dixon1960]. A sample is censored when one or more observations at the extremes are missing, with the number and positions of the missing ones known. Censoring can arise naturally, when the magnitude of an observation is known only to be more extreme than the others, or it can be imposed by the experimenter who, from past experience, knows that extreme observations are so unreliable that their magnitude should not be used as observed [Dixon1960].

Winsor proposed that, for the magnitude of an extreme observation that is poorly known or unknown, one should use the magnitude of the next-largest (or next-smallest) observation: the censored extreme is not eliminated, but replaced by the value of its surviving neighbor [Dixon1960]. Dixon shows that, when the symmetry of the censoring is maintained (or an appropriate correction is applied), this "winsorized" practice produces mean estimators whose efficiency is barely distinguishable from that of the best linear estimators [Dixon1960]. For symmetric censoring, with $i$ observations censored at each extreme of an ordered sample $x_1 \le x_2 \le \dots \le x_N$, the mean estimator is

$$m_W = \frac{(i+1)x_{i+1} + x_{i+2} + \dots + x_{N-i-1} + (i+1)x_{N-i}}{N} ,$$

where the double weights $(i+1)$ on the two innermost ordered values still observed reflect the fact that each of them also stands in for the censored neighbor [Dixon1960]. On a sample of ten observations with central values $\dots, 108, 111, 119, 121, 125, \dots$, while the best linear estimator gives $118{,}9$, the winsorized formula gives

$$m_W = \frac{4(111) + 119 + 121 + 4(125)}{10} = 118{,}4 .$$

The efficiencies tabulated by Dixon never fall below $0{,}962$ for estimating the mean relative to the best linear estimator, and remain at or above $0{,}965$ for estimating the standard deviation obtained from suitable combinations of ranges [Dixon1960]. Winsorization thus caps extreme values — replacing them with their less extreme counterpart — rather than discarding them, and pays a negligible efficiency cost when the data truly come from a normal distribution.

## Shrinkage as a Principle

Winsorization, beta adjustment, and covariance matrix regularization are particular expressions of a single principle, **shrinkage**: shrinking a noisy estimate toward a structured target reduces its mean squared error. The general principle traces back to [Stein1956] and to the technique of [JamesStein1961].

The justification is the decomposition of mean squared error into variance and squared bias. For an estimator $\Sigma^*$ of a true quantity $\Sigma$, with the appropriate quadratic norm,

$$E\big[\|\Sigma^* - \Sigma\|^2\big] = E\big[\|\Sigma^* - E[\Sigma^*]\|^2\big] + \|E[\Sigma^*] - \Sigma\|^2 ,$$

that is, mean squared error equals variance plus squared bias [LedoitWolf2004]. The structured target has all bias and no variance; the raw sample estimate has all variance and no bias; the combination of the two represents the optimal trade-off between the two types of error, and it is precisely the idea of trading off bias against variance already central to the original James-Stein shrinkage technique [LedoitWolf2004].

A concrete and transparent form of the same principle is the Bayesian combination of two signals. If a raw estimate $b$ has precision $1/s_b^2$ and a prior piece of information indicates a value $b'$ with precision $1/s_{b'}^2$, the combined estimate is the precision-weighted average,

$$b'' = \frac{b'/s_{b'}^2 + b/s_b^2}{1/s_{b'}^2 + 1/s_b^2}, \qquad \frac{1}{s_{b''}^2} = \frac{1}{s_{b'}^2} + \frac{1}{s_b^2} ,$$

where the precision of the combined estimate is the sum of the two precisions [Vasicek1973]. The raw estimate is thus pulled toward the target $b'$ to a degree that grows the more imprecise it is. In the following two sections this scheme is applied first to the beta of a single security, then to the entire covariance matrix.

## Shrinkage of Beta

Beta is the sensitivity of a security's return to that of the market portfolio, and it is customary to estimate it from past data via least-squares regression [Vasicek1973]. Writing the linear process $y_t = \alpha + \beta x_t + e_t$ for the excess returns of the security $y_t$ and of the market $x_t$, the least-squares estimate is

$$b = \frac{\sum_t (y_t - \bar{y})(x_t - \bar{x})}{\sum_t (x_t - \bar{x})^2},$$

with estimated variance $s_b^2 = s^2/\sum_t (x_t - \bar{x})^2$, where $s_b$ is the standard error of the estimate [Vasicek1973]. This estimate is unbiased, $E(b\,|\,\beta) = \beta$, and of minimum variance in its class [Vasicek1973].

Vasicek observes, however, that unbiasedness does not reflect the desired property. Unbiasedness describes the distribution of the estimate given the true value of the parameter; the real situation is the opposite, since it is the sample estimate that is known, and on this basis — together with any prior information — we want to infer the distribution of the parameter [Vasicek1973]. The relevant prior information is the **cross-sectional** distribution of betas. On the New York market betas cluster around one, with most values between $0{,}5$ and $1{,}5$; a low sample estimate such as $0{,}2$ is therefore more likely the result of underestimation than of overestimation, and taking it as an unbiased estimate is not correct [Vasicek1973].

Assuming for beta an approximately normal prior density with mean $b'$ and variance $s_{b'}^2$ equal to that of the cross-sectional distribution, and an improper prior density for the other parameters, the posterior density of beta is, for $T$ greater than $20$, approximately normal with mean

$$b'' = \frac{b'/s_{b'}^2 + b/s_b^2}{1/s_{b'}^2 + 1/s_b^2}, \qquad s_{b''}^2 = \frac{1}{1/s_{b'}^2 + 1/s_b^2}$$

[Vasicek1973]. The Bayesian estimate $b''$ is interpreted as an adjustment of the sample estimate $b$ toward the best prior estimate $b'$, with a degree of adjustment proportional to the precision $h = 1/s_b^2$ of the sample estimate [Vasicek1973]. For the New York market population the prior parameters are approximately $b' = 1$ and $s_{b'} = 0{,}5$: in this case the regression coefficient estimated from the sample is linearly adjusted toward one, to a degree depending on its standard error $s_b$ [Vasicek1973]. The estimate $b''$, while not unbiased in the sense of sampling theory, is preferable because it minimizes the *estimation* squared error — the expectation of $(\hat\beta - \beta)^2$ with respect to the posterior distribution — rather than the *sampling* error [Vasicek1973]. As the sample size grows, $s_b^2 \to 0$ and the degree of adjustment vanishes, so that $b''$ remains consistent, $\operatorname{plim} b'' = \beta$ [Vasicek1973].

Applied practice has adopted simplified versions of the same scheme. Merrill Lynch's Security Risk Evaluation service uses the formula

$$b'' = 1 + k(b-1),$$

with $k$ a constant common to all securities, interpretable as the slope of the cross-sectional regression of beta estimates on those from a preceding non-overlapping period; this amounts to assuming the same variance $s_b^2$ for all securities, and has the effect of over-adjusting the more accurate estimates and under-adjusting the less accurate ones [Vasicek1973]. Similarly, data providers compute **adjusted beta** by averaging the estimate with $1{,}0$; Bloomberg's formula is

$$\text{Adjusted Beta} = \tfrac{2}{3}\,\beta_i + \tfrac{1}{3}(1{,}0)$$

[BerkDeMarzo2019]. Two reasons support this adjustment toward one: estimates that are extreme relative to historical or industry norms are suspect because they are largely due to estimation error, and there is evidence that betas tend to regress toward the mean value of $1{,}0$ over time [BerkDeMarzo2019]. This latter tendency was documented empirically by [Blume1971], who showed that estimated betas shift toward one in subsequent periods. For the same need to reduce estimation error, many practitioners prefer to use industry-average betas rather than those of individual securities [BerkDeMarzo2019], a solution that the factor models of [[05 Modelli fattoriali]] make systematic. Vasicek finally notes that, when estimating the beta of a portfolio composed of $N$ securities, the prior variance $s_{b'}^2$ to be used is the cross-sectional dispersion of betas of portfolios of size $N$, which under cross-sectional independence of the residuals is reduced by a factor of $1/\sqrt{N}$ relative to that of individual securities [Vasicek1973].

## Linear Shrinkage of the Covariance Matrix

The same principle, applied to the entire covariance matrix, produces the linear shrinkage estimator of [LedoitWolf2004]. Consider the sample covariance matrix $S = XX'/n$, where $X$ is $p \times n$. To make an estimator well-conditioned at all costs one could impose an arbitrary structure, but in the absence of information about the true structure it is generally misspecified and produces an overly biased estimator [LedoitWolf2004]. The approach followed is the weighted average of the sample matrix and a well-conditioned structured estimator, with the weight chosen optimally.

The structured target requires that all variances be equal and all covariances be zero, that is, the identity matrix scaled by the average variance $\mu = \langle \Sigma, I\rangle$ [LedoitWolf2004]. Measuring distance with the Frobenius norm, one seeks the linear combination $\Sigma^* = \rho_1 I + \rho_2 S$ that minimizes the expected quadratic loss $E[\|\Sigma^* - \Sigma\|^2]$. Introducing the four scalars

$$\mu = \langle \Sigma, I\rangle, \quad \alpha^2 = \|\Sigma - \mu I\|^2, \quad \beta^2 = E[\|S - \Sigma\|^2], \quad \delta^2 = E[\|S - \mu I\|^2],$$

the Pythagorean relation $\alpha^2 + \beta^2 = \delta^2$ holds (since $E[S] = \Sigma$), and the solution to the problem is

$$\Sigma^* = \frac{\beta^2}{\delta^2}\,\mu I + \frac{\alpha^2}{\delta^2}\, S ,$$

with optimal loss $E[\|\Sigma^* - \Sigma\|^2] = \alpha^2\beta^2/\delta^2$ [LedoitWolf2004]. Reparametrizing as $\Sigma^* = \rho\,\mu I + (1-\rho) S$, the weight placed on the target $\mu I$ is $\rho = \beta^2/\delta^2$, the **shrinkage intensity** [LedoitWolf2004]. The percentage relative improvement in average loss (PRIAL) over the sample matrix is

$$\frac{E[\|S - \Sigma\|^2] - E[\|\Sigma^* - \Sigma\|^2]}{E[\|S - \Sigma\|^2]} = \frac{\beta^2}{\delta^2} ,$$

equal to the intensity itself: everything is governed by the ratio $\beta^2/\delta^2$, a normalized measure of the error in the sample matrix $S$. Intuitively, if $S$ is relatively accurate it is not worth shrinking it much, and shrinking it would help little anyway; if $S$ is relatively inaccurate it is worth shrinking it a lot, and there is much to be gained [LedoitWolf2004].

The same result can be read off the eigenvalues. Letting $\lambda_1, \dots, \lambda_p$ be the true eigenvalues and $l_1, \dots, l_p$ the sample ones, their mean coincides with $\mu$, and

$$\frac{1}{p}E\Big[\sum_{i=1}^{p}(l_i - \mu)^2\Big] = \frac{1}{p}\sum_{i=1}^{p}(\lambda_i - \mu)^2 + E[\|S - \Sigma\|^2] :$$

the sample eigenvalues are more spread out around their mean than the true ones, and the excess spread equals the error of the sample matrix [LedoitWolf2004]. This implies that the largest sample eigenvalues are biased upward and the smallest ones downward; the estimate is thus improved by shrinking the eigenvalues toward their mean,

$$\lambda_i^* = \frac{\beta^2}{\delta^2}\,\mu + \frac{\alpha^2}{\delta^2}\, l_i, \qquad i = 1, \dots, p .$$

This is a linear shrinkage of each eigenvalue toward the centroid $\mu$, with **the same intensity** for all of them [LedoitWolf2004]. The underlying cause of the excess dispersion is the error in the sample eigenvectors: since the statistician knows that the eigenvectors associated with extreme eigenvalues are the least reliable, a cautious stance must be maintained on extremely small and large eigenvalues, which the sample matrix does not do [LedoitWolf2004].

The estimator $\Sigma^*$, however, depends on unobservable quantities. Its usable counterpart replaces the four scalars with consistent estimators,

$$S_n^* = \frac{b^2}{d^2}\,m I + \frac{a^2}{d^2}\, S ,$$

where $m = \langle S, I\rangle$ estimates $\mu$, $d^2 = \|S - mI\|^2$ estimates $\delta^2$, $b^2$ estimates $\beta^2$ (truncated to $d^2$), and $a^2 = d^2 - b^2$ estimates $\alpha^2$ [LedoitWolf2004]. Under the framework of *general asymptotics*, in which the number of variables $p$ and the number of observations $n$ both tend to infinity with a bounded ratio $p/n$, $S_n^*$ is consistent and has the same asymptotic expected loss as $\Sigma^*$; it is furthermore the estimator with asymptotically minimal quadratic risk among all linear combinations of the identity and the sample matrix, including those that would use posterior knowledge of the true matrix [LedoitWolf2004]. Unlike $S$, it is always invertible even when $p > n$, and is typically well-conditioned [LedoitWolf2004]. It can be interpreted as an empirical Bayes estimator, in which shrinkage toward the target $\mu I$ plays the role of the prior information [LedoitWolf2004]; it is precisely the type of structured estimate that the optimization discussed in [[03 Limiti della MPT e il modello di Black-Litterman]] needs.

## Nonlinear Shrinkage of Eigenvalues

A look at the equation of [MarcenkoPastur1967], which governs the relationship between sample eigenvalues and population eigenvalues under large-dimensional asymptotics, shows that linear shrinkage is merely the first-order approximation to a problem that is fundamentally nonlinear in nature [LedoitWolf2012]. Linear shrinkage applies the same intensity to all sample eigenvalues, regardless of their position: if the intensity is, say, $0{,}5$, every eigenvalue is moved halfway toward the common mean [LedoitWolf2012]. How good this approximation is depends on the situation: when $p/n$ is small and/or the population eigenvalues are dispersed, higher-order effects become pronounced and linear shrinkage improves little on the sample matrix [LedoitWolf2012].

The remedy is to move to **nonlinear** shrinkage, which applies an individualized shrinkage intensity to each sample eigenvalue [LedoitWolf2012]. The natural class to search within is that of **rotation-equivariant** estimators, for which rotating the data with an orthogonal matrix rotates the estimator in the same way; such estimators share their eigenvectors with the sample matrix and can differ only in their eigenvalues [LedoitWolf2012]. Every rotation-equivariant estimator thus has the form $U_n D_n U_n'$, with $U_n$ the matrix of sample eigenvectors and $D_n = \operatorname{Diag}(d_1, \dots, d_p)$ diagonal [LedoitWolf2012].

Searching within this class for the matrix closest to $\Sigma_n$ in the Frobenius norm, the finite-sample optimal solution has diagonal elements

$$d_i^* = u_i' \, \Sigma_n \, u_i, \qquad i = 1, \dots, p ,$$

which measure how the $i$-th sample eigenvector $u_i$ relates to the population covariance matrix as a whole [LedoitWolf2012]. This quantity depends on the unobservable true matrix. Generalizing the Marčenko–Pastur equation, [LedoitPeche2011] show that $d_i^*$ can be approximated by the *oracle* quantity

$$d_i^{\,or} = \frac{\lambda_i}{\big|\,1 - c - c\,\lambda_i\,\breve{m}_F(\lambda_i)\,\big|^2}, \qquad i = 1, \dots, p ,$$

where $\lambda_i$ is the $i$-th sample eigenvalue, $c$ the limit of $p/n$, and $\breve{m}_F$ the Stieltjes transform of the limiting distribution of the sample eigenvalues [LedoitWolf2012]. The decisive advantage is that this formula does not depend on the population covariance matrix, but only on the distribution of the sample eigenvalues [LedoitWolf2012]. Since the value of the denominator varies with $\lambda_i$, the shrunk eigenvalues are obtained by applying a **nonlinear** transformation to the sample ones: this is what distinguishes the estimator from linear shrinkage [LedoitWolf2012].

The oracle estimator $S_n^{\,or} = U_n D_n^{\,or} U_n'$ is not, however, usable, because $\breve{m}_F$ is defined from the limiting — continuous — distribution of the eigenvalues, whereas the observed one is, by construction, a step function [LedoitWolf2012]. The central contribution of the work is to construct a *bona fide* estimator by consistently estimating, uniformly in $\lambda$, the oracle intensities: having obtained a consistent estimator $\breve{m}_{F_{\widehat{H}_n,\widehat{c}_n}}(\lambda)$ of the Stieltjes transform, one defines

$$\widehat{S}_n = U_n \widehat{D}_n U_n', \qquad \widehat{d}_i = \frac{\lambda_i}{\big|\,1 - \widehat{c}_n - \widehat{c}_n \lambda_i\,\breve{m}_{F_{\widehat{H}_n,\widehat{c}_n}}(\lambda_i)\,\big|^2}, \qquad i = 1, \dots, p ,$$

which converges almost surely to its own oracle in the Frobenius norm [LedoitWolf2012]. The nonlinear estimator has the potential to asymptotically match the linear estimator of [LedoitWolf2004] and often to do much better, especially when linear shrinkage does not offer a sufficient improvement over the sample matrix; since the magnitude of the higher-order effects depends on the population covariance matrix, which is unobservable, it is always more prudent a priori to use nonlinear shrinkage [LedoitWolf2012]. When the precision matrix, that is, the inverse of the covariance matrix, is required, the superior approach is not to invert the previous estimator but to estimate it directly by nonlinearly shrinking the inverses of the sample eigenvalues, with the corresponding oracle estimator $P_n^{\,or} = U_n A_n^{\,or} U_n'$ having elements $a_i^{\,or} = \lambda_i^{-1}\big(1 - c - 2c\,\lambda_i\,\mathrm{Re}[\breve{m}_F(\lambda_i)]\big)$ [LedoitWolf2012].

## References

- **[BerkDeMarzo2019]** Berk, J. & DeMarzo, P. (2019). Corporate Finance (4th ed.). Pearson. [Chapters 10, "Capital Markets and the Pricing of Risk," and 12, "Estimating the Cost of Capital," with an appendix on beta forecasting].
- **[Blume1971]** Blume, M. E. (1971). On the Assessment of Risk. The Journal of Finance, 26(1), 1–10.
- **[Dixon1960]** Dixon, W. J. (1960). Simplified Estimation from Censored Normal Samples. The Annals of Mathematical Statistics, 31(2), 385–391.
- **[JamesStein1961]** James, W. & Stein, C. (1961). Estimation with Quadratic Loss. Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability, 1, 361–379.
- **[LedoitPeche2011]** Ledoit, O. & Péché, S. (2011). Eigenvectors of Some Large Sample Covariance Matrix Ensembles. Probability Theory and Related Fields, 151(1–2), 233–264.
- **[LedoitWolf2004]** Ledoit, O. & Wolf, M. (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. Journal of Multivariate Analysis, 88(2), 365–411.
- **[LedoitWolf2012]** Ledoit, O. & Wolf, M. (2012). Nonlinear Shrinkage Estimation of Large-Dimensional Covariance Matrices. The Annals of Statistics, 40(2), 1024–1060.
- **[MarcenkoPastur1967]** Marčenko, V. A. & Pastur, L. A. (1967). Distribution of Eigenvalues for Some Sets of Random Matrices. Mathematics of the USSR-Sbornik, 1(4), 457–483.
- **[Markowitz1952]** Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77–91.
- **[Stein1956]** Stein, C. (1956). Inadmissibility of the Usual Estimator for the Mean of a Multivariate Normal Distribution. Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability, 1, 197–206.
- **[Vasicek1973]** Vasicek, O. A. (1973). A Note on Using Cross-Sectional Information in Bayesian Estimation of Security Betas. The Journal of Finance, 28(5), 1233–1239.
