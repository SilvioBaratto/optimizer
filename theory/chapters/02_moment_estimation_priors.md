# Moment Estimation and Prior Construction

Portfolio optimization requires estimates of the joint return distribution: at minimum, expected returns and the covariance matrix. The quality of these estimates determines whether the optimizer discovers genuine risk-return tradeoffs or merely exploits estimation noise. Classical mean-variance optimization is notoriously sensitive to its inputs: small perturbations in expected returns can produce wildly different portfolio weights, and covariance matrices estimated from finite samples inherit substantial sampling error that propagates directly into allocation decisions. This chapter presents the principal moment estimation methods, organized by the statistical quantity they target, and shows how they compose into prior distributions that feed the optimization pipeline. The progression moves from expected returns through covariance matrices to factor models, culminating in the assembly of these components into prior objects that serve as the interface between estimation and optimization.

## Expected Return Estimation

Expected returns are simultaneously the most influential and the most difficult inputs to estimate. A portfolio's optimal weights depend linearly on expected returns in the unconstrained mean-variance problem, so estimation errors in the mean vector translate directly, and often dramatically, into misallocation. Empirically, the signal-to-noise ratio of expected return estimates is far lower than that of covariance estimates: with monthly data, estimating means to useful precision requires decades of observations, whereas the covariance matrix stabilizes over much shorter horizons. This asymmetry motivates the range of estimators discussed below, which trade off between fidelity to historical data and stability through various forms of regularization.

### Empirical Mean

The simplest and most direct estimator of expected returns is the sample average:

$$
\hat{\mu}_i = \frac{1}{T}\sum_{t=1}^{T} r_{i,t}
$$

where $r_{i,t}$ denotes the return of asset $i$ at time $t$ and $T$ is the total number of observations. The sample mean is an unbiased estimator of the true expected return under stationarity, and its variance decreases as $\sigma_i^2 / T$. However, this convergence is slow relative to the precision demanded by portfolio optimization. Merton (1980) demonstrated that estimation error in expected returns dominates portfolio construction for samples shorter than approximately 25 years of monthly data. The core difficulty is that equity return distributions have high variance relative to their means: annualized Sharpe ratios rarely exceed unity, implying that the signal (the mean) is small relative to the noise (the standard deviation). As a result, the sample mean is an unbiased but extremely noisy estimator, and portfolios constructed from raw sample means tend to exhibit extreme and unstable weights.

### Shrinkage Toward the Grand Mean

Shrinkage estimation addresses the instability of sample means by pulling individual estimates toward a common central value. The shrinkage estimator takes the form:

$$
\hat{\mu}_i^{\text{shrunk}} = (1 - \alpha)\,\bar{\mu}_i + \alpha \cdot \bar{\mu}_{\text{grand}}
$$

where $\bar{\mu}_i$ is the sample mean of asset $i$, $\bar{\mu}_{\text{grand}} = \frac{1}{N}\sum_{i=1}^{N} \bar{\mu}_i$ is the grand mean across all $N$ assets, and $\alpha \in [0, 1]$ is the shrinkage intensity. When $\alpha = 0$, the estimator reduces to the sample mean; when $\alpha = 1$, all assets receive the same expected return, and any mean-variance optimizer will produce a minimum-variance portfolio.

The theoretical justification derives from the James-Stein estimator, which demonstrates that for $N \geq 3$ Gaussian random variables, the sample mean is inadmissible: there exists a shrinkage estimator that uniformly dominates it in terms of total squared error. The James-Stein result provides the theoretically optimal $\alpha$ for Gaussian returns, balancing the bias introduced by shrinkage against the variance reduction it achieves. In practice, shrinkage reduces the dispersion of expected return estimates, compressing extreme values toward the cross-sectional average. This compression produces more diversified portfolios that are less sensitive to estimation noise in any individual asset's mean return.

### Exponentially Weighted Mean

When the return-generating process is non-stationary (as during regime changes, structural breaks, or evolving market microstructure) equal weighting of all historical observations may be inappropriate. The exponentially weighted mean assigns geometrically decaying weights to past observations:

$$
\hat{\mu}_i = \sum_{t=1}^{T} w_t \cdot r_{i,t}, \quad w_t \propto (1 - \alpha)^{T-t}
$$

where $\alpha \in (0, 1)$ is the decay parameter. Higher values of $\alpha$ concentrate weight on more recent observations, producing estimates that adapt quickly to changes in the return distribution. The effective sample size of the estimator is approximately $1/\alpha$, so $\alpha = 0.06$ corresponds to roughly 17 observations of effective history.

The trade-off is between responsiveness and stability. Aggressive decay (large $\alpha$) enables the estimator to track genuine regime shifts, but it also increases susceptibility to recent anomalies or transient market dislocations. Conservative decay (small $\alpha$) approaches the equal-weighted sample mean and inherits its stability at the cost of slower adaptation.

### Equilibrium (CAPM) Returns

Rather than estimating expected returns from historical data, the equilibrium approach derives implied returns from the assumption that observed market capitalization weights represent an optimal portfolio. Starting from the first-order condition of a mean-variance investor holding the market portfolio, the implied excess returns are:

$$
\boldsymbol{\Pi} = \delta \, \boldsymbol{\Sigma} \, \mathbf{w}_{\text{mkt}}
$$

where $\boldsymbol{\Sigma}$ is the covariance matrix, $\mathbf{w}_{\text{mkt}}$ is the vector of market capitalization weights, and $\delta$ is the risk aversion coefficient. The risk aversion parameter is typically estimated from the market portfolio's risk-return characteristics:

$$
\delta = \frac{\mathbb{E}[R_{\text{mkt}}] - R_f}{\sigma_{\text{mkt}}^2}
$$

where $\mathbb{E}[R_{\text{mkt}}] - R_f$ is the expected market excess return and $\sigma_{\text{mkt}}^2$ is the variance of market returns.

This approach has a crucial advantage: it requires no historical return estimation for individual assets. The only inputs are the covariance matrix (which is estimated with greater precision than means) and the risk aversion parameter. The resulting implied returns are internally consistent with market equilibrium and produce well-diversified portfolios when used as inputs to mean-variance optimization. For this reason, equilibrium returns serve as the most stable and theoretically grounded baseline for the Black-Litterman framework, where they function as the prior distribution that is subsequently updated with investor views.

### Estimator Selection Guidance

The choice of expected return estimator should reflect the available data, the investment horizon, and the role the estimates play in the broader portfolio construction pipeline. The following table summarizes recommended choices under common conditions:

| Condition | Recommended Estimator | Rationale |
|---|---|---|
| Long history ($T > 500$), stable markets | Sample mean | Sufficient data for reliable estimation |
| Short history or regime uncertainty | Shrinkage toward grand mean | Reduces extreme estimates toward consensus |
| Recent regime change suspected | Exponentially weighted mean | Weights recent observations more heavily |
| Black-Litterman or view-based framework | Equilibrium returns | Stable prior for Bayesian updating |
| Regime-switching dynamics | HMM-blended expected returns | State-conditional estimates adapt to prevailing regime |

In practice, the estimator choice interacts with downstream optimization. Mean-variance optimization amplifies estimation error in expected returns, so estimators with lower variance (shrinkage, equilibrium) tend to produce superior out-of-sample performance even if they introduce modest bias. Risk-parity and minimum-variance approaches, which do not use expected returns at all, sidestep this problem entirely, a design choice that itself constitutes an implicit statement about the difficulty of mean estimation.

## Covariance Matrix Estimation

The covariance matrix governs the risk structure of the portfolio. Although covariance estimates are generally more reliable than mean estimates for a given sample size, high-dimensional portfolios introduce challenges of their own: when the number of assets $N$ approaches or exceeds the number of observations $T$, the sample covariance matrix becomes ill-conditioned or singular, and its eigenvalues exhibit systematic biases. The estimators presented below address these challenges through shrinkage, spectral cleaning, sparsity assumptions, and adaptive weighting.

### Sample Covariance

The unbiased sample covariance matrix is:

$$
\hat{\boldsymbol{\Sigma}} = \frac{1}{T-1}\sum_{t=1}^{T}(\mathbf{r}_t - \hat{\boldsymbol{\mu}})(\mathbf{r}_t - \hat{\boldsymbol{\mu}})^\top
$$

where $\mathbf{r}_t \in \mathbb{R}^N$ is the vector of asset returns at time $t$ and $\hat{\boldsymbol{\mu}}$ is the sample mean vector. This estimator is unbiased and maximum likelihood under Gaussian returns, but its quality degrades rapidly as the ratio $N/T$ increases. For $N/T > 0.1$, the eigenvalue spectrum of $\hat{\boldsymbol{\Sigma}}$ is distorted: the largest eigenvalues are biased upward and the smallest are biased downward. When $N > T$, the matrix is singular and cannot be inverted, making it unusable for standard mean-variance optimization.

Even when $T > N$, the effective condition number of $\hat{\boldsymbol{\Sigma}}$ may be large enough to produce numerically unstable portfolio weights. The inverse covariance matrix, which appears in the optimal portfolio formula $\mathbf{w}^* = \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}$, amplifies small eigenvalues, causing extreme sensitivity to estimation noise in the least-variance eigenvectors.

### Ledoit-Wolf Shrinkage

The Ledoit-Wolf estimator addresses the eigenvalue distortion of the sample covariance by shrinking it toward a structured target matrix:

$$
\hat{\boldsymbol{\Sigma}}_{\text{LW}} = \delta^* \, \mathbf{F} + (1 - \delta^*) \, \mathbf{S}
$$

where $\mathbf{S}$ is the sample covariance matrix, $\mathbf{F}$ is a structured target, and $\delta^* \in [0, 1]$ is the analytically optimal shrinkage intensity. The optimal $\delta^*$ minimizes the expected Frobenius norm of the estimation error $\|\hat{\boldsymbol{\Sigma}}_{\text{LW}} - \boldsymbol{\Sigma}_{\text{true}}\|_F^2$ and admits a closed-form expression that depends only on the sample data.

A common target is the constant-correlation model:

$$
\mathbf{F} = \bar{\rho} \, \mathbf{D}\mathbf{J}\mathbf{D} + (1 - \bar{\rho})\,\mathbf{D}^2
$$

where $\mathbf{D} = \text{diag}(\sigma_1, \ldots, \sigma_N)$ contains the sample standard deviations, $\mathbf{J}$ is the matrix of ones, and $\bar{\rho}$ is the average pairwise sample correlation. This target preserves individual asset volatilities while imposing a common correlation structure, providing a well-conditioned matrix that the sample covariance is shrunk toward.

Empirical studies consistently find that Ledoit-Wolf shrinkage reduces portfolio volatility forecast errors by 30--50\% compared to the raw sample covariance, with the largest improvements occurring in high-dimensional settings where $N/T$ is substantial. The improvement arises because shrinkage corrects the systematic eigenvalue distortion: overestimated large eigenvalues are pulled down, underestimated small eigenvalues are pulled up, and the resulting matrix is better conditioned.

### Oracle Approximating Shrinkage

The Oracle Approximating Shrinkage (OAS) estimator extends the Ledoit-Wolf framework by computing an analytically optimal shrinkage intensity without requiring the specification of a particular target structure. It approximates the oracle shrinkage (the intensity that would be chosen if the true covariance matrix were known) using only observable quantities from the sample. This estimator adapts automatically to the data characteristics, adjusting its shrinkage intensity based on the dimensionality ratio $N/T$ and the structure of the sample covariance.

OAS typically performs comparably to Ledoit-Wolf while offering greater flexibility in the choice of structured target.

### Random Matrix Theory Denoising

Random matrix theory provides a principled framework for separating signal from noise in the eigenvalue spectrum of the sample covariance matrix. Under the null hypothesis that returns are independent with common variance $\sigma^2$, the Marchenko-Pastur distribution describes the limiting eigenvalue density of the sample covariance. The upper bound of this distribution is:

$$
\lambda_+ = \sigma^2 \left(1 + \sqrt{N/T}\right)^2
$$

Eigenvalues of $\hat{\boldsymbol{\Sigma}}$ that fall below $\lambda_+$ are indistinguishable from pure noise and carry no exploitable information about the true covariance structure. The denoising procedure replaces these noise eigenvalues with their average while preserving the eigenvalues and eigenvectors above the threshold, which are assumed to reflect genuine systematic risk factors.

Formally, the denoised covariance matrix is reconstructed as:

$$
\hat{\boldsymbol{\Sigma}}_{\text{denoised}} = \sum_{k : \lambda_k > \lambda_+} \lambda_k \, \mathbf{v}_k \mathbf{v}_k^\top + \bar{\lambda}_{\text{noise}} \sum_{k : \lambda_k \leq \lambda_+} \mathbf{v}_k \mathbf{v}_k^\top
$$

where $\bar{\lambda}_{\text{noise}}$ is the average of the noise eigenvalues, ensuring the trace (total variance) is preserved. This approach is particularly effective when $N/T$ is large, as a greater fraction of eigenvalues falls within the noise band.

### Detoning

Detoning removes the dominant market factor, corresponding to the largest eigenvalue and its associated eigenvector, from the covariance matrix. In most equity covariance matrices, the first principal component captures the market mode: the tendency of all stocks to move together. While this common factor is a genuine feature of the return distribution, it can obscure the underlying correlation structure that is relevant for diversification.

The detoned covariance matrix is:

$$
\hat{\boldsymbol{\Sigma}}_{\text{detoned}} = \hat{\boldsymbol{\Sigma}} - \lambda_1 \, \mathbf{v}_1 \mathbf{v}_1^\top
$$

where $\lambda_1$ is the largest eigenvalue and $\mathbf{v}_1$ is the corresponding eigenvector. The resulting matrix reveals the residual correlation structure net of the common market mode. This is useful when the objective is to construct portfolios that are diversified in a factor-neutral sense rather than merely market-directional. Detoning is often applied in combination with denoising: first the noise eigenvalues are cleaned, then the market factor is removed.

### Gerber Statistic

The Gerber statistic provides a robust measure of co-movement that is less sensitive to the distributional assumptions underlying the Pearson correlation. Rather than measuring linear association across all return magnitudes, the Gerber statistic focuses exclusively on co-movements that exceed a significance threshold, filtering out small random fluctuations that may reflect noise rather than genuine co-dependence.

The Gerber covariance between assets $i$ and $j$ is defined through the concordance and discordance counts:

$$
G_{ij}(\theta) = \frac{N^{++}_{ij} + N^{--}_{ij} - N^{+-}_{ij} - N^{-+}_{ij}}{N^{++}_{ij} + N^{--}_{ij} + N^{+-}_{ij} + N^{-+}_{ij}}
$$

where $N^{++}_{ij}$ counts the number of periods where both $|r_{i,t}| > \theta_i$ and $|r_{j,t}| > \theta_j$ with the same sign, and $N^{+-}_{ij}$ counts periods where both exceed their thresholds but with opposite signs. The thresholds $\theta_i$ are typically set as a fraction of each asset's standard deviation. Observations where either asset's return falls within the threshold band are excluded from the count entirely.

This construction makes the Gerber statistic robust to non-Gaussian return distributions, particularly those with heavy tails. By ignoring small returns, it avoids contamination from the many near-zero observations that dominate the sample but carry little information about tail dependence. The resulting covariance matrix tends to be better conditioned and more informative about the co-movement structure during periods of market stress.

### Exponentially Weighted Covariance

Analogous to the exponentially weighted mean, the exponentially weighted covariance assigns geometrically decaying weights to past observations, producing an estimator that adapts to changing volatility and correlation regimes:

$$
\hat{\boldsymbol{\Sigma}}_t = (1 - \alpha)\,\hat{\boldsymbol{\Sigma}}_{t-1} + \alpha \cdot \mathbf{r}_t \mathbf{r}_t^\top
$$

where $\alpha \in (0, 1)$ controls the decay rate. Small values of $\alpha$ produce a slowly evolving estimate close to the equal-weighted sample covariance, while large values create a responsive estimate dominated by recent observations. The effective window length is approximately $1/\alpha$ observations.

This estimator is particularly valuable when volatility and correlation are time-varying, a well-documented empirical regularity. During periods of market stress, correlations tend to increase, and an exponentially weighted estimator captures this dynamic more rapidly than the sample covariance. The trade-off is the same as for the exponentially weighted mean: responsiveness to genuine regime changes comes at the cost of increased susceptibility to transient fluctuations.

### Graphical Lasso

The Graphical Lasso estimates a sparse precision matrix (inverse covariance matrix) by solving the $\ell_1$-penalized maximum likelihood problem:

$$
\hat{\boldsymbol{\Theta}} = \arg\max_{\boldsymbol{\Theta} \succ 0} \left\{ \log\det\boldsymbol{\Theta} - \text{tr}(\mathbf{S}\boldsymbol{\Theta}) - \lambda \|\boldsymbol{\Theta}\|_1 \right\}
$$

where $\mathbf{S}$ is the sample covariance, $\boldsymbol{\Theta} = \boldsymbol{\Sigma}^{-1}$ is the precision matrix, and $\lambda > 0$ is the regularization parameter. The $\ell_1$ penalty forces many entries of $\hat{\boldsymbol{\Theta}}$ to exactly zero, producing a sparse graphical model of conditional dependencies. A zero entry $\hat{\Theta}_{ij} = 0$ implies that assets $i$ and $j$ are conditionally independent given all other assets: their partial correlation is zero.

Sparsity in the precision matrix is a natural assumption for large equity universes: while many pairs of stocks exhibit marginal correlation (through shared exposure to common factors), far fewer pairs have significant conditional dependence after accounting for other assets. The sparse precision matrix directly yields the optimal portfolio weights in the minimum-variance problem, as $\mathbf{w}_{\text{MV}} \propto \boldsymbol{\Theta} \mathbf{1}$, making this estimator particularly efficient for large-scale portfolio construction. The regularization parameter $\lambda$ is typically selected by cross-validation.

### Implied Covariance

The implied covariance estimator incorporates forward-looking information from options markets by blending implied volatilities with historical correlation estimates:

$$
\hat{\boldsymbol{\Sigma}}_{\text{implied}} = f(\boldsymbol{\sigma}_{\text{IV}},\, \boldsymbol{\rho}_{\text{historical}})
$$

where $\boldsymbol{\sigma}_{\text{IV}}$ is the vector of implied volatilities extracted from option prices and $\boldsymbol{\rho}_{\text{historical}}$ is the historical correlation matrix. The construction replaces the diagonal elements of the historical covariance (the variances) with squared implied volatilities while retaining the off-diagonal correlation structure from historical data.

This approach has a theoretical motivation: option-implied volatilities are forward-looking, reflecting the market's consensus expectation of future realized volatility over the option's life. They incorporate information beyond what is available in the historical return series, including anticipated events (earnings announcements, policy decisions) and current risk sentiment. Empirical evidence suggests that implied volatilities are generally superior forecasters of future realized volatility compared to historical estimates.

The practical requirement is access to implied volatility data, which must be routed through the estimation pipeline alongside return data via metadata routing.

### Estimator Selection Guidance

The following table maps portfolio construction scenarios to recommended covariance estimators:

| Condition | Recommended Estimator | Rationale |
|---|---|---|
| $N < 100$, $T > 500$ | Sample covariance | Sufficient data for reliable estimation |
| $N > 100$, $T < 5N$ | Ledoit-Wolf or OAS | Shrinkage corrects overfit eigenvalues |
| Fat-tailed returns | Gerber statistic | Robust to non-Gaussian co-movement |
| Known factor structure | Factor model covariance | Exploits structural dimensionality reduction |
| Large $N > 500$ | Graphical Lasso (cross-validated) | Sparse precision matrix |
| Regime-sensitive strategies | Exponentially weighted covariance | Adapts to changing volatility |
| Discrete regime shifts | HMM regime-conditional covariance | State-specific estimates blended by filtered probabilities |
| High noise ratio $N/T > 0.5$ | Random matrix theory denoising | Separates signal from noise |
| Options data available | Implied covariance | Forward-looking volatility |

The covariance estimator choice should reflect both the dimensionality challenge and the downstream optimization objective. For minimum-variance portfolios, covariance accuracy is paramount because the optimizer depends entirely on the risk model. For mean-variance or Black-Litterman portfolios, covariance quality matters for risk estimation but interacts with expected return estimates; a well-conditioned covariance matrix prevents the optimizer from generating extreme positions along poorly estimated eigenvectors.

## Factor Model Construction

Factor models decompose asset returns into systematic and idiosyncratic components. The canonical factors (value, momentum, quality, and growth) capture distinct sources of systematic return variation that have been extensively documented in the empirical asset pricing literature. Within the portfolio optimization pipeline, factor models serve primarily as a dimensionality reduction device: rather than estimating the full $N \times N$ covariance matrix directly, the factor model estimates a $K \times K$ factor covariance matrix plus $N$ specific variances, drastically reducing the number of parameters and the associated estimation error.

### Loading Matrix Estimation

The factor model relates asset returns to a set of $K$ common factors through a linear regression:

$$
r_{nt} = \alpha_n + \sum_{k=1}^{K} \beta_{nk} \cdot f_{kt} + u_{nt}
$$

where $r_{nt}$ is the return of asset $n$ at time $t$, $f_{kt}$ is the return of factor $k$ at time $t$, $\beta_{nk}$ is the loading of asset $n$ on factor $k$, $\alpha_n$ is the intercept, and $u_{nt}$ is the idiosyncratic residual with $\mathbb{E}[u_{nt}] = 0$ and $\text{Cov}(u_{nt}, f_{kt}) = 0$ for all $k$.

The loading matrix $\mathbf{B} \in \mathbb{R}^{N \times K}$ is estimated by ordinary least squares regression:

$$
\hat{\mathbf{B}} = (\mathbf{Y}^\top \mathbf{Y})^{-1} \mathbf{Y}^\top \mathbf{X}
$$

where $\mathbf{X} \in \mathbb{R}^{T \times N}$ contains asset returns and $\mathbf{Y} \in \mathbb{R}^{T \times K}$ contains factor returns. Each column of $\hat{\mathbf{B}}$ gives the factor exposures of one asset, and these exposures determine how systematic risk propagates from factor space to asset space.

### Dimensionality Reduction

The central benefit of the factor model is the dramatic reduction in the number of parameters to estimate. The full covariance matrix requires $\frac{N(N+1)}{2}$ parameters, while the factor model requires only $\frac{K(K+1)}{2}$ parameters for the factor covariance plus $N$ specific variances, for a total of $\frac{K(K+1)}{2} + N$. The savings are substantial in high-dimensional settings: for $N = 500$ assets and $K = 50$ factors, the factor model estimates $\frac{50 \cdot 51}{2} + 500 = 1{,}775$ parameters compared to $\frac{500 \cdot 501}{2} = 125{,}250$ for the full covariance, a reduction of 98.6\%.

The implied covariance structure under the factor model is:

$$
\boldsymbol{\Sigma} = \mathbf{B}\,\mathbf{F}\,\mathbf{B}^\top + \mathbf{D}
$$

where $\mathbf{B} \in \mathbb{R}^{N \times K}$ is the loading matrix, $\mathbf{F} \in \mathbb{R}^{K \times K}$ is the factor covariance matrix, and $\mathbf{D} \in \mathbb{R}^{N \times N}$ is a diagonal matrix of specific (idiosyncratic) variances. This decomposition ensures that the estimated covariance is positive semi-definite by construction (provided $\mathbf{F}$ is positive semi-definite and $\mathbf{D}$ has non-negative entries), and it is always invertible when the specific variances are strictly positive, regardless of the ratio $N/T$.

### Factor Covariance and Specific Risk

The factor covariance matrix $\mathbf{F}$ is estimated in the lower-dimensional factor space, where $K \ll N$ ensures that even the sample covariance is well-conditioned. Any of the covariance estimators discussed in the previous section (Ledoit-Wolf, exponentially weighted covariance, random matrix theory denoising, and others) can be applied to factor returns rather than asset returns, providing additional regularization in factor space.

The specific risk matrix $\mathbf{D}$ is estimated from the residuals of the factor regression. Specifically, $D_{nn} = \hat{\sigma}^2_{u_n} = \frac{1}{T-K-1}\sum_{t=1}^{T} \hat{u}_{nt}^2$, where $\hat{u}_{nt} = r_{nt} - \hat{\alpha}_n - \sum_k \hat{\beta}_{nk} f_{kt}$ are the regression residuals. The diagonality assumption (that idiosyncratic returns are uncorrelated across assets) is a strong but useful simplification that ensures $\mathbf{D}$ is well-conditioned and that the factor model captures all systematic co-movement through the common factors.

A further advantage of the factor model structure is that it facilitates view specification in the Black-Litterman framework. Views can be expressed directly on factor returns rather than individual asset returns, and these factor-level views propagate to asset-level expectations through the loading matrix: $\mathbb{E}[\mathbf{r}] = \mathbf{B}\,\mathbb{E}[\mathbf{f}]$. This is conceptually natural when views derive from macroeconomic analysis (which naturally targets factor premia) rather than individual stock picking.

### Factor Model Prior

The factor model assembles loading matrix, factor covariance, and specific risk into a complete prior distribution that can replace or complement the empirical prior in downstream optimization. The factor model prior accepts a specification for how factor-level moments are estimated. For example, using the sample mean and covariance of factor returns as the factor-level distribution, these moments are then mapped to asset space through the loading matrix.

Factor returns are passed as auxiliary data alongside the asset return matrix, allowing factor models to participate in cross-validation and pipeline composition alongside other estimators.

## Regime-Switching Models

Financial return distributions exhibit persistent shifts in their statistical properties across different market environments. Bull markets, bear markets, high-volatility crises, and calm recovery periods each produce distinct patterns of means, variances, and correlations. The static estimators discussed above treat the return-generating process as stationary, averaging over regime differences to produce a single set of moment estimates. Regime-switching models explicitly model the transitions between these states, producing time-varying moment estimates that adapt to the prevailing market environment.

### Hidden Markov Models for Regime Detection

The Hidden Markov Model (HMM), introduced to financial econometrics by Hamilton (1989), posits that observed asset returns are generated by a process governed by a discrete latent state that evolves as a first-order Markov chain.

**Generative model.** Let $z_t \in \{1, \ldots, S\}$ denote the latent regime at time $t$. State transitions are governed by a transition matrix $\mathbf{A} \in \mathbb{R}^{S \times S}$:

$$
p(z_t = j \mid z_{t-1} = i) = A_{ij}
$$

with $A_{ij} \geq 0$ and $\sum_{j=1}^{S} A_{ij} = 1$ for each row $i$. The initial state distribution is $\pi_{0,s} = p(z_1 = s)$. Conditional on the latent state, the $N$-dimensional return vector is drawn from a state-specific Gaussian emission:

$$
\mathbf{r}_t \mid z_t = s \sim \mathcal{N}(\boldsymbol{\mu}_s, \boldsymbol{\Sigma}_s)
$$

where $\boldsymbol{\mu}_s \in \mathbb{R}^N$ and $\boldsymbol{\Sigma}_s \in \mathbb{R}^{N \times N}$ are the state-conditional mean vector and covariance matrix. The joint distribution of the complete observation and state sequences factorizes as:

$$
p(\mathbf{r}_{1:T}, z_{1:T}) = p(z_1) \prod_{t=2}^{T} p(z_t \mid z_{t-1}) \prod_{t=1}^{T} p(\mathbf{r}_t \mid z_t)
$$

Two-state models (bull/bear or low-volatility/high-volatility) capture the dominant regime structure in equity markets. Three-state models add an intermediate regime that accommodates transitional periods. The number of states $S$ is a structural hyperparameter, selected by information criteria (AIC, BIC) or cross-validated predictive performance.

**Parameter estimation via Baum-Welch.** The model parameters $\theta = \{\mathbf{A}, \boldsymbol{\pi}_0, \{\boldsymbol{\mu}_s, \boldsymbol{\Sigma}_s\}_{s=1}^{S}\}$ are estimated from observed returns using the Baum-Welch algorithm, a special case of expectation-maximization (EM). The algorithm alternates between computing expected sufficient statistics under the current parameter estimates and updating those parameters to maximize the expected complete-data log-likelihood.

The E-step employs the forward-backward algorithm. The forward variable $\alpha_t(s) = p(\mathbf{r}_{1:t}, z_t = s)$ satisfies the recursion:

$$
\alpha_t(s) = \left[\sum_{s'=1}^{S} \alpha_{t-1}(s') \, A_{s',s}\right] p(\mathbf{r}_t \mid z_t = s)
$$

initialized with $\alpha_1(s) = \pi_{0,s} \, p(\mathbf{r}_1 \mid z_1 = s)$. The backward variable $\beta_t(s) = p(\mathbf{r}_{t+1:T} \mid z_t = s)$ satisfies:

$$
\beta_t(s) = \sum_{s'=1}^{S} A_{s,s'} \, p(\mathbf{r}_{t+1} \mid z_{t+1} = s') \, \beta_{t+1}(s')
$$

initialized with $\beta_T(s) = 1$. The smoothed state probabilities and pairwise transition probabilities follow as:

$$
\gamma_t(s) = p(z_t = s \mid \mathbf{r}_{1:T}) = \frac{\alpha_t(s) \, \beta_t(s)}{p(\mathbf{r}_{1:T})}
$$

$$
\xi_t(i, j) = p(z_{t-1} = i, \, z_t = j \mid \mathbf{r}_{1:T}) = \frac{\alpha_{t-1}(i) \, A_{ij} \, p(\mathbf{r}_t \mid z_t = j) \, \beta_t(j)}{p(\mathbf{r}_{1:T})}
$$

The M-step updates the parameters using these sufficient statistics:

$$
\hat{A}_{ij} = \frac{\sum_{t=2}^{T} \xi_t(i,j)}{\sum_{t=2}^{T} \gamma_{t-1}(i)}, \qquad \hat{\boldsymbol{\mu}}_s = \frac{\sum_{t=1}^{T} \gamma_t(s) \, \mathbf{r}_t}{\sum_{t=1}^{T} \gamma_t(s)}
$$

$$
\hat{\boldsymbol{\Sigma}}_s = \frac{\sum_{t=1}^{T} \gamma_t(s) \, (\mathbf{r}_t - \hat{\boldsymbol{\mu}}_s)(\mathbf{r}_t - \hat{\boldsymbol{\mu}}_s)^\top}{\sum_{t=1}^{T} \gamma_t(s)}
$$

The E-step and M-step alternate until convergence, monotonically increasing the observed-data log-likelihood $\log p(\mathbf{r}_{1:T} \mid \theta)$ at each iteration.

**Online filtering.** For production deployment, forward filtering provides the causal filtered state probability at the current time:

$$
p(z_t = s \mid \mathbf{r}_{1:t}) \propto p(\mathbf{r}_t \mid z_t = s) \sum_{s'=1}^{S} A_{s',s} \, p(z_{t-1} = s' \mid \mathbf{r}_{1:t-1})
$$

This recursive update uses only past and current observations, maintaining the temporal causality required for real-time portfolio construction. No future data contaminates the state estimate, a property that is critical for backtest validity.

### Regime-Conditional Moment Estimation

The HMM's state-conditional parameters yield time-varying moment estimates through probability-weighted blending. Given the filtered state probabilities at time $t$, the blended expected return vector is:

$$
\hat{\boldsymbol{\mu}}_t = \sum_{s=1}^{S} p(z_t = s \mid \mathbf{r}_{1:t}) \, \boldsymbol{\mu}_s
$$

The blended covariance matrix must account for both within-state and between-state variation in means:

$$
\hat{\boldsymbol{\Sigma}}_t = \sum_{s=1}^{S} p(z_t = s \mid \mathbf{r}_{1:t}) \left[\boldsymbol{\Sigma}_s + (\boldsymbol{\mu}_s - \hat{\boldsymbol{\mu}}_t)(\boldsymbol{\mu}_s - \hat{\boldsymbol{\mu}}_t)^\top\right]
$$

The second term inflates the blended covariance by the cross-state dispersion of means. When the regime is ambiguous (filtered probabilities near uniform), this term is large, producing appropriately conservative risk estimates. When the regime is clearly identified (one state probability near unity), the blended moments approximate the conditional moments of the dominant state.

During a high-probability bear market regime, the blended estimates shift toward the bear-state parameters: lower expected returns, higher volatilities, and elevated correlations. During a bull regime, the reverse holds. Transitions between regimes are smooth, governed by the continuous evolution of filtered probabilities, which avoids the instability that hard regime switching would introduce.

These time-varying moments serve as direct inputs to the empirical prior and, through it, to the optimization pipeline. Any of the covariance estimators discussed in the preceding sections (shrinkage, denoising, Graphical Lasso) can be applied within each regime to regularize the state-conditional covariance estimates, a technique that is particularly valuable when the effective sample size per regime is small.

### Deep Markov Models

Hidden Markov Models are constrained by their discrete state space: the number of distinct regime configurations is fixed at $S$, and transitions between regimes are instantaneous. Deep Markov Models (DMMs) generalize this framework by replacing discrete latent states with continuous latent vectors and parameterizing the transition and emission distributions with neural networks. This generalization enables the model to capture complex, nonlinear dynamics in the return-generating process that discrete-state HMMs cannot represent.

**Generative model.** The DMM posits a continuous latent state $\mathbf{z}_t \in \mathbb{R}^d$ that evolves through a neural-network-parameterized transition distribution:

$$
p_\theta(\mathbf{z}_t \mid \mathbf{z}_{t-1}) = \mathcal{N}\!\left(\mathbf{z}_t \;\middle|\; \boldsymbol{\mu}_\theta^{\text{trans}}(\mathbf{z}_{t-1}),\; \text{diag}\!\left(\boldsymbol{\sigma}_\theta^{\text{trans}}(\mathbf{z}_{t-1})^2\right)\right)
$$

where $\boldsymbol{\mu}_\theta^{\text{trans}}$ and $\boldsymbol{\sigma}_\theta^{\text{trans}}$ are outputs of a neural network (the transition network) with parameters $\theta$. Observed returns are generated from a state-dependent emission distribution:

$$
p_\theta(\mathbf{r}_t \mid \mathbf{z}_t) = \mathcal{N}\!\left(\mathbf{r}_t \;\middle|\; \boldsymbol{\mu}_\theta^{\text{emit}}(\mathbf{z}_t),\; \text{diag}\!\left(\boldsymbol{\sigma}_\theta^{\text{emit}}(\mathbf{z}_t)^2\right)\right)
$$

The joint distribution factorizes as:

$$
p_\theta(\mathbf{r}_{1:T}, \mathbf{z}_{1:T}) = p(\mathbf{z}_1) \prod_{t=2}^{T} p_\theta(\mathbf{z}_t \mid \mathbf{z}_{t-1}) \prod_{t=1}^{T} p_\theta(\mathbf{r}_t \mid \mathbf{z}_t)
$$

**Gated transitions.** A refinement introduces gating mechanisms that interpolate between linear and nonlinear dynamics:

$$
\boldsymbol{\mu}_\theta^{\text{trans}}(\mathbf{z}_{t-1}) = (1 - \mathbf{g}_t) \odot (\mathbf{W}\mathbf{z}_{t-1} + \mathbf{b}) + \mathbf{g}_t \odot \mathbf{h}_\theta(\mathbf{z}_{t-1})
$$

where $\mathbf{g}_t = \sigma(\mathbf{W}_g \mathbf{z}_{t-1} + \mathbf{b}_g)$ is a sigmoid gate, $\mathbf{W}\mathbf{z}_{t-1} + \mathbf{b}$ is the linear component, $\mathbf{h}_\theta$ is a nonlinear neural network, and $\odot$ denotes elementwise multiplication. When the gate activations are near zero, the transition is approximately linear, resembling a linear state-space model. When the gate activations approach one, the transition is fully nonlinear. This adaptive complexity allows the model to allocate nonlinearity only where the data requires it.

**Variational inference.** The posterior $p_\theta(\mathbf{z}_{1:T} \mid \mathbf{r}_{1:T})$ is intractable due to the nonlinear dependencies introduced by the neural networks. Amortized variational inference addresses this by introducing a structured inference network $q_\phi(\mathbf{z}_{1:T} \mid \mathbf{r}_{1:T})$ that factorizes as:

$$
q_\phi(\mathbf{z}_{1:T} \mid \mathbf{r}_{1:T}) = \prod_{t=1}^{T} q_\phi(\mathbf{z}_t \mid \mathbf{z}_{t-1}, \mathbf{r}_{t:T})
$$

Each factor conditions on both the previous latent state and future observations, which are processed by a backward-running recurrent neural network. The inference network and generative model are trained jointly by maximizing the evidence lower bound (ELBO):

$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi}\!\left[\sum_{t=1}^{T} \log p_\theta(\mathbf{r}_t \mid \mathbf{z}_t)\right] - \sum_{t=1}^{T} \text{KL}\!\left[q_\phi(\mathbf{z}_t \mid \cdot) \,\|\, p_\theta(\mathbf{z}_t \mid \mathbf{z}_{t-1})\right]
$$

The first term rewards accurate reconstruction of observed returns. The second term penalizes deviation of the approximate posterior from the generative prior, regularizing the latent dynamics against overfitting.

**Predictive moment extraction.** Given observations $\mathbf{r}_{1:t}$ up to the current time, the DMM produces a distribution over the next-period latent state and consequently a predictive distribution over returns. The predictive moments are:

$$
\hat{\boldsymbol{\mu}}_{t+1} = \mathbb{E}_{p(\mathbf{z}_{t+1} \mid \mathbf{r}_{1:t})}\!\left[\boldsymbol{\mu}_\theta^{\text{emit}}(\mathbf{z}_{t+1})\right]
$$

$$
\hat{\boldsymbol{\Sigma}}_{t+1} = \mathbb{E}_{p(\mathbf{z}_{t+1} \mid \mathbf{r}_{1:t})}\!\left[\text{diag}\!\left(\boldsymbol{\sigma}_\theta^{\text{emit}}(\mathbf{z}_{t+1})^2\right) + \boldsymbol{\mu}_\theta^{\text{emit}}(\mathbf{z}_{t+1})\boldsymbol{\mu}_\theta^{\text{emit}}(\mathbf{z}_{t+1})^\top\right] - \hat{\boldsymbol{\mu}}_{t+1}\hat{\boldsymbol{\mu}}_{t+1}^\top
$$

These expectations are approximated by Monte Carlo sampling from the posterior predictive distribution. The resulting time-varying moments substitute for or complement the static estimators discussed earlier in this chapter.

### Model Selection Guidance

The regime-switching models occupy a distinct position in the estimator landscape, complementing rather than replacing the static estimators.

| Condition | Recommended Approach | Rationale |
|---|---|---|
| Clear regime structure (bull/bear) | HMM with 2--3 states | Interpretable, parsimonious, well-understood |
| Smooth, continuous regime variation | Deep Markov Model | Captures nonlinear dynamics in continuous latent space |
| Limited data ($T < 500$) | Static estimators (shrinkage, factor model) | Insufficient data for reliable regime inference |
| High-dimensional universe ($N > 100$) | HMM on factor returns, mapped to assets | Reduces dimensionality of emission model |
| Regime transitions known a priori | Fixed-regime conditional estimation | Avoids estimation of transition dynamics |

HMMs are preferred when interpretability matters and the regime structure is expected to be discrete and low-dimensional. DMMs are preferred when the underlying dynamics are complex, the data is abundant, and the practitioner is willing to trade interpretability for representational capacity. Static estimators remain appropriate when regime dynamics are not a primary concern or when the sample is too short for reliable regime inference.

## The Empirical Prior

The empirical prior assembles an expected return estimator and a covariance estimator into a single object that provides the complete input specification for downstream optimization. This assembly is the critical interface between the estimation stage and the optimization stage of the portfolio construction pipeline: the optimizer receives a prior distribution and extracts the moments it requires, without needing to know how those moments were estimated.

The empirical prior accepts an expected return estimator and a covariance estimator as components, allowing any combination of the estimators discussed above. A typical composition might pair shrinkage toward the grand mean for expected returns with Ledoit-Wolf shrinkage for the covariance matrix, producing a prior that is regularized in both the mean and covariance dimensions. The modularity of this design means that changing the covariance estimator (for instance, switching from Ledoit-Wolf to random matrix theory denoising) requires modifying a single component without altering any other part of the pipeline.

For multi-period optimization, the prior supports log-normal return assumptions. When enabled, the prior compounds single-period moments to produce multi-period expected returns and covariances consistent with the geometric growth properties of wealth accumulation. Setting the investment horizon (for instance, 252 periods for annualization when working with daily data) applies the appropriate compounding correction. This adjustment is important because mean-variance optimization at long horizons should account for the volatility drag that reduces compound returns below arithmetic returns:

$$
\mathbb{E}[\log(1 + R_p)] \approx \mathbb{E}[R_p] - \frac{1}{2}\sigma_p^2
$$

The prior serves as the modular interface that decouples estimation from optimization. Any estimator conforming to the prior protocol can be substituted without downstream changes, enabling systematic comparison of estimation approaches through cross-validation.

## LLM-Augmented Moment Estimation

Large language models introduce a qualitative information channel into the otherwise quantitative moment estimation pipeline. While the estimators discussed above operate exclusively on numerical return data, LLMs can process unstructured text (economic reports, central bank communications, earnings transcripts, news articles) and translate qualitative assessments into quantitative adjustments to the estimation process. This augmentation does not replace statistical estimation but rather informs the configuration and parameterization of the estimators.

**Risk aversion calibration.** The equilibrium return estimator depends critically on the risk aversion parameter $\delta$, which determines the scale of implied returns. LLMs can classify the current macroeconomic regime (early-cycle, mid-cycle, late-cycle, or recession) by processing leading indicators such as purchasing managers' indices, yield curve slopes, and credit spreads. This regime classification maps to appropriate risk aversion parameters: expansionary regimes warrant lower risk aversion (higher implied returns, more aggressive positioning), while late-cycle or recessionary regimes warrant higher risk aversion (lower implied returns, more defensive positioning). The mapping from regime to $\delta$ translates qualitative macroeconomic assessment into a precise numerical input for the equilibrium return estimator.

**Factor weight adaptation.** Business cycle phase influences which systematic factors are likely to deliver premiums over the subsequent investment horizon. LLMs can recommend shifting factor emphasis based on current economic conditions; for instance, tilting toward quality and value factors in late-cycle environments where earnings resilience and margin of safety become more important, or toward momentum and growth factors in early-cycle recoveries where economic acceleration favors high-beta, growth-oriented exposures. These recommendations adjust the loading matrix or factor covariance inputs in the factor model prior, shaping the risk model to reflect the anticipated regime.

**Covariance regime selection.** The choice of covariance estimator itself can be informed by qualitative analysis. News sentiment analysis can detect whether the current environment favors responsive estimators (such as exponentially weighted covariance during regime transitions, when correlations and volatilities are shifting rapidly) or stable estimators (such as Ledoit-Wolf during calm periods when the return distribution is approximately stationary). An LLM monitoring news flow and policy communications can flag transitions between these regimes, triggering automated switches in the covariance estimation methodology.

**Confidence calibration.** Perhaps most subtly, LLMs can assess the reliability of moment estimates by cross-referencing quantitative signals with qualitative information. When historical factor premia are consistent with the narrative embedded in economic data and corporate earnings, the moment estimates can be trusted with higher confidence. When quantitative signals diverge from the qualitative picture (for instance, when momentum signals remain positive but leading indicators suggest deterioration) the LLM can flag the discrepancy, prompting wider uncertainty bands in the prior or greater shrinkage toward neutral estimates. This meta-level assessment of estimate reliability adds a layer of robustness that purely quantitative systems lack.

\newpage
