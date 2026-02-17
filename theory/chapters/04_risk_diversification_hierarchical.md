# Risk Measures, Diversification, and Hierarchical Methods

Portfolio construction requires explicit choices about how risk is measured, how it is distributed across positions, and whether the optimization should respect the natural clustering structure of assets. These choices are not incidental; they define the very meaning of an "optimal" portfolio. A variance-minimizing allocation differs fundamentally from one that minimizes conditional drawdowns, and an allocation that respects hierarchical asset structure differs from one that treats the investment universe as a flat collection of securities. This chapter presents the full taxonomy of risk measures, the risk budgeting and maximum diversification frameworks that allocate risk systematically, and the hierarchical methods that exploit asset correlation structure to produce stable allocations without matrix inversion. The integration of large language models into each of these layers introduces adaptive, forward-looking intelligence that complements the purely quantitative machinery.

## Risk Measures: From Variance to Tail Risk

The choice of risk measure determines what the optimizer considers "risky" and therefore shapes portfolio composition in profound ways. Variance treats upside and downside symmetrically, penalizing favorable returns with the same weight as adverse ones. Tail risk measures concentrate attention on the extreme left tail of the return distribution, where losses are largest and most consequential. Drawdown measures target the investor's lived experience of peak-to-trough declines, which often drives behavioral responses more powerfully than abstract distributional statistics. Any of these measures can serve interchangeably as the objective or constraint in mean-risk optimization, and this architectural uniformity allows the practitioner to explore the full risk-measure landscape without altering the optimization framework.

### Variance and Standard Deviation

Variance remains the foundational risk measure in portfolio theory, originating in the mean-variance framework. Portfolio variance is defined as

$$\sigma_P^2 = \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}$$

where $\mathbf{w}$ denotes the vector of portfolio weights and $\boldsymbol{\Sigma}$ the covariance matrix of asset returns. The portfolio standard deviation follows directly:

$$\sigma_P = \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}$$

Variance is symmetric: it penalizes upside deviations from the mean with precisely the same severity as downside deviations. This symmetry is both its analytical strength and its conceptual limitation. From an optimization standpoint, variance is convex and quadratic in the weight vector, admitting closed-form solutions under linear constraints and efficient numerical solution under more general constraint sets. Its analytical tractability has made it the default risk measure for decades, and its properties are thoroughly understood.

When variance serves as the objective, the resulting optimization problem is a convex quadratic program, solvable with high reliability by standard solvers.

### Semi-Variance and Semi-Deviation

Semi-variance addresses the conceptual limitation of variance by restricting attention to returns that fall below the mean. It is defined as

$$\text{SemiVar}_P = \frac{1}{T}\sum_{t=1}^{T} \left[\min(r_{P,t} - \bar{r}_P, 0)\right]^2$$

where $r_{P,t}$ denotes the portfolio return in period $t$, $\bar{r}_P$ the mean portfolio return, and $T$ the number of observations. Semi-deviation is the square root of semi-variance. By zeroing out positive deviations, semi-variance captures only the downside component of return dispersion, better reflecting investor loss aversion. An asset that frequently produces large positive surprises but rarely declines will appear far less risky under semi-variance than under variance.

The semi-deviation follows as

$$\text{SemiDev}_P = \sqrt{\text{SemiVar}_P}$$

Semi-variance is not as analytically convenient as full variance, since it depends on the realized return path rather than solely on the covariance matrix. However, it remains a convex function of portfolio weights under standard conditions, and its optimization proceeds through sample-based formulations.

### Mean Absolute Deviation

The mean absolute deviation offers an alternative dispersion measure that is more robust to outliers than variance:

$$\text{MAD}_P = \frac{1}{T}\sum_{t=1}^{T} |r_{P,t} - \bar{r}_P|$$

Because MAD uses absolute deviations rather than squared deviations, extreme observations exert less influence on the risk estimate. This makes MAD particularly suitable when the return distribution contains occasional outliers that would disproportionately inflate variance. The MAD-based optimization can be formulated as a linear program through standard auxiliary variable techniques, which can be computationally advantageous for large portfolios.

### Value-at-Risk

Value-at-Risk at confidence level $\alpha$ represents the loss threshold that is exceeded with probability $\alpha$:

$$\text{VaR}_\alpha(P) = -F_{r_P}^{-1}(\alpha)$$

where $F_{r_P}^{-1}$ is the inverse cumulative distribution function (quantile function) of portfolio returns. For instance, at $\alpha = 0.05$, VaR is the loss level such that only five percent of return realizations are worse. VaR is intuitive and widely reported, but it possesses a critical theoretical deficiency: it is not a coherent risk measure. Specifically, VaR can violate subadditivity, meaning that the VaR of a combined portfolio can exceed the sum of the individual VaR contributions. This implies that diversification may appear to increase risk under VaR, a paradoxical and undesirable property.

VaR provides no information about the magnitude of losses beyond the threshold. Two portfolios with identical VaR may have vastly different tail behavior. Despite these limitations, VaR remains useful as a reporting metric and regulatory benchmark.

### Conditional Value-at-Risk

Conditional Value-at-Risk (also known as Expected Shortfall) remedies the deficiencies of VaR by averaging over the entire tail beyond the VaR threshold:

$$\text{CVaR}_\alpha(\mathbf{w}) = -\frac{1}{\alpha}\int_0^{\alpha} F_{\mathbf{w}^\top \mathbf{r}}^{-1}(p)\,dp$$

CVaR represents the expected loss conditional on the loss exceeding VaR. It is a coherent risk measure, satisfying all four axioms of coherence: monotonicity, translation invariance, positive homogeneity, and crucially, subadditivity. This last property ensures that diversification is always recognized as risk-reducing under CVaR. CVaR is convex in portfolio weights, making it amenable to efficient optimization. The sample-based formulation replaces the integral with an average over the worst $\lfloor \alpha T \rfloor$ return observations, and the Rockafellar-Uryasev reformulation converts CVaR minimization into a linear program.

CVaR is more conservative than VaR because it accounts for the severity of tail losses, not merely their frequency. This makes it the preferred risk measure for portfolios where tail risk is a primary concern.

### Entropic Value-at-Risk

The Entropic Value-at-Risk provides an even tighter bound on tail risk than CVaR:

$$\text{EVaR}_\alpha(\mathbf{w}) = \inf_{z > 0}\left\{\frac{1}{z}\ln\left(\frac{M_{\mathbf{w}^\top\mathbf{r}}(z)}{\alpha}\right)\right\}$$

where $M_{\mathbf{w}^\top\mathbf{r}}(z) = \mathbb{E}[e^{z \cdot \mathbf{w}^\top\mathbf{r}}]$ is the moment-generating function of portfolio returns. EVaR is derived from the Chernoff bound on tail probabilities and dominates CVaR: for any portfolio, $\text{EVaR}_\alpha \geq \text{CVaR}_\alpha$. This makes EVaR a more conservative risk measure that is better suited for distributions with heavy tails, where the moment-generating function captures information about higher-order moments that CVaR does not fully exploit.

EVaR is coherent and convex, and its optimization can be handled through exponential cone programming. It is particularly valuable when the return distribution exhibits significant kurtosis or skewness.

### Worst Realization

The worst realization is the most conservative point-in-time risk measure, defined simply as the maximum single-period loss observed in the sample:

$$\text{WR}(\mathbf{w}) = \max_{t \in \{1,\ldots,T\}} \left(-r_{P,t}\right)$$

Minimizing worst realization produces a minimax portfolio that guards against the single worst historical outcome. This measure is extremely conservative and may produce overly defensive allocations in practice, but it serves as a useful bound and stress-test criterion.

### Maximum Drawdown

Maximum drawdown captures the largest peak-to-trough decline in portfolio value over the evaluation period:

$$\text{MDD} = \max_{t \in [0,T]}\left(\frac{\max_{s \in [0,t]} V_s - V_t}{\max_{s \in [0,t]} V_s}\right)$$

where $V_t$ is the portfolio value at time $t$. Unlike point-in-time risk measures, maximum drawdown reflects the path-dependent experience of the investor. A portfolio may have low variance yet suffer severe drawdowns if negative returns cluster in time. Maximum drawdown is directly relevant to investor psychology, as prolonged declines often trigger premature liquidation and behavioral errors.

Maximum drawdown is not convex in general, but appropriate reformulations enable its use as an optimization objective.

### Average Drawdown

Average drawdown computes the mean of all drawdown episodes over the evaluation period, providing a more representative picture of typical drawdown behavior than the single worst case captured by maximum drawdown. A portfolio that experiences many moderate drawdowns may have a high average drawdown despite a moderate maximum drawdown, signaling persistent capital impairment. This measure is less sensitive to a single extreme event and therefore more stable as an optimization objective.

### Conditional Drawdown-at-Risk

Conditional Drawdown-at-Risk applies the CVaR concept to the distribution of drawdowns rather than returns:

$$\text{CDaR}_\alpha(\mathbf{w}) = -\frac{1}{\alpha}\int_0^{\alpha} F_{\text{DD}(\mathbf{w})}^{-1}(p)\,dp$$

where $F_{\text{DD}(\mathbf{w})}$ is the cumulative distribution function of drawdowns for portfolio $\mathbf{w}$. CDaR represents the expected drawdown in the worst $\alpha$ fraction of drawdown episodes. It is the drawdown analogue of CVaR and inherits its coherence properties within the drawdown domain. CDaR is particularly valuable for investors whose risk tolerance is framed in terms of peak-to-trough declines rather than return dispersion.

### Entropic Drawdown-at-Risk

Entropic Drawdown-at-Risk applies the EVaR tightening principle to the drawdown distribution, yielding a more conservative drawdown risk measure than CDaR. Just as EVaR dominates CVaR for return distributions, EDaR dominates CDaR for drawdown distributions, providing tighter tail bounds when drawdown episodes exhibit heavy-tailed behavior. EDaR is suitable for portfolios where protection against extreme drawdown scenarios is paramount.

### Ulcer Index

The Ulcer Index computes the root mean square of drawdowns, thereby penalizing both the depth and duration of drawdown episodes:

$$\text{UI} = \sqrt{\frac{1}{T}\sum_{t=1}^{T} D_t^2}$$

where $D_t$ is the drawdown at time $t$. By squaring drawdowns before averaging, the Ulcer Index places greater weight on severe drawdowns while still accounting for persistent moderate drawdowns. It provides a single scalar summary of the entire drawdown experience.

### Risk Measure Selection Guidance

The choice of risk measure should reflect the investor's loss preferences, the distributional characteristics of the asset universe, and the computational requirements of the optimization. The following table summarizes the recommended mappings:

| Investor Preference | Recommended Risk Measure | Rationale |
|---|---|---|
| Traditional mean-variance | Variance | Analytical tractability, well-studied properties |
| Loss aversion (downside focus) | Semi-variance or CVaR | Penalizes only adverse outcomes |
| Tail risk management | CVaR or EVaR | Captures extreme loss behavior coherently |
| Drawdown sensitivity | CDaR or maximum drawdown | Targets peak-to-trough investor experience |
| Conservative tail protection | EVaR or EDaR | Tightest bounds on tail and drawdown risk |
| Robust to outliers | MAD | Less sensitive to extreme observations |

No single risk measure is universally superior. In practice, the most informative approach is to optimize under multiple risk measures and compare the resulting allocations, identifying positions that are robust across measures and those that are highly sensitive to the risk definition.

## Risk Budgeting and Equal Risk Contribution

### Risk Contribution Framework

Risk budgeting decomposes total portfolio risk into contributions attributable to each asset, enabling the portfolio manager to specify how much risk each position should bear. The framework rests on Euler's theorem for homogeneous functions: if the risk measure $\rho(\mathbf{w})$ is positively homogeneous of degree one in weights (as standard deviation is), then it decomposes exactly into asset-level contributions.

The marginal risk contribution of asset $i$ measures the sensitivity of portfolio risk to a marginal increase in the weight of asset $i$:

$$\text{MRC}_i = \frac{\partial \sigma_P}{\partial w_i} = \frac{(\boldsymbol{\Sigma}\mathbf{w})_i}{\sigma_P}$$

The component risk contribution multiplies the marginal contribution by the asset weight, yielding the absolute risk attributable to position $i$:

$$\text{RC}_i = w_i \cdot \text{MRC}_i = w_i \cdot \frac{(\boldsymbol{\Sigma}\mathbf{w})_i}{\sigma_P}$$

The Euler decomposition guarantees that these contributions sum exactly to total portfolio risk:

$$\sum_{i=1}^{N} \text{RC}_i = \sigma_P$$

This decomposition provides a complete accounting of risk: no risk is "unattributed," and the manager can assess whether the risk allocation across positions aligns with investment convictions.

### Equal Risk Contribution Portfolios

The equal risk contribution portfolio requires each asset to contribute identically to total portfolio risk:

$$w_i \cdot (\boldsymbol{\Sigma}\mathbf{w})_i = \frac{\sigma_P^2}{N} \quad \text{for all } i$$

This system of $N$ non-linear equations does not admit a closed-form solution and must be solved iteratively. The resulting allocation lies between equal weighting and minimum variance: it avoids the extreme concentration that minimum variance can produce while incorporating correlation structure that equal weighting ignores. Equal risk contribution portfolios are particularly attractive because they require no expected return estimates, relying solely on covariance information. This eliminates the largest source of estimation error in portfolio optimization.

The equal risk contribution approach embodies a principled agnosticism: absent strong views on expected returns, the most defensible allocation is one where no single asset dominates portfolio risk. In the risk budgeting framework, this corresponds to setting the budget vector to equal allocations across all assets.

### Custom Risk Budgets

When the investor holds differential convictions about asset attractiveness or risk tolerance, custom risk budgets allow explicit specification of the desired risk allocation. Given a budget vector $\mathbf{b}$ with $b_i > 0$ for all $i$ and $\sum_{i=1}^{N} b_i = 1$, the custom risk budget portfolio satisfies:

$$w_i \cdot (\boldsymbol{\Sigma}\mathbf{w})_i = b_i \cdot \sigma_P^2 \quad \text{for all } i$$

Assets assigned larger budgets receive greater weight (all else equal), while those with smaller budgets are constrained to contribute less risk. The budget vector translates qualitative views about asset desirability into a quantitative risk allocation target.

### Extension to Tail Risk Measures

The risk budgeting framework generalizes naturally beyond variance to any convex, positively homogeneous risk measure $\rho(\mathbf{w})$. The generalized risk contribution of asset $i$ is defined via the gradient:

$$\text{RC}_i = w_i \cdot \frac{\partial \rho}{\partial w_i}$$

and the Euler decomposition continues to hold: $\sum_i \text{RC}_i = \rho(\mathbf{w})$.

CVaR-based risk budgeting accounts for fat tails and asymmetric return distributions, allocating risk based on each asset's contribution to expected tail loss rather than to variance. CDaR-based risk budgeting targets drawdown contributions, directly addressing the investor's experience of capital impairment. The risk budgeting optimizer accepts any of the risk measures discussed above, enabling seamless exploration of risk budgeting under alternative risk definitions.

## Maximum Diversification

The maximum diversification portfolio maximizes the diversification ratio, defined as

$$\text{DR}(\mathbf{w}) = \frac{\mathbf{w}^\top \boldsymbol{\sigma}}{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}} = \frac{\sum_{i=1}^{N} w_i \sigma_i}{\sigma_P}$$

where $\boldsymbol{\sigma}$ is the vector of individual asset volatilities and $\sigma_P$ is the portfolio volatility. The numerator represents the weighted average of individual volatilities, which is the portfolio volatility that would prevail if all pairwise correlations were unity. The denominator is the actual portfolio volatility, which is lower due to imperfect correlation. The diversification ratio therefore quantifies the risk reduction achieved through diversification.

A diversification ratio of exactly 1.0 indicates zero diversification benefit: the portfolio behaves as though it holds a single asset. Higher ratios indicate greater diversification, with the maximum achievable ratio depending on the correlation structure of the asset universe. The maximum diversification portfolio is equivalent to the minimum variance portfolio when all assets are first standardized to unit volatility, revealing an elegant connection between the two approaches: maximum diversification seeks the allocation that extracts the greatest correlation-driven risk reduction, independent of individual asset volatility levels.

## Distance Measures and Codependence

Hierarchical portfolio methods require a distance or codependence measure between assets to define the clustering structure. The choice of distance measure determines what notion of "similarity" drives the hierarchical decomposition, and different measures can produce substantially different dendrograms and therefore different allocations.

### Pearson Distance

The most widely used distance measure transforms the Pearson linear correlation coefficient into a metric:

$$d_{ij} = \sqrt{\frac{1}{2}(1 - \rho_{ij}^{\text{Pearson}})}$$

This transformation maps perfectly correlated assets ($\rho = 1$) to zero distance and uncorrelated assets ($\rho = 0$) to distance $1/\sqrt{2}$. Perfectly negatively correlated assets ($\rho = -1$) map to distance 1. Pearson distance captures linear dependence and is the standard choice for most applications.

### Kendall Distance

Kendall distance is derived from Kendall's tau rank correlation coefficient. Kendall's tau measures the concordance between two variables: the probability that two randomly selected observations exhibit the same ordering in both variables minus the probability they exhibit opposite orderings. The resulting distance is robust to non-linear monotonic relationships, as it depends only on the ranks of observations rather than their magnitudes. This makes it less sensitive to outliers and distributional assumptions than Pearson distance.

### Spearman Distance

Spearman distance is based on the Spearman rank correlation, which is simply the Pearson correlation computed on the ranks of the observations. It captures monotonic but not necessarily linear dependence, occupying a middle ground between Pearson's linearity assumption and the more general dependence captured by non-parametric measures. Spearman distance shares Kendall distance's robustness to outliers while being computationally simpler.

### Distance Correlation

Distance correlation, introduced in the statistical literature as a measure of dependence between random vectors, captures both linear and non-linear dependence. Its defining property is that distance correlation equals zero if and only if the two variables are statistically independent. This is a strictly stronger condition than zero Pearson correlation, which only implies independence under joint normality. Distance correlation therefore detects non-linear relationships that Pearson, Kendall, and Spearman measures may miss entirely.

This measure is particularly valuable in asset universes where non-linear dependence structures arise, such as during market stress when correlations exhibit threshold effects.

### Mutual Information

Mutual information is an information-theoretic measure that quantifies the total statistical dependence between two variables:

$$I(X; Y) = \int \int p(x, y) \ln \frac{p(x, y)}{p(x)p(y)} \, dx \, dy$$

Mutual information is zero if and only if the variables are independent, and it captures all forms of dependence: linear, non-linear, and higher-order. Unlike correlation-based measures, mutual information is not limited to monotonic relationships. Its estimation requires binning or kernel density estimation, introducing a degree of sensitivity to the estimation procedure. It provides the most general characterization of statistical dependence among the available measures.

## Hierarchical Clustering

Agglomerative hierarchical clustering constructs a dendrogram, a tree-structured representation of asset similarity, by iteratively merging the closest pairs of assets or clusters. Beginning with $N$ singleton clusters (one per asset), the algorithm proceeds as follows: at each step, the two clusters with the smallest inter-cluster distance are merged, and the distance matrix is updated. This process continues until all assets belong to a single cluster, producing a complete hierarchical decomposition.

The choice of linkage method determines how inter-cluster distance is computed from pairwise asset distances:

- **Single linkage** uses the minimum distance between any pair of assets across two clusters, tending to produce elongated, chain-like clusters.
- **Complete linkage** uses the maximum distance, producing compact, spherical clusters but being sensitive to outliers.
- **Average linkage** uses the mean distance, offering a balance between the two extremes.
- **Ward linkage** minimizes the total within-cluster variance at each merge, producing the most compact and balanced clusters. It is the most commonly used linkage method in portfolio applications.
- **Weighted linkage** assigns equal weight to each cluster regardless of size when computing inter-cluster distances.

The resulting dendrogram reveals the natural grouping structure of the asset universe: assets that cluster together at low merge distances are highly similar (close in the chosen distance metric), while those that merge only at high distances are dissimilar. This hierarchical structure is exploited by hierarchical risk parity, hierarchical equal risk contribution, and nested clusters optimization to produce allocations that respect asset relationships.

## Hierarchical Risk Parity

Hierarchical Risk Parity, introduced by Lopez de Prado (2016), addresses the fundamental instability of mean-variance optimization by replacing matrix inversion with a hierarchical allocation procedure. The algorithm proceeds in three steps.

**Step 1: Distance-based clustering.** Compute pairwise distances between all assets using the chosen distance measure (Pearson, Kendall, Spearman, distance correlation, or mutual information). Apply hierarchical clustering with the chosen linkage method to build a dendrogram encoding the asset similarity structure.

**Step 2: Quasi-diagonalization.** Reorder the rows and columns of the covariance matrix according to the leaf ordering of the dendrogram. This quasi-diagonalization places similar assets adjacent to one another, concentrating large covariance entries near the diagonal. The reordered matrix is not truly diagonal, but it is organized so that the hierarchical structure is geometrically apparent.

**Step 3: Recursive bisection.** Allocate risk top-down through the dendrogram. At the root, the full asset universe is split into the two clusters defined by the top-level merge. Capital is divided between these clusters inversely proportional to their respective variances:

$$\alpha = 1 - \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2}$$

The left cluster receives weight $\alpha$ and the right cluster receives weight $1 - \alpha$. This bisection is applied recursively at each level of the dendrogram until individual assets are reached, at which point each asset's weight is the product of all the allocation fractions along its path from the root.

The key advantages of HRP are substantial. First, no matrix inversion is required at any stage, eliminating the numerical instability that plagues mean-variance optimization when the covariance matrix is ill-conditioned or nearly singular. Second, the algorithm handles singular covariance matrices naturally, which arise whenever the number of assets exceeds the number of return observations. Third, by respecting the hierarchical clustering structure, HRP produces allocations that are more stable over time: small perturbations in the covariance matrix cause small changes in the dendrogram and hence small changes in weights, in contrast to the erratic weight swings of unconstrained mean-variance optimization.

HRP accepts parameters for the distance measure, clustering method, and risk measure used in the bisection step. The risk measure need not be variance; any of the measures discussed above can be used, enabling CVaR-based or drawdown-based hierarchical risk parity.

## Hierarchical Equal Risk Contribution

Hierarchical Equal Risk Contribution (HERC) extends HRP by replacing the simple inverse-variance bisection with an equal risk contribution allocation within each cluster. At each node in the dendrogram, rather than dividing capital proportional to inverse variance, the algorithm solves for weights such that each sub-cluster contributes equally to the risk of the parent cluster. This combines the stability advantages of hierarchical methods (no matrix inversion, respect for clustering structure, robustness to covariance estimation error) with the risk-parity properties of equal risk contribution.

The result is a multi-level risk parity: each cluster at every level of the hierarchy, and each asset within its cluster, contributes equally to total risk at its respective level. This produces allocations that are both stable (from the hierarchical structure) and risk-balanced (from the equal contribution principle). The diversification is achieved simultaneously across the hierarchical levels, ensuring that risk concentration does not arise at any scale.

HERC accepts the same parameterization options as HRP: distance measure, clustering method, and risk measure. The choice of risk measure determines what "equal risk contribution" means at each level, whether equal variance contribution, equal CVaR contribution, or equal drawdown contribution, among other possibilities.

## Nested Clusters Optimization

Nested Clusters Optimization (NCO) takes a different approach to exploiting hierarchical structure. Rather than allocating through recursive bisection, NCO uses the clustering to decompose a large optimization problem into smaller, better-conditioned sub-problems.

**Stage 1: Inner optimization.** For each cluster identified by the hierarchical clustering, an independent optimization is performed over the assets within that cluster. If a cluster contains $n_k$ assets, the inner optimization operates on an $n_k \times n_k$ covariance sub-matrix. Each cluster's optimization produces a set of intra-cluster weights, effectively reducing the cluster to a single composite asset with known return and risk characteristics.

**Stage 2: Outer optimization.** The composite assets from all $K$ clusters are collected, and a second optimization is performed over these $K$ composite assets to determine the inter-cluster allocation. The final weight of each individual asset is the product of its intra-cluster weight and its cluster's inter-cluster weight.

The dimensional reduction is the key benefit. Consider a universe of $N = 500$ assets partitioned into $K = 20$ clusters of approximately 25 assets each. The inner optimizations each handle $25 \times 25$ covariance matrices, and the outer optimization handles a $20 \times 20$ matrix. Both are well-conditioned even with limited return history, whereas direct optimization of the full $500 \times 500$ matrix may be severely ill-conditioned. This decomposition dramatically improves the numerical stability and statistical reliability of the optimization.

A further advantage of NCO is modularity: the inner and outer stages can use entirely different optimization strategies. For instance, the inner stage might use minimum variance to produce concentrated intra-cluster allocations, while the outer stage uses risk parity across clusters. This flexibility enables the practitioner to tailor the optimization approach to the characteristics of each hierarchical level.

## Regime-Driven Risk Adaptation

### Markov-Driven Risk Measure Selection

The risk measures discussed above need not be applied statically across all market environments. The Hidden Markov Model framework formalized in the moment estimation chapter provides filtered regime probabilities that can drive dynamic selection of the risk measure most appropriate to current conditions.

Given $S$ regimes with filtered probabilities $p(z_t = s \mid \mathbf{r}_{1:t})$ and a regime-specific risk measure mapping $\rho_s$, the effective portfolio risk at time $t$ is the probability-weighted combination:

$$
\rho_t(\mathbf{w}) = \sum_{s=1}^{S} p(z_t = s \mid \mathbf{r}_{1:t}) \, \rho_s(\mathbf{w})
$$

In a two-state model, the mapping might assign variance as the risk measure in the low-volatility state and CVaR at the 95\% confidence level in the high-volatility state:

$$
\rho_t(\mathbf{w}) = p(z_t = 1 \mid \mathbf{r}_{1:t}) \cdot \sigma^2(\mathbf{w}) \;+\; p(z_t = 2 \mid \mathbf{r}_{1:t}) \cdot \text{CVaR}_{0.05}(\mathbf{w})
$$

As the filtered probability of the crisis state increases, the effective risk measure transitions smoothly from variance toward CVaR, automatically increasing the portfolio's sensitivity to tail risk. The continuous nature of the filtered probabilities ensures that the transition is gradual rather than abrupt, preventing the portfolio instability that hard switching would produce.

When the Deep Markov Model framework is used instead of a discrete HMM, the continuous latent state $\mathbf{z}_t$ can parameterize the risk measure continuously. A mapping $\rho(\mathbf{w}; \mathbf{z}_t)$ that depends on the latent state enables risk sensitivity to vary along a continuum rather than switching between a finite set of predefined measures. In practice, this is achieved by defining a convex combination weight $\lambda(\mathbf{z}_t) \in [0, 1]$ (output by a neural network) that interpolates between two anchor risk measures.

### Regime-Conditional Risk Budgets

Regime probabilities also inform the calibration of risk budgets. Denoting by $\mathbf{b}_s$ the risk budget vector appropriate for regime $s$, the blended budget at time $t$ is:

$$
\mathbf{b}_t = \sum_{s=1}^{S} p(z_t = s \mid \mathbf{r}_{1:t}) \, \mathbf{b}_s
$$

In expansionary regimes, the budget might allocate greater risk to cyclical sectors with strong earnings momentum. In contractionary regimes, risk shifts toward defensive sectors with stable cash flows and lower economic sensitivity. The smooth blending ensures that portfolio restructuring at regime boundaries is gradual, limiting unnecessary turnover.

For a two-state model with sector groups $\{G_1, \ldots, G_K\}$, the regime-conditional budgets take the form:

$$
b_{k,1} = \frac{w_{k}^{\text{expansion}}}{\sum_{j} w_{j}^{\text{expansion}}}, \qquad b_{k,2} = \frac{w_{k}^{\text{contraction}}}{\sum_{j} w_{j}^{\text{contraction}}}
$$

where $w_k^{\text{expansion}}$ and $w_k^{\text{contraction}}$ reflect the desired risk allocation to sector group $k$ under each regime. The blended budget $\mathbf{b}_t$ then governs the risk budgeting optimization at each rebalancing date, producing allocations that rotate sector exposures in response to changing macroeconomic conditions without requiring manual intervention.

### Integration with Hierarchical Methods

Regime probabilities integrate naturally with the hierarchical methods discussed above. At each rebalancing date, the regime-conditional covariance matrix $\hat{\boldsymbol{\Sigma}}_t$ enters the distance computation and clustering step, producing a regime-adaptive dendrogram. Clusters that form during stress periods differ from those in calm periods, as correlations shift and factor structures change. By re-clustering at each rebalancing date using the current regime-conditional covariance, the hierarchical methods adapt their asset groupings to the prevailing market structure.

In HRP and HERC, the risk measure used in the bisection or equal-risk-contribution step can itself be regime-dependent: variance during calm regimes and CVaR during stress regimes. This produces allocations that are simultaneously hierarchically structured, risk-balanced, and regime-adaptive.

## LLM-Augmented Risk and Diversification

### Regime-Dependent Risk Measure Selection

The choice of risk measure need not be static. Different market regimes favor different risk measures, and large language models can provide the regime classification that drives dynamic risk measure selection. In calm, low-volatility environments, variance-based optimization may suffice, as return distributions are approximately symmetric and tail events are rare. When conditions deteriorate and tail risk becomes elevated (as indicated by widening credit spreads, rising implied volatility, or deteriorating economic indicators) switching to CVaR or EVaR captures the increased importance of extreme losses. In environments where drawdowns are the primary concern, perhaps due to leverage constraints or client sensitivity to capital impairment, CDaR or maximum drawdown becomes the appropriate objective.

An LLM processing current economic data, central bank communications, and market commentary can classify the prevailing regime along these dimensions and recommend the risk measure most appropriate for current conditions. This recommendation feeds into a dynamic risk measure selection policy: the optimization framework remains fixed, but the risk measure parameter rotates based on the LLM's regime assessment. The result is an adaptive portfolio construction process that selects the risk lens most relevant to the current environment.

### Stress Scenario Design

Traditional stress testing relies on replaying historical crisis episodes, an approach that is inherently backward-looking and may miss novel risk configurations. Large language models can complement historical analysis by designing forward-looking stress scenarios derived from current conditions. The process involves analyzing the current constellation of risk factors (leverage levels across sectors, valuation extremes, geopolitical tensions, policy uncertainty, supply chain vulnerabilities) and constructing conditional scenarios that specify how these factors might interact adversely.

For example, an LLM might identify that the combination of elevated corporate leverage and tightening monetary policy creates a specific vulnerability in credit-sensitive sectors, and design a stress scenario in which credit spreads widen sharply while equity markets decline and interest rates remain elevated. This scenario may not correspond to any single historical episode but is nonetheless plausible and relevant. The resulting stress test complements purely historical scenario analysis by introducing forward-looking, context-specific risk assessment.

The stress scenarios can be translated into return shocks applied to the portfolio, enabling the optimizer to evaluate how different allocations perform under adverse conditions that the LLM identifies as currently relevant. This integration produces stress tests that evolve with the risk environment rather than remaining anchored to past crises.

### Risk Budget Calibration from Sector Outlook

Custom risk budgets require the practitioner to specify how much risk each asset or sector should bear. Large language models can inform this specification by processing sector-level research, earnings trends, and macroeconomic sensitivity analysis. Sectors with favorable outlooks (supported by earnings momentum, favorable policy conditions, or structural demand growth) receive larger risk budgets, allowing the optimizer to allocate greater risk to these areas. Sectors facing headwinds (margin compression, regulatory pressure, or cyclical vulnerability) receive smaller risk budgets, constraining their contribution to total portfolio risk.

This process translates qualitative sector analysis into quantitative risk budget vectors that the risk budgeting optimizer can consume directly. The LLM serves as the bridge between unstructured research and structured optimization inputs, converting narrative assessments into the numerical budget vector $\mathbf{b}$ that parameterizes the risk budgeting problem. By updating these budgets as conditions evolve, the portfolio maintains an alignment between qualitative investment views and quantitative risk allocation that would be difficult to achieve through purely manual processes.

The combination of LLM-driven risk budgets with the full flexibility of the risk measure taxonomy creates a powerful adaptive framework: not only does the risk budget shift with the investment outlook, but the underlying risk measure can also rotate with the market regime, producing a portfolio that is responsive to both the level and the nature of risk in the current environment.

\newpage
