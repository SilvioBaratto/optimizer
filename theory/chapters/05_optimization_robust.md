# Portfolio Optimization and Robust Methods

With prior distributions constructed and risk measures selected, the optimization stage determines portfolio weights that best satisfy the investor's objectives subject to practical constraints. This chapter presents the optimization formulations, from classical mean-risk to distributionally robust methods, along with the constraint specifications, ensemble approaches, and naive baselines that form the complete optimization toolkit.

## Objective Functions

The choice of objective function determines how the optimizer trades off return against risk. Each formulation identifies a distinct point or region on the efficient frontier, and the investor's mandate dictates which is appropriate.

### Minimize Risk

The simplest formulation seeks the portfolio with the smallest possible risk exposure:

$$
\min_{\mathbf{w}} \; \rho(\mathbf{w})
$$

where $\rho$ denotes any convex risk measure (variance, CVaR, maximum drawdown, or any of the coherent measures discussed in prior chapters). This objective produces the global minimum risk point on the efficient frontier. Because no expected return estimate enters the formulation, the optimizer avoids the well-documented estimation error in $\boldsymbol{\mu}$ entirely.

### Maximize Return

When the investor has a fixed risk budget, the optimizer seeks the highest expected return that remains within that budget:

$$
\max_{\mathbf{w}} \; \mathbf{w}^\top \boldsymbol{\mu} \quad \text{subject to} \quad \rho(\mathbf{w}) \leq \rho_{\max}
$$

The constraint $\rho_{\max}$ represents the maximum tolerable risk level. This formulation traces the upper boundary of the feasible set for a given risk threshold.

### Maximize Utility

The utility-based formulation balances expected return against risk through a risk aversion parameter $\lambda$:

$$
\max_{\mathbf{w}} \; \mathbf{w}^\top \boldsymbol{\mu} - \frac{\lambda}{2} \rho(\mathbf{w})
$$

The parameter $\lambda > 0$ governs the tradeoff: larger values penalize risk more heavily, producing more conservative allocations. When $\rho(\mathbf{w}) = \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}$, this reduces to the classical Markowitz quadratic utility.

### Maximize Ratio

The ratio objective identifies the tangency portfolio, the allocation with the highest reward per unit of risk:

$$
\max_{\mathbf{w}} \; \frac{\mathbf{w}^\top \boldsymbol{\mu}}{\rho(\mathbf{w})}
$$

When $\rho$ is the portfolio standard deviation, this yields the maximum Sharpe ratio portfolio. When $\rho$ is the conditional value-at-risk, it yields the maximum CVaR ratio portfolio, and similarly for any other risk measure. The tangency portfolio is the point at which a ray from the origin is tangent to the efficient frontier.

## Constraints

Optimization without constraints produces mathematically elegant but practically infeasible portfolios. The constraint apparatus translates real-world investment mandates into mathematical restrictions on the weight vector $\mathbf{w}$.

### Position Limits

Box constraints bound each individual weight:

$$
w_i^{\min} \leq w_i \leq w_i^{\max} \quad \forall \; i = 1, \ldots, N
$$

The long-only constraint sets $w_i^{\min} = 0$, prohibiting short positions. Maximum position limits $w_i^{\max}$ (typically in the range $0.05$ to $0.10$ for diversified strategies) prevent excessive concentration in any single asset.

### Budget Constraint

The budget constraint governs the total capital deployment:

$$
\sum_{i=1}^{N} w_i = 1
$$

This enforces full investment. A relaxed variant permits partial investment:

$$
\sum_{i=1}^{N} w_i \leq B
$$

where $B \leq 1$ allows a cash position.

### Group and Sector Constraints

Group constraints impose bounds on the aggregate weight allocated to predefined asset groups (sectors, geographies, asset classes):

$$
g_s^{\min} \leq \sum_{i \in G_s} w_i \leq g_s^{\max} \quad \text{for each group } s
$$

where $G_s$ denotes the set of assets belonging to group $s$. Active weight bounds restrict deviations from a benchmark allocation:

$$
\Delta_s^{\min} \leq \sum_{i \in G_s} (w_i - w_{B,i}) \leq \Delta_s^{\max}
$$

where $w_{B,i}$ is the benchmark weight of asset $i$.

### Transaction Costs and Fees

Practical portfolio management incurs costs that must be internalized within the optimization. Transaction costs penalize turnover relative to the current portfolio $\mathbf{w}_0$:

$$
\sum_{i=1}^{N} c_i \, |w_i - w_{i,0}|
$$

where $c_i$ is the per-asset transaction cost (bid-ask spread plus commissions). Management fees impose a continuous drag proportional to position size:

$$
\sum_{i=1}^{N} f_i \, |w_i|
$$

where $f_i$ captures the expense ratio for ETFs or funds held within the portfolio. Both terms are integrated directly into the objective function, ensuring that the optimizer accounts for implementation costs when determining weights.

### Regularization

Regularization penalties improve out-of-sample stability by penalizing extreme or fragmented allocations.

The $L_1$ penalty (lasso regularization) encourages sparsity by driving small positions to exactly zero:

$$
\kappa_1 \|\mathbf{w}\|_1 = \kappa_1 \sum_{i=1}^{N} |w_i|
$$

The $L_2$ penalty (ridge regularization) shrinks extreme weights toward uniformity:

$$
\kappa_2 \|\mathbf{w}\|_2^2 = \kappa_2 \sum_{i=1}^{N} w_i^2
$$

The $L_2$ penalty is mathematically equivalent to adding $\kappa_2 \mathbf{I}$ to the covariance matrix, which improves its conditioning and reduces sensitivity to estimation noise. Typical values lie in the range $\kappa_2 \in [0.001, 0.1]$.

### Custom Linear Constraints

For investment mandates that transcend the standard constraint types, the framework supports general linear inequality constraints:

$$
\mathbf{A} \mathbf{w} \leq \mathbf{b}
$$

where $\mathbf{A} \in \mathbb{R}^{m \times N}$ and $\mathbf{b} \in \mathbb{R}^m$ encode $m$ linear restrictions. This general form subsumes turnover limits, tracking error bounds, portfolio beta constraints, and ESG score requirements as special cases.

## Robust Optimization Under Parameter Uncertainty

Classical mean-variance optimization treats $\hat{\boldsymbol{\mu}}$ and $\hat{\boldsymbol{\Sigma}}$ as known with certainty, yet these are merely estimates subject to substantial sampling error. Robust optimization acknowledges this uncertainty by optimizing against the worst case within a plausible set of parameter values.

### Mean Uncertainty Sets

The robust counterpart replaces the point estimate $\hat{\boldsymbol{\mu}}$ with an uncertainty set $\mathcal{U}_\mu$:

$$
\max_{\mathbf{w}} \; \min_{\boldsymbol{\mu} \in \mathcal{U}_\mu} \; \mathbf{w}^\top \boldsymbol{\mu} - \frac{\lambda}{2} \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}
$$

The inner minimization finds the least favorable expected return vector within the uncertainty set, while the outer maximization selects weights that perform best under this adversarial scenario.

The ellipsoidal uncertainty set takes the form:

$$
\mathcal{U}_\mu = \left\{ \boldsymbol{\mu} : (\boldsymbol{\mu} - \hat{\boldsymbol{\mu}})^\top \mathbf{S}_\mu^{-1} (\boldsymbol{\mu} - \hat{\boldsymbol{\mu}}) \leq \kappa^2 \right\}
$$

where $\mathbf{S}_\mu$ is the covariance of the estimator $\hat{\boldsymbol{\mu}}$ and $\kappa$ is calibrated from a confidence level (typically $90\%$ to $95\%$). The parameter $\kappa$ controls the size of the uncertainty region: larger values produce more conservative portfolios that hedge against greater estimation error.

Bootstrap-based approaches construct empirical confidence regions by resampling with replacement from the historical return series.

### Covariance Uncertainty Sets

Analogously, the covariance matrix is subject to estimation error, particularly in the off-diagonal elements that govern asset co-movements. Robust covariance uncertainty sets protect against misestimation of the risk structure by considering a neighborhood of plausible covariance matrices around $\hat{\boldsymbol{\Sigma}}$.

Bootstrap-based sampling constructs confidence regions for $\boldsymbol{\Sigma}$ by repeatedly resampling historical returns and computing the sample covariance for each resample. The resulting distribution of covariance estimates delineates the uncertainty region.

## Distributionally Robust CVaR

Distributionally robust optimization extends the uncertainty set concept from parameters to the entire return distribution. Rather than assuming a fixed distribution $P_0$ (typically the empirical distribution of historical returns), the optimizer considers all distributions within a Wasserstein ball centered on $P_0$:

$$
\min_{\mathbf{w}} \; \sup_{P \in \mathcal{B}_\epsilon(P_0)} \; \text{CVaR}_\alpha^P(\mathbf{w})
$$

where $\mathcal{B}_\epsilon(P_0)$ denotes the set of all probability distributions whose Wasserstein distance from the empirical distribution $P_0$ does not exceed $\epsilon$:

$$
\mathcal{B}_\epsilon(P_0) = \left\{ P : W(P, P_0) \leq \epsilon \right\}
$$

The radius $\epsilon$ (typically in the range $0.01$ to $0.05$) controls the degree of conservatism. A larger $\epsilon$ enlarges the ambiguity set, producing more defensive portfolios that are robust to greater distributional perturbations. This formulation provides protection against regime changes, fat tails, and model misspecification without requiring parametric distributional assumptions.

## Synthetic Data and Stress Testing

### Vine Copula Framework

Historical return samples are inherently limited in size and may fail to capture extreme events, future regime shifts, or tail dependencies that are plausible but unobserved. Vine copula models address this limitation by fitting flexible multivariate distributions that faithfully reproduce the statistical structure of asset returns.

The vine copula decomposition separates the multivariate distribution into three components:

1. **Marginal distributions** for each asset, fitted individually using flexible parametric families such as Student-$t$, Johnson SU, or Normal Inverse Gaussian distributions.
2. **Bivariate copulas** that capture pairwise dependence structures, selected from families including Gaussian, Student-$t$, Clayton, Gumbel, and Joe copulas. Each pair may use a different copula family, accommodating asymmetric and tail-dependent relationships.
3. **Vine structure** that decomposes the full $N$-dimensional dependence into a sequence of conditional bivariate relationships, organized as a tree structure where each level conditions on variables from previous levels.

This decomposition is both parsimonious and expressive: it requires only $\binom{N}{2}$ bivariate copula selections rather than direct specification of the full $N$-dimensional distribution.

### Scenario Generation

Given a fitted vine copula model, one can generate thousands of synthetic return scenarios that preserve the statistical properties of the historical data (including marginal shapes, pairwise dependence, tail dependence, and asymmetric co-movements) while extending well beyond the historical sample. These synthetic scenarios serve as inputs to scenario-based optimizers, enriching the information set available for portfolio construction.

### Conditional Stress Testing

Conditional simulation fixes one or more variables at specified stress levels and generates consistent scenarios for the remaining assets, conditioned on the stressed values:

$$
\mathbf{r}_{\text{synthetic}} \sim F\!\left(\mathbf{r} \mid r_{\text{market}} = -0.30\right)
$$

This produces scenarios where, for example, the broad market has declined by $30\%$, and all other asset returns are drawn from their conditional distribution given this market shock. The vine copula structure makes such conditional sampling tractable, as the conditional distributions decompose naturally along the vine tree.

Applications include factor crash scenarios (equity market drawdown, interest rate spikes), sector-specific stress tests, and correlation regime shifts where historical co-movements break down.

## Benchmark Tracking

Enhanced index strategies and constrained active mandates require portfolios that remain close to a benchmark while seeking modest outperformance. The benchmark tracking formulation minimizes tracking error:

$$
\min_{\mathbf{w}} \; \text{TE}(\mathbf{w}, \mathbf{w}_B) \quad \text{subject to} \quad \text{TE} \leq \text{TE}_{\text{target}}
$$

where the tracking error is defined as:

$$
\text{TE} = \sqrt{\text{Var}(r_P - r_B)} = \sqrt{(\mathbf{w} - \mathbf{w}_B)^\top \boldsymbol{\Sigma} (\mathbf{w} - \mathbf{w}_B)}
$$

and $r_P$, $r_B$ denote the portfolio and benchmark returns respectively. The target tracking error $\text{TE}_{\text{target}}$ constrains how far the portfolio may deviate from the benchmark in risk space, balancing the desire for active returns against mandate compliance.

## Naive Allocation Methods

Naive allocation methods require minimal or no parameter estimation and serve as essential baselines against which more sophisticated optimizers must demonstrate improvement.

### Equal Weighted

The equal-weighted portfolio assigns identical weight to each asset:

$$
w_i = \frac{1}{N} \quad \forall \; i = 1, \ldots, N
$$

This allocation requires no estimation whatsoever (neither expected returns nor covariances) and is therefore immune to estimation error. Despite its simplicity, the equal-weighted portfolio has proven to be a surprisingly competitive baseline, often outperforming mean-variance optimized portfolios on an out-of-sample basis when the number of assets is moderate and estimation windows are short.

### Inverse Volatility

The inverse volatility portfolio scales each position inversely to its estimated volatility:

$$
w_i = \frac{1/\sigma_i}{\sum_{j=1}^{N} 1/\sigma_j}
$$

This allocation assigns larger weights to less volatile assets, implementing a rudimentary form of risk-based allocation. It is most appropriate when assets have similar Sharpe ratios, in which case equalizing risk contributions approximates the tangency portfolio. Since only marginal volatilities (not the full covariance matrix) are required, estimation error is substantially reduced relative to mean-variance optimization.

## Ensemble Optimization

Model uncertainty is pervasive in portfolio optimization: different formulations, risk measures, and estimation methods yield different weight vectors, and no single approach dominates across all market conditions. Ensemble methods address this by combining multiple optimization strategies to reduce model risk.

Stacking, the primary ensemble approach, proceeds in three stages:

1. **Independent optimization.** A collection of diverse optimizers is run independently, each embodying different modeling assumptions. For example, one may use mean-risk optimization with Sharpe ratio maximization, another hierarchical risk parity, and a third risk budgeting with equal risk contributions.

2. **Sub-portfolio construction.** Each optimizer's output weight vector defines a sub-portfolio. These sub-portfolios span a range of risk-return characteristics reflecting the diversity of underlying assumptions.

3. **Meta-optimization.** A final meta-optimizer allocates capital across the sub-portfolios, treating each as a single composite asset. This meta-level allocation diversifies across model assumptions: if mean-variance overestimates expected returns while hierarchical risk parity is overly conservative, the stacking framework blends both, reducing the impact of any single model's errors.

\newpage
