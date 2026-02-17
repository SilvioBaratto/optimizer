# View Integration and Bayesian Updating

Raw moment estimates, however carefully constructed, reflect only backward-looking statistical patterns. Active portfolio management requires incorporating forward-looking views: beliefs about expected returns, volatilities, correlations, or tail behavior that deviate from what history alone implies. This chapter presents three complementary frameworks for view integration and establishes LLM-driven view generation as a systematic approach to populating these frameworks.

## The Black-Litterman Framework

### The Foundational Problem

Traditional **mean-variance optimization** suffers from extreme sensitivity to expected return estimates. Small changes in forecasted returns produce dramatically different portfolios; the optimizer aggressively exploits estimation error by taking extreme positions in assets with overstated returns. The resulting allocations are unstable, unintuitive, and dominated by noise rather than signal.

**Black and Litterman (1992)** resolved this instability through Bayesian inference: rather than treating expected returns as known inputs to be estimated directly, the framework begins with **equilibrium returns** as a neutral prior distribution and then incorporates **active views** with explicit confidence levels. The posterior distribution over expected returns blends market-implied information with the investor's subjective or model-driven beliefs, producing portfolios that are stable, diversified, and responsive to conviction in proportion to its precision.

### The Equilibrium Prior

Market equilibrium returns $\boldsymbol{\Pi}$ represent the expected returns implied by current market capitalization weights under the assumption that markets are in equilibrium:

$$
\boldsymbol{\Pi} = \delta \boldsymbol{\Sigma} \mathbf{w}_{\text{mkt}}
$$

where:

- $\delta$ is the **risk aversion coefficient**, reflecting the aggregate investor's trade-off between expected return and variance
- $\boldsymbol{\Sigma}$ is the **covariance matrix** of asset returns
- $\mathbf{w}_{\text{mkt}}$ is the vector of **market capitalization weights**, serving as the prior portfolio

The interpretation is straightforward: if all investors held the market portfolio with risk aversion $\delta$, equilibrium expected returns must satisfy the reverse optimization equation above. This provides a **neutral starting point** reflecting collective market wisdom rather than any individual forecast. The equilibrium prior anchors the model, ensuring that in the absence of active views, the resulting portfolio coincides with the market portfolio.

**Risk aversion estimation** derives from observable market quantities:

$$
\delta = \frac{E[R_{\text{mkt}}] - R_f}{\sigma_{\text{mkt}}^2}
$$

where $E[R_{\text{mkt}}] - R_f$ is the equity risk premium and $\sigma_{\text{mkt}}^2$ is market variance.

### Active View Specification

Investors express **active views** through the linear constraint system:

$$
\mathbf{P}\boldsymbol{\mu} = \mathbf{Q} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Omega})
$$

where:

- $\mathbf{P}$ is a $K \times N$ **pick matrix** defining which assets each view concerns
- $\mathbf{Q}$ is a $K \times 1$ vector of **view returns** (expected outcomes)
- $\boldsymbol{\Omega}$ is a $K \times K$ **view uncertainty matrix** (diagonal for independent views)

Three canonical view types arise from different structures of $\mathbf{P}$:

**Absolute views** specify an expected return for a single asset. The corresponding row of $\mathbf{P}$ contains a single entry of one in the column of the target asset and zeros elsewhere, with $Q$ equal to the expected return:

$$
\mathbf{P} = [0 \; \cdots \; 0 \; 1 \; 0 \; \cdots \; 0], \quad Q = q
$$

For example, an absolute view that a particular equity will return ten percent annualized places one in its column and sets $Q = 0.10$.

**Relative views** express an expected outperformance of one asset over another. The pick matrix contains $+1$ for the outperforming asset and $-1$ for the underperforming asset:

$$
\mathbf{P} = [0 \; \cdots \; 1_{(i)} \; \cdots \; -1_{(j)} \; \cdots \; 0], \quad Q = q
$$

where $q$ represents the expected return differential.

**Basket views** extend relative views to groups of assets. The pick matrix assigns fractional weights to group members, enabling views on sector-level or theme-level performance:

$$
\mathbf{P} = \left[\frac{1}{n_1} \; \cdots \; \frac{1}{n_1} \; 0 \; \cdots \; 0\right], \quad Q = q
$$

Views can be specified using **string expressions** that are automatically parsed into the $\mathbf{P}$ and $\mathbf{Q}$ matrices. This design permits programmatic view generation, an essential capability for LLM-driven pipelines that produce views as structured text.

### Bayesian Posterior Returns

The **Black-Litterman posterior expected returns** combine equilibrium and views via Bayesian updating:

$$
\mathbb{E}[\boldsymbol{\mu}] = \bar{\boldsymbol{\mu}} = \left[(\tau\boldsymbol{\Sigma})^{-1} + \mathbf{P}^\top\boldsymbol{\Omega}^{-1}\mathbf{P}\right]^{-1} \left[(\tau\boldsymbol{\Sigma})^{-1}\boldsymbol{\Pi} + \mathbf{P}^\top\boldsymbol{\Omega}^{-1}\mathbf{Q}\right]
$$

with posterior uncertainty:

$$
\mathbf{M} = \left[(\tau\boldsymbol{\Sigma})^{-1} + \mathbf{P}^\top\boldsymbol{\Omega}^{-1}\mathbf{P}\right]^{-1}
$$

The parameter $\tau$ is the **uncertainty scaling parameter**, typically set between $0.025$ and $0.05$, representing the relative uncertainty in equilibrium return estimates compared to the covariance matrix. A smaller $\tau$ implies greater confidence in the equilibrium prior, anchoring the posterior more firmly to market-implied returns.

The **intuition** behind the posterior formula is that of a precision-weighted average. The precision of the equilibrium prior is $(\tau\boldsymbol{\Sigma})^{-1}$ and the precision of the views is $\mathbf{P}^\top\boldsymbol{\Omega}^{-1}\mathbf{P}$. The posterior combines these two sources of information in proportion to their respective precisions:

- **High view confidence** (small $\boldsymbol{\Omega}$): posterior returns tilt strongly toward the active views
- **Low view confidence** (large $\boldsymbol{\Omega}$): posterior returns remain close to the equilibrium prior
- **No views**: posterior returns reduce exactly to the equilibrium $\boldsymbol{\Pi}$

This graceful degradation ensures that the framework never produces worse results than the equilibrium baseline, regardless of view quality.

### View Uncertainty Calibration

The specification of the view uncertainty matrix $\boldsymbol{\Omega}$ is critical to the practical success of the Black-Litterman framework. Three approaches, each with distinct advantages, have gained prominence.

The **He-Litterman proportional approach** sets view uncertainty proportional to the variance of the view portfolio:

$$
\omega_k = \tau \cdot \mathbf{P}_k^\top \boldsymbol{\Sigma} \mathbf{P}_k
$$

This ensures that views on volatile assets or asset combinations carry proportionally more uncertainty, preventing the optimizer from over-reacting to views on inherently noisy assets. The approach requires no subjective inputs beyond $\tau$ itself.

The **Idzorek (2005) extension** introduces an explicit view strength parameter $\alpha_k$:

$$
\omega_k = \frac{1}{\alpha_k} \cdot \mathbf{P}_k^\top \boldsymbol{\Sigma} \mathbf{P}_k
$$

where $\alpha_k \in [0, \infty)$ calibrates the degree to which each view influences the posterior:

- $\alpha_k = 1.0$: full confidence, the view is treated as effectively certain
- $\alpha_k = 0.5$: moderate confidence, the view produces a meaningful but tempered tilt
- $\alpha_k = 0.1$: minimal confidence, the view produces only a marginal adjustment

This parameterization is particularly amenable to LLM-driven confidence calibration, as the strength parameter maps naturally to a continuous confidence score.

**Direct specification from forecast error** derives view uncertainty from the historical accuracy of the view-generating process:

$$
\omega_k = \text{Var}(Q_k - \text{Realized Return}_k)
$$

estimated from a track record of past views and their outcomes. This approach is the most principled when sufficient historical data exists, as it directly captures the empirical reliability of the forecasting methodology.

### Black-Litterman Factor Model

Rather than specifying views on individual assets, the factor model variant targets **factor returns**:

$$
\mathbf{P}_f \boldsymbol{\mu}_f = \mathbf{Q}_f + \boldsymbol{\epsilon}_f
$$

where $\boldsymbol{\mu}_f$ is the vector of expected factor returns. A view that "the momentum factor will return twelve percent annually" or "quality will outperform value by four percent" operates in the factor space. Asset-level expectations are then derived through the factor loading matrix:

$$
\bar{\boldsymbol{\mu}} = \mathbf{B}\bar{\boldsymbol{\mu}}_f
$$

where $\mathbf{B}$ is the $N \times K$ matrix of asset-to-factor loadings. This approach offers three advantages. First, it is more **parsimonious**: a small number of factor views implicitly generates views on all assets through their factor exposures. Second, it is more **stable**: factors are better-estimated than individual asset returns, producing smoother posterior distributions. Third, it is more **interpretable**: views on well-defined investment themes (value, momentum, quality, size) are easier to formulate, justify, and communicate than views on hundreds of individual securities.

This architecture is realized by combining a factor model with a Black-Litterman prior applied at the factor level, so that views on factors propagate to asset-level expectations through the loading matrix.

## Entropy Pooling: Non-Linear View Integration

### Limitations of Classical Black-Litterman

The classical Black-Litterman framework, for all its elegance, restricts views to **linear functions of expected returns** ($\mathbf{P}\boldsymbol{\mu} = \mathbf{Q}$) and assumes Gaussian uncertainty on those views. This formulation excludes a wide range of practically important beliefs:

- Views on **variance**: "volatility of asset $i$ will double over the next quarter"
- Views on **correlation**: "the correlation between assets $i$ and $j$ will increase to 0.80"
- Views on **skewness**: "asset $i$ will exhibit negative skew due to downside tail risk"
- Views on **tail risk**: "the 95\% CVaR of asset $i$ will reach eight percent"

These limitations motivate a more general framework capable of incorporating views on arbitrary distributional properties.

### Minimum Kullback-Leibler Divergence Framework

**Entropy Pooling**, introduced by Meucci (2008), overcomes these limitations by adjusting **scenario probabilities** rather than modifying return parameters directly. Given $S$ historical or simulated scenarios with baseline (equal) probabilities $\mathbf{p}_0 = (1/S, \ldots, 1/S)$, the framework seeks new probabilities $\mathbf{p}^*$ that satisfy the investor's view constraints while remaining as close as possible to the prior distribution.

The optimization problem minimizes the **Kullback-Leibler divergence** between the new and prior probability vectors:

$$
\mathbf{p}^* = \arg\min_{\mathbf{p}} \sum_{s=1}^{S} p_s \ln\left(\frac{p_s}{p_{0,s}}\right)
$$

subject to the normalization constraint $\sum_s p_s = 1$, non-negativity $p_s \geq 0$, and arbitrary view constraints expressed as moment conditions on the scenario-weighted distribution.

The Kullback-Leibler divergence provides the natural measure of **information loss**: among all probability distributions satisfying the view constraints, the solution $\mathbf{p}^*$ is the one that introduces the least additional structure beyond what the views require. This principle of minimum relative entropy ensures that the posterior distribution reflects the views and nothing more; no spurious correlations or distributional artifacts are introduced.

### View Types Supported

Entropy Pooling accommodates views on any moment or distributional property that can be expressed as a constraint on scenario probabilities. The following catalogue covers the principal view types, each with its mathematical constraint formulation.

**Mean views** on expected returns take the form of equality or inequality constraints:

$$
\sum_{s=1}^{S} p_s \cdot r_{i,s} = \mu_i^{\text{view}} \quad \text{(equality)}, \qquad \sum_{s=1}^{S} p_s \cdot r_{i,s} \geq \mu_i^{\text{lower}} \quad \text{(inequality)}
$$

Equality views fix the expected return precisely, while inequality views set a floor or ceiling, allowing the optimizer to determine the exact level that minimizes information loss.

**Variance views** constrain the second central moment:

$$
\sum_{s=1}^{S} p_s \cdot (r_{i,s} - \bar{r}_i)^2 = \sigma_i^{2,\text{view}}
$$

enabling beliefs about future volatility levels to be incorporated without altering expected returns or correlations beyond what the variance constraint requires.

**Correlation views** constrain the normalized cross-moment between two assets:

$$
\frac{\sum_s p_s (r_{i,s} - \bar{r}_i)(r_{j,s} - \bar{r}_j)}{\sigma_i^{\text{view}} \cdot \sigma_j^{\text{view}}} = \rho_{ij}^{\text{view}}
$$

This is particularly valuable for expressing beliefs about regime changes in co-movement structure; for instance, that correlations will converge toward one during a stress period.

**Skewness views** constrain the standardized third moment:

$$
\sum_{s=1}^{S} p_s \cdot \left(\frac{r_{i,s} - \bar{r}_i}{\sigma_i}\right)^3 = \gamma_i^{\text{view}}
$$

allowing the investor to express beliefs about distributional asymmetry: negative skew for assets facing downside tail risk, positive skew for assets with optionality or convex payoff structures.

**CVaR views** constrain the expected loss in the tail:

$$
-\frac{1}{\alpha} \sum_{s \in \text{worst } \alpha S} p_s \cdot r_{i,s} = \text{CVaR}_i^{\text{view}}
$$

where $\alpha$ defines the tail probability (typically five percent). This enables direct expression of tail risk beliefs without parametric distributional assumptions.

**Group views** aggregate across asset classifications:

$$
\sum_{i \in G} w_i \cdot \bar{r}_i^{\text{view}} = \mu_G^{\text{view}}
$$

where groups $G$ can represent sectors, themes, factor exposures, or any other asset classification. Group views are natural for top-down allocation, where the investor has convictions about broad categories rather than individual securities.

### Views Relative to Prior

A particularly powerful feature of Entropy Pooling allows expressing views **relative to the prior distribution**, using multiplicative adjustments without specifying absolute values:

- "Variance of asset $i$ will be four times its prior level"
- "Correlation between assets $i$ and $j$ will be half the prior level"
- "Expected return of asset $i$ will be twenty percent above its prior level"

Relative views eliminate the need to compute absolute moment values manually, substantially reducing calibration error. The investor need only specify the **direction and magnitude of deviation** from the current distributional estimate, which is often a more natural and robust form of belief expression than absolute forecasts.

## Opinion Pooling: Combining Multiple Expert Views

### The Multi-Expert Problem

When multiple sources generate views (quantitative models, fundamental analysis, macro strategies, LLM-based signals) their opinions must be combined into a single coherent input for the optimization framework. Each source may possess different accuracy characteristics, different domains of expertise, and potentially conflicting conclusions. Naive approaches such as simple averaging ignore these heterogeneities, while ad hoc reconciliation introduces subjective bias and lacks theoretical grounding.

### Consensus Distribution with Credibility Weights

**Opinion Pooling** provides a principled framework for multi-expert combination. Given $M$ expert posterior distributions $\{P_1, P_2, \ldots, P_M\}$ and a base prior $P_0$, the consensus distribution is formed as:

$$
P^* = \left(1 - \sum_{m=1}^{M} \pi_m\right) P_0 + \sum_{m=1}^{M} \pi_m P_m
$$

where $\pi_m \in [0, 1]$ represents the **credibility weight** assigned to expert $m$, and the constraint $\sum_m \pi_m \leq 1$ ensures the base prior retains residual influence. The base prior acts as a shrinkage target, anchoring the consensus toward the equilibrium or empirical distribution when expert opinions are uncertain or contradictory.

The credibility weights $\pi_m$ govern the influence of each expert on the final distribution. When $\sum_m \pi_m = 1$, the base prior receives zero weight and the consensus is a pure mixture of expert opinions. When $\sum_m \pi_m < 1$, the residual weight $1 - \sum_m \pi_m$ accrues to the base prior, providing a regularization effect that prevents any single expert from dominating the consensus.

### Expert Weight Calibration

Calibration of expert weights $\pi_m$ can proceed along several dimensions:

**Historical accuracy** provides the most direct calibration signal. By tracking each expert's information coefficient (the correlation between forecasted and realized returns) over time, the weight assignment reflects demonstrated predictive ability. Experts with consistently higher information coefficients receive larger weights.

**Domain relevance** adjusts weights according to the scope of each expert's competence. A macro-focused model may receive higher weight for interest-rate-sensitive assets but lower weight for idiosyncratic stock-level views. Conversely, a bottom-up fundamental model may be weighted more heavily for individual security views than for broad sector or factor calls.

**Source diversification** penalizes redundant signals from correlated information sources. If two experts rely on overlapping data or similar methodologies, their combined weight should reflect their incremental (not total) information content. This prevents double-counting and ensures the consensus benefits from genuinely diverse perspectives.

### Conflicting View Resolution

The Opinion Pooling framework handles disagreement naturally, without requiring manual reconciliation. If one expert is bullish and another bearish on the same asset, the consensus distribution reflects the **probability-weighted blend** of their views. The base prior acts as an anchor when experts disagree sharply, tempering the consensus toward the neutral equilibrium estimate.

This automatic conflict resolution is a significant advantage over sequential or hierarchical view integration schemes, where the order of incorporation can influence the final result. Under Opinion Pooling, all experts contribute simultaneously, and the consensus depends only on their respective credibility weights and posterior distributions.

## LLM-Driven View Generation and Integration

### Structured View Generation from Multi-Factor Analysis

Large language models introduce a qualitatively new approach to view generation by processing **comprehensive multi-dimensional data** for each asset in the investment universe. The analytical dimensions span:

**Valuation**: price-to-earnings, price-to-book, price-to-sales ratios compared against sector norms and historical distributions. Persistent deviations from fair value generate mean-reversion views; justified deviations (supported by growth or quality differentials) produce continuation views.

**Momentum**: three-month, six-month, and twelve-month returns, relative strength index, and trend strength metrics. The LLM evaluates whether momentum signals reflect genuine information diffusion or speculative excess, distinguishing sustainable trends from fragile overextensions.

**Quality**: return on equity, profit margins, cash flow stability, and financial leverage. High-quality firms with durable competitive advantages generate positive expected return adjustments, while deteriorating quality metrics trigger negative views.

**Growth**: revenue and earnings growth rates, forward guidance trajectories, and estimate revision trends. The LLM assesses whether current growth rates are sustainable, accelerating, or decelerating, translating the assessment into directional views.

**Technical**: support and resistance levels, volume patterns, and market microstructure signals. These shorter-horizon indicators modulate view confidence rather than generating primary directional views.

**Analyst consensus**: ratings distributions, price target dispersions, and estimate revision momentum. The LLM synthesizes consensus information while detecting potential herding or stale estimates.

The output of this multi-factor analysis is a **structured view** for each asset, comprising an expected return estimate, a confidence level, and a natural-language rationale. The confidence calibration follows a systematic scale:

- **High confidence** ($0.8$ to $1.0$): all analytical factors aligned, clear identifiable catalyst, strong data quality
- **Medium confidence** ($0.6$ to $0.8$): four to five factors aligned, some residual uncertainty, mixed technical signals
- **Low confidence** ($0.0$ to $0.6$): mixed or conflicting signals, data gaps, contradictory indicators across dimensions

These structured views map directly into the Black-Litterman $\mathbf{Q}$ vector (expected returns) and $\boldsymbol{\Omega}$ matrix (uncertainty), with the confidence score governing the Idzorek-style view strength parameter $\alpha_k$.

### Macroeconomic Regime Classification and Regime-Adjusted Confidence

LLMs contribute to view integration not only through individual asset analysis but through **macroeconomic regime classification**, which calibrates the global parameters of the optimization framework.

The classification processes a broad set of leading indicators:

- Manufacturing purchasing managers' indices and their diffusion components
- Yield curve slope and term structure dynamics
- Credit spreads and their rate of change
- Unemployment claims and labor market breadth indicators
- Corporate earnings trends and profit margin trajectories
- Central bank communications and forward guidance shifts
- Commodity price dynamics and supply-demand balances

From these inputs, the LLM assigns the current environment to one of four canonical **business cycle phases**:

1. **Early-cycle (recovery)**: leading indicators inflecting upward, credit conditions easing, earnings troughing
2. **Mid-cycle (expansion)**: broad-based growth, moderate inflation, stable credit conditions
3. **Late-cycle (slowdown)**: leading indicators peaking, credit tightening, margin compression beginning
4. **Recession (contraction)**: negative output growth, rising unemployment, credit stress, earnings declining

Each phase implies distinct **regime-adjusted parameters** for the optimization framework:

**Risk aversion $\delta$** scales with the business cycle: higher values in late-cycle and recession phases reflect more conservative risk preferences and the empirical observation that risk premia widen during downturns. Lower values in early-cycle phases permit more aggressive positioning when the risk-reward balance favors risk-taking.

**Uncertainty $\tau$** scales with market volatility relative to its long-term average. Elevated realized and implied volatility during stress periods widens the uncertainty band around equilibrium returns, causing the posterior to anchor more firmly to the prior and reducing the influence of active views, a prudent response when forecast reliability deteriorates.

**Factor emphasis** shifts across the cycle: quality and value factors receive greater weight in late-cycle and recessionary environments, where capital preservation and fundamental soundness dominate. Momentum and growth factors receive greater emphasis in early-cycle and expansion phases, where trend persistence and earnings acceleration drive returns.

The regime classification feeds directly into Black-Litterman prior calibration, adjusting $\delta$, $\tau$, and the relative weighting of factor-level views according to the identified phase.

The LLM-based regime classification discussed here can be complemented by the formal Hidden Markov Model framework developed in the moment estimation chapter. HMM filtered probabilities provide a quantitative, data-driven regime indicator that serves as an input to, or a cross-check on, the LLM's qualitative assessment. When both approaches agree on the prevailing regime, the resulting parameter adjustments carry greater conviction. When they disagree, the practitioner can increase uncertainty parameters ($\tau$, $\boldsymbol{\Omega}$) to reflect the ambiguity, a conservative response appropriate for periods of conflicting signals. Deep Markov Models extend this further by providing continuous regime characterizations that capture subtler transitions than the four discrete phases above.

### Multi-LLM Opinion Pooling

Different prompts, models, or analytical frameworks applied to the same data function as **independent experts**, each producing a distinct set of views with associated confidence levels. This multiplicity maps naturally to the Opinion Pooling framework:

- Each LLM-expert generates a posterior distribution over asset returns (views with confidences)
- The opinion pooling estimator combines these posteriors with calibrated credibility weights $\pi_m$
- The base prior $P_0$ absorbs residual weight, ensuring stability when LLM-experts disagree

The principal benefit of multi-LLM opinion pooling is **robustness to individual model biases**. Any single language model may exhibit systematic tendencies: recency bias, anchoring to salient narratives, or over-weighting certain data types. By combining multiple models with diverse analytical orientations, these biases partially cancel, and the consensus captures a broader range of analytical perspectives.

**Weight calibration** for LLM-experts follows the same principles as for traditional expert combination. Each model's historical accuracy is tracked through its information coefficient over successive rebalancing periods. Models demonstrating consistently higher predictive accuracy receive larger credibility weights, while models with poor or deteriorating track records are down-weighted. This adaptive calibration ensures that the opinion pool evolves toward reliance on the most reliable analytical sources.

### News Sentiment as View Confidence Modifier

LLMs process news feeds, earnings call transcripts, and analyst reports to extract **sentiment signals** that modulate view confidence rather than serving as primary view sources. The distinction is important: sentiment is treated as second-order information that adjusts the precision of existing views, not as a direct input to expected return estimation.

The sentiment-confidence interaction operates through several channels:

**Reinforcement**: strong positive sentiment aligned with a bullish fundamental view increases the confidence parameter $\alpha_k$, tightening the view uncertainty $\omega_k$ and amplifying the view's influence on the posterior. The rationale is that when both quantitative signals and qualitative narrative point in the same direction, the probability of the view being correct is higher.

**Contradiction**: negative sentiment opposing a bullish fundamental view decreases confidence, widening $\omega_k$ and attenuating the view's effect. Similarly, positive sentiment opposing a bearish view reduces the conviction in the negative outlook. The framework does not discard the view entirely but softens its impact in proportion to the degree of contradiction.

**Ambiguity**: when sentiment signals are mixed or inconclusive, overall confidence is reduced across all views for the affected asset. This conservative response reflects the increased uncertainty that accompanies conflicting information sources.

**Temporal decay** governs the weighting of sentiment signals over time. Recent news and earnings releases receive substantially more weight than older coverage, reflecting the rapid incorporation of information into market prices. A half-life parameter controls the rate of decay, with typical values ranging from one to four weeks depending on the asset's information environment and liquidity.

\newpage
