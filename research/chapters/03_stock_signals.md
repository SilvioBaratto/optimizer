# Stock Signal Generation

The generation of stock-level investment signals represents the core analytical engine that translates raw financial data and macroeconomic context into actionable investment recommendations. This chapter presents a comprehensive mathematical framework for multi-factor signal construction, employing institutional-grade cross-sectional standardization techniques that ensure robust, regime-adaptive portfolio signals. The methodology integrates fundamental analysis, technical metrics, and macroeconomic regime classifications through a sophisticated seven-pass processing pipeline that achieves statistical rigor while maintaining computational efficiency.

## Theoretical Foundations

### Multi-Factor Asset Pricing Framework

The signal generation methodology builds upon established asset pricing theory, extending the Fama-French factor models with contemporary enhancements suited for practical portfolio construction. The fundamental signal generation function can be expressed as:

$$
S_i = f(F_{\text{value},i}, F_{\text{momentum},i}, F_{\text{quality},i}, F_{\text{growth},i}, R_{\text{macro}}, A_{\text{risk},i})
$$

where $S_i$ represents the investment signal for stock $i$, $F$ denotes factor exposures, $R_{\text{macro}}$ captures macroeconomic regime effects, and $A_{\text{risk}}$ incorporates stock-specific risk adjustments.

The four-factor structure reflects both theoretical motivation and empirical evidence:

**Value Factor**: Captures the tendency of undervalued securities to outperform over medium-term horizons, drawing from fundamental principles that market prices eventually converge to intrinsic values. Extensive empirical evidence demonstrates persistent value premiums across markets and time periods.

**Momentum Factor**: Exploits the empirical observation that securities exhibiting strong past performance tend to continue outperforming over 6-12 month horizons, a pattern documented across asset classes. The momentum anomaly reflects behavioral biases and gradual information diffusion.

**Quality Factor**: Identifies companies with superior operational efficiency, profitability, and financial health. Quality factors provide defensive characteristics during market stress while maintaining participation in bull markets, making them valuable diversifiers.

**Growth Factor**: Measures the trajectory of fundamental business metrics, incorporating both trailing performance and forward-looking expectations. Growth signals capture companies expanding market share, revenue, and profitability.

The equal-weighting baseline (25% allocation to each factor) reflects the institutional practice of diversification across return sources in the absence of strong prior beliefs about factor timing.

### Cross-Sectional Standardization Paradigm

A fundamental design choice distinguishes between time-series and cross-sectional perspectives. The framework explicitly adopts **cross-sectional ranking**, asking not "Has stock $i$ performed well?" but rather "Is stock $i$ in the top quintile relative to all available securities?"

For any metric $X$, the z-score transformation employs cross-sectional statistics:

$$
z_{i,t} = \frac{X_{i,t} - \mu_{X,t}}{\sigma_{X,t}}
$$

where $\mu_{X,t}$ and $\sigma_{X,t}$ represent the mean and standard deviation computed across all stocks in the universe at time $t$. This cross-sectional approach ensures:

1. **Regime Invariance**: Rankings remain meaningful whether the market is rising, falling, or sideways
2. **Market Neutrality**: Signals identify relative rather than absolute attractiveness
3. **Temporal Consistency**: Z-scores maintain comparable scale across different market environments

The cross-sectional paradigm contrasts with time-series standardization, which would compare each stock's current metrics to its own historical distribution. While time-series approaches can capture mean-reversion within individual securities, they fail to provide the cross-sectional rankings essential for portfolio construction.

### Statistical Challenges in Multi-Stock Universes

Applying cross-sectional standardization to thousands of securities introduces statistical complications absent in single-asset analysis. The presence of extreme outliers—distressed companies, acquisition targets, or data errors—can severely contaminate sample statistics if handled naively.

Consider a universe of 1,000 stocks where Book-to-Price ratios range from 0.10 to 0.60 for 99.5% of observations, but five distressed companies exhibit ratios exceeding 10.0 due to market capitalizations approaching zero. Computing the sample mean and standard deviation from this contaminated distribution yields:

$$
\mu_{\text{naive}} = \frac{1}{1000}\sum_{i=1}^{1000} X_i \approx 0.30 \text{ (heavily inflated)}
$$

$$
\sigma_{\text{naive}} = \sqrt{\frac{1}{999}\sum_{i=1}^{1000}(X_i - \mu)^2} \approx 1.2 \text{ (severely inflated)}
$$

These contaminated statistics render standardization meaningless: stocks with Book-to-Price of 0.50 (genuinely high) would receive z-scores near zero, while truly typical values near 0.25 would appear artificially depressed.

The solution requires **robust statistical estimation** that identifies and mitigates outlier influence before standardization. The iterative approach employed achieves this through successive refinement, a problem that motivates the multi-pass architecture detailed in Section 3.

## Factor Construction Methodology

### Value Factor: Inverse Price Multiples

The value factor aggregates four complementary valuation metrics, each expressed as the inverse of traditional price multiples to ensure directional consistency (higher values indicate cheaper valuations):

$$
F_{\text{value},i} = \frac{1}{4}\sum_{j \in \{B/P, E/P, S/P, D/P\}} z_{j,i}
$$

**Book-to-Price Ratio ($B/P$)**: Computed as the reciprocal of the Price-to-Book ratio, measuring market capitalization relative to accounting book value. The metric anchors to balance sheet fundamentals:

$$
B/P_i = \frac{\text{Book Value per Share}_i}{\text{Price per Share}_i}
$$

Historical S&P 500 norms establish $\mu_{B/P} = 0.25$ (implying average P/B of 4.0) with $\sigma_{B/P} = 0.15$. When cross-sectional standardization is enabled, these static norms are replaced by universe-specific statistics computed in Pass 1.5.

**Earnings-to-Price Ratio ($E/P$)**: The reciprocal of the P/E ratio, relating annual earnings to market valuation:

$$
E/P_i = \frac{\text{EPS}_i}{\text{Price per Share}_i}
$$

with market norms $\mu_{E/P} = 0.05$ (P/E of 20) and $\sigma_{E/P} = 0.03$. This metric captures profitability-based valuation but can be undefined or misleading for companies with negative or cyclically depressed earnings.

**Sales-to-Price Ratio ($S/P$)**: Evaluates revenue relative to market capitalization:

$$
S/P_i = \frac{\text{Revenue per Share}_i}{\text{Price per Share}_i}
$$

with typical parameters $\mu_{S/P} = 0.50$ (P/S of 2.0) and $\sigma_{S/P} = 0.30$. Unlike earnings-based metrics, sales-based valuation remains defined for unprofitable companies and provides more stable signals during earnings volatility.

**Dividend Yield ($D/P$)**: Measures annual cash distributions relative to price:

$$
D/P_i = \frac{\text{Annual Dividend per Share}_i}{\text{Price per Share}_i}
$$

Historical norms: $\mu_{D/P} = 0.02$ (2% yield), $\sigma_{D/P} = 0.015$. Approximately 40% of stocks in typical universes pay no dividend, particularly growth-oriented technology companies. The framework handles missing dividends by computing value scores from available metrics only, avoiding imputation that would bias signals.

The equal-weighting across available metrics (typically 3-4 out of 4) reflects institutional practice: in the absence of strong beliefs about which valuation metric is superior, diversification across valuation perspectives improves robustness.

### Momentum Factor: Intermediate-Horizon Returns

The momentum factor employs the well-established 12-month minus 1-month return specification, documented extensively in academic literature beginning with Jegadeesh and Titman (1993):

$$
F_{\text{momentum},i} = \text{Return}_{i,t-21:t-252}
$$

where:

$$
\text{Return}_{i,t-21:t-252} = \frac{P_{i,t-21}}{P_{i,t-252}} - 1
$$

The construction excludes the most recent month ($t$ to $t-21$) to eliminate short-term reversal effects while capturing intermediate-term price trends. Empirical research demonstrates that 6-12 month returns predict subsequent performance, but 1-month returns exhibit mean-reversion due to microstructure effects and liquidity provision.

Cross-sectional standardization transforms raw returns into comparable signals:

$$
z_{\text{momentum},i,t} = \frac{\text{Return}_{i,t-21:t-252} - \mu_{\text{Return},t}}{\sigma_{\text{Return},t}}
$$

where $\mu_{\text{Return},t}$ and $\sigma_{\text{Return},t}$ represent the cross-sectional mean and standard deviation of 12-1 month returns across the universe. Historical S&P 500 benchmarks suggest $\mu \approx 0.08$ (8% average return) and $\sigma \approx 0.20$ (20% return dispersion), though cross-sectional statistics adapt these norms to current market conditions.

The momentum factor requires minimum data availability: at least 252 trading days (approximately one calendar year) of price history must exist to compute the metric. Securities with insufficient history receive neutral momentum scores (zero z-score) to avoid biasing the composite signal.

### Quality Factor: Profitability and Operational Efficiency

Quality factors identify companies with sustainable competitive advantages, efficient operations, and strong balance sheets. The composite quality measure aggregates three complementary metrics:

$$
F_{\text{quality},i} = \frac{1}{3}(z_{\text{ROE},i} + z_{\text{margin},i} + z_{\text{Sharpe},i})
$$

**Return on Equity (ROE)**: Measures profitability relative to shareholder capital:

$$
\text{ROE}_i = \frac{\text{Net Income}_i}{\text{Shareholders' Equity}_i}
$$

with typical cross-sectional parameters $\mu_{\text{ROE}} = 0.15$ (15% return on equity) and $\sigma_{\text{ROE}} = 0.10$. High ROE indicates efficient capital deployment and pricing power, though interpretation requires care for financial institutions and companies with high leverage.

**Profit Margin**: Evaluates operational efficiency through the relationship between earnings and revenue:

$$
\text{Margin}_i = \frac{\text{Net Income}_i}{\text{Revenue}_i}
$$

Historical norms: $\mu_{\text{margin}} = 0.10$ (10% net margin), $\sigma_{\text{margin}} = 0.08$. Margins vary systematically across industries, with software companies exhibiting substantially higher margins than retailers. Cross-sectional standardization captures these structural differences.

**Sharpe Ratio**: Provides risk-adjusted return perspective by relating excess returns to volatility:

$$
\text{Sharpe}_i = \frac{\mu_{R_i} - R_f}{\sigma_{R_i}}
$$

where $\mu_{R_i}$ represents the stock's historical average return, $R_f$ denotes the risk-free rate, and $\sigma_{R_i}$ measures return volatility. Typical values: $\mu_{\text{Sharpe}} = 0.5$, $\sigma_{\text{Sharpe}} = 0.5$. The Sharpe ratio complements accounting-based metrics by incorporating market-determined risk assessments.

#### Distress Penalty Adjustment

The quality score undergoes risk-based adjustment to penalize companies exhibiting financial distress indicators. Five distress signals trigger cumulative penalties:

1. **Negative Book Value** (penalty: -2.0): Indicates accumulated losses exceeding invested capital, a severe red flag suggesting potential bankruptcy
2. **Negative ROE with High Leverage** (penalty: -1.5): Combines unprofitability with substantial debt, magnifying financial fragility
3. **Revenue Decline with High Leverage** (penalty: -1.0): Shrinking top-line amid leverage suggests deteriorating business fundamentals
4. **Altman Z-Score < 1.81** (penalty: -1.0): The Altman Z-score combines multiple financial ratios into a bankruptcy prediction model; scores below 1.81 indicate high distress probability
5. **Negative Operating Cash Flow and Margins** (penalty: -1.5): Inability to generate cash from operations alongside negative margins indicates unsustainable business model

The total penalty is capped at -3.0 to prevent complete signal domination by distress indicators:

$$
F_{\text{quality},i}^{\text{adjusted}} = F_{\text{quality},i}^{\text{base}} + \max(\text{Penalty}_i, -3.0)
$$

This adjustment ensures that fundamentally weak companies receive appropriately negative quality signals even if individual metrics appear superficially acceptable.

#### Inflation Risk Adjustment

When forecast inflation exceeds 3% annually, an additional penalty applies to quality signals:

$$
\text{Inflation Penalty}_i = -\max(0, \pi_{\text{forecast},6m} - 3.0) \times 0.1
$$

where $\pi_{\text{forecast},6m}$ represents the 6-month inflation forecast. High inflation erodes the real value of nominally stable cash flows characteristic of quality stocks, particularly affecting companies with limited pricing power. The 10% per percentage point penalty (0.1 coefficient) provides moderate adjustment without overwhelming the base quality signal.

### Growth Factor: Expansion Metrics

The growth factor measures business expansion through both trailing performance and forward-looking expectations:

$$
F_{\text{growth},i} = \frac{1}{2}(z_{\text{revenue growth},i} + z_{\text{earnings growth},i})
$$

where each component combines historical and forecast metrics:

$$
z_{\text{revenue growth},i} = 0.6 \times z_{\text{revenue},i}^{\text{trailing}} + 0.4 \times z_{\text{GDP forecast},i}
$$

$$
z_{\text{earnings growth},i} = 0.6 \times z_{\text{earnings},i}^{\text{trailing}} + 0.4 \times z_{\text{earnings forecast},i}
$$

**Trailing Growth Metrics**: Historical revenue and earnings growth rates over the past fiscal year, subject to ±50% winsorization to prevent extreme outliers from distorting cross-sectional statistics. Market norms: $\mu_{\text{revenue growth}} = 0.05$ (5%), $\sigma_{\text{revenue growth}} = 0.10$; $\mu_{\text{earnings growth}} = 0.07$ (7%), $\sigma_{\text{earnings growth}} = 0.15$.

**Forward-Looking Components**: The 40% allocation to forecast metrics incorporates forward-looking information, enabling the growth factor to anticipate turning points 3-6 months before they fully manifest in trailing data. GDP forecasts provide macroeconomic context for revenue expectations, while analyst earnings forecasts reflect company-specific projections.

The 60/40 trailing-forecast blend balances the reliability of realized performance against the predictive value of expectations. Pure historical metrics lag fundamental changes, while pure forecasts can reflect excessive optimism or pessimism. The blend achieves pragmatic compromise.

## Cross-Sectional Standardization: Seven-Pass Architecture

### Motivation for Multi-Pass Processing

The statistical challenge of computing robust cross-sectional statistics in the presence of outliers motivates a sophisticated multi-pass architecture. A naive single-pass approach—fetching data, computing sample means and standard deviations, and immediately classifying signals—suffers from fatal flaws:

**Contamination Problem**: Extreme outliers bias sample statistics, rendering standardization ineffective. A universe containing 995 stocks with earnings yields between 2-8% plus 5 distressed companies with yields exceeding 500% (due to near-zero market capitalizations) would exhibit severely inflated mean and standard deviation.

**Circular Dependency**: Outlier detection requires standardized scores (values beyond ±3σ suggest outliers), but standardization requires clean statistics. Using contaminated statistics to identify outliers creates a circular problem.

**Distribution Assumptions**: Classification schemes assuming normally distributed z-scores fail when raw metrics exhibit fat tails, skewness, or other non-normality that persists after naive standardization.

The solution employs iterative refinement through seven distinct processing passes, each serving a specific statistical purpose.

### Pass 1: Raw Fundamentals Collection

The initial pass fetches comprehensive financial data for all instruments in the universe without performing any standardization or signal calculation. For each security, the system retrieves:

- Historical price data: 2 years (approximately 500 trading days) of adjusted close prices, volumes, and splits
- Fundamental metrics: Balance sheet items (book value, total assets, total debt), income statement items (revenue, net income, operating income), and cash flow data
- Technical indicators: Calculated from price data (volatility, beta, maximum drawdown)
- Country classification: For macroeconomic regime integration

This pass explicitly avoids z-score calculation, storing only raw values. The separation prevents premature standardization using contaminated statistics and enables subsequent passes to compute proper cross-sectional norms.

Output structure: Array of tuples containing raw data:

$$
\text{Pass1}_{\text{output}} = \{(\text{Instrument}_i, \text{Metrics}_i, \text{Info}_i, \text{Country}_i)\}_{i=1}^{N}
$$

where $N$ represents the number of valid securities surviving initial data quality filters.

### Pass 1.5: Robust Cross-Sectional Statistics Calculation

Pass 1.5 implements the critical iterative outlier removal procedure that resolves the contamination problem. For each fundamental metric (Book-to-Price, Earnings-to-Price, ROE, etc.), the algorithm applies three iterations of ±3σ filtering:

**Iteration 1**: Compute initial sample statistics from all available data:

$$
\mu_1 = \frac{1}{N}\sum_{i=1}^{N} X_i
$$

$$
\sigma_1 = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(X_i - \mu_1)^2}
$$

Calculate z-scores using these potentially contaminated statistics:

$$
z_{i,1} = \frac{X_i - \mu_1}{\sigma_1}
$$

Remove observations where $|z_{i,1}| > 3.0$, creating filtered dataset $\mathcal{D}_1$ with $N_1 \leq N$ observations.

**Iteration 2**: Recalculate statistics using only the filtered dataset:

$$
\mu_2 = \frac{1}{N_1}\sum_{i \in \mathcal{D}_1} X_i
$$

$$
\sigma_2 = \sqrt{\frac{1}{N_1-1}\sum_{i \in \mathcal{D}_1}(X_i - \mu_2)^2}
$$

Apply second-round filtering, removing observations where $|z_{i,2}| > 3.0$ relative to the updated statistics, yielding $\mathcal{D}_2$ with $N_2 \leq N_1$ observations.

**Iteration 3**: Final refinement using twice-filtered data:

$$
\mu_{\text{robust}} = \frac{1}{N_2}\sum_{i \in \mathcal{D}_2} X_i
$$

$$
\sigma_{\text{robust}} = \sqrt{\frac{1}{N_2-1}\sum_{i \in \mathcal{D}_2}(X_i - \mu_{\text{robust}})^2}
$$

Empirical observation demonstrates that three iterations typically achieve convergence: the third iteration removes few if any additional observations, indicating statistical stability.

**Convergence Example**: For a Book-to-Price distribution across 1,000 stocks:

- Iteration 1: Removes 12 extreme outliers (B/P > 2.0), reducing $N$ from 1,000 to 988
- Iteration 2: Removes 3 secondary outliers newly identified, reducing $N$ to 985
- Iteration 3: Removes 0 additional outliers, confirming convergence

The resulting robust statistics $(\mu_{\text{robust}}, \sigma_{\text{robust}})$ provide clean anchors for subsequent standardization, free from outlier contamination.

Output structure: Dictionary mapping each factor and metric to robust statistics:

$$
\text{CrossSectionalStats} = \{(\text{Factor}, \text{Metric}, \mu_{\text{robust}}, \sigma_{\text{robust}})\}
$$

### Pass 1B: Z-Score Recalculation with Universe Statistics

Pass 1B reconstructs factor z-scores using the robust cross-sectional statistics computed in Pass 1.5, replacing the static market norms with universe-specific values. This recalculation ensures that standardization reflects actual market conditions rather than historical approximations.

For each stock $i$ and metric $X$:

$$
z_{X,i}^{\text{universe}} = \frac{X_i - \mu_{X}^{\text{robust}}}{\sigma_{X}^{\text{robust}}}
$$

These metric-level z-scores aggregate into factor scores following the specifications in Section 2:

$$
F_{\text{value},i} = \frac{1}{4}\sum_{j \in \{B/P, E/P, S/P, D/P\}} z_{j,i}^{\text{universe}}
$$

with analogous constructions for momentum, quality, and growth factors. The composite signal combines factors through equal weighting:

$$
Z_{\text{raw},i} = \frac{1}{4}\sum_{k \in \{\text{value, momentum, quality, growth}\}} F_{k,i}
$$

James-Stein shrinkage applies to moderate extreme scores:

$$
Z_{\text{shrunk},i} = 0.7 \times Z_{\text{raw},i}
$$

The shrinkage factor of 0.7 balances information signal with mean-reversion, reducing estimation error for extreme observations while preserving meaningful differentiation across the distribution.

Macroeconomic regime adjustments integrate business cycle positioning:

$$
Z_{\text{adjusted},i} = Z_{\text{shrunk},i} \times (1 + \alpha_{\text{sector},i} \times \beta_{\text{regime}})
$$

where $\alpha_{\text{sector},i}$ represents sector-specific multipliers and $\beta_{\text{regime}}$ captures the current business cycle phase. Conservative multipliers limit adjustments to ±5% to maintain tracking error within institutional risk budgets.

Output: Enhanced dataset with composite z-scores:

$$
\text{Pass1B}_{\text{output}} = \{(\text{Instrument}_i, Z_{\text{adjusted},i}, F_{k,i}, \text{Metrics}_i)\}_{i=1}^{N}
$$

### Pass 2: Standardization via Winsorization and Scaling

Pass 2 applies two-stage standardization to ensure the composite z-score distribution exhibits exact $\mathcal{N}(0,1)$ properties suitable for percentile-based classification.

**Stage 1: Winsorization**: Clips extreme values to prevent isolated outliers from dominating subsequent scaling:

$$
Z_{\text{winsorized},i} = \max(-10, \min(Z_{\text{adjusted},i}, 10))
$$

The ±10σ threshold affects fewer than 0.1% of observations under normality but prevents extreme scores (e.g., $Z = 50$ from data errors or unique circumstances) from distorting the StandardScaler transformation.

**Stage 2: StandardScaler Transformation**: Applies exact normalization to achieve zero mean and unit variance:

$$
Z_{\text{final},i} = \frac{Z_{\text{winsorized},i} - \mu_{\text{winsorized}}}{\sigma_{\text{winsorized}}}
$$

where $\mu_{\text{winsorized}}$ and $\sigma_{\text{winsorized}}$ are computed from the winsorized distribution. This transformation guarantees:

$$
\frac{1}{N}\sum_{i=1}^{N} Z_{\text{final},i} = 0 \quad \text{(exactly)}
$$

$$
\frac{1}{N-1}\sum_{i=1}^{N} (Z_{\text{final},i})^2 = 1 \quad \text{(exactly)}
$$

The exact standardization ensures percentile cutoffs calculated from theoretical $\mathcal{N}(0,1)$ distributions provide accurate quintile boundaries.

### Pass 2.5: Factor-Level Robust Statistics

Pass 2.5 validates that individual factor z-scores (value, momentum, quality, growth) maintain proper standardization properties before combination. The validation applies the same iterative outlier removal procedure from Pass 1.5 to each factor's distribution:

For factor $F_k$ with z-scores $\{F_{k,i}\}_{i=1}^{N}$:

1. Compute robust mean $\mu_{F_k}^{\text{robust}}$ and standard deviation $\sigma_{F_k}^{\text{robust}}$ via three iterations of ±3σ filtering
2. Verify that $|\mu_{F_k}^{\text{robust}}| < 0.3$ and $0.7 < \sigma_{F_k}^{\text{robust}} < 1.3$

Substantial deviations from $\mathcal{N}(0,1)$ properties indicate potential issues in factor construction or data quality requiring investigation.

### Pass 2.6: Factor Correlation Validation

Pass 2.6 constructs the correlation matrix across the four factor z-scores and validates against expected patterns documented in institutional portfolio construction literature:

$$
\mathbf{C} = \begin{bmatrix}
1 & \rho_{V,M} & \rho_{V,Q} & \rho_{V,G} \\
\rho_{V,M} & 1 & \rho_{M,Q} & \rho_{M,G} \\
\rho_{V,Q} & \rho_{M,Q} & 1 & \rho_{Q,G} \\
\rho_{V,G} & \rho_{M,G} & \rho_{Q,G} & 1
\end{bmatrix}
$$

Expected correlation ranges, derived from empirical factor research:

- **Value-Momentum**: $\rho_{V,M} \in [-0.4, -0.2]$ — Negative correlation provides diversification benefit, as value and momentum strategies perform differently across market cycles
- **Quality-Value**: $\rho_{V,Q} \in [0.0, 0.2]$ — Low positive correlation indicates near-independence, enabling complementary portfolio contributions
- **Quality-Momentum**: $\rho_{M,Q} \in [0.2, 0.4]$ — Moderate positive correlation reflects shared exposure to profitable companies with positive price trends
- **Growth-Momentum**: $\rho_{G,M} \approx 0$ — Near-zero correlation suggests orthogonal information content

Correlation patterns outside expected ranges flag data quality issues, definitional problems, or structural market regime changes warranting investigation. High positive correlations ($\rho > 0.7$) between supposedly independent factors indicate potential redundancy and multicollinearity concerns.

### Pass 3: Signal Classification and Database Persistence

The final pass converts standardized z-scores into discrete investment signals through percentile-based classification, applying momentum filters for refinement, and persisting results to the database for portfolio optimization.

**Quintile Classification**: The system classifies stocks into five equally-sized buckets based on cross-sectional ranking:

$$
S_i = \begin{cases}
\text{LARGE\_GAIN} & \text{if } Z_{\text{final},i} \geq p_{80} \\
\text{SMALL\_GAIN} & \text{if } p_{60} \leq Z_{\text{final},i} < p_{80} \\
\text{NEUTRAL} & \text{if } p_{40} \leq Z_{\text{final},i} < p_{60} \\
\text{SMALL\_DECLINE} & \text{if } p_{20} \leq Z_{\text{final},i} < p_{40} \\
\text{LARGE\_DECLINE} & \text{if } Z_{\text{final},i} < p_{20}
\end{cases}
$$

where $p_k$ denotes the $k$-th percentile of the $Z_{\text{final}}$ distribution. For theoretical $\mathcal{N}(0,1)$ distributions, these percentiles correspond to:

$$
p_{20} = -0.84, \quad p_{40} = -0.25, \quad p_{60} = 0.25, \quad p_{80} = 0.84
$$

**Momentum Filter Adjustments**: Two-tier momentum-based refinements override initial classifications when strong price trends conflict with composite signals:

**Level 2 - Negative Momentum Downgrade**: If 12-month returns fall below -15% and initial signal indicates SMALL\_GAIN or LARGE\_GAIN, downgrade to NEUTRAL:

$$
S_i^{\text{adjusted}} = \begin{cases}
\text{NEUTRAL} & \text{if } S_i \in \{\text{SMALL\_GAIN, LARGE\_GAIN}\} \text{ and } R_{i,12m} < -0.15 \\
S_i & \text{otherwise}
\end{cases}
$$

This adjustment prevents recommending stocks with strongly negative momentum despite favorable fundamental signals.

**Level 3 - Positive Momentum Upgrade**: If 12-month returns exceed +40% and initial signal is SMALL\_GAIN, upgrade to LARGE\_GAIN:

$$
S_i^{\text{final}} = \begin{cases}
\text{LARGE\_GAIN} & \text{if } S_i = \text{SMALL\_GAIN} \text{ and } R_{i,12m} > 0.40 \\
S_i^{\text{adjusted}} & \text{otherwise}
\end{cases}
$$

These thresholds (-15% for downgrade, +40% for upgrade) balance responsiveness to price trends against false signals from short-term volatility.

## Signal Classification Framework

### Percentile-Based vs. Threshold-Based Approaches

The framework employs percentile-based classification rather than fixed z-score thresholds, a distinction with important implications for signal distribution properties.

**Fixed Threshold Approach** (not used): Classify based on absolute z-score values:

$$
\text{LARGE\_GAIN if } Z_i > 0.84, \quad \text{NEUTRAL if } |Z_i| \leq 0.25, \text{ etc.}
$$

This approach assumes the z-score distribution precisely follows $\mathcal{N}(0,1)$. While Pass 2 standardization ensures approximate normality, residual skewness or kurtosis can create unequal bucket sizes, with some quintiles containing 15% of stocks and others 25%.

**Percentile Approach** (implemented): Classify based on cross-sectional rank:

$$
\text{LARGE\_GAIN if } \text{rank}(Z_i) \geq 80\text{th percentile}
$$

This approach guarantees exact 20/20/20/20/20 distribution regardless of the z-score distribution's shape. Percentile cutoffs adapt to the empirical distribution, accommodating any residual non-normality after standardization.

### Distribution Tracking and Validation

The system maintains cumulative distribution tracking through database persistence, enabling three-tier classification logic that prioritizes historical consistency while adapting to changing market conditions.

**Tier 1 - Saved Distribution**: If a validated distribution exists from previous runs with sufficient sample size ($n \geq 50$), use its empirically derived percentile thresholds:

$$
p_k^{\text{historical}} = \text{percentile}(\{Z_{\text{final},i,t'}\}_{t' < t}, k)
$$

This approach maintains classification consistency across time, ensuring signals remain comparable to historical patterns.

**Tier 2 - Empirical Percentiles**: If no saved distribution exists but the current sample is large ($n \geq 100$), compute empirical percentiles directly from current data:

$$
p_k^{\text{empirical}} = \text{percentile}(\{Z_{\text{final},i,t}\}_{i=1}^{N_t}, k)
$$

**Tier 3 - Theoretical Thresholds**: For small samples ($n < 100$) without saved distributions, use theoretical $\mathcal{N}(0,1)$ percentiles:

$$
p_k^{\text{theoretical}} = \Phi^{-1}(k/100)
$$

where $\Phi^{-1}$ denotes the inverse cumulative distribution function of the standard normal.

This three-tier hierarchy balances statistical reliability (larger samples provide better percentile estimates) with consistency (historical distributions maintain comparability) and practical necessity (small samples require theoretical fallbacks).

### Distribution Quality Validation

Before accepting a distribution for Tier 1 classification, the system validates several statistical properties:

$$
|\bar{Z}| < 0.3, \quad 0.7 < \sigma_Z < 1.3
$$

$$
\left|\frac{n_k}{N} - 0.20\right| < 0.05 \quad \text{for each quintile } k \in \{1,2,3,4,5\}
$$

$$
|p_k^{\text{empirical}} - p_k^{\text{theoretical}}| < 0.2 \quad \text{for } k \in \{20, 40, 60, 80\}
$$

Distributions failing these criteria receive validation warnings and may be rejected from Tier 1 usage pending investigation of data quality or methodological issues.

## Statistical Rigor and Quality Assurance

### Iterative Outlier Removal Convergence

The three-iteration outlier removal procedure typically achieves statistical convergence, defined as minimal change in estimated parameters between successive iterations. Convergence can be formalized as:

$$
|\mu_{j+1} - \mu_j| < \epsilon_{\mu} \quad \text{and} \quad |\sigma_{j+1} - \sigma_j| < \epsilon_{\sigma}
$$

where $j$ denotes iteration number and $\epsilon$ represents tolerance thresholds. Empirical analysis across diverse universes demonstrates that iteration 3 typically satisfies $\epsilon_{\mu} = 0.01$ and $\epsilon_{\sigma} = 0.01$, confirming stabilization.

The iterative procedure's effectiveness stems from the **shrinking contamination principle**: each iteration removes the most extreme outliers relative to current estimates, gradually purifying the sample. Initial iterations remove gross outliers (data errors, acquisition targets), while later iterations refine by removing secondary outliers that become visible only after gross outliers are eliminated.

### Winsorization Strategy and Thresholds

Winsorization at ±10σ reflects a carefully chosen threshold that balances outlier mitigation with information preservation. Under standard normality, the probability of observing values beyond ±10σ is approximately:

$$
P(|Z| > 10) = 2\Phi(-10) \approx 1.5 \times 10^{-23}
$$

In practical terms, a universe of 10,000 stocks would expect zero observations beyond ±10σ under pure normality. Any such observations reflect either:

1. Data errors (incorrect prices, stale fundamentals, corporate action mishandling)
2. Truly exceptional circumstances (bankruptcy, acquisition, fraud revelation)
3. Residual non-normality in the composite z-score distribution

Winsorization clips these extreme values to ±10, preventing them from dominating the StandardScaler transformation while preserving their classification as extreme positive or negative signals.

Alternative thresholds present tradeoffs:
- Lower thresholds (±5σ): More aggressive outlier mitigation but risk clipping genuine extreme signals
- Higher thresholds (±15σ): Preserve more extreme information but allow larger influence on standardization

The ±10σ choice provides pragmatic compromise, affecting fewer than 0.1% of observations while protecting against scale distortion.

### Factor Correlation Interpretation

The factor correlation matrix provides diagnostic information about signal quality and independence. Expected correlation patterns reflect underlying economic relationships:

**Value-Momentum Negative Correlation** ($\rho_{V,M} \in [-0.4, -0.2]$): Value strategies identify undervalued stocks that may have experienced poor recent performance (negative momentum), while momentum strategies favor recent winners regardless of valuation. The negative correlation indicates these factors capture distinct return sources, providing diversification benefit when combined.

**Quality-Value Low Correlation** ($\rho_{V,Q} \in [0.0, 0.2]$): Quality characteristics (high ROE, stable margins) can occur in both expensive and cheap stocks. Luxury goods manufacturers may exhibit high quality but trade at premium valuations, while cyclical industrials may offer value but show moderate quality metrics. The low correlation enables independent contributions to portfolio selection.

**Quality-Momentum Positive Correlation** ($\rho_{M,Q} \in [0.2, 0.4]$): Companies with improving fundamentals and operational excellence tend to exhibit positive price momentum as markets gradually recognize quality improvements. The moderate positive correlation is natural and indicates shared exposure to improving business fundamentals, while remaining sufficiently low to provide diversification benefits.

Deviations from expected correlations warrant investigation:
- **Unexpected high positive correlations** ($\rho > 0.7$): Suggest factors may be measuring similar underlying characteristics, reducing diversification benefit
- **Sign reversals**: Indicate potential data quality issues or fundamental market structure changes
- **Regime-dependent shifts**: Business cycle transitions can temporarily alter factor relationships, requiring monitoring

### Minimum Sample Size Requirements

Statistical reliability depends critically on sample size, with different analytical procedures requiring different minimum thresholds:

**Cross-Sectional Statistics** ($n \geq 50$): Computing robust mean and standard deviation via iterative outlier removal requires sufficient observations to ensure stable parameter estimates after removing outliers. With three iterations potentially removing up to 5% of observations each, starting samples below 50 risk excessive depletion.

**Percentile Estimation** ($n \geq 100$): Accurate percentile estimation, particularly for extreme percentiles ($p_{20}$, $p_{80}$), requires larger samples. The 20th percentile estimate from 100 observations corresponds to the 20th ordered statistic, providing reasonable precision. Smaller samples yield volatile percentile estimates sensitive to individual observations.

**Distribution Validation** ($n \geq 30$): Statistical tests for normality and bucket balance require minimum sample sizes for meaningful inference. With $n = 30$, each quintile should contain approximately 6 observations under perfect balance, sufficient for rough validation though not rigorous testing.

**Correlation Analysis** ($n \geq 50$): Estimating correlation matrices with four factors requires sufficient observations to ensure stable correlation coefficient estimates. Smaller samples yield correlation estimates with large standard errors, limiting diagnostic value.

These thresholds represent minimum viable sample sizes; larger samples (ideally $n \geq 500$ for institutional applications) provide superior statistical properties and more stable signals.

## Database Integration and Persistence

### Signal Distribution Model

The SignalDistribution database model captures statistical properties of z-score distributions for historical tracking and classification consistency. Key fields include:

**Distributional Statistics**:
- Sample size $n$
- Mean $\bar{Z}$ (expected: $\approx 0$)
- Standard deviation $\sigma_Z$ (expected: $\approx 1$)
- Median (expected: $\approx 0$)
- Skewness and kurtosis measures

**Percentile Thresholds**:
- $p_{20}, p_{40}, p_{60}, p_{80}$ for quintile classification

**Validation Flags**:
- Boolean indicating whether distribution passes quality checks
- Timestamp of distribution calculation
- Universe description (which stocks included)

The model enables temporal tracking of signal distribution evolution, supporting analyses of:
- Distribution stability across time
- Regime-dependent distribution shifts
- Data quality trends through validation failure rates

### Stock Signal Model

Individual stock signals persist through the StockSignal database model, recording:

**Signal Classification**:
- Signal type (LARGE\_GAIN through LARGE\_DECLINE)
- Signal date and generation timestamp
- Confidence level

**Underlying Metrics**:
- Composite z-score
- Individual factor z-scores (value, momentum, quality, growth)
- Raw metric values (for audit and analysis)

**Risk Characteristics**:
- Volatility level, beta risk, leverage risk, liquidity risk classifications
- Maximum drawdown, Sharpe ratio, Sortino ratio

**Performance Data**:
- Current price, volume, daily return
- Historical annualized return and volatility

This comprehensive persistence enables:
- Backtesting signal performance by tracking subsequent returns for stocks with each signal type
- Portfolio construction using current signals with full transparency to underlying factors
- Audit trails documenting the complete signal generation process from raw data through final classification

## Summary

The stock signal generation methodology achieves institutional-grade rigor through mathematical precision, robust statistical procedures, and comprehensive validation. The seven-pass architecture solves the fundamental challenge of computing reliable cross-sectional statistics in the presence of outliers, enabling consistent signal generation across diverse market conditions.

The four-factor framework—value, momentum, quality, and growth—captures complementary return sources with negative to low correlations, providing diversification within the signal generation process itself. Cross-sectional standardization ensures signals reflect relative rather than absolute attractiveness, maintaining meaningful rankings across bull and bear markets.

Percentile-based classification with momentum filters translates continuous z-scores into discrete investment signals while maintaining flexibility for diverse portfolio construction approaches. Comprehensive database persistence enables backtesting, performance attribution, and continuous signal refinement.

The integration of macroeconomic regime classifications, detailed in Chapter 2, adapts signal weights and sector exposures to business cycle dynamics, enhancing signal informativeness during regime transitions. The mathematical signals provide cost-effective, scalable foundations that complement and enable subsequent portfolio optimization processes examined in following chapters.
