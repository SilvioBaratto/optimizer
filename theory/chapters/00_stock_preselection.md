# Quantitative Stock Pre-Selection

Before any portfolio optimizer receives its inputs, a prior question must be answered: which instruments deserve consideration? The preceding chapters assume a well-defined investment universe, but that universe does not materialize spontaneously. It is the product of a systematic pre-selection process that enforces investability, scores each candidate on multiple dimensions of expected return, and delivers a concentrated set of high-conviction instruments to the optimization pipeline. This chapter specifies that process. The treatment moves from coarse investability screens through individual factor construction and composite scoring to regime-conditional tilts and statistical validation, culminating in the interface between pre-selection output and portfolio optimization input.

## Investability Screening

A stock must be tradeable before it can be investable. Investability screening enforces minimum standards of market capitalization, liquidity, price level, listing history, and data availability that ensure every instrument admitted to the optimization universe can be bought and sold in meaningful size without excessive market impact or stale information. These screens are not alpha signals; they are necessary conditions for the subsequent factor analysis to produce actionable results.

### Market Capitalization and Liquidity Requirements

Free-float market capitalization serves as the primary size filter. The free-float adjustment excludes shares held by insiders, governments, and strategic investors that are unlikely to trade, focusing instead on the capitalization available to portfolio managers. For a developed-market universe targeting 500 to 2000 names, a minimum free-float market capitalization of approximately 200 million USD at entry excludes microcap stocks whose execution costs and price impact would erode any factor premium. Major index providers impose analogous screens: MSCI requires minimum absolute and relative capitalization thresholds calibrated by market segment, CRSP sets entry at 15 million USD with removal at 10 million USD for its total market index, and FTSE Russell requires 30 million USD for Russell US index eligibility. The thresholds adopted here are deliberately tighter than the broadest indices to reflect a universe oriented toward institutional execution quality rather than maximal coverage.

Liquidity screens complement the size filter. Average daily dollar volume over trailing twelve-month and three-month windows captures both structural liquidity and recent trading activity. A twelve-month average of at least 750 thousand USD and a three-month average of at least 500 thousand USD ensure that a position of several million dollars can be established or liquidated within a few trading sessions without moving the price substantially. Trading frequency, measured as the fraction of trading days with nonzero volume, must exceed 95 percent over the trailing year. These thresholds are simplified analogs of the annualized traded value ratio used by MSCI, which requires a 20 percent ratio over both three-month and twelve-month windows in developed markets, and the turnover-based screens used by CRSP.

### Price and Listing Filters

Minimum price thresholds eliminate penny stocks, whose bid-ask spreads consume a disproportionate share of any expected return. For US-listed equities, a minimum price of 3 USD at entry (2 USD at exit) is stricter than the 1 USD floor imposed by FTSE Russell but appropriate for a universe where execution quality matters. European exchanges apply comparable thresholds in local currency. These filters remove instruments that are dominated by microstructure noise rather than fundamental information.

Listing history requirements ensure sufficient data for momentum, volatility, and factor computation. A minimum of 252 trading days of price history, corresponding to approximately one calendar year, provides the lookback needed for twelve-month momentum and annualized volatility estimation. An additional IPO seasoning filter of 60 trading days excludes recently listed stocks whose early price dynamics are dominated by allocation effects, retail enthusiasm, and thin float rather than fundamental value. Financial statement availability further constrains the universe: at least three years of annual reports or eight quarters of quarterly reports with core accounting items (total assets, total equity, total revenue, net income, operating cash flow) must be present to compute stable valuation and profitability ratios.

### Hysteresis and Buffer Design

Applying identical thresholds for entry and exit generates excessive turnover at the boundary, as marginal stocks oscillate in and out of the universe with small fluctuations in capitalization or volume. Index providers address this through buffer mechanisms: CRSP uses packeting around capitalization breakpoints, FTSE Russell maintains separate entry and exit price thresholds, and MSCI applies buffer zones around investability cutoffs. The pre-selection module adopts the same principle by setting exit thresholds below entry thresholds for every screen. A stock enters the universe when its free-float market capitalization reaches 200 million USD but is removed only when it falls below 150 million USD. Liquidity entry requires 750 thousand USD in twelve-month average daily dollar volume; exit triggers at 500 thousand USD. This asymmetry reduces one-way monthly turnover attributable to investability changes by roughly half compared to symmetric thresholds.

The following table summarizes the recommended default parameters:

| Filter | Entry Threshold | Exit Threshold |
|---|---|---|
| Free-float market cap | $\geq$ 200M USD and $\geq$ 10th percentile by exchange | $\geq$ 150M USD and $\geq$ 7.5th percentile |
| 12-month avg. daily dollar volume | $\geq$ 750K USD | $\geq$ 500K USD |
| 3-month avg. daily dollar volume | $\geq$ 500K USD | $\geq$ 350K USD |
| Trading frequency (12-month) | $\geq$ 95% of trading days | $\geq$ 90% |
| Price (US exchanges) | $\geq$ 3 USD | $\geq$ 2 USD |
| Price (European exchanges) | $\geq$ 2 local currency | $\geq$ 1.5 local currency |
| Trading history | $\geq$ 252 trading days | --- |
| IPO seasoning | $\geq$ 60 trading days since first price | --- |
| Financial statements | $\geq$ 3 annual or 8 quarterly reports | --- |

## Factor Taxonomy and Empirical Evidence

The empirical asset pricing literature has identified hundreds of cross-sectional return predictors. Harvey, Liu, and Zhu catalogued over 300 published factors through 2015 and showed that conventional significance thresholds are far too permissive given the scale of collective data mining. This proliferation demands discipline: the factors retained for pre-selection scoring must rest on robust economic rationale, survive out-of-sample testing, and remain implementable after accounting for transaction costs. The taxonomy below focuses on factors with deep empirical support, measured by decades of evidence across multiple markets.

### Value

The value premium is among the oldest and most extensively documented anomalies in equity markets. The Fama-French five-factor model measures value through book-to-market equity, with the HML (high minus low) factor delivering an annualized long-short premium of approximately 3 to 5 percent over the 1963--2014 sample period, at an annualized volatility of 12 to 15 percent and a Sharpe ratio of roughly 0.3 to 0.4. Asness, Moskowitz, and Pedersen confirm this premium across equity markets, government bonds, currencies, and commodity futures, establishing value as a pervasive phenomenon rather than a US equity artifact.

A composite value score combining book-to-price, earnings yield, cash-flow yield, sales-to-price, and EBITDA-to-enterprise-value provides a more robust signal than any single ratio. Typical monthly cross-sectional rank information coefficients for well-constructed value composites range from 0.03 to 0.05. The signal is slow-moving, with predictive power extending to horizons of 12 to 36 months, but the premium is highly cyclical, with prolonged drawdowns particularly in the post-2007 period.

Value correlates negatively with profitability (approximately $-0.3$ to $-0.5$) because profitable firms command higher valuations. This negative correlation makes the value-profitability combination particularly attractive for diversification. Value and momentum are historically near-orthogonal, providing further diversification, though both can suffer simultaneously during crisis reversals.

### Profitability and Quality

Novy-Marx demonstrated that gross profitability, defined as gross profits scaled by total assets, predicts cross-sectional returns with power comparable to book-to-market. The gross profitability premium, measured as a long-short return of the most profitable minus least profitable firms, averages approximately 0.52 percent per month (6.2 percent annualized) with a $t$-statistic of 4.49 over the 1963--2010 period, and a Sharpe ratio of approximately 0.85. The Fama-French RMW (robust minus weak operating profitability) factor confirms this finding at a somewhat lower magnitude of roughly 3.6 percent annualized.

The profitability signal is persistent, with monthly rank information coefficients in the 0.02 to 0.04 range and predictive power extending to multi-year horizons. Profitability factors have high capacity because they tend to favor large, established, liquid firms. The strongest alphas appear even among the largest and most liquid stocks, making profitability one of the most implementable factor premiums.

### Investment Conservatism

The Fama-French CMA (conservative minus aggressive) factor captures the tendency of firms with low asset growth to outperform firms with high asset growth. The annual asset growth rate, measured as the log change in total assets over the prior year, defines the sorting variable. The CMA factor yields approximately 3 to 3.5 percent annualized over the 1963--2013 sample with significant $t$-statistics. Monthly information coefficients are modest at 0.01 to 0.02, reflecting a structural tilt rather than a tactical timing signal. Investment conservatism correlates positively with profitability, which partly explains the redundancy of the HML factor after controlling for both RMW and CMA in the five-factor model.

### Momentum

Price momentum is the cross-sectional strategy of buying recent winners and selling recent losers. The standard construction uses cumulative returns over months $t-12$ through $t-1$, skipping the most recent month to avoid short-term reversal effects. Jegadeesh and Titman reported annualized long-short returns of approximately 12 percent with $t$-statistics exceeding 4 and Sharpe ratios in the 0.6 to 0.8 range. Carhart incorporated momentum as the fourth factor in his extension of the Fama-French three-factor model. Monthly rank information coefficients for twelve-minus-one-month momentum typically range from 0.04 to 0.06, peaking at the one-month horizon and decaying by roughly 50 percent at six months. McLean and Pontiff document that post-publication returns are somewhat lower than in-sample estimates, but momentum remains economically meaningful.

Momentum is approximately orthogonal to value and profitability in normal markets, providing diversification, but it is subject to severe crash risk during sharp market reversals, as concentrated winner and loser portfolios unwind simultaneously.

### Low Risk

Low-volatility and low-beta portfolios have historically delivered market-like returns with 25 to 30 percent lower volatility, implying higher Sharpe ratios and CAPM alpha of 2 to 4 percent per year in developed equity markets. As a stock-selection factor, low volatility has modest monthly information coefficients of approximately 0.01 to 0.02, functioning primarily as a long-horizon risk adjuster rather than a short-term return predictor. Low-risk factors correlate positively with quality and dividend yield, reflecting overlapping defensive characteristics, and negatively with high-beta speculative stocks.

### Liquidity

Amihud's illiquidity measure, defined as the average ratio of absolute daily return to daily dollar volume, captures the price impact of trading. Illiquid stocks command a return premium of approximately 2 to 4 percent per year on a long-short basis after controlling for size and value. Monthly information coefficients range from 0.01 to 0.03, with the effect concentrated in small-capitalization stocks and persisting over longer horizons. Liquidity is highly correlated with size, and illiquid stocks often overlap with cheap and small firms, limiting the incremental diversification this factor provides beyond what value and size already capture. Nevertheless, a modest liquidity tilt within the investable universe can capture residual compensation for bearing execution risk.

### Dividend Yield and Payout

High-dividend-yield portfolios earn 1 to 3 percent annualized excess returns relative to the market, though this premium is largely subsumed by value and quality in modern factor models. The more robust finding is defensive performance: high-yield portfolios exhibit shallower drawdowns during market stress. Monthly information coefficients are approximately 0.01 to 0.02, slow-moving and persistent over multi-year horizons. Dividend yield correlates strongly with value and low volatility, so naive combination adds less diversification than the raw factor count might suggest.

### Analyst Sentiment and Revisions

Recommendation upgrades and target price increases predict excess returns of approximately 3 to 8 percent annualized for extreme quintiles over horizons of 3 to 12 months, though alphas shrink to 1 to 3 percent after controlling for standard factors. Monthly information coefficients of 0.02 to 0.04 decay quickly, with a half-life of 3 to 6 months, as information is incorporated into prices. Revisions correlate positively with momentum and capture overlapping directional information, so their incremental contribution to a composite that already includes momentum is modest.

### Ownership and Insider Activity

Net insider buying predicts excess returns of approximately 3 to 8 percent annualized over 6 to 12 months for high-signal quintiles, particularly in small-capitalization stocks, as Lakonishok and Lee document. Institutional ownership changes show smaller but positive effects. Monthly information coefficients range from 0.02 to 0.04, decaying over approximately 12 months. These signals partially overlap with value, momentum, and analyst revisions, providing additive but not orthogonal information.

### Cross-Factor Correlation Structure

Understanding the pairwise correlation structure among factors is essential for composite construction. The most valuable combinations pair negatively correlated factors: value with profitability ($\rho \approx -0.3$ to $-0.5$) and value with momentum (low or mildly negative correlation). These pairings improve composite Sharpe ratios by capturing distinct sources of return while partially hedging each factor's drawdowns. Conversely, momentum, analyst revisions, and short-term trend factors share similar directional information, and low volatility, dividend yield, and quality overlap in their defensive profiles. Combining highly correlated factors within the same composite group and weighting groups rather than individual factors avoids double-counting.

## Factor Construction Methodology

Raw factor values must be transformed into cross-sectionally comparable scores before they can enter a composite. The construction methodology addresses distributional pathology, sector and country biases, missing data, and temporal alignment.

### Cross-Sectional Standardization

Two standardization approaches suit different distributional characteristics. For approximately normal factors such as momentum, a simple $z$-score transformation suffices:

$$
z_{i} = \frac{f_{i} - \bar{f}}{\sigma_f}
$$

where $\bar{f}$ and $\sigma_f$ are the cross-sectional mean and standard deviation computed after winsorization. For heavy-tailed or skewed factors such as valuation ratios and illiquidity measures, a rank-based normal-score transformation is more robust. Each stock receives its cross-sectional rank, the rank is mapped to a uniform quantile, and the quantile is passed through the inverse normal distribution:

$$
z_{i} = \Phi^{-1}\!\left(\frac{\text{rank}(f_i) - 0.5}{N}\right)
$$

This rank-normal transformation eliminates the influence of extreme outliers while preserving ordinal information, following the practice advocated by Asness and Frazzini for heavy-tailed factors. Prior to either transformation, winsorization at the 1st and 99th percentiles (or 2.5th and 97.5th for mildly skewed factors) caps the influence of extreme observations that may reflect data errors rather than genuine dispersion.

### Sector and Country Neutralization

Factor scores often exhibit systematic sector or country biases. Value scores tend to be uniformly high in capital-intensive sectors such as financials and utilities, while momentum scores cluster in whichever sector has recently outperformed. If left uncorrected, these biases cause the composite score to function partly as a sector bet rather than a pure stock-selection signal.

Neutralization proceeds by regressing factor scores on sector and country indicator variables and retaining only the residuals:

$$
z_{i} = \alpha + \sum_{s} \beta_s \, S_{i,s} + \sum_{c} \gamma_c \, C_{i,c} + \varepsilon_i
$$

The residual $\varepsilon_i$ represents the stock-specific factor exposure after removing the component explained by sector and country membership. Optionally, the residuals are re-standardized to mean zero and unit standard deviation. An equivalent approach computes $z$-scores separately within each sector (or sector-country group) and stacks the results.

### Missing Data Treatment

Not all factors are available for every stock. Analyst coverage is sparse for smaller firms, insider transaction data arrives irregularly, and recently listed companies may lack sufficient financial history for all accounting ratios. The recommended treatment assigns a neutral score of zero (the cross-sectional mean after standardization) to any stock with a missing factor value and records a binary coverage flag. In composite scoring, each factor's contribution is weighted by its coverage flag, so that missing factors neither help nor hurt the overall score. This approach avoids imputation artifacts while ensuring that stocks with partial factor coverage are not systematically penalized.

### Point-in-Time Alignment

Ensuring that factor scores use only information available at the time of the investment decision is critical for avoiding look-ahead bias. Price-based and volume-based factors (momentum, volatility, liquidity) use data through the last business day of the prior month. Annual financial statements are assumed available 90 calendar days after the fiscal period end, and quarterly statements 45 days after the period end. At each rebalance date, the most recent report satisfying this publication lag constraint is used. Analyst data carries a five-day buffer to avoid same-day incorporation of recommendations that may not yet be widely disseminated. Macroeconomic indicators use data as of the end of the prior month, with an additional one-month lag for indicators subject to revision.

## Composite Scoring

### Equal-Weight Baseline

DeMiguel, Garlappi, and Uppal demonstrated that naive $1/N$ allocation frequently outperforms optimized portfolios in out-of-sample tests because estimation error in means and covariances overwhelms the theoretical gains from optimization. The same logic applies in factor space: equal-weighting reasonably predictive, low-correlated factor groups is robust and difficult to improve upon when sample sizes are limited.

The baseline composite organizes individual factors into groups (value, quality, investment, momentum, low risk, liquidity, dividend, sentiment, ownership) and computes the average sector-neutral $z$-score within each group. Group-level scores are then combined with weights that assign full weight to core groups (value, quality, momentum, low risk) and half weight to supplementary groups (liquidity, dividend, sentiment, ownership), normalized to sum to unity:

$$
\text{AlphaScore}_i = \sum_{g} w_g \cdot \overline{z}_{i,g}
$$

where $\overline{z}_{i,g}$ is the average standardized score of stock $i$ across factors in group $g$, and $w_g$ are the group weights.

### Information-Coefficient-Weighted Combinations

Rolling information coefficients provide a data-driven alternative to equal weighting. For each factor $f$ and each month $\tau$ in a trailing window, the Spearman rank correlation between factor scores and subsequent one-month returns yields a time series of IC realizations. The IC ratio, defined as the mean IC divided by its standard deviation, measures signal reliability:

$$
\text{ICIR}_f = \frac{\overline{\text{IC}}_f}{\sigma_{\text{IC},f}}
$$

Setting factor weights proportional to $\max(\text{ICIR}_f, 0)$ and renormalizing overweights factors with historically stable predictive power and underweights noisy or unreliable signals. This approach tends to tilt toward momentum and profitability, which typically exhibit the highest IC ratios, at the expense of noisier sentiment and ownership signals.

### Factor Interaction and Multicollinearity Diagnostics

Before finalizing composite weights, the correlation structure among factor-mimicking portfolio returns should be examined. The variance inflation factor for factor $f$,

$$
\text{VIF}_f = \frac{1}{1 - R_f^2}
$$

where $R_f^2$ is the coefficient of determination from regressing factor $f$ on all other factors, quantifies multicollinearity. Values exceeding 5 to 10 indicate that the factor is nearly a linear combination of others and should be merged into an existing group rather than included independently. Principal component analysis on standardized factor scores can further identify the effective dimensionality: retaining components explaining 70 to 80 percent of cross-sectional variance yields orthogonal factors suitable for regression-based expected return modeling.

Machine learning methods, including ridge regression and gradient boosting, can capture nonlinear factor interactions that linear composites miss. Gu, Kelly, and Xiu report that tree-based and neural-network models roughly double the Sharpe ratio of linear strategies on large US stock panels spanning several decades. However, for panels of 500 to 2000 stocks over 3 to 5 years, the gains are modest, on the order of 0.01 to 0.02 improvement in monthly rank IC, and the risk of overfitting is substantial. Shallow gradient-boosted trees with strong regularization represent the practical frontier; deeper architectures require more data than a typical institutional sample provides.

## Stock Selection and Turnover Control

### Fixed-Count Versus Quantile Selection

Two approaches convert composite scores into a selected universe. Fixed-count selection sorts all stocks by composite score and retains the top $N$ names (for example, 100 from a 1000-stock universe). Quantile selection retains all stocks in the top $Q$ percent (for example, the top quintile). Fixed-count selection guarantees a predictable universe size, which simplifies capacity planning and optimizer configuration, but it concentrates selection-boundary risk around the $N$-th ranked stock. Quantile selection provides style purity regardless of universe size fluctuations but can produce variable portfolio breadth. For a 500 to 2000 stock investable universe with monthly rebalancing, top-quintile selection typically yields 100 to 400 names.

### Buffer Zones and Hysteresis

Index providers use buffer bands around selection boundaries to reduce turnover. The same principle applies to factor-based selection. Under a fixed-count scheme, a stock currently in the selected set is retained if its new rank falls within $N + \text{buffer}$, where the buffer is typically 20 percent of the target count. New stocks enter in order of rank until the target count is met. Under quantile selection, a stock enters when its rank crosses into the top $Q_{\text{in}}$ percent (for example, 20 percent) and exits only when it falls below $Q_{\text{out}}$ percent (for example, 30 percent). This asymmetric boundary reduces one-way monthly turnover from the 25 to 35 percent range typical of unbuffered multi-factor selection to approximately 15 to 20 percent.

### Transaction Cost and Turnover Analysis

Turnover directly determines the implementation cost of any factor strategy. Single-factor momentum portfolios with monthly rebalancing exhibit one-way turnover of 50 to 100 percent per month, making them expensive in isolation. Multi-factor composites combining value, quality, and momentum reduce turnover to approximately 15 to 25 percent monthly because the factors have different rebalancing dynamics: value is slow-moving while momentum turns over rapidly. Quarterly rebalancing roughly halves these figures.

Net alpha after costs provides the definitive assessment of whether a selection strategy is worth implementing:

$$
\alpha_{\text{net}} = \alpha_{\text{gross}} - \text{Turnover} \times c
$$

where $c$ is the per-unit round-trip transaction cost. Novy-Marx and Velikov estimate that trading costs range from 5 to 20 basis points per unit of turnover for liquid developed-market equities, depending on capitalization and execution method. If gross alpha from selection is 3 percent annualized and annual one-way turnover is 240 percent (20 percent monthly), costs at 20 basis points per unit consume approximately 48 basis points, leaving roughly 2.5 percent net alpha before optimization decisions.

### Sector Balance Constraints

Unconstrained factor scoring can produce severe sector concentrations: value scores cluster in financials and energy, momentum clusters in whichever sector has recently rallied. Imposing sector balance at the selection stage ensures the optimizer receives a diversified starting universe. For each sector $s$, the selected universe weight $w_s^{\text{sel}}$ is constrained to lie within a band around the parent universe weight:

$$
w_s^{\text{parent}} - \delta \leq w_s^{\text{sel}} \leq w_s^{\text{parent}} + \delta
$$

where $\delta$ is an absolute tolerance (for example, 5 percentage points). Implementation proceeds by computing target counts per sector proportional to parent weights, selecting the top-ranked stocks within each sector up to the target count, and adjusting to reach the overall target through marginal additions or removals in sectors with the largest score gaps.

## Regime-Conditional Factor Tilts

### Macro Regime Definitions

Factor premiums are not stationary across the business cycle. MSCI research and the Invesco dynamic multifactor framework document that value, momentum, quality, and low-volatility factors exhibit distinct performance patterns across macro regimes defined by combinations of GDP growth, unemployment dynamics, inflation, and yield curve slope. A four-regime classification based on these indicators provides a tractable framework:

- **Expansion**: above-trend growth, declining unemployment, positive yield curve slope
- **Slowdown**: below-trend but positive growth, rising unemployment
- **Recession**: negative growth, elevated and rising unemployment, flat or inverted yield curve
- **Recovery**: growth turning positive from negative, declining unemployment, positive yield curve slope

Macro regimes are not equally frequent. Expansions dominate the historical record, accounting for roughly 60 to 75 percent of months, with recessions clustered but infrequent at 10 to 15 percent.

### Factor-Regime Interaction Evidence

Empirical studies consistently find that value and size premiums are strongest during expansions and recoveries, when risk appetite is high and distressed firms benefit from improving conditions. Quality and low-volatility premiums are strongest during slowdowns and recessions, when investors seek defensive characteristics and earnings resilience. Momentum performs well during stable trends but suffers severely at regime turning points, particularly the onset of recoveries, when recent losers rally sharply. Dividend yield behaves defensively, outperforming during slowdowns and recessions. These patterns, documented by Gupta, Kassam, Suryanarayanan, and Varga among others, provide the empirical basis for regime-conditional factor tilts.

### Regime-Aware Scoring

Regime-conditional scoring applies multiplicative tilts to the baseline group weights:

$$
w_g^{\text{regime}} = w_g^{\text{baseline}} \times m_g(R_t)
$$

where $m_g(R_t)$ is a regime-dependent multiplier for factor group $g$ given the current regime $R_t$. The tilted weights are renormalized to sum to unity. Representative multipliers increase value and size weights by 20 to 40 percent during expansions and recoveries while reducing them during recessions, and increase quality and low-volatility weights by 30 to 60 percent during slowdowns and recessions while reducing them during expansions.

### Detection Lag and Out-of-Sample Efficacy

Real-time regime detection lags the true regime by one to three months due to publication delays in macroeconomic data and the need for confirmation across multiple indicators. Asness notes skeptically that much of the theoretical benefit of factor timing is eroded once detection lags and transaction costs are incorporated. Hodges, Hogan, Peterson, and Ang find that macro-timed multifactor strategies improve Sharpe ratios by 0.1 to 0.2 before costs, with realized improvements smaller but still potentially meaningful when tilts are moderate and turnover is controlled.

Regime conditioning should therefore be viewed as a moderate tilt on top of robust static factors rather than an aggressive timing strategy. The baseline equal-weight composite provides the structural foundation; regime tilts adjust it at the margin.

## Validation Protocol

### Information Coefficient Analysis

For each factor $f$, the monthly cross-sectional Spearman rank correlation between standardized scores $z_{i,f,t}$ and subsequent one-month returns $r_{i,t+1}$ yields the information coefficient $\text{IC}_{f,t}$. The time-series mean, standard deviation, and IC ratio summarize predictive power and consistency. Statistical significance requires Newey-West adjustment for autocorrelation in the IC series, with a lag of 3 to 6 months appropriate for monthly factor data. Harvey, Liu, and Zhu recommend $t$-statistic thresholds of approximately 3.0 when many factors have been tested, to control the false discovery rate implied by collective data mining across the literature. For a pre-specified factor set of 20 or fewer, a pragmatic rule treats $|t| \geq 3.0$ as strong evidence, $2.0 \leq |t| < 3.0$ as tentative evidence requiring out-of-sample confirmation, and $|t| < 2.0$ as insufficient.

### Quantile Spread Returns

Sorting the investable universe into quintiles by factor score and computing the return spread between the top and bottom quintiles provides a direct measure of economic magnitude. For each factor and month:

$$
\text{LS}_{f,t+1} = \bar{r}_{Q_{\text{high}},t+1} - \bar{r}_{Q_{\text{low}},t+1}
$$

Annualized spreads are compared against literature benchmarks: approximately 3 to 5 percent for value, 6 percent for profitability, 8 to 12 percent for momentum, and 2 to 4 percent for illiquidity. Spreads substantially below these benchmarks may indicate implementation issues or sample-specific attenuation, while spreads substantially above suggest potential look-ahead bias or survivorship contamination.

### Factor-Mimicking Portfolios

Constructing long-short factor-mimicking portfolios within the investable universe provides return time series for estimating factor premia, computing cross-factor correlations, and testing regime-conditional performance. Each month, stocks are ranked by sector-neutral factor score, with the top 30 percent forming the long leg and the bottom 30 percent forming the short leg, both equal-weighted. Optionally, the long and short legs are scaled to be dollar-neutral and beta-neutral. These factor return series serve as the inputs for the cross-factor correlation analysis described above and for the factor premium estimation used in the integration with portfolio optimization.

### Multiple Testing Correction

With ten or more factors and multiple construction variants, the risk of discovering spurious predictability through data mining is substantial. Harvey, Liu, and Zhu show that naive $|t| > 1.96$ thresholds drastically understate the false discovery rate when hundreds of factors have been tested collectively across the academic literature. For the pre-specified factor set used here, family-wise error rate control via the Bonferroni or Holm procedure, or false discovery rate control via the Benjamini-Hochberg procedure, provides a disciplined framework for distinguishing robust factors from statistical artifacts.

### Out-of-Sample Protocols

Given limited history of 3 to 5 years, time-series block cross-validation provides the most reliable out-of-sample assessment. Training on the first 36 months and validating on the subsequent 12 months, then rolling forward by 6 months and repeating, yields multiple non-overlapping test periods. For each factor and composite, out-of-sample IC and long-short spreads are computed per fold and aggregated across folds to assess stability. Combinatorial purged cross-validation offers a complementary approach that generates a larger number of test paths from the same data, at the cost of greater computational expense.

A meaningful stock-selection signal should outperform an equal-weighted benchmark of all investable names by at least 2 to 3 percent annualized gross, deliver a Sharpe improvement of at least 0.1 to 0.2, and exhibit statistically significant IC series with $t$-statistics above 2.5 to 3.0 across multiple out-of-sample folds.

## Integration with Portfolio Optimization

### Mapping Factor Scores to Expected Returns

Factor scores translate into expected return estimates through a linear mapping:

$$
\mathbb{E}[r_i] = r_f + \lambda_{\text{Mkt}} \beta_i + \sum_{g} \lambda_g \cdot \overline{z}_{i,g}
$$

where $r_f$ is the risk-free rate, $\lambda_{\text{Mkt}}$ is the market risk premium, $\beta_i$ is the market beta, and $\lambda_g$ are factor risk premia estimated as the time-series means of factor-mimicking portfolio returns. These expected returns serve as the $\boldsymbol{\mu}$ vector in mean-variance optimization or as the basis for view formation in the Black-Litterman framework.

### Black-Litterman with Factor Views

The Black-Litterman model provides a natural mechanism for incorporating factor-based views into portfolio construction. The equilibrium expected return vector $\boldsymbol{\Pi} = \delta \boldsymbol{\Sigma} \mathbf{w}_{\text{mkt}}$ serves as the prior, reflecting the market's implicit return expectations. Factor views are expressed through a pick matrix $\mathbf{P}$ whose rows encode the factor exposures of long-short portfolios, and a view vector $\mathbf{Q}$ containing the expected factor premia. The posterior expected returns are:

$$
\boldsymbol{\mu}_{\text{BL}} = \boldsymbol{\Pi} + \boldsymbol{\Sigma} \mathbf{P}^\top \left(\mathbf{P} \boldsymbol{\Sigma} \mathbf{P}^\top + \boldsymbol{\Omega}\right)^{-1} (\mathbf{Q} - \mathbf{P} \boldsymbol{\Pi})
$$

where $\boldsymbol{\Omega}$ is the view uncertainty covariance matrix, typically set proportional to the view variance to reflect confidence in each factor view. This formulation tilts the equilibrium returns toward the factor views in proportion to view precision, producing well-diversified portfolios that incorporate factor information without the extreme weights that unconstrained mean-variance optimization generates from noisy expected return estimates.

### Factor Exposure Constraints

Factor scores also enable linear exposure constraints within the optimizer. The portfolio's exposure to factor group $g$ is:

$$
\text{Exposure}_g = \sum_{i} w_i \cdot z_{i,g}
$$

Constraining this quantity to lie within specified bounds (for example, $0.3 \leq \text{Exposure}_{\text{Value}} \leq 1.5$ standard deviations) ensures that the optimized portfolio maintains target factor tilts. These are linear constraints in the portfolio weights and integrate directly into quadratic or conic optimization programs without altering the computational complexity.

## Data Limitations

Several constraints bound what the pre-selection module can achieve with the available data. The absence of intraday data precludes microstructure factors such as realized intraday volatility, effective bid-ask spreads, and order-book imbalance; liquidity estimation relies entirely on daily volume and range measures. The absence of options data rules out implied volatility, skewness, and variance risk premium signals that are informative for certain strategies. No ESG or governance data is available, restricting screening to purely financial criteria.

Survivorship bias poses a systematic concern. If the estimation sample includes only currently listed instruments, historical returns and factor premia are upwardly biased because the worst-performing stocks, those that delisted, defaulted, or were acquired at distressed valuations, are systematically excluded. Elton, Gruber, and Blake estimate that survivorship bias inflates average performance by approximately 0.9 percent per annum in mutual fund samples, with potentially larger effects in equity factor research where delisted stocks disproportionately populate the losing side of value and momentum sorts. Backtested excess returns below approximately 2 percent annualized should be treated as potentially illusory, and evaluation should emphasize Sharpe ratios, information coefficients, and robustness across subperiods rather than raw return magnitudes.

\newpage
