// ============================================================================
// STOCK SIGNAL GENERATION - LLM-BASED TRADING SIGNAL ANALYSIS
// ============================================================================
// Institutional-grade stock signal generation using comprehensive yfinance data.
// Analyzes valuation, momentum, quality, growth, and technical factors to
// classify stocks into trading signal categories.
//
// Signal Types:
// - large_decline: Significant downward expectation (> 3% potential decline)
// - small_decline: Minor downward expectation (0-3% potential decline)
// - neutral: No significant movement expected (�0.5%)
// - small_gain: Minor upward expectation (0-3% potential gain)
// - large_gain: Significant upward expectation (> 3% potential gain)
// ============================================================================

// ============================================================================
// MACRO REGIME DATA
// ============================================================================

class MacroRegimeData {
  // Country and timestamp
  country string @description("Country code (USA, Germany, etc.)")
  assessment_date string @description("Date of macro assessment (YYYY-MM-DD)")

  // Business cycle regime
  regime string @description("Current regime: EARLY_CYCLE, MID_CYCLE, LATE_CYCLE, RECESSION, UNCERTAIN")
  confidence float @description("Regime classification confidence (0.0-1.0)")
  rationale string @description("Brief rationale for regime classification")

  // Key market indicators
  ism_pmi float? @description("ISM Manufacturing PMI (if available)")
  ism_signal string? @description("ISM signal: STRONG_EXPANSION, MILD_EXPANSION, MILD_CONTRACTION, DEEP_CONTRACTION")

  yield_curve_2s10s float? @description("2s10s yield spread in bps (if available)")
  yield_curve_signal string? @description("Yield curve signal: STEEP, NORMAL, FLAT, INVERTED")

  hy_spread float? @description("High-yield credit spread in bps (if available)")
  credit_spread_signal string? @description("Credit spread signal: TIGHT, NEUTRAL, WIDENING, STRESS")

  vix float? @description("VIX volatility index (if available)")

  // Recession risk
  recession_risk_6m float @description("6-month recession probability (0.0-1.0)")
  recession_risk_12m float @description("12-month recession probability (0.0-1.0)")

  // Economic indicators
  gdp_growth_yy float? @description("GDP growth year-over-year %")
  unemployment float? @description("Unemployment rate %")
  inflation float? @description("Inflation rate %")

  // Portfolio positioning context
  recommended_overweights string[] @description("Sectors currently recommended for overweight")
  recommended_underweights string[] @description("Sectors currently recommended for underweight")
  factor_exposure string? @description("Recommended factor exposure: GROWTH_MOMENTUM, QUALITY_DEFENSIVE, BALANCED")

  // Risk monitoring
  primary_risks string[] @description("2-3 main macro risks to monitor")
}

// ============================================================================
// ENUMS AND OUTPUT TYPES
// ============================================================================

enum SignalType {
  // Bearish Signals
  LARGE_DECLINE @alias("large_decline") @description(#"
    Significant downward movement expected (> 3%).
    Indicators: Overvaluation, negative momentum, deteriorating fundamentals,
    bearish technicals, negative analyst sentiment.
  "#)

  SMALL_DECLINE @alias("small_decline") @description(#"
    Minor downward movement expected (0-3%).
    Indicators: Slight overvaluation, weakening momentum, mixed fundamentals,
    neutral to slightly bearish technicals.
  "#)

  // Neutral
  NEUTRAL @alias("neutral") @description(#"
    No significant movement expected (�0.5%).
    Indicators: Fair valuation, sideways momentum, stable fundamentals,
    neutral technicals, mixed analyst sentiment.
  "#)

  // Bullish Signals
  SMALL_GAIN @alias("small_gain") @description(#"
    Minor upward movement expected (0-3%).
    Indicators: Slight undervaluation, positive momentum, improving fundamentals,
    neutral to slightly bullish technicals.
  "#)

  LARGE_GAIN @alias("large_gain") @description(#"
    Significant upward movement expected (> 3%).
    Indicators: Strong undervaluation, strong positive momentum, excellent fundamentals,
    bullish technicals, positive analyst consensus.
  "#)
}

// Confidence level for signal classification
enum ConfidenceLevel {
  LOW @alias("low") @description("< 60% confidence due to mixed signals or limited data")
  MEDIUM @alias("medium") @description("60-80% confidence with moderate signal alignment")
  HIGH @alias("high") @description("> 80% confidence with strong signal alignment across factors")
}

// Key factors driving the signal
class SignalDrivers {
  // Valuation Assessment (-1 to +1: -1 = very overvalued, +1 = very undervalued)
  valuation_score float @description("Composite valuation score based on P/E, P/B, price vs targets")
  valuation_summary string @description("Brief valuation assessment (1-2 sentences)")

  // Momentum Assessment (-1 to +1: -1 = strong downtrend, +1 = strong uptrend)
  momentum_score float @description("Price momentum based on 52-week performance, moving averages")
  momentum_summary string @description("Brief momentum assessment (1-2 sentences)")

  // Quality Assessment (0 to 1: quality of business fundamentals)
  quality_score float @description("Business quality based on margins, ROE, debt levels")
  quality_summary string @description("Brief quality assessment (1-2 sentences)")

  // Growth Assessment (-1 to +1: -1 = contracting, +1 = high growth)
  growth_score float @description("Growth prospects based on revenue/earnings growth")
  growth_summary string @description("Brief growth assessment (1-2 sentences)")

  // Technical Assessment (-1 to +1: -1 = very bearish, +1 = very bullish)
  technical_score float? @description("Technical indicators: RSI, moving averages, volume trends")
  technical_summary string? @description("Brief technical assessment if data available")

  // Analyst Sentiment (-1 to +1: -1 = strong sell consensus, +1 = strong buy consensus)
  analyst_score float? @description("Analyst consensus and ratings")
  analyst_summary string? @description("Brief analyst sentiment summary")
}

// Risk factors and considerations
class RiskFactors {
  // Market risk factors
  volatility_level string @description("Low/Medium/High - historical volatility assessment")
  beta_risk string? @description("Market sensitivity assessment if beta available")

  // Company-specific risks
  debt_risk string @description("Assessment of debt levels and financial leverage")
  liquidity_risk string @description("Assessment of trading liquidity and market depth")

  // Data quality concerns
  data_gaps string[] @description("List of missing or unreliable data points that affected analysis")

  // Key risks summary
  primary_risks string[] @description("Top 3-5 key risks to the signal thesis")
}

// Main output structure matching database schema
class StockSignalOutput {
  // Signal Classification (required)
  signal_type SignalType @description("Primary trading signal classification")

  // Date and Price Data
  signal_date string @description("Date for the signal (YYYY-MM-DD format)")
  close_price float? @description("Current/most recent closing price")
  open_price float? @description("Most recent opening price")

  // Daily metrics
  daily_return float? @description("Most recent daily return percentage")
  volume float? @description("Recent trading volume")

  // Technical indicators
  volatility float? @description("Historical volatility (annualized standard deviation)")
  rsi float? @description("Relative Strength Index if available from yfinance")

  // Analysis metadata
  data_quality_score float @description(#"
    Data quality score 0-1 based on completeness of input data.
    1.0 = all key fields present, 0.5 = moderate data, 0.0 = minimal data
  "#)

  confidence_level ConfidenceLevel @description("Confidence in signal classification")

  // Detailed analysis (required)
  analysis_notes string @description(#"
    Concise analysis explaining the signal classification.
    Maximum 2-3 sentences covering:
    1. Primary signal driver and rationale
    2. Key supporting factors or risks
    3. Overall conviction level
    Professional, institutional tone. Be concise and actionable.
  "#)

  // Structured signal drivers
  signal_drivers SignalDrivers @description("Quantified assessment across key factors")

  // Risk assessment
  risk_factors RiskFactors @description("Identified risks and data quality concerns")

  // Price targets (if derivable from data)
  upside_potential_pct float? @description("Estimated upside % to fair value or target")
  downside_risk_pct float? @description("Estimated downside % risk to support levels")
}

// ============================================================================
// MAIN SIGNAL GENERATION FUNCTION
// ============================================================================

function GenerateStockSignal(
  stock_data: ComprehensiveYFinanceData,
  target_date: string,
  macro_regime: MacroRegimeData?,
  recent_news: NewsArticle[]?,
  calculated_open_price: float?,
  calculated_volatility: float?,
  calculated_rsi: float?
) -> StockSignalOutput {
  client Model

  prompt #"
    You are an institutional quantitative analyst generating a trading signal for {{ stock_data.ticker }} ({{ stock_data.company_name }}) as of {{ target_date }}.

    === STOCK DATA ===
    Ticker: {{ stock_data.ticker }} | Data Completeness: {{ stock_data.data_completeness_pct }}% | Fields: {{ stock_data.yfinance_info_dict_size }}

    === CALCULATED METRICS (use these exact values) ===
    {% if calculated_open_price %}Open: ${{ calculated_open_price }} | {% endif %}{% if calculated_volatility %}Vol: {{ (calculated_volatility * 100) | round(2) }}% | {% endif %}{% if calculated_rsi %}RSI: {{ calculated_rsi | round(2) }}{% endif %}

    {% if macro_regime %}
    === MACRO CONTEXT: {{ macro_regime.country }} ({{ macro_regime.regime }}, {{ (macro_regime.confidence * 100) | round(0) }}% confidence) ===
    Recession Risk: 6m={{ (macro_regime.recession_risk_6m * 100) | round(0) }}%, 12m={{ (macro_regime.recession_risk_12m * 100) | round(0) }}%
    {% if macro_regime.ism_pmi %}ISM={{ macro_regime.ism_pmi }} ({{ macro_regime.ism_signal }}) | {% endif %}{% if macro_regime.yield_curve_2s10s %}Curve={{ macro_regime.yield_curve_2s10s }}bps ({{ macro_regime.yield_curve_signal }}) | {% endif %}{% if macro_regime.hy_spread %}HY={{ macro_regime.hy_spread }}bps{% endif %}
    Overweight: {{ macro_regime.recommended_overweights | join(', ') }} | Underweight: {{ macro_regime.recommended_underweights | join(', ') }}

    REGIME RULES:
    {% if macro_regime.regime == 'RECESSION' or macro_regime.recession_risk_12m > 0.4 %}
    RECESSION: Quality=35%, Val=25%, Mom=15%, Growth=10%. Require ROE>15%, D/E<1.0, +FCF. Downgrade unless defensive.
    {% elif macro_regime.regime == 'LATE_CYCLE' %}
    LATE_CYCLE: Quality=30%, Val=25%, Growth=15%, Mom=15%. Favor defensives, strong balance sheets.
    {% elif macro_regime.regime == 'EARLY_CYCLE' %}
    EARLY_CYCLE: Mom=30%, Growth=25%, Val=15%, Quality=15%. Favor cyclicals, growth stocks.
    {% else %}
    MID_CYCLE: Val=25%, Mom=20%, Quality=20%, Growth=20%. Balanced approach.
    {% endif %}
    Sector alignment: Overweight +0.2, Underweight -0.2 to composite score.
    {% endif %}

    {% if recent_news and recent_news | length > 0 %}
    === NEWS ({{ recent_news | length }} articles) ===
    {% for article in recent_news %}
    [{{ loop.index }}] {{ article.title }} ({{ article.publisher }}, {{ article.date }})
    {% if article.summary %}Summary: {{ article.summary }}{% endif %}
    {% endfor %}

    NEWS RULES: Bullish=earnings beats, innovations, partnerships, upgrades. Bearish=misses, lawsuits, layoffs, downgrades. Weight recent articles higher. Integrate sentiment into momentum/analyst scores.
    {% endif %}

    {% if stock_data.company_info %}
    === COMPANY ===
    {{ stock_data.company_info.sector }} / {{ stock_data.company_info.industry }} / {{ stock_data.company_info.country }} | Employees: {{ stock_data.company_info.fullTimeEmployees }}
    {{ stock_data.company_info.longBusinessSummary }}
    {% endif %}

    {% if stock_data.price_data %}
    === PRICE & VALUATION ===
    Price: ${{ stock_data.price_data.currentPrice }} | 52W: ${{ stock_data.price_data.fiftyTwoWeekLow }}-${{ stock_data.price_data.fiftyTwoWeekHigh }} (Chg: {{ stock_data.price_data.fiftyTwoWeekChangePercent }}%, From High: {{ stock_data.price_data.fiftyTwoWeekHighChangePercent }}%)
    {% if stock_data.price_data.targetMeanPrice %}Target: ${{ stock_data.price_data.targetMeanPrice }} (Upside: {{ ((stock_data.price_data.targetMeanPrice - stock_data.price_data.currentPrice) / stock_data.price_data.currentPrice * 100) }}%) | {% endif %}P/B: {{ stock_data.price_data.priceToBook }} | P/S: {{ stock_data.price_data.priceToSalesTrailing12Months }}
    {% endif %}

    {% if stock_data.technical_data %}
    === TECHNICALS ===
    P/E: {{ stock_data.technical_data.trailingPE }} (Fwd: {{ stock_data.technical_data.forwardPE }}) | EPS: ${{ stock_data.technical_data.epsTrailingTwelveMonths }} (Fwd: ${{ stock_data.technical_data.forwardEps }}) | Beta: {{ stock_data.technical_data.beta }}
    MAs: 50D=${{ stock_data.technical_data.fiftyDayAverage }} ({{ stock_data.technical_data.fiftyDayAverageChangePercent }}%), 200D=${{ stock_data.technical_data.twoHundredDayAverage }} ({{ stock_data.technical_data.twoHundredDayAverageChangePercent }}%)
    {% endif %}

    {% if stock_data.financial_metrics %}
    === FINANCIALS ===
    Margins: Profit={{ stock_data.financial_metrics.profitMargins }}, Gross={{ stock_data.financial_metrics.grossMargins }}, Op={{ stock_data.financial_metrics.operatingMargins }}
    Returns: ROE={{ stock_data.financial_metrics.returnOnEquity }}, ROA={{ stock_data.financial_metrics.returnOnAssets }}
    Growth: Rev={{ stock_data.financial_metrics.revenueGrowth }}, Earn={{ stock_data.financial_metrics.earningsGrowth }}, QtrEarn={{ stock_data.financial_metrics.earningsQuarterlyGrowth }}
    Balance: D/E={{ stock_data.financial_metrics.debtToEquity }}, Current={{ stock_data.financial_metrics.currentRatio }}, Quick={{ stock_data.financial_metrics.quickRatio }}
    Cash Flow: OCF=${{ stock_data.financial_metrics.operatingCashflow }}, FCF=${{ stock_data.financial_metrics.freeCashflow }} | Cash=${{ stock_data.financial_metrics.totalCash }}, Debt=${{ stock_data.financial_metrics.totalDebt }}
    {% endif %}

    {% if stock_data.market_data %}
    === MARKET DATA ===
    MCap: ${{ stock_data.market_data.marketCap }} | Vol: {{ stock_data.market_data.volume }} (10d: {{ stock_data.market_data.averageVolume10days }}, 3m: {{ stock_data.market_data.averageDailyVolume3Month }})
    {% if stock_data.market_data.sharesShort %}Short: {{ stock_data.market_data.shortPercentOfFloat }}% float | {% endif %}Inst: {{ stock_data.market_data.heldPercentInstitutions }}, Insider: {{ stock_data.market_data.heldPercentInsiders }}
    {% endif %}

    {% if stock_data.analyst_data %}
    === ANALYSTS ===
    {{ stock_data.analyst_data.recommendationKey }} (Score: {{ stock_data.analyst_data.recommendationMean }}/5, {{ stock_data.analyst_data.numberOfAnalystOpinions }} analysts)
    {% endif %}

    {% if stock_data.dividend_data %}
    === DIVIDEND ===
    Yield: {{ stock_data.dividend_data.dividendYield }} | Rate: ${{ stock_data.dividend_data.dividendRate }} | 5Y Avg: {{ stock_data.dividend_data.fiveYearAvgDividendYield }}
    {% endif %}

    {% if stock_data.governance %}
    === GOVERNANCE ===
    Risk: Overall={{ stock_data.governance.overallRisk }}, Board={{ stock_data.governance.boardRisk }}, Comp={{ stock_data.governance.compensationRisk }}
    {% endif %}

    === FACTOR ANALYSIS (Fama-French + Carhart) ===

    1. VALUATION (-1 to +1): P/E (<15 value, 15-25 fair, >30 growth/overvalued), P/B, FCF yield (>5% attractive), price vs targets. Score: -1 (very overvalued) to +1 (deep value).

    2. MOMENTUM (-1 to +1): 52W performance, MA position (above 50D+200D = strong uptrend +0.5/+1.0, below both = downtrend -0.5/-1.0), relative strength vs sector. Score: -1 (strong downtrend) to +1 (strong uptrend).

    3. QUALITY (0 to 1): ROE (>15% high, <10% low), ROA (>8% strong, <3% weak), margins (Op>15% excellent, <5% weak), D/E (<0.5 conservative, >2.0 leveraged), current ratio (>1.5 good, <1.0 weak), FCF (positive/growing). Score: 0 (poor quality) to 1 (fortress balance sheet).

    4. GROWTH (-1 to +1): Rev growth (>20% high, 5-10% moderate, <0% contraction), Earn growth (>15% strong), sustainable/profitable growth. Avoid excessive M&A. Score: -1 (contracting) to +1 (sustainable high growth).

    5. TECHNICAL (-1 to +1, if available): RSI (<30 oversold, >70 overbought), MA trends, volume patterns, beta (>1.5 high-beta, <0.7 defensive). Score: -1 (bearish) to +1 (bullish).

    6. ANALYST (-1 to +1, if available): Rec mean (1-1.5 Strong Buy, 2.5-3.5 Hold, 4.5-5 Sell), coverage breadth, price target upside. Score: -1 (sell consensus) to +1 (buy consensus).

    SIZE (informational): Mega >$200B, Large $10-200B, Mid $2-10B, Small $300M-2B, Micro <$300M. Note in risk_factors.

    === SIGNAL CLASSIFICATION ===

    COMPOSITE: Weight factors by regime (already specified above), add sector alignment (±0.2).

    RISK ADJUSTMENTS:
    - D/E>2.5 (RECESSION/LATE) or >4.0 (any): Downgrade 1 category
    - Vol<$1M: LOW confidence | Vol<$5M: note liquidity risk
    - Neg FCF+low growth: max neutral | P/E>50 w/o growth: cap small_gain | P/E>100: cap neutral
    - Data<50%: LOW confidence, cap small_gain/decline

    THRESHOLDS:
    LARGE_GAIN: >+0.4 + 3+ aligned factors | SMALL_GAIN: +0.15 to +0.4 | NEUTRAL: -0.15 to +0.15 | SMALL_DECLINE: -0.4 to -0.15 | LARGE_DECLINE: <-0.4 + 3+ aligned

    OVERRIDES: RECESSION+D/E>2.0+neg FCF=max small_decline | LATE_CYCLE+P/E>30+weak mom=neutral | Distressed (int cov<1.5)=max small_decline

    === OUTPUT REQUIREMENTS ===

    1. signal_type: Use thresholds above with risk adjustments
    2. confidence_level: HIGH (>80%, data>70%, 4+ aligned) | MEDIUM (60-80%, data 50-70%, 2-3 aligned) | LOW (<60%, data<50% OR conflicting)
    3. analysis_notes: 2-3 sentences. S1: Primary thesis+driver. S2: Supporting factors/risks. S3: Conviction+positioning. Institutional tone.
    4. signal_drivers: Score each factor per framework above. 1-2 sentence summary with specific metrics.
       - Valuation (-1 to +1): P/E, P/B, FCF yield, vs targets
       - Momentum (-1 to +1): 52W perf, MA position, relative strength
       - Quality (0 to 1): ROE, ROA, margins, D/E, FCF
       - Growth (-1 to +1): Rev/Earn growth, sustainability
       - Technical (-1 to +1, opt): RSI, MAs, beta
       - Analyst (-1 to +1, opt): Rec consensus, target upside
    5. risk_factors:
       - volatility_level: Low (<20%), Medium (20-40%), High (>40%)
       - beta_risk (opt): Market sensitivity if beta available
       - debt_risk: D/E, interest coverage, flag if concerning
       - liquidity_risk: ADV, market cap, flag if <$5M ADV or micro-cap
       - data_gaps[]: List missing data
       - primary_risks[]: Top 3-5 risks (macro, company, valuation, liquidity)
    6. data_quality_score: 0.9-1.0 (comprehensive) | 0.7-0.9 (good) | 0.5-0.7 (moderate) | 0.3-0.5 (limited) | 0-0.3 (minimal)
    7. Price targets (opt): upside_potential_pct (analyst target or fair value), downside_risk_pct (support levels, bear case). Express as %.
    8. Price/volume: signal_date={{ target_date }}, close_price (currentPrice), open_price={{ calculated_open_price if calculated_open_price else "null" }}, volume, volatility={{ calculated_volatility if calculated_volatility else "null" }}, rsi={{ calculated_rsi if calculated_rsi else "null" }}

    CHECKS: ✓ Coherent scores ✓ Conservative confidence ✓ Concise notes ✓ Specific risks ✓ Honest data_gaps ✓ Data-driven targets

    {{ ctx.output_format }}
  "#
}