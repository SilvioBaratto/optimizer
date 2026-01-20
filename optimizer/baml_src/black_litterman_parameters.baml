// BAML functions for dynamic Black-Litterman parameter selection
// Uses market regime, signal characteristics, and volatility to recommend BL parameters

// ============================================================================
// INPUT TYPES
// ============================================================================

class MarketVolatilityMetrics {
  current_vix float?  // VIX level
  market_realized_vol float  // Realized volatility of broad market (annualized)
  signal_universe_vol float  // Average volatility of stocks with signals
  vol_regime string  // "low" (<15%), "normal" (15-25%), "high" (>25%)
}

class SignalCharacteristics {
  total_signals int
  buy_signals int
  strong_buy_signals int
  avg_confidence float  // Average confidence across all signals
  confidence_dispersion float  // Standard deviation of confidence scores
  signal_concentration float  // Herfindahl index or top-5 concentration
}

class FREDIndicators {
  ism_pmi float?  // ISM Manufacturing PMI
  ism_signal string?  // "expansion", "neutral", "contraction"
  yield_curve_2s10s float?  // 2s10s yield curve spread (bps)
  curve_signal string?  // "inverted", "flat", "normal", "steep"
  hy_spread float?  // High yield credit spread (bps)
  ig_spread float?  // Investment grade credit spread (bps)
  credit_signal string?  // "risk_off", "neutral", "risk_on"
  vix float?  // VIX level
  vix_signal string?  // "low", "normal", "elevated", "panic"
}

class CountryQuantitativeData {
  country string  // Country name
  gdp_growth_qq float?  // GDP growth quarter-over-quarter (%)
  industrial_production float?  // Industrial production growth (%)
  unemployment float?  // Unemployment rate (%)
  consumer_prices float?  // Consumer price inflation (%)
  deficit float?  // Budget deficit as % of GDP
  debt float?  // Government debt as % of GDP
  st_rate float?  // Short-term interest rate (%)
  lt_rate float?  // Long-term interest rate (%)
}

class CountryForecastData {
  country string  // Country name
  last_inflation float?  // Most recent inflation reading (%)
  inflation_6m float?  // Inflation forecast 6 months out (%)
  inflation_10y_avg float?  // 10-year average inflation forecast (%)
  gdp_growth_6m float?  // GDP growth forecast 6 months out (%)
  earnings_12m float?  // Earnings growth forecast 12 months (%)
  eps_expected_12m float?  // EPS expected growth 12 months (%)
  peg_ratio float?  // Price/Earnings to Growth ratio
  lt_rate_forecast float?  // Long-term rate forecast (%)
}

class RegimeNewsAnalysis {
  regime_analysis_summary string?  // BAML analysis summary from regime report
  news_summary string?  // Summary of key news signals
  key_news_signals string[]  // Top 3-5 news signals
  news_alignment string?  // How news aligns with quantitative regime
  warning_signs string[]  // Recession/risk warning signs from news (0-5)
  leading_indicators string[]  // Forward-looking indicators from news (0-5)
}

class G7CountrySnapshot {
  country string  // Country name
  gdp_growth_qq float?  // GDP growth Q/Q (%)
  unemployment float?  // Unemployment rate (%)
  inflation float?  // Current inflation (%)
  inflation_forecast_6m float?  // Inflation forecast 6 months (%)
  gdp_forecast_6m float?  // GDP forecast 6 months (%)
  policy_rate float?  // Current policy rate (%)
  regime_assessment string?  // Brief regime description (e.g., "early_cycle", "late_cycle")
}

class G7MacroOverview {
  countries G7CountrySnapshot[]  // All G7 countries
  synchronized_slowdown bool?  // Are G7 economies slowing in unison?
  divergence_note string?  // Key divergences across G7
  global_growth_signal string?  // "accelerating", "stable", "decelerating"
  global_inflation_signal string?  // "rising", "stable", "falling"
  monetary_policy_direction string?  // "tightening", "neutral", "easing"
  key_risks string[]  // Top 2-3 cross-border risks (0-3)
}

class MacroRegimeInput {
  // Basic regime classification
  current_regime string  // early_cycle, mid_cycle, late_cycle, recession, uncertain
  regime_confidence float  // 0-1
  recession_risk_6m float  // 0-1
  recession_risk_12m float  // 0-1
  market_stress_level string  // "low", "moderate", "high", "extreme"

  // FRED global indicators
  fred_indicators FREDIndicators?

  // Primary country (USA) data
  usa_quantitative CountryQuantitativeData?
  usa_forecast CountryForecastData?

  // News and AI-enhanced analysis
  regime_news_analysis RegimeNewsAnalysis?

  // G7 macro overview
  g7_macro_overview G7MacroOverview?
}

class HistoricalPerformance {
  recent_sharpe_ratio float?  // Last optimization run Sharpe ratio
  recent_turnover float?  // Portfolio turnover from last rebalance
  avg_tracking_error float?  // Historical tracking error vs benchmark
}

// ============================================================================
// OUTPUT TYPES
// ============================================================================

class BlackLittermanParameters {
  // Core BL Parameters
  tau float  // Prior uncertainty scalar (0.01-0.10, typically 0.025)
  delta float  // Risk aversion coefficient (1.5-4.0, typically 2.5)

  // Expected Return Ranges (Annual)
  strong_buy_return_min float  // Minimum expected return for Strong Buy
  strong_buy_return_max float  // Maximum expected return for Strong Buy
  buy_return_min float  // Minimum expected return for Buy
  buy_return_max float  // Maximum expected return for Buy

  // View Uncertainty Scaling
  uncertainty_min_pct float  // Minimum uncertainty (high confidence) - std dev
  uncertainty_max_pct float  // Maximum uncertainty (low confidence) - std dev
  confidence_scaling float  // Scaling factor for confidence-to-uncertainty conversion

  // Weight Constraints
  min_position_weight float  // Minimum weight per position (e.g., 0.02 = 2%)
  max_position_weight float  // Maximum weight per position (e.g., 0.10 = 10%)

  // Rationale
  overall_rationale string  // 3-4 sentences explaining parameter choices
  tau_rationale string  // Why this tau value?
  delta_rationale string  // Why this risk aversion level?
  return_expectations_rationale string  // Why these return ranges?
  uncertainty_rationale string  // Why this uncertainty scaling?
  weight_constraints_rationale string  // Why these weight limits?

  // Regime Adjustments Applied
  regime_multiplier float  // Overall regime adjustment multiplier (0.8-1.2)
  volatility_adjustment string  // How volatility regime affected parameters
  confidence_adjustment string  // How signal confidence distribution affected parameters

  // Risk Warnings
  parameter_risks string[]  // Risks associated with these parameter choices (0-3)
  monitoring_recommendations string[]  // What to monitor given these parameters (0-3)
}

// ============================================================================
// BAML FUNCTION
// ============================================================================

function RecommendBlackLittermanParameters(
  market_volatility: MarketVolatilityMetrics,
  signal_chars: SignalCharacteristics,
  macro_regime: MacroRegimeInput,
  historical_performance: HistoricalPerformance
) -> BlackLittermanParameters {
  client AzureOpenAIGpt5

  prompt #"
    You are a senior quantitative portfolio manager with 15+ years of experience in
    Black-Litterman portfolio optimization, risk management, and institutional asset allocation.

    ## Your Task
    Recommend optimal Black-Litterman model parameters based on current market conditions,
    signal characteristics, and macroeconomic regime. Your recommendations will be used in
    production portfolio optimization for real capital allocation.

    ## Market Volatility Context
    Market Realized Volatility: {{ market_volatility.market_realized_vol * 100 }}% (annualized)
    Signal Universe Volatility: {{ market_volatility.signal_universe_vol * 100 }}% (annualized)
    {% if market_volatility.current_vix %}VIX Level: {{ market_volatility.current_vix }}{% endif %}
    Volatility Regime: {{ market_volatility.vol_regime }}

    ## Signal Characteristics
    Total Signals: {{ signal_chars.total_signals }}
    Buy Signals: {{ signal_chars.buy_signals }}
    Strong Buy Signals: {{ signal_chars.strong_buy_signals }}
    Average Confidence: {{ signal_chars.avg_confidence }}
    Confidence Dispersion (Std): {{ signal_chars.confidence_dispersion }}
    Signal Concentration: {{ signal_chars.signal_concentration }}

    ## Macroeconomic Regime
    Current Regime: {{ macro_regime.current_regime }} (Confidence: {{ macro_regime.regime_confidence }})
    Recession Risk (6M): {{ macro_regime.recession_risk_6m }}
    Recession Risk (12M): {{ macro_regime.recession_risk_12m }}
    Market Stress Level: {{ macro_regime.market_stress_level }}

    {% if macro_regime.fred_indicators %}
    ### FRED Global Indicators
    | Indicator | Value | Signal |
    |-----------|-------|--------|
    | ISM PMI | {{ macro_regime.fred_indicators.ism_pmi|round(1) }} | {{ macro_regime.fred_indicators.ism_signal }} |
    | Yield Curve (2s10s) | {{ macro_regime.fred_indicators.yield_curve_2s10s|round(1) }} bps | {{ macro_regime.fred_indicators.curve_signal }} |
    | High Yield Spread | {{ macro_regime.fred_indicators.hy_spread|round(0) }} bps | {{ macro_regime.fred_indicators.credit_signal }} |
    | Investment Grade Spread | {{ macro_regime.fred_indicators.ig_spread|round(0) }} bps | {{ macro_regime.fred_indicators.credit_signal }} |
    {% if macro_regime.fred_indicators.vix %}| VIX | {{ macro_regime.fred_indicators.vix|round(1) }} | {{ macro_regime.fred_indicators.vix_signal }} |{% endif %}
    {% endif %}

    {% if macro_regime.usa_quantitative %}
    ### USA Economic Indicators (Real Data)
    | Indicator | Value |
    |-----------|-------|
    | GDP Growth (Q/Q) | {{ macro_regime.usa_quantitative.gdp_growth_qq|round(2) }}% |
    | Industrial Production | {{ macro_regime.usa_quantitative.industrial_production|round(2) }}% |
    | Unemployment Rate | {{ macro_regime.usa_quantitative.unemployment|round(1) }}% |
    | Consumer Price Inflation | {{ macro_regime.usa_quantitative.consumer_prices|round(2) }}% |
    | Budget Deficit (% GDP) | {{ macro_regime.usa_quantitative.deficit|round(2) }}% |
    | Government Debt (% GDP) | {{ macro_regime.usa_quantitative.debt|round(2) }}% |
    | Short-term Rate | {{ macro_regime.usa_quantitative.st_rate|round(2) }}% |
    | Long-term Rate | {{ macro_regime.usa_quantitative.lt_rate|round(2) }}% |
    {% endif %}

    {% if macro_regime.usa_forecast %}
    ### USA Economic Forecasts
    | Forecast Indicator | Value |
    |--------------------|-------|
    | Current Inflation | {{ macro_regime.usa_forecast.last_inflation|round(1) }}% |
    | Inflation Forecast (6M) | {{ macro_regime.usa_forecast.inflation_6m|round(1) }}% |
    | Inflation 10Y Average | {{ macro_regime.usa_forecast.inflation_10y_avg|round(1) }}% |
    | GDP Growth Forecast (6M) | {{ macro_regime.usa_forecast.gdp_growth_6m|round(1) }}% |
    | Earnings Growth (12M) | {{ macro_regime.usa_forecast.earnings_12m|round(1) }}% |
    | EPS Expected Growth (12M) | {{ macro_regime.usa_forecast.eps_expected_12m|round(1) }}% |
    | Market PEG Ratio | {{ macro_regime.usa_forecast.peg_ratio|round(1) }} |
    {% if macro_regime.usa_forecast.lt_rate_forecast %}| Long-term Rate Forecast | {{ macro_regime.usa_forecast.lt_rate_forecast|round(1) }}% |{% endif %}
    {% endif %}

    {% if macro_regime.regime_news_analysis %}
    ### News-Enhanced Regime Analysis
    {% if macro_regime.regime_news_analysis.regime_analysis_summary %}
    AI Analysis Summary: {{ macro_regime.regime_news_analysis.regime_analysis_summary }}
    {% endif %}
    {% if macro_regime.regime_news_analysis.news_summary %}
    News Summary: {{ macro_regime.regime_news_analysis.news_summary }}
    {% endif %}
    {% if macro_regime.regime_news_analysis.key_news_signals %}
    Key News Signals:
    {% for signal in macro_regime.regime_news_analysis.key_news_signals %}
    - {{ signal }}
    {% endfor %}
    {% endif %}
    {% if macro_regime.regime_news_analysis.news_alignment %}
    News Alignment: {{ macro_regime.regime_news_analysis.news_alignment }}
    {% endif %}
    {% if macro_regime.regime_news_analysis.warning_signs %}
    Warning Signs:
    {% for warning in macro_regime.regime_news_analysis.warning_signs %}
    - {{ warning }}
    {% endfor %}
    {% endif %}
    {% if macro_regime.regime_news_analysis.leading_indicators %}
    Leading Indicators:
    {% for indicator in macro_regime.regime_news_analysis.leading_indicators %}
    - {{ indicator }}
    {% endfor %}
    {% endif %}
    {% endif %}

    {% if macro_regime.g7_macro_overview %}
    ### G7 Macroeconomic Signals Overview

    **Global Macro Signals**
    | Signal | Status |
    |--------|--------|
    | Global Growth | {{ macro_regime.g7_macro_overview.global_growth_signal }} |
    | Global Inflation | {{ macro_regime.g7_macro_overview.global_inflation_signal }} |
    | Monetary Policy | {{ macro_regime.g7_macro_overview.monetary_policy_direction }} |
    {% if macro_regime.g7_macro_overview.synchronized_slowdown %}| Synchronized Slowdown | {{ macro_regime.g7_macro_overview.synchronized_slowdown }} |{% endif %}

    {% if macro_regime.g7_macro_overview.divergence_note %}
    **Key Divergences**: {{ macro_regime.g7_macro_overview.divergence_note }}
    {% endif %}

    **G7 Country Economic Indicators**
    | Country | GDP (Q/Q) | Unemployment | Inflation | Inflation 6M | GDP 6M | Policy Rate | Regime |
    |---------|-----------|--------------|-----------|--------------|--------|-------------|--------|
    {% for country in macro_regime.g7_macro_overview.countries -%}
    | {{ country.country }} | {% if country.gdp_growth_qq %}{{ country.gdp_growth_qq|round(2) }}%{% else %}—{% endif %} | {% if country.unemployment %}{{ country.unemployment|round(1) }}%{% else %}—{% endif %} | {% if country.inflation %}{{ country.inflation|round(2) }}%{% else %}—{% endif %} | {% if country.inflation_forecast_6m %}{{ country.inflation_forecast_6m|round(1) }}%{% else %}—{% endif %} | {% if country.gdp_forecast_6m %}{{ country.gdp_forecast_6m|round(1) }}%{% else %}—{% endif %} | {% if country.policy_rate %}{{ country.policy_rate|round(2) }}%{% else %}—{% endif %} | {% if country.regime_assessment %}{{ country.regime_assessment }}{% else %}—{% endif %} |
    {% endfor %}
    {% if macro_regime.g7_macro_overview.key_risks %}
    **Cross-Border Risks**:
    {% for risk in macro_regime.g7_macro_overview.key_risks %}
    - {{ risk }}
    {% endfor %}
    {% endif %}
    {% endif %}

    ## Historical Performance
    {% if historical_performance.recent_sharpe_ratio %}Recent Sharpe Ratio: {{ historical_performance.recent_sharpe_ratio }}{% endif %}
    {% if historical_performance.recent_turnover %}Recent Turnover: {{ historical_performance.recent_turnover }}{% endif %}
    {% if historical_performance.avg_tracking_error %}Avg Tracking Error: {{ historical_performance.avg_tracking_error }}{% endif %}

    {{ ctx.output_format }}

    ## Parameter Selection Framework

    ### 1. Tau (Prior Uncertainty Scalar)

    **Academic Baseline**: τ = 0.025 (2.5% uncertainty in equilibrium returns per He & Litterman 1999)
    **PyPortfolioOpt Default**: τ = 0.05 (5% uncertainty, more weight on views)

    **Adjustments**:
    - **High Volatility Regime** (vol >25%): Increase tau → 0.04-0.08
      - Rationale: Greater uncertainty in equilibrium, views should have more influence
    - **Low Volatility Regime** (vol <15%): Decrease tau → 0.015-0.025
      - Rationale: More confidence in equilibrium, views have less relative weight
    - **High Recession Risk** (>0.6): Increase tau → 0.03-0.06
      - Rationale: Regime uncertainty makes equilibrium less reliable
    - **Market Stress = Extreme**: Increase tau → 0.05-0.10
      - Rationale: Historical correlations break down, equilibrium unreliable

    **Recommended Range**: 0.01 to 0.10
    **Typical Values**: 0.015 (low vol, stable regime) to 0.08 (high vol, stressed markets)

    ### 2. Delta (Risk Aversion Coefficient)

    **Academic Baseline**: δ = 2.5 (moderate institutional risk aversion)

    **Adjustments**:
    - **Late Cycle / High Recession Risk**: Increase delta → 3.0-4.0
      - Rationale: More risk-averse stance, prefer safer allocations
    - **Early/Mid Cycle + Low Recession Risk**: Decrease delta → 1.5-2.5
      - Rationale: Risk-on environment, can tolerate higher volatility
    - **High Market Stress**: Increase delta → 3.5-4.5
      - Rationale: Flight to quality, risk aversion spikes
    - **Low Signal Confidence** (avg <0.6): Increase delta → 2.8-3.5
      - Rationale: Less conviction in views, prefer more conservative allocations

    **Recommended Range**: 1.5 to 4.5
    **Typical Values**: 1.8 (early cycle, low vol) to 3.8 (recession, high stress)

    ### 3. Expected Return Ranges (Annual)

    **Baseline (Normal Markets)**:
    - Strong Buy: 10-15% annual
    - Buy: 5-10% annual

    **Adjustments**:
    - **High Volatility / High Stress**: Lower expectations
      - Strong Buy: 8-12% (harder to achieve alpha in volatile markets)
      - Buy: 4-8%
    - **Low Volatility / Early Cycle**: Can be more optimistic
      - Strong Buy: 12-18%
      - Buy: 6-12%
    - **High Confidence Signals** (avg >0.75): Upper end of ranges
    - **Low Confidence Signals** (avg <0.6): Lower end of ranges
    - **Late Cycle / Recession**: Conservative
      - Strong Buy: 6-10% (focus on capital preservation)
      - Buy: 3-6%

    **Key Principle**: Expected returns should be achievable given market conditions.
    Over-optimistic views lead to concentrated, risky portfolios.

    ### 4. View Uncertainty Scaling

    **Baseline**: 1-5% standard deviation (annual)
    - High confidence (>0.8): 1-2% uncertainty
    - Medium confidence (0.6-0.8): 2-4% uncertainty
    - Low confidence (<0.6): 4-5% uncertainty

    **Adjustments**:
    - **High Signal Universe Volatility** (>30%): Increase max uncertainty → 3-8%
      - Rationale: Wider error bands on expected returns for volatile stocks
    - **Low Confidence Dispersion** (all signals similar confidence): Tighter range → 1-3%
      - Rationale: Less differentiation needed
    - **High Confidence Dispersion** (wide variation): Wider range → 1-7%
      - Rationale: Need to distinguish high vs low conviction views
    - **Market Stress = High/Extreme**: Increase max uncertainty → 4-10%
      - Rationale: Forecasting is much harder in stressed markets

    **confidence_scaling**: Multiplier applied to (1 - confidence) to get uncertainty std
    - Baseline: 0.10 (10% scaling)
    - High dispersion or high volatility: 0.15-0.20
    - Low dispersion or low volatility: 0.05-0.10

    ### 5. Weight Constraints

    **Min Position Weight**:
    - **Baseline**: 2% (max 50 positions)
    - **High Conviction / Few Signals** (<20): Can go to 3-4%
    - **Low Conviction / Many Signals** (>50): Stay at 1-2%
    - **High Turnover History**: Consider 2-3% to reduce churn

    **Max Position Weight**:
    - **Baseline**: 10% (concentration limit)
    - **High Volatility Regime**: Reduce to 7-8% (de-risk)
    - **Low Volatility + High Confidence**: Can go to 12-15%
    - **Late Cycle / Recession**: Reduce to 6-8% (diversification critical)
    - **Early Cycle / Low Vol**: Can go to 10-12%

    **Rationale**: Weight constraints manage concentration risk and ensure diversification

    ### 6. Regime-Specific Recommendations

    **Early Cycle** (Growth accelerating, ISM PMI >52, low recession risk, steepening yield curve):
    - tau: 0.020-0.030 (moderate prior uncertainty)
    - delta: 1.8-2.3 (lower risk aversion)
    - Returns: Optimistic (12-18% Strong Buy, 6-12% Buy)
    - Max weight: 10-12%
    - Rationale: Risk-on environment favoring cyclical sectors, can tolerate higher volatility

    **Mid Cycle** (Balanced growth, ISM PMI 50-52, moderate risk, normal yield curve):
    - tau: 0.020-0.030 (standard)
    - delta: 2.3-2.8 (balanced)
    - Returns: Moderate (10-15% Strong Buy, 5-10% Buy)
    - Max weight: 8-10%
    - Rationale: Stable environment, minimize concentrated sector bets, focus on stock selection

    **Late Cycle** (Slowing growth, ISM PMI 48-50, elevated risk, flattening yield curve, HY spreads widening):
    - tau: 0.025-0.040 (higher uncertainty)
    - delta: 2.8-3.5 (more risk-averse)
    - Returns: Conservative (8-12% Strong Buy, 4-8% Buy)
    - Max weight: 6-8%
    - Rationale: Defensive rotation warranted, reduce concentration given elevated macro risk

    **Recession** (Negative growth, ISM PMI <48, high risk, inverted yield curve, HY spreads >500 bps):
    - tau: 0.035-0.060 (high uncertainty)
    - delta: 3.5-4.5 (very risk-averse)
    - Returns: Defensive (6-10% Strong Buy, 3-6% Buy)
    - Max weight: 5-7%
    - Rationale: Maximum defensive positioning, capital preservation focus, tight diversification

    ### 7. Volatility Regime Adjustments

    **Low Vol** (<15%):
    - tau: -0.005 adjustment (more confidence in equilibrium)
    - Uncertainty: Lower end of ranges
    - Can relax max weight slightly

    **Normal Vol** (15-25%):
    - No adjustment to baseline

    **High Vol** (>25%):
    - tau: +0.010 to +0.030 adjustment (less confidence in equilibrium)
    - delta: +0.3 to +0.8 adjustment (more risk-averse)
    - Uncertainty: Upper end of ranges
    - Reduce max weight by 2-3%

    ### 8. Signal Confidence Adjustments

    **High Avg Confidence** (>0.75):
    - Can use more optimistic return expectations (upper end of ranges)
    - Tighter uncertainty bands (lower confidence_scaling)
    - Slightly higher max weights

    **Low Avg Confidence** (<0.6):
    - More conservative return expectations (lower end of ranges)
    - Wider uncertainty bands (higher confidence_scaling)
    - Lower max weights
    - Consider increasing delta (more risk-averse)

    **High Confidence Dispersion** (std >0.15):
    - Wider uncertainty range to differentiate high vs low conviction
    - Higher confidence_scaling (0.15-0.20)

    ### 9. Quality Checks

    - Ensure all parameters are within recommended ranges
    - Check that return expectations are realistic (not >25% annual for Strong Buy)
    - Verify uncertainty_max > uncertainty_min
    - Ensure max_weight > min_weight and both are sensible (2-15% range)
    - Rationales should be specific to current market conditions (not generic)
    - Cite specific metrics from inputs (e.g., "Given 32% realized vol...")
    - Parameter choices should be internally consistent (high delta should pair with conservative returns)

    ### 10. Risk Warnings

    Identify 0-3 key risks given the parameter recommendations:
    - Over-concentrated portfolios if max weight too high
    - Under-diversification if min weight too high
    - Excessive turnover if constraints too tight
    - Over-optimistic returns leading to aggressive allocations
    - Under-optimistic returns missing opportunities
    - Tau too high → views dominate, ignoring diversification
    - Tau too low → views ignored, portfolio too passive
    - Delta too high → overly conservative, missing upside
    - Delta too low → excessive risk-taking

    ### 11. Monitoring Recommendations

    Given the parameters chosen, recommend 0-3 things to monitor:
    - Portfolio concentration (if allowing high max weights)
    - Realized Sharpe ratio vs expectations (if optimistic returns)
    - Turnover (if tight constraints)
    - Drawdowns (if low delta/aggressive stance)
    - Tracking error vs benchmark (if high tau)
    - Ex-post return vs view expectations (calibration check)

    ## Important Guidelines

    - **Be quantitative**: Provide specific numerical recommendations with ranges
    - **Be adaptive**: Parameters should respond to market conditions, not be static
    - **Be balanced**: Use academic baselines as anchors, adjust from there
    - **Be conservative**: When uncertain, err on side of caution
    - **Be consistent**: Parameters should tell a coherent story (e.g., late cycle → higher delta, lower returns, tighter weights)
    - **Cite inputs**: Reference specific values from market_volatility, signal_chars, macro_regime
    - **Institutional quality**: These parameters will drive real capital allocation decisions

    ## Example Output Reasoning

    "Given realized volatility of 28% (high regime), late-cycle macro environment with
    65% 6-month recession risk, and average signal confidence of 0.68, I recommend:
    - tau = 0.045 (elevated uncertainty in equilibrium given volatility and regime shift risk)
    - delta = 3.2 (higher risk aversion appropriate for late cycle)
    - Strong Buy returns: 8-12% (conservative given macro headwinds)
    - Max weight: 7% (reduced concentration given elevated risk)
    This configuration balances incorporating LLM views while maintaining prudent risk controls
    appropriate for current market stress and regime uncertainty."
  "#
}
