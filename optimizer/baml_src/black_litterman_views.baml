// BAML function for generating Black-Litterman views from stock signals
// Converts stock analyzer signals into structured investment views with confidence levels

// ============================================================================
// INPUT TYPES
// ============================================================================

class StockSignalData {
  ticker string
  yfinance_ticker string
  sector string?
  industry string?

  // Signal classification
  signal_type string  // LARGE_GAIN, SMALL_GAIN, NEUTRAL, SMALL_DECLINE, LARGE_DECLINE
  confidence_level string  // HIGH, MEDIUM, LOW

  // Factor scores (normalized -1 to +1)
  valuation_score float?
  momentum_score float?
  quality_score float?
  growth_score float?
  technical_score float?

  // Performance metrics
  annualized_return float?
  volatility float?
  sharpe_ratio float?
  beta float?
  alpha float?

  // Risk factors
  volatility_level string?  // LOW, MEDIUM, HIGH
  beta_risk string?  // LOW, MEDIUM, HIGH
  debt_risk string?  // LOW, MEDIUM, HIGH, UNKNOWN
  liquidity_risk string?  // LOW, MEDIUM, HIGH

  // Projections
  upside_potential_pct float?
  downside_risk_pct float?

  // Price data
  close_price float?
  daily_return float?
}

class MacroRegimeContext {
  current_regime string  // EARLY_CYCLE, MID_CYCLE, LATE_CYCLE, RECESSION, UNCERTAIN
  regime_confidence float  // 0-1
  recession_risk_6m float?  // 0-1
  recession_risk_12m float?  // 0-1

  // Market indicators
  vix float?
  hy_spread float?  // High yield credit spread (bps)
  yield_curve_2s10s float?  // 2s10s spread (bps)

  // Sector recommendations from regime analysis
  recommended_overweights string[]?
  recommended_underweights string[]?
}

class SectorContext {
  sector_name string
  avg_sector_return float?  // Average return expectation for sector
  sector_momentum string?  // "strong", "moderate", "weak", "negative"
  sector_valuation string?  // "cheap", "fair", "expensive"
  num_signals_in_sector int?  // How many other signals in this sector
}

// ============================================================================
// OUTPUT TYPES
// ============================================================================

class BlackLittermanView {
  ticker string

  // View specification
  expected_return float  // Annual expected return (e.g., 0.12 = 12%)
  @description(#"
    Expected annual return for this asset. Range: -0.30 to +0.30 (-30% to +30%).
    Typical values: Strong Buy: 0.08-0.15, Buy: 0.04-0.10, Sell: -0.05 to -0.15
  "#)

  confidence float  // View confidence 0-1
  @description(#"
    Confidence level in this view (0 = no confidence, 1 = absolute certainty).
    HIGH: 0.75-0.95, MEDIUM: 0.50-0.75, LOW: 0.20-0.50
  "#)

  view_uncertainty float  // Standard deviation of expected return (e.g., 0.02 = 2%)
  @description(#"
    Uncertainty (standard deviation) of the expected return forecast.
    Derived from confidence: higher confidence → lower uncertainty.
    Typical range: 0.01 (1%) to 0.08 (8%)
  "#)

  // Supporting analysis
  rationale string
  @description(#"
    2-3 sentence explanation of why this expected return is justified.
    Reference specific factors: valuation, momentum, quality, growth.
    Mention regime considerations and risk adjustments.
  "#)

  factor_contributions string
  @description(#"
    Breakdown of which factors support the view (e.g., "Momentum +3%, Valuation +2%, Quality +1%")
  "#)

  risk_adjustments string
  @description(#"
    Any penalties applied due to risk factors (e.g., "High debt: -2%, Low liquidity: -1%")
  "#)

  regime_adjustment string
  @description(#"
    How the macro regime influenced the view (e.g., "Late cycle: reduced return by 2%")
  "#)
}

// ============================================================================
// BAML FUNCTION
// ============================================================================

function GenerateBlackLittermanView(
  signal: StockSignalData,
  macro_regime: MacroRegimeContext,
  sector_context: SectorContext,
  news_signals: StockNewsSignals?
) -> BlackLittermanView {
  client AzureOpenAIGpt5

  prompt #"
    You are a senior quantitative analyst generating Black-Litterman investment views
    for institutional portfolio optimization. Your view will be used in production capital
    allocation decisions and must be rigorous, well-calibrated, and explainable.

    ## Input Signal Data

    **Stock:** {{ signal.ticker }} ({{ signal.yfinance_ticker }})
    **Sector:** {{ signal.sector|default("Unknown") }} / {{ signal.industry|default("Unknown") }}
    **Signal Type:** {{ signal.signal_type }}
    **Confidence Level:** {{ signal.confidence_level }}
    **Close Price:** ${{ signal.close_price|default("N/A") }}

    **Factor Scores (normalized -1 to +1):**
    | Factor | Score |
    |--------|-------|
    | Valuation | {{ signal.valuation_score|default("N/A") }} |
    | Momentum | {{ signal.momentum_score|default("N/A") }} |
    | Quality | {{ signal.quality_score|default("N/A") }} |
    | Growth | {{ signal.growth_score|default("N/A") }} |
    | Technical | {{ signal.technical_score|default("N/A") }} |

    **Performance Metrics:**
    | Metric | Value |
    |--------|-------|
    | Annualized Return | {{ (signal.annualized_return * 100)|round(1)|default("N/A") }}% |
    | Volatility | {{ (signal.volatility * 100)|round(1)|default("N/A") }}% |
    | Sharpe Ratio | {{ signal.sharpe_ratio|round(2)|default("N/A") }} |
    | Beta | {{ signal.beta|round(2)|default("N/A") }} |
    | Alpha | {{ (signal.alpha * 100)|round(1)|default("N/A") }}% |

    **Risk Factors:**
    - Volatility Risk: {{ signal.volatility_level|default("UNKNOWN") }}
    - Beta Risk: {{ signal.beta_risk|default("UNKNOWN") }}
    - Debt Risk: {{ signal.debt_risk|default("UNKNOWN") }}
    - Liquidity Risk: {{ signal.liquidity_risk|default("UNKNOWN") }}

    **Projections:**
    - Upside Potential: {{ signal.upside_potential_pct|round(1)|default("N/A") }}%
    - Downside Risk: {{ signal.downside_risk_pct|round(1)|default("N/A") }}%

    ## Macro Regime Context

    **Current Regime:** {{ macro_regime.current_regime }} (Confidence: {{ (macro_regime.regime_confidence * 100)|round(0) }}%)
    {% if macro_regime.recession_risk_6m %}**Recession Risk (6M):** {{ (macro_regime.recession_risk_6m * 100)|round(0) }}%{% endif %}
    {% if macro_regime.recession_risk_12m %}**Recession Risk (12M):** {{ (macro_regime.recession_risk_12m * 100)|round(0) }}%{% endif %}

    **Market Indicators:**
    {% if macro_regime.vix %}- VIX: {{ macro_regime.vix|round(1) }}{% endif %}
    {% if macro_regime.hy_spread %}- HY Spread: {{ macro_regime.hy_spread|round(0) }} bps{% endif %}
    {% if macro_regime.yield_curve_2s10s %}- Yield Curve (2s10s): {{ macro_regime.yield_curve_2s10s|round(0) }} bps{% endif %}

    {% if macro_regime.recommended_overweights %}
    **Regime Recommends Overweight:** {{ macro_regime.recommended_overweights|join(", ") }}
    {% endif %}
    {% if macro_regime.recommended_underweights %}
    **Regime Recommends Underweight:** {{ macro_regime.recommended_underweights|join(", ") }}
    {% endif %}

    ## Sector Context

    **Sector:** {{ sector_context.sector_name }}
    {% if sector_context.avg_sector_return %}- Average Sector Return Expectation: {{ (sector_context.avg_sector_return * 100)|round(1) }}%{% endif %}
    {% if sector_context.sector_momentum %}- Sector Momentum: {{ sector_context.sector_momentum }}{% endif %}
    {% if sector_context.sector_valuation %}- Sector Valuation: {{ sector_context.sector_valuation }}{% endif %}
    {% if sector_context.num_signals_in_sector %}- Other Signals in Sector: {{ sector_context.num_signals_in_sector }}{% endif %}

    {% if news_signals %}
    ## Recent News Analysis

    **Dominant Themes:** {{ news_signals.dominant_themes|join(", ") }}
    **Overall News Bias:** {{ news_signals.news_bias }} (Confidence: {{ news_signals.confidence_assessment }})

    **Company Momentum:** {{ news_signals.company_momentum }}
    {{ news_signals.momentum_summary }}

    **Earnings Outlook:** {{ news_signals.earnings_outlook }}
    {{ news_signals.earnings_summary }}

    **Management Sentiment:** {{ news_signals.management_sentiment }}
    {{ news_signals.management_summary }}

    **Competitive Position:** {{ news_signals.competitive_position }}
    {{ news_signals.competitive_summary }}

    {% if news_signals.key_catalysts %}
    **Key Catalysts:** {{ news_signals.key_catalysts|join(", ") }}
    {% endif %}

    {% if news_signals.key_risks %}
    **Key Risks:** {{ news_signals.key_risks|join(", ") }}
    {% endif %}

    **IMPORTANT:** Integrate this news analysis into your view generation. Recent news provides forward-looking context that may confirm, challenge, or refine the quantitative signal. Consider:
    - Does news support the signal direction and magnitude?
    - Are there specific catalysts that could drive outperformance?
    - Are there risks that should temper the expected return or reduce confidence?
    - How does management sentiment align with the quantitative factors?

    {% else %}
    ## Recent News Analysis

    No recent news analysis available. Generate view based solely on quantitative signal, macro regime, and sector context.

    {% endif %}

    {{ ctx.output_format }}

    ## View Generation Framework

    Your task is to generate a Black-Litterman view specifying an **expected return** and
    **confidence level** for this stock over the next 12 months. Follow this systematic process:

    ### Step 1: Baseline Return from Signal Type

    Start with baseline return expectations based on signal type and confidence:

    **LARGE_GAIN (HIGH confidence):**
    - Baseline: 12-18% annual
    - Use upper end if factors strongly aligned

    **LARGE_GAIN (MEDIUM confidence):**
    - Baseline: 10-15% annual

    **LARGE_GAIN (LOW confidence):**
    - Baseline: 8-12% annual

    **SMALL_GAIN (HIGH confidence):**
    - Baseline: 8-12% annual

    **SMALL_GAIN (MEDIUM confidence):**
    - Baseline: 5-10% annual

    **SMALL_GAIN (LOW confidence):**
    - Baseline: 3-8% annual

    **NEUTRAL / DECLINE signals:**
    - These should not generate bullish views
    - Expected return: -5% to +3%

    ### Step 2: Regime-Adaptive Factor Weighting

    Adjust the baseline based on regime-specific factor importance:

    **EARLY_CYCLE:**
    - Weight Momentum: 35% (strongest)
    - Weight Growth: 25%
    - Weight Quality: 20%
    - Weight Valuation: 20%
    - Rationale: Cyclical upswing favors momentum and growth

    **MID_CYCLE:**
    - Weight all factors equally: 25% each
    - Rationale: Balanced environment, no strong tilts

    **LATE_CYCLE:**
    - Weight Quality: 30% (defensive)
    - Weight Valuation: 25%
    - Weight Momentum: 25%
    - Weight Growth: 20%
    - Rationale: Focus on quality and value as cycle matures

    **RECESSION:**
    - Weight Quality: 35% (most important)
    - Weight Valuation: 25%
    - Weight Momentum: 20%
    - Weight Growth: 20%
    - Rationale: Quality and valuation critical in downturns

    **UNCERTAIN:**
    - Weight Quality: 30%
    - Weight Valuation: 25%
    - Weight Momentum: 23%
    - Weight Growth: 22%
    - Rationale: Slight defensive tilt due to uncertainty

    **Calculation:**
    ```
    weighted_score = (valuation * w_val) + (momentum * w_mom) + (quality * w_qual) + (growth * w_growth)
    ```

    If weighted_score > 0.3: Add +2% to baseline
    If weighted_score > 0.5: Add +4% to baseline
    If weighted_score < -0.2: Subtract -2% from baseline
    If weighted_score < -0.4: Subtract -4% from baseline

    ### Step 3: Risk Penalty Adjustments

    Apply penalties for elevated risk factors:

    **High Debt Risk (debt_risk = HIGH):**
    - Penalty: -10% to -20% of expected return
    - Rationale: Financial leverage amplifies downside in stressed markets
    - Example: 12% baseline → 10% after -2% penalty

    **Low Liquidity (liquidity_risk = HIGH):**
    - Penalty: -5% to -15% of expected return
    - Rationale: Illiquid stocks harder to exit, wider spreads

    **High Volatility (volatility_level = HIGH):**
    - Penalty: -5% of expected return
    - Rationale: Higher risk requires margin of safety

    **High Beta Risk (beta_risk = HIGH):**
    - Penalty: -3% in LATE_CYCLE or RECESSION regimes only
    - Rationale: Amplified downside when markets turn

    **Missing Data (any factor score is null):**
    - Penalty: -10% of expected return
    - Reduce confidence by 0.10
    - Rationale: Less information = more uncertainty

    ### Step 4: Regime-Specific Adjustments

    Further adjust based on macro regime:

    **LATE_CYCLE (recession risk > 0.40):**
    - Reduce expected returns by 20-30%
    - Example: 12% → 8-10%
    - Rationale: Capital preservation becomes priority

    **RECESSION (recession risk > 0.60):**
    - Reduce expected returns by 30-40%
    - Example: 12% → 7-8%
    - Focus on quality, avoid cyclicals

    **EARLY_CYCLE (recession risk < 0.20):**
    - Can use upper end of baseline ranges
    - Rationale: Risk-on environment

    **High VIX (> 30):**
    - Reduce expected returns by 10-15%
    - Increase uncertainty
    - Rationale: Elevated market stress

    **Inverted Yield Curve (< -50 bps):**
    - Reduce expected returns by 5-10%
    - Rationale: Recession leading indicator

    ### Step 5: Sector Alignment Check

    Check if stock's sector aligns with regime recommendations:

    **Sector in recommended_overweights:**
    - Boost: +1% to +2%
    - Rationale: Regime favors this sector

    **Sector in recommended_underweights:**
    - Penalty: -2% to -3%
    - Rationale: Regime headwinds for sector

    **Sector valuation expensive + regime = LATE_CYCLE:**
    - Extra penalty: -2%

    ### Step 6: Confidence Calibration

    Map signal confidence + adjustments to view confidence (0-1):

    **HIGH confidence signal:**
    - Base confidence: 0.80-0.95
    - If all 4 factors aligned (all positive): 0.90-0.95
    - If 3/4 factors aligned: 0.80-0.85
    - If risk penalties applied: -0.05 to -0.10

    **MEDIUM confidence signal:**
    - Base confidence: 0.60-0.75
    - If strong factor alignment: 0.70-0.75
    - If weak factor alignment: 0.60-0.65
    - If risk penalties applied: -0.05 to -0.10

    **LOW confidence signal:**
    - Base confidence: 0.40-0.60
    - If some positive factors: 0.50-0.60
    - If weak factors: 0.40-0.50
    - If risk penalties applied: -0.05

    **Missing data penalty:**
    - Reduce confidence by 0.10 for each missing major factor score

    **High regime uncertainty:**
    - If regime = UNCERTAIN: -0.05 to -0.10 confidence

    ### Step 7: View Uncertainty Calculation

    Convert confidence to view uncertainty (standard deviation):

    ```
    view_uncertainty = base_uncertainty * (1 - confidence)
    ```

    Where base_uncertainty depends on volatility regime:
    - Low volatility (< 20%): base = 0.10 (10%)
    - Normal volatility (20-30%): base = 0.15 (15%)
    - High volatility (> 30%): base = 0.20 (20%)

    **Examples:**
    - Confidence 0.80, normal vol: uncertainty = 0.15 * (1 - 0.80) = 0.03 (3%)
    - Confidence 0.60, high vol: uncertainty = 0.20 * (1 - 0.60) = 0.08 (8%)
    - Confidence 0.90, low vol: uncertainty = 0.10 * (1 - 0.90) = 0.01 (1%)

    **Bounds:** Ensure 0.01 ≤ view_uncertainty ≤ 0.10

    ### Step 8: Quality Checks

    Before finalizing, verify:

    - [ ] Expected return is reasonable: -30% to +30% annual
    - [ ] Typical strong buy: 8-18% depending on regime
    - [ ] Confidence is calibrated: HIGH signal → 0.75-0.95, MEDIUM → 0.50-0.75, LOW → 0.20-0.60
    - [ ] View uncertainty makes sense: high confidence → low uncertainty (1-3%)
    - [ ] Rationale mentions specific factors and regime considerations
    - [ ] Risk adjustments are documented
    - [ ] Regime alignment is checked
    - [ ] If recession risk > 0.5, expected returns should be conservative (< 10%)

    ### Step 9: Generate Structured Output

    Provide:
    - **expected_return**: Final calculated annual return (as decimal, e.g., 0.12)
    - **confidence**: Final confidence level (0-1)
    - **view_uncertainty**: Standard deviation of return forecast (as decimal, e.g., 0.03)
    - **rationale**: 2-3 sentences explaining the view
    - **factor_contributions**: Breakdown of factor impacts (e.g., "Momentum +3%, Quality +2%")
    - **risk_adjustments**: List of penalties applied (e.g., "High debt: -2%, High vol: -1%")
    - **regime_adjustment**: How regime affected the view (e.g., "Late cycle: reduced return by 3%")

    ## Important Guidelines

    - **Be conservative:** When in doubt, lower expected returns and reduce confidence
    - **Be quantitative:** Provide specific numerical adjustments, not vague statements
    - **Be consistent:** Similar signals should produce similar views
    - **Be explainable:** Rationale should be clear and reference specific inputs
    - **Respect regime:** Late cycle / recession should produce more conservative views
    - **Mind the risks:** Don't ignore debt, liquidity, or volatility warnings
    - **Calibrate confidence:** Don't over-confidence views; market is uncertain
    - **Use upside_potential_pct as a sanity check:** If it says +5%, don't forecast +20%

    ## Example Reasoning Process

    "Given LARGE_GAIN signal with HIGH confidence, I start with 14% baseline. The stock has
    strong momentum (0.73) and quality (0.61) scores, adding +3%. However, it has HIGH debt
    risk, applying -2% penalty. The late-cycle regime with 52% recession risk warrants
    reducing the return by 25%, bringing it to 11.25%. The sector (Technology) is in the
    regime's recommended overweights, adding +1%. Final view: 12.3% expected return with
    0.78 confidence (reduced from 0.85 due to debt risk). View uncertainty of 2.2% reflects
    the high confidence tempered by late-cycle macro uncertainty."
  "#
}
