enum BusinessCycleRegime {
  EARLY_CYCLE
  MID_CYCLE
  LATE_CYCLE
  RECESSION
  UNCERTAIN
}

enum ISMSignal {
  STRONG_EXPANSION // >52
  MILD_EXPANSION // 50-52
  MILD_CONTRACTION // 45-50
  DEEP_CONTRACTION // <45
}

enum YieldCurveSignal {
  STEEP // >100bps
  NORMAL // 50-100bps
  FLAT // 0-50bps
  INVERTED // <0bps
}

enum CreditSpreadSignal {
  TIGHT // <350bps
  NEUTRAL // 350-500bps
  WIDENING // 500-700bps
  STRESS // >700bps
}

enum FactorExposure {
  GROWTH_MOMENTUM
  QUALITY_DEFENSIVE
  BALANCED
}

class EconomicIndicators {
  // GDP Indicators (from Il Sole + Trading Economics)
  gdp_growth_qq float?  // Quarterly growth rate
  gdp_growth_yy float?  // Year-over-year growth rate
  gdp_forecast_6m float?  // 6-month forecast

  // Labor Market
  unemployment float?  // Unemployment rate (%)

  // Production & Capacity
  industrial_production float?  // Industrial production growth (%)
  capacity_utilization float?  // Capacity utilization (%)

  // Inflation & Prices
  inflation float?  // Headline inflation rate (%)
  inflation_mom float?  // Month-over-month inflation (%)
  core_inflation float?  // Core inflation rate (%)
  inflation_forecast float?  // Inflation forecast

  // Consumption & Retail
  retail_sales_mom float?  // Retail sales month-over-month growth (%)

  // Corporate & Earnings
  earnings_growth_forecast float?  // Earnings growth forecast (%)

  // Confidence Indicators
  business_confidence float?  // Business confidence index
  consumer_confidence float?  // Consumer confidence index

  // External & Trade
  trade_balance float?  // Trade balance (USD billions)
  current_account float?  // Current account balance (USD billions)
  current_account_gdp float?  // Current account as % of GDP

  // Fiscal Position
  government_debt_gdp float?  // Government debt as % of GDP
  budget_balance_gdp float?  // Budget balance as % of GDP
}

class MarketIndicators {
  // PMI Indicators (Manufacturing Activity)
  ism_pmi float?  // US ISM PMI (global proxy from FRED)
  manufacturing_pmi float?  // Country-specific manufacturing PMI
  services_pmi float?  // Services sector PMI
  composite_pmi float?  // Composite PMI (both manufacturing & services)

  // Interest Rates & Policy
  interest_rate float?  // Central bank policy rate (%)

  // Yield Curve & Bond Markets
  yield_curve_2s10s float?  // 10Y-2Y treasury spread (basis points)
  bond_yield_2y float?  // 2-year government bond yield (%)
  bond_yield_5y float?  // 5-year government bond yield (%)
  bond_yield_10y float?  // 10-year government bond yield (%)
  bond_yield_30y float?  // 30-year government bond yield (%)

  // Credit Markets & Risk
  hy_spread float?  // High-yield credit spread (basis points)

  // Volatility & Sentiment
  vix float?  // VIX volatility index
  vix_signal string?  // VIX regime classification
}

class BusinessCycleClassification {
  // Core Classification
  regime BusinessCycleRegime

  confidence float
  @description(#"
    0.0 to 1.0
  "#)

  rationale string
  @description(#"
    Concise explanation (2-3 sentences maximum)
  "#)

  // Key Decision Indicators (simplified signals)
  ism_signal ISMSignal
  yield_curve_signal YieldCurveSignal
  credit_spread_signal CreditSpreadSignal

  // Recession Risk Assessment
  recession_risk_6m float
  @description(#"
    0.0 to 1.0
  "#)

  recession_risk_12m float
  @description(#"
    0.0 to 1.0
  "#)

  recession_drivers string[]
  @description(#"
    2-4 key recession risk factors
  "#)

  // Portfolio Positioning (ACTIONABLE - directly usable)
  sector_tilts map<string, float>
  @description(#"
    Sector -> weight adjustment between -0.10 and +0.10
  "#)

  recommended_overweights string[]
  @description(#"
    3-5 sectors to overweight with brief rationale
  "#)

  recommended_underweights string[]
  @description(#"
    3-5 sectors to underweight with brief rationale
  "#)

  factor_exposure FactorExposure

  // Risk Monitoring
  primary_risks string[]
  @description(#"
    2-3 main risks to watch
  "#)

  conflicting_signals string[]
  @description(#"
    Any major contradictions (empty if none)
  "#)
}

function ClassifyBusinessCycleWithLLM(
  today: string,
  economic_indicators: EconomicIndicators,
  market_indicators: MarketIndicators,
  news_signals: MacroNewsSignals,
  country: string
) -> BusinessCycleClassification {
  client Model

  prompt #"
    {{ _.role("system") }}
    You are an expert macroeconomic strategist with 15+ years of experience in business cycle analysis and portfolio construction. Your specialty is synthesizing multi-source economic data into actionable investment insights.

    Your core competencies:
    - Pattern recognition across historical business cycles
    - Data-driven regime classification using quantitative frameworks
    - Risk assessment through leading and coincident indicator analysis
    - Sector rotation strategies aligned with macroeconomic phases

    Your analytical approach is systematic, evidence-based, and grounded in economic theory. You prioritize country-specific data over global proxies and always quantify confidence levels based on signal alignment.

    {{ _.role("user") }}
    # OBJECTIVE
    Classify the current business cycle regime for {{ country }} using a structured analytical framework. Provide high-confidence classification with actionable portfolio positioning recommendations.

    **Analysis Date:** {{ today }}

    ---

    # INPUT DATA HIERARCHY

    ## PRIMARY SOURCE: Country-Specific Economic Indicators ({{ country }})
    **GDP Growth**
    - Quarter-over-Quarter: {{ economic_indicators.gdp_growth_qq }}%
    - Year-over-Year: {{ economic_indicators.gdp_growth_yy }}%
    - 6-Month Forecast: {{ economic_indicators.gdp_forecast_6m }}%

    **Labor Market**
    - Unemployment Rate: {{ economic_indicators.unemployment }}%

    **Production & Capacity**
    - Industrial Production Growth: {{ economic_indicators.industrial_production }}%
    - Capacity Utilization: {{ economic_indicators.capacity_utilization }}%

    **Inflation & Prices**
    - Headline Inflation: {{ economic_indicators.inflation }}%
    - Inflation MoM: {{ economic_indicators.inflation_mom }}%
    - Core Inflation: {{ economic_indicators.core_inflation }}%
    - Inflation Forecast (6M): {{ economic_indicators.inflation_forecast }}%

    **Consumption & Retail**
    - Retail Sales MoM: {{ economic_indicators.retail_sales_mom }}%

    **Corporate Earnings**
    - Earnings Growth Forecast: {{ economic_indicators.earnings_growth_forecast }}%

    **Confidence Indicators**
    - Business Confidence: {{ economic_indicators.business_confidence }}
    - Consumer Confidence: {{ economic_indicators.consumer_confidence }}

    **External Balance & Trade**
    - Trade Balance: {{ economic_indicators.trade_balance }} USD billions
    - Current Account: {{ economic_indicators.current_account }} USD billions
    - Current Account (% GDP): {{ economic_indicators.current_account_gdp }}%

    **Fiscal Position**
    - Government Debt (% GDP): {{ economic_indicators.government_debt_gdp }}%
    - Budget Balance (% GDP): {{ economic_indicators.budget_balance_gdp }}%

    ---

    ## SECONDARY SOURCE: Market Indicators (Global Context & Confirmation)
    **PMI Indicators (Manufacturing & Services Activity)**
    - ISM PMI (US proxy): {{ market_indicators.ism_pmi }}
      └─ Classification: >52 (expansion) | 50-52 (neutral) | 45-50 (mild contraction) | <45 (deep contraction)
    - Manufacturing PMI ({{ country }}): {{ market_indicators.manufacturing_pmi }}
    - Services PMI ({{ country }}): {{ market_indicators.services_pmi }}
    - Composite PMI ({{ country }}): {{ market_indicators.composite_pmi }}

    **Interest Rates & Monetary Policy**
    - Central Bank Policy Rate: {{ market_indicators.interest_rate }}%

    **Yield Curve & Government Bonds**
    - Yield Curve Spread (10Y-2Y): {{ market_indicators.yield_curve_2s10s }} basis points
      └─ Classification: >100bps (steep/accommodative) | 50-100bps (normal) | 0-50bps (flat/caution) | <0bps (inverted/recession warning)
    - 2-Year Bond Yield: {{ market_indicators.bond_yield_2y }}%
    - 5-Year Bond Yield: {{ market_indicators.bond_yield_5y }}%
    - 10-Year Bond Yield: {{ market_indicators.bond_yield_10y }}%
    - 30-Year Bond Yield: {{ market_indicators.bond_yield_30y }}%

    **Credit Markets & Risk**
    - High-Yield Credit Spread: {{ market_indicators.hy_spread }} basis points
      └─ Classification: <350bps (tight/optimistic) | 350-500bps (neutral) | 500-700bps (widening/stress) | >700bps (severe stress)

    **Volatility & Market Sentiment**
    - VIX Level: {{ market_indicators.vix }} ({{ market_indicators.vix_signal }})

    ## TERTIARY SOURCE: News Sentiment & Narrative Analysis
    **Thematic Trends**: {{ news_signals.dominant_themes | join(', ') }}

    **Economic Sentiment Breakdown**
    - Growth Outlook: {{ news_signals.growth_sentiment }} - {{ news_signals.growth_summary }}
    - Inflation Expectations: {{ news_signals.inflation_sentiment }} - {{ news_signals.inflation_summary }}
    - Labor Market Sentiment: {{ news_signals.labor_market_sentiment }} - {{ news_signals.labor_market_summary }}

    **Policy & Risk**
    - Central Bank Policy Stance: {{ news_signals.policy_stance }} - {{ news_signals.policy_summary }}
    - Market Risk Appetite: {{ news_signals.risk_appetite }}

    **Regime Narrative**: {{ news_signals.regime_narrative }} - {{ news_signals.regime_rationale }}

    ---

    # BUSINESS CYCLE CLASSIFICATION FRAMEWORK

    ## Regime Definitions & Characteristics

    ### EARLY_CYCLE
    **Economic Characteristics:**
    - GDP accelerating from trough (QQ growth turning positive and rising)
    - Unemployment rate declining from peak levels
    - Industrial production rebounding strongly (>3% YY)
    - ISM PMI expanding (>52)
    - Earnings growth expectations improving sharply

    **Market Characteristics:**
    - Yield curve steep (>100bps) - accommodative monetary policy
    - Credit spreads tightening from stressed levels (<350bps)
    - VIX declining from elevated levels
    - Risk appetite increasing

    **Typical Duration:** 6-12 months post-recession trough

    ### MID_CYCLE
    **Economic Characteristics:**
    - GDP growth steady and moderate (2.0-3.5% annualized)
    - Unemployment rate stable in healthy range (4.0-5.5%)
    - Moderate inflation (1.5-3.0%)
    - ISM PMI in expansion zone but not overheated (50-55)
    - Balanced economic conditions with no major imbalances

    **Market Characteristics:**
    - Yield curve normal slope (50-100bps)
    - Credit spreads neutral (350-500bps)
    - Volatility subdued (VIX 12-20)
    - Steady risk appetite

    **Typical Duration:** 18-36 months - longest phase of cycle

    ### LATE_CYCLE
    **Economic Characteristics:**
    - GDP growth decelerating (slowing QQ growth rates)
    - Unemployment at cycle lows (<4.0%) - tight labor market
    - Inflation pressures building or elevated (>3.0%)
    - ISM PMI falling toward 50 or below
    - Tightening financial conditions
    - Earnings growth forecasts declining

    **Market Characteristics:**
    - Yield curve flattening or inverted (<50bps or negative)
    - Credit spreads beginning to widen (>500bps)
    - Volatility rising (VIX >20)
    - Risk appetite deteriorating
    - Central bank in tightening mode or pausing after hikes

    **Typical Duration:** 6-18 months before recession

    ### RECESSION
    **Economic Characteristics:**
    - GDP growth negative (two consecutive quarters typical)
    - Unemployment rising (typically >6.5% and increasing)
    - Industrial production contracting (negative YY growth)
    - ISM PMI in deep contraction (<45)
    - Earnings declining broadly

    **Market Characteristics:**
    - Credit spreads wide and volatile (>500bps, often >700bps)
    - VIX elevated (>30)
    - Significant market stress and risk-off sentiment
    - Central bank easing aggressively

    **Typical Duration:** 6-18 months

    ### UNCERTAIN
    **When to Use:**
    - Conflicting signals across data sources (e.g., strong GDP but contracting PMI)
    - Transition periods between regimes with ambiguous data
    - Data quality issues or missing critical indicators
    - Confidence score would be <0.60 for any single regime
    - Major structural changes making historical patterns unreliable

    ---

    # ANALYTICAL REASONING FRAMEWORK

    Use this systematic chain-of-thought process. **Document your reasoning explicitly for each step.**

    ## STEP 1: DATA QUALITY & SIGNAL EXTRACTION

    **Task:** Extract and classify all available signals according to the framework.

    ### 1A. Classify Market Indicators
    - ISM PMI {{ market_indicators.ism_pmi }} → Classify as: STRONG_EXPANSION (>52) | MILD_EXPANSION (50-52) | MILD_CONTRACTION (45-50) | DEEP_CONTRACTION (<45)
    - Yield Curve {{ market_indicators.yield_curve_2s10s }}bps → Classify as: STEEP (>100) | NORMAL (50-100) | FLAT (0-50) | INVERTED (<0)
    - Credit Spread {{ market_indicators.hy_spread }}bps → Classify as: TIGHT (<350) | NEUTRAL (350-500) | WIDENING (500-700) | STRESS (>700)

    ### 1B. Assess Data Availability & Quality
    - Which indicators are missing or null? List any gaps.
    - Are there data inconsistencies or outliers that need special consideration?

    ### 1C. Extract Key Metrics from Country-Specific Data
    From the Economic and Market Indicators, identify:
    - Most recent PMI readings (Manufacturing, Services, Composite)
    - Business/Consumer Confidence trends (rising, falling, stable)
    - Bond yield curve shape (use actual {{ country }} government bonds: 2Y, 5Y, 10Y, 30Y)
    - Retail sales growth trajectory
    - Employment/unemployment trends
    - Industrial production & capacity utilization
    - Trade balance and fiscal position

    **Output for Step 1:** List the classified signals and note data quality.

    ---

    ## STEP 2: REGIME PATTERN MATCHING

    **Task:** Compare observed data patterns against each regime definition. Use weighted scoring.

    ### 2A. Scoring Weights by Data Source Priority
    Apply these weights when evaluating regime fit:
    - **Country-specific economic indicators (GDP, unemployment, inflation, trade, fiscal):** 50% weight
    - **Country-specific market indicators (PMI, bond yields, confidence):** 30% weight
    - **Global market indicators (ISM, HY spreads, VIX):** 15% weight
    - **News sentiment:** 5% weight

    ### 2B. Pattern Matching Analysis
    For each regime (EARLY_CYCLE, MID_CYCLE, LATE_CYCLE, RECESSION), evaluate:

    **EARLY_CYCLE Pattern Match:**
    - Is GDP growth accelerating from low levels? (Check QQ vs YY growth trends)
    - Is ISM >52 in expansion?
    - Is yield curve steep (>100bps)?
    - Are credit spreads tight (<350bps)?
    - Is unemployment declining?
    - Are earnings forecasts improving?
    - What is the overall pattern match score (0-100)?

    **MID_CYCLE Pattern Match:**
    - Is GDP growth stable in 2-3.5% range?
    - Is ISM in 50-55 range?
    - Is yield curve normal (50-100bps)?
    - Are credit spreads neutral (350-500bps)?
    - Is unemployment stable in healthy range (4-5.5%)?
    - Are conditions balanced without major imbalances?
    - What is the overall pattern match score (0-100)?

    **LATE_CYCLE Pattern Match:**
    - Is GDP growth decelerating (compare current to 6M forecast)?
    - Is ISM falling toward/below 50?
    - Is yield curve flat/inverted (<50bps)?
    - Are credit spreads widening (>500bps)?
    - Is unemployment at cycle lows (<4%)?
    - Is inflation elevated (>3%)?
    - Are financial conditions tightening?
    - What is the overall pattern match score (0-100)?

    **RECESSION Pattern Match:**
    - Is GDP growth negative?
    - Is ISM <45 in deep contraction?
    - Is unemployment >6.5% and rising?
    - Are credit spreads in stress territory (>500bps)?
    - Are earnings declining?
    - Is VIX elevated (>30)?
    - What is the overall pattern match score (0-100)?

    ### 2C. Identify Conflicting Signals
    - List any indicators that contradict the highest-scoring regime
    - Assess whether contradictions are minor (lagging indicators) or major (leading indicators diverging)

    **Output for Step 2:** Rank regimes by pattern match score. Identify the leading candidate and note any significant conflicting signals.

    ---

    ## STEP 3: CONFIDENCE CALIBRATION

    **Task:** Quantify confidence level based on signal alignment and data quality.

    ### 3A. Calculate Base Confidence
    Use this calibration rubric:

    **High Confidence (0.85-1.0):**
    - All primary indicators (GDP, unemployment, industrial production, comprehensive PMI data) align with single regime
    - Market indicators (ISM, yield curve, credit spreads) confirm the same regime
    - News sentiment supports the classification
    - No major conflicting signals
    - Regime is clearly established (not in transition)

    **Moderate-High Confidence (0.70-0.85):**
    - Most primary indicators align with single regime
    - 1-2 minor contradictions from lagging or less critical indicators
    - Market indicators mostly confirm
    - Regime is likely but may be in early transition phase

    **Mixed Confidence (0.60-0.70):**
    - Primary indicators show mixed signals
    - 2-3 significant contradictions between leading indicators
    - Market indicators diverge from economic indicators
    - Possible transition period between regimes
    - Consider UNCERTAIN regime classification

    **Low Confidence (<0.60):**
    - Major contradictions across data sources
    - Missing critical data points
    - Structural changes making historical patterns unreliable
    - **MUST classify as UNCERTAIN regime**

    ### 3B. Adjust for Data Quality
    - Reduce confidence by 0.05-0.15 if multiple critical indicators (GDP, unemployment, PMI) are null
    - Reduce confidence by 0.10 if major conflicting signals exist between leading indicators
    - Reduce confidence by 0.05 if country-specific data (PMI, confidence indices) is missing

    ### 3C. Adjust for Regime Characteristics
    - Mid-cycle classifications: Can achieve confidence up to 0.95 (most stable regime)
    - Early-cycle classifications: Cap confidence at 0.90 (transition from recession is volatile)
    - Late-cycle classifications: Cap confidence at 0.85 (most uncertain/complex phase)
    - Recession classifications: Can achieve 0.95 if data clearly negative across board

    **Output for Step 3:** Final confidence score with justification.

    ---

    ## STEP 4: RECESSION RISK QUANTIFICATION

    **Task:** Calculate 6-month and 12-month recession probabilities using additive risk model.

    ### 4A. Start with Base Risk
    - 6-month baseline: 0.05 (5%)
    - 12-month baseline: 0.10 (10%)

    ### 4B. Add Risk Factors (cumulative, each factor adds risk)
    **Economic Deterioration (+0.10 to +0.20 each):**
    - GDP growth negative: +0.20 (6M), +0.15 (12M)
    - GDP growth <1%: +0.10 (6M), +0.08 (12M)
    - Unemployment >6.5%: +0.15 (6M), +0.10 (12M)
    - Industrial production negative YY: +0.12 (6M), +0.10 (12M)
    - ISM PMI <45 (deep contraction): +0.20 (6M), +0.15 (12M)
    - ISM PMI 45-50 (mild contraction): +0.10 (6M), +0.08 (12M)

    **Market Stress Indicators (+0.08 to +0.15 each):**
    - Yield curve inverted: +0.15 (6M), +0.20 (12M) - leading indicator
    - Credit spreads >700bps (stress): +0.15 (6M), +0.12 (12M)
    - Credit spreads 500-700bps (widening): +0.08 (6M), +0.08 (12M)
    - VIX >30: +0.10 (6M), +0.08 (12M)

    **Sentiment & Narrative (+0.05 to +0.10 each):**
    - News dominated by layoff/contraction themes: +0.08 (6M), +0.08 (12M)
    - Policy stance defensive/emergency easing: +0.10 (6M), +0.08 (12M)

    ### 4C. Subtract Risk Factors (each factor reduces risk)
    **Economic Strength (-0.05 to -0.15 each):**
    - GDP growth >3%: -0.10 (6M), -0.08 (12M)
    - ISM PMI >55: -0.10 (6M), -0.08 (12M)
    - Unemployment <4.5%: -0.05 (6M), -0.03 (12M)
    - Strong earnings growth forecast (>10%): -0.08 (6M), -0.05 (12M)

    **Market Optimism (-0.03 to -0.10 each):**
    - Credit spreads <300bps: -0.08 (6M), -0.05 (12M)
    - Yield curve >100bps (steep): -0.05 (6M), -0.03 (12M)
    - VIX <15: -0.03 (6M), -0.02 (12M)

    ### 4D. Final Calculation
    - Sum all applicable risk factors
    - Cap maximum at 0.95 (never assign 100% certainty)
    - Floor minimum at 0.01 (always maintain non-zero tail risk)
    - Ensure 12M risk ≥ 6M risk (longer horizon should not have lower risk)

    ### 4E. Identify Recession Drivers
    List the 2-4 most significant factors contributing to recession risk:
    - Be specific with data points (e.g., "ISM PMI at 44 signals manufacturing contraction")
    - Prioritize leading indicators over lagging
    - Include forward-looking concerns if confidence is moderate

    **Output for Step 4:** 6M and 12M recession risk scores with driver list.

    ---

    ## STEP 5: PORTFOLIO POSITIONING STRATEGY

    **Task:** Apply regime-specific sector tilts and generate actionable recommendations.

    ### 5A. Apply Exact Regime Tilts

    **EARLY_CYCLE Positioning:**
    ```
    Technology: +0.04
    Industrials: +0.04
    Consumer Discretionary: +0.05
    Financials: +0.03
    Real Estate: +0.03
    Materials: +0.03
    Utilities: -0.04
    Consumer Staples: -0.03
    Factor: GROWTH_MOMENTUM
    ```

    **MID_CYCLE Positioning:**
    ```
    Technology: +0.01
    Communication Services: +0.01
    (All other sectors: 0.00 or minimal tilts)
    Factor: BALANCED
    ```

    **LATE_CYCLE Positioning:**
    ```
    Energy: +0.05
    Healthcare: +0.02
    Consumer Staples: +0.03
    Utilities: +0.02
    Technology: -0.03
    Consumer Discretionary: -0.03
    Financials: -0.02
    Industrials: -0.02
    Factor: QUALITY_DEFENSIVE
    ```

    **RECESSION Positioning:**
    ```
    Consumer Staples: +0.06
    Healthcare: +0.05
    Utilities: +0.04
    Technology: -0.06
    Consumer Discretionary: -0.05
    Financials: -0.05
    Industrials: -0.04
    Real Estate: -0.04
    Factor: QUALITY_DEFENSIVE
    ```

    **UNCERTAIN Positioning:**
    - Use MID_CYCLE tilts as neutral baseline
    - Reduce all tilt magnitudes by 50% (e.g., +0.04 becomes +0.02)
    - Increase defensive quality bias slightly
    - Factor: BALANCED

    ### 5B. Generate Overweight Recommendations (3-5 sectors)
    For each recommended overweight:
    - Cite the regime logic (e.g., "Early cycle favors cyclical growth sectors")
    - Reference specific data supporting the tilt (e.g., "Accelerating GDP supports industrials")
    - Keep rationale concise (15-25 words)

    Format: "Sector: Rationale citing regime and data"
    Example: "Technology: Early cycle growth momentum favors innovation leaders; earnings growth forecast of 12% supports"

    ### 5C. Generate Underweight Recommendations (3-5 sectors)
    For each recommended underweight:
    - Cite the regime logic (e.g., "Late cycle risks elevate vulnerability of cyclicals")
    - Reference specific data supporting the tilt (e.g., "Decelerating GDP pressures discretionary spending")
    - Keep rationale concise (15-25 words)

    Format: "Sector: Rationale citing regime and data"
    Example: "Consumer Discretionary: Late cycle slowdown pressures discretionary spending; unemployment rising toward 5%"

    ### 5D. Assign Factor Exposure
    Based on regime classification, assign the appropriate factor strategy:
    - EARLY_CYCLE → GROWTH_MOMENTUM
    - MID_CYCLE → BALANCED
    - LATE_CYCLE → QUALITY_DEFENSIVE
    - RECESSION → QUALITY_DEFENSIVE
    - UNCERTAIN → BALANCED

    **Output for Step 5:** Complete sector_tilts map, overweight/underweight arrays, and factor exposure.

    ---

    ## STEP 6: RISK IDENTIFICATION & MONITORING

    **Task:** Identify forward-looking risks and document conflicting signals.

    ### 6A. Primary Risks (2-3 most important)
    Focus on:
    - Leading indicators that could signal regime transition
    - Macro vulnerabilities specific to {{ country }}
    - Policy risks (central bank, fiscal, geopolitical)
    - Market structure risks (liquidity, volatility regime change)

    Format: Clear, specific risk statements (10-20 words each)
    Example: "Yield curve inversion deepening suggests recession risk within 12-18 months"
    Example: "Central bank may overtighten if inflation remains elevated above 4%"

    ### 6B. Conflicting Signals
    Document any significant contradictions:
    - Leading vs lagging indicator divergence
    - Economic data vs market pricing discrepancies
    - Country-specific vs global indicator conflicts
    - Return empty array if no major contradictions exist

    Format: "Indicator A shows X, but Indicator B shows Y"
    Example: "GDP growth remains positive at 2.1%, but ISM PMI at 48 signals manufacturing contraction"

    **Output for Step 6:** Primary risks array and conflicting signals array.

    ---

    # OUTPUT REQUIREMENTS

    Provide your complete analysis following the structure defined in:
    {{ ctx.output_format }}

    # EXECUTION INSTRUCTION

    Now analyze the provided data for {{ country }} using the systematic 6-step framework. Document your reasoning explicitly for each step, then provide the final structured classification according to {{ ctx.output_format }}.

    Begin your analysis.
  "#
}
