class NewsArticle {
  title string
  publisher string
  date string
  link string?
  summary string?  // Article summary from yfinance API (short snippet)
  full_content string?  // Full article content fetched from URL (preferred for analysis)
}

enum GrowthSentiment {
  ACCELERATING
  STABLE
  SLOWING
  CONTRACTING
}

enum InflationSentiment {
  RISING
  STABLE
  FALLING
  ELEVATED
}

enum LaborMarketSentiment {
  STRONG
  HEALTHY
  SOFTENING
  WEAK
}

enum PolicyStance {
  DOVISH
  NEUTRAL
  HAWKISH
  TIGHTENING
  EASING
}

enum RiskAppetite {
  RISK_ON
  NEUTRAL
  CAUTIOUS
  RISK_OFF
  FEAR
}

enum RegimeNarrative {
  EARLY_CYCLE
  MID_CYCLE
  LATE_CYCLE
  RECESSION
  UNCERTAIN
}

class MacroNewsSignals {
  // Dominant themes across all articles
  dominant_themes string[]
  @description(#"
    Top 3-5 themes (e.g., "Fed hawkish tone", "recession fears rising")
  "#)

  // Growth outlook
  growth_sentiment GrowthSentiment
  growth_summary string
  @description(#"
    2-3 sentences on GDP, earnings, business activity trends
  "#)

  // Inflation outlook
  inflation_sentiment InflationSentiment
  inflation_summary string
  @description(#"
    2-3 sentences on inflation trends and central bank response
  "#)

  // Labor market
  labor_market_sentiment LaborMarketSentiment
  labor_market_summary string
  @description(#"
    2-3 sentences on unemployment, hiring, layoffs
  "#)

  // Central bank policy
  policy_stance PolicyStance
  policy_summary string
  @description(#"
    2-3 sentences on monetary policy tone and expectations
  "#)

  // Market conditions
  risk_appetite RiskAppetite

  // Regime indicators
  regime_narrative RegimeNarrative
  regime_rationale string
  @description(#"
    2-3 sentences explaining why news suggests this regime
  "#)
}

function SummarizeNewsArticles(
  news: NewsArticle[],
  country: string
) -> MacroNewsSignals {
  client AzureOpenAIGpt5

  prompt #"
    {{ _.role("system") }}
    You are a senior macroeconomic analyst specializing in extracting actionable signals from financial news for business cycle analysis. Your outputs feed directly into a business cycle classification model that determines portfolio positioning.

    Core competencies:
    - Synthesize multiple news sources into coherent macroeconomic narratives
    - Classify sentiment consistently using predefined categories
    - Extract forward-looking signals that predict cycle transitions
    - Maintain objectivity by balancing bullish and bearish signals

    {{ _.role("user") }}
    # OBJECTIVE
    Analyze {{ news | length }} news articles for {{ country }} and extract structured macroeconomic signals that will be used by a downstream business cycle classifier.

    # INPUT DATA: NEWS ARTICLES ({{ country }})

    {% for article in news %}
    ## Article {{ loop.index }} | {{ article.date }}
    **Title:** {{ article.title }}
    **Source:** {{ article.publisher }}
    {% if article.full_content %}
    **Full Content:**
    {{ article.full_content }}
    {% elif article.summary %}
    **Summary:**
    {{ article.summary }}
    {% endif %}

    {% endfor %}

    # ANALYSIS FRAMEWORK

    ## Step 1: Identify Cross-Article Patterns
    Before extracting signals, scan all {{ news | length }} articles to identify:
    - Recurring themes mentioned in 2+ articles
    - Consensus vs. divergent viewpoints
    - Forward-looking statements vs. historical commentary
    - Data points with specific numbers (GDP, inflation, unemployment, earnings)

    ## Step 2: Extract Structured Signals

    ### 2.1 Dominant Themes (3-5 themes)
    Extract the most frequently discussed macroeconomic topics across ALL articles.
    - Format: Concise phrases (4-6 words each)
    - Examples: "Fed hawkish tone persists", "Soft landing narrative strengthening", "Inflation persistence concerns", "Labor market cooling signals"
    - Prioritize themes that appear in multiple articles

    ### 2.2 Growth Outlook
    **Sentiment Classification:**
    - ACCELERATING: GDP growth accelerating, PMI rising, earnings upgrades, expansion momentum
    - STABLE: Steady growth, moderate GDP (2-3%), flat PMI, stable earnings
    - SLOWING: Decelerating growth, PMI declining but >50, earnings downgrades, weakening activity
    - CONTRACTING: Negative growth, PMI <50, recession signals, severe slowdown

    **Summary Requirements:**
    - 2-3 sentences maximum
    - Include specific data points (e.g., "GDP forecast at 2.1%", "PMI fell to 48.5")
    - Focus on: GDP trends, business activity (PMI), corporate guidance, capex plans
    - Synthesize across articles - avoid redundancy

    ### 2.3 Inflation Outlook
    **Sentiment Classification:**
    - RISING: Inflation accelerating, upside surprises, wage pressure building
    - STABLE: Inflation steady at current level, contained within target range
    - FALLING: Disinflation trend, prices declining, deflation risks
    - ELEVATED: Inflation high but not necessarily rising (e.g., sticky at 4-5%)

    **Summary Requirements:**
    - 2-3 sentences maximum
    - Include specific data (CPI, PCE, wage growth percentages)
    - Focus on: Price pressures, central bank inflation targets, inflation expectations
    - Note divergence between headline and core if relevant

    ### 2.4 Labor Market
    **Sentiment Classification:**
    - STRONG: Unemployment <4%, robust hiring, tight labor market, wage growth >4%
    - HEALTHY: Balanced market, unemployment 4-5%, moderate hiring, stable wages
    - SOFTENING: Unemployment rising, hiring slowdown, layoff announcements, quits rate declining
    - WEAK: High unemployment >6%, mass layoffs, severe hiring freeze

    **Summary Requirements:**
    - 2-3 sentences maximum
    - Include specific metrics (unemployment rate, job openings, layoff counts)
    - Focus on: Employment trends, hiring/firing activity, wage dynamics
    - Weight company announcements by credibility (major corporations > startups)

    ### 2.5 Central Bank Policy
    **Stance Classification:**
    - DOVISH: Signaling rate cuts, accommodative language, prioritizing growth
    - NEUTRAL: Balanced approach, data-dependent, no clear directional bias
    - HAWKISH: Prioritizing inflation control, signaling higher-for-longer rates
    - TIGHTENING: Actively raising rates or reducing balance sheet
    - EASING: Actively cutting rates or expanding balance sheet

    **Summary Requirements:**
    - 2-3 sentences maximum
    - Prioritize official central bank communications over analyst speculation
    - Include rate expectations (e.g., "Market pricing 2 cuts by year-end")
    - Note any shifts in forward guidance

    ### 2.6 Risk Appetite
    **Classification:**
    - RISK_ON: Optimistic tone, rally themes, risk assets favored, bullish sentiment
    - NEUTRAL: Balanced sentiment, cautious optimism, wait-and-see approach
    - CAUTIOUS: Defensive positioning emerging, uncertainty rising, volatility concerns
    - RISK_OFF: Flight to safety, sell-off themes, bearish tone
    - FEAR: Panic signals, crisis language, extreme risk aversion

    **Assessment Criteria:**
    - Aggregate sentiment across all articles (not just market commentary)
    - Consider: Equity market tone, credit market conditions, investor positioning
    - Weight recent developments more heavily than historical context

    ### 2.7 Regime Narrative
    **Classification Criteria:**
    - EARLY_CYCLE: Recovery/rebound themes, stimulus effectiveness, confidence restoration, bottoming signals
    - MID_CYCLE: Goldilocks/balanced growth, contained inflation, stable conditions, low volatility
    - LATE_CYCLE: Peak growth warnings, overheating concerns, policy tightening, slowdown anticipation
    - RECESSION: Contraction language, layoffs, earnings collapses, credit stress, downturn confirmation
    - UNCERTAIN: Mixed/conflicting signals, transition period, unclear trajectory, divergent views

    **Rationale Requirements:**
    - 2-3 sentences explaining classification
    - Reference specific supporting evidence from articles
    - Acknowledge conflicting signals if present

    ## Step 3: Quality Control Checklist
    Before finalizing output, verify:
    - [ ] All summaries are 2-3 sentences (not bullets, not paragraphs)
    - [ ] Sentiment classifications match the defined criteria above
    - [ ] Specific numbers included where available (GDP %, inflation %, unemployment %)
    - [ ] Analysis reflects balance of articles (not cherry-picked)
    - [ ] Dominant themes appear in multiple articles (not one-off stories)
    - [ ] Forward-looking information prioritized over historical data
    - [ ] Credible sources weighted higher (central banks > analysts > pundits)

    # CRITICAL CONSTRAINTS

    1. **Synthesis Over Summarization**: Do NOT list individual article takeaways. Extract the aggregate signal.
    2. **Conciseness**: Each summary field must be exactly 2-3 sentences. No bullet points, no lists within summaries.
    3. **Consistency**: Use the exact sentiment/classification options provided. Do not invent new categories.
    4. **Data-Driven**: Include specific numerical data points whenever available (percentages, basis points, counts).
    5. **Objectivity**: Reflect the actual balance of sentiment across articles. Avoid confirmation bias.
    6. **Downstream Context**: This output feeds a business cycle classifier that needs clean, consistent, actionable signals.

    # OUTPUT FORMAT

    {{ ctx.output_format }}

    ---

    Now analyze all {{ news | length }} articles for {{ country }} following the framework above. Think step-by-step through the analysis framework before generating the structured output.
  "#
}