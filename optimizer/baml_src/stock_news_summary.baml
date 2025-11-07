// Stock News Summarization for Black-Litterman View Generation
// Similar to macro news summarization but focused on company-specific signals

// NewsArticle class is already defined in summarize_news.baml
// We'll import and reuse it

enum CompanyMomentum {
  ACCELERATING
  POSITIVE
  STABLE
  SLOWING
  NEGATIVE
}

enum EarningsOutlook {
  BEATS_EXPECTED
  MEETS_EXPECTED
  BELOW_EXPECTED
  UNCERTAIN
}

enum ManagementSentiment {
  VERY_BULLISH
  BULLISH
  NEUTRAL
  CAUTIOUS
  PESSIMISTIC
}

enum CompetitivePosition {
  STRENGTHENING
  STABLE
  WEAKENING
  UNDER_PRESSURE
}

class StockNewsSignals {
  // Overall narrative
  dominant_themes string[]
  @description(#"
    Top 3-5 themes from company news (e.g., "Strong earnings beat", "Product launch success", "Regulatory concerns")
  "#)

  // Business momentum
  company_momentum CompanyMomentum
  momentum_summary string
  @description(#"
    2-3 sentences on business growth, revenue trends, market share changes
  "#)

  // Earnings and financials
  earnings_outlook EarningsOutlook
  earnings_summary string
  @description(#"
    2-3 sentences on recent/expected earnings, guidance, profitability
  "#)

  // Management and strategy
  management_sentiment ManagementSentiment
  management_summary string
  @description(#"
    2-3 sentences on management commentary, strategic initiatives, capital allocation
  "#)

  // Competitive dynamics
  competitive_position CompetitivePosition
  competitive_summary string
  @description(#"
    2-3 sentences on competitive positioning, industry trends, market dynamics
  "#)

  // Specific catalysts or risks
  key_catalysts string[]
  @description(#"
    List of specific positive catalysts mentioned (e.g., "New product launch", "Partnership announced")
  "#)

  key_risks string[]
  @description(#"
    List of specific risks or concerns (e.g., "Regulatory investigation", "Supply chain issues")
  "#)

  // Overall assessment
  news_bias string
  @description(#"
    Overall tone: "BULLISH", "NEUTRAL", or "BEARISH"
  "#)

  confidence_assessment string
  @description(#"
    How clear/actionable the news signal is: "HIGH", "MEDIUM", "LOW"
  "#)
}

function SummarizeStockNews(
  ticker: string,
  news: NewsArticle[]
) -> StockNewsSignals {
  client Model

  prompt #"
    {{ _.role("system") }}
    You are a senior equity analyst specializing in extracting actionable investment signals from company news for Black-Litterman portfolio optimization. Your outputs feed directly into a quantitative model that determines expected returns and position sizing.

    Core competencies:
    - Synthesize multiple news sources into coherent company narratives
    - Extract forward-looking signals that predict stock performance
    - Classify sentiment consistently using predefined categories
    - Balance positive and negative signals objectively

    {{ _.role("user") }}
    # OBJECTIVE
    Analyze {{ news | length }} news articles for {{ ticker }} and extract structured investment signals for Black-Litterman view generation.

    # INPUT DATA: NEWS ARTICLES ({{ ticker }})

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
    Scan all {{ news | length }} articles to identify:
    - Recurring themes mentioned in 2+ articles
    - Consensus vs. divergent viewpoints
    - Forward-looking statements (guidance, forecasts, launches) vs. historical results
    - Specific numbers (revenue, earnings, growth rates, margins)
    - Management commentary and tone shifts

    ## Step 2: Extract Structured Signals

    ### 2.1 Dominant Themes (3-5 themes)
    Extract the most frequently discussed company-specific topics.
    - Format: Concise phrases (4-6 words each)
    - Examples: "Earnings beat analyst estimates", "New product launch announced", "Regulatory approval delayed", "Market share gains evident"
    - Prioritize themes that appear in multiple articles or from credible sources

    ### 2.2 Company Momentum
    **Classification:**
    - ACCELERATING: Revenue/earnings accelerating, market share gains, strong growth trajectory
    - POSITIVE: Solid growth, positive trends, expanding margins, good execution
    - STABLE: Steady performance, in-line results, maintaining position
    - SLOWING: Decelerating growth, margin pressure, weakening metrics
    - NEGATIVE: Declining revenue/earnings, losing share, deteriorating fundamentals

    **Summary Requirements:**
    - 2-3 sentences maximum
    - Include specific metrics (revenue growth %, EPS, guidance)
    - Focus on: Business trends, growth rates, operational performance

    ### 2.3 Earnings Outlook
    **Classification:**
    - BEATS_EXPECTED: Recent beat, raised guidance, positive surprises
    - MEETS_EXPECTED: In-line results, confirmed guidance, expectations met
    - BELOW_EXPECTED: Miss, lowered guidance, disappointments
    - UNCERTAIN: Mixed signals, wide analyst range, unclear visibility

    **Summary Requirements:**
    - 2-3 sentences maximum
    - Include specific numbers (EPS, revenue, guidance ranges)
    - Note analyst revisions and consensus changes

    ### 2.4 Management Sentiment
    **Classification:**
    - VERY_BULLISH: Highly confident, raising targets, aggressive expansion
    - BULLISH: Optimistic, confident in strategy, positive outlook
    - NEUTRAL: Balanced, cautious optimism, data-dependent
    - CAUTIOUS: Acknowledging headwinds, tempering expectations
    - PESSIMISTIC: Defensive, cutting guidance, significant concerns

    **Summary Requirements:**
    - 2-3 sentences maximum
    - Quote specific management commentary when impactful
    - Focus on: Strategy updates, capital allocation, outlook statements

    ### 2.5 Competitive Position
    **Classification:**
    - STRENGTHENING: Gaining share, product/tech advantages, widening moat
    - STABLE: Maintaining position, competitive dynamics unchanged
    - WEAKENING: Losing share, facing new competition, eroding advantages
    - UNDER_PRESSURE: Significant competitive threats, market share losses

    **Summary Requirements:**
    - 2-3 sentences maximum
    - Include market share data if available
    - Note competitive actions (new products, pricing, M&A)

    ### 2.6 Catalysts and Risks
    **Key Catalysts** (2-5 items):
    - Specific positive events or developments
    - Examples: "FDA approval", "Major contract win", "Strategic partnership", "Product launch"
    - Quantify impact where possible

    **Key Risks** (2-5 items):
    - Specific concerns or headwinds
    - Examples: "Regulatory investigation", "Supply chain disruption", "Customer concentration", "Debt refinancing"
    - Assess materiality

    ### 2.7 Overall Assessment
    **News Bias:**
    - BULLISH: Predominantly positive news, multiple catalysts, positive surprises
    - NEUTRAL: Balanced news flow, mixed signals, no clear direction
    - BEARISH: Predominantly negative news, risks elevated, negative surprises

    **Confidence Assessment:**
    - HIGH: Clear signal, consistent across articles, actionable information
    - MEDIUM: Moderately clear, some ambiguity, reasonable conviction
    - LOW: Unclear signal, highly mixed messages, limited actionability

    ## Step 3: Quality Control
    - [ ] All summaries are 2-3 sentences
    - [ ] Specific numbers included where available
    - [ ] Forward-looking information prioritized
    - [ ] Balanced view (not cherry-picked)
    - [ ] Credible sources weighted appropriately
    - [ ] Catalysts and risks are specific and material

    # CRITICAL CONSTRAINTS

    1. **Synthesis**: Extract aggregate signal, don't list individual articles
    2. **Conciseness**: Exactly 2-3 sentences per summary field
    3. **Consistency**: Use exact enum values, don't invent categories
    4. **Data-Driven**: Include numbers when available
    5. **Objectivity**: Reflect true balance of news sentiment
    6. **Actionability**: Focus on signals that impact stock returns

    # OUTPUT FORMAT

    {{ ctx.output_format }}

    ---

    Now analyze all {{ news | length }} articles for {{ ticker }} following the framework above. Think step-by-step before generating structured output.
  "#
}
