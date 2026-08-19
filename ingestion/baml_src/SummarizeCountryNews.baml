// SummarizeCountryNews.baml
// Generates a daily country-level macro news summary from aggregated article text.
// Output feeds into the macro_news_summaries table (Phase 2 of the news pipeline).

enum SentimentLabel {
  BULLISH   @description("Predominantly positive macro outlook; risk-on signals dominate.")
  BEARISH   @description("Predominantly negative macro outlook; risk-off signals dominate.")
  NEUTRAL   @description("Mixed or unclear signals; no strong directional bias.")
  MIXED     @description("Significant bullish and bearish signals coexist; high uncertainty.")
}

class CountryNewsSummary {
  summary string
    @description("Concise macro summary (3-5 sentences) of the day's news for the country. Cover key economic events, policy signals, market reactions, and forward implications. Focus on market-moving themes only.")
  sentiment SentimentLabel
    @description("Overall macro sentiment derived from the news.")
  sentiment_score float
    @description("Numeric sentiment score in [-1.0, 1.0]. +1.0 = strongly bullish, -1.0 = strongly bearish, 0.0 = neutral.")
}

function SummarizeCountryNews(country: string, news_text: string) -> CountryNewsSummary {
  client Model
  prompt #"
    You are a macro research analyst producing a daily country intelligence briefing.
    Country: {{ country }}

    Guidelines for macro-relevant analysis:
    - Focus on: central bank policy, GDP/growth signals, inflation, employment, PMI/activity data,
      fiscal policy, credit conditions, geopolitical risk, currency moves, and equity market implications.
    - Ignore: celebrity news, sports, unrelated domestic politics unless they have clear macro impact.
    - Sentiment calibration:
        BULLISH  / score > +0.3 : growth-positive signals, dovish pivot, strong data beats
        MIXED    / score near 0 : conflicting signals, policy uncertainty, mixed data
        NEUTRAL  / score near 0 : no meaningful macro signal in today's news
        BEARISH  / score < -0.3 : recession risk, hawkish surprise, major data miss, systemic stress
    - sentiment_score must be in [-1.0, 1.0]. Use the full range; intermediate values are encouraged.
    - summary should be concise (3-5 sentences). Focus on market-moving themes only.

    News articles for {{ country }}:
    {{ news_text }}

    {{ ctx.output_format }}
  "#
}

test SummarizeCountryNewsUSA {
  functions [SummarizeCountryNews]
  args {
    country "USA"
    news_text "Fed holds rates at 5.25%-5.5%, signals two cuts in 2025. Non-farm payrolls beat at +250k vs +185k expected. CPI came in at 3.1% YoY, above the 2.9% forecast. ISM Manufacturing PMI: 51.2, first expansion in six months. Treasury yields rose 10bps on the hot inflation print."
  }
}

test SummarizeCountryNewsGermany {
  functions [SummarizeCountryNews]
  args {
    country "Germany"
    news_text "German GDP contracted 0.3% in Q4 2024, second consecutive quarter of negative growth. IFO Business Climate index fell to 84.7, lowest since 2020. ECB holds rates but Lagarde signals readiness to cut if data deteriorates further. German 10Y bund yield fell 8bps to 2.21%."
  }
}
