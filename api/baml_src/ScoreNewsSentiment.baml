// ScoreNewsSentiment.baml
// LLM-based financial news sentiment scoring for Idzorek alpha adjustment.
// Each news headline is scored in [-1, 1]; scores feed into the
// exponential-decay signal used to adjust view confidence in issue #13.

class NewsArticle {
  title string  @description("News headline to score.")
}

class NewsSentimentOutput {
  // One score per input article, in the same order as articles[].
  // Each value is in [-1.0, 1.0].
  scores float[]  @description("Sentiment score for each article: +1=strongly bullish, -1=strongly bearish, 0=neutral.")
  reasoning string @description("Brief overall summary of the sentiment assessment.")
}

function ScoreNewsSentiment(
  ticker: string,
  articles: NewsArticle[]
) -> NewsSentimentOutput {
  client CustomSonnet45
  prompt #"
    You are a financial analyst scoring news sentiment for {{ ticker }}.

    Rate each headline below using these guidelines:
    - +1.0 = Strongly bullish  (earnings beat, major contract win, strong buy upgrade, record revenue)
    - +0.5 = Moderately bullish (guidance raised, positive product launch, mild analyst upgrade)
    -  0.0 = Neutral            (routine personnel change, minor announcement, unclear impact)
    - -0.5 = Moderately bearish (guidance cut, regulatory inquiry, mild analyst downgrade)
    - -1.0 = Strongly bearish   (earnings miss, lawsuit, mass layoffs, major loss, credit downgrade)

    Rules:
    - Return exactly one score per headline, in the same order as the input.
    - Scores must be in [-1.0, 1.0].
    - Intermediate values (e.g. 0.3, -0.7) are allowed and encouraged for nuance.
    - Focus on the likely PRICE IMPACT for {{ ticker }}, not general market sentiment.

    Headlines:
    {{ articles }}

    {{ ctx.output_format }}
  "#
}

test ScoreNewsSentimentApple {
  functions [ScoreNewsSentiment]
  args {
    ticker "AAPL"
    articles [
      { title "Apple reports record Q4 earnings, beats EPS by 15%" }
      { title "Apple faces antitrust probe in EU over App Store policies" }
      { title "Apple names new VP of Engineering" }
    ]
  }
}
