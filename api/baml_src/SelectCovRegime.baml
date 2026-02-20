// SelectCovRegime.baml
// Selects the appropriate covariance estimator based on market conditions and news sentiment.

enum CovEstimatorChoice {
  EMPIRICAL          @description("Standard sample covariance; use in calm, data-rich environments.")
  LEDOIT_WOLF        @description("Ledoit-Wolf shrinkage; robust default for most regimes.")
  OAS                @description("Oracle Approximating Shrinkage; works well with small N/T ratio.")
  SHRUNK             @description("Manual shrinkage; use when you want explicit shrinkage control.")
  EW                 @description("Exponentially weighted; use in trending or high-volatility regimes.")
  GERBER             @description("Gerber statistic; robust to outliers and fat tails.")
  GRAPHICAL_LASSO_CV @description("Sparse precision matrix; use when few factors drive returns.")
  DENOISE            @description("Random matrix theory denoising; clean covariance in normal regimes.")
  DETONE             @description("Denoised + market mode removed; for diversified factor models.")
  IMPLIED            @description("Option-implied covariance; use when options market data is available.")
}

class CovRegimeSelection {
  estimator CovEstimatorChoice @description("Recommended covariance estimator for current conditions.")
  confidence float @description("Confidence in this selection, in [0.0, 1.0].")
  rationale string @description("Brief explanation referencing the sentiment and volatility regime.")
}

function SelectCovRegime(
  news_headlines: string[],
  avg_sentiment_score: float,
  realized_vol_30d: float
) -> CovRegimeSelection {
  client CustomSonnet45
  prompt #"
    You are a quantitative risk model specialist. Select the most appropriate covariance
    estimator for portfolio construction given current market conditions.

    Decision guidelines:
    - avg_sentiment_score < -0.3 (negative) AND realized_vol_30d > 0.20: use EW (trending vol)
    - avg_sentiment_score in [-0.3, 0.3] AND realized_vol_30d in [0.10, 0.20]: use LEDOIT_WOLF
    - avg_sentiment_score > 0.3 (positive) AND realized_vol_30d < 0.10: use DENOISE (stable)
    - Extreme stress (realized_vol_30d > 0.35): use GERBER (outlier-robust)
    - When few news topics dominate (concentrated): use GRAPHICAL_LASSO_CV
    - Default fallback: LEDOIT_WOLF

    Sentiment score: {{ avg_sentiment_score }} (range [-1.0, 1.0])
    30-day realized volatility (annualised): {{ realized_vol_30d }}

    Recent news headlines:
    {% for h in news_headlines %}
    - {{ h }}
    {% endfor %}

    {{ ctx.output_format }}
  "#
}

test SelectCovRegimeHighVol {
  functions [SelectCovRegime]
  args {
    news_headlines [
      "Fed signals aggressive rate hikes amid persistent inflation",
      "Bank of England warns of recession risk",
      "Tech sector selloff accelerates, Nasdaq down 15% YTD"
    ]
    avg_sentiment_score -0.65
    realized_vol_30d 0.28
  }
}

test SelectCovRegimeCalm {
  functions [SelectCovRegime]
  args {
    news_headlines [
      "S&P 500 hits all-time high on strong earnings season",
      "Fed holds rates, markets cheer soft landing narrative"
    ]
    avg_sentiment_score 0.55
    realized_vol_30d 0.09
  }
}
