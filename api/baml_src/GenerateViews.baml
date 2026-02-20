// GenerateViews.baml
// LLM-driven Black-Litterman view generation from multi-factor per-asset data.

class AssetFactorData {
  ticker string @description("Asset ticker symbol (e.g. 'AAPL').")

  // Valuation (z-scores or raw ratios; null if unavailable)
  trailing_pe float? @description("Trailing price-to-earnings ratio.")
  price_to_book float? @description("Price-to-book ratio.")
  ev_to_ebitda float? @description("Enterprise value / EBITDA ratio.")

  // Momentum
  momentum_12_1m float? @description("12-1 month price momentum (decimal, e.g. 0.18 = +18%).")
  momentum_1m float? @description("1-month price momentum (decimal). Negative = recent reversal.")
  rsi_14 float? @description("14-day RSI in [0, 100]. >70 overbought, <30 oversold.")

  // Quality
  return_on_equity float? @description("Return on equity (decimal, e.g. 0.15 = 15%).")
  debt_to_equity float? @description("Debt-to-equity ratio.")
  profit_margins float? @description("Net profit margin (decimal).")

  // Growth
  revenue_growth_yoy float? @description("Revenue growth year-over-year (decimal).")
  earnings_growth_yoy float? @description("Earnings growth year-over-year (decimal).")

  // Technical
  pct_from_52w_high float? @description("Distance from 52-week high (negative means below high, e.g. -0.15 = 15% below).")
  pct_from_52w_low float? @description("Distance from 52-week low (positive, e.g. 0.30 = 30% above low).")

  // Analyst consensus
  recommendation_mean float? @description("Mean analyst recommendation: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell.")
  target_upside float? @description("(target_mean_price - current_price) / current_price. Positive = upside.")
  analyst_count int? @description("Number of analysts covering the stock.")
}

class AssetView {
  asset string @description("Ticker symbol of the asset this view is about.")
  direction int @description("View direction: +1 (bullish/outperform) or -1 (bearish/underperform).")
  magnitude_bps float @description("Expected annual excess return in basis points (e.g. 200 = 2% p.a.). Always positive; sign is captured by direction.")
  confidence float @description("View confidence in (0, 1). Higher = more certain. Maps to Idzorek alpha_k.")
  reasoning string @description("One-sentence explanation referencing the factor evidence.")
}

class ViewOutput {
  views AssetView[] @description("List of asset views. Only include assets with sufficiently strong factor evidence.")
  idzorek_alphas map<string, float> @description("Asset ticker -> Idzorek alpha_k in (0, 1). Must include an entry for every asset in views[].")
  rationale string @description("Brief overall narrative explaining the view generation process and dominant themes.")
}

function GenerateViews(assets: AssetFactorData[]) -> ViewOutput {
  client CustomSonnet45
  prompt #"
    You are a quantitative portfolio manager constructing Black-Litterman views from
    multi-factor asset data. Your task is to generate a set of high-conviction views
    that will be fed into the Black-Litterman prior estimation model.

    Guidelines:
    - Only generate a view for an asset when the factor evidence is sufficiently strong.
    - direction = +1 means you expect the asset to generate above-equilibrium returns.
    - direction = -1 means you expect below-equilibrium returns.
    - magnitude_bps: typical range is 50-500 bps/yr. Use higher values for stronger conviction.
    - confidence: calibrate to [0.1, 0.9]. Avoid extremes. High confidence = small Omega uncertainty.
    - idzorek_alphas must have a key for every asset that appears in views[].
    - The alphas in idzorek_alphas must match the confidence values in views[].

    Factor interpretation:
    - Low P/E + Low P/B → value opportunity → bullish signal
    - High 12-1m momentum + moderate RSI → momentum continuation → bullish
    - High ROE + low debt + high margins → quality → bullish
    - Revenue + earnings growth above average → growth → bullish
    - Close to 52w high but NOT overbought (RSI<70) → momentum confirmation
    - Recommendation mean < 2.5 → analyst bullish; > 3.5 → analyst bearish
    - Target upside > 0.15 → strong analyst bullish signal

    Asset data:
    {{ assets }}

    {{ ctx.output_format }}
  "#
}

test GenerateViewsMixed {
  functions [GenerateViews]
  args {
    assets [
      {
        ticker "AAPL"
        trailing_pe 28.5
        price_to_book 45.2
        ev_to_ebitda 21.0
        momentum_12_1m 0.22
        momentum_1m 0.03
        rsi_14 62.0
        return_on_equity 1.47
        debt_to_equity 1.76
        profit_margins 0.26
        revenue_growth_yoy 0.05
        earnings_growth_yoy 0.12
        pct_from_52w_high -0.03
        pct_from_52w_low 0.28
        recommendation_mean 1.8
        target_upside 0.12
        analyst_count 45
      }
      {
        ticker "INTC"
        trailing_pe 35.0
        price_to_book 1.2
        ev_to_ebitda 18.0
        momentum_12_1m -0.28
        momentum_1m -0.06
        rsi_14 32.0
        return_on_equity 0.04
        debt_to_equity 0.68
        profit_margins 0.02
        revenue_growth_yoy -0.12
        earnings_growth_yoy -0.45
        pct_from_52w_high -0.41
        pct_from_52w_low 0.05
        recommendation_mean 3.8
        target_upside -0.05
        analyst_count 38
      }
    ]
  }
}
