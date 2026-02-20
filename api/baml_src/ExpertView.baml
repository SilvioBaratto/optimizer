// ExpertView.baml
// Parameterised expert view generation for Multi-LLM Opinion Pooling (issue #12).
// Reuses AssetFactorData, AssetView, and ViewOutput from GenerateViews.baml.
//
// Each function represents a distinct analyst persona with a different
// investment philosophy, enabling independent expert views for pooling.

enum ExpertPersona {
  VALUE_INVESTOR   @description("Focuses on valuation multiples: P/E, P/B, EV/EBITDA. Prefers cheap assets with earnings support.")
  MOMENTUM_TRADER  @description("Focuses on price momentum, RSI, and trend strength. Prefers assets with strong 12-1m momentum.")
  MACRO_ANALYST    @description("Focuses on macro sensitivity: revenue growth, earnings revisions, analyst consensus, and cycle positioning.")
}

function GenerateExpertView(
  assets: AssetFactorData[],
  persona: ExpertPersona
) -> ViewOutput {
  client CustomSonnet45
  prompt #"
    You are an investment expert generating Black-Litterman views.
    Your persona and investment philosophy: {{ persona }}

    {% if persona == ExpertPersona.VALUE_INVESTOR %}
    VALUE INVESTOR PHILOSOPHY:
    You focus exclusively on valuation:
    - Seek assets with trailing_pe < sector average and low price_to_book → bullish
    - Avoid assets where ev_to_ebitda is elevated vs. history → bearish
    - Strong analyst upside (target_upside > 0.15) confirms value → boost confidence
    - Ignore momentum signals; they are noise to you
    {% elif persona == ExpertPersona.MOMENTUM_TRADER %}
    MOMENTUM TRADER PHILOSOPHY:
    You focus exclusively on price momentum and technical signals:
    - High 12-1m momentum (> 0.15) with RSI < 70 → strong bullish
    - Negative 12-1m momentum AND recent reversal (1m < -0.03) → strong bearish
    - Distance from 52w high < -0.05 with recovering momentum → setup
    - Ignore valuation; momentum persists regardless of P/E
    {% elif persona == ExpertPersona.MACRO_ANALYST %}
    MACRO ANALYST PHILOSOPHY:
    You focus on growth, quality, and analyst signals:
    - High revenue and earnings growth YoY → bullish fundamental backdrop
    - Strong ROE + low debt/equity → quality buffer
    - Analyst recommendation_mean < 2.0 with narrow dispersion → strong buy signal
    - Earnings revisions positive (earnings_growth_yoy improving) → upgrade catalyst
    {% endif %}

    Guidelines for all personas:
    - Only generate views where your philosophy gives a clear signal.
    - direction = +1 (outperform) or -1 (underperform).
    - magnitude_bps: 50-500 bps/yr. Match conviction.
    - confidence in (0, 1). Calibrate to evidence strength.
    - idzorek_alphas must contain an entry for every asset in views[].

    Asset data:
    {{ assets }}

    {{ ctx.output_format }}
  "#
}

// Convenience wrappers for each persona (for simpler BAML client calls)

function GenerateValueView(assets: AssetFactorData[]) -> ViewOutput {
  client CustomSonnet45
  prompt #"
    You are a VALUE INVESTOR generating Black-Litterman views.

    Focus ONLY on valuation signals:
    - Low trailing_pe + low price_to_book → bullish
    - High ev_to_ebitda → bearish
    - Positive target_upside confirms value thesis → raise confidence
    - Ignore momentum and growth signals

    direction = +1 (bullish) or -1 (bearish).
    magnitude_bps = expected annual excess return in basis points (50–500).
    confidence ∈ (0, 1).
    idzorek_alphas must cover all assets in views[].

    Asset data:
    {{ assets }}

    {{ ctx.output_format }}
  "#
}

function GenerateMomentumView(assets: AssetFactorData[]) -> ViewOutput {
  client CustomOpus45
  prompt #"
    You are a MOMENTUM TRADER generating Black-Litterman views.

    Focus ONLY on momentum and technical signals:
    - momentum_12_1m > 0.15 AND rsi_14 < 70 → bullish continuation
    - momentum_12_1m < -0.10 AND momentum_1m < 0 → bearish continuation
    - Close to 52w high (pct_from_52w_high > -0.05) → momentum confirmation
    - Ignore valuation (P/E, P/B are noise)

    direction = +1 (bullish) or -1 (bearish).
    magnitude_bps = expected annual excess return in basis points (50–500).
    confidence ∈ (0, 1).
    idzorek_alphas must cover all assets in views[].

    Asset data:
    {{ assets }}

    {{ ctx.output_format }}
  "#
}

function GenerateMacroView(assets: AssetFactorData[]) -> ViewOutput {
  client CustomSonnet45
  prompt #"
    You are a MACRO ANALYST generating Black-Litterman views.

    Focus ONLY on growth, quality, and analyst consensus:
    - High revenue_growth_yoy + earnings_growth_yoy → fundamental tailwind → bullish
    - Strong return_on_equity + low debt_to_equity → quality → bullish
    - recommendation_mean < 2.0 → analyst consensus buy → bullish
    - recommendation_mean > 3.5 OR target_upside < 0 → analyst bearish → bearish
    - Ignore price momentum and short-term technicals

    direction = +1 (bullish) or -1 (bearish).
    magnitude_bps = expected annual excess return in basis points (50–500).
    confidence ∈ (0, 1).
    idzorek_alphas must cover all assets in views[].

    Asset data:
    {{ assets }}

    {{ ctx.output_format }}
  "#
}

test ExpertViewValue {
  functions [GenerateValueView]
  args {
    assets [
      {
        ticker "AAPL"
        trailing_pe 28.0
        price_to_book 45.0
        ev_to_ebitda 21.0
        momentum_12_1m 0.20
        momentum_1m 0.02
        rsi_14 60.0
        return_on_equity 1.4
        debt_to_equity 1.8
        profit_margins 0.25
        revenue_growth_yoy 0.06
        earnings_growth_yoy 0.13
        pct_from_52w_high -0.03
        pct_from_52w_low 0.28
        recommendation_mean 1.9
        target_upside 0.18
        analyst_count 45
      }
    ]
  }
}

test ExpertViewMomentum {
  functions [GenerateMomentumView]
  args {
    assets [
      {
        ticker "NVDA"
        trailing_pe 55.0
        price_to_book 30.0
        ev_to_ebitda 40.0
        momentum_12_1m 0.85
        momentum_1m 0.05
        rsi_14 65.0
        return_on_equity 0.90
        debt_to_equity 0.40
        profit_margins 0.55
        revenue_growth_yoy 1.22
        earnings_growth_yoy 2.10
        pct_from_52w_high -0.02
        pct_from_52w_low 0.95
        recommendation_mean 1.5
        target_upside 0.08
        analyst_count 50
      }
    ]
  }
}
