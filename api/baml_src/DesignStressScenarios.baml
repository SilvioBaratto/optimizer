// DesignStressScenarios.baml
// LLM-driven forward-looking stress scenario design for tail risk events.
// Historical simulation only captures risks that have materialised; this
// function generates plausible but not-yet-observed shocks that feed into
// build_synthetic_data() for conditional stress testing.

class StressScenario {
  name string
    @description("Short label for the scenario, e.g. 'Global Recession 2026'.")
  description string
    @description("2-4 sentence narrative explaining the macro/geopolitical trigger.")
  shocks map<string, float>
    @description("Ticker → expected return shock over the horizon. Values in (-1, 1). E.g. {\"SPY\": -0.15, \"GLD\": 0.08}.")
  probability float
    @description("Subjective probability that this scenario materialises. Must be in (0, 1).")
  horizon_days int
    @description("Time horizon over which the shocks occur, in trading days.")
}

function DesignStressScenarios(
  tickers: string[],
  macro_context: string,
  n_scenarios: int
) -> StressScenario[] {
  client CustomSonnet45
  prompt #"
    You are a senior risk manager designing forward-looking stress scenarios for a
    portfolio of the following assets: {{ tickers | join(", ") }}.

    Current macro context provided by the user:
    {{ macro_context }}

    Design exactly {{ n_scenarios }} plausible but not-yet-materialised stress scenarios.

    Rules:
    1. Each scenario must be DISTINCT — cover different risk factors
       (e.g. recession, inflation shock, geopolitical crisis, credit event, sector crash).
    2. At least ONE scenario must represent a broad market drawdown (>10% loss on most equities).
    3. Shock values are expected RETURNS over the scenario horizon (not multipliers).
       Valid range: (-1.0, 1.0). Negative = loss, positive = gain.
    4. Every ticker in the portfolio must have a shock entry.
    5. Probability must be in (0, 1) — subjective forward probability.
    6. horizon_days is the number of trading days over which the shock unfolds.
    7. Safe-haven assets (GLD, TLT, SHV, BIL) should have positive or mildly negative
       shocks in risk-off scenarios and negative shocks in inflation/stagflation scenarios.

    Scenario diversity examples (do not copy, use as inspiration):
    - Deep recession: equities -25 to -40%, rates rally, gold +15%
    - Inflation shock / stagflation: equities -15%, gold +10%, bonds -20%
    - Geopolitical escalation: energy +20%, tech -20%, defense +15%
    - Credit crunch: high-yield spreads widen, financials -30%, treasuries +10%
    - Technology sector crash: tech -40%, value sectors flat to +5%

    {{ ctx.output_format }}
  "#
}

test DesignStressScenariosBasic {
  functions [DesignStressScenarios]
  args {
    tickers ["SPY", "QQQ", "GLD", "TLT", "XLE"]
    macro_context "US CPI running at 4.2%, Fed on hold, geopolitical tension in Middle East, AI sector valuations elevated."
    n_scenarios 3
  }
}
