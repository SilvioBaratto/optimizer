// CalibrateRiskBudget.baml
// LLM-driven translation of qualitative sector outlook to a numeric
// risk budget vector for use with build_risk_budgeting().

class RiskBudgetOutput {
  sector_budgets map<string, float>
    @description("Sector name → relative risk budget weight. All values are non-negative. Higher = more risk budget.")
  asset_budgets map<string, float>
    @description("Asset ticker → relative risk budget weight. All values are non-negative. Higher = more risk budget.")
  rationale string
    @description("2-4 sentence explanation of the budget allocation rationale.")
}

function CalibrateRiskBudget(
  sector_outlook: string,
  sector_universe: string[],
  asset_sector_map: map<string, string>
) -> RiskBudgetOutput {
  client Model
  prompt #"
    You are a quantitative risk manager translating a qualitative sector outlook
    into a precise risk budget vector for a risk-parity portfolio.

    Sector universe: {{ sector_universe | join(", ") }}

    Asset-to-sector mapping:
    {{ asset_sector_map }}

    Analyst sector outlook:
    {{ sector_outlook }}

    Instructions:
    1. Assign a relative risk budget weight to each sector based on the outlook.
       - "overweight" / "bullish" sectors get HIGHER weights.
       - "underweight" / "bearish" sectors get LOWER weights.
       - Neutral sectors get moderate weights.
    2. sector_budgets must cover ALL sectors in the sector universe.
    3. All values must be non-negative.
    4. For asset_budgets: give each asset a weight reflecting its sector's budget.
       Assets in the same sector should have similar weights.
    5. asset_budgets must cover ALL assets in asset_sector_map.
    6. Write a brief rationale explaining your allocation choices.

    {{ ctx.output_format }}
  "#
}

test CalibrateRiskBudgetBasic {
  functions [CalibrateRiskBudget]
  args {
    sector_outlook "Overweight Technology and Healthcare. Underweight Energy and Utilities. Neutral on Financials."
    sector_universe ["Technology", "Healthcare", "Energy", "Utilities", "Financials"]
    asset_sector_map {
      "AAPL": "Technology"
      "MSFT": "Technology"
      "JNJ": "Healthcare"
      "XOM": "Energy"
      "NEE": "Utilities"
      "JPM": "Financials"
    }
  }
}
