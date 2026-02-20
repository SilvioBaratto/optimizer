// CalibrateRiskBudget.baml
// LLM-driven translation of qualitative sector outlook to a numeric
// risk budget vector for use with build_risk_budgeting().

class RiskBudgetOutput {
  sector_budgets map<string, float>
    @description("Sector name → risk budget weight. All values are non-negative and sum to 1.0.")
  asset_budgets map<string, float>
    @description("Asset ticker → risk budget weight. Derived from sector_budgets by equal-weighting assets within each sector. Values are non-negative and sum to 1.0.")
  rationale string
    @description("2-4 sentence explanation of the budget allocation rationale.")
}

function CalibrateRiskBudget(
  sector_outlook: string,
  sector_universe: string[],
  asset_sector_map: map<string, string>
) -> RiskBudgetOutput {
  client CustomSonnet45
  prompt #"
    You are a quantitative risk manager translating a qualitative sector outlook
    into a precise risk budget vector for a risk-parity portfolio.

    Sector universe: {{ sector_universe | join(", ") }}

    Asset-to-sector mapping:
    {% for asset, sector in asset_sector_map %}
    - {{ asset }}: {{ sector }}
    {% endfor %}

    Analyst sector outlook:
    {{ sector_outlook }}

    Instructions:
    1. Assign a risk budget weight to each sector in the universe based on the outlook.
       - Sectors described as "overweight", "positive", "bullish", or "favoured"
         receive HIGHER budget weights (they are allowed to take more risk).
       - Sectors described as "underweight", "negative", "bearish", or "avoid"
         receive LOWER budget weights.
       - Neutral sectors receive weights proportional to 1/N (equal budget baseline).
    2. sector_budgets must cover ALL sectors in the sector universe.
    3. All sector budget values must be non-negative.
    4. sector_budgets must sum to exactly 1.0.
    5. For asset_budgets: distribute each sector's budget equally among all assets
       mapped to that sector (asset_budget = sector_budget / n_assets_in_sector).
    6. asset_budgets must cover ALL assets in asset_sector_map.
    7. asset_budgets must also sum to exactly 1.0.
    8. Write a brief rationale explaining your allocation choices.

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
