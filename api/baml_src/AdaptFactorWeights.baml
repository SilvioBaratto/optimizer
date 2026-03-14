// AdaptFactorWeights.baml
// Classifies business cycle phase and returns factor group weight multipliers.

enum BusinessCyclePhase {
  EARLY_EXPANSION
  MID_EXPANSION
  LATE_EXPANSION
  RECESSION
}

class FactorWeightAdaptation {
  phase BusinessCyclePhase @description("Detected business cycle phase.")
  weights map<string, float> @description("Factor group -> relative weight multiplier. Higher = overweight, lower = underweight.")
  rationale string @description("Brief explanation of the phase classification and weight adjustments.")
}

function AdaptFactorWeights(macro_indicators: string, factor_groups: string[]) -> FactorWeightAdaptation {
  client Model
  prompt #"
    You are a quantitative macro strategist. Based on the macro indicators provided,
    classify the current business cycle phase and recommend factor weight multipliers.

    Business cycle phases and typical factor tilts:
    - EARLY_EXPANSION: favour momentum (MTUM), quality (QUAL); underweight value (VLUE)
    - MID_EXPANSION: balanced; slight overweight momentum and low-volatility (USMV)
    - LATE_EXPANSION: favour low-volatility (USMV), dividend (DVY); underweight momentum
    - RECESSION: favour value (VLUE), quality (QUAL); underweight momentum (MTUM)

    Each multiplier must be positive (minimum 0.1). Higher values = overweight, lower = underweight.

    Macro indicators:
    {{ macro_indicators }}

    Factor groups to weight:
    {{ factor_groups }}

    {{ ctx.output_format }}
  "#
}

test AdaptFactorWeightsExpansion {
  functions [AdaptFactorWeights]
  args {
    macro_indicators "ISM Manufacturing PMI: 55.3. Unemployment: 4.2%. 10Y-2Y spread: +120bps. Leading indicators index: rising."
    factor_groups ["momentum", "value", "quality", "low_volatility"]
  }
}

test AdaptFactorWeightsRecession {
  functions [AdaptFactorWeights]
  args {
    macro_indicators "ISM PMI: 47.1. Unemployment: 6.8%. Yield curve inverted: -50bps. Leading indicators: declining for 8 months."
    factor_groups ["momentum", "value", "quality", "low_volatility"]
  }
}
