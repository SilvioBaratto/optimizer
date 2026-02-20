// ClassifyMacroRegime.baml
// Classifies the current business cycle phase from macro indicators and
// returns calibrated Black-Litterman δ (risk aversion) and τ (uncertainty scaling).

class MacroRegimeCalibration {
  phase BusinessCyclePhase @description("Classified business cycle phase.")
  delta float @description("Black-Litterman risk aversion scalar. Range [1.0, 10.0]. Higher = more conservative equilibrium prior.")
  tau float @description("Black-Litterman uncertainty scaling. Range [0.001, 0.1]. Higher = more weight on views vs. equilibrium.")
  confidence float @description("Classification confidence in [0.0, 1.0].")
  rationale string @description("Brief explanation of the phase classification and parameter choices.")
}

function ClassifyMacroRegime(macro_summary: string) -> MacroRegimeCalibration {
  client CustomSonnet45
  prompt #"
    You are a macro strategist calibrating Black-Litterman portfolio model parameters.
    Classify the current business cycle phase and return appropriate δ (delta) and τ (tau).

    Phase definitions and parameter guidelines:
    - EARLY_EXPANSION: GDP recovering, unemployment falling, PMI >50 and rising, yield curve steepening.
      → delta: 2.0–2.5, tau: 0.05 (moderate risk aversion, significant view weighting)
    - MID_EXPANSION: Solid GDP growth, low unemployment, PMI >55, credit spreads tight.
      → delta: 2.5–3.0, tau: 0.025 (moderate risk aversion, balanced view weighting)
    - LATE_EXPANSION: Growth slowing, PMI plateauing/falling, yield curve flattening, inflation elevated.
      → delta: 3.0–4.0, tau: 0.01 (elevated risk aversion, low view weighting — trust equilibrium more)
    - RECESSION: GDP contracting, unemployment rising, PMI <50, yield curve inverted or credit spreads wide.
      → delta: 4.0–6.0, tau: 0.05 (high risk aversion, significant view weighting — views signal turning point)

    Hard constraints:
    - delta MUST be in [1.0, 10.0]
    - tau MUST be in [0.001, 0.1]
    - confidence MUST be in [0.0, 1.0]

    Macro indicators summary:
    {{ macro_summary }}

    {{ ctx.output_format }}
  "#
}

test ClassifyMacroRegimeExpansion {
  functions [ClassifyMacroRegime]
  args {
    macro_summary "US GDP growth: +3.1% YoY. Unemployment: 3.8%. ISM PMI: 56.2 (rising). 10Y-2Y spread: +120bps (steepening). CPI: +2.3% (near target). Industrial production: +2.1%."
  }
}

test ClassifyMacroRegimeRecession {
  functions [ClassifyMacroRegime]
  args {
    macro_summary "US GDP growth: -1.2% YoY (second consecutive contraction). Unemployment: 6.5% (rising). ISM PMI: 44.1 (falling). 10Y-2Y spread: -80bps (inverted). CPI: +4.1% (elevated). Credit spreads: 450bps."
  }
}

test ClassifyMacroRegimeLateExpansion {
  functions [ClassifyMacroRegime]
  args {
    macro_summary "US GDP growth: +1.8% YoY (slowing). Unemployment: 4.2%. ISM PMI: 50.5 (plateauing). 10Y-2Y spread: +15bps (flattening). CPI: +3.8% (elevated). Leading indicators: declining for 3 months."
  }
}
