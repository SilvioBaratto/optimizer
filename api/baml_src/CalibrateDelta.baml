// CalibrateDelta.baml
// Calibrates the Black-Litterman risk aversion scalar (delta) from macro regime text.

class DeltaCalibration {
  delta float @description("Risk aversion scalar in [1.0, 10.0]. Higher = more risk-averse prior.")
  rationale string @description("Brief explanation of why this delta was chosen.")
}

function CalibrateDelta(macro_text: string) -> DeltaCalibration {
  client CustomSonnet45
  prompt #"
    You are a quantitative portfolio strategist. Your task is to calibrate the
    Black-Litterman risk aversion scalar (delta) based on a macro regime description.

    Guidelines:
    - delta = 1.0–3.0: benign / expansionary environment (equities favoured, low uncertainty)
    - delta = 3.0–6.0: neutral / mixed signals (balanced positioning)
    - delta = 6.0–10.0: recessionary / high-uncertainty environment (defensive positioning)

    Macro context:
    {{ macro_text }}

    Return a JSON object with:
    - delta: float clamped to [1.0, 10.0]
    - rationale: one-sentence explanation

    {{ ctx.output_format }}
  "#
}

test CalibrateDeltaExpansion {
  functions [CalibrateDelta]
  args {
    macro_text "GDP growth accelerated to 3.2% YoY. The Fed held rates steady signalling a soft landing. Credit spreads are tight and equity volatility (VIX) is at 12."
  }
}

test CalibrateDeltaRecession {
  functions [CalibrateDelta]
  args {
    macro_text "The Fed raised rates by 75bps for the third consecutive meeting. GDP contracted for two quarters. Credit spreads widened to 500bps and unemployment rose to 6.5%."
  }
}
