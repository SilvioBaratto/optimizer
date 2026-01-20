from __future__ import annotations

from typing import Dict, List

from optimizer.src.macro_regime.strategies.protocol import MacroRegime


class FactorTimingCalculator:
    """
    Calculate factor exposure recommendations based on business cycle regime.
    """

    # Factor timing recommendations by regime
    # Values: 'overweight', 'neutral', 'underweight'
    REGIME_TIMING: Dict[MacroRegime, Dict[str, str]] = {
        MacroRegime.EARLY_CYCLE: {
            "Value": "overweight",
            "Momentum": "overweight",
            "Quality": "neutral",
            "Growth": "overweight",
            "Low Volatility": "underweight",
        },
        MacroRegime.MID_CYCLE: {
            "Value": "neutral",
            "Momentum": "neutral",
            "Quality": "overweight",
            "Growth": "neutral",
            "Low Volatility": "neutral",
        },
        MacroRegime.LATE_CYCLE: {
            "Value": "overweight",
            "Momentum": "underweight",
            "Quality": "overweight",
            "Growth": "underweight",
            "Low Volatility": "overweight",
        },
        MacroRegime.RECESSION: {
            "Value": "neutral",
            "Momentum": "underweight",
            "Quality": "overweight",
            "Growth": "underweight",
            "Low Volatility": "overweight",
        },
        MacroRegime.UNCERTAIN: {
            "Value": "neutral",
            "Momentum": "neutral",
            "Quality": "overweight",
            "Growth": "neutral",
            "Low Volatility": "overweight",
        },
    }

    # Factor characteristics
    FACTOR_CHARACTERISTICS: Dict[str, Dict[str, str]] = {
        "Value": {
            "description": "Cheap stocks relative to book value, earnings, or cash flow",
            "risk_profile": "medium",
            "cycle_sensitivity": "early_late",
        },
        "Momentum": {
            "description": "Stocks with strong recent price performance",
            "risk_profile": "high",
            "cycle_sensitivity": "early_cycle",
        },
        "Quality": {
            "description": "Stocks with strong ROE, low leverage, stable earnings",
            "risk_profile": "low",
            "cycle_sensitivity": "defensive",
        },
        "Growth": {
            "description": "Stocks with high expected earnings growth",
            "risk_profile": "high",
            "cycle_sensitivity": "early_mid",
        },
        "Low Volatility": {
            "description": "Stocks with lower price volatility",
            "risk_profile": "low",
            "cycle_sensitivity": "defensive",
        },
    }

    def __init__(self):
        pass

    def get_timing(self, regime: MacroRegime) -> Dict[str, str]:
        """
        Get factor timing recommendations for the given regime.
        """
        return self.REGIME_TIMING.get(regime, {}).copy()

    def get_overweight_factors(self, regime: MacroRegime) -> List[str]:
        """Get factors to overweight for the regime."""
        timing = self.get_timing(regime)
        return [factor for factor, rec in timing.items() if rec == "overweight"]

    def get_underweight_factors(self, regime: MacroRegime) -> List[str]:
        """Get factors to underweight for the regime."""
        timing = self.get_timing(regime)
        return [factor for factor, rec in timing.items() if rec == "underweight"]

    def get_defensive_factors(self) -> List[str]:
        """Get list of defensive factors."""
        return [
            factor
            for factor, chars in self.FACTOR_CHARACTERISTICS.items()
            if chars["cycle_sensitivity"] == "defensive"
        ]

    def get_cyclical_factors(self) -> List[str]:
        """Get list of cyclical factors."""
        return [
            factor
            for factor, chars in self.FACTOR_CHARACTERISTICS.items()
            if chars["cycle_sensitivity"] in ["early_cycle", "early_mid", "early_late"]
        ]

    def get_factor_exposure_style(self, regime: MacroRegime) -> str:
        """
        Get overall factor exposure style for the regime.
        """
        timing = self.get_timing(regime)

        # Count recommendations
        defensive_overweight = sum(
            1 for f in ["Quality", "Low Volatility"] if timing.get(f) == "overweight"
        )
        growth_overweight = sum(1 for f in ["Growth", "Momentum"] if timing.get(f) == "overweight")

        if growth_overweight >= 2:
            return "growth_momentum"
        elif defensive_overweight >= 2:
            return "quality_defensive"
        return "balanced"

    def apply_recession_overlay(
        self,
        timing: Dict[str, str],
        recession_risk: float,
    ) -> Dict[str, str]:
        """
        Adjust factor timing based on recession risk.

        Args:
            timing: Base factor timing
            recession_risk: Recession probability (0-1)

        Returns:
            Adjusted factor timing
        """
        if recession_risk < 0.25:
            return timing

        adjusted = timing.copy()

        # Increase defensive factors
        if recession_risk >= 0.40:
            adjusted["Quality"] = "overweight"
            adjusted["Low Volatility"] = "overweight"
            adjusted["Momentum"] = "underweight"
            adjusted["Growth"] = "underweight"
        elif recession_risk >= 0.25:
            if adjusted.get("Quality") == "neutral":
                adjusted["Quality"] = "overweight"
            if adjusted.get("Momentum") == "overweight":
                adjusted["Momentum"] = "neutral"

        return adjusted

    def get_factor_weight_adjustments(
        self,
        timing: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Convert timing recommendations to weight adjustments.
        """
        weight_map = {
            "overweight": 0.10,  # +10%
            "neutral": 0.0,
            "underweight": -0.10,  # -10%
        }

        return {factor: weight_map.get(rec, 0.0) for factor, rec in timing.items()}

    def calculate_factor_score(
        self,
        stock_factors: Dict[str, float],
        timing: Dict[str, str],
    ) -> float:
        """
        Calculate a stock's factor alignment score.
        """
        score = 0.0
        weight_adjustments = self.get_factor_weight_adjustments(timing)

        for factor, exposure in stock_factors.items():
            adjustment = weight_adjustments.get(factor, 0.0)
            # Positive adjustment + high exposure = good
            # Negative adjustment + low exposure = good
            score += adjustment * (exposure - 0.5)

        return score
