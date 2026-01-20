"""
Sector Tilt Calculator - Sector allocation recommendations by regime.

Based on institutional methodology from portfolio_guideline/guide.md pages 73-79.
"""

from __future__ import annotations

from typing import Dict, List

from optimizer.src.macro_regime.strategies.protocol import MacroRegime


class SectorTiltCalculator:
    """
    Calculate sector allocation recommendations based on business cycle regime.

    From guide.md pages 73-79: Different sectors perform differently
    across the business cycle.
    """

    # Sector tilt recommendations by regime
    # Values represent weight adjustments (e.g., +0.05 = overweight by 5%)
    REGIME_TILTS: Dict[MacroRegime, Dict[str, float]] = {
        MacroRegime.EARLY_CYCLE: {
            "Consumer Discretionary": +0.05,
            "Industrials": +0.04,
            "Technology": +0.04,
            "Financials": +0.03,
            "Real Estate": +0.03,
            "Materials": +0.02,
            "Utilities": -0.04,
            "Consumer Staples": -0.03,
            "Healthcare": -0.02,
        },
        MacroRegime.MID_CYCLE: {
            # Minimal tilts - focus on stock selection (guide.md page 76)
            "Technology": +0.01,
            "Communication Services": +0.01,
            "Materials": -0.01,
            "Utilities": -0.01,
        },
        MacroRegime.LATE_CYCLE: {
            "Energy": +0.05,
            "Materials": +0.03,
            "Consumer Staples": +0.03,
            "Healthcare": +0.02,
            "Utilities": +0.02,
            "Technology": -0.03,
            "Consumer Discretionary": -0.03,
            "Financials": -0.02,
        },
        MacroRegime.RECESSION: {
            "Consumer Staples": +0.06,
            "Healthcare": +0.05,
            "Utilities": +0.04,
            "Technology": -0.06,
            "Consumer Discretionary": -0.05,
            "Financials": -0.05,
            "Industrials": -0.04,
            "Real Estate": -0.03,
        },
        MacroRegime.UNCERTAIN: {},
    }

    # Sector characteristics
    SECTOR_CHARACTERISTICS: Dict[str, Dict[str, str]] = {
        "Consumer Discretionary": {
            "beta": "high",
            "cyclicality": "high",
            "duration": "long",
        },
        "Consumer Staples": {
            "beta": "low",
            "cyclicality": "defensive",
            "duration": "short",
        },
        "Energy": {
            "beta": "high",
            "cyclicality": "late_cycle",
            "duration": "short",
        },
        "Financials": {
            "beta": "high",
            "cyclicality": "early_cycle",
            "duration": "medium",
        },
        "Healthcare": {
            "beta": "low",
            "cyclicality": "defensive",
            "duration": "long",
        },
        "Industrials": {
            "beta": "high",
            "cyclicality": "early_cycle",
            "duration": "medium",
        },
        "Materials": {
            "beta": "high",
            "cyclicality": "late_cycle",
            "duration": "short",
        },
        "Real Estate": {
            "beta": "medium",
            "cyclicality": "rate_sensitive",
            "duration": "long",
        },
        "Technology": {
            "beta": "high",
            "cyclicality": "growth",
            "duration": "long",
        },
        "Communication Services": {
            "beta": "medium",
            "cyclicality": "mixed",
            "duration": "medium",
        },
        "Utilities": {
            "beta": "low",
            "cyclicality": "defensive",
            "duration": "long",
        },
    }

    def __init__(self):
        pass

    def get_tilts(self, regime: MacroRegime) -> Dict[str, float]:
        """
        Get sector allocation tilts for the given regime.

        Args:
            regime: Current business cycle regime

        Returns:
            Dictionary of sector -> weight adjustment
        """
        return self.REGIME_TILTS.get(regime, {}).copy()

    def get_overweights(self, regime: MacroRegime) -> List[str]:
        """Get sectors to overweight for the regime."""
        tilts = self.get_tilts(regime)
        return [sector for sector, tilt in tilts.items() if tilt > 0]

    def get_underweights(self, regime: MacroRegime) -> List[str]:
        """Get sectors to underweight for the regime."""
        tilts = self.get_tilts(regime)
        return [sector for sector, tilt in tilts.items() if tilt < 0]

    def get_defensive_sectors(self) -> List[str]:
        """Get list of defensive sectors."""
        return [
            sector
            for sector, chars in self.SECTOR_CHARACTERISTICS.items()
            if chars["cyclicality"] == "defensive"
        ]

    def get_cyclical_sectors(self) -> List[str]:
        """Get list of cyclical sectors."""
        return [
            sector
            for sector, chars in self.SECTOR_CHARACTERISTICS.items()
            if chars["cyclicality"] in ["high", "early_cycle", "late_cycle"]
        ]

    def apply_recession_overlay(
        self,
        tilts: Dict[str, float],
        recession_risk: float,
    ) -> Dict[str, float]:
        """
        Adjust sector tilts based on recession risk.

        Args:
            tilts: Base sector tilts
            recession_risk: Recession probability (0-1)

        Returns:
            Adjusted sector tilts
        """
        if recession_risk < 0.20:
            return tilts

        adjusted = tilts.copy()
        defensive = self.get_defensive_sectors()
        cyclical = self.get_cyclical_sectors()

        # Scale adjustment by recession risk
        adjustment_factor = min(1.0, (recession_risk - 0.20) * 2)

        for sector in defensive:
            current = adjusted.get(sector, 0)
            adjusted[sector] = current + 0.02 * adjustment_factor

        for sector in cyclical:
            current = adjusted.get(sector, 0)
            adjusted[sector] = current - 0.02 * adjustment_factor

        return adjusted

    def get_sector_beta(self, sector: str) -> str:
        """Get beta characteristic for a sector."""
        return self.SECTOR_CHARACTERISTICS.get(sector, {}).get("beta", "medium")

    def calculate_portfolio_beta_tilt(
        self,
        weights: Dict[str, float],
    ) -> str:
        """
        Calculate overall portfolio beta tilt from sector weights.

        Args:
            weights: Dictionary of sector weights

        Returns:
            "aggressive", "neutral", or "defensive"
        """
        beta_score = 0.0
        total_weight = 0.0

        beta_map = {"high": 1.0, "medium": 0.0, "low": -1.0}

        for sector, weight in weights.items():
            beta = self.get_sector_beta(sector)
            beta_score += beta_map.get(beta, 0.0) * weight
            total_weight += weight

        if total_weight > 0:
            avg_beta = beta_score / total_weight
        else:
            avg_beta = 0.0

        if avg_beta > 0.3:
            return "aggressive"
        elif avg_beta < -0.3:
            return "defensive"
        return "neutral"
