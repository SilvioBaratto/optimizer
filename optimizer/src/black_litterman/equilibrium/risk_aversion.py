"""
Risk Aversion Estimation - Estimate market risk aversion coefficient.
"""

import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskAversionEstimatorImpl:
    """
    Estimate market risk aversion coefficient (δ) for Black-Litterman.

    Risk aversion controls the balance between the equilibrium returns
    and investor views. Higher δ means:
    - More weight on equilibrium (market expectations)
    - Less aggressive deviations from market weights

    Typical values:
    - δ = 2.5-3.5 for normal markets
    - δ = 4.0+ for risk-off periods
    - δ = 2.0 or less for risk-on periods

    Methods:
    1. Historical: Derived from market Sharpe ratio
    2. Implied: Reverse-optimized from benchmark
    3. Fixed: Use preset values
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def estimate(
        self,
        market_returns: pd.Series,
        risk_free_rate: float,
        method: str = "historical",
    ) -> float:
        """
        Estimate the implied market risk aversion.

        Args:
            market_returns: Historical market returns (e.g., S&P 500)
            risk_free_rate: Risk-free rate (annualized, decimal)
            method: Estimation method ('historical', 'implied')

        Returns:
            Risk aversion coefficient (typically 1.5 to 5.0)
        """
        if method == "historical":
            return self._estimate_historical(market_returns, risk_free_rate)
        elif method == "implied":
            return self._estimate_implied(market_returns, risk_free_rate)
        else:
            self._logger.warning(f"Unknown method {method}, using historical")
            return self._estimate_historical(market_returns, risk_free_rate)

    def _estimate_historical(
        self,
        market_returns: pd.Series,
        risk_free_rate: float,
    ) -> float:
        """
        Estimate risk aversion from historical market Sharpe ratio.

        Formula: δ = (E[R_m] - r_f) / σ²_m

        This assumes the market portfolio is optimal, so the market
        Sharpe ratio equals the risk aversion times market volatility.
        """
        # Annualize returns
        if len(market_returns) > 0:
            mean_return = market_returns.mean() * 252
            std_return = market_returns.std() * np.sqrt(252)
        else:
            self._logger.warning("No market returns provided, using defaults")
            return 2.5

        # Calculate variance
        variance = std_return ** 2

        if variance <= 0:
            self._logger.warning("Market variance is zero or negative, using default δ=2.5")
            return 2.5

        # Risk aversion = excess return / variance
        excess_return = mean_return - risk_free_rate
        delta = excess_return / variance

        # Bound to reasonable range
        delta = max(0.5, min(delta, 8.0))

        self._logger.info(
            f"Historical risk aversion: δ = {delta:.2f} "
            f"(market return={mean_return:.2%}, vol={std_return:.2%})"
        )

        return delta

    def _estimate_implied(
        self,
        market_returns: pd.Series,
        risk_free_rate: float,
    ) -> float:
        """
        Estimate implied risk aversion using market premium assumption.

        Uses a target equity risk premium (typically 4-6%) to derive δ.
        """
        # Assume a reasonable equity risk premium
        TARGET_ERP = 0.05  # 5% equity risk premium

        # Market volatility
        if len(market_returns) > 0:
            std_return = market_returns.std() * np.sqrt(252)
        else:
            std_return = 0.16  # Typical long-run market vol

        variance = std_return ** 2

        if variance <= 0:
            return 2.5

        # δ = ERP / σ²
        delta = TARGET_ERP / variance

        # Bound to reasonable range
        delta = max(0.5, min(delta, 8.0))

        self._logger.info(f"Implied risk aversion: δ = {delta:.2f} (assuming ERP={TARGET_ERP:.1%})")

        return delta


class RegimeAdjustedRiskAversion:
    """
    Adjust risk aversion based on macro regime.

    Different regimes warrant different risk appetites:
    - Early cycle: Lower risk aversion (more aggressive)
    - Mid cycle: Neutral risk aversion
    - Late cycle: Higher risk aversion (more defensive)
    - Recession: Much higher risk aversion
    """

    # Regime multipliers for risk aversion
    REGIME_MULTIPLIERS = {
        "early_cycle": 0.85,
        "mid_cycle": 1.0,
        "late_cycle": 1.3,
        "recession": 1.6,
        "uncertain": 1.15,
    }

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def adjust_for_regime(
        self,
        base_delta: float,
        regime: str,
        recession_risk: Optional[float] = None,
    ) -> float:
        """
        Adjust risk aversion based on macro regime.

        Args:
            base_delta: Base risk aversion estimate
            regime: Macro regime (early_cycle, mid_cycle, late_cycle, recession)
            recession_risk: Optional recession probability (0-1)

        Returns:
            Adjusted risk aversion coefficient
        """
        regime_lower = regime.lower().replace(" ", "_")
        multiplier = self.REGIME_MULTIPLIERS.get(regime_lower, 1.0)

        # Additional adjustment for recession risk
        if recession_risk is not None and recession_risk > 0.3:
            # Increase risk aversion as recession probability rises
            risk_adjustment = 1.0 + (recession_risk - 0.3) * 0.5
            multiplier *= risk_adjustment
            self._logger.debug(f"Recession risk adjustment: {risk_adjustment:.2f}")

        adjusted_delta = base_delta * multiplier

        # Bound to reasonable range
        adjusted_delta = max(1.0, min(adjusted_delta, 10.0))

        self._logger.info(
            f"Regime-adjusted risk aversion: {base_delta:.2f} × {multiplier:.2f} = {adjusted_delta:.2f} "
            f"(regime={regime})"
        )

        return adjusted_delta

    def get_by_country(
        self,
        country: str,
        regime: str,
        base_delta: Optional[float] = None,
    ) -> float:
        """
        Get risk aversion for a specific country and regime.

        Different countries have different risk premiums:
        - US: Lower risk aversion (deep, liquid markets)
        - Europe: Moderate risk aversion
        - Emerging: Higher risk aversion

        Args:
            country: Country code
            regime: Macro regime
            base_delta: Optional base risk aversion (default calculated by country)

        Returns:
            Country and regime-adjusted risk aversion
        """
        # Country base risk aversion
        country_base = {
            "USA": 2.5,
            "UK": 2.8,
            "Germany": 3.0,
            "France": 3.0,
            "Japan": 3.5,
            "Canada": 2.6,
            "Australia": 2.8,
        }

        if base_delta is None:
            base_delta = country_base.get(country, 3.0)

        return self.adjust_for_regime(base_delta, regime)

    def get_multi_country(
        self,
        weights_by_country: Dict[str, float],
        regime: str,
    ) -> float:
        """
        Get weighted average risk aversion for multi-country portfolio.

        Args:
            weights_by_country: Portfolio weights by country
            regime: Macro regime

        Returns:
            Weighted risk aversion
        """
        total_weight = sum(weights_by_country.values())
        if total_weight <= 0:
            return 2.5

        weighted_delta = 0.0
        for country, weight in weights_by_country.items():
            country_delta = self.get_by_country(country, regime)
            weighted_delta += (weight / total_weight) * country_delta

        return weighted_delta
