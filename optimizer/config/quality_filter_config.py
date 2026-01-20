"""
Quality Filter Configuration - Externalized thresholds for quality filtering.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class QualityFilterConfig:
    """
    Configuration for quality-based stock filtering.

    All thresholds and parameters for the QualityFilter are externalized here.
    This enables easy testing and environment-based configuration.

    Attributes:
        # Risk-adjusted return thresholds
        min_sharpe_ratio: Minimum Sharpe ratio (risk-adjusted return)
        min_sortino_ratio: Minimum Sortino ratio (downside risk adjusted)
        min_calmar_ratio: Minimum Calmar ratio (return/drawdown)
        min_information_ratio: Minimum Information ratio (alpha consistency)

        # Risk thresholds
        max_volatility: Maximum annualized volatility (decimal, e.g., 0.40 for 40%)
        max_max_drawdown: Maximum drawdown (negative decimal, e.g., -0.30 for -30%)
        max_beta: Maximum market beta (None to disable)

        # Quality thresholds
        min_quality_score: Minimum BAML quality score (0-1 scale)
        min_data_quality_score: Minimum data quality score (0-1 scale)

        # Liquidity thresholds
        min_daily_dollar_volume: Minimum daily dollar volume ($)
        min_close_price: Minimum stock price ($)

        # Alpha requirement
        require_positive_alpha: Whether to require positive alpha
    """

    # Risk-adjusted return thresholds
    min_sharpe_ratio: float = 0.5
    min_sortino_ratio: Optional[float] = 0.6
    min_calmar_ratio: Optional[float] = 0.8
    min_information_ratio: Optional[float] = 0.4

    # Risk thresholds
    max_volatility: float = 0.40
    max_max_drawdown: float = -0.30
    max_beta: Optional[float] = None

    # Quality thresholds
    min_quality_score: Optional[float] = 0.55
    min_data_quality_score: float = 0.6

    # Liquidity thresholds
    min_daily_dollar_volume: Optional[float] = 10_000_000  # $10M
    min_close_price: float = 5.0

    # Alpha requirement
    require_positive_alpha: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "min_sharpe_ratio": self.min_sharpe_ratio,
            "min_sortino_ratio": self.min_sortino_ratio,
            "min_calmar_ratio": self.min_calmar_ratio,
            "min_information_ratio": self.min_information_ratio,
            "max_volatility": self.max_volatility,
            "max_max_drawdown": self.max_max_drawdown,
            "max_beta": self.max_beta,
            "min_quality_score": self.min_quality_score,
            "min_data_quality_score": self.min_data_quality_score,
            "min_daily_dollar_volume": self.min_daily_dollar_volume,
            "min_close_price": self.min_close_price,
            "require_positive_alpha": self.require_positive_alpha,
        }

    @classmethod
    def from_env(cls) -> "QualityFilterConfig":
        """
        Create configuration from environment variables.

        Environment variables:
        - QUALITY_MIN_SHARPE
        - QUALITY_MIN_SORTINO
        - QUALITY_MIN_CALMAR
        - QUALITY_MIN_IR
        - QUALITY_MAX_VOLATILITY
        - QUALITY_MAX_DRAWDOWN
        - QUALITY_MAX_BETA
        - QUALITY_MIN_QUALITY_SCORE
        - QUALITY_MIN_DATA_QUALITY
        - QUALITY_MIN_DOLLAR_VOLUME
        - QUALITY_MIN_PRICE
        - QUALITY_REQUIRE_ALPHA
        """

        def get_float(name: str, default: Optional[float]) -> Optional[float]:
            val = os.getenv(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                return default

        def get_bool(name: str, default: bool) -> bool:
            val = os.getenv(name)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        return cls(
            min_sharpe_ratio=get_float("QUALITY_MIN_SHARPE", 0.5) or 0.5,
            min_sortino_ratio=get_float("QUALITY_MIN_SORTINO", 0.6),
            min_calmar_ratio=get_float("QUALITY_MIN_CALMAR", 0.8),
            min_information_ratio=get_float("QUALITY_MIN_IR", 0.4),
            max_volatility=get_float("QUALITY_MAX_VOLATILITY", 0.40) or 0.40,
            max_max_drawdown=get_float("QUALITY_MAX_DRAWDOWN", -0.30) or -0.30,
            max_beta=get_float("QUALITY_MAX_BETA", None),
            min_quality_score=get_float("QUALITY_MIN_QUALITY_SCORE", 0.55),
            min_data_quality_score=get_float("QUALITY_MIN_DATA_QUALITY", 0.6) or 0.6,
            min_daily_dollar_volume=get_float("QUALITY_MIN_DOLLAR_VOLUME", 10_000_000),
            min_close_price=get_float("QUALITY_MIN_PRICE", 5.0) or 5.0,
            require_positive_alpha=get_bool("QUALITY_REQUIRE_ALPHA", False),
        )

    @classmethod
    def conservative(cls) -> "QualityFilterConfig":
        """Create conservative configuration with stricter thresholds."""
        return cls(
            min_sharpe_ratio=0.7,
            min_sortino_ratio=0.8,
            min_calmar_ratio=1.0,
            min_information_ratio=0.5,
            max_volatility=0.30,
            max_max_drawdown=-0.20,
            max_beta=1.2,
            min_quality_score=0.65,
            min_data_quality_score=0.7,
            min_daily_dollar_volume=20_000_000,
            min_close_price=10.0,
            require_positive_alpha=True,
        )

    @classmethod
    def aggressive(cls) -> "QualityFilterConfig":
        """Create aggressive configuration with looser thresholds."""
        return cls(
            min_sharpe_ratio=0.3,
            min_sortino_ratio=0.4,
            min_calmar_ratio=0.5,
            min_information_ratio=0.2,
            max_volatility=0.50,
            max_max_drawdown=-0.40,
            max_beta=None,
            min_quality_score=0.40,
            min_data_quality_score=0.5,
            min_daily_dollar_volume=5_000_000,
            min_close_price=3.0,
            require_positive_alpha=False,
        )

    @classmethod
    def for_regime(cls, regime: str) -> "QualityFilterConfig":
        """
        Create regime-appropriate configuration.

        Args:
            regime: Macro regime (early_cycle, mid_cycle, late_cycle, recession)

        Returns:
            Regime-appropriate QualityFilterConfig
        """
        if regime == "recession":
            # Very conservative in recession
            return cls(
                min_sharpe_ratio=0.8,
                min_sortino_ratio=0.9,
                min_calmar_ratio=1.2,
                min_information_ratio=0.5,
                max_volatility=0.25,
                max_max_drawdown=-0.15,
                max_beta=0.8,
                min_quality_score=0.70,
                min_data_quality_score=0.7,
                min_daily_dollar_volume=25_000_000,
                min_close_price=15.0,
                require_positive_alpha=True,
            )
        elif regime == "late_cycle":
            # More conservative in late cycle
            return cls(
                min_sharpe_ratio=0.6,
                min_sortino_ratio=0.7,
                min_calmar_ratio=1.0,
                min_information_ratio=0.4,
                max_volatility=0.35,
                max_max_drawdown=-0.25,
                max_beta=1.0,
                min_quality_score=0.60,
                min_data_quality_score=0.65,
                min_daily_dollar_volume=15_000_000,
                min_close_price=10.0,
                require_positive_alpha=True,
            )
        elif regime == "early_cycle":
            # More aggressive in early cycle
            return cls(
                min_sharpe_ratio=0.4,
                min_sortino_ratio=0.5,
                min_calmar_ratio=0.6,
                min_information_ratio=0.3,
                max_volatility=0.45,
                max_max_drawdown=-0.35,
                max_beta=1.5,
                min_quality_score=0.45,
                min_data_quality_score=0.55,
                min_daily_dollar_volume=8_000_000,
                min_close_price=5.0,
                require_positive_alpha=False,
            )
        else:
            # Default (mid_cycle or uncertain)
            return cls()
