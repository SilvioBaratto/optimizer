"""
Quality Filter - Filter stocks by quantitative quality metrics.
"""

from typing import Tuple, Optional, Dict, Any

from optimizer.config.quality_filter_config import QualityFilterConfig
from optimizer.domain.models.stock_signal import StockSignalDTO
from optimizer.src.risk_management.filters.protocol import StockFilterImpl


class QualityFilterImpl(StockFilterImpl):
    """
    Implementation of quality-based stock filtering.

    Filters stocks based on risk-adjusted returns, risk metrics,
    and quality scores. Uses externalized configuration.

    Filter criteria:
    - Sharpe ratio (risk-adjusted return)
    - Sortino ratio (downside risk)
    - Calmar ratio (return/drawdown)
    - Information ratio (alpha consistency)
    - Volatility (price stability)
    - Max drawdown (tail risk)
    - Quality score (BAML fundamentals)
    - Dollar volume (liquidity)
    - Beta (market sensitivity)
    - Data quality (reliability)
    """

    def __init__(self, config: Optional[QualityFilterConfig] = None):
        """
        Initialize quality filter.

        Args:
            config: QualityFilterConfig with thresholds (defaults if None)
        """
        super().__init__()
        self._config = config or QualityFilterConfig()

    @property
    def name(self) -> str:
        """Filter name."""
        return "QualityFilter"

    @property
    def config(self) -> QualityFilterConfig:
        """Get filter configuration."""
        return self._config

    def _filter_single(self, signal: StockSignalDTO) -> Tuple[bool, Optional[str]]:
        """
        Apply all quality filters to a single signal.

        Returns on first failure for efficiency.
        """
        # Sharpe ratio (required)
        if signal.sharpe_ratio is None:
            return False, "missing_sharpe"
        if signal.sharpe_ratio < self._config.min_sharpe_ratio:
            return False, f"sharpe_ratio({signal.sharpe_ratio:.2f}<{self._config.min_sharpe_ratio})"

        # Sortino ratio (if configured)
        if self._config.min_sortino_ratio is not None:
            if signal.sortino_ratio is None:
                return False, "missing_sortino"
            if signal.sortino_ratio < self._config.min_sortino_ratio:
                return False, f"sortino_ratio({signal.sortino_ratio:.2f}<{self._config.min_sortino_ratio})"

        # Calmar ratio (if configured)
        if self._config.min_calmar_ratio is not None:
            if signal.calmar_ratio is None:
                return False, "missing_calmar"
            if signal.calmar_ratio < self._config.min_calmar_ratio:
                return False, f"calmar_ratio({signal.calmar_ratio:.2f}<{self._config.min_calmar_ratio})"

        # Information ratio (if configured)
        if self._config.min_information_ratio is not None:
            if signal.information_ratio is None:
                return False, "missing_information_ratio"
            if signal.information_ratio < self._config.min_information_ratio:
                return False, f"information_ratio({signal.information_ratio:.2f}<{self._config.min_information_ratio})"

        # Volatility
        if signal.volatility is not None:
            if signal.volatility > self._config.max_volatility:
                return False, f"volatility({signal.volatility:.2f}>{self._config.max_volatility})"

        # Max drawdown
        if signal.max_drawdown is not None:
            if signal.max_drawdown < self._config.max_max_drawdown:
                return False, f"max_drawdown({signal.max_drawdown:.2f}<{self._config.max_max_drawdown})"

        # Quality score (if configured)
        if self._config.min_quality_score is not None:
            if signal.quality_score is None:
                return False, "missing_quality_score"
            if signal.quality_score < self._config.min_quality_score:
                return False, f"quality_score({signal.quality_score:.2f}<{self._config.min_quality_score})"

        # Dollar volume (if configured)
        if self._config.min_daily_dollar_volume is not None:
            dollar_volume = signal.daily_dollar_volume
            if dollar_volume is None:
                return False, "missing_dollar_volume"
            if dollar_volume < self._config.min_daily_dollar_volume:
                return False, f"dollar_volume(${dollar_volume/1e6:.1f}M<${self._config.min_daily_dollar_volume/1e6:.0f}M)"

        # Price
        if signal.close_price is None:
            return False, "missing_price"
        if signal.close_price < self._config.min_close_price:
            return False, f"price(${signal.close_price:.2f}<${self._config.min_close_price})"

        # Beta (if configured)
        if self._config.max_beta is not None:
            if signal.beta is not None and signal.beta > self._config.max_beta:
                return False, f"beta({signal.beta:.2f}>{self._config.max_beta})"

        # Alpha (if required)
        if self._config.require_positive_alpha:
            if signal.alpha is None:
                return False, "missing_alpha"
            if signal.alpha <= 0:
                return False, f"negative_alpha({signal.alpha:.4f})"

        # Data quality
        if signal.data_quality_score is not None:
            if signal.data_quality_score < self._config.min_data_quality_score:
                return False, f"data_quality({signal.data_quality_score:.2f}<{self._config.min_data_quality_score})"

        return True, None

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of filter configuration."""
        return {
            "name": self.name,
            "min_sharpe_ratio": self._config.min_sharpe_ratio,
            "min_sortino_ratio": self._config.min_sortino_ratio,
            "min_calmar_ratio": self._config.min_calmar_ratio,
            "min_information_ratio": self._config.min_information_ratio,
            "max_volatility": f"{self._config.max_volatility:.0%}",
            "max_max_drawdown": f"{self._config.max_max_drawdown:.0%}",
            "min_quality_score": self._config.min_quality_score,
            "min_dollar_volume": f"${self._config.min_daily_dollar_volume/1e6:.0f}M" if self._config.min_daily_dollar_volume else None,
            "min_price": f"${self._config.min_close_price:.2f}",
            "max_beta": self._config.max_beta,
            "require_positive_alpha": self._config.require_positive_alpha,
        }

    @classmethod
    def for_regime(cls, regime: str) -> "QualityFilterImpl":
        """
        Create regime-appropriate quality filter.

        Args:
            regime: Macro regime (early_cycle, mid_cycle, late_cycle, recession)

        Returns:
            QualityFilterImpl with regime-appropriate configuration
        """
        config = QualityFilterConfig.for_regime(regime)
        return cls(config=config)
