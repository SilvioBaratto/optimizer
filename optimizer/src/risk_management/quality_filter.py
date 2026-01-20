#!/usr/bin/env python3
from typing import List, Tuple, Optional
from dataclasses import dataclass

from optimizer.database.models.stock_signals import StockSignal
from optimizer.database.models.universe import Instrument


@dataclass
class QualityMetrics:
    """Enhanced quality filter statistics."""

    total_signals: int
    passed_sharpe: int
    passed_sortino: int
    passed_information_ratio: int
    passed_calmar: int
    passed_volatility: int
    passed_drawdown: int
    passed_price: int
    passed_alpha: int
    passed_quality_score: int
    passed_dollar_volume: int
    passed_beta: int
    passed_data_quality: int
    passed_all: int

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        return (self.passed_all / self.total_signals) if self.total_signals > 0 else 0.0


class QualityFilter:
    """
    Enhanced quality filter for concentrated portfolios.
    """

    def __init__(
        self,
        min_sharpe_ratio: float = 0.5,
        max_volatility: float = 0.40,
        max_max_drawdown: float = -0.30,
        min_close_price: float = 5.0,
        require_positive_alpha: bool = False,
        min_data_quality_score: float = 0.6,
        min_sortino_ratio: Optional[float] = 0.6,  # Downside risk focus
        min_information_ratio: Optional[float] = 0.4,  # Alpha consistency
        min_daily_dollar_volume: Optional[float] = 10_000_000,  # $10M liquidity
        min_quality_score: Optional[float] = 0.55,  # BAML business quality (0-1 scale)
        min_calmar_ratio: Optional[float] = 0.8,  # Return/MaxDrawdown
        max_beta: Optional[float] = None,  # Market sensitivity cap (regime-dependent)
    ):
        """
        Initialize enhanced quality filter.
        """
        # Existing
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_volatility = max_volatility
        self.max_max_drawdown = max_max_drawdown
        self.min_close_price = min_close_price
        self.require_positive_alpha = require_positive_alpha
        self.min_data_quality_score = min_data_quality_score

        # Phase 1 (Essential)
        self.min_sortino_ratio = min_sortino_ratio
        self.min_information_ratio = min_information_ratio
        self.min_daily_dollar_volume = min_daily_dollar_volume
        self.min_quality_score = min_quality_score

        # Phase 2 (Valuable)
        self.min_calmar_ratio = min_calmar_ratio
        self.max_beta = max_beta

    def passes_sharpe_filter(self, signal: StockSignal) -> bool:
        """Check if stock meets minimum Sharpe ratio (risk-adjusted return)."""
        if signal.sharpe_ratio is None:
            return False  # Conservative: reject if missing
        return signal.sharpe_ratio >= self.min_sharpe_ratio

    def passes_volatility_filter(self, signal: StockSignal) -> bool:
        """Check if stock volatility is below maximum (stability check)."""
        if signal.volatility is None:
            return True  # Allow if missing (lenient)
        return signal.volatility <= self.max_volatility

    def passes_drawdown_filter(self, signal: StockSignal) -> bool:
        """Check if max drawdown is within acceptable range."""
        if signal.max_drawdown is None:
            return True  # Allow if missing (lenient)
        # max_drawdown is negative, so check if it's greater than threshold
        # e.g., -0.25 > -0.30 passes (smaller drawdown is better)
        return signal.max_drawdown >= self.max_max_drawdown

    def passes_price_filter(self, signal: StockSignal) -> bool:
        """Check minimum price to avoid penny stocks."""
        if signal.close_price is None:
            return False  # Conservative: reject if missing
        return signal.close_price >= self.min_close_price

    def passes_alpha_filter(self, signal: StockSignal) -> bool:
        """Check if stock has positive alpha (excess return)."""
        if not self.require_positive_alpha:
            return True

        if signal.alpha is None:
            return False  # Conservative: reject if missing when required
        return signal.alpha > 0

    def passes_data_quality_filter(self, signal: StockSignal) -> bool:
        """Check if signal has sufficient data quality."""
        if signal.data_quality_score is None:
            return True  # Allow if missing (lenient)
        return signal.data_quality_score >= self.min_data_quality_score

    # ========== NEW ENHANCED FILTERS ==========

    def passes_sortino_filter(self, signal: StockSignal) -> bool:
        """
        Check if stock meets minimum Sortino ratio.

        Sortino focuses on DOWNSIDE risk (more relevant than Sharpe's total risk).
        Better for asymmetric return distributions common in stocks.

        STRICT: Rejects if missing (critical risk metric).
        """
        if self.min_sortino_ratio is None:
            return True  # Filter disabled

        if signal.sortino_ratio is None:
            return False  # STRICT: Reject if missing (critical metric)

        return signal.sortino_ratio >= self.min_sortino_ratio

    def passes_information_ratio_filter(self, signal: StockSignal) -> bool:
        """
        Check if stock has consistent alpha generation.
        """
        if self.min_information_ratio is None:
            return True  # Filter disabled

        if signal.information_ratio is None:
            return False  # STRICT: Reject if missing (critical metric)

        return signal.information_ratio >= self.min_information_ratio

    def passes_calmar_filter(self, signal: StockSignal) -> bool:
        """
        Check if stock has acceptable return/drawdown ratio.
        """
        if self.min_calmar_ratio is None:
            return True  # Filter disabled

        if signal.calmar_ratio is None:
            return False  # STRICT: Reject if missing (critical metric)

        return signal.calmar_ratio >= self.min_calmar_ratio

    def passes_quality_score_filter(self, signal: StockSignal) -> bool:
        """
        Check if stock has minimum business quality score.
        """
        if self.min_quality_score is None:
            return True  # Filter disabled

        if signal.quality_score is None:
            return False  # STRICT: Reject if missing (critical metric)

        return signal.quality_score >= self.min_quality_score

    def passes_dollar_volume_filter(self, signal: StockSignal) -> bool:
        """
        Check if stock has minimum daily dollar volume.
        """
        if self.min_daily_dollar_volume is None:
            return True  # Filter disabled

        if signal.volume is None or signal.close_price is None:
            return False  # STRICT: Reject if missing (critical metric)

        # Calculate dollar volume (shares * price)
        daily_dollar_volume = signal.volume * signal.close_price

        return daily_dollar_volume >= self.min_daily_dollar_volume

    def passes_beta_filter(self, signal: StockSignal) -> bool:
        """
        Check if stock beta is within acceptable range.
        """
        if self.max_beta is None:
            return True  # Filter disabled

        if signal.beta is None:
            return True  # LENIENT: Allow if missing (situational filter)

        return signal.beta <= self.max_beta

    def passes_all_filters(self, signal: StockSignal) -> Tuple[bool, List[str]]:
        """
        Check if signal passes all quality filters.
        """
        failed_filters = []

        # Existing filters
        if not self.passes_sharpe_filter(signal):
            failed_filters.append("sharpe_ratio")

        if not self.passes_volatility_filter(signal):
            failed_filters.append("volatility")

        if not self.passes_drawdown_filter(signal):
            failed_filters.append("max_drawdown")

        if not self.passes_price_filter(signal):
            failed_filters.append("price")

        if not self.passes_alpha_filter(signal):
            failed_filters.append("alpha")

        if not self.passes_data_quality_filter(signal):
            failed_filters.append("data_quality")

        # NEW enhanced filters
        if not self.passes_sortino_filter(signal):
            failed_filters.append("sortino_ratio")

        if not self.passes_information_ratio_filter(signal):
            failed_filters.append("information_ratio")

        if not self.passes_calmar_filter(signal):
            failed_filters.append("calmar_ratio")

        if not self.passes_quality_score_filter(signal):
            failed_filters.append("quality_score")

        if not self.passes_dollar_volume_filter(signal):
            failed_filters.append("dollar_volume")

        if not self.passes_beta_filter(signal):
            failed_filters.append("beta")

        return (len(failed_filters) == 0, failed_filters)

    def filter_signals(
        self, signals: List[Tuple[StockSignal, Instrument]]
    ) -> List[Tuple[StockSignal, Instrument]]:
        """
        Filter signals by enhanced quality criteria.
        """
        passed_all_list = []

        for signal, instrument in signals:
            # Check all filters
            passes, _ = self.passes_all_filters(signal)
            if passes:
                passed_all_list.append((signal, instrument))

        return passed_all_list

    def get_filter_summary(self) -> str:
        """Get summary string of ALL filter criteria."""
        summary_lines = ["Enhanced Quality Filters (v2.0):"]
        summary_lines.append("  EXISTING:")
        summary_lines.append(f"    - Sharpe ratio >= {self.min_sharpe_ratio:.2f}")
        summary_lines.append(f"    - Volatility <= {self.max_volatility:.0%}")
        summary_lines.append(f"    - Max drawdown >= {self.max_max_drawdown:.0%}")
        summary_lines.append(f"    - Price >= ${self.min_close_price:.2f}")
        summary_lines.append(f"    - Positive alpha required: {self.require_positive_alpha}")
        summary_lines.append(f"    - Data quality >= {self.min_data_quality_score:.1f}")

        summary_lines.append("  PHASE 1 ENHANCEMENTS (HIGH PRIORITY):")
        if self.min_sortino_ratio is not None:
            summary_lines.append(f"    - Sortino ratio >= {self.min_sortino_ratio:.1f}")
        if self.min_information_ratio is not None:
            summary_lines.append(f"    - Information ratio >= {self.min_information_ratio:.1f}")
        if self.min_daily_dollar_volume is not None:
            summary_lines.append(f"    - Dollar volume >= ${self.min_daily_dollar_volume/1e6:.0f}M")
        if self.min_quality_score is not None:
            summary_lines.append(
                f"    - Quality score >= {self.min_quality_score:.2f} ({self.min_quality_score*100:.0f}%)"
            )

        summary_lines.append("  PHASE 2 ENHANCEMENTS (MEDIUM PRIORITY):")
        if self.min_calmar_ratio is not None:
            summary_lines.append(f"    - Calmar ratio >= {self.min_calmar_ratio:.1f}")
        if self.max_beta is not None:
            summary_lines.append(f"    - Beta <= {self.max_beta:.1f}")

        return "\n".join(summary_lines)
