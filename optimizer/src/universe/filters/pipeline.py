"""
Filter Pipeline - Composite filter that chains multiple filters.

Implements the Composite pattern to treat a collection of filters as a single filter.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List

from optimizer.domain.protocols.universe import InstrumentFilter


@dataclass
class FilterPipelineImpl:
    """
    Composite filter that chains multiple filters together.

    Filters are applied in the order they are added. A stock must pass
    ALL filters to be included in the universe. On first failure, the
    pipeline short-circuits and returns the rejection reason.

    Tracks statistics for each filter:
    - passed: Number of stocks that passed this filter
    - failed: Number of stocks that failed this filter

    Example:
        pipeline = FilterPipelineImpl()
        pipeline.add_filter(MarketCapFilter(config))
        pipeline.add_filter(PriceFilter(config))
        pipeline.add_filter(LiquidityFilter(config))

        passed, reason = pipeline.apply(data, "AAPL")

        # Get statistics
        stats = pipeline.get_summary()
        for filter_name, counts in stats.items():
            print(f"{filter_name}: {counts['passed']} passed, {counts['failed']} failed")
    """

    _filters: List[InstrumentFilter] = field(default_factory=list)
    _stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add_filter(self, filter: InstrumentFilter) -> "FilterPipelineImpl":
        """
        Add a filter to the pipeline.

        Filters are applied in the order they are added.

        Args:
            filter: InstrumentFilter to add

        Returns:
            Self for method chaining
        """
        self._filters.append(filter)
        self._stats[filter.name] = {"passed": 0, "failed": 0}
        return self

    def apply(self, data: Dict[str, Any], yf_ticker: str) -> Tuple[bool, str]:
        """
        Apply all filters to instrument data.

        Short-circuits on first failure - remaining filters are not evaluated.

        Args:
            data: yfinance info dictionary
            yf_ticker: Yahoo Finance ticker symbol

        Returns:
            Tuple of (passed, reason):
            - passed: True if instrument passes ALL filters
            - reason: "Passed all filters (details)" or "[FilterName] rejection reason"
        """
        if not self._filters:
            return True, "No filters configured"

        for f in self._filters:
            passed, reason = f.filter(data, yf_ticker)

            if not passed:
                self._stats[f.name]["failed"] += 1
                return False, f"[{f.name}] {reason}"

            self._stats[f.name]["passed"] += 1

        return True, "Passed all filters"

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for all filters in the pipeline.

        Returns:
            Dictionary mapping filter_name -> {"passed": count, "failed": count}
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset all filter statistics to zero."""
        for name in self._stats:
            self._stats[name] = {"passed": 0, "failed": 0}

    def get_filters(self) -> List[InstrumentFilter]:
        """
        Get list of all filters in the pipeline.

        Returns:
            List of InstrumentFilter objects in execution order
        """
        return self._filters.copy()

    def get_pipeline_summary(self) -> str:
        """
        Get a human-readable summary of the pipeline configuration and statistics.

        Returns:
            Multi-line string describing all filters and their stats
        """
        lines = ["Filter Pipeline Summary:", "=" * 50]

        for f in self._filters:
            stats = self._stats.get(f.name, {"passed": 0, "failed": 0})
            total = stats["passed"] + stats["failed"]
            pass_rate = (stats["passed"] / total * 100) if total > 0 else 0
            lines.append(
                f"  {f.name}: {stats['passed']} passed, "
                f"{stats['failed']} failed ({pass_rate:.1f}% pass rate)"
            )

        total_passed = min(
            (self._stats.get(f.name, {}).get("passed", 0) for f in self._filters),
            default=0,
        )
        lines.append("=" * 50)
        lines.append(f"Final: {total_passed} stocks passed all filters")

        return "\n".join(lines)
