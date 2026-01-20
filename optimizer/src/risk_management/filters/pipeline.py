"""
Filter Pipeline - Compose multiple filters into a processing pipeline.
"""

from typing import List, Tuple, Dict, Any, Optional

from optimizer.domain.models.stock_signal import StockSignalDTO
from optimizer.src.risk_management.filters.protocol import StockFilterImpl


class FilterPipelineImpl:
    """
    Implementation of the FilterPipeline protocol.

    Composes multiple filters into a pipeline where signals must pass
    all filters to be included in the output.

    Features:
    - Ordered filter execution
    - Detailed statistics per filter
    - Early termination (stop filtering a signal on first failure)

    Example:
        pipeline = FilterPipelineImpl()
        pipeline.add_filter(QualityFilterImpl(config))
        pipeline.add_filter(AffordabilityFilter(max_price=75.0))
        pipeline.add_filter(SectorFilter(max_sector_count=3))

        passed, stats = pipeline.filter_batch(signals)
    """

    def __init__(self, name: str = "FilterPipeline"):
        """
        Initialize filter pipeline.

        Args:
            name: Name for identification
        """
        self._name = name
        self._filters: List[StockFilterImpl] = []

    @property
    def name(self) -> str:
        """Pipeline name."""
        return self._name

    def add_filter(self, filter: StockFilterImpl) -> "FilterPipelineImpl":
        """
        Add a filter to the pipeline.

        Filters are applied in the order they are added.

        Args:
            filter: StockFilter to add

        Returns:
            Self for chaining
        """
        self._filters.append(filter)
        return self

    def remove_filter(self, filter_name: str) -> bool:
        """
        Remove a filter by name.

        Args:
            filter_name: Name of filter to remove

        Returns:
            True if filter was found and removed
        """
        for i, f in enumerate(self._filters):
            if f.name == filter_name:
                self._filters.pop(i)
                return True
        return False

    def get_filters(self) -> List[StockFilterImpl]:
        """
        Get list of all filters in the pipeline.

        Returns:
            List of StockFilter objects in execution order
        """
        return list(self._filters)

    def filter_single(
        self,
        signal: StockSignalDTO,
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        """
        Apply all filters to a single signal.

        Args:
            signal: Signal to filter

        Returns:
            Tuple of (passes, results):
            - passes: True if signal passes all filters
            - results: Dictionary of {filter_name: rejection_reason or None}
        """
        results: Dict[str, Optional[str]] = {}
        all_passed = True

        for f in self._filters:
            passes, reason = f.filter(signal)
            results[f.name] = reason

            if not passes:
                all_passed = False
                # Continue to collect all failures (don't early terminate)

        return all_passed, results

    def filter_batch(
        self,
        signals: List[StockSignalDTO],
        early_terminate: bool = True,
    ) -> Tuple[List[StockSignalDTO], Dict[str, Dict[str, int]]]:
        """
        Apply all filters to a batch of signals.

        Args:
            signals: List of StockSignalDTO to evaluate
            early_terminate: If True, stop filtering a signal on first failure

        Returns:
            Tuple of (passed_signals, stats):
            - passed_signals: Signals that passed all filters
            - stats: Dictionary of {filter_name: {reason: count}}
        """
        if not signals:
            return [], {}

        passed_signals: List[StockSignalDTO] = []
        stats: Dict[str, Dict[str, int]] = {f.name: {} for f in self._filters}

        # Track signals at each stage
        current_signals = signals
        stage_counts = [len(signals)]

        for filter_idx, f in enumerate(self._filters):
            next_signals: List[StockSignalDTO] = []

            for signal in current_signals:
                passes, reason = f.filter(signal)

                if passes:
                    next_signals.append(signal)
                else:
                    reason_key = reason or "unknown"
                    stats[f.name][reason_key] = stats[f.name].get(reason_key, 0) + 1

            current_signals = next_signals
            stage_counts.append(len(current_signals))

        passed_signals = current_signals

        return passed_signals, stats

    def filter_batch_parallel(
        self,
        signals: List[StockSignalDTO],
    ) -> Tuple[List[StockSignalDTO], Dict[str, Dict[str, int]]]:
        """
        Apply all filters in parallel (all filters to each signal).

        Unlike sequential filtering, this applies all filters to each signal
        and collects all rejection reasons.

        Args:
            signals: List of StockSignalDTO to evaluate

        Returns:
            Tuple of (passed_signals, stats)
        """
        if not signals:
            return [], {}

        passed_signals: List[StockSignalDTO] = []
        stats: Dict[str, Dict[str, int]] = {f.name: {} for f in self._filters}

        for signal in signals:
            all_passed = True

            for f in self._filters:
                passes, reason = f.filter(signal)

                if not passes:
                    all_passed = False
                    reason_key = reason or "unknown"
                    stats[f.name][reason_key] = stats[f.name].get(reason_key, 0) + 1

            if all_passed:
                passed_signals.append(signal)

        return passed_signals, stats

    def get_pipeline_summary(self) -> str:
        """
        Get a human-readable summary of the pipeline configuration.

        Returns:
            Multi-line string describing all filters and their settings
        """
        lines = [f"Filter Pipeline: {self._name}", "=" * 50]

        for i, f in enumerate(self._filters, 1):
            lines.append(f"\n{i}. {f.name}")
            config = f.get_config_summary()
            for key, value in config.items():
                if key != "name":
                    lines.append(f"   - {key}: {value}")

        return "\n".join(lines)

    def get_config(self) -> Dict[str, Any]:
        """
        Get pipeline configuration as dictionary.

        Returns:
            Dictionary with pipeline configuration
        """
        return {
            "name": self._name,
            "filter_count": len(self._filters),
            "filters": [f.get_config_summary() for f in self._filters],
        }

    def __len__(self) -> int:
        """Number of filters in pipeline."""
        return len(self._filters)

    def __repr__(self) -> str:
        """String representation."""
        return f"FilterPipelineImpl(name='{self._name}', filters={len(self._filters)})"
