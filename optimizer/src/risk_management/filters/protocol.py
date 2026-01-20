"""
Stock Filter Protocol Implementation - Base class for all filters.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any

from optimizer.domain.models.stock_signal import StockSignalDTO


class StockFilterImpl(ABC):
    """
    Abstract base class implementing the StockFilter protocol.

    Provides common functionality for all filter implementations:
    - Batch filtering with statistics
    - Template method for filter logic

    Subclasses should implement:
    - _filter_single(): Core filtering logic for one signal
    - name property: Human-readable filter name
    """

    def __init__(self):
        """Initialize filter."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable filter name."""
        pass

    @abstractmethod
    def _filter_single(self, signal: StockSignalDTO) -> Tuple[bool, Optional[str]]:
        """
        Core filtering logic for a single signal.

        Args:
            signal: StockSignalDTO to evaluate

        Returns:
            Tuple of (passes, reason):
            - passes: True if signal passes filter
            - reason: Human-readable reason if rejected (None if passes)
        """
        pass

    def filter(self, signal: StockSignalDTO) -> Tuple[bool, Optional[str]]:
        """
        Check if a single signal passes the filter.

        This is the public interface that wraps _filter_single.

        Args:
            signal: StockSignalDTO to evaluate

        Returns:
            Tuple of (passes, reason)
        """
        passes, reason = self._filter_single(signal)
        return passes, reason

    def filter_batch(
        self,
        signals: List[StockSignalDTO],
    ) -> Tuple[List[StockSignalDTO], Dict[str, int]]:
        """
        Filter a batch of signals.

        Args:
            signals: List of StockSignalDTO to evaluate

        Returns:
            Tuple of (passed_signals, rejection_counts):
            - passed_signals: Signals that passed the filter
            - rejection_counts: Count of rejections by reason
        """
        passed: List[StockSignalDTO] = []
        rejection_counts: Dict[str, int] = {}

        for signal in signals:
            passes, reason = self.filter(signal)

            if passes:
                passed.append(signal)
            elif reason:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        return passed, rejection_counts

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of filter configuration.

        Subclasses can override to include specific thresholds.

        Returns:
            Dictionary of configuration values
        """
        return {"name": self.name}


class CompositeFilter(StockFilterImpl):
    """
    Composite filter that combines multiple filters using AND logic.

    A signal must pass ALL component filters to pass the composite.
    """

    def __init__(self, filters: Optional[List[StockFilterImpl]] = None):
        """
        Initialize composite filter.

        Args:
            filters: Optional list of filters to include
        """
        super().__init__()
        self._filters: List[StockFilterImpl] = filters or []

    @property
    def name(self) -> str:
        """Composite filter name."""
        if not self._filters:
            return "CompositeFilter(empty)"
        filter_names = ", ".join(f.name for f in self._filters)
        return f"CompositeFilter({filter_names})"

    def add_filter(self, filter: StockFilterImpl) -> None:
        """Add a filter to the composite."""
        self._filters.append(filter)

    def remove_filter(self, filter_name: str) -> bool:
        """Remove a filter by name."""
        for i, f in enumerate(self._filters):
            if f.name == filter_name:
                self._filters.pop(i)
                return True
        return False

    def _filter_single(self, signal: StockSignalDTO) -> Tuple[bool, Optional[str]]:
        """
        Check if signal passes ALL component filters.

        Returns on first failure with the failing filter's reason.
        """
        for f in self._filters:
            passes, reason = f.filter(signal)
            if not passes:
                return False, f"{f.name}: {reason}"
        return True, None

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration of all component filters."""
        return {
            "name": self.name,
            "filters": [f.get_config_summary() for f in self._filters],
        }
