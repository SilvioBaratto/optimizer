"""
Filter Protocols - Stock filtering interfaces.

Defines the contract for stock filters used in portfolio construction.
Follows the Strategy and Composite patterns for flexibility.

Design Principles:
- Single Responsibility: Each filter handles one filtering criterion
- Open/Closed: New filters can be added without modifying existing code
- Liskov Substitution: All filters are interchangeable
- Interface Segregation: Minimal interface for maximum flexibility
"""

from typing import Protocol, List, Tuple, Optional, Dict, Any, runtime_checkable


@runtime_checkable
class StockFilter(Protocol):
    """
    Protocol for individual stock filters.

    Each filter implementation should:
    - Have a single responsibility (one filtering criterion)
    - Be stateless (configuration via constructor)
    - Return a tuple of (passes, reason) for detailed feedback

    Example implementations:
    - QualityFilter: Sharpe ratio, volatility, drawdown
    - AffordabilityFilter: Price constraints
    - SectorFilter: Sector concentration limits
    - CorrelationFilter: Pairwise correlation limits
    """

    @property
    def name(self) -> str:
        """Human-readable filter name for logging."""
        ...

    def filter(self, signal: Any) -> Tuple[bool, Optional[str]]:
        """
        Check if a single signal passes the filter.

        Args:
            signal: StockSignalDTO to evaluate

        Returns:
            Tuple of (passes, reason):
            - passes: True if signal passes filter
            - reason: Human-readable reason if rejected (None if passes)
        """
        ...

    def filter_batch(
        self,
        signals: List[Any],  # List[StockSignalDTO]
    ) -> Tuple[List[Any], Dict[str, int]]:  # Returns (passed, rejection_counts)
        """
        Filter a batch of signals.

        Args:
            signals: List of StockSignalDTO to evaluate

        Returns:
            Tuple of (passed_signals, rejection_counts):
            - passed_signals: Signals that passed the filter
            - rejection_counts: Count of rejections by reason
        """
        ...


class FilterPipeline(Protocol):
    """
    Protocol for composing multiple filters into a pipeline.

    Implements the Composite pattern to treat a collection of filters
    as a single filter. Filters are applied in sequence.

    Example usage:
        pipeline = FilterPipelineImpl()
        pipeline.add_filter(QualityFilter(min_sharpe=0.5))
        pipeline.add_filter(AffordabilityFilter(max_price=75.0))
        pipeline.add_filter(SectorFilter(max_sector_weight=0.15))

        passed_signals, stats = pipeline.filter_batch(signals)
    """

    def add_filter(self, filter: StockFilter) -> None:
        """
        Add a filter to the pipeline.

        Filters are applied in the order they are added.

        Args:
            filter: StockFilter to add
        """
        ...

    def remove_filter(self, filter_name: str) -> bool:
        """
        Remove a filter by name.

        Args:
            filter_name: Name of filter to remove

        Returns:
            True if filter was found and removed
        """
        ...

    def get_filters(self) -> List[StockFilter]:
        """
        Get list of all filters in the pipeline.

        Returns:
            List of StockFilter objects in execution order
        """
        ...

    def filter_batch(
        self,
        signals: List[Any],  # List[StockSignalDTO]
    ) -> Tuple[List[Any], Dict[str, Dict[str, int]]]:
        """
        Apply all filters to a batch of signals.

        Args:
            signals: List of StockSignalDTO to evaluate

        Returns:
            Tuple of (passed_signals, stats):
            - passed_signals: Signals that passed all filters
            - stats: Dictionary of {filter_name: {reason: count}}
        """
        ...

    def get_pipeline_summary(self) -> str:
        """
        Get a human-readable summary of the pipeline configuration.

        Returns:
            Multi-line string describing all filters and their settings
        """
        ...


class CorrelationFilter(Protocol):
    """
    Specialized protocol for correlation-based filtering.

    This filter requires access to historical price data and operates
    on the entire batch rather than individual signals.

    Different from StockFilter because it considers relationships
    between stocks rather than individual stock characteristics.
    """

    @property
    def max_correlation(self) -> float:
        """Maximum allowed pairwise correlation."""
        ...

    @property
    def max_cluster_size(self) -> int:
        """Maximum stocks allowed per correlation cluster."""
        ...

    def build_correlation_matrix(
        self,
        signals: List[Any],  # List[StockSignalDTO]
        lookback_days: int = 252,
    ) -> Any:  # Returns pd.DataFrame
        """
        Build correlation matrix from historical prices.

        Args:
            signals: Signals to build matrix for
            lookback_days: Historical period for correlation

        Returns:
            DataFrame with pairwise correlations
        """
        ...

    def select_diversified(
        self,
        signals: List[Any],
        correlation_matrix: Any,  # pd.DataFrame
        target_count: int,
    ) -> List[Any]:  # Returns List[StockSignalDTO]
        """
        Select diversified stocks using correlation constraints.

        Args:
            signals: Candidate signals
            correlation_matrix: Precomputed correlation matrix
            target_count: Target number of stocks to select

        Returns:
            List of selected StockSignalDTO objects
        """
        ...
