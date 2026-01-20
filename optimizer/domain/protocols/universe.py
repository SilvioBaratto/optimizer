"""
Universe Building Protocols - Interface definitions for universe construction.

Defines contracts for:
- InstrumentFilter: Single filter in the pipeline
- FilterPipeline: Composite of multiple filters
- TickerMapper: Maps Trading212 symbols to yfinance tickers
- UniverseRepository: Data access for universe building

Design Principles:
- No concrete implementations (Protocol pattern)
- No database imports (clean domain layer)
- runtime_checkable for isinstance() support
"""

from typing import Protocol, runtime_checkable, Optional, Dict, List, Any, Tuple
from datetime import datetime


@runtime_checkable
class InstrumentFilter(Protocol):
    """
    Protocol for instrument filters in the universe building pipeline.

    Each filter implementation should:
    - Have a single responsibility (one filtering criterion)
    - Be stateless (configuration via constructor)
    - Return a tuple of (passed, reason) for detailed feedback

    Example implementations:
    - MarketCapFilter: Minimum market cap
    - PriceFilter: Price range validation
    - LiquidityFilter: ADV thresholds by market cap segment
    - DataCoverageFilter: Institutional data completeness
    - HistoricalDataFilter: Historical data availability
    """

    @property
    def name(self) -> str:
        """Human-readable filter name for logging and statistics."""
        ...

    def filter(self, data: Dict[str, Any], yf_ticker: str) -> Tuple[bool, str]:
        """
        Apply filter to instrument data.

        Args:
            data: yfinance info dictionary with financial data
            yf_ticker: Yahoo Finance ticker symbol

        Returns:
            Tuple of (passed, reason):
            - passed: True if instrument passes filter
            - reason: Human-readable reason (explanation if passed, rejection if failed)
        """
        ...


@runtime_checkable
class FilterPipeline(Protocol):
    """
    Protocol for composing multiple filters into a pipeline.

    Implements the Composite pattern to treat a collection of filters
    as a single filter. Filters are applied in sequence - a stock must
    pass ALL filters to be included in the universe.

    Example usage:
        pipeline = FilterPipelineImpl()
        pipeline.add_filter(MarketCapFilter(config))
        pipeline.add_filter(PriceFilter(config))
        pipeline.add_filter(LiquidityFilter(config))

        passed, reason = pipeline.apply(data, yf_ticker)
    """

    def add_filter(self, filter: InstrumentFilter) -> "FilterPipeline":
        """
        Add a filter to the pipeline.

        Filters are applied in the order they are added.

        Args:
            filter: InstrumentFilter to add

        Returns:
            Self for method chaining
        """
        ...

    def apply(self, data: Dict[str, Any], yf_ticker: str) -> Tuple[bool, str]:
        """
        Apply all filters to instrument data.

        Args:
            data: yfinance info dictionary
            yf_ticker: Yahoo Finance ticker symbol

        Returns:
            Tuple of (passed, reason):
            - passed: True if instrument passes ALL filters
            - reason: Final status (or first failure reason)
        """
        ...

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for all filters in the pipeline.

        Returns:
            Dictionary mapping filter_name -> {"passed": count, "failed": count}
        """
        ...

    def reset_stats(self) -> None:
        """Reset all filter statistics to zero."""
        ...


@runtime_checkable
class TickerMapper(Protocol):
    """
    Protocol for mapping Trading212 symbols to yfinance tickers.

    Responsible for discovering the correct Yahoo Finance ticker
    for a given Trading212 symbol and exchange.

    Implementations may:
    - Use suffix mapping (exchange -> Yahoo suffix)
    - Try multiple ticker formats
    - Use caching for previously discovered mappings
    """

    def discover(
        self, symbol: str, exchange_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Discover the correct yfinance ticker for a Trading212 symbol.

        Args:
            symbol: Trading212 short name (e.g., 'AAPL', 'VOW3')
            exchange_name: Exchange name for suffix determination

        Returns:
            yfinance ticker string if found (e.g., 'AAPL', 'VOW3.DE')
            None if ticker cannot be discovered
        """
        ...


@runtime_checkable
class TickerCache(Protocol):
    """
    Protocol for caching ticker mappings.

    Provides persistent caching of T212 -> yfinance ticker mappings
    to avoid redundant API calls.
    """

    def get_mapping(
        self, symbol: str, exchange_name: str, max_age_days: int = 90
    ) -> Optional[str]:
        """
        Get cached ticker mapping if available and fresh.

        Args:
            symbol: Trading212 short name
            exchange_name: Exchange name
            max_age_days: Maximum age of cached mapping in days

        Returns:
            yfinance ticker if cached and fresh, None otherwise
        """
        ...

    def save_mapping(self, symbol: str, exchange_name: str, yf_ticker: str) -> None:
        """
        Save successful ticker mapping to cache.

        Args:
            symbol: Trading212 short name
            exchange_name: Exchange name
            yf_ticker: Discovered yfinance ticker
        """
        ...


@runtime_checkable
class UniverseRepository(Protocol):
    """
    Protocol for universe data access.

    Handles persistence of exchanges and instruments during universe building.
    Returns domain objects (DTOs), not ORM models.

    Implementations should:
    - Handle database transactions internally
    - Convert between ORM models and DTOs
    - Support batch operations for performance
    """

    def save_exchange(self, exchange_data: Dict[str, Any]) -> Any:
        """
        Save or update an exchange.

        Args:
            exchange_data: Exchange data from Trading212 API with keys:
                - id: Trading212 exchange ID
                - name: Exchange name

        Returns:
            ExchangeDTO with generated UUID
        """
        ...

    def get_exchange_by_t212_id(self, t212_exchange_id: int) -> Optional[Any]:
        """
        Get exchange by Trading212 ID.

        Args:
            t212_exchange_id: Trading212 exchange ID

        Returns:
            ExchangeDTO or None if not found
        """
        ...

    def save_instrument(
        self, instrument_data: Dict[str, Any], exchange_id: Any
    ) -> Optional[Any]:
        """
        Save or update an instrument.

        Args:
            instrument_data: Instrument data with keys:
                - ticker: Trading212 ticker
                - shortName: Short ticker name
                - name: Full name
                - isin: ISIN code
                - type: Instrument type
                - currencyCode: Trading currency
                - maxOpenQuantity: Position limit
                - addedOn: Date added
                - yfinanceTicker: Discovered yfinance ticker

            exchange_id: UUID of parent exchange

        Returns:
            InstrumentDTO or None if save failed
        """
        ...

    def save_instruments_batch(
        self, instruments_data: List[Dict[str, Any]], exchange_id: Any
    ) -> int:
        """
        Save multiple instruments in a batch.

        Args:
            instruments_data: List of instrument data dictionaries
            exchange_id: UUID of parent exchange

        Returns:
            Number of instruments saved
        """
        ...

    def clear_all(self) -> Tuple[int, int]:
        """
        Clear all universe data (exchanges and instruments).

        Returns:
            Tuple of (deleted_exchanges, deleted_instruments)
        """
        ...

    def get_instrument_count(self) -> int:
        """
        Get total number of instruments.

        Returns:
            Total instrument count
        """
        ...

    def get_exchange_count(self) -> int:
        """
        Get total number of exchanges.

        Returns:
            Total exchange count
        """
        ...


@runtime_checkable
class Trading212ApiClient(Protocol):
    """
    Protocol for Trading212 API access.

    Handles fetching exchange and instrument metadata from the Trading212 API.
    """

    def get_exchanges(self) -> List[Dict[str, Any]]:
        """
        Fetch all exchanges from Trading212 API.

        Returns:
            List of exchange dictionaries with keys:
                - id: Exchange ID
                - name: Exchange name
                - workingSchedules: List of schedule dicts
        """
        ...

    def get_instruments(self) -> List[Dict[str, Any]]:
        """
        Fetch all instruments from Trading212 API.

        Returns:
            List of instrument dictionaries with keys:
                - ticker: Trading212 ticker (e.g., 'AAPL_US_EQ')
                - shortName: Short name (e.g., 'AAPL')
                - name: Full company name
                - isin: ISIN code
                - type: 'STOCK', 'ETF', etc.
                - currencyCode: Trading currency
                - workingScheduleId: Schedule ID (links to exchange)
                - maxOpenQuantity: Position limit
                - addedOn: Date added
        """
        ...
