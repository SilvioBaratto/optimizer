"""
Repository Protocols - Data access interfaces.

These protocols define the contract for data access, allowing business logic
to be independent of the underlying storage mechanism (PostgreSQL, etc.).

Design Principles:
- Methods return domain objects (DTOs), not ORM models
- No SQLAlchemy imports allowed
- Type hints for all parameters and return values
- Optional filtering via keyword arguments
"""

from typing import Protocol, List, Optional, Dict, Any
from datetime import date
from uuid import UUID


class SignalRepository(Protocol):
    """
    Protocol for stock signal data access.

    Implementations should:
    - Return StockSignalDTO objects
    - Handle database transactions internally
    - Raise appropriate exceptions for errors
    """

    def get_by_date(
        self,
        signal_date: date,
        signal_types: Optional[List[str]] = None,
        min_sharpe: Optional[float] = None,
        sector: Optional[str] = None,
    ) -> List[Any]:  # Returns List[StockSignalDTO]
        """
        Fetch signals for a specific date with optional filters.

        Args:
            signal_date: Date to fetch signals for
            signal_types: Filter by signal types (e.g., ['LARGE_GAIN'])
            min_sharpe: Minimum Sharpe ratio filter
            sector: Filter by sector

        Returns:
            List of StockSignalDTO objects
        """
        ...

    def get_latest_date(self, signal_type: Optional[str] = None) -> Optional[date]:
        """
        Get the most recent signal date in the database.

        Args:
            signal_type: Optional filter by signal type

        Returns:
            Most recent date, or None if no signals exist
        """
        ...

    def get_by_tickers(
        self,
        tickers: List[str],
        signal_date: Optional[date] = None,
    ) -> Dict[str, Any]:  # Returns Dict[str, StockSignalDTO]
        """
        Fetch signals for specific tickers.

        Args:
            tickers: List of ticker symbols
            signal_date: Optional date (defaults to most recent)

        Returns:
            Dictionary mapping ticker -> StockSignalDTO
        """
        ...

    def get_large_gain_signals(
        self,
        signal_date: Optional[date] = None,
        min_sharpe: Optional[float] = None,
        max_volatility: Optional[float] = None,
    ) -> List[Any]:  # Returns List[StockSignalDTO]
        """
        Fetch LARGE_GAIN signals with quality filters.

        This is a convenience method for the most common use case.

        Args:
            signal_date: Date to fetch (defaults to most recent)
            min_sharpe: Minimum Sharpe ratio
            max_volatility: Maximum volatility

        Returns:
            List of StockSignalDTO objects
        """
        ...


class InstrumentRepository(Protocol):
    """
    Protocol for instrument (stock) metadata access.

    Instruments are the static reference data for tradeable securities.
    """

    def get_by_ticker(self, ticker: str) -> Optional[Any]:  # Returns Optional[InstrumentDTO]
        """
        Fetch a single instrument by Trading212 ticker.

        Args:
            ticker: Trading212 ticker symbol (e.g., 'AAPL_US_EQ')

        Returns:
            InstrumentDTO or None if not found
        """
        ...

    def get_by_tickers(self, tickers: List[str]) -> Dict[str, Any]:  # Returns Dict[str, InstrumentDTO]
        """
        Fetch multiple instruments by tickers.

        Args:
            tickers: List of Trading212 ticker symbols

        Returns:
            Dictionary mapping ticker -> InstrumentDTO
        """
        ...

    def get_yfinance_mapping(self, tickers: List[str]) -> Dict[str, Optional[str]]:
        """
        Get mapping from Trading212 tickers to Yahoo Finance tickers.

        Args:
            tickers: List of Trading212 ticker symbols

        Returns:
            Dictionary mapping t212_ticker -> yfinance_ticker (or None)
        """
        ...

    def get_by_exchange(self, exchange_name: str) -> List[Any]:  # Returns List[InstrumentDTO]
        """
        Fetch all instruments from a specific exchange.

        Args:
            exchange_name: Exchange name (e.g., 'NYSE', 'NASDAQ')

        Returns:
            List of InstrumentDTO objects
        """
        ...

    def get_active_instruments(self) -> List[Any]:  # Returns List[InstrumentDTO]
        """
        Fetch all active (tradeable) instruments.

        Returns:
            List of InstrumentDTO objects
        """
        ...


class MacroRegimeRepository(Protocol):
    """
    Protocol for macro regime assessment data access.

    Macro regimes are business cycle classifications (early_cycle, mid_cycle, etc.)
    with associated recommendations.
    """

    def get_latest_by_country(self, country: str) -> Optional[Any]:  # Returns Optional[MacroRegimeDTO]
        """
        Fetch the most recent regime assessment for a country.

        Args:
            country: Country code (e.g., 'USA', 'Germany')

        Returns:
            MacroRegimeDTO or None if not found
        """
        ...

    def get_by_date_range(
        self,
        country: str,
        start_date: date,
        end_date: date,
    ) -> List[Any]:  # Returns List[MacroRegimeDTO]
        """
        Fetch regime assessments for a date range.

        Args:
            country: Country code
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of MacroRegimeDTO objects
        """
        ...

    def get_all_countries_latest(self) -> Dict[str, Any]:  # Returns Dict[str, MacroRegimeDTO]
        """
        Fetch the most recent assessment for all countries.

        Returns:
            Dictionary mapping country -> MacroRegimeDTO
        """
        ...


class PortfolioRepository(Protocol):
    """
    Protocol for portfolio data access.

    Portfolios are collections of positions with weights and metadata.
    """

    def save(self, portfolio: Any) -> UUID:  # Takes PortfolioDTO, returns portfolio ID
        """
        Save a new portfolio to the database.

        Args:
            portfolio: PortfolioDTO to save

        Returns:
            UUID of the saved portfolio
        """
        ...

    def get_by_id(self, portfolio_id: UUID) -> Optional[Any]:  # Returns Optional[PortfolioDTO]
        """
        Fetch a portfolio by its ID.

        Args:
            portfolio_id: Portfolio UUID

        Returns:
            PortfolioDTO or None if not found
        """
        ...

    def get_by_date(self, portfolio_date: date) -> List[Any]:  # Returns List[PortfolioDTO]
        """
        Fetch all portfolios for a specific date.

        Args:
            portfolio_date: Portfolio date

        Returns:
            List of PortfolioDTO objects
        """
        ...

    def get_latest(self, limit: int = 10) -> List[Any]:  # Returns List[PortfolioDTO]
        """
        Fetch the most recent portfolios.

        Args:
            limit: Maximum number of portfolios to return

        Returns:
            List of PortfolioDTO objects
        """
        ...

    def update_positions(
        self,
        portfolio_id: UUID,
        positions: List[Any],  # List[PositionDTO]
    ) -> None:
        """
        Update positions for an existing portfolio.

        Args:
            portfolio_id: Portfolio UUID
            positions: New list of PositionDTO objects
        """
        ...
