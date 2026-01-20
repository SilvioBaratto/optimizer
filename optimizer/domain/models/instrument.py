"""
Instrument Domain Model - Pure Python representation of tradeable instruments.

Instruments are the static reference data for stocks, ETFs, and other securities.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID


@dataclass(frozen=True)
class ExchangeDTO:
    """
    Data Transfer Object for stock exchanges.

    Represents a trading venue (NYSE, NASDAQ, etc.) with metadata.
    """

    id: UUID
    exchange_id: int  # Trading212 exchange ID
    exchange_name: str
    is_active: bool = True
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "exchange_id": self.exchange_id,
            "exchange_name": self.exchange_name,
            "is_active": self.is_active,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


@dataclass(frozen=True)
class InstrumentDTO:
    """
    Data Transfer Object for instruments (stocks/securities).

    Immutable representation of a tradeable instrument with:
    - Trading212 identifiers
    - Yahoo Finance mapping
    - Basic metadata

    Attributes:
        id: Unique instrument identifier
        exchange_id: Reference to exchange
        ticker: Trading212 ticker symbol (e.g., 'AAPL_US_EQ')
        short_name: Short ticker name (e.g., 'AAPL')
        name: Full company name
        isin: ISIN code
        instrument_type: Type (STOCK, ETF, etc.)
        currency_code: Trading currency (USD, EUR, GBP)
        yfinance_ticker: Yahoo Finance ticker for data fetching
        is_active: Whether instrument is currently tradeable
        max_open_quantity: Trading212 position limit
        added_on: Date added to Trading212
        last_validated: Last data validation timestamp
    """

    # Required fields
    id: UUID
    exchange_id: UUID
    ticker: str
    short_name: str

    # Optional fields
    name: Optional[str] = None
    isin: Optional[str] = None
    instrument_type: str = "STOCK"
    currency_code: Optional[str] = None
    yfinance_ticker: Optional[str] = None
    is_active: bool = True
    max_open_quantity: Optional[float] = None
    added_on: Optional[datetime] = None
    last_validated: Optional[datetime] = None

    # Denormalized exchange info (for convenience)
    exchange_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "exchange_id": str(self.exchange_id),
            "ticker": self.ticker,
            "short_name": self.short_name,
            "name": self.name,
            "isin": self.isin,
            "instrument_type": self.instrument_type,
            "currency_code": self.currency_code,
            "yfinance_ticker": self.yfinance_ticker,
            "is_active": self.is_active,
            "max_open_quantity": self.max_open_quantity,
            "added_on": self.added_on.isoformat() if self.added_on else None,
            "last_validated": self.last_validated.isoformat() if self.last_validated else None,
            "exchange_name": self.exchange_name,
        }

    def to_dict_summary(self) -> Dict[str, Any]:
        """Return a summary dictionary (for API responses)."""
        return {
            "id": str(self.id),
            "ticker": self.ticker,
            "short_name": self.short_name,
            "name": self.name,
            "yfinance_ticker": self.yfinance_ticker,
            "exchange_name": self.exchange_name,
        }

    @property
    def is_us_stock(self) -> bool:
        """Check if this is a US-listed stock."""
        if self.ticker:
            return "_US_" in self.ticker
        return False

    @property
    def is_european_stock(self) -> bool:
        """Check if this is a European-listed stock."""
        if self.yfinance_ticker:
            european_suffixes = [".L", ".DE", ".PA", ".AS", ".MI", ".MC"]
            return any(self.yfinance_ticker.endswith(suffix) for suffix in european_suffixes)
        return False

    @property
    def country(self) -> str:
        """
        Infer country from ticker/exchange.

        Returns country code based on yfinance ticker suffix or ticker pattern.
        """
        if self.yfinance_ticker:
            suffix_to_country = {
                ".L": "UK",
                ".DE": "Germany",
                ".PA": "France",
                ".AS": "Netherlands",
                ".MI": "Italy",
                ".MC": "Spain",
                ".SW": "Switzerland",
                ".TO": "Canada",
                ".AX": "Australia",
                ".HK": "Hong Kong",
                ".T": "Japan",
            }
            for suffix, country in suffix_to_country.items():
                if self.yfinance_ticker.endswith(suffix):
                    return country

        # Default to USA for US tickers
        if self.ticker and "_US_" in self.ticker:
            return "USA"

        return "Unknown"
