"""
Universe Building Protocols - Interface definitions for universe construction.

Defines contracts for:
- InstrumentFilter: Single filter in the pipeline
- FilterPipeline: Composite of multiple filters
- TickerMapper: Maps Trading212 symbols to yfinance tickers
- TickerCache: Caches ticker mappings
- Trading212ApiClient: Trading212 API access
- UniverseRepository: Data access for universe building
"""

from typing import Protocol, runtime_checkable, Optional, Dict, List, Any, Tuple


@runtime_checkable
class InstrumentFilter(Protocol):
    @property
    def name(self) -> str: ...

    def filter(self, data: Dict[str, Any], yf_ticker: str) -> Tuple[bool, str]: ...


@runtime_checkable
class FilterPipeline(Protocol):
    def add_filter(self, filter: InstrumentFilter) -> "FilterPipeline": ...
    def apply(self, data: Dict[str, Any], yf_ticker: str) -> Tuple[bool, str]: ...
    def get_summary(self) -> Dict[str, Dict[str, int]]: ...
    def reset_stats(self) -> None: ...


@runtime_checkable
class TickerMapper(Protocol):
    def discover(
        self, symbol: str, exchange_name: Optional[str] = None
    ) -> Optional[str]: ...


@runtime_checkable
class TickerCache(Protocol):
    def get_mapping(
        self, symbol: str, exchange_name: str, max_age_days: int = 90
    ) -> Optional[str]: ...

    def save_mapping(self, symbol: str, exchange_name: str, yf_ticker: str) -> None: ...


@runtime_checkable
class Trading212ApiClient(Protocol):
    def get_exchanges(self) -> List[Dict[str, Any]]: ...
    def get_instruments(self) -> List[Dict[str, Any]]: ...


@runtime_checkable
class UniverseRepository(Protocol):
    def save_exchange(self, exchange_data: Dict[str, Any]) -> Any: ...
    def save_instruments_batch(
        self, instruments_data: List[Dict[str, Any]], exchange_id: Any
    ) -> int: ...
    def clear_all(self) -> Tuple[int, int]: ...
    def get_instrument_count(self) -> int: ...
    def get_exchange_count(self) -> int: ...
