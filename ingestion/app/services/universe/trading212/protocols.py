"""Structural interfaces for universe construction components.

Ingestion applies no investability filtering, so there are no filter/pipeline
protocols here — screening lives in the downstream fund layer.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TickerMapper(Protocol):
    """Resolve a Trading 212 symbol + exchange to a yfinance ticker."""

    def discover(self, symbol: str, exchange_name: str | None = None) -> str | None: ...


@runtime_checkable
class TickerCache(Protocol):
    """Per-build in-memory store for resolved (symbol, exchange) → ticker mappings."""

    def get_mapping(
        self, symbol: str, exchange_name: str, max_age_days: int = 90
    ) -> str | None: ...

    def save_mapping(self, symbol: str, exchange_name: str, yf_ticker: str) -> None: ...


@runtime_checkable
class Trading212ApiClient(Protocol):
    """Fetch the raw exchange and instrument metadata lists from T212."""

    def get_exchanges(self) -> list[dict[str, Any]]: ...
    def get_instruments(self) -> list[dict[str, Any]]: ...


@runtime_checkable
class UniverseRepository(Protocol):
    """Persist and query universe state (exchanges, instruments, delistings)."""

    def save_exchange(self, exchange_data: dict[str, Any]) -> Any: ...
    def save_instruments_batch(
        self, instruments_data: list[dict[str, Any]], exchange_id: Any
    ) -> int: ...
    def clear_all(self) -> tuple[int, int]: ...
    def get_instrument_count(self) -> int: ...
    def get_exchange_count(self) -> int: ...
