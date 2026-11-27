"""Structural interfaces for universe construction components.

Ingestion applies no investability filtering, so there are no filter/pipeline
protocols here — screening lives in the downstream fund layer.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TickerMapper(Protocol):
    """Resolve a source symbol + exchange to a yfinance ticker."""

    def discover(self, symbol: str, exchange_name: str | None = None) -> str | None: ...


@runtime_checkable
class UniverseSourceClient(Protocol):
    """Fetch the raw exchange and instrument lists a universe is built from."""

    def get_exchanges(self) -> list[dict[str, Any]]: ...
    def get_instruments(self) -> list[dict[str, Any]]: ...


@runtime_checkable
class LiveSymbolReporter(Protocol):
    """Optional ``UniverseSourceClient`` extension for sources that can vouch for liveness."""

    def get_live_symbols(self) -> set[str]:
        """Symbols the source saw trading, whether or not they were admitted."""
        ...


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
