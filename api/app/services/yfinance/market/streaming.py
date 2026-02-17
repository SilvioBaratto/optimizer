"""Sub-client for WebSocket streaming via ``yf.WebSocket`` / ``yf.AsyncWebSocket``."""

from __future__ import annotations

import logging
from typing import Any, Callable

import yfinance as yf

logger = logging.getLogger(__name__)


class StreamingClient:
    """Synchronous WebSocket wrapper around ``yf.WebSocket``."""

    def __init__(self) -> None:
        self._ws: yf.WebSocket | None = None

    def subscribe(self, symbols: list[str]) -> None:
        logger.debug("Subscribing to %d symbols", len(symbols))
        if self._ws is None:
            self._ws = yf.WebSocket()
        self._ws.subscribe(symbols)

    def unsubscribe(self, symbols: list[str]) -> None:
        logger.debug("Unsubscribing from %d symbols", len(symbols))
        if self._ws is not None:
            self._ws.unsubscribe(symbols)

    def listen(
        self,
        message_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if self._ws is None:
            raise RuntimeError("No active WebSocket. Call subscribe() first.")
        self._ws.listen(message_handler=message_handler)

    def close(self) -> None:
        logger.debug("Closing WebSocket")
        if self._ws is not None:
            self._ws.close()
            self._ws = None


class AsyncStreamingClient:
    """Async WebSocket wrapper around ``yf.AsyncWebSocket``."""

    def __init__(self) -> None:
        self._ws: yf.AsyncWebSocket | None = None

    async def subscribe(self, symbols: list[str]) -> None:
        logger.debug("Async subscribing to %d symbols", len(symbols))
        if self._ws is None:
            self._ws = yf.AsyncWebSocket()
        await self._ws.subscribe(symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        logger.debug("Async unsubscribing from %d symbols", len(symbols))
        if self._ws is not None:
            await self._ws.unsubscribe(symbols)

    async def listen(
        self,
        message_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if self._ws is None:
            raise RuntimeError("No active AsyncWebSocket. Call subscribe() first.")
        await self._ws.listen(message_handler=message_handler)

    async def close(self) -> None:
        logger.debug("Closing AsyncWebSocket")
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
