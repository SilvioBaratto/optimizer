"""Sub-client for WebSocket streaming via ``yf.WebSocket`` / ``yf.AsyncWebSocket``.

Both clients enforce a handler-first lifecycle: ``set_handler()`` must be called
before ``subscribe()`` or ``listen()`` — otherwise a ``RuntimeError`` is raised.
``_ws`` lazy construction and mutation is guarded by a ``threading.Lock``
(``StreamingClient``) or ``asyncio.Lock`` (``AsyncStreamingClient``) so concurrent
callers cannot double-construct the underlying WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

_HANDLER_REQUIRED_MSG = (
    "set_handler() must be called before subscribe() / listen()"
)


class StreamingClient:
    """Synchronous WebSocket wrapper around ``yf.WebSocket``."""

    def __init__(self) -> None:
        self._ws: yf.WebSocket | None = None
        self._lock = threading.Lock()
        self._handler: Callable[[dict[str, Any]], None] | None = None

    def set_handler(self, handler: Callable[[dict[str, Any]], None]) -> None:
        self._handler = handler

    def subscribe(self, symbols: list[str]) -> None:
        if self._handler is None:
            raise RuntimeError(_HANDLER_REQUIRED_MSG)
        logger.debug("Subscribing to %d symbols", len(symbols))
        with self._lock:
            if self._ws is None:
                self._ws = yf.WebSocket()
            self._ws.subscribe(symbols)

    def unsubscribe(self, symbols: list[str]) -> None:
        logger.debug("Unsubscribing from %d symbols", len(symbols))
        with self._lock:
            if self._ws is not None:
                self._ws.unsubscribe(symbols)

    def listen(self) -> None:
        if self._handler is None:
            raise RuntimeError(_HANDLER_REQUIRED_MSG)
        with self._lock:
            ws = self._ws
        if ws is None:
            raise RuntimeError("No active WebSocket. Call subscribe() first.")
        ws.listen(message_handler=self._handler)

    def close(self) -> None:
        logger.debug("Closing WebSocket")
        with self._lock:
            if self._ws is not None:
                self._ws.close()
                self._ws = None


class AsyncStreamingClient:
    """Async WebSocket wrapper around ``yf.AsyncWebSocket``."""

    def __init__(self) -> None:
        self._ws: yf.AsyncWebSocket | None = None
        self._lock: asyncio.Lock | None = None
        self._handler: Callable[[dict[str, Any]], None] | None = None

    def set_handler(self, handler: Callable[[dict[str, Any]], None]) -> None:
        self._handler = handler

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def subscribe(self, symbols: list[str]) -> None:
        if self._handler is None:
            raise RuntimeError(_HANDLER_REQUIRED_MSG)
        logger.debug("Async subscribing to %d symbols", len(symbols))
        async with self._ensure_lock():
            if self._ws is None:
                self._ws = yf.AsyncWebSocket()
            await self._ws.subscribe(symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        logger.debug("Async unsubscribing from %d symbols", len(symbols))
        async with self._ensure_lock():
            if self._ws is not None:
                await self._ws.unsubscribe(symbols)

    async def listen(self) -> None:
        if self._handler is None:
            raise RuntimeError(_HANDLER_REQUIRED_MSG)
        async with self._ensure_lock():
            ws = self._ws
        if ws is None:
            raise RuntimeError("No active AsyncWebSocket. Call subscribe() first.")
        await ws.listen(message_handler=self._handler)

    async def close(self) -> None:
        logger.debug("Closing AsyncWebSocket")
        async with self._ensure_lock():
            if self._ws is not None:
                await self._ws.close()
                self._ws = None
