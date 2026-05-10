"""Cycle 4 streaming lock + handler-first contract.

Validates:
1. ``StreamingClient`` and ``AsyncStreamingClient`` guard ``_ws`` mutation with
   a lock (``threading.Lock`` / ``asyncio.Lock``); 20 concurrent ``subscribe()``
   calls produce exactly one underlying ``yf.WebSocket`` / ``yf.AsyncWebSocket``.
2. ``subscribe()`` and ``listen()`` raise ``RuntimeError`` (mentioning
   ``set_handler``) when called before ``set_handler()``.
3. ``isinstance(StreamingClient(), StreamingClientProtocol)`` and
   ``isinstance(AsyncStreamingClient(), AsyncStreamingClientProtocol)`` both
   return ``True``.

Issue: #602
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.services.yfinance.market.streaming import (
    AsyncStreamingClient,
    StreamingClient,
)
from app.services.yfinance.protocols import (
    AsyncStreamingClientProtocol,
    StreamingClientProtocol,
)

_THREAD_COUNT: int = 20


def _noop_handler(_msg: dict[str, Any]) -> None:
    return None


def test_when_20_threads_subscribe_then_yf_websocket_constructed_once() -> None:
    counter = {"n": 0}
    counter_lock = threading.Lock()

    def fake_ws_factory(*_args: Any, **_kwargs: Any) -> MagicMock:
        with counter_lock:
            counter["n"] += 1
        return MagicMock()

    client = StreamingClient()
    client.set_handler(_noop_handler)
    barrier = threading.Barrier(_THREAD_COUNT)
    errors: list[BaseException] = []

    with patch(
        "app.services.yfinance.market.streaming.yf.WebSocket",
        side_effect=fake_ws_factory,
    ):

        def worker() -> None:
            try:
                barrier.wait(timeout=10.0)
                client.subscribe(["AAPL"])
            except BaseException as exc:
                with counter_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(_THREAD_COUNT)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

    assert not errors, f"Worker threads raised: {errors!r}"
    assert counter["n"] == 1, f"yf.WebSocket constructed {counter['n']} times, expected 1"


async def test_when_20_tasks_subscribe_async_then_yf_async_websocket_constructed_once() -> (
    None
):
    counter = {"n": 0}

    class _FakeAsyncWs:
        async def subscribe(self, symbols: list[str]) -> None:
            await asyncio.sleep(0)

        async def unsubscribe(self, symbols: list[str]) -> None:
            await asyncio.sleep(0)

        async def listen(self, **kwargs: Any) -> None:
            await asyncio.sleep(0)

        async def close(self) -> None:
            await asyncio.sleep(0)

    def sync_factory(*args: Any, **kwargs: Any) -> _FakeAsyncWs:
        counter["n"] += 1
        return _FakeAsyncWs()

    client = AsyncStreamingClient()
    client.set_handler(_noop_handler)
    start_event = asyncio.Event()

    with patch(
        "app.services.yfinance.market.streaming.yf.AsyncWebSocket",
        side_effect=sync_factory,
    ):

        async def worker() -> None:
            await start_event.wait()
            await client.subscribe(["AAPL"])

        tasks = [asyncio.create_task(worker()) for _ in range(_THREAD_COUNT)]
        await asyncio.sleep(0)
        start_event.set()
        await asyncio.gather(*tasks)

    assert counter["n"] == 1, (
        f"yf.AsyncWebSocket constructed {counter['n']} times, expected 1"
    )


def test_when_subscribe_called_without_set_handler_then_runtime_error() -> None:
    client = StreamingClient()
    with pytest.raises(RuntimeError, match="set_handler"):
        client.subscribe(["AAPL"])


def test_when_listen_called_without_set_handler_then_runtime_error() -> None:
    client = StreamingClient()
    with pytest.raises(RuntimeError, match="set_handler"):
        client.listen()


async def test_when_async_subscribe_called_without_set_handler_then_runtime_error() -> (
    None
):
    client = AsyncStreamingClient()
    with pytest.raises(RuntimeError, match="set_handler"):
        await client.subscribe(["AAPL"])


async def test_when_async_listen_called_without_set_handler_then_runtime_error() -> (
    None
):
    client = AsyncStreamingClient()
    with pytest.raises(RuntimeError, match="set_handler"):
        await client.listen()


def test_when_streaming_client_built_then_satisfies_streaming_protocol() -> None:
    assert isinstance(StreamingClient(), StreamingClientProtocol)


def test_when_async_streaming_client_built_then_satisfies_async_protocol() -> None:
    assert isinstance(AsyncStreamingClient(), AsyncStreamingClientProtocol)
