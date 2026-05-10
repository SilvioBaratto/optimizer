"""Cycle 4 torture test: sub-client properties migrate to ``functools.cached_property``.

Validates that all 13 ``YFinanceClient`` sub-client properties are constructed
exactly once even when 50 threads race on first access, that the cache lives
under the un-prefixed attribute name in ``__dict__`` (not ``_<name>``), and that
``reset_instance`` evicts every cached entry before nulling the singleton.

Issue: #601
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import pytest

from app.services.market_data.yfinance._facade import YFinanceClient

_SUBCLIENT_NAMES: tuple[str, ...] = (
    "financials",
    "analysis",
    "holders",
    "corporate_actions",
    "metadata",
    "funds",
    "market",
    "sectors",
    "search",
    "screener",
    "calendars",
    "streaming",
    "async_streaming",
)

_PATCH_TARGETS: dict[str, str] = {
    "financials": (
        "app.services.market_data.yfinance.ticker.financials.FinancialsClient.__init__"
    ),
    "analysis": "app.services.market_data.yfinance.ticker.analysis.AnalysisClient.__init__",
    "holders": "app.services.market_data.yfinance.ticker.holders.HoldersClient.__init__",
    "corporate_actions": (
        "app.services.market_data.yfinance.ticker.corporate_actions"
        ".CorporateActionsClient.__init__"
    ),
    "metadata": "app.services.market_data.yfinance.ticker.metadata.MetadataClient.__init__",
    "funds": "app.services.market_data.yfinance.ticker.funds.FundsClient.__init__",
    "market": "app.services.market_data.yfinance.market.market.MarketClient.__init__",
    "sectors": (
        "app.services.market_data.yfinance.market.sector_industry.SectorIndustryClient.__init__"
    ),
    "search": "app.services.market_data.yfinance.market.search.SearchClient.__init__",
    "screener": "app.services.market_data.yfinance.market.screener.ScreenerClient.__init__",
    "calendars": "app.services.market_data.yfinance.market.calendars.CalendarsClient.__init__",
    "streaming": "app.services.market_data.yfinance.market.streaming.StreamingClient.__init__",
    "async_streaming": (
        "app.services.market_data.yfinance.market.streaming.AsyncStreamingClient.__init__"
    ),
}

_THREAD_COUNT: int = 50


@pytest.fixture
def fresh_client() -> Iterator[YFinanceClient]:
    """Yield a freshly-built singleton; reset before and after each test."""
    YFinanceClient.reset_instance()
    client = YFinanceClient.get_instance()
    yield client
    YFinanceClient.reset_instance()


def _make_counting_init(
    counters: dict[str, int],
    counter_lock: threading.Lock,
    name: str,
    original_init: Any,
) -> Any:
    """Wrap *original_init* so each invocation increments ``counters[name]``."""

    def counting_init(self: Any, *args: Any, **kwargs: Any) -> None:
        with counter_lock:
            counters[name] += 1
        original_init(self, *args, **kwargs)

    return counting_init


def test_when_50_threads_first_access_each_subclient_then_constructor_runs_once(
    fresh_client: YFinanceClient,
) -> None:
    counters: dict[str, int] = dict.fromkeys(_SUBCLIENT_NAMES, 0)
    counter_lock = threading.Lock()
    barrier = threading.Barrier(_THREAD_COUNT)
    errors: list[BaseException] = []

    with ExitStack() as stack:
        for name, target in _PATCH_TARGETS.items():
            module_path, _, _ = target.rpartition(".")
            cls_path, _, cls_name = module_path.rpartition(".")
            module_name = cls_path
            module = __import__(module_name, fromlist=[cls_name])
            cls = getattr(module, cls_name)
            original_init = cls.__init__
            wrapper = _make_counting_init(counters, counter_lock, name, original_init)
            stack.enter_context(patch.object(cls, "__init__", wrapper))

        def worker() -> None:
            try:
                barrier.wait(timeout=10.0)
                for prop in _SUBCLIENT_NAMES:
                    getattr(fresh_client, prop)
            except BaseException as exc:
                with counter_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(_THREAD_COUNT)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

    assert not errors, f"Worker threads raised: {errors!r}"
    for name in _SUBCLIENT_NAMES:
        assert counters[name] == 1, (
            f"{name!r} constructor invoked {counters[name]} times, expected 1"
        )


def test_when_subclient_property_accessed_then_cached_under_unprefixed_name(
    fresh_client: YFinanceClient,
) -> None:
    for name in _SUBCLIENT_NAMES:
        getattr(fresh_client, name)

    for name in _SUBCLIENT_NAMES:
        assert name in fresh_client.__dict__, (
            f"cached_property '{name}' missing from __dict__"
        )
        assert f"_{name}" not in fresh_client.__dict__, (
            f"underscore-prefixed '_{name}' must not exist (legacy hasattr pattern)"
        )


def test_when_reset_instance_called_then_cached_entries_evicted_before_null() -> None:
    YFinanceClient.reset_instance()
    client = YFinanceClient.get_instance()
    for name in _SUBCLIENT_NAMES:
        getattr(client, name)
    for name in _SUBCLIENT_NAMES:
        assert name in client.__dict__

    YFinanceClient.reset_instance()

    for name in _SUBCLIENT_NAMES:
        assert name not in client.__dict__, (
            f"reset_instance must pop cached '{name}' from __dict__"
        )
    assert YFinanceClient._instance is None


def test_when_property_accessed_twice_then_returns_identical_instance(
    fresh_client: YFinanceClient,
) -> None:
    for name in _SUBCLIENT_NAMES:
        first = getattr(fresh_client, name)
        second = getattr(fresh_client, name)
        assert first is second, f"{name!r} not cached: distinct instances returned"


def test_when_facade_module_inspected_then_no_hasattr_underscore_pattern() -> None:
    import inspect

    from app.services.market_data.yfinance import _facade

    source = inspect.getsource(_facade)
    for name in _SUBCLIENT_NAMES:
        assert f'hasattr(self, "_{name}")' not in source, (
            f"legacy hasattr guard for '_{name}' still present in _facade.py"
        )
        assert f"self._{name} =" not in source, (
            f"legacy '_{name}' assignment still present in _facade.py"
        )
