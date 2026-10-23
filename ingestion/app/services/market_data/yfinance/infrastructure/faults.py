"""Surface transient Yahoo failures that yfinance folds into empty results.

In its default mode yfinance catches rate limits, server errors and network
faults inside its scrapers, returns an empty frame and caches it on the
``yf.Ticker``. An empty result alone therefore cannot be told apart from
Yahoo's "no data" answer, and the retry layer would accept it as final.

``YfData._make_request`` is the one place every request passes, so a thin
wrapper there notes the fault for the call in progress. Sub-clients read the
note once the call returns and retry only when the result is also unusable.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from yfinance.data import YfData
from yfinance.exceptions import YFRateLimitError

# 404 is deliberately absent: Yahoo answers it for a data module a symbol lacks.
_TRANSIENT_STATUSES = frozenset({401, 403, 408, 429})
_SERVER_ERROR_FLOOR = 500
_WITNESS_MARKER = "_portopt_fault_witness"

_install_lock = threading.Lock()
_local = threading.local()


class YahooTransientError(Exception):
    """Yahoo answered with a status that a later attempt may not repeat."""

    def __init__(self, status: int) -> None:
        super().__init__(f"Yahoo answered HTTP {status}")
        self.status = status


@dataclass
class FaultWatch:
    """What one watched call saw while yfinance made its requests.

    Attributes:
        fault: The most telling transient fault, or ``None`` when every request
            succeeded. A rate limit outranks other faults so the circuit
            breaker still trips.
        symbols: Symbols whose cached ``yf.Ticker`` the call used, so a caller
            can drop tickers that now hold swallowed-failure state.
    """

    fault: Exception | None = None
    symbols: set[str] = field(default_factory=set)

    def record(self, fault: Exception) -> None:
        """Keep *fault* unless an equal or more telling one is already held."""
        if self.fault is None or (
            isinstance(fault, YFRateLimitError)
            and not isinstance(self.fault, YFRateLimitError)
        ):
            self.fault = fault


@contextmanager
def watch_for_faults() -> Iterator[FaultWatch]:
    """Collect transient faults raised by yfinance requests on this thread.

    Watches nest: the previous one is restored on exit. Other threads are never
    affected, so concurrent fetch workers keep separate records.

    Yields:
        The watch to read after the wrapped call returns.
    """
    previous: FaultWatch | None = getattr(_local, "watch", None)
    watch = FaultWatch()
    _local.watch = watch
    try:
        yield watch
    finally:
        _local.watch = previous


def note_ticker(symbol: str) -> None:
    """Tell the active watch that *symbol*'s cached ticker is in use."""
    watch: FaultWatch | None = getattr(_local, "watch", None)
    if watch is not None:
        watch.symbols.add(symbol)


def forget_cached_responses() -> None:
    """Drop yfinance's response cache so a retry reaches Yahoo again.

    ``YfData.cache_get`` memoises whole responses, error statuses included, so
    a retried URL would otherwise replay the failure without a request.
    """
    clear = getattr(YfData.cache_get, "cache_clear", None)
    if clear is not None:
        clear()


def install_fault_witness() -> None:
    """Wrap ``YfData._make_request`` so watched calls record transient faults.

    Idempotent and process-wide. Outside a :func:`watch_for_faults` block the
    wrapper only forwards the call and its result unchanged.
    """
    with _install_lock:
        original = YfData._make_request
        if getattr(original, _WITNESS_MARKER, False):
            return

        @functools.wraps(original)
        def witnessed(self: YfData, *args: Any, **kwargs: Any) -> Any:
            """Forward one request, noting an escaping error or a transient status.

            Returns:
                The response of the wrapped request, unchanged.

            Raises:
                Exception: Whatever the wrapped request raised.
            """
            watch: FaultWatch | None = getattr(_local, "watch", None)
            try:
                response = original(self, *args, **kwargs)
            except Exception as exc:
                if watch is not None:
                    watch.record(exc)
                raise
            status: int = getattr(response, "status_code", 0)
            if watch is not None and (
                status >= _SERVER_ERROR_FLOOR or status in _TRANSIENT_STATUSES
            ):
                watch.record(YahooTransientError(status))
            return response

        setattr(witnessed, _WITNESS_MARKER, True)
        setattr(YfData, "_make_request", witnessed)  # noqa: B010
