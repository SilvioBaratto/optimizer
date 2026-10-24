"""Separate Yahoo's final "no data" answers from failures worth retrying.

yfinance raises some "no data" answers even in its default mode (an empty
upgrade/downgrade history, a missing price series, an HTTP error outside the
paths it guards). Repeating those requests returns the same answer, so the
retry loop stops on them while network faults and rate limits keep retrying.

A request that still fails after its retries is either Yahoo being unreachable
or refusing service (a transport error) or Yahoo's content being unusable;
only the first says anything about the connection, so callers that watch for
outages tell them apart with :func:`is_transport_error`.
"""

from __future__ import annotations

from http import HTTPStatus

import requests
from yfinance.exceptions import (
    YFDataException,
    YFRateLimitError,
    YFTickerMissingError,
)

from app.services.infrastructure.circuit_breaker import CircuitOpenError

from .faults import YahooTransientError


def is_symbol_not_found(error: BaseException) -> bool:
    """Report whether *error* is an HTTP 404 from Yahoo.

    Reads ``error.response.status_code`` by duck typing, so it covers both the
    ``requests`` and the ``curl_cffi`` ``HTTPError`` that yfinance may raise.

    Args:
        error: Any exception raised by a yfinance call.

    Returns:
        True only when the exception carries a response with status 404.
    """
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == HTTPStatus.NOT_FOUND


def is_no_data_error(error: BaseException) -> bool:
    """Report whether *error* is Yahoo answering that it holds no such data.

    Covers a 404, yfinance's missing-ticker family (no prices, no timezone, no
    earnings dates) and ``YFDataException``, which yfinance raises for an empty
    or unparseable payload. Rate limits, network faults and timeouts are
    excluded because a later attempt may succeed.

    Args:
        error: Any exception raised by a yfinance call.

    Returns:
        True when repeating the request would return the same empty answer.
    """
    return isinstance(
        error, (YFTickerMissingError, YFDataException)
    ) or is_symbol_not_found(error)


def is_transport_error(error: BaseException) -> bool:
    """Report whether *error* means Yahoo could not be reached or refused service.

    Covers dropped, reset and timed-out connections, HTTP error statuses
    (rate limits and server faults), yfinance's rate-limit error, a transient
    status that yfinance swallowed and the fault witness raised again, and a circuit
    breaker that stopped calling after repeated rate limits. Yahoo's no-data
    answers and a body it sent but that cannot be parsed (missing keys, invalid
    JSON) are its content, not the connection, and are excluded. ``curl_cffi``,
    the HTTP stack of yfinance 1.x, shares no base class with ``requests``, so
    its errors are recognised by origin.

    Args:
        error: Any exception raised by a yfinance call.

    Returns:
        True when the failure is about reaching Yahoo rather than what it said.
    """
    if is_no_data_error(error) or isinstance(
        error, requests.exceptions.JSONDecodeError
    ):
        return False
    if isinstance(
        error,
        (
            ConnectionError,
            TimeoutError,
            CircuitOpenError,
            YFRateLimitError,
            YahooTransientError,
            requests.exceptions.RequestException,
        ),
    ):
        return True
    return type(error).__module__.partition(".")[0] == "curl_cffi"
