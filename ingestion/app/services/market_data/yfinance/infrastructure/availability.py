"""Separate Yahoo's final "no data" answers from failures worth retrying.

yfinance raises some "no data" answers even in its default mode (an empty
upgrade/downgrade history, a missing price series, an HTTP error outside the
paths it guards). Repeating those requests returns the same answer, so the
retry loop stops on them while network faults and rate limits keep retrying.
"""

from __future__ import annotations

from http import HTTPStatus

from yfinance.exceptions import YFDataException, YFTickerMissingError


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
