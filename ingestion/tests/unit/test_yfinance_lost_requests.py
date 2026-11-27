"""A yfinance request that never got an answer is distinguishable from "no data".

``fetch_with_resilience`` returns ``None`` both when Yahoo answers that it holds
nothing and when every attempt failed (DNS, TLS, a truncated body, throttling).
Callers that must not mistake an outage for an answer opt in with
``raise_on_lost`` and get :class:`RequestLostError` for the second case only.

Tickers are mocks; ``yf.Ticker`` is never constructed against the network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import requests
from yfinance.exceptions import YFRateLimitError, YFTickerMissingError

from app.services.market_data.yfinance import RequestLostError
from app.services.market_data.yfinance.ticker import MetadataClient


@pytest.fixture(autouse=True)
def _no_backoff_sleep() -> Iterator[None]:
    with patch("app.services.infrastructure.retry._full_jitter", return_value=0.0):
        yield


def _metadata(options: PropertyMock) -> tuple[MetadataClient, MagicMock]:
    ticker = MagicMock()
    type(ticker).options = options
    cache = MagicMock()
    cache.get.return_value = ticker
    breaker = MagicMock()
    client = MetadataClient(
        cache=cache,
        rate_limiter=MagicMock(),
        circuit_breaker=breaker,
        default_max_retries=3,
    )
    return client, breaker


_EXPIRATIONS = ("2026-12-18", "2027-01-15")

_LOST = [
    pytest.param(
        requests.exceptions.ConnectionError("Failed to resolve 'query1.finance'"),
        id="dns-failure",
    ),
    pytest.param(json.JSONDecodeError("Expecting value", "<html>", 0), id="bad-json"),
    pytest.param(requests.exceptions.SSLError("handshake failed"), id="tls-failure"),
    pytest.param(YFRateLimitError(), id="rate-limited"),
]


@pytest.mark.criterion("REQ-ING-021")
@pytest.mark.parametrize("failure", _LOST)
def test_every_attempt_failing_raises_when_requested(failure: Exception) -> None:
    """Validates ``MetadataClient.fetch_options_expirations``: it raises when lost.

    With ``raise_on_lost`` set, a request whose every attempt failed ends in
    ``RequestLostError`` caused by the last fault, after the whole retry budget.
    """
    options = PropertyMock(side_effect=failure)
    client, _ = _metadata(options)

    with pytest.raises(RequestLostError) as excinfo:
        client.fetch_options_expirations("AAPL", raise_on_lost=True)

    assert excinfo.value.__cause__ is failure
    assert options.call_count == 3  # the whole retry budget was spent first


@pytest.mark.criterion("REQ-ING-021")
@pytest.mark.parametrize("failure", _LOST)
def test_every_attempt_failing_returns_none_by_default(failure: Exception) -> None:
    """Validates ``MetadataClient.fetch_options_expirations``: no raise by default."""
    client, _ = _metadata(PropertyMock(side_effect=failure))

    assert client.fetch_options_expirations("AAPL") is None


@pytest.mark.criterion("REQ-ING-021")
def test_rate_limit_still_trips_the_breaker_before_raising() -> None:
    """Validates ``MetadataClient.fetch_options_expirations``: a rate limit trips.

    The breaker is tripped by the throttled attempts before the loss is raised.
    """
    client, breaker = _metadata(PropertyMock(side_effect=YFRateLimitError()))

    with pytest.raises(RequestLostError):
        client.fetch_options_expirations("AAPL", raise_on_lost=True)

    assert breaker.trigger.called


@pytest.mark.criterion("REQ-ING-021")
def test_breaker_refusing_every_attempt_counts_as_lost() -> None:
    """Validates ``MetadataClient.fetch_options_expirations``: an open breaker is lost.

    A breaker that refuses every attempt sends no request and counts as a loss.
    """
    options = PropertyMock(return_value=_EXPIRATIONS)
    client, breaker = _metadata(options)
    breaker.check.side_effect = RuntimeError("circuit breaker open")

    with pytest.raises(RequestLostError):
        client.fetch_options_expirations("AAPL", raise_on_lost=True)

    options.assert_not_called()


@pytest.mark.criterion("REQ-ING-021")
@pytest.mark.parametrize("answer", [None, ()])
def test_yahoo_listing_no_options_is_an_answer_not_a_loss(
    answer: tuple[str, ...] | None,
) -> None:
    """Validates ``MetadataClient.fetch_options_expirations``: no data is an answer.

    A listing without options returns ``None`` after one attempt, not a loss.
    """
    options = PropertyMock(return_value=answer)
    client, _ = _metadata(options)

    assert client.fetch_options_expirations("AAPL", raise_on_lost=True) is None
    assert options.call_count == 1


@pytest.mark.criterion("REQ-ING-021")
def test_no_data_exception_is_an_answer_not_a_loss() -> None:
    """Validates ``MetadataClient.fetch_options_expirations``: a no-data fault is final.

    Yahoo's missing-ticker exception is an answer, so it is not retried or raised.
    """
    options = PropertyMock(side_effect=YFTickerMissingError("AAPL", "no ticker"))
    client, _ = _metadata(options)

    assert client.fetch_options_expirations("AAPL", raise_on_lost=True) is None
    assert options.call_count == 1


@pytest.mark.criterion("REQ-ING-021")
def test_transient_failure_followed_by_an_answer_is_not_lost() -> None:
    """Validates ``MetadataClient.fetch_options_expirations``: a retry can recover.

    A transient fault followed by a real answer returns the answer, with no loss.
    """
    options = PropertyMock(
        side_effect=[requests.exceptions.ConnectionError("reset"), _EXPIRATIONS]
    )
    client, _ = _metadata(options)

    assert client.fetch_options_expirations("AAPL", raise_on_lost=True) == _EXPIRATIONS


@pytest.mark.criterion("REQ-ING-021")
def test_answer_followed_by_nothing_to_retry_is_not_lost_after_an_earlier_failure() -> (
    None
):
    """Validates ``MetadataClient.fetch_options_expirations``: the last attempt rules.

    The first attempt failed and the second returned an empty tuple (Yahoo's "no
    options listed" answer). The final state is an answer, so the earlier failure
    must not leak out as a loss.
    """
    options = PropertyMock(side_effect=[requests.exceptions.ConnectionError("x"), ()])
    client, _ = _metadata(options)

    assert client.fetch_options_expirations("AAPL", raise_on_lost=True) is None
