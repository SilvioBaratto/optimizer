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


def _metadata(isin: PropertyMock) -> tuple[MetadataClient, MagicMock]:
    ticker = MagicMock()
    type(ticker).isin = isin
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


_LOST = [
    pytest.param(
        requests.exceptions.ConnectionError("Failed to resolve 'query1.finance'"),
        id="dns-failure",
    ),
    pytest.param(json.JSONDecodeError("Expecting value", "<html>", 0), id="bad-json"),
    pytest.param(requests.exceptions.SSLError("handshake failed"), id="tls-failure"),
    pytest.param(YFRateLimitError(), id="rate-limited"),
]


@pytest.mark.parametrize("failure", _LOST)
def test_every_attempt_failing_raises_when_requested(failure: Exception) -> None:
    isin = PropertyMock(side_effect=failure)
    client, _ = _metadata(isin)

    with pytest.raises(RequestLostError) as excinfo:
        client.fetch_isin("AAPL", raise_on_lost=True)

    assert excinfo.value.__cause__ is failure
    assert isin.call_count == 3  # the whole retry budget was spent first


@pytest.mark.parametrize("failure", _LOST)
def test_every_attempt_failing_returns_none_by_default(failure: Exception) -> None:
    client, _ = _metadata(PropertyMock(side_effect=failure))

    assert client.fetch_isin("AAPL") is None


def test_rate_limit_still_trips_the_breaker_before_raising() -> None:
    client, breaker = _metadata(PropertyMock(side_effect=YFRateLimitError()))

    with pytest.raises(RequestLostError):
        client.fetch_isin("AAPL", raise_on_lost=True)

    assert breaker.trigger.called


def test_breaker_refusing_every_attempt_counts_as_lost() -> None:
    isin = PropertyMock(return_value="US123")
    client, breaker = _metadata(isin)
    breaker.check.side_effect = RuntimeError("rate limit persists")

    with pytest.raises(RequestLostError):
        client.fetch_isin("AAPL", raise_on_lost=True)

    isin.assert_not_called()


@pytest.mark.parametrize("answer", [None, "", "-"])
def test_yahoo_having_no_isin_is_an_answer_not_a_loss(answer: str | None) -> None:
    isin = PropertyMock(return_value=answer)
    client, _ = _metadata(isin)

    assert client.fetch_isin("AAPL", raise_on_lost=True) is None
    assert isin.call_count == 1


def test_no_data_exception_is_an_answer_not_a_loss() -> None:
    isin = PropertyMock(side_effect=YFTickerMissingError("AAPL", "no ticker"))
    client, _ = _metadata(isin)

    assert client.fetch_isin("AAPL", raise_on_lost=True) is None
    assert isin.call_count == 1


def test_transient_failure_followed_by_an_answer_is_not_lost() -> None:
    isin = PropertyMock(
        side_effect=[requests.exceptions.ConnectionError("reset"), "US0378331005"]
    )
    client, _ = _metadata(isin)

    assert client.fetch_isin("AAPL", raise_on_lost=True) == "US0378331005"


def test_answer_followed_by_nothing_to_retry_is_not_lost_after_an_earlier_failure() -> (
    None
):
    # The first attempt failed, the second returned "-" (Yahoo's no-ISIN marker):
    # the final state is an answer, so the earlier failure must not leak out.
    isin = PropertyMock(side_effect=[requests.exceptions.ConnectionError("x"), "-"])
    client, _ = _metadata(isin)

    assert client.fetch_isin("AAPL", raise_on_lost=True) is None
