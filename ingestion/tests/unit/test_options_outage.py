"""The options sweep must not report an outage as a clean, empty run.

``fetch_options_expirations`` returned ``None`` for a ticker without options and for
a request that never got an answer, and the sweep turned both into "no options,
processed". Through a Yahoo outage every instrument then ended as processed with
no error and the job completed.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch
from uuid import uuid4

import pytest
import requests

from app.services.market_data.options_service import run_bulk_options_fetch
from app.services.market_data.yfinance import RequestLostError
from app.services.market_data.yfinance.ticker import MetadataClient
from app.services.market_data.yfinance_data_service import YahooUnreachableError

M = "app.services.market_data.options_service"


@pytest.fixture(autouse=True)
def _no_backoff_sleep() -> Iterator[None]:
    with patch("app.services.infrastructure.retry._full_jitter", return_value=0.0):
        yield


def _metadata(options: PropertyMock) -> MetadataClient:
    ticker = MagicMock()
    type(ticker).options = options
    cache = MagicMock()
    cache.get.return_value = ticker
    return MetadataClient(
        cache=cache,
        rate_limiter=MagicMock(),
        circuit_breaker=MagicMock(),
        default_max_retries=2,
    )


class TestExpirationsLookup:
    """Telling a lost request apart from a ticker without options."""

    def test_a_lost_request_raises_when_requested(self) -> None:
        """Raise once every attempt failed, after spending the retry budget."""
        options = PropertyMock(side_effect=requests.exceptions.ConnectionError("down"))
        client = _metadata(options)

        with pytest.raises(RequestLostError):
            client.fetch_options_expirations("AAPL", raise_on_lost=True)

        assert options.call_count == 2

    def test_a_ticker_without_options_is_an_answer(self) -> None:
        """Return None without raising when Yahoo lists no expirations."""
        client = _metadata(PropertyMock(return_value=()))

        assert client.fetch_options_expirations("AAPL", raise_on_lost=True) is None

    def test_default_keeps_returning_none_for_a_lost_request(self) -> None:
        """Do not change the contract for callers that did not opt in."""
        client = _metadata(
            PropertyMock(side_effect=requests.exceptions.ConnectionError("down"))
        )

        assert client.fetch_options_expirations("AAPL") is None


def _session_ctx() -> MagicMock:
    @contextmanager
    def _cm():
        yield MagicMock(name="session")

    dbm = MagicMock()
    dbm.get_session = _cm
    return dbm


def _repo(count: int) -> MagicMock:
    repo = MagicMock(name="repo")
    repo.get_instruments_with_yfinance_ticker.return_value = [
        SimpleNamespace(id=uuid4(), yfinance_ticker=f"T{i}") for i in range(count)
    ]
    repo.get_options_as_of_bulk.return_value = {}
    return repo


class TestSweepThroughAnOutage:
    """What the sweep reports when Yahoo does not answer."""

    def test_a_lost_expirations_request_is_an_instrument_error(self) -> None:
        """Count the instrument as failed, not processed, and name it."""
        repo = _repo(2)
        yf = MagicMock()
        yf.metadata.fetch_options_expirations.side_effect = [
            RequestLostError("lost"),
            (),
        ]
        progress = MagicMock()
        with (
            patch(f"{M}.YFinanceRepository", return_value=repo),
            patch(f"{M}.call_with_timeout", lambda fn, _t, **_: fn()),
            patch("app.database.database_manager", _session_ctx()),
        ):
            result = run_bulk_options_fetch(yf, on_progress=progress)

        assert result["error_count"] == 1
        assert result["instruments_processed"] == 1
        errors = progress.call_args_list[-1].kwargs["errors"]
        assert errors and errors[0].startswith("T0")

    def test_the_expirations_lookup_asks_to_be_told_about_lost_requests(self) -> None:
        """Opt in to raise_on_lost on every expirations call."""
        repo = _repo(1)
        yf = MagicMock()
        yf.metadata.fetch_options_expirations.return_value = ()
        with (
            patch(f"{M}.YFinanceRepository", return_value=repo),
            patch(f"{M}.call_with_timeout", lambda fn, _t, **_: fn()),
            patch("app.database.database_manager", _session_ctx()),
        ):
            run_bulk_options_fetch(yf)

        yf.metadata.fetch_options_expirations.assert_called_once_with(
            "T0", raise_on_lost=True
        )

    def test_a_sweep_in_which_most_requests_got_no_answer_fails_the_job(self) -> None:
        """End as an outage, like the other yfinance sweeps, instead of completed."""
        from app.services.market_data.yfinance import record_failure

        repo = _repo(4)
        yf = MagicMock()

        def _lost(_ticker: str, **_kw: object) -> None:
            record_failure("ConnectionError: down")
            raise RequestLostError("lost")

        yf.metadata.fetch_options_expirations.side_effect = _lost
        progress = MagicMock()
        with (
            patch(f"{M}.YFinanceRepository", return_value=repo),
            patch(f"{M}.call_with_timeout", lambda fn, _t, **_: fn()),
            patch("app.database.database_manager", _session_ctx()),
            pytest.raises(YahooUnreachableError),
        ):
            run_bulk_options_fetch(yf, on_progress=progress)

        statuses = [c.kwargs.get("status") for c in progress.call_args_list]
        assert "completed" not in statuses
