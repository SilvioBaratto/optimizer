"""Trading 212 annotation-step contract.

Attaches T212 tickers to the yfinance universe by ISIN. ISIN is fetched lazily
(injected here), T212 metadata is mocked, and the repo + session are patched —
no network, no DB. Skips cleanly when Trading 212 is not configured.
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import requests
from yfinance.exceptions import YFRateLimitError

from app.services.market_data.yfinance import MetadataClient, RequestLostError
from app.services.universe import t212_annotate

M = "app.services.universe.t212_annotate"


@contextmanager
def _patched(repo: MagicMock, client: MagicMock):
    sessions: list[MagicMock] = []

    @contextmanager
    def _session_cm():
        session = MagicMock()
        sessions.append(session)
        yield session

    dm = MagicMock()
    dm.get_session = _session_cm
    with (
        patch(f"{M}.database_manager", dm),
        patch(f"{M}.UniverseRepository", return_value=repo),
        patch(f"{M}.build_trading212_client", return_value=client),
    ):
        yield sessions


def _inst(
    ticker: str,
    exchange_id: int = 1,
    *,
    currency_code: str | None = None,
    t212_ticker: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        exchange_id=exchange_id,
        currency_code=currency_code,
        t212_ticker=t212_ticker,
    )


def _client(instruments: list[dict]) -> MagicMock:
    client = MagicMock()
    client.get_instruments.return_value = instruments
    return client


def test_annotates_instrument_matched_by_isin() -> None:
    repo = MagicMock()
    repo.get_active_instruments.return_value = [_inst("AAPL")]
    repo.set_t212_ticker.return_value = True
    client = _client([{"ticker": "AAPL_US_EQ", "isin": "US123"}])

    with _patched(repo, client):
        summary = t212_annotate.run_t212_annotate(isin_lookup=lambda t: "US123")

    repo.set_t212_ticker.assert_called_once_with(
        ticker="AAPL", exchange_id=1, t212_ticker="AAPL_US_EQ"
    )
    assert summary["annotated"] == 1
    assert summary["skipped"] is False


def test_no_isin_match_annotates_nothing() -> None:
    repo = MagicMock()
    repo.get_active_instruments.return_value = [_inst("AAPL")]
    client = _client([{"ticker": "AAPL_US_EQ", "isin": "US123"}])

    with _patched(repo, client):
        summary = t212_annotate.run_t212_annotate(isin_lookup=lambda t: "OTHER")

    repo.set_t212_ticker.assert_not_called()
    assert summary["annotated"] == 0


def test_missing_isin_skips_instrument() -> None:
    repo = MagicMock()
    repo.get_active_instruments.return_value = [_inst("AAPL")]
    client = _client([{"ticker": "AAPL_US_EQ", "isin": "US123"}])

    with _patched(repo, client):
        summary = t212_annotate.run_t212_annotate(isin_lookup=lambda t: None)

    repo.set_t212_ticker.assert_not_called()
    assert summary["annotated"] == 0


def test_annotates_across_multiple_instruments() -> None:
    repo = MagicMock()
    repo.get_active_instruments.return_value = [
        _inst("AAPL"),
        _inst("VOD", 2),
    ]
    repo.set_t212_ticker.return_value = True
    client = _client(
        [
            {"ticker": "AAPL_US_EQ", "isin": "US1"},
            {"ticker": "VODl_EQ", "isin": "GB1"},
        ]
    )
    isins = {"AAPL": "US1", "VOD": "GB1"}

    with _patched(repo, client):
        summary = t212_annotate.run_t212_annotate(isin_lookup=lambda t: isins[t])

    assert summary["annotated"] == 2
    assert summary["total"] == 2


_WORLD_ETF_LINES = [
    {"ticker": "EUNLd_EQ", "isin": "IE00B4L5Y983", "currencyCode": "EUR"},
    {"ticker": "SWDAl_EQ", "isin": "IE00B4L5Y983", "currencyCode": "GBX"},
    {"ticker": "SWDAd_EQ", "isin": "IE00B4L5Y983", "currencyCode": "USD"},
]


def _annotate_one(
    instrument: SimpleNamespace, t212_lines: list[dict], isin: str | None
) -> tuple[MagicMock, dict]:
    repo = MagicMock()
    repo.get_active_instruments.return_value = [instrument]
    repo.set_t212_ticker.return_value = True
    with _patched(repo, _client(t212_lines)):
        summary = t212_annotate.run_t212_annotate(isin_lookup=lambda t: isin)
    return repo, summary


def test_picks_the_t212_line_quoted_in_the_instrument_currency() -> None:
    # Xetra EUR line: the EUR T212 variant, not whichever line the API lists last.
    repo, summary = _annotate_one(
        _inst("EUNL.DE", currency_code="EUR"), _WORLD_ETF_LINES, "IE00B4L5Y983"
    )

    repo.set_t212_ticker.assert_called_once_with(
        ticker="EUNL.DE", exchange_id=1, t212_ticker="EUNLd_EQ"
    )
    assert summary["annotated"] == 1


def test_pence_quoted_instrument_matches_the_gbx_t212_line() -> None:
    # yfinance says GBp, Trading 212 says GBX: same unit, different spelling.
    lines = [*_WORLD_ETF_LINES[1:], _WORLD_ETF_LINES[0]]
    repo, _ = _annotate_one(_inst("SWDA.L", currency_code="GBp"), lines, "IE00B4L5Y983")

    repo.set_t212_ticker.assert_called_once_with(
        ticker="SWDA.L", exchange_id=1, t212_ticker="SWDAl_EQ"
    )


def test_major_currency_match_is_used_when_no_line_shares_the_quote_unit() -> None:
    lines = [
        {"ticker": "AAAl_EQ", "isin": "GB1", "currencyCode": "GBX"},
        {"ticker": "AAAd_EQ", "isin": "GB1", "currencyCode": "EUR"},
    ]
    repo, _ = _annotate_one(_inst("AAA.L", currency_code="GBP"), lines, "GB1")

    repo.set_t212_ticker.assert_called_once_with(
        ticker="AAA.L", exchange_id=1, t212_ticker="AAAl_EQ"
    )


def test_no_annotation_when_several_t212_lines_and_none_in_the_currency() -> None:
    repo, summary = _annotate_one(
        _inst("X.SW", currency_code="CHF"), _WORLD_ETF_LINES, "IE00B4L5Y983"
    )

    repo.set_t212_ticker.assert_not_called()
    assert summary["annotated"] == 0


def test_no_annotation_when_two_t212_lines_share_the_instrument_currency() -> None:
    lines = [
        {"ticker": "EUNLd_EQ", "isin": "IE1", "currencyCode": "EUR"},
        {"ticker": "EUNLp_EQ", "isin": "IE1", "currencyCode": "EUR"},
    ]
    repo, summary = _annotate_one(_inst("EUNL.DE", currency_code="EUR"), lines, "IE1")

    repo.set_t212_ticker.assert_not_called()
    assert summary["annotated"] == 0


def test_sole_t212_line_is_attached_whatever_its_currency() -> None:
    # Nothing to confuse it with: it is the only way to trade this ISIN.
    lines = [{"ticker": "SWDAl_EQ", "isin": "IE1", "currencyCode": "GBX"}]
    repo, _ = _annotate_one(_inst("EUNL.DE", currency_code="EUR"), lines, "IE1")

    repo.set_t212_ticker.assert_called_once_with(
        ticker="EUNL.DE", exchange_id=1, t212_ticker="SWDAl_EQ"
    )


def test_stale_t212_ticker_is_cleared_when_the_isin_left_trading212() -> None:
    lines = [{"ticker": "OTHER_EQ", "isin": "US999", "currencyCode": "USD"}]
    repo, summary = _annotate_one(
        _inst("AAPL", currency_code="USD", t212_ticker="AAPL_US_EQ"), lines, "US123"
    )

    repo.set_t212_ticker.assert_called_once_with(
        ticker="AAPL", exchange_id=1, t212_ticker=None
    )
    assert summary["annotated"] == 0
    assert summary["cleared"] == 1


def test_stale_t212_ticker_is_cleared_when_the_match_became_ambiguous() -> None:
    repo, summary = _annotate_one(
        _inst("X.SW", currency_code="CHF", t212_ticker="OLD_EQ"),
        _WORLD_ETF_LINES,
        "IE00B4L5Y983",
    )

    repo.set_t212_ticker.assert_called_once_with(
        ticker="X.SW", exchange_id=1, t212_ticker=None
    )
    assert summary["cleared"] == 1


def test_t212_ticker_is_kept_when_yahoo_returns_no_isin() -> None:
    # No ISIN is "unknown", not "not on Trading 212": nothing proves it stale.
    repo, summary = _annotate_one(
        _inst("AAPL", t212_ticker="AAPL_US_EQ"),
        [{"ticker": "AAPL_US_EQ", "isin": "US123", "currencyCode": "USD"}],
        None,
    )

    repo.set_t212_ticker.assert_not_called()
    assert summary["cleared"] == 0


def test_t212_ticker_is_kept_when_the_isin_lookup_errors() -> None:
    repo = MagicMock()
    repo.get_active_instruments.return_value = [_inst("AAPL", t212_ticker="AAPL_US_EQ")]

    def lookup(ticker: str) -> str:
        raise t212_annotate.IsinLookupError("throttled")

    client = _client([{"ticker": "AAPL_US_EQ", "isin": "US123"}])
    with _patched(repo, client):
        summary = t212_annotate.run_t212_annotate(isin_lookup=lookup)

    repo.set_t212_ticker.assert_not_called()
    assert summary["cleared"] == 0


def test_unmatched_instrument_without_a_t212_ticker_is_left_alone() -> None:
    repo, _ = _annotate_one(
        _inst("AAPL"),
        [{"ticker": "OTHER_EQ", "isin": "US999", "currencyCode": "USD"}],
        "US123",
    )

    repo.set_t212_ticker.assert_not_called()


def _yf_client(*, isin: str | None, lost: Exception | None = None) -> MagicMock:
    yf_client = MagicMock()
    if lost is not None:
        yf_client.metadata.fetch_isin.side_effect = lost
    else:
        yf_client.metadata.fetch_isin.return_value = isin
    return yf_client


def test_default_isin_lookup_goes_through_the_shared_yfinance_client(
    monkeypatch,
) -> None:
    yf_client = _yf_client(isin="US999")
    monkeypatch.setattr(f"{M}.get_yfinance_client", lambda: yf_client)
    monkeypatch.setattr(
        "yfinance.Ticker", MagicMock(side_effect=AssertionError("raw yfinance"))
    )

    assert t212_annotate._default_isin_lookup("AAPL") == "US999"
    yf_client.metadata.fetch_isin.assert_called_once_with("AAPL", raise_on_lost=True)


def test_default_isin_lookup_returns_none_when_yahoo_has_no_isin(monkeypatch) -> None:
    monkeypatch.setattr(f"{M}.get_yfinance_client", lambda: _yf_client(isin=None))
    assert t212_annotate._default_isin_lookup("AAPL") is None


def test_default_isin_lookup_raises_when_the_request_was_lost(monkeypatch) -> None:
    # A lost request is not Yahoo's "no ISIN", whatever the cause.
    lost = RequestLostError("every attempt failed")
    monkeypatch.setattr(
        f"{M}.get_yfinance_client", lambda: _yf_client(isin=None, lost=lost)
    )

    with pytest.raises(t212_annotate.IsinLookupError, match="AAPL") as excinfo:
        t212_annotate._default_isin_lookup("AAPL")

    assert excinfo.value.__cause__ is lost


def _real_metadata_client(failure: Exception | None, isin: str | None = None):
    """Shared yfinance client whose only live part is the real MetadataClient."""
    ticker = MagicMock()
    type(ticker).isin = PropertyMock(side_effect=failure, return_value=isin)
    cache = MagicMock()
    cache.get.return_value = ticker
    # A closed breaker is what a DNS or garbled-body outage leaves behind: only a
    # rate limit ever opens it.
    breaker = MagicMock(is_active=False)
    metadata = MetadataClient(
        cache=cache,
        rate_limiter=MagicMock(),
        circuit_breaker=breaker,
        default_max_retries=1,
    )
    return SimpleNamespace(metadata=metadata, circuit_breaker=breaker)


_OUTAGES = [
    pytest.param(
        requests.exceptions.ConnectionError("Failed to resolve 'query1.finance'"),
        id="dns-failure",
    ),
    pytest.param(json.JSONDecodeError("Expecting value", "<html>", 0), id="bad-json"),
    pytest.param(YFRateLimitError(), id="rate-limited"),
]


@pytest.mark.parametrize("failure", _OUTAGES)
def test_outage_fails_the_step_instead_of_annotating_nothing(
    monkeypatch, failure: Exception
) -> None:
    # Through the real MetadataClient: only the throttled case used to be noticed,
    # an unreachable or garbled source finished as a clean run with annotated=0.
    monkeypatch.setattr(
        f"{M}.get_yfinance_client", lambda: _real_metadata_client(failure)
    )
    limit = t212_annotate._MAX_CONSECUTIVE_LOOKUP_ERRORS
    repo = MagicMock()
    repo.get_active_instruments.return_value = [_inst(f"T{i}") for i in range(limit)]
    on_progress = MagicMock()

    with _patched(repo, _client([{"ticker": "T0_EQ", "isin": "ISIN0"}])):
        with pytest.raises(t212_annotate.IsinLookupError, match="consecutive"):
            t212_annotate.run_t212_annotate(on_progress=on_progress)

    assert not any(
        c.kwargs.get("status") == "completed" for c in on_progress.call_args_list
    )


def test_yahoo_having_no_isin_for_every_ticker_is_not_an_outage(monkeypatch) -> None:
    monkeypatch.setattr(
        f"{M}.get_yfinance_client", lambda: _real_metadata_client(None, isin="-")
    )
    limit = t212_annotate._MAX_CONSECUTIVE_LOOKUP_ERRORS
    repo = MagicMock()
    repo.get_active_instruments.return_value = [
        _inst(f"T{i}") for i in range(limit * 2)
    ]

    with _patched(repo, _client([{"ticker": "T0_EQ", "isin": "ISIN0"}])):
        summary = t212_annotate.run_t212_annotate()

    assert summary["lookup_errors"] == 0
    assert summary["annotated"] == 0


def test_isolated_lookup_error_does_not_abort_the_step() -> None:
    repo = MagicMock()
    repo.get_active_instruments.return_value = [_inst(t) for t in ("A", "B", "C")]
    repo.set_t212_ticker.return_value = True
    client = _client([{"ticker": f"{t}_EQ", "isin": f"ISIN{t}"} for t in "AC"])

    def lookup(ticker: str) -> str:
        if ticker == "B":
            raise t212_annotate.IsinLookupError("throttled")
        return f"ISIN{ticker}"

    with _patched(repo, client):
        summary = t212_annotate.run_t212_annotate(isin_lookup=lookup)

    assert summary["annotated"] == 2
    assert summary["lookup_errors"] == 1


def test_step_fails_instead_of_completing_when_lookups_keep_erroring() -> None:
    limit = t212_annotate._MAX_CONSECUTIVE_LOOKUP_ERRORS
    repo = MagicMock()
    repo.get_active_instruments.return_value = [
        _inst(f"T{i}") for i in range(limit * 3)
    ]
    repo.set_t212_ticker.return_value = True
    client = _client([{"ticker": "T0_EQ", "isin": "ISIN0"}])
    on_progress = MagicMock()
    calls: list[str] = []

    def lookup(ticker: str) -> str:
        calls.append(ticker)
        if ticker == "T0":
            return "ISIN0"
        raise t212_annotate.IsinLookupError("throttled")

    with _patched(repo, client) as sessions:
        with pytest.raises(t212_annotate.IsinLookupError, match="consecutive"):
            t212_annotate.run_t212_annotate(isin_lookup=lookup, on_progress=on_progress)

    assert len(calls) == limit + 1  # aborted early, not after the whole universe
    repo.set_t212_ticker.assert_called_once()  # work done before the outage is kept
    sessions[0].commit.assert_called()
    assert not any(
        c.kwargs.get("status") == "completed" for c in on_progress.call_args_list
    )


def test_successful_lookup_resets_the_consecutive_error_streak() -> None:
    limit = t212_annotate._MAX_CONSECUTIVE_LOOKUP_ERRORS
    tickers = [f"T{i}" for i in range(limit * 2)]
    repo = MagicMock()
    repo.get_active_instruments.return_value = [_inst(t) for t in tickers]
    repo.set_t212_ticker.return_value = True
    client = _client([{"ticker": "OK_EQ", "isin": "OKISIN"}])

    def lookup(ticker: str) -> str:
        if int(ticker[1:]) % limit == limit - 1:  # one success per `limit` calls
            return "OKISIN"
        raise t212_annotate.IsinLookupError("throttled")

    with _patched(repo, client):
        summary = t212_annotate.run_t212_annotate(isin_lookup=lookup)

    assert summary["total"] == len(tickers)
    assert summary["lookup_errors"] == len(tickers) - 2


def test_skips_cleanly_when_trading212_not_configured() -> None:
    from app.services.universe.universe_build_service import (
        Trading212NotConfiguredError,
    )

    with patch(
        f"{M}.build_trading212_client",
        side_effect=Trading212NotConfiguredError("no key"),
    ):
        summary = t212_annotate.run_t212_annotate(isin_lookup=lambda t: "US123")

    assert summary["skipped"] is True
    assert summary["annotated"] == 0
