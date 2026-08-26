"""Trading 212 annotation-step contract (SPEC D14, task T11b).

Attaches T212 tickers to the yfinance universe by ISIN. ISIN is fetched lazily
(injected here), T212 metadata is mocked, and the repo + session are patched —
no network, no DB. Skips cleanly when Trading 212 is not configured.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.universe import t212_annotate

M = "app.services.universe.t212_annotate"


@contextmanager
def _patched(repo: MagicMock, client: MagicMock):
    @contextmanager
    def _session_cm():
        yield MagicMock()

    dm = MagicMock()
    dm.get_session = _session_cm
    with (
        patch(f"{M}.database_manager", dm),
        patch(f"{M}.UniverseRepository", return_value=repo),
        patch(f"{M}.build_trading212_client", return_value=client),
    ):
        yield


def _client(instruments: list[dict]) -> MagicMock:
    client = MagicMock()
    client.get_instruments.return_value = instruments
    return client


def test_annotates_instrument_matched_by_isin() -> None:
    repo = MagicMock()
    repo.get_active_instruments.return_value = [
        SimpleNamespace(ticker="AAPL", exchange_id=1)
    ]
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
    repo.get_active_instruments.return_value = [
        SimpleNamespace(ticker="AAPL", exchange_id=1)
    ]
    client = _client([{"ticker": "AAPL_US_EQ", "isin": "US123"}])

    with _patched(repo, client):
        summary = t212_annotate.run_t212_annotate(isin_lookup=lambda t: "OTHER")

    repo.set_t212_ticker.assert_not_called()
    assert summary["annotated"] == 0


def test_missing_isin_skips_instrument() -> None:
    repo = MagicMock()
    repo.get_active_instruments.return_value = [
        SimpleNamespace(ticker="AAPL", exchange_id=1)
    ]
    client = _client([{"ticker": "AAPL_US_EQ", "isin": "US123"}])

    with _patched(repo, client):
        summary = t212_annotate.run_t212_annotate(isin_lookup=lambda t: None)

    repo.set_t212_ticker.assert_not_called()
    assert summary["annotated"] == 0


def test_annotates_across_multiple_instruments() -> None:
    repo = MagicMock()
    repo.get_active_instruments.return_value = [
        SimpleNamespace(ticker="AAPL", exchange_id=1),
        SimpleNamespace(ticker="VOD", exchange_id=2),
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


def test_default_isin_lookup_uses_yfinance(monkeypatch) -> None:
    ticker = MagicMock()
    ticker.isin = "US999"
    monkeypatch.setattr("yfinance.Ticker", lambda _t: ticker)
    assert t212_annotate._default_isin_lookup("AAPL") == "US999"


def test_default_isin_lookup_swallows_errors(monkeypatch) -> None:
    def _boom(_t):
        raise RuntimeError("network down")

    monkeypatch.setattr("yfinance.Ticker", _boom)
    assert t212_annotate._default_isin_lookup("AAPL") is None


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
