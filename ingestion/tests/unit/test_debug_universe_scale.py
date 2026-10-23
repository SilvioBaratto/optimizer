"""Behaviour of the per-instrument probe in ``scripts/debug_universe_scale.py``.

The probe is driven with autospec'd collaborators, so a call to a method the
real mapper or client no longer has fails here the way it would in production.
No network, database or API key is touched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from app.services.market_data.yfinance import YFinanceClient
from app.services.universe.trading212.ticker_mapper import YFinanceTickerMapper

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "debug_universe_scale.py"

_INFO = {f"field_{i}": i for i in range(5)}


def _load_script() -> ModuleType:
    # Loaded by path: the repository root also carries a ``scripts`` directory,
    # so importing ``scripts.debug_universe_scale`` is ambiguous.
    spec = importlib.util.spec_from_file_location(
        "debug_universe_scale_script", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """The script module, loaded once for the whole file."""
    return _load_script()


def _frame(rows: int = 3) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=rows, freq="B")
    return pd.DataFrame({"Close": range(rows)}, index=index)


@pytest.fixture
def mapper():
    """A mapper that resolves every symbol to ``AAPL``."""
    mapped = create_autospec(YFinanceTickerMapper, instance=True)
    mapped.discover.return_value = "AAPL"
    return mapped


@pytest.fixture
def ticker():
    """The cached ``yf.Ticker`` the probe reads history from: three rows."""
    fake = MagicMock()
    fake.history.return_value = _frame()
    return fake


@pytest.fixture
def client(ticker):
    """A client with a usable profile whose ``get_ticker`` returns *ticker*."""
    fake = create_autospec(YFinanceClient, instance=True)
    fake.fetch_info.return_value = _INFO
    fake.get_ticker.return_value = ticker
    return fake


def _inst() -> dict:
    return {"shortName": "AAPL_US_EQ"}


def test_probe_takes_info_ok_from_the_client_fetch_info(script, mapper, client):
    """The info check goes through the client, not a mapper method that is gone."""
    out = script._probe(mapper, client, "NASDAQ", _inst())

    assert out["mapped"] is True
    assert out["info_ok"] is True
    assert out["hist_rows"] == 3
    client.fetch_info.assert_called_once_with("AAPL", max_retries=1, min_fields=5)


def test_probe_reports_info_not_ok_when_yahoo_has_no_profile(script, mapper, client):
    """A ``None`` profile is a clean negative, not an error."""
    client.fetch_info.return_value = None

    out = script._probe(mapper, client, "NASDAQ", _inst())

    assert out["info_ok"] is False
    assert out.get("hist_err") is None


def test_probe_records_a_worker_error_instead_of_raising(script, mapper, client):
    """One failing symbol must not abort the pool through ``fut.result()``."""
    client.fetch_info.side_effect = RuntimeError("boom")

    out = script._probe(mapper, client, "NASDAQ", _inst())

    assert out["mapped"] is True
    assert "boom" in out["error"]
    assert "idx" in out


def test_probe_re_requests_an_empty_history_up_to_hist_retries(
    script, mapper, client, ticker
):
    """The facade treats an empty answer as final, so the retry lives in the probe."""
    ticker.history.side_effect = [_frame(0), _frame(0), _frame(4)]

    out = script._probe(mapper, client, "NASDAQ", _inst(), hist_retries=3)

    assert out["hist_rows"] == 4
    assert ticker.history.call_count == 3


def test_probe_makes_a_single_history_request_by_default(
    script, mapper, client, ticker
):
    """Without ``--hist-retries`` an empty answer is not asked for again."""
    ticker.history.side_effect = None
    ticker.history.return_value = _frame(0)

    out = script._probe(mapper, client, "NASDAQ", _inst())

    assert out["hist_rows"] == 0
    assert ticker.history.call_count == 1


def test_probe_records_a_history_rate_limit_error(script, mapper, client, ticker):
    """A throttled request must not be filed as a genuine empty answer.

    The facade swallows every exception and returns ``None``, which is
    indistinguishable from "Yahoo has no data"; the probe reads the Ticker
    directly so the rate-limit error, the only one yfinance re-raises from
    ``Ticker.history``, reaches the report.
    """
    ticker.history.side_effect = YFRateLimitError()

    out = script._probe(mapper, client, "NASDAQ", _inst())

    assert out["hist_rows"] == 0
    assert out["hist_err"] is not None
    assert "YFRateLimitError" in out["hist_err"]


def test_probe_reports_a_genuine_empty_history_without_an_error(
    script, mapper, client, ticker
):
    """An empty frame is Yahoo's answer: zero rows and no error."""
    ticker.history.return_value = _frame(0)

    out = script._probe(mapper, client, "NASDAQ", _inst())

    assert out["hist_rows"] == 0
    assert out["hist_err"] is None


def test_probe_clears_the_error_when_a_retry_succeeds(script, mapper, client, ticker):
    """A transient failure that a retry recovers is not an error in the report."""
    ticker.history.side_effect = [YFRateLimitError(), _frame(5)]

    out = script._probe(mapper, client, "NASDAQ", _inst(), hist_retries=2)

    assert out["hist_rows"] == 5
    assert out["hist_err"] is None


def test_fetch_history_raw_gives_up_after_the_attempt_budget(script, client, ticker):
    """The serial re-fetch reports the last failure once attempts run out."""
    ticker.history.side_effect = YFRateLimitError()

    hist, err = script.fetch_history_raw(client, "AAPL", attempts=3)

    assert hist is None
    assert err is not None and "YFRateLimitError" in err
    assert ticker.history.call_count == 3


def test_fetch_history_raw_always_makes_one_request(script, client, ticker):
    """A non-positive budget still performs the request rather than skipping it."""
    hist, err = script.fetch_history_raw(client, "AAPL", attempts=0)

    assert err is None
    assert hist is not None
    assert ticker.history.call_count == 1


def test_fetch_history_raw_docstring_claims_only_rate_limit_errors(script):
    """``Ticker.history`` re-raises only rate limits; the docstring must say so.

    Every other failure becomes an empty frame because ``hide_exceptions``
    stays on, so promising "any exception" would send a reader hunting for
    errors this helper can never report.
    """
    doc = " ".join((script.fetch_history_raw.__doc__ or "").split())

    assert "any exception" not in doc
    assert "YFRateLimitError" in doc


def test_fetch_history_raw_keeps_price_repair_on(script, client, ticker):
    """The raw request matches the facade default that needs scikit-learn."""
    script.fetch_history_raw(client, "AAPL", attempts=1)

    ticker.history.assert_called_once_with(period="5y", repair=True)


def test_main_reports_short_histories_without_a_config_history_floor(
    script, mapper, client, monkeypatch, capsys
):
    """The universe config carries no history threshold, so the script owns its own.

    ``UniverseBuilderConfig`` dropped every size / history threshold, and ``main``
    must still run to the end - classifying a short listing as YOUNG - rather than
    crash on the missing attribute after the rate-limited universe fetch.
    """
    monkeypatch.setattr(
        sys, "argv", ["debug_universe_scale", "--limit", "2", "--workers", "2"]
    )
    monkeypatch.setattr(
        script,
        "_real_universe_pairs",
        lambda cfg: [("NASDAQ", _inst()), ("NYSE", _inst())],
    )
    monkeypatch.setattr(script, "YFinanceTickerMapper", lambda cfg: mapper)
    monkeypatch.setattr(
        script, "YFinanceClient", SimpleNamespace(get_instance=lambda: client)
    )

    script.main()

    out = capsys.readouterr().out
    floor = script._MIN_TRADING_DAYS
    assert f"fetch_history >={floor} : 0" in out
    assert f"SHORT HISTORY (0 < rows < {floor}" in out
    assert "YOUNG (genuine short listing): 2" in out
