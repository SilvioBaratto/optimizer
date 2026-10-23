"""Behaviour of the per-instrument probe in ``scripts/debug_universe_scale.py``.

The probe is driven with autospec'd collaborators, so a call to a method the
real mapper or client no longer has fails here the way it would in production.
No network, database or API key is touched.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import create_autospec

import pandas as pd
import pytest

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
def client():
    """A client with a usable profile and a three-row history."""
    fake = create_autospec(YFinanceClient, instance=True)
    fake.fetch_info.return_value = _INFO
    fake.fetch_history.return_value = _frame()
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


def test_probe_re_requests_an_empty_history_up_to_hist_retries(script, mapper, client):
    """The facade treats an empty answer as final, so the retry lives in the probe."""
    client.fetch_history.side_effect = [None, None, _frame(4)]

    out = script._probe(mapper, client, "NASDAQ", _inst(), hist_retries=3)

    assert out["hist_rows"] == 4
    assert client.fetch_history.call_count == 3


def test_probe_makes_a_single_history_request_by_default(script, mapper, client):
    """Without ``--hist-retries`` an empty answer is not asked for again."""
    client.fetch_history.return_value = None

    out = script._probe(mapper, client, "NASDAQ", _inst())

    assert out["hist_rows"] == 0
    assert client.fetch_history.call_count == 1


def test_fetch_history_retrying_empties_gives_up_after_the_attempt_budget(
    script, client
):
    """The serial re-fetch must report 'still empty' once attempts run out."""
    client.fetch_history.return_value = None

    hist = script.fetch_history_retrying_empties(client, "AAPL", attempts=3)

    assert hist is None
    assert client.fetch_history.call_count == 3


def test_fetch_history_retrying_empties_always_makes_one_request(script, client):
    """A non-positive budget still performs the request rather than skipping it."""
    hist = script.fetch_history_retrying_empties(client, "AAPL", attempts=0)

    assert hist is not None
    assert client.fetch_history.call_count == 1
