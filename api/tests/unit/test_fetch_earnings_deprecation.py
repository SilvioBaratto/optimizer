"""Regression tests for issue #591: dead `fetch_earnings` path removal.

Asserts that ``YFinanceDataService.fetch_and_store`` never invokes any code
that triggers the ``'Ticker.earnings' is deprecated`` warning emitted by
yfinance 1.3.0, and that the protocol/wrapper no longer expose
``fetch_earnings``.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pandas as pd

from app.services.yfinance.protocols.interfaces import FinancialsClientProtocol
from app.services.yfinance.ticker.financials import FinancialsClient
from app.services.yfinance_data_service import YFinanceDataService

_DEPRECATION_MESSAGE = "'Ticker.earnings' is deprecated"


class _StubFinancials:
    """Stub financials sub-client whose ``fetch_earnings`` would raise the
    deprecated-attribute warning if it were ever called.
    """

    def __init__(self, income_df: pd.DataFrame | None = None) -> None:
        self._income_df = income_df

    def fetch_income_stmt(self, *_: Any, **__: Any) -> pd.DataFrame | None:
        return self._income_df

    def fetch_balance_sheet(self, *_: Any, **__: Any) -> pd.DataFrame | None:
        return None

    def fetch_cashflow(self, *_: Any, **__: Any) -> pd.DataFrame | None:
        return None

    def fetch_sec_filings(self, *_: Any, **__: Any) -> list[dict[str, Any]] | None:
        return None

    def fetch_earnings(self, *_: Any, **__: Any) -> pd.DataFrame | None:
        warnings.warn(
            f"{_DEPRECATION_MESSAGE}, please replace your usage of `Ticker.earnings`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return None


def _build_yf_client(financials: _StubFinancials) -> MagicMock:
    """Return a MagicMock yf client whose only real attribute is ``financials``."""
    yf_client = MagicMock(name="yf_client")
    yf_client.financials = financials
    yf_client.fetch_info.return_value = None
    yf_client.fetch_history.return_value = None
    return yf_client


def _build_repo() -> MagicMock:
    repo = MagicMock(name="repo")
    repo.get_staleness_info.return_value = None
    repo.upsert_financial_statements.return_value = 0
    repo.upsert_profile.return_value = 0
    repo.upsert_price_history.return_value = 0
    return repo


def _run_fetch_and_store(
    income_df: pd.DataFrame | None = None,
) -> SimpleNamespace:
    financials = _StubFinancials(income_df=income_df)
    yf_client = _build_yf_client(financials)
    repo = _build_repo()
    service = YFinanceDataService(repo=repo, yf_client=yf_client)
    service.fetch_and_store(
        instrument_id=uuid4(),
        yfinance_ticker="AAPL",
        period="5y",
        mode="full",
    )
    return SimpleNamespace(repo=repo, yf_client=yf_client)


def test_when_fetch_and_store_runs_then_no_ticker_earnings_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run_fetch_and_store()

    earnings_warnings = [
        w for w in caught if _DEPRECATION_MESSAGE in str(w.message)
    ]
    assert earnings_warnings == [], (
        f"Expected zero 'Ticker.earnings' deprecation warnings, "
        f"got: {[str(w.message) for w in earnings_warnings]}"
    )


def test_when_income_statement_returns_data_then_upsert_uses_income_statement_type() -> None:
    income_df = pd.DataFrame(
        {pd.Timestamp("2025-12-31"): [1_000.0]},
        index=["TotalRevenue"],
    )

    result = _run_fetch_and_store(income_df=income_df)

    upsert_calls = result.repo.upsert_financial_statements.call_args_list
    statement_types = [
        call.args[2] if len(call.args) >= 3 else call.kwargs.get("statement_type")
        for call in upsert_calls
    ]
    assert "income_statement" in statement_types
    assert "earnings" not in statement_types


def test_when_inspecting_financials_client_then_fetch_earnings_is_absent() -> None:
    assert not hasattr(FinancialsClient, "fetch_earnings")


def test_when_inspecting_financials_protocol_then_fetch_earnings_is_absent() -> None:
    assert not hasattr(FinancialsClientProtocol, "fetch_earnings")
