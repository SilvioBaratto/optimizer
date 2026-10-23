"""A DB error in one yfinance category must not poison the rest of the ticker.

All categories of one ticker share a single Session. PostgreSQL leaves the
transaction aborted after a failed statement, so without a SAVEPOINT per
category every later write fails with ``InFailedSqlTransaction`` and the final
COMMIT silently becomes a ROLLBACK. SQLite has no aborted-transaction state, so
these tests emulate it on the connection: after a failed statement every
statement is refused until the transaction (or a savepoint) is rolled back.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository
from sqlalchemy import event, text

from app.services.market_data.yfinance_data_service import YFinanceDataService

_ABORTED_MSG = "current transaction is aborted, commands ignored until end of block"


@contextlib.contextmanager
def _emulated_pg_abort(connection: Any) -> Iterator[dict[str, bool]]:
    state = {"aborted": False}

    def _on_error(_ctx: Any) -> None:
        state["aborted"] = True

    def _guard(_conn: Any, _cursor: Any, statement: str, *_a: Any) -> None:
        head = statement.lstrip().upper()
        if head.startswith("ROLLBACK TO"):
            state["aborted"] = False
            return
        if head.startswith(("SAVEPOINT", "RELEASE")):
            return
        if state["aborted"]:
            raise RuntimeError(_ABORTED_MSG)

    # handle_error can only be registered on the Engine, not a Connection.
    event.listen(connection.engine, "handle_error", _on_error)
    event.listen(connection, "before_cursor_execute", _guard)
    try:
        yield state
    finally:
        event.remove(connection.engine, "handle_error", _on_error)
        event.remove(connection, "before_cursor_execute", _guard)


def _instrument(db_session) -> Instrument:
    ex = Exchange(name="Savepoint Exchange")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(
        ticker="SVPT",
        short_name="SVPT",
        exchange_id=ex.id,
        instrument_type="STOCK",
        asset_class="equity",
    )
    db_session.add(inst)
    db_session.flush()
    return inst


def _yf_client() -> MagicMock:
    yf = MagicMock(name="yf_client")
    yf.fetch_info.return_value = None
    yf.financials.fetch_sec_filings.return_value = None
    yf.analysis.fetch_analyst_price_targets.return_value = None
    yf.fetch_history.return_value = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Close": [10.2, 11.2],
            "Volume": [1000, 2000],
        },
        index=pd.to_datetime(["2026-08-24", "2026-08-25"]),
    )
    yf.holders.fetch_institutional_holders.return_value = pd.DataFrame(
        {"Holder": ["Vanguard", "Vanguard"], "Shares": [1, 2]}
    )
    yf.holders.fetch_mutualfund_holders.return_value = pd.DataFrame(
        {"Holder": ["Vanguard Total Stock Market"], "Shares": [10], "Value": [100]}
    )
    yf.holders.fetch_insider_transactions.return_value = None
    yf.get_ticker.return_value = MagicMock(news=[])
    return yf


def test_failed_category_write_does_not_abort_later_categories(db_session) -> None:
    """A failing holders write is reported once and later categories still land."""
    inst = _instrument(db_session)
    repo = YFinanceRepository(db_session)

    def _failing_write(*_args: Any, **_kwargs: Any) -> int:
        db_session.execute(text("INSERT INTO table_that_does_not_exist VALUES (1)"))
        return 1

    repo.upsert_institutional_holders = _failing_write  # type: ignore[method-assign]
    service = YFinanceDataService(repo, _yf_client())

    with _emulated_pg_abort(db_session.get_bind()) as state:
        result = service.fetch_and_store(
            inst.id, "SVPT", period="1mo", mode="full", asset_class="equity"
        )

    assert [e.split(":")[0] for e in result["errors"]] == ["institutional_holders"]
    assert result["counts"]["prices"] == 2
    assert result["counts"]["mutualfund_holders"] == 1
    assert len(repo.get_mutualfund_holders(inst.id)) == 1
    assert state["aborted"] is False


def test_failed_etf_write_rolls_back_the_whole_etf_snapshot(db_session) -> None:
    """A failing ETF holdings write leaves no half-written metadata snapshot."""
    from portopt_db.repositories.market_data.etf_metadata_repository import (
        ETFMetadataRepository,
    )

    inst = _instrument(db_session)
    repo = YFinanceRepository(db_session)
    yf = _yf_client()
    yf.funds.fetch_fund_profile.return_value = {"aum": 1e9, "nav": 10.0}
    yf.funds.fetch_funds_data.return_value = {
        "asset_classes": {"stockPosition": 0.6, "bondPosition": 0.4},
        "top_holdings": [{"symbol": "AAA", "name": "A", "weight": 0.1}],
    }
    service = YFinanceDataService(repo, yf)

    def _failing_holdings(*_args: Any, **_kwargs: Any) -> int:
        db_session.execute(text("INSERT INTO table_that_does_not_exist VALUES (1)"))
        return 1

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ETFMetadataRepository, "upsert_holdings", _failing_holdings)
        with _emulated_pg_abort(db_session.get_bind()) as state:
            result = service.fetch_and_store(
                inst.id, "SVPT", period="1mo", mode="full", asset_class="fixed_income"
            )
        assert state["aborted"] is False

    etf_repo = ETFMetadataRepository(db_session)
    assert [e.split(":")[0] for e in result["errors"]] == ["etf_metadata"]
    assert etf_repo.get_metadata(inst.id) is None
    assert etf_repo.get_asset_classes(inst.id) is None
