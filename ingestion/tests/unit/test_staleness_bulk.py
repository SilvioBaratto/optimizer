"""SPEC review #4 — batched staleness (one grouped query per category for a
whole sweep) must match the per-instrument get_staleness_info shape/values.
Runs against real SQLite via the db_session fixture.
"""

from __future__ import annotations

import datetime as dt

from portopt_db.models.market_data.yfinance_data import PriceHistory, TickerProfile
from portopt_db.models.universe.universe import Exchange, Instrument

from app.repositories.market_data.yfinance_repository import YFinanceRepository


def _instrument(db_session, ticker: str) -> Instrument:
    ex = Exchange(name=f"EX-{ticker}")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(
        ticker=ticker,
        short_name=ticker,
        exchange_id=ex.id,
        instrument_type="EQUITY",
        asset_class="equity",
    )
    db_session.add(inst)
    db_session.flush()
    return inst


def test_bulk_matches_single_and_covers_empty(db_session) -> None:
    repo = YFinanceRepository(db_session)
    a = _instrument(db_session, "AAA")
    b = _instrument(db_session, "BBB")  # no rows at all

    db_session.add(PriceHistory(instrument_id=a.id, date=dt.date(2024, 6, 1), close=10))
    db_session.add(TickerProfile(instrument_id=a.id, symbol="AAA"))
    db_session.flush()

    bulk = repo.get_staleness_info_bulk([a.id, b.id])

    # Every requested id present with the full key set (all-None for the empty one).
    assert set(bulk) == {a.id, b.id}
    single_a = repo.get_staleness_info(a.id)
    assert set(bulk[a.id]) == set(single_a)
    assert (
        bulk[a.id]["price_max_date"]
        == dt.date(2024, 6, 1)
        == single_a["price_max_date"]
    )
    assert bulk[a.id]["profile_updated_at"] is not None

    # B has no data → all values None (matches a fresh instrument's single query).
    assert bulk[b.id]["price_max_date"] is None
    assert all(v is None for v in bulk[b.id].values())


def test_empty_id_list_returns_empty(db_session) -> None:
    repo = YFinanceRepository(db_session)
    assert repo.get_staleness_info_bulk([]) == {}
