"""MarketStructureRepository upserts (sector snapshot / industries / top companies)."""

from __future__ import annotations

import datetime as dt

from portopt_db.repositories.market_data.market_structure_repository import (
    MarketStructureRepository,
)

_AS_OF = dt.date(2026, 8, 27)


def test_upserts_are_idempotent(db_session) -> None:
    repo = MarketStructureRepository(db_session)
    n_ind = n_co = 0
    for _ in range(2):
        repo.upsert_sector_snapshot(
            "technology",
            "US",
            _AS_OF,
            name="Technology",
            symbol="^YH311",
            market_cap=1.5e13,
            market_weight=0.31,
            companies_count=800,
            industries_count=12,
            employee_count=5_000_000,
        )
        n_ind = repo.upsert_industries(
            "technology",
            "US",
            _AS_OF,
            [{"key": "semiconductors", "name": "Semiconductors"}],
        )
        n_co = repo.upsert_top_companies(
            "technology",
            "US",
            _AS_OF,
            [{"symbol": "AAPL", "name": "Apple", "weight": 0.07, "rating": "buy"}],
        )
    db_session.flush()

    assert n_ind == 1 and n_co == 1
    snap = repo.get_sector_snapshot("technology", "US")
    assert snap is not None and float(snap.market_weight) == 0.31
    assert repo.get_latest_sector_as_of("technology", "US") == _AS_OF
