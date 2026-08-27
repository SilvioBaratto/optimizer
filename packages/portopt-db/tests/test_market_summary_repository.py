"""MarketSummaryRepository upsert."""

from __future__ import annotations

import datetime as dt

from portopt_db.models.market_data.market_summary import MarketSummary
from portopt_db.repositories.market_data.market_summary_repository import (
    MarketSummaryRepository,
)

_AS_OF = dt.date(2026, 8, 27)


def test_upsert_idempotent(db_session) -> None:
    repo = MarketSummaryRepository(db_session)
    rows = [
        {
            "symbol": "^GSPC",
            "short_name": "S&P 500",
            "price": 5600.5,
            "change": 12.3,
            "change_percent": 0.42,
            "previous_close": 5588.2,
            "market_state": "REGULAR",
        }
    ]
    assert repo.upsert_summaries("US", _AS_OF, rows) == 1
    assert repo.upsert_summaries("US", _AS_OF, rows) == 1  # idempotent
    db_session.flush()

    got = db_session.query(MarketSummary).all()
    assert len(got) == 1
    assert float(got[0].price) == 5600.5
