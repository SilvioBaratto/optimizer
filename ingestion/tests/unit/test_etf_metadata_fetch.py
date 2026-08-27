"""T5 — ETF metadata fetch/store in the data service.

An ETF (fixed_income / multi_asset) instrument pulls fund metadata via the funds
sub-client and persists it into the etf_* tables; staleness-gated in incremental
mode; a non-fund yields nothing without crashing.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

from portopt_db.models.universe.universe import Exchange, Instrument

from app.repositories.market_data.etf_metadata_repository import (
    ETFMetadataRepository,
)
from app.repositories.market_data.yfinance_repository import YFinanceRepository
from app.services.market_data.yfinance_data_service import (
    DEFAULT_THRESHOLDS,
    YFinanceDataService,
)

_NOW = dt.datetime(2026, 8, 25, 12, 0, tzinfo=dt.timezone.utc)


def _fi_instrument(db_session) -> Instrument:
    ex = Exchange(name="Deutsche Börse Xetra")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(
        ticker="JAGA.DE",
        short_name="JAGA",
        exchange_id=ex.id,
        instrument_type="ETF",
        asset_class="fixed_income",
    )
    db_session.add(inst)
    db_session.flush()
    return inst


def _service(db_session, yf: MagicMock) -> YFinanceDataService:
    return YFinanceDataService(YFinanceRepository(db_session), yf, request_timeout=5.0)


def test_fetch_etf_metadata_writes_all_tables(db_session) -> None:
    inst = _fi_instrument(db_session)
    yf = MagicMock()
    yf.funds.fetch_fund_profile.return_value = {
        "aum": 1e9,
        "nav": 100.0,
        "fund_family": "JPMorgan",
        "legal_type": "ETF",
        "expense_ratio": 0.001,
        "base_currency": "EUR",
    }
    yf.funds.fetch_funds_data.return_value = {
        "asset_classes": {
            "stockPosition": 0.0,
            "bondPosition": 1.0,
            "cashPosition": 0.0,
            "otherPosition": 0.0,
        },
        "top_holdings": [{"symbol": "UST", "name": "US Treasury", "weight": 0.05}],
        "sector_weightings": {"government": 0.6},
    }
    svc = _service(db_session, yf)

    counts: dict[str, int] = {}
    errors: list[str] = []
    svc._fetch_etf_metadata(
        inst.id, "JAGA.DE", "full", DEFAULT_THRESHOLDS, _NOW, counts, errors, []
    )
    db_session.flush()

    etf_repo = ETFMetadataRepository(db_session)
    assert etf_repo.get_metadata(inst.id) is not None
    ac = etf_repo.get_asset_classes(inst.id)
    assert ac is not None and float(ac.bond_pct) == 1.0
    assert counts.get("etf_metadata") == 1
    assert counts.get("etf_holdings") == 1
    assert counts.get("etf_sector_weights") == 1
    assert not errors


def test_fetch_etf_metadata_writes_depth_tables(db_session) -> None:
    """SPEC A8: equity/bond holdings, bond ratings, fund operations, and the
    fund overview (category/description) are persisted from funds_data depth."""
    from portopt_db.models.market_data.etf_metadata import (
        ETFBondHoldings,
        ETFBondRating,
        ETFEquityHoldings,
        ETFFundOperations,
        ETFMetadata,
    )
    from sqlalchemy import select

    inst = _fi_instrument(db_session)
    yf = MagicMock()
    yf.funds.fetch_fund_profile.return_value = {
        "aum": 1e9,
        "nav": 100.0,
        "fund_family": "JPMorgan",
        "legal_type": "ETF",
        "expense_ratio": 0.001,
        "base_currency": "EUR",
    }
    yf.funds.fetch_funds_data.return_value = {
        "asset_classes": {"stockPosition": 0.0, "bondPosition": 1.0},
        "top_holdings": [],
        "sector_weightings": {},
        "equity_holdings": {"priceToEarnings": 15.0, "priceToBook": 2.0},
        "bond_holdings": {"duration": 5.5, "maturity": 7.0, "creditQuality": 3.0},
        "fund_operations": {
            "annualReportExpenseRatio": 0.001,
            "annualHoldingsTurnover": 0.2,
            "totalNetAssets": 1.0e9,
        },
        "bond_ratings": {"aaa": 0.5, "bbb": 0.2},
        "fund_overview": {"categoryName": "Ultrashort Bond", "legalType": "ETF"},
        "description": "A short-duration bond ETF.",
    }
    svc = _service(db_session, yf)

    counts: dict[str, int] = {}
    errors: list[str] = []
    svc._fetch_etf_metadata(
        inst.id, "JAGA.DE", "full", DEFAULT_THRESHOLDS, _NOW, counts, errors, []
    )
    db_session.flush()

    eq = db_session.execute(
        select(ETFEquityHoldings).where(ETFEquityHoldings.instrument_id == inst.id)
    ).scalar_one()
    assert float(eq.price_to_earnings) == 15.0
    bh = db_session.execute(
        select(ETFBondHoldings).where(ETFBondHoldings.instrument_id == inst.id)
    ).scalar_one()
    assert float(bh.duration) == 5.5
    ratings = (
        db_session.execute(
            select(ETFBondRating).where(ETFBondRating.instrument_id == inst.id)
        )
        .scalars()
        .all()
    )
    assert {r.rating for r in ratings} == {"aaa", "bbb"}
    ops = db_session.execute(
        select(ETFFundOperations).where(ETFFundOperations.instrument_id == inst.id)
    ).scalar_one()
    assert float(ops.annual_report_expense_ratio) == 0.001
    meta = db_session.execute(
        select(ETFMetadata).where(ETFMetadata.instrument_id == inst.id)
    ).scalar_one()
    assert meta.category == "Ultrashort Bond"
    assert meta.description == "A short-duration bond ETF."
    assert not errors


def test_incremental_skips_when_metadata_is_fresh(db_session) -> None:
    inst = _fi_instrument(db_session)
    ETFMetadataRepository(db_session).upsert_metadata(
        inst.id,
        aum=1e9,
        nav=1.0,
        fund_family=None,
        legal_type=None,
        expense_ratio=None,
        base_currency=None,
        as_of=dt.date(2026, 8, 25),
    )
    db_session.flush()

    yf = MagicMock()
    svc = _service(db_session, yf)
    skipped: list[str] = []
    svc._fetch_etf_metadata(
        inst.id,
        "JAGA.DE",
        "incremental",
        DEFAULT_THRESHOLDS,
        _NOW,
        {},
        [],
        skipped,
    )

    assert "etf_metadata" in skipped
    yf.funds.fetch_fund_profile.assert_not_called()


def test_dispatch_calls_etf_branch_only_for_funds(db_session) -> None:
    """fetch_and_store runs the ETF metadata branch for fixed_income /
    multi_asset instruments, and skips it for equity."""
    from unittest.mock import patch

    inst = _fi_instrument(db_session)
    yf = MagicMock()
    yf.fetch_info.return_value = None
    yf.fetch_history.return_value = None
    svc = _service(db_session, yf)

    with patch.object(svc, "_fetch_etf_metadata") as branch:
        svc.fetch_and_store(inst.id, "JAGA.DE", mode="full", asset_class="equity")
        assert branch.call_count == 0
        svc.fetch_and_store(inst.id, "JAGA.DE", mode="full", asset_class="fixed_income")
        assert branch.call_count == 1
        svc.fetch_and_store(inst.id, "V60A.DE", mode="full", asset_class="multi_asset")
        assert branch.call_count == 2


def test_none_profile_and_funds_data_is_safe(db_session) -> None:
    inst = _fi_instrument(db_session)
    yf = MagicMock()
    yf.funds.fetch_fund_profile.return_value = None
    yf.funds.fetch_funds_data.return_value = None
    svc = _service(db_session, yf)

    counts: dict[str, int] = {}
    errors: list[str] = []
    svc._fetch_etf_metadata(
        inst.id, "X", "full", DEFAULT_THRESHOLDS, _NOW, counts, errors, []
    )

    assert counts == {}
    assert not errors
