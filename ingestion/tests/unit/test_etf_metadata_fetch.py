"""T5 — ETF metadata fetch/store in the data service.

An ETF (fixed_income / multi_asset) instrument pulls fund metadata via the funds
sub-client and persists it into the etf_* tables; staleness-gated in incremental
mode; a non-fund yields nothing without crashing.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.market_data.etf_metadata_repository import (
    ETFMetadataRepository,
)
from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository

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
    """Equity/bond holdings, bond ratings, fund operations, and the
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
        # Real yfinance funds_data depth keys are display labels, not camelCase.
        "equity_holdings": {"Price/Earnings": 15.0, "Price/Book": 2.0},
        "bond_holdings": {"Duration": 5.5, "Maturity": 7.0, "Credit Quality": 3.0},
        "fund_operations": {
            "Annual Report Expense Ratio": 0.001,
            "Annual Holdings Turnover": 0.2,
            "Total Net Assets": 1.0e9,
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


def test_fetch_etf_metadata_stores_major_unit_base_currency(db_session) -> None:
    """Yahoo quotes an LSE-listed UCITS ETF in pence ('GBp'); a sub-unit code is
    not a currency a fund's NAV or AUM is expressed in, so the major is stored."""
    inst = _fi_instrument(db_session)
    yf = MagicMock()
    yf.funds.fetch_fund_profile.return_value = {
        "aum": 1e9,
        "nav": 100.0,
        "base_currency": "GBp",
    }
    yf.funds.fetch_funds_data.return_value = None
    svc = _service(db_session, yf)

    svc._fetch_etf_metadata(
        inst.id, "JAGA.L", "full", DEFAULT_THRESHOLDS, _NOW, {}, [], []
    )
    db_session.flush()

    meta = ETFMetadataRepository(db_session).get_metadata(inst.id)
    assert meta is not None
    assert meta.base_currency == "GBP"


def test_fetch_etf_metadata_keeps_absent_base_currency_null(db_session) -> None:
    inst = _fi_instrument(db_session)
    yf = MagicMock()
    yf.funds.fetch_fund_profile.return_value = {"aum": 1e9, "nav": 100.0}
    yf.funds.fetch_funds_data.return_value = None
    svc = _service(db_session, yf)

    svc._fetch_etf_metadata(
        inst.id, "JAGA.DE", "full", DEFAULT_THRESHOLDS, _NOW, {}, [], []
    )
    db_session.flush()

    meta = ETFMetadataRepository(db_session).get_metadata(inst.id)
    assert meta is not None
    assert meta.base_currency is None


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


def test_unreported_allocation_does_not_overwrite_the_latest_snapshot(
    db_session,
) -> None:
    """A fund whose profile is populated but whose positions Yahoo omits must
    not write an all-NULL asset-class row: readers take the newest as_of, so it
    would shadow the last real allocation."""
    inst = _fi_instrument(db_session)
    etf_repo = ETFMetadataRepository(db_session)
    etf_repo.upsert_asset_classes(
        inst.id,
        dt.date(2026, 8, 24),
        stock_pct=0.0,
        bond_pct=0.98,
        cash_pct=0.02,
        other_pct=0.0,
    )
    db_session.flush()

    yf = MagicMock()
    yf.funds.fetch_fund_profile.return_value = None
    yf.funds.fetch_funds_data.return_value = {
        "asset_classes": {},
        "top_holdings": [],
        "sector_weightings": {},
        "fund_overview": {"categoryName": "Global Bond"},
        "description": "A bond ETF.",
    }
    svc = _service(db_session, yf)

    counts: dict[str, int] = {}
    errors: list[str] = []
    svc._fetch_etf_metadata(
        inst.id, "AGGB.L", "full", DEFAULT_THRESHOLDS, _NOW, counts, errors, []
    )
    db_session.flush()

    latest = etf_repo.get_asset_classes(inst.id)
    assert latest is not None
    assert latest.as_of == dt.date(2026, 8, 24)
    assert float(latest.bond_pct) == 0.98
    assert "etf_asset_classes" not in counts
    assert counts.get("etf_metadata") == 1
    assert not errors


def test_allocation_without_any_mapped_bucket_is_not_stored(db_session) -> None:
    inst = _fi_instrument(db_session)
    yf = MagicMock()
    yf.funds.fetch_fund_profile.return_value = None
    yf.funds.fetch_funds_data.return_value = {
        "asset_classes": {"preferredPosition": 0.1, "convertiblePosition": 0.0},
        "top_holdings": [{"symbol": "UST", "name": "US Treasury", "weight": 0.05}],
        "sector_weightings": {},
    }
    svc = _service(db_session, yf)

    counts: dict[str, int] = {}
    svc._fetch_etf_metadata(
        inst.id, "AGGB.L", "full", DEFAULT_THRESHOLDS, _NOW, counts, [], []
    )
    db_session.flush()

    assert ETFMetadataRepository(db_session).get_asset_classes(inst.id) is None
    assert "etf_asset_classes" not in counts
    assert counts.get("etf_holdings") == 1
