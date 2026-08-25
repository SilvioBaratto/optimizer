"""T0 — stock+bond schema: asset_class tagging on instruments + etf_* metadata.

Verifies the DB can store a tagged fixed-income instrument and its fund metadata,
and that a stock defaults to the ``equity`` asset class.
"""

from __future__ import annotations

import datetime as dt

from app.models.market_data.etf_metadata import (
    ETFAssetClass,
    ETFHolding,
    ETFMetadata,
    ETFSectorWeight,
)
from app.models.universe.universe import Exchange, Instrument
from app.services.universe.trading212.enums import (
    AssetClass,
    DurationBucket,
    FiSubclass,
)


def _exchange(db_session, name: str = "Deutsche Börse Xetra") -> Exchange:
    ex = Exchange(name=name)
    db_session.add(ex)
    db_session.flush()
    return ex


def test_stock_defaults_to_equity_asset_class(db_session) -> None:
    ex = _exchange(db_session)
    inst = Instrument(
        ticker="EN.PA",
        short_name="EN",
        exchange_id=ex.id,
        instrument_type="STOCK",
    )
    db_session.add(inst)
    db_session.flush()
    db_session.refresh(inst)

    assert inst.asset_class == AssetClass.EQUITY.value == "equity"
    assert inst.fi_subclass is None
    assert inst.duration_bucket is None


def test_fixed_income_instrument_and_etf_metadata_round_trip(db_session) -> None:
    ex = _exchange(db_session)
    inst = Instrument(
        ticker="JAGA.DE",
        short_name="JAGA",
        exchange_id=ex.id,
        instrument_type="ETF",
        asset_class=AssetClass.FIXED_INCOME.value,
        fi_subclass=FiSubclass.AGGREGATE.value,
        duration_bucket=DurationBucket.INTERMEDIATE.value,
    )
    db_session.add(inst)
    db_session.flush()

    as_of = dt.date(2026, 8, 25)
    db_session.add_all(
        [
            ETFMetadata(
                instrument_id=inst.id,
                aum=79_000_000_000.0,
                nav=193.9,
                fund_family="JPMorgan",
                legal_type="ETF",
                expense_ratio=0.0025,
                base_currency="EUR",
                as_of=as_of,
            ),
            ETFAssetClass(
                instrument_id=inst.id,
                as_of=as_of,
                stock_pct=0.0,
                bond_pct=1.0,
                cash_pct=0.0,
                other_pct=0.0,
            ),
            ETFHolding(
                instrument_id=inst.id,
                as_of=as_of,
                holding_symbol="US912810TM",
                holding_name="US Treasury",
                weight=0.01,
            ),
            ETFSectorWeight(
                instrument_id=inst.id,
                as_of=as_of,
                sector="government",
                weight=0.6,
            ),
        ]
    )
    db_session.flush()

    assert inst.asset_class == "fixed_income"
    ac = db_session.query(ETFAssetClass).filter_by(instrument_id=inst.id).one()
    assert float(ac.bond_pct) == 1.0
    assert len(inst.etf_metadata) == 1
    assert len(inst.etf_holdings) == 1
