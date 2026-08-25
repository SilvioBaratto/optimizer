"""T3 — ETF metadata repository: idempotent upserts on the natural keys.

Convert-then-rerun: the same payload twice converges to one row per table
(no duplicates); a changed payload updates in place.
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
from app.repositories.market_data.etf_metadata_repository import (
    ETFMetadataRepository,
)

_AS_OF = dt.date(2026, 8, 25)


def _instrument(db_session) -> Instrument:
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


def test_metadata_upsert_is_idempotent_and_updates(db_session) -> None:
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)

    for _ in range(2):
        repo.upsert_metadata(
            inst.id,
            aum=1_000_000_000.0,
            nav=100.0,
            fund_family="JPMorgan",
            legal_type="ETF",
            expense_ratio=0.001,
            base_currency="EUR",
            as_of=_AS_OF,
        )
    db_session.flush()
    assert db_session.query(ETFMetadata).filter_by(instrument_id=inst.id).count() == 1

    repo.upsert_metadata(
        inst.id,
        aum=2_000_000_000.0,
        nav=101.0,
        fund_family="JPMorgan",
        legal_type="ETF",
        expense_ratio=0.001,
        base_currency="EUR",
        as_of=_AS_OF,
    )
    db_session.flush()
    row = db_session.query(ETFMetadata).filter_by(instrument_id=inst.id).one()
    assert float(row.aum) == 2_000_000_000.0


def test_asset_classes_upsert_is_idempotent_and_updates(db_session) -> None:
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)

    for bond in (1.0, 1.0, 0.9):
        repo.upsert_asset_classes(
            inst.id,
            _AS_OF,
            stock_pct=0.0,
            bond_pct=bond,
            cash_pct=0.0,
            other_pct=0.0,
        )
    db_session.flush()

    row = repo.get_asset_classes(inst.id)
    assert row is not None
    assert float(row.bond_pct) == 0.9
    assert db_session.query(ETFAssetClass).filter_by(instrument_id=inst.id).count() == 1


def test_holdings_upsert_is_idempotent(db_session) -> None:
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)
    holdings = [
        {"symbol": "UST", "name": "US Treasury", "weight": 0.05},
        {"symbol": "BUND", "name": "Bund", "weight": 0.03},
    ]
    for _ in range(2):
        repo.upsert_holdings(inst.id, _AS_OF, holdings)
    db_session.flush()
    assert db_session.query(ETFHolding).filter_by(instrument_id=inst.id).count() == 2


def test_reads_return_none_when_absent(db_session) -> None:
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)
    assert repo.get_metadata(inst.id) is None
    assert repo.get_asset_classes(inst.id) is None


def test_empty_holdings_and_sectors_return_zero(db_session) -> None:
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)
    assert repo.upsert_holdings(inst.id, _AS_OF, []) == 0
    assert repo.upsert_sector_weights(inst.id, _AS_OF, {}) == 0


def test_sector_weights_upsert_is_idempotent(db_session) -> None:
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)
    weights = {"government": 0.6, "corporate": 0.4}
    for _ in range(2):
        repo.upsert_sector_weights(inst.id, _AS_OF, weights)
    db_session.flush()
    assert (
        db_session.query(ETFSectorWeight).filter_by(instrument_id=inst.id).count() == 2
    )
