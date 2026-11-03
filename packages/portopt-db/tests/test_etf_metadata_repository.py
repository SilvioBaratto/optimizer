"""ETF metadata repository: idempotent upserts on the natural keys.

Convert-then-rerun: the same payload twice converges to one row per table
(no duplicates); a changed payload updates in place.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError

from portopt_db.models.market_data.etf_metadata import (
    ETFAssetClass,
    ETFBondHoldings,
    ETFEquityHoldings,
    ETFFundOperations,
    ETFHolding,
    ETFMetadata,
    ETFSectorWeight,
)
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.market_data.etf_metadata_repository import (
    ETFMetadataRepository,
    SectorSnapshot,
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


def test_metadata_upsert_with_none_keeps_previously_stored_values(db_session) -> None:
    """A partial Yahoo response (None = not reported this time) must not erase
    the last known fund metadata; supplied fields still update."""
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)
    repo.upsert_metadata(
        inst.id,
        aum=1_000_000_000.0,
        nav=100.0,
        fund_family="JPMorgan",
        legal_type="ETF",
        expense_ratio=0.001,
        base_currency="EUR",
        as_of=_AS_OF,
        category="Ultrashort Bond",
        description="A short-duration bond ETF.",
    )
    db_session.flush()

    later = dt.date(2026, 9, 1)
    repo.upsert_metadata(
        inst.id,
        aum=None,
        nav=None,
        fund_family=None,
        legal_type=None,
        expense_ratio=None,
        base_currency="USD",
        as_of=later,
    )
    db_session.flush()
    db_session.expire_all()

    row = db_session.query(ETFMetadata).filter_by(instrument_id=inst.id).one()
    assert float(row.aum) == 1_000_000_000.0
    assert float(row.nav) == 100.0
    assert row.fund_family == "JPMorgan"
    assert row.legal_type == "ETF"
    assert float(row.expense_ratio) == 0.001
    assert row.category == "Ultrashort Bond"
    assert row.description == "A short-duration bond ETF."
    assert row.base_currency == "USD"
    assert row.as_of == later


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


def test_asset_classes_preserve_preferred_and_convertible_so_weights_sum_to_one(
    db_session,
) -> None:
    # PFF-shaped payload: Yahoo reports six positions and 93% of the fund sits in
    # preferreds. Dropping that bucket leaves the stored weights summing to ~6%.
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)

    repo.upsert_asset_classes(
        inst.id,
        _AS_OF,
        stock_pct=0.0514,
        bond_pct=0.0001,
        cash_pct=0.0111,
        preferred_pct=0.932,
        convertible_pct=0.002,
        other_pct=0.0035,
    )
    db_session.flush()

    row = repo.get_asset_classes(inst.id)
    assert row is not None
    assert float(row.preferred_pct) == pytest.approx(0.932)
    assert float(row.convertible_pct) == pytest.approx(0.002)
    total = sum(
        float(v)
        for v in (
            row.stock_pct,
            row.bond_pct,
            row.cash_pct,
            row.preferred_pct,
            row.convertible_pct,
            row.other_pct,
        )
    )
    assert total == pytest.approx(1.0, abs=1e-3)


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


def test_holdings_dedup_duplicate_symbols_in_one_batch(db_session) -> None:
    # yfinance can repeat a symbol; a batch ON CONFLICT touching the same key
    # twice would raise on PostgreSQL. Dedup keeps the last, count reflects it.
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)
    n = repo.upsert_holdings(
        inst.id,
        _AS_OF,
        [
            {"symbol": "UST", "name": "A", "weight": 0.1},
            {"symbol": "UST", "name": "B", "weight": 0.2},
        ],
    )
    db_session.flush()
    assert n == 1
    row = db_session.query(ETFHolding).filter_by(instrument_id=inst.id).one()
    assert float(row.weight) == 0.2


def test_holdings_payload_carries_no_nan_name_or_weight(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing holding name arrives as pandas NaN and must reach SQL as NULL.

    SQLite silently stores float NaN as NULL, so the PostgreSQL behaviour (the
    text 'NaN' in a varchar column) is only visible on the upsert payload.
    """
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)
    captured: list[dict] = []
    monkeypatch.setattr(
        repo, "_upsert", lambda _model, rows, **_kw: captured.extend(rows)
    )

    repo.upsert_holdings(
        inst.id,
        _AS_OF,
        [
            {"symbol": "AAA", "name": float("nan"), "weight": float("nan")},
            {"symbol": "BBB", "name": "Beta", "weight": float("inf")},
            {"symbol": "CCC", "name": "Gamma", "weight": 0.1},
        ],
    )

    assert [r["holding_name"] for r in captured] == [None, "Beta", "Gamma"]
    assert [r["weight"] for r in captured] == [None, None, 0.1]


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


def test_depth_upserts_map_yfinance_display_labels(db_session) -> None:
    """funds_data depth dicts are keyed by yfinance display labels
    ("Price/Earnings", "Duration", "Annual Report Expense Ratio"), not camelCase.
    The repo must map those onto the model columns; absent labels stay NULL.
    """
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)

    repo.upsert_equity_holdings(
        inst.id,
        _AS_OF,
        {"Price/Earnings": 22.5, "Price/Book": 3.1},
    )
    repo.upsert_bond_holdings(inst.id, _AS_OF, {"Duration": 6.2, "Maturity": 8.0})
    repo.upsert_fund_operations(
        inst.id,
        _AS_OF,
        {"Annual Report Expense Ratio": 0.0009, "Total Net Assets": 5.0e8},
    )
    db_session.flush()

    eq = db_session.query(ETFEquityHoldings).filter_by(instrument_id=inst.id).one()
    assert float(eq.price_to_earnings) == 22.5
    assert float(eq.price_to_book) == 3.1
    assert eq.price_to_sales is None

    bh = db_session.query(ETFBondHoldings).filter_by(instrument_id=inst.id).one()
    assert float(bh.duration) == 6.2
    assert float(bh.maturity) == 8.0

    ops = db_session.query(ETFFundOperations).filter_by(instrument_id=inst.id).one()
    assert float(ops.annual_report_expense_ratio) == 0.0009
    assert float(ops.total_net_assets) == 5.0e8
    assert ops.annual_holdings_turnover is None


def test_depth_upserts_accept_camelcase_fallback(db_session) -> None:
    """Forward-compat: camelCase keys still map if yfinance switches spelling."""
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)
    repo.upsert_bond_holdings(inst.id, _AS_OF, {"duration": 4.0})
    repo.upsert_equity_holdings(inst.id, _AS_OF, {"priceToEarnings": 24.2})
    db_session.flush()
    bh = db_session.query(ETFBondHoldings).filter_by(instrument_id=inst.id).one()
    assert float(bh.duration) == 4.0
    eq = db_session.query(ETFEquityHoldings).filter_by(instrument_id=inst.id).one()
    assert float(eq.price_to_earnings) == 24.2


def test_failed_upsert_rolls_back_to_a_savepoint_and_spares_the_transaction(
    db_session,
) -> None:
    # The ingestion caller swallows a failed ETF write and still commits the
    # instrument's other writes from the same transaction. PostgreSQL aborts the
    # whole transaction on a failed statement, which turns that commit into a
    # silent rollback unless the write ran inside a SAVEPOINT.
    repo = ETFMetadataRepository(db_session)
    inst = _instrument(db_session)
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
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(statement)

    connection = db_session.get_bind()
    event.listen(connection, "before_cursor_execute", record)
    try:
        with pytest.raises(IntegrityError):
            repo.upsert_sector_weights(inst.id, _AS_OF, {None: 0.5})  # type: ignore[dict-item]
    finally:
        event.remove(connection, "before_cursor_execute", record)

    insert_at = next(
        i for i, s in enumerate(statements) if s.startswith("INSERT INTO etf_sector")
    )
    assert any(s.startswith("SAVEPOINT") for s in statements[:insert_at])
    assert any(s.startswith("ROLLBACK TO SAVEPOINT") for s in statements[insert_at:])

    repo.upsert_asset_classes(
        inst.id, _AS_OF, stock_pct=0.0, bond_pct=1.0, cash_pct=0.0, other_pct=0.0
    )
    db_session.flush()
    meta = repo.get_metadata(inst.id)
    assert meta is not None
    assert float(meta.aum) == 1_000_000_000.0
    assert repo.get_asset_classes(inst.id) is not None


def _etf(
    db_session, yfinance_ticker: str, *, exchange: str = "Xetra Test"
) -> Instrument:
    """An ETF instrument whose yfinance symbol is ``yfinance_ticker``."""
    ex = db_session.query(Exchange).filter_by(name=exchange).one_or_none()
    if ex is None:
        ex = Exchange(name=exchange)
        db_session.add(ex)
        db_session.flush()
    inst = Instrument(
        ticker=yfinance_ticker.split(".")[0],
        short_name=yfinance_ticker,
        yfinance_ticker=yfinance_ticker,
        exchange_id=ex.id,
        instrument_type="ETF",
        asset_class="equity",
    )
    db_session.add(inst)
    db_session.flush()
    return inst


def _record_statements(db_session) -> tuple[list[str], object]:
    """Start capturing every SQL statement on the session's connection.

    Returns the capture list and the listener to pass to ``_stop_recording``.
    """
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(statement)

    event.listen(db_session.get_bind(), "before_cursor_execute", record)
    return statements, record


def _stop_recording(db_session, record) -> None:
    event.remove(db_session.get_bind(), "before_cursor_execute", record)


class TestGetLatestSectorWeights:
    """``get_latest_sector_weights`` reads the newest sector snapshot per ETF."""

    def test_returns_only_the_newest_snapshot(self, db_session) -> None:
        """An older as_of for the same instrument is ignored."""
        repo = ETFMetadataRepository(db_session)
        inst = _etf(db_session, "VWCE.DE")
        repo.upsert_sector_weights(
            inst.id, dt.date(2026, 7, 1), {"technology": 0.9, "energy": 0.1}
        )
        repo.upsert_sector_weights(
            inst.id, dt.date(2026, 9, 1), {"technology": 0.3, "healthcare": 0.7}
        )
        db_session.flush()

        result = repo.get_latest_sector_weights(["VWCE.DE"])

        assert set(result) == {"VWCE.DE"}
        assert result["VWCE.DE"].as_of == dt.date(2026, 9, 1)
        assert result["VWCE.DE"].weights == {"technology": 0.3, "healthcare": 0.7}

    def test_each_instrument_uses_its_own_newest_as_of(self, db_session) -> None:
        """Snapshots of different ETFs are compared per instrument, not globally."""
        repo = ETFMetadataRepository(db_session)
        a = _etf(db_session, "AAA.DE")
        b = _etf(db_session, "BBB.DE")
        repo.upsert_sector_weights(a.id, dt.date(2026, 6, 1), {"energy": 1.0})
        repo.upsert_sector_weights(b.id, dt.date(2026, 9, 1), {"utilities": 1.0})
        db_session.flush()

        result = repo.get_latest_sector_weights(["AAA.DE", "BBB.DE"])

        assert result["AAA.DE"].as_of == dt.date(2026, 6, 1)
        assert result["AAA.DE"].weights == {"energy": 1.0}
        assert result["BBB.DE"].as_of == dt.date(2026, 9, 1)

    def test_unknown_ticker_and_etf_without_sectors_are_absent(
        self, db_session
    ) -> None:
        """A ticker with no instrument, or an ETF with no sector rows, is omitted."""
        repo = ETFMetadataRepository(db_session)
        with_rows = _etf(db_session, "VWCE.DE")
        _etf(db_session, "EGLN.L", exchange="LSE Test")
        repo.upsert_sector_weights(with_rows.id, _AS_OF, {"energy": 1.0})
        db_session.flush()

        result = repo.get_latest_sector_weights(["VWCE.DE", "EGLN.L", "NOPE.DE"])

        assert set(result) == {"VWCE.DE"}

    def test_empty_input_returns_empty_dict_without_a_query(self, db_session) -> None:
        """No tickers means no database round trip."""
        repo = ETFMetadataRepository(db_session)
        statements, record = _record_statements(db_session)
        try:
            assert repo.get_latest_sector_weights([]) == {}
        finally:
            _stop_recording(db_session, record)

        assert statements == []

    def test_zero_kept_and_null_preserved(self, db_session) -> None:
        """Yahoo's 0.0 for an empty sector stays 0.0; a NULL weight stays None."""
        repo = ETFMetadataRepository(db_session)
        inst = _etf(db_session, "VWCE.DE")
        repo.upsert_sector_weights(
            inst.id,
            _AS_OF,
            {"technology": 0.25, "energy": 0.0, "utilities": None},  # type: ignore[dict-item]
        )
        db_session.flush()

        weights = repo.get_latest_sector_weights(["VWCE.DE"])["VWCE.DE"].weights

        assert weights["energy"] == 0.0
        assert weights["utilities"] is None
        assert weights["technology"] == 0.25
        assert isinstance(weights["technology"], float)

    def test_weights_are_not_normalised(self, db_session) -> None:
        """Weights come back as stored even when they do not sum to one."""
        repo = ETFMetadataRepository(db_session)
        inst = _etf(db_session, "VWCE.DE")
        repo.upsert_sector_weights(inst.id, _AS_OF, {"technology": 0.2, "energy": 0.1})
        db_session.flush()

        weights = repo.get_latest_sector_weights(["VWCE.DE"])["VWCE.DE"].weights

        assert sum(w for w in weights.values() if w is not None) == pytest.approx(0.3)

    def test_runs_as_one_query_for_many_tickers(self, db_session) -> None:
        """The read costs a single statement regardless of how many tickers."""
        repo = ETFMetadataRepository(db_session)
        tickers = [f"T{i}.DE" for i in range(6)]
        for t in tickers:
            repo.upsert_sector_weights(_etf(db_session, t).id, _AS_OF, {"energy": 1.0})
        db_session.flush()
        statements, record = _record_statements(db_session)
        try:
            result = repo.get_latest_sector_weights(tickers)
        finally:
            _stop_recording(db_session, record)

        assert set(result) == set(tickers)
        assert len(statements) == 1

    def test_shared_yfinance_ticker_keeps_the_newer_instrument_snapshot(
        self, db_session
    ) -> None:
        """Two instrument rows sharing a symbol never mix their sector weights."""
        repo = ETFMetadataRepository(db_session)
        old = _etf(db_session, "DUP.DE", exchange="Venue A")
        new = _etf(db_session, "DUP.DE", exchange="Venue B")
        repo.upsert_sector_weights(old.id, dt.date(2026, 6, 1), {"energy": 1.0})
        repo.upsert_sector_weights(new.id, dt.date(2026, 9, 1), {"utilities": 1.0})
        db_session.flush()

        snapshot = repo.get_latest_sector_weights(["DUP.DE"])["DUP.DE"]

        assert snapshot.as_of == dt.date(2026, 9, 1)
        assert snapshot.weights == {"utilities": 1.0}

    def test_snapshot_is_immutable(self) -> None:
        """``SectorSnapshot`` is a frozen dataclass."""
        snap = SectorSnapshot(as_of=_AS_OF, weights={"energy": 1.0})
        with pytest.raises(AttributeError):
            snap.as_of = dt.date(2026, 1, 1)  # type: ignore[misc]
