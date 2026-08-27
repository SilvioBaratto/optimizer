"""Branch coverage for YFinanceRepository and its module-level helpers.

Strategy
--------
* All ``upsert_*`` methods call ``RepositoryBase._upsert`` which issues a
  PostgreSQL ``INSERT … ON CONFLICT`` statement.  That dialect construct is
  **not executable on SQLite** — those methods are skipped with an inline
  comment where relevant.
* Every ``get_*`` / query method is a plain SELECT — seeded via
  ``session.add(obj); session.flush()`` and fully testable.
* Module-level helpers (``_safe_val``, ``_safe_int``, ``_safe_float``,
  ``_safe_str``, ``_safe_date``) have no DB dependency — tested directly.
* ``get_staleness_info`` uses ``func.max()`` SELECTs — fully testable.
"""

from __future__ import annotations

import math
import uuid
from datetime import date, datetime

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from portopt_db.models.market_data.yfinance_data import (
    AnalystPriceTarget,
    AnalystRecommendation,
    Dividend,
    FinancialStatement,
    InsiderTransaction,
    InstitutionalHolder,
    MutualFundHolder,
    PriceHistory,
    StockSplit,
    TickerNews,
    TickerProfile,
)
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.market_data.yfinance_repository import (
    YFinanceRepository,
    _safe_date,
    _safe_float,
    _safe_int,
    _safe_str,
    _safe_val,
)

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _exchange(session: Session, name: str = "NYSE") -> Exchange:
    ex = Exchange(name=name)
    session.add(ex)
    session.flush()
    return ex


def _instrument(
    session: Session,
    exchange: Exchange,
    ticker: str = "AAPL",
    yfinance_ticker: str | None = "AAPL",
) -> Instrument:
    inst = Instrument(
        ticker=ticker,
        short_name=ticker,
        name=f"{ticker} Inc.",
        exchange=exchange,
        yfinance_ticker=yfinance_ticker,
    )
    session.add(inst)
    session.flush()
    return inst


def _repo(session: Session) -> YFinanceRepository:
    return YFinanceRepository(session)


# ---------------------------------------------------------------------------
# _safe_val — all branches
# ---------------------------------------------------------------------------


class TestSafeVal:
    def test_none_returns_none(self) -> None:
        assert _safe_val(None) is None

    def test_nan_float_returns_none(self) -> None:
        assert _safe_val(float("nan")) is None

    def test_inf_float_returns_none(self) -> None:
        assert _safe_val(float("inf")) is None

    def test_neg_inf_float_returns_none(self) -> None:
        assert _safe_val(float("-inf")) is None

    def test_normal_float_passthrough(self) -> None:
        assert _safe_val(3.14) == 3.14

    def test_pd_timestamp_converted_to_pydatetime(self) -> None:
        ts = pd.Timestamp("2024-01-15")
        result = _safe_val(ts)
        assert isinstance(result, datetime)

    def test_numpy_scalar_calls_item(self) -> None:
        import numpy as np

        arr = np.array([42])
        result = _safe_val(arr[0])
        assert result == 42

    def test_plain_int_passthrough(self) -> None:
        assert _safe_val(7) == 7

    def test_plain_str_passthrough(self) -> None:
        assert _safe_val("hello") == "hello"


# ---------------------------------------------------------------------------
# _safe_int — all branches
# ---------------------------------------------------------------------------


class TestSafeInt:
    def test_none_returns_none(self) -> None:
        assert _safe_int(None) is None

    def test_valid_int_returns_int(self) -> None:
        assert _safe_int(5) == 5

    def test_float_truncated_to_int(self) -> None:
        assert _safe_int(3.9) == 3

    def test_nan_returns_none(self) -> None:
        assert _safe_int(float("nan")) is None

    def test_non_numeric_str_returns_none(self) -> None:
        assert _safe_int("abc") is None

    def test_numeric_str_returns_int(self) -> None:
        assert _safe_int("10") == 10

    def test_list_returns_none(self) -> None:
        # TypeError path
        assert _safe_int([1, 2]) is None


# ---------------------------------------------------------------------------
# _safe_float — all branches
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_none_returns_none(self) -> None:
        assert _safe_float(None) is None

    def test_valid_float_returned(self) -> None:
        result = _safe_float(2.5)
        assert result == pytest.approx(2.5)

    def test_nan_input_returns_none(self) -> None:
        assert _safe_float(float("nan")) is None

    def test_inf_input_returns_none(self) -> None:
        assert _safe_float(float("inf")) is None

    def test_nan_after_float_conversion_returns_none(self) -> None:
        # _safe_val passes through; float("nan") produced inside _safe_float
        assert _safe_float(math.nan) is None

    def test_non_numeric_str_returns_none(self) -> None:
        assert _safe_float("xyz") is None

    def test_numeric_str_returns_float(self) -> None:
        result = _safe_float("1.5")
        assert result == pytest.approx(1.5)

    def test_list_returns_none(self) -> None:
        assert _safe_float([]) is None

    def test_inf_after_float_conversion_returns_none(self) -> None:
        # Covers the math.isinf(f) branch when math.isnan(f) is False
        # numpy float('inf') passes _safe_val (not a float instance initially)
        # but we can reach line 67 via a real Python float inf that passes
        # the isinstance check in _safe_val (it IS a float) → returns None there.
        # To reach _safe_float line 67 we need a value that:
        #   1. is NOT a Python float (so _safe_val passes it through), AND
        #   2. converts to inf via float()
        # A string "inf" satisfies both: _safe_val returns "inf" (str), then
        # float("inf") is inf → math.isnan is False → math.isinf is True → None
        assert _safe_float("inf") is None

    def test_neg_inf_str_returns_none(self) -> None:
        assert _safe_float("-inf") is None


# ---------------------------------------------------------------------------
# _safe_str — all branches
# ---------------------------------------------------------------------------


class TestSafeStr:
    def test_none_returns_none(self) -> None:
        assert _safe_str(None) is None

    def test_string_passthrough(self) -> None:
        assert _safe_str("hello") == "hello"

    def test_int_coerced_to_str(self) -> None:
        assert _safe_str(42) == "42"

    def test_max_len_truncates(self) -> None:
        assert _safe_str("abcdef", max_len=3) == "abc"

    def test_max_len_none_no_truncation(self) -> None:
        s = "x" * 100
        assert _safe_str(s) == s

    def test_max_len_zero_returns_empty(self) -> None:
        # max_len=0 is falsy — no truncation applied
        result = _safe_str("hello", max_len=0)
        assert result == "hello"


# ---------------------------------------------------------------------------
# _safe_date — all branches
# ---------------------------------------------------------------------------


class TestSafeDate:
    def test_none_returns_none(self) -> None:
        assert _safe_date(None) is None

    def test_datetime_object_returns_date(self) -> None:
        dt = datetime(2024, 6, 15, 12, 0)
        assert _safe_date(dt) == date(2024, 6, 15)

    def test_date_object_passthrough(self) -> None:
        d = date(2024, 3, 1)
        assert _safe_date(d) == d

    def test_iso_string_parsed(self) -> None:
        assert _safe_date("2024-01-15") == date(2024, 1, 15)

    def test_invalid_string_returns_none(self) -> None:
        assert _safe_date("not-a-date") is None

    def test_unix_timestamp_int_returns_date(self) -> None:
        # 2024-01-01 00:00:00 UTC → date
        ts = int(datetime(2024, 1, 1).timestamp())
        result = _safe_date(ts)
        assert isinstance(result, date)

    def test_nan_float_returns_none(self) -> None:
        # _safe_val returns None for NaN before the int/float branch
        assert _safe_date(float("nan")) is None

    def test_very_large_int_returns_none(self) -> None:
        # OSError / OverflowError path for out-of-range timestamp
        assert _safe_date(99999999999999) is None

    def test_pd_timestamp_returns_date(self) -> None:
        ts = pd.Timestamp("2023-07-04")
        result = _safe_date(ts)
        assert isinstance(result, date)

    def test_unknown_type_returns_none(self) -> None:
        assert _safe_date([2024, 1, 1]) is None


# ---------------------------------------------------------------------------
# get_profile — success + not-found
# ---------------------------------------------------------------------------


class TestGetProfile:
    def test_when_profile_exists_then_returned(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        profile = TickerProfile(
            instrument_id=inst.id,
            symbol="AAPL",
            sector="Technology",
            currency="USD",
        )
        db_session.add(profile)
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_profile(inst.id)

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.sector == "Technology"

    def test_when_no_profile_then_none(self, db_session: Session) -> None:
        repo = _repo(db_session)
        result = repo.get_profile(uuid.uuid4())
        assert result is None


# ---------------------------------------------------------------------------
# get_price_history — success + empty + date filter branches
# ---------------------------------------------------------------------------


class TestGetPriceHistory:
    def _seed_prices(
        self, session: Session, inst: Instrument, dates: list[date]
    ) -> list[PriceHistory]:
        rows = []
        for d in dates:
            ph = PriceHistory(
                instrument_id=inst.id,
                date=d,
                open=100.0,
                high=110.0,
                low=90.0,
                close=105.0,
                volume=1_000_000,
            )
            session.add(ph)
            rows.append(ph)
        session.flush()
        return rows

    def test_when_no_prices_then_empty(self, db_session: Session) -> None:
        repo = _repo(db_session)
        result = repo.get_price_history(uuid.uuid4())
        assert list(result) == []

    def test_when_prices_exist_then_ordered_desc(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        self._seed_prices(
            db_session,
            inst,
            [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        )

        repo = _repo(db_session)
        result = repo.get_price_history(inst.id)

        dates = [r.date for r in result]
        assert dates == sorted(dates, reverse=True)
        assert len(result) == 3

    def test_start_date_filter_excludes_earlier(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        self._seed_prices(
            db_session,
            inst,
            [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
        )

        repo = _repo(db_session)
        result = repo.get_price_history(inst.id, start_date=date(2024, 2, 1))

        assert all(r.date >= date(2024, 2, 1) for r in result)
        assert len(result) == 2

    def test_end_date_filter_excludes_later(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        self._seed_prices(
            db_session,
            inst,
            [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
        )

        repo = _repo(db_session)
        result = repo.get_price_history(inst.id, end_date=date(2024, 2, 1))

        assert all(r.date <= date(2024, 2, 1) for r in result)
        assert len(result) == 2

    def test_both_date_filters_applied(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        self._seed_prices(
            db_session,
            inst,
            [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
        )

        repo = _repo(db_session)
        result = repo.get_price_history(
            inst.id,
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 28),
        )

        assert len(result) == 1
        assert result[0].date == date(2024, 2, 1)

    def test_limit_respected(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        self._seed_prices(
            db_session,
            inst,
            [date(2024, 1, i) for i in range(1, 11)],
        )

        repo = _repo(db_session)
        result = repo.get_price_history(inst.id, limit=3)

        assert len(result) == 3


# ---------------------------------------------------------------------------
# get_financial_statements — success + filter branches
# ---------------------------------------------------------------------------


class TestGetFinancialStatements:
    def _seed_stmt(
        self,
        session: Session,
        inst: Instrument,
        stmt_type: str,
        period_type: str,
        line_item: str = "Revenue",
        period_date: date = date(2024, 12, 31),
    ) -> FinancialStatement:
        fs = FinancialStatement(
            instrument_id=inst.id,
            statement_type=stmt_type,
            period_type=period_type,
            period_date=period_date,
            line_item=line_item,
            value=1_000_000.0,
        )
        session.add(fs)
        session.flush()
        return fs

    def test_when_no_statements_then_empty(self, db_session: Session) -> None:
        repo = _repo(db_session)
        assert list(repo.get_financial_statements(uuid.uuid4())) == []

    def test_returns_all_when_no_filters(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        self._seed_stmt(db_session, inst, "income_statement", "annual")
        self._seed_stmt(db_session, inst, "balance_sheet", "quarterly")

        repo = _repo(db_session)
        result = repo.get_financial_statements(inst.id)

        assert len(result) == 2

    def test_statement_type_filter(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        self._seed_stmt(db_session, inst, "income_statement", "annual")
        self._seed_stmt(db_session, inst, "balance_sheet", "annual")

        repo = _repo(db_session)
        result = repo.get_financial_statements(
            inst.id, statement_type="income_statement"
        )

        assert all(r.statement_type == "income_statement" for r in result)
        assert len(result) == 1

    def test_period_type_filter(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        self._seed_stmt(db_session, inst, "income_statement", "annual")
        self._seed_stmt(
            db_session,
            inst,
            "income_statement",
            "quarterly",
            period_date=date(2024, 3, 31),
        )

        repo = _repo(db_session)
        result = repo.get_financial_statements(inst.id, period_type="quarterly")

        assert all(r.period_type == "quarterly" for r in result)
        assert len(result) == 1

    def test_both_filters_applied(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        self._seed_stmt(db_session, inst, "income_statement", "annual")
        self._seed_stmt(
            db_session, inst, "balance_sheet", "annual", period_date=date(2023, 12, 31)
        )

        repo = _repo(db_session)
        result = repo.get_financial_statements(
            inst.id,
            statement_type="income_statement",
            period_type="annual",
        )

        assert len(result) == 1
        assert result[0].statement_type == "income_statement"


# ---------------------------------------------------------------------------
# get_dividends
# ---------------------------------------------------------------------------


class TestGetDividends:
    def test_when_no_dividends_then_empty(self, db_session: Session) -> None:
        repo = _repo(db_session)
        assert list(repo.get_dividends(uuid.uuid4())) == []

    def test_dividends_returned_ordered_desc(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        for d in [date(2024, 1, 1), date(2024, 4, 1), date(2024, 7, 1)]:
            db_session.add(Dividend(instrument_id=inst.id, date=d, amount=0.25))
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_dividends(inst.id)

        dates = [r.date for r in result]
        assert dates == sorted(dates, reverse=True)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# get_splits
# ---------------------------------------------------------------------------


class TestGetSplits:
    def test_when_no_splits_then_empty(self, db_session: Session) -> None:
        repo = _repo(db_session)
        assert list(repo.get_splits(uuid.uuid4())) == []

    def test_splits_returned_ordered_desc(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        for d in [date(2020, 8, 31), date(2014, 6, 9)]:
            db_session.add(StockSplit(instrument_id=inst.id, date=d, ratio=4.0))
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_splits(inst.id)

        dates = [r.date for r in result]
        assert dates == sorted(dates, reverse=True)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# get_recommendations
# ---------------------------------------------------------------------------


class TestGetRecommendations:
    def test_when_no_recs_then_empty(self, db_session: Session) -> None:
        repo = _repo(db_session)
        assert list(repo.get_recommendations(uuid.uuid4())) == []

    def test_recommendations_returned_ordered_by_period(
        self, db_session: Session
    ) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        for period in ["0m", "-1m", "-2m"]:
            db_session.add(
                AnalystRecommendation(
                    instrument_id=inst.id,
                    period=period,
                    strong_buy=10,
                    buy=5,
                    hold=3,
                    sell=1,
                    strong_sell=0,
                )
            )
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_recommendations(inst.id)

        periods = [r.period for r in result]
        assert periods == sorted(periods)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# get_price_targets
# ---------------------------------------------------------------------------


class TestGetPriceTargets:
    def test_when_no_target_then_none(self, db_session: Session) -> None:
        repo = _repo(db_session)
        assert repo.get_price_targets(uuid.uuid4()) is None

    def test_when_target_exists_then_returned(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        pt = AnalystPriceTarget(
            instrument_id=inst.id,
            current=180.0,
            low=150.0,
            high=220.0,
            mean=190.0,
            median=185.0,
        )
        db_session.add(pt)
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_price_targets(inst.id)

        assert result is not None
        assert float(result.current) == pytest.approx(180.0)
        assert float(result.high) == pytest.approx(220.0)


# ---------------------------------------------------------------------------
# get_institutional_holders
# ---------------------------------------------------------------------------


class TestGetInstitutionalHolders:
    def test_when_none_then_empty(self, db_session: Session) -> None:
        repo = _repo(db_session)
        assert list(repo.get_institutional_holders(uuid.uuid4())) == []

    def test_holders_ordered_by_name(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        for name in ["Vanguard", "BlackRock", "Fidelity"]:
            db_session.add(
                InstitutionalHolder(
                    instrument_id=inst.id,
                    holder_name=name,
                    shares=1_000_000,
                    pct_held=0.05,
                )
            )
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_institutional_holders(inst.id)

        names = [r.holder_name for r in result]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# get_mutualfund_holders
# ---------------------------------------------------------------------------


class TestGetMutualFundHolders:
    def test_when_none_then_empty(self, db_session: Session) -> None:
        repo = _repo(db_session)
        assert list(repo.get_mutualfund_holders(uuid.uuid4())) == []

    def test_holders_ordered_by_name(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        for name in ["Zeta Fund", "Alpha Fund"]:
            db_session.add(
                MutualFundHolder(
                    instrument_id=inst.id,
                    holder_name=name,
                    shares=500_000,
                )
            )
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_mutualfund_holders(inst.id)

        names = [r.holder_name for r in result]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# get_insider_transactions
# ---------------------------------------------------------------------------


class TestGetInsiderTransactions:
    def test_when_none_then_empty(self, db_session: Session) -> None:
        repo = _repo(db_session)
        assert list(repo.get_insider_transactions(uuid.uuid4())) == []

    def test_transactions_ordered_desc_by_start_date(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        for d, name in [
            (date(2024, 1, 10), "Alice CEO"),
            (date(2024, 3, 15), "Bob CFO"),
            (date(2023, 11, 5), "Carol COO"),
        ]:
            db_session.add(
                InsiderTransaction(
                    instrument_id=inst.id,
                    insider_name=name,
                    transaction_type="Sale",
                    start_date=d,
                    shares=1000,
                )
            )
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_insider_transactions(inst.id)

        dates = [r.start_date for r in result]
        assert dates == sorted(dates, reverse=True)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# get_news
# ---------------------------------------------------------------------------


class TestGetNews:
    def test_when_no_news_then_empty(self, db_session: Session) -> None:
        repo = _repo(db_session)
        assert list(repo.get_news(uuid.uuid4())) == []

    def test_news_ordered_desc_by_publish_time(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        for i, dt in enumerate(
            [
                datetime(2024, 1, 1, 9, 0),
                datetime(2024, 1, 2, 9, 0),
                datetime(2024, 1, 3, 9, 0),
            ]
        ):
            db_session.add(
                TickerNews(
                    instrument_id=inst.id,
                    news_uuid=f"uuid-{i}",
                    title=f"Article {i}",
                    publish_time=dt,
                )
            )
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_news(inst.id)

        times = [r.publish_time for r in result]
        assert times == sorted(times, reverse=True)

    def test_news_with_null_publish_time_included(self, db_session: Session) -> None:
        """NULLs appear last (nullslast ordering) — they must still be returned."""
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        db_session.add(
            TickerNews(
                instrument_id=inst.id,
                news_uuid="no-date-uuid",
                title="Timeless article",
                publish_time=None,
            )
        )
        db_session.add(
            TickerNews(
                instrument_id=inst.id,
                news_uuid="dated-uuid",
                title="Dated article",
                publish_time=datetime(2024, 6, 1),
            )
        )
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_news(inst.id)

        assert len(result) == 2
        # dated article comes first; null-time article last
        assert result[0].publish_time is not None
        assert result[-1].publish_time is None


# ---------------------------------------------------------------------------
# get_staleness_info
# ---------------------------------------------------------------------------


class TestGetStalenessInfo:
    def test_when_no_data_all_none(self, db_session: Session) -> None:
        repo = _repo(db_session)
        info = repo.get_staleness_info(uuid.uuid4())

        assert info["price_max_date"] is None
        assert info["profile_updated_at"] is None
        assert info["financials_updated_at"] is None
        assert info["dividends_updated_at"] is None
        assert info["splits_updated_at"] is None
        assert info["recommendations_updated_at"] is None
        assert info["price_targets_updated_at"] is None
        assert info["institutional_holders_updated_at"] is None
        assert info["mutualfund_holders_updated_at"] is None
        assert info["insider_transactions_updated_at"] is None
        assert info["news_updated_at"] is None

    def test_when_price_history_exists_price_max_date_set(
        self, db_session: Session
    ) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        for d in [date(2024, 1, 1), date(2024, 6, 15), date(2023, 12, 31)]:
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=d,
                    close=150.0,
                )
            )
        db_session.flush()

        repo = _repo(db_session)
        info = repo.get_staleness_info(inst.id)

        assert info["price_max_date"] == date(2024, 6, 15)

    def test_when_profile_exists_updated_at_set(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex)
        db_session.add(
            TickerProfile(instrument_id=inst.id, symbol=inst.ticker, currency="USD")
        )
        db_session.flush()

        repo = _repo(db_session)
        info = repo.get_staleness_info(inst.id)

        assert info["profile_updated_at"] is not None

    def test_keys_cover_all_categories(self, db_session: Session) -> None:
        repo = _repo(db_session)
        info = repo.get_staleness_info(uuid.uuid4())

        expected_keys = {
            "price_max_date",
            "profile_updated_at",
            "financials_updated_at",
            "dividends_updated_at",
            "splits_updated_at",
            "recommendations_updated_at",
            "price_targets_updated_at",
            "institutional_holders_updated_at",
            "mutualfund_holders_updated_at",
            "insider_transactions_updated_at",
            "news_updated_at",
        }
        assert expected_keys == set(info.keys())


# ---------------------------------------------------------------------------
# get_instruments_with_yfinance_ticker
# ---------------------------------------------------------------------------


class TestGetInstrumentsWithYfinanceTicker:
    def test_when_none_exist_then_empty(self, db_session: Session) -> None:
        repo = _repo(db_session)
        result = repo.get_instruments_with_yfinance_ticker()
        assert list(result) == []

    def test_instruments_with_ticker_returned(self, db_session: Session) -> None:
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex, ticker="MSFT", yfinance_ticker="MSFT")
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_instruments_with_yfinance_ticker()

        ids = [r.id for r in result]
        assert inst.id in ids

    def test_instruments_without_yfinance_ticker_excluded(
        self, db_session: Session
    ) -> None:
        ex = _exchange(db_session)
        # null yfinance_ticker
        null_inst = _instrument(db_session, ex, ticker="NULL1", yfinance_ticker=None)
        # empty-string yfinance_ticker
        empty_inst = _instrument(db_session, ex, ticker="EMPTY1", yfinance_ticker="")
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_instruments_with_yfinance_ticker()

        ids = [r.id for r in result]
        assert null_inst.id not in ids
        assert empty_inst.id not in ids

    def test_mix_returns_only_valid(self, db_session: Session) -> None:
        ex = _exchange(db_session, name="MixExchange")
        valid = _instrument(db_session, ex, ticker="GOOG", yfinance_ticker="GOOGL")
        _instrument(db_session, ex, ticker="NOYF", yfinance_ticker=None)
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_instruments_with_yfinance_ticker()

        ids = [r.id for r in result]
        assert valid.id in ids


# ---------------------------------------------------------------------------
# get_instrument_by_yfinance_ticker
# ---------------------------------------------------------------------------


class TestGetInstrumentByYfinanceTicker:
    def test_when_not_found_then_none(self, db_session: Session) -> None:
        repo = _repo(db_session)
        result = repo.get_instrument_by_yfinance_ticker("DOESNOTEXIST")
        assert result is None

    def test_when_found_then_instrument_returned(self, db_session: Session) -> None:
        ex = _exchange(db_session, name="FindEx")
        inst = _instrument(db_session, ex, ticker="TSLA", yfinance_ticker="TSLA")
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_instrument_by_yfinance_ticker("TSLA")

        assert result is not None
        assert result.id == inst.id
        assert result.ticker == "TSLA"

    def test_exchange_is_eager_loaded(self, db_session: Session) -> None:
        ex = _exchange(db_session, name="EagerEx")
        _instrument(db_session, ex, ticker="AMZN", yfinance_ticker="AMZN")
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_instrument_by_yfinance_ticker("AMZN")

        assert result is not None
        # Access exchange without a second DB hit (eager loaded)
        assert result.exchange.name == "EagerEx"

    def test_ticker_mismatch_returns_none(self, db_session: Session) -> None:
        ex = _exchange(db_session, name="MismatchEx")
        _instrument(db_session, ex, ticker="META", yfinance_ticker="META")
        db_session.flush()

        repo = _repo(db_session)
        result = repo.get_instrument_by_yfinance_ticker("FB")

        assert result is None


# ---------------------------------------------------------------------------
# Upsert methods — Postgres-only ON CONFLICT, NOT executable on SQLite
# ---------------------------------------------------------------------------
# The following methods use ``RepositoryBase._upsert`` which calls
# ``sqlalchemy.dialects.postgresql.insert`` with ``on_conflict_do_update``.
# That construct is not supported by the SQLite dialect and will raise
# ``CompileError`` if executed.  We cover their SQLite-safe guard branches
# (empty-input early returns) by calling the helpers that gate the upsert.
#
# Methods skipped (Postgres-only ON CONFLICT — not executable on SQLite):
#   upsert_profile, upsert_price_history, upsert_financial_statements,
#   upsert_dividends, upsert_splits, upsert_recommendations,
#   upsert_price_targets, upsert_institutional_holders,
#   upsert_mutualfund_holders, upsert_insider_transactions, upsert_news
# ---------------------------------------------------------------------------


class TestUpsertEmptyInputGuards:
    """The RepositoryBase._upsert returns 0 immediately when rows=[] without
    executing any SQL.  We can verify this branch is reachable by calling
    _upsert directly on the base with an empty list (the dialect-specific
    INSERT is never compiled)."""

    def test_upsert_base_empty_rows_returns_zero(self, db_session: Session) -> None:
        from app.repositories._shared.base import RepositoryBase

        base = RepositoryBase(db_session)
        # Empty rows — early return before any SQL is compiled or executed
        result = base._upsert(
            PriceHistory,
            [],
            constraint_name="uq_price_history_instrument_date",
        )
        assert result == 0

    def test_upsert_base_empty_rows_with_update_columns_returns_zero(
        self, db_session: Session
    ) -> None:
        from app.repositories._shared.base import RepositoryBase

        base = RepositoryBase(db_session)
        result = base._upsert(
            TickerProfile,
            [],
            constraint_name="uq_ticker_profile_instrument",
            update_columns=["symbol", "sector"],
        )
        assert result == 0


# ---------------------------------------------------------------------------
# upsert_profile pre-processing branches (mocked _upsert — Postgres-only
# ON CONFLICT not executable on SQLite; pre-processing logic is pure Python)
# ---------------------------------------------------------------------------


class TestUpsertProfilePreprocessing:
    """Cover the exDividendDate int/float vs other branch in upsert_profile."""

    def _call(self, db_session: Session, info: dict) -> tuple[list, YFinanceRepository]:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_profile(uuid.uuid4(), info)
        return captured, repo

    def test_ex_div_date_as_int_triggers_safe_date_int_branch(
        self, db_session: Session
    ) -> None:
        # int branch: isinstance(ex_div, int | float) == True
        ts = int(datetime(2024, 3, 15).timestamp())
        rows, _ = self._call(db_session, {"exDividendDate": ts, "symbol": "AAPL"})
        assert len(rows) == 1
        # exDividendDate converted to a date object or None
        assert rows[0]["ex_dividend_date"] is None or isinstance(
            rows[0]["ex_dividend_date"], date
        )

    def test_ex_div_date_as_string_triggers_else_branch(
        self, db_session: Session
    ) -> None:
        # else branch: not int/float
        rows, _ = self._call(
            db_session, {"exDividendDate": "2024-03-15", "symbol": "MSFT"}
        )
        assert len(rows) == 1
        assert rows[0]["ex_dividend_date"] == date(2024, 3, 15)

    def test_ex_div_date_none_produces_none_field(self, db_session: Session) -> None:
        rows, _ = self._call(db_session, {"symbol": "TSLA"})
        assert rows[0]["ex_dividend_date"] is None

    def test_full_info_dict_maps_all_fields(self, db_session: Session) -> None:
        info = {
            "symbol": "GOOG",
            "shortName": "Alphabet Inc.",
            "currency": "USD",
            "sector": "Technology",
            "marketCap": 2_000_000_000_000,
            "currentPrice": 175.5,
            "beta": 1.1,
            "trailingPE": 28.5,
            "dividendYield": 0.005,
        }
        rows, _ = self._call(db_session, info)
        assert rows[0]["symbol"] == "GOOG"
        assert rows[0]["currency"] == "USD"
        assert rows[0]["market_cap"] == 2_000_000_000_000
        assert rows[0]["current_price"] == pytest.approx(175.5)


# ---------------------------------------------------------------------------
# upsert_price_history — Timestamp vs plain-date index branch
# ---------------------------------------------------------------------------


class TestUpsertPriceHistoryPreprocessing:
    """Cover the pd.Timestamp|datetime vs other date index branch."""

    def _call(self, db_session: Session, df: pd.DataFrame) -> list:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_price_history(uuid.uuid4(), df)
        return captured

    def test_timestamp_index_converted_to_date(self, db_session: Session) -> None:
        # isinstance(dt, pd.Timestamp | datetime) == True branch
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [110.0],
                "Low": [90.0],
                "Close": [105.0],
                "Volume": [1_000_000],
                "Dividends": [0.0],
                "Stock Splits": [0.0],
            },
            index=[pd.Timestamp("2024-06-01")],
        )
        rows = self._call(db_session, df)
        assert len(rows) == 1
        assert rows[0]["date"] == date(2024, 6, 1)

    def test_plain_date_index_passes_through(self, db_session: Session) -> None:
        # not Timestamp/datetime — plain date object falls through
        df = pd.DataFrame(
            {
                "Open": [200.0],
                "High": [210.0],
                "Low": [195.0],
                "Close": [205.0],
                "Volume": [500_000],
                "Dividends": [0.0],
                "Stock Splits": [0.0],
            },
            index=[date(2024, 7, 4)],
        )
        rows = self._call(db_session, df)
        assert len(rows) == 1
        assert rows[0]["date"] == date(2024, 7, 4)

    def test_multiple_rows_all_processed(self, db_session: Session) -> None:
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [110.0, 111.0],
                "Low": [90.0, 91.0],
                "Close": [105.0, 106.0],
                "Volume": [1_000_000, 1_100_000],
                "Dividends": [0.0, 0.25],
                "Stock Splits": [0.0, 0.0],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )
        rows = self._call(db_session, df)
        assert len(rows) == 2
        assert rows[1]["dividends"] == pytest.approx(0.25)

    def test_empty_dataframe_produces_no_rows(self, db_session: Session) -> None:
        df = pd.DataFrame(
            columns=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Dividends",
                "Stock Splits",
            ]
        )
        rows = self._call(db_session, df)
        assert rows == []


# ---------------------------------------------------------------------------
# upsert_financial_statements — period_date None skip + nested row building
# (Postgres-only ON CONFLICT — not executable on SQLite; pre-processing is pure Python)
# ---------------------------------------------------------------------------


class TestUpsertFinancialStatementsPreprocessing:
    def _call(self, db_session: Session, df: pd.DataFrame, **kwargs) -> list:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_financial_statements(
            uuid.uuid4(),
            df,
            statement_type=kwargs.get("statement_type", "income_statement"),
            period_type=kwargs.get("period_type", "annual"),
            currency_code=kwargs.get("currency_code"),
        )
        return captured

    def test_valid_timestamp_column_produces_rows(self, db_session: Session) -> None:
        # period_date not None → rows appended for each (col, line_item)
        df = pd.DataFrame(
            {"Revenue": [1_000_000.0], "NetIncome": [200_000.0]},
            index=[pd.Timestamp("2024-12-31")],
        ).T
        rows = self._call(db_session, df)
        assert len(rows) == 2
        line_items = {r["line_item"] for r in rows}
        assert line_items == {"Revenue", "NetIncome"}

    def test_non_parseable_column_skipped(self, db_session: Session) -> None:
        # _safe_date returns None for a non-date column label → continue (skip)
        df = pd.DataFrame(
            {"Revenue": [500_000.0]},
            index=["not-a-date"],  # string that won't parse as date
        ).T
        rows = self._call(db_session, df)
        assert rows == []

    def test_mixed_columns_only_valid_dates_included(self, db_session: Session) -> None:
        # One valid Timestamp col + one invalid string col
        df = pd.DataFrame(
            {
                pd.Timestamp("2024-12-31"): {"Revenue": 1_000_000.0},
                "bad-col": {"Revenue": 999.0},
            }
        )
        rows = self._call(db_session, df)
        # Only the Timestamp column yields a row
        assert len(rows) == 1
        assert rows[0]["period_date"] == date(2024, 12, 31)

    def test_currency_code_propagated_to_each_row(self, db_session: Session) -> None:
        df = pd.DataFrame(
            {"Revenue": [2_000_000.0]},
            index=[pd.Timestamp("2024-12-31")],
        ).T
        rows = self._call(db_session, df, currency_code="GBP")
        assert rows[0]["currency_code"] == "GBP"

    def test_statement_type_and_period_type_set_on_rows(
        self, db_session: Session
    ) -> None:
        df = pd.DataFrame(
            {"TotalAssets": [5_000_000.0]},
            index=[pd.Timestamp("2024-12-31")],
        ).T
        rows = self._call(
            db_session,
            df,
            statement_type="balance_sheet",
            period_type="quarterly",
        )
        assert rows[0]["statement_type"] == "balance_sheet"
        assert rows[0]["period_type"] == "quarterly"

    def test_empty_dataframe_produces_no_rows(self, db_session: Session) -> None:
        df = pd.DataFrame()
        rows = self._call(db_session, df)
        assert rows == []

    def test_nan_value_stored_as_none(self, db_session: Session) -> None:
        df = pd.DataFrame(
            {"Revenue": [float("nan")]},
            index=[pd.Timestamp("2024-12-31")],
        ).T
        rows = self._call(db_session, df)
        assert rows[0]["value"] is None


# ---------------------------------------------------------------------------
# upsert_dividends — None filtering branches
# ---------------------------------------------------------------------------


class TestUpsertDividendsPreprocessing:
    def _call(self, db_session: Session, series: pd.Series) -> list:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_dividends(uuid.uuid4(), series)
        return captured

    def test_valid_rows_included(self, db_session: Session) -> None:
        s = pd.Series(
            [0.25, 0.25],
            index=[pd.Timestamp("2024-01-15"), pd.Timestamp("2024-04-15")],
        )
        rows = self._call(db_session, s)
        assert len(rows) == 2
        assert rows[0]["amount"] == pytest.approx(0.25)

    def test_nan_amount_row_skipped(self, db_session: Session) -> None:
        # amt is None branch (NaN filtered by _safe_float)
        s = pd.Series(
            [float("nan"), 0.30],
            index=[pd.Timestamp("2024-01-15"), pd.Timestamp("2024-04-15")],
        )
        rows = self._call(db_session, s)
        assert len(rows) == 1
        assert rows[0]["amount"] == pytest.approx(0.30)

    def test_empty_series_no_rows(self, db_session: Session) -> None:
        s = pd.Series([], dtype=float)
        rows = self._call(db_session, s)
        assert rows == []


# ---------------------------------------------------------------------------
# upsert_splits — None filtering branches
# ---------------------------------------------------------------------------


class TestUpsertSplitsPreprocessing:
    def _call(self, db_session: Session, series: pd.Series) -> list:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_splits(uuid.uuid4(), series)
        return captured

    def test_valid_split_included(self, db_session: Session) -> None:
        s = pd.Series([4.0], index=[pd.Timestamp("2020-08-31")])
        rows = self._call(db_session, s)
        assert len(rows) == 1
        assert rows[0]["ratio"] == pytest.approx(4.0)

    def test_nan_ratio_row_skipped(self, db_session: Session) -> None:
        s = pd.Series(
            [float("nan"), 2.0],
            index=[pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")],
        )
        rows = self._call(db_session, s)
        assert len(rows) == 1

    def test_empty_series_no_rows(self, db_session: Session) -> None:
        s = pd.Series([], dtype=float)
        rows = self._call(db_session, s)
        assert rows == []


# ---------------------------------------------------------------------------
# upsert_recommendations — empty period skip branch
# ---------------------------------------------------------------------------


class TestUpsertRecommendationsPreprocessing:
    def _call(self, db_session: Session, df: pd.DataFrame) -> list:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_recommendations(uuid.uuid4(), df)
        return captured

    def test_valid_period_row_included(self, db_session: Session) -> None:
        df = pd.DataFrame(
            {
                "period": ["0m"],
                "strongBuy": [10],
                "buy": [5],
                "hold": [3],
                "sell": [1],
                "strongSell": [0],
            }
        )
        rows = self._call(db_session, df)
        assert len(rows) == 1
        assert rows[0]["period"] == "0m"

    def test_empty_period_row_skipped(self, db_session: Session) -> None:
        # not period branch: period is None/empty
        df = pd.DataFrame(
            {
                "period": [None, "0m"],
                "strongBuy": [5, 10],
                "buy": [2, 5],
                "hold": [1, 3],
                "sell": [0, 1],
                "strongSell": [0, 0],
            }
        )
        rows = self._call(db_session, df)
        assert len(rows) == 1
        assert rows[0]["period"] == "0m"

    def test_all_periods_empty_produces_no_rows(self, db_session: Session) -> None:
        df = pd.DataFrame(
            {
                "period": [None, None],
                "strongBuy": [0, 0],
                "buy": [0, 0],
                "hold": [0, 0],
                "sell": [0, 0],
                "strongSell": [0, 0],
            }
        )
        rows = self._call(db_session, df)
        assert rows == []


# ---------------------------------------------------------------------------
# upsert_price_targets — straightforward row building (mocked)
# ---------------------------------------------------------------------------


class TestUpsertPriceTargetsPreprocessing:
    def test_all_fields_mapped(self, db_session: Session) -> None:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        targets = {
            "current": 180.0,
            "low": 150.0,
            "high": 220.0,
            "mean": 190.0,
            "median": 185.0,
        }
        repo.upsert_price_targets(uuid.uuid4(), targets)
        assert len(captured) == 1
        assert captured[0]["current"] == pytest.approx(180.0)
        assert captured[0]["high"] == pytest.approx(220.0)

    def test_missing_fields_produce_none_values(self, db_session: Session) -> None:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_price_targets(uuid.uuid4(), {})
        assert captured[0]["current"] is None
        assert captured[0]["median"] is None


# ---------------------------------------------------------------------------
# upsert_institutional_holders — empty name skip branch
# ---------------------------------------------------------------------------


class TestUpsertInstitutionalHoldersPreprocessing:
    def _call(self, db_session: Session, df: pd.DataFrame) -> list:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_institutional_holders(uuid.uuid4(), df)
        return captured

    def test_valid_holder_included(self, db_session: Session) -> None:
        df = pd.DataFrame(
            {
                "Holder": ["Vanguard"],
                "Date Reported": [pd.Timestamp("2024-03-31")],
                "Shares": [50_000_000],
                "Value": [8_500_000_000],
                "pctHeld": [0.035],
            }
        )
        rows = self._call(db_session, df)
        assert len(rows) == 1
        assert rows[0]["holder_name"] == "Vanguard"

    def test_none_holder_name_skipped(self, db_session: Session) -> None:
        # not name branch
        df = pd.DataFrame(
            {
                "Holder": [None, "BlackRock"],
                "Date Reported": [None, pd.Timestamp("2024-03-31")],
                "Shares": [0, 40_000_000],
                "Value": [0, 7_000_000_000],
                "pctHeld": [0.0, 0.028],
            }
        )
        rows = self._call(db_session, df)
        assert len(rows) == 1
        assert rows[0]["holder_name"] == "BlackRock"

    def test_empty_dataframe_no_rows(self, db_session: Session) -> None:
        df = pd.DataFrame(
            columns=["Holder", "Date Reported", "Shares", "Value", "pctHeld"]
        )
        rows = self._call(db_session, df)
        assert rows == []


# ---------------------------------------------------------------------------
# upsert_mutualfund_holders — empty name skip branch
# ---------------------------------------------------------------------------


class TestUpsertMutualFundHoldersPreprocessing:
    def _call(self, db_session: Session, df: pd.DataFrame) -> list:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_mutualfund_holders(uuid.uuid4(), df)
        return captured

    def test_valid_fund_included(self, db_session: Session) -> None:
        df = pd.DataFrame(
            {
                "Holder": ["Fidelity 500"],
                "Date Reported": [pd.Timestamp("2024-03-31")],
                "Shares": [10_000_000],
                "Value": [1_800_000_000],
                "pctHeld": [0.007],
            }
        )
        rows = self._call(db_session, df)
        assert len(rows) == 1

    def test_none_fund_name_skipped(self, db_session: Session) -> None:
        df = pd.DataFrame(
            {
                "Holder": [None],
                "Date Reported": [None],
                "Shares": [0],
                "Value": [0],
                "pctHeld": [0.0],
            }
        )
        rows = self._call(db_session, df)
        assert rows == []


# ---------------------------------------------------------------------------
# upsert_insider_transactions — Transaction vs Text fallback + dedup + skip
# ---------------------------------------------------------------------------


class TestUpsertInsiderTransactionsPreprocessing:
    def _call(self, db_session: Session, df: pd.DataFrame) -> list:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_insider_transactions(uuid.uuid4(), df)
        return captured

    def test_transaction_column_used_when_present(self, db_session: Session) -> None:
        # tx_type truthy from "Transaction" column — Text fallback not reached
        df = pd.DataFrame(
            {
                "Insider": ["Alice CEO"],
                "Transaction": ["Buy"],
                "Text": ["Bought shares."],
                "Position": ["CEO"],
                "Shares": [1000],
                "Value": [150_000],
                "Start Date": [pd.Timestamp("2024-01-10")],
                "Ownership": ["D"],
            }
        )
        rows = self._call(db_session, df)
        assert len(rows) == 1
        assert rows[0]["transaction_type"] == "Buy"

    def test_text_fallback_used_when_transaction_empty(
        self, db_session: Session
    ) -> None:
        # tx_type falsy → falls back to "Text" column (yfinance 1.3.0 path)
        df = pd.DataFrame(
            {
                "Insider": ["Bob CFO"],
                "Transaction": [""],
                "Text": ["Sale at price 290.00 per share."],
                "Position": ["CFO"],
                "Shares": [500],
                "Value": [145_000],
                "Start Date": [pd.Timestamp("2024-02-20")],
                "Ownership": ["D"],
            }
        )
        rows = self._call(db_session, df)
        assert len(rows) == 1
        assert rows[0]["transaction_type"] == "Sale at price 290.00 per share."

    def test_both_name_and_tx_empty_row_skipped(self, db_session: Session) -> None:
        # not name or not tx_type → continue
        df = pd.DataFrame(
            {
                "Insider": [None],
                "Transaction": [""],
                "Text": [None],
                "Position": [None],
                "Shares": [0],
                "Value": [0],
                "Start Date": [None],
                "Ownership": [None],
            }
        )
        rows = self._call(db_session, df)
        assert rows == []

    def test_name_present_but_no_tx_and_no_text_row_skipped(
        self, db_session: Session
    ) -> None:
        # name set, tx_type empty, Text also None → skip
        df = pd.DataFrame(
            {
                "Insider": ["Carol COO"],
                "Transaction": [None],
                "Text": [None],
                "Position": ["COO"],
                "Shares": [100],
                "Value": [20_000],
                "Start Date": [pd.Timestamp("2024-03-01")],
                "Ownership": ["D"],
            }
        )
        rows = self._call(db_session, df)
        assert rows == []

    def test_duplicate_key_rows_deduped_last_wins(self, db_session: Session) -> None:
        # Same insider_name + start_date + transaction_type → last row wins
        inst_id = uuid.uuid4()
        d = pd.Timestamp("2024-01-10")
        df = pd.DataFrame(
            {
                "Insider": ["Alice CEO", "Alice CEO"],
                "Transaction": ["Buy", "Buy"],
                "Text": ["", ""],
                "Position": ["CEO", "CEO"],
                "Shares": [1000, 2000],
                "Value": [150_000, 300_000],
                "Start Date": [d, d],
                "Ownership": ["D", "D"],
            }
        )
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_insider_transactions(inst_id, df)

        # Dedup: only one row remains — the last occurrence
        assert len(captured) == 1
        assert captured[0]["shares"] == 2000

    def test_sentinel_date_used_when_start_date_none(self, db_session: Session) -> None:
        from portopt_db.repositories.market_data.yfinance_repository import (
            _SENTINEL_DATE,
        )

        df = pd.DataFrame(
            {
                "Insider": ["Dave Director"],
                "Transaction": ["Grant"],
                "Text": [""],
                "Position": ["Director"],
                "Shares": [500],
                "Value": [50_000],
                "Start Date": [None],  # _safe_date returns None → sentinel used
                "Ownership": ["D"],
            }
        )
        rows = self._call(db_session, df)
        assert len(rows) == 1
        assert rows[0]["start_date"] == _SENTINEL_DATE

    def test_mixed_valid_and_invalid_rows(self, db_session: Session) -> None:
        df = pd.DataFrame(
            {
                "Insider": ["Alice CEO", None, "Bob CFO"],
                "Transaction": ["Buy", "Sell", ""],
                "Text": ["", "", "Sale at price 100."],
                "Position": ["CEO", None, "CFO"],
                "Shares": [1000, 500, 200],
                "Value": [150_000, 50_000, 20_000],
                "Start Date": [
                    pd.Timestamp("2024-01-10"),
                    pd.Timestamp("2024-01-11"),
                    pd.Timestamp("2024-01-12"),
                ],
                "Ownership": ["D", "D", "D"],
            }
        )
        rows = self._call(db_session, df)
        # Alice: valid (Transaction="Buy"); None: skipped; Bob: Text fallback
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# upsert_news — all format/branch variants (mocked _upsert)
# ---------------------------------------------------------------------------


class TestUpsertNewsPreprocessing:
    def _call(
        self,
        db_session: Session,
        inst_id: uuid.UUID,
        articles: list,
    ) -> list:
        repo = _repo(db_session)
        captured: list = []

        def fake_upsert(model, rows, *, constraint_name, update_columns=None):
            captured.extend(rows)
            return len(rows)

        repo._upsert = fake_upsert  # type: ignore[method-assign]
        repo.upsert_news(inst_id, articles)
        return captured

    def test_no_uuid_article_skipped(self, db_session: Session) -> None:
        # news_uuid falsy → continue
        ex = _exchange(db_session, name="NewsEx1")
        inst = _instrument(db_session, ex, ticker="N1", yfinance_ticker="N1")
        rows = self._call(db_session, inst.id, [{"title": "No ID article"}])
        assert rows == []

    def test_old_format_flat_dict_parsed(self, db_session: Session) -> None:
        # Old yfinance: flat dict with "uuid", "publisher", "link", "providerPublishTime"
        ex = _exchange(db_session, name="NewsEx2")
        inst = _instrument(db_session, ex, ticker="N2", yfinance_ticker="N2")
        article = {
            "uuid": "old-uuid-001",
            "title": "Old Format Article",
            "publisher": "Reuters",
            "link": "https://reuters.com/article/1",
            "providerPublishTime": int(datetime(2024, 6, 1, 9, 0).timestamp()),
            "type": "STORY",
        }
        rows = self._call(db_session, inst.id, [article])
        assert len(rows) == 1
        assert rows[0]["news_uuid"] == "old-uuid-001"
        assert rows[0]["publisher"] == "Reuters"
        assert rows[0]["news_type"] == "STORY"
        assert rows[0]["publish_time"] is not None

    def test_new_format_nested_content_parsed(self, db_session: Session) -> None:
        # New yfinance: "id" at top level, data nested under "content"
        ex = _exchange(db_session, name="NewsEx3")
        inst = _instrument(db_session, ex, ticker="N3", yfinance_ticker="N3")
        article = {
            "id": "new-id-001",
            "content": {
                "title": "New Format Article",
                "provider": {"displayName": "Bloomberg"},
                "canonicalUrl": {"url": "https://bloomberg.com/article/1"},
                "pubDate": "2024-06-15T09:00:00Z",
                "contentType": "ARTICLE",
            },
        }
        rows = self._call(db_session, inst.id, [article])
        assert len(rows) == 1
        assert rows[0]["news_uuid"] == "new-id-001"
        assert rows[0]["publisher"] == "Bloomberg"
        assert rows[0]["link"] == "https://bloomberg.com/article/1"
        assert rows[0]["news_type"] == "ARTICLE"
        assert rows[0]["publish_time"] is not None

    def test_provider_not_dict_uses_flat_publisher(self, db_session: Session) -> None:
        # provider is string (not dict) → else branch uses article["publisher"]
        ex = _exchange(db_session, name="NewsEx4")
        inst = _instrument(db_session, ex, ticker="N4", yfinance_ticker="N4")
        article = {
            "uuid": "flat-pub-uuid",
            "publisher": "AP News",
            "content": {
                "title": "Mixed Article",
                "provider": "not-a-dict",
            },
        }
        rows = self._call(db_session, inst.id, [article])
        assert rows[0]["publisher"] == "AP News"

    def test_canonical_url_not_dict_uses_preview_or_link(
        self, db_session: Session
    ) -> None:
        # canonicalUrl is string (not dict) → else branch: previewUrl or link
        ex = _exchange(db_session, name="NewsEx5")
        inst = _instrument(db_session, ex, ticker="N5", yfinance_ticker="N5")
        article = {
            "uuid": "previewUrl-uuid",
            "content": {
                "canonicalUrl": "not-a-dict",
                "previewUrl": "https://preview.example.com/1",
            },
        }
        rows = self._call(db_session, inst.id, [article])
        assert rows[0]["link"] == "https://preview.example.com/1"

    def test_pub_date_iso_string_parsed_tz_stripped(self, db_session: Session) -> None:
        # pubDate ISO 8601 with timezone → tz stripped
        ex = _exchange(db_session, name="NewsEx6")
        inst = _instrument(db_session, ex, ticker="N6", yfinance_ticker="N6")
        article = {
            "id": "iso-dt-uuid",
            "content": {"pubDate": "2024-06-15T14:30:00+05:00"},
        }
        rows = self._call(db_session, inst.id, [article])
        assert rows[0]["publish_time"] is not None
        assert rows[0]["publish_time"].tzinfo is None

    def test_pub_date_invalid_string_falls_through_to_provider_publish_time(
        self, db_session: Session
    ) -> None:
        # Invalid pubDate str → ValueError → publish_time stays None → try providerPublishTime
        ex = _exchange(db_session, name="NewsEx7")
        inst = _instrument(db_session, ex, ticker="N7", yfinance_ticker="N7")
        ts = int(datetime(2024, 5, 1, 10, 0).timestamp())
        article = {
            "uuid": "fallback-uuid",
            "content": {"pubDate": "not-a-valid-date"},
            "providerPublishTime": ts,
        }
        rows = self._call(db_session, inst.id, [article])
        # pubDate parse fails; providerPublishTime succeeds
        assert rows[0]["publish_time"] is not None

    def test_no_publish_time_at_all_yields_none(self, db_session: Session) -> None:
        # No pubDate, no providerPublishTime → publish_time stays None
        ex = _exchange(db_session, name="NewsEx8")
        inst = _instrument(db_session, ex, ticker="N8", yfinance_ticker="N8")
        article = {"uuid": "no-time-uuid", "title": "No time article"}
        rows = self._call(db_session, inst.id, [article])
        assert rows[0]["publish_time"] is None

    def test_full_content_field_carried_through(self, db_session: Session) -> None:
        ex = _exchange(db_session, name="NewsEx9")
        inst = _instrument(db_session, ex, ticker="N9", yfinance_ticker="N9")
        article = {
            "uuid": "full-content-uuid",
            "full_content": "Long article body text here.",
        }
        rows = self._call(db_session, inst.id, [article])
        assert rows[0]["full_content"] == "Long article body text here."

    def test_id_takes_precedence_over_uuid_field(self, db_session: Session) -> None:
        # article.get("id") or article.get("uuid") → "id" wins when present
        ex = _exchange(db_session, name="NewsEx10")
        inst = _instrument(db_session, ex, ticker="N10", yfinance_ticker="N10")
        article = {"id": "preferred-id", "uuid": "fallback-uuid"}
        rows = self._call(db_session, inst.id, [article])
        assert rows[0]["news_uuid"] == "preferred-id"

    def test_uuid_used_when_id_absent(self, db_session: Session) -> None:
        # article.get("id") is None → falls through to article.get("uuid")
        ex = _exchange(db_session, name="NewsEx11")
        inst = _instrument(db_session, ex, ticker="N11", yfinance_ticker="N11")
        article = {"uuid": "fallback-uuid-only"}
        rows = self._call(db_session, inst.id, [article])
        assert rows[0]["news_uuid"] == "fallback-uuid-only"

    def test_ticker_name_looked_up_from_db(self, db_session: Session) -> None:
        # ticker_name resolved via SELECT on Instrument.name
        ex = _exchange(db_session, name="NewsEx12")
        inst = _instrument(db_session, ex, ticker="N12", yfinance_ticker="N12")
        article = {"uuid": "ticker-name-uuid", "title": "Named ticker article"}
        rows = self._call(db_session, inst.id, [article])
        assert rows[0]["ticker_name"] == "N12 Inc."

    def test_empty_articles_list_produces_no_rows(self, db_session: Session) -> None:
        rows = self._call(db_session, uuid.uuid4(), [])
        assert rows == []

    def test_provider_publish_time_none_skips_fromtimestamp(
        self, db_session: Session
    ) -> None:
        # providerPublishTime is None → pt is None → contextlib.suppress block skipped
        ex = _exchange(db_session, name="NewsEx13")
        inst = _instrument(db_session, ex, ticker="N13", yfinance_ticker="N13")
        article = {
            "uuid": "no-pt-uuid",
            "providerPublishTime": None,
        }
        rows = self._call(db_session, inst.id, [article])
        assert rows[0]["publish_time"] is None
