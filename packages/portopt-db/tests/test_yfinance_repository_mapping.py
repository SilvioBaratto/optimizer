"""Row-mapping / dedup coverage for the PostgreSQL-only Phase-A upserts.

These upserts call ``self._upsert(..., constraint_name=...)`` (ON CONFLICT), so
they never run against the SQLite test engine — only their service wiring was
covered (mock repo). Here ``_upsert`` is patched to capture the rows the repo
builds, so the defensive column mapping, ``_safe_*`` coercion, and in-batch
dedup are asserted directly. yfinance shapes mirror the 1.6.0 payloads.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock
from uuid import uuid4

import pandas as pd
import pytest
from sqlalchemy.dialects import postgresql

from portopt_db.repositories.market_data.yfinance_repository import (
    YFinanceRepository,
    _safe_date,
)

_IID = uuid4()


class TestSafeDate:
    """pd.NaT subclasses datetime, so it must be coerced to None, not NaT."""

    def test_nat_is_none(self) -> None:
        assert _safe_date(pd.NaT) is None

    def test_none_is_none(self) -> None:
        assert _safe_date(None) is None

    def test_timestamp_becomes_date(self) -> None:
        assert _safe_date(pd.Timestamp("2024-03-31")) == dt.date(2024, 3, 31)

    def test_isoformat_string(self) -> None:
        assert _safe_date("2024-05-01") == dt.date(2024, 5, 1)

    def test_us_month_day_year_label_without_zero_padding(self) -> None:
        # yfinance valuation-measure column labels are "M/D/YYYY".
        assert _safe_date("9/30/2025") == dt.date(2025, 9, 30)
        assert _safe_date("12/31/2024") == dt.date(2024, 12, 31)

    def test_non_date_label_is_none(self) -> None:
        assert _safe_date("Current") is None


@pytest.fixture
def repo() -> YFinanceRepository:
    r = YFinanceRepository(MagicMock(name="session"))
    r._upsert = MagicMock(return_value=0)  # type: ignore[method-assign]
    return r


def _rows(repo: YFinanceRepository) -> list[dict]:
    """The ``rows`` positional arg passed to the last ``_upsert`` call."""
    return repo._upsert.call_args.args[1]


def _constraint(repo: YFinanceRepository) -> str:
    return repo._upsert.call_args.kwargs["constraint_name"]


class TestProfileRatioUnits:
    """Yahoo quotes dividendYield, fiveYearAvgDividendYield and debtToEquity in
    percent but trailingAnnualDividendYield and payoutRatio as fractions; the
    row must carry one unit."""

    def test_percent_quoted_fields_are_stored_as_fractions(self, repo) -> None:
        repo.upsert_profile(
            _IID,
            {
                "dividendYield": 0.31,
                "fiveYearAvgDividendYield": 0.47,
                "trailingAnnualDividendYield": 0.003015,
                "payoutRatio": 0.1204,
                "debtToEquity": 78.445,
            },
        )
        row = _rows(repo)[0]
        assert row["dividend_yield"] == pytest.approx(0.0031)
        assert row["five_year_avg_dividend_yield"] == pytest.approx(0.0047)
        assert row["debt_to_equity"] == pytest.approx(0.78445)
        # Already fractions at the source.
        assert row["trailing_annual_dividend_yield"] == 0.003015
        assert row["payout_ratio"] == 0.1204

    def test_dividend_yield_agrees_with_the_trailing_fraction(self, repo) -> None:
        repo.upsert_profile(
            _IID, {"dividendYield": 2.22, "trailingAnnualDividendYield": 0.0218}
        )
        row = _rows(repo)[0]
        assert row["dividend_yield"] == pytest.approx(
            row["trailing_annual_dividend_yield"], rel=0.05
        )

    def test_absent_percent_fields_stay_none(self, repo) -> None:
        repo.upsert_profile(_IID, {"symbol": "X"})
        row = _rows(repo)[0]
        assert row["dividend_yield"] is None
        assert row["five_year_avg_dividend_yield"] is None
        assert row["debt_to_equity"] is None


class TestDegradedInfoKeepsStoredDescription:
    """When quoteSummary fails, yfinance falls back to the v7 quote payload alone:
    a non-empty ``info`` with no sector, industry, country, website or summary.
    The ON CONFLICT update must not overwrite what an earlier full fetch stored
    with NULL for those columns."""

    @staticmethod
    def _set_clause(session: MagicMock) -> str:
        stmt = session.execute.call_args.args[0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        return sql.split("DO UPDATE SET", 1)[1]

    @pytest.fixture
    def session(self) -> MagicMock:
        return MagicMock(name="session")

    @pytest.fixture
    def degraded_info(self) -> dict:
        return {
            "symbol": "AAPL",
            "shortName": "Apple",
            "regularMarketPrice": 190.0,
            "marketCap": 3e12,
        }

    def test_missing_descriptive_fields_do_not_overwrite_stored_values(
        self, session, degraded_info
    ) -> None:
        YFinanceRepository(session).upsert_profile(_IID, degraded_info)
        update = self._set_clause(session)
        for column in (
            "sector",
            "industry",
            "country",
            "website",
            "long_business_summary",
            "isin",
        ):
            assert f"{column} = excluded.{column}" not in update

    def test_present_descriptive_fields_are_refreshed(self, session) -> None:
        info = {"symbol": "AAPL", "sector": "Technology", "industry": "Hardware"}
        YFinanceRepository(session).upsert_profile(_IID, info)
        update = self._set_clause(session)
        assert "sector = excluded.sector" in update
        assert "industry = excluded.industry" in update

    def test_market_fields_still_mirror_the_latest_response(
        self, session, degraded_info
    ) -> None:
        # A dropped dividend or lost analyst coverage must clear, not linger.
        YFinanceRepository(session).upsert_profile(_IID, degraded_info)
        update = self._set_clause(session)
        assert "dividend_yield = excluded.dividend_yield" in update
        assert "target_mean_price = excluded.target_mean_price" in update

    def test_row_is_still_stamped_on_conflict(self, session, degraded_info) -> None:
        YFinanceRepository(session).upsert_profile(_IID, degraded_info)
        assert "updated_at = clock_timestamp()" in self._set_clause(session)

    def test_extras_keep_stored_sector_and_industry_keys(
        self, session, degraded_info
    ) -> None:
        YFinanceRepository(session).upsert_profile_extras(_IID, degraded_info)
        update = self._set_clause(session)
        assert "sector_key = excluded.sector_key" not in update
        assert "industry_key = excluded.industry_key" not in update
        assert "shares_short = excluded.shares_short" in update


class TestAnalystEstimates:
    def test_earnings_estimate_maps_period_and_fields(self, repo) -> None:
        df = pd.DataFrame(
            {
                "numberOfAnalysts": [10, 12],
                "avg": [1.5, 1.7],
                "low": [1.4, 1.6],
                "high": [1.6, 1.8],
                "yearAgoEps": [1.2, 1.3],
                "growth": [0.1, 0.2],
            },
            index=["0q", "+1q"],
        )
        repo.upsert_earnings_estimate(_IID, df)
        rows = _rows(repo)
        assert [r["period"] for r in rows] == ["0q", "+1q"]
        assert rows[0]["num_analysts"] == 10
        assert rows[0]["year_ago_eps"] == 1.2
        assert _constraint(repo) == "uq_earnings_estimate_instrument_period"

    def test_growth_estimates_defensive_column_names(self, repo) -> None:
        # Short spellings (stock/index) rather than *Trend must still map.
        df = pd.DataFrame({"stock": [0.1], "index": [0.4]}, index=["0q"])
        repo.upsert_growth_estimates(_IID, df)
        r = _rows(repo)[0]
        assert (r["stock_trend"], r["index_trend"]) == (0.1, 0.4)

    def test_eps_trend_maps_period_and_snapshot_ages(self, repo) -> None:
        # index = period labels, columns = snapshot ages — neither axis is a date.
        df = pd.DataFrame(
            {
                "current": [1.50, 6.10],
                "7daysAgo": [1.51, 6.11],
                "30daysAgo": [1.52, 6.14],
                "60daysAgo": [1.55, 6.20],
                "90daysAgo": [1.58, 6.25],
            },
            index=["0q", "0y"],
        )
        repo.upsert_eps_trend(_IID, df)
        rows = _rows(repo)
        assert [r["period"] for r in rows] == ["0q", "0y"]
        assert rows[0]["current_estimate"] == 1.50
        assert rows[0]["days_ago_7"] == 1.51
        assert rows[1]["days_ago_90"] == 6.25
        assert _constraint(repo) == "uq_eps_trend_instrument_period"

    def test_eps_revisions_maps_period_and_counts(self, repo) -> None:
        # Yahoo spells only the 7-day down column with a capital D.
        df = pd.DataFrame(
            {
                "upLast7days": [2, 3],
                "upLast30days": [5, 7],
                "downLast30days": [1, 2],
                "downLast7Days": [4, 1],
                "currency": ["USD", "USD"],
            },
            index=["0q", "+1q"],
        )
        repo.upsert_eps_revisions(_IID, df)
        rows = _rows(repo)
        assert [r["period"] for r in rows] == ["0q", "+1q"]
        assert rows[0]["up_last_7days"] == 2
        assert rows[0]["down_last_7days"] == 4
        assert rows[1]["down_last_7days"] == 1
        assert rows[1]["down_last_30days"] == 2
        assert _constraint(repo) == "uq_eps_revisions_instrument_period"

    def test_eps_revisions_accepts_lowercase_down_last_7days(self, repo) -> None:
        df = pd.DataFrame(
            {
                "upLast7days": [2],
                "upLast30days": [5],
                "downLast7days": [3],
                "downLast30days": [1],
            },
            index=["0q"],
        )
        repo.upsert_eps_revisions(_IID, df)
        assert _rows(repo)[0]["down_last_7days"] == 3


class TestEstimateCurrency:
    """Yahoo labels each forward-estimate period with the currency the figures
    are quoted in, which can differ from the listing currency (a USD ADR with
    TWD estimates). Without the label the numbers cannot be read as money."""

    _FRAMES = [
        ("upsert_earnings_estimate", {"avg": [1.5, 1.7]}),
        ("upsert_revenue_estimate", {"avg": [1.0e12, 1.1e12]}),
        ("upsert_eps_trend", {"current": [1.5, 6.1]}),
        ("upsert_eps_revisions", {"upLast7days": [2, 3]}),
    ]

    @staticmethod
    def _frame(columns: dict, currency: str | None) -> pd.DataFrame:
        df = pd.DataFrame(columns, index=["0q", "+1q"])
        if currency is not None:
            df["currency"] = currency
        return df

    @pytest.mark.parametrize(("method", "columns"), _FRAMES)
    def test_rows_carry_the_currency_yahoo_reports(self, repo, method, columns) -> None:
        getattr(repo, method)(_IID, self._frame(columns, "TWD"))
        assert [r["currency"] for r in _rows(repo)] == ["TWD", "TWD"]

    @pytest.mark.parametrize(("method", "columns"), _FRAMES)
    def test_frame_without_a_currency_column_stores_null(
        self, repo, method, columns
    ) -> None:
        getattr(repo, method)(_IID, self._frame(columns, None))
        assert [r["currency"] for r in _rows(repo)] == [None, None]

    @pytest.mark.parametrize(("method", "columns"), _FRAMES)
    def test_unknown_currency_does_not_erase_a_stored_label(
        self, repo, method, columns
    ) -> None:
        getattr(repo, method)(_IID, self._frame(columns, None))
        update_columns = repo._upsert.call_args.kwargs.get("update_columns")
        assert update_columns is not None
        assert "currency" not in update_columns
        assert "updated_at" in update_columns

    @pytest.mark.parametrize(("method", "columns"), _FRAMES)
    def test_known_currency_relabels_the_stored_row(
        self, repo, method, columns
    ) -> None:
        getattr(repo, method)(_IID, self._frame(columns, "USD"))
        update_columns = repo._upsert.call_args.kwargs.get("update_columns")
        assert update_columns is None or "currency" in update_columns


class TestPriceHistory:
    def test_nat_index_row_dropped(self, repo) -> None:
        # A NaT in the DatetimeIndex must not become date=NaT in a NOT NULL col.
        df = pd.DataFrame(
            {"Close": [1.0, 2.0]},
            index=[pd.Timestamp("2024-01-02"), pd.NaT],
        )
        repo.upsert_price_history(_IID, df)
        rows = _rows(repo)
        assert len(rows) == 1
        assert rows[0]["date"] == dt.date(2024, 1, 2)


class TestEarningsTimeline:
    def test_earnings_history_dedups_on_period_date(self, repo) -> None:
        # Two source rows on the same day collapse to one conflict key; without
        # in-batch dedup PostgreSQL raises "cannot affect row a second time".
        df = pd.DataFrame(
            {
                "epsEstimate": [1.0, 1.5],
                "epsActual": [1.1, 1.6],
                "epsDifference": [0.1, 0.1],
                "surprisePercent": [10.0, 5.0],
            },
            index=pd.to_datetime(["2024-03-31", "2024-03-31"]),
        )
        repo.upsert_earnings_history(_IID, df)
        rows = _rows(repo)
        assert len(rows) == 1
        assert rows[0]["eps_estimate"] == 1.5  # last-wins

    def test_surprise_percent_uses_one_unit_across_history_and_dates(
        self, repo
    ) -> None:
        # Yahoo's earnings_history carries the surprise as a fraction (0.0674)
        # while the earnings-dates calendar carries percent (6.74) for the same
        # quarter; both land in a column named surprise_percent.
        history = pd.DataFrame(
            {"epsActual": [2.37], "surprisePercent": [0.0674]},
            index=pd.to_datetime(["2026-06-30"]),
        )
        repo.upsert_earnings_history(_IID, history)
        from_history = _rows(repo)[0]["surprise_percent"]

        dates = pd.DataFrame(
            {"Reported EPS": [2.37], "Surprise(%)": [6.74]},
            index=pd.to_datetime(["2026-06-30"]),
        )
        repo.upsert_earnings_dates(_IID, dates)
        from_dates = _rows(repo)[0]["surprise_percent"]

        assert from_history == pytest.approx(6.74)
        assert from_dates == pytest.approx(6.74)

    def test_earnings_history_missing_surprise_stays_none(self, repo) -> None:
        history = pd.DataFrame(
            {"epsActual": [2.37]}, index=pd.to_datetime(["2026-06-30"])
        )
        repo.upsert_earnings_history(_IID, history)
        assert _rows(repo)[0]["surprise_percent"] is None

    def test_earnings_dates_dedups_on_calendar_date(self, repo) -> None:
        # Distinct tz datetimes on the same calendar date collapse to one key.
        df = pd.DataFrame(
            {
                "EPS Estimate": [1.0, 1.2],
                "Reported EPS": [1.1, 1.3],
                "Surprise(%)": [10.0, 8.0],
            },
            index=pd.to_datetime(["2024-05-01 08:00", "2024-05-01 20:00"]),
        )
        repo.upsert_earnings_dates(_IID, df)
        rows = _rows(repo)
        assert len(rows) == 1

    def test_earnings_history_drops_undated_rows(self, repo) -> None:
        df = pd.DataFrame(
            {
                "epsEstimate": [1.0, 2.0],
                "epsActual": [1.1, 2.1],
                "epsDifference": [0.1, 0.1],
                "surprisePercent": [10.0, 5.0],
            },
            index=[pd.Timestamp("2024-03-31"), pd.NaT],
        )
        repo.upsert_earnings_history(_IID, df)
        rows = _rows(repo)
        assert len(rows) == 1
        assert rows[0]["period_date"] == dt.date(2024, 3, 31)

    def test_earnings_history_surprise_is_stored_in_percent_points(self, repo) -> None:
        # Yahoo's earnings_history reports surprisePercent as a fraction (0.1181
        # = 11.81%) while get_earnings_dates reports "Surprise(%)" already in
        # percent points. Both land in a column named surprise_percent, so the
        # fraction must be rescaled or the same event reads 100x apart.
        history = pd.DataFrame(
            {
                "epsEstimate": [3.0],
                "epsActual": [3.3543],
                "epsDifference": [0.3543],
                "surprisePercent": [0.1181],
            },
            index=pd.to_datetime(["2026-06-30"]),
        )
        repo.upsert_earnings_history(_IID, history)
        history_value = _rows(repo)[0]["surprise_percent"]

        dates = pd.DataFrame(
            {
                "EPS Estimate": [3.0],
                "Reported EPS": [3.3543],
                "Surprise(%)": [11.81],
            },
            index=pd.to_datetime(["2026-07-29"]),
        )
        repo.upsert_earnings_dates(_IID, dates)
        dates_value = _rows(repo)[0]["surprise_percent"]

        assert history_value == pytest.approx(11.81)
        assert history_value == pytest.approx(dates_value)

    def test_earnings_history_missing_surprise_stays_null(self, repo) -> None:
        df = pd.DataFrame(
            {
                "epsEstimate": [1.0],
                "epsActual": [1.1],
                "epsDifference": [0.1],
                "surprisePercent": [float("nan")],
            },
            index=pd.to_datetime(["2024-03-31"]),
        )
        repo.upsert_earnings_history(_IID, df)
        assert _rows(repo)[0]["surprise_percent"] is None

    def test_earnings_dates_defensive_columns(self, repo) -> None:
        df = pd.DataFrame(
            {
                "EPS Estimate": [1.0],
                "Reported EPS": [1.1],
                "Surprise(%)": [10.0],
            },
            index=pd.to_datetime(["2024-05-01"]),
        )
        repo.upsert_earnings_dates(_IID, df)
        r = _rows(repo)[0]
        assert r["earnings_date"] == dt.date(2024, 5, 1)
        assert r["eps_estimate"] == 1.0
        assert r["eps_actual"] == 1.1


class TestAnalystActions:
    def test_dedup_on_date_firm_tograde(self, repo) -> None:
        idx = pd.to_datetime(["2024-01-01", "2024-01-01"])
        df = pd.DataFrame(
            {
                "Firm": ["Goldman", "Goldman"],
                "ToGrade": ["Buy", "Buy"],
                "FromGrade": ["Hold", "Hold"],
                "Action": ["up", "up"],
            },
            index=idx,
        )
        repo.upsert_analyst_actions(_IID, df)
        rows = _rows(repo)
        assert len(rows) == 1
        assert rows[0]["firm"] == "Goldman"
        assert rows[0]["to_grade"] == "Buy"
        assert rows[0]["from_grade"] == "Hold"


class TestSecFilings:
    def test_dedup_and_key_fallbacks(self, repo) -> None:
        filings = [
            {"date": "2024-02-01", "type": "10-K", "title": "A", "edgarUrl": "u"},
            {"date": "2024-02-01", "type": "10-K", "title": "A", "edgarUrl": "u"},
            {"epochDate": "2024-05-01", "form": "10-Q", "description": "B"},
        ]
        repo.upsert_sec_filings(_IID, filings)
        rows = _rows(repo)
        assert len(rows) == 2
        forms = {r["form_type"] for r in rows}
        assert forms == {"10-K", "10-Q"}


class TestCorpActionExtras:
    def test_shares_outstanding_series_dedup(self, repo) -> None:
        s = pd.Series(
            [1000, 1050, 1050],
            index=pd.to_datetime(["2024-01-01", "2024-06-01", "2024-06-01"]),
        )
        repo.upsert_shares_outstanding(_IID, s)
        rows = _rows(repo)
        assert len(rows) == 2
        assert rows[0]["shares"] == 1000

    def test_shares_outstanding_accepts_dataframe(self, repo) -> None:
        df = pd.DataFrame({"x": [2000]}, index=pd.to_datetime(["2024-01-01"]))
        repo.upsert_shares_outstanding(_IID, df)
        rows = _rows(repo)
        assert len(rows) == 1 and rows[0]["shares"] == 2000


class TestHoldersExtras:
    def test_major_holders_key_lookup(self, repo) -> None:
        df = pd.DataFrame(
            {"Value": [0.01, 0.60, 0.61, 5000]},
            index=[
                "insidersPercentHeld",
                "institutionsPercentHeld",
                "institutionsFloatPercentHeld",
                "institutionsCount",
            ],
        )
        repo.upsert_major_holders(_IID, df)
        r = _rows(repo)[0]
        assert r["insiders_percent_held"] == 0.01
        assert r["institutions_percent_held"] == 0.60
        assert r["institutions_count"] == 5000

    def test_insider_purchases_label_lookup(self, repo) -> None:
        df = pd.DataFrame(
            {
                "Insider Purchases Last 6m": [
                    "Purchases",
                    "Sales",
                    "Net Shares Purchased (Sold)",
                    "Total Insider Shares Held",
                ],
                "Shares": [100, 40, 60, 5000],
                "Trans": [2, 1, 1, None],
            }
        )
        repo.upsert_insider_purchases(_IID, df)
        r = _rows(repo)[0]
        assert r["purchase_shares"] == 100
        assert r["sale_shares"] == 40
        assert r["net_shares"] == 60
        assert r["total_insider_shares"] == 5000

    def test_insider_roster_dedup_and_columns(self, repo) -> None:
        df = pd.DataFrame(
            {
                "Name": ["JANE DOE", "JANE DOE"],
                "Position": ["CEO", "CEO"],
                "Most Recent Transaction": ["Buy", "Buy"],
                "Latest Transaction Date": ["2024-05-01", "2024-05-01"],
                "Shares Owned Directly": [1000, 1000],
            }
        )
        repo.upsert_insider_roster(_IID, df)
        rows = _rows(repo)
        assert len(rows) == 1
        r = rows[0]
        assert r["insider_name"] == "JANE DOE"
        assert r["position"] == "CEO"
        assert r["latest_transaction_date"] == dt.date(2024, 5, 1)
        assert r["shares_owned_directly"] == 1000


class TestProfileExtras:
    def test_maps_special_info_keys(self, repo) -> None:
        info = {
            "sharesShort": 100_000,
            "shortRatio": 1.2,
            "52WeekChange": 0.15,
            "SandP52WeekChange": 0.10,
            "sectorKey": "technology",
            "industryKey": "semiconductors",
            "overallRisk": 3,
            "shareHolderRightsRisk": 4,
            "heldPercentInsiders": 0.01,
        }
        repo.upsert_profile_extras(_IID, info)
        r = _rows(repo)[0]
        assert r["shares_short"] == 100_000
        assert r["fifty_two_week_change"] == 0.15
        assert r["sandp_52_week_change"] == 0.10
        assert r["sector_key"] == "technology"
        assert r["overall_risk"] == 3
        assert r["shareholder_rights_risk"] == 4
        assert _constraint(repo) == "uq_ticker_profile_extras_instrument"


class TestOptionChain:
    def test_dedup_and_instrument_id_injection(self, repo) -> None:
        base = {
            "as_of": dt.date(2026, 8, 27),
            "expiry": dt.date(2026, 9, 18),
            "option_type": "call",
            "strike": 1.0,
            "contract_symbol": "X260918C00001000",
        }
        repo.upsert_option_chain(_IID, [dict(base), dict(base)])
        rows = _rows(repo)
        assert len(rows) == 1  # dedup on (as_of, contract_symbol)
        assert rows[0]["instrument_id"] == _IID
        assert _constraint(repo) == "uq_option_contract"

    def test_rows_without_symbol_or_as_of_skipped(self, repo) -> None:
        repo.upsert_option_chain(
            _IID,
            [
                {"as_of": None, "contract_symbol": "A"},
                {"as_of": dt.date(2026, 8, 27), "contract_symbol": None},
            ],
        )
        # nothing valid → _upsert not called (returns 0 early)
        assert repo._upsert.call_count == 0


class TestUpsertNewsKeepsScrapedBody:
    """News is re-fetched every run and the same articles reappear for days, so
    a failed re-scrape (no ``full_content``) must not erase a stored body."""

    @staticmethod
    def _conflict_updates(articles: list[dict]) -> list[str]:
        """SQL of every INSERT ... ON CONFLICT issued for ``articles`` on PG."""
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.sql.dml import Insert

        executed: list = []
        session = MagicMock(name="session")
        session.execute.side_effect = lambda stmt: executed.append(stmt) or MagicMock()
        YFinanceRepository(session).upsert_news(_IID, articles)
        return [
            str(s.compile(dialect=postgresql.dialect()))
            for s in executed
            if isinstance(s, Insert)
        ]

    def test_article_without_body_leaves_stored_full_content_untouched(self) -> None:
        (sql,) = self._conflict_updates([{"uuid": "a1", "title": "T"}])
        assert "title = excluded.title" in sql
        assert "full_content = excluded.full_content" not in sql

    def test_article_with_body_replaces_stored_full_content(self) -> None:
        (sql,) = self._conflict_updates(
            [{"uuid": "a1", "title": "T", "full_content": "Body"}]
        )
        assert "full_content = excluded.full_content" in sql

    def test_mixed_batch_only_protects_the_articles_without_a_body(self) -> None:
        with_body, without_body = self._conflict_updates(
            [
                {"uuid": "a1", "title": "T", "full_content": "Body"},
                {"uuid": "a2", "title": "U"},
            ]
        )
        assert "full_content = excluded.full_content" in with_body
        assert "full_content = excluded.full_content" not in without_body

    def test_returns_total_rows_written_across_both_groups(self) -> None:
        session = MagicMock(name="session")
        n = YFinanceRepository(session).upsert_news(
            _IID,
            [
                {"uuid": "a1", "full_content": "Body"},
                {"uuid": "a2"},
                {"uuid": "a3"},
            ],
        )
        assert n == 3


class TestInBatchDedup:
    """One INSERT ... ON CONFLICT cannot hit the same key twice, so every
    batch must be collapsed on its conflict key before it reaches ``_upsert``."""

    def test_price_history_same_date_keeps_last(self, repo) -> None:
        df = pd.DataFrame(
            {"Close": [1.0, 2.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-02"]),
        )
        repo.upsert_price_history(_IID, df)
        rows = _rows(repo)
        assert len(rows) == 1
        assert rows[0]["close"] == 2.0

    def test_dividends_same_date_keeps_last(self, repo) -> None:
        s = pd.Series(
            [0.2, 0.3], index=pd.to_datetime(["2024-02-01 09:00", "2024-02-01 16:00"])
        )
        repo.upsert_dividends(_IID, s)
        rows = _rows(repo)
        assert [r["amount"] for r in rows] == [0.3]

    def test_splits_same_date_keeps_last(self, repo) -> None:
        s = pd.Series(
            [2.0, 3.0], index=pd.to_datetime(["2024-02-01 09:00", "2024-02-01 16:00"])
        )
        repo.upsert_splits(_IID, s)
        assert [r["ratio"] for r in _rows(repo)] == [3.0]

    def test_recommendations_same_period_keeps_last(self, repo) -> None:
        df = pd.DataFrame(
            {
                "period": ["0m", "0m"],
                "strongBuy": [1, 5],
                "buy": [1, 1],
                "hold": [1, 1],
                "sell": [1, 1],
                "strongSell": [1, 1],
            }
        )
        repo.upsert_recommendations(_IID, df)
        rows = _rows(repo)
        assert len(rows) == 1
        assert rows[0]["strong_buy"] == 5

    def test_financial_statements_duplicate_line_items_keep_last(self, repo) -> None:
        df = pd.DataFrame(
            [[1.0, 10.0], [2.0, 20.0]],
            index=["TotalRevenue", "TotalRevenue"],
            columns=pd.to_datetime(["2024-12-31", "2023-12-31"]),
        )
        repo.upsert_financial_statements(_IID, df, "income_statement", "annual")
        rows = _rows(repo)
        assert len(rows) == 2
        by_date = {r["period_date"]: r["value"] for r in rows}
        assert by_date == {dt.date(2024, 12, 31): 2.0, dt.date(2023, 12, 31): 20.0}

    @pytest.mark.parametrize(
        "method", ["upsert_institutional_holders", "upsert_mutualfund_holders"]
    )
    def test_holders_same_name_keeps_latest_reported(self, repo, method) -> None:
        df = pd.DataFrame(
            {
                "Holder": ["Vanguard", "Vanguard"],
                "Date Reported": pd.to_datetime(["2024-06-30", "2024-03-31"]),
                "Shares": [200, 100],
                "Value": [2000, 1000],
                "pctHeld": [0.02, 0.01],
            }
        )
        getattr(repo, method)(_IID, df)
        rows = _rows(repo)
        assert len(rows) == 1
        assert rows[0]["shares"] == 200
        assert rows[0]["date_reported"] == dt.date(2024, 6, 30)

    def test_holders_names_equal_after_truncation_collapse(self, repo) -> None:
        long_a = "X" * 500 + "a"
        long_b = "X" * 500 + "b"
        df = pd.DataFrame(
            {
                "Holder": [long_a, long_b],
                "Date Reported": pd.to_datetime(["2024-06-30", "2024-06-30"]),
                "Shares": [1, 2],
            }
        )
        repo.upsert_institutional_holders(_IID, df)
        assert len(_rows(repo)) == 1

    def test_news_same_uuid_keeps_last(self, repo) -> None:
        articles = [
            {"id": "n1", "content": {"title": "first"}},
            {"id": "n1", "content": {"title": "second"}},
            {"id": "n2", "content": {"title": "other"}},
        ]
        repo.upsert_news(_IID, articles)
        rows = _rows(repo)
        assert [(r["news_uuid"], r["title"]) for r in rows] == [
            ("n1", "second"),
            ("n2", "other"),
        ]

    @pytest.mark.parametrize(
        ("method", "frame"),
        [
            ("upsert_earnings_estimate", {"avg": [1.0, 2.0]}),
            ("upsert_revenue_estimate", {"avg": [1.0, 2.0]}),
            ("upsert_growth_estimates", {"stockTrend": [0.1, 0.2]}),
            ("upsert_eps_trend", {"current": [1.0, 2.0]}),
            ("upsert_eps_revisions", {"upLast7days": [1, 2]}),
        ],
    )
    def test_period_indexed_estimates_keep_one_row_per_period(
        self, repo, method, frame
    ) -> None:
        df = pd.DataFrame(frame, index=["0q", "0q"])
        getattr(repo, method)(_IID, df)
        assert [r["period"] for r in _rows(repo)] == ["0q"]


class TestValuationMeasures:
    """Real yfinance 1.6.0 ``Ticker.valuation`` shape: measure labels as the
    index, a ``Current`` column plus ``M/D/YYYY`` period-end string columns."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Current": [3.0e12, 31.5],
                "9/30/2025": [2.9e12, 30.1],
                "6/30/2025": [2.8e12, 29.0],
            },
            index=["Market Cap", "Trailing P/E"],
        )

    def test_period_end_labels_are_parsed(self, repo) -> None:
        repo.upsert_financial_statements(
            _IID, self._frame(), "valuation_measures", "point_in_time"
        )
        rows = _rows(repo)
        by_key = {(r["period_date"], r["line_item"]): r["value"] for r in rows}
        assert by_key[(dt.date(2025, 9, 30), "Market Cap")] == 2.9e12
        assert by_key[(dt.date(2025, 6, 30), "Trailing P/E")] == 29.0

    def test_current_column_is_stamped_with_the_as_of_date(self, repo) -> None:
        as_of = dt.date(2026, 10, 23)
        repo.upsert_financial_statements(
            _IID,
            self._frame(),
            "valuation_measures",
            "point_in_time",
            as_of=as_of,
        )
        by_key = {(r["period_date"], r["line_item"]): r["value"] for r in _rows(repo)}
        assert by_key[(as_of, "Market Cap")] == 3.0e12
        assert by_key[(as_of, "Trailing P/E")] == 31.5
        assert len(by_key) == 6

    def test_current_column_is_skipped_without_an_as_of_date(self, repo) -> None:
        repo.upsert_financial_statements(
            _IID, self._frame(), "valuation_measures", "point_in_time"
        )
        assert len(_rows(repo)) == 4

    def test_warns_when_no_column_is_a_period_date(self, repo, caplog) -> None:
        df = pd.DataFrame({"Current": [1.0], "TTM": [2.0]}, index=["Market Cap"])
        with caplog.at_level("WARNING"):
            repo.upsert_financial_statements(
                _IID, df, "valuation_measures", "point_in_time"
            )
        assert "no column" in caplog.text.lower()
