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
        # Short spellings (stock/industry/...) rather than *Trend must still map.
        df = pd.DataFrame(
            {"stock": [0.1], "industry": [0.2], "sector": [0.3], "index": [0.4]},
            index=["0q"],
        )
        repo.upsert_growth_estimates(_IID, df)
        r = _rows(repo)[0]
        assert (r["stock_trend"], r["industry_trend"]) == (0.1, 0.2)
        assert (r["sector_trend"], r["index_trend"]) == (0.3, 0.4)


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
        assert len(rows) == 1  # the NaT row is dropped
        assert rows[0]["period_date"] == dt.date(2024, 3, 31)

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


class TestEsgScores:
    def test_metric_lookup_from_single_column_frame(self, repo) -> None:
        df = pd.DataFrame(
            {"esgScores": [22.5, 1.0, 2.0, 3.0, 4.0]},
            index=[
                "totalEsg",
                "environmentScore",
                "socialScore",
                "governanceScore",
                "highestControversy",
            ],
        )
        repo.upsert_esg_scores(_IID, df)
        r = _rows(repo)[0]
        assert r["total_esg"] == 22.5
        assert r["highest_controversy"] == 4.0


class TestSecFilings:
    def test_dedup_and_key_fallbacks(self, repo) -> None:
        filings = [
            {"date": "2024-02-01", "type": "10-K", "title": "A", "edgarUrl": "u"},
            {"date": "2024-02-01", "type": "10-K", "title": "A", "edgarUrl": "u"},
            {"epochDate": "2024-05-01", "form": "10-Q", "description": "B"},
        ]
        repo.upsert_sec_filings(_IID, filings)
        rows = _rows(repo)
        assert len(rows) == 2  # first two dedup
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
        assert len(rows) == 2  # duplicate 2024-06-01 collapsed
        assert rows[0]["shares"] == 1000

    def test_shares_outstanding_accepts_dataframe(self, repo) -> None:
        df = pd.DataFrame({"x": [2000]}, index=pd.to_datetime(["2024-01-01"]))
        repo.upsert_shares_outstanding(_IID, df)
        rows = _rows(repo)
        assert len(rows) == 1 and rows[0]["shares"] == 2000

    def test_capital_gains_series(self, repo) -> None:
        s = pd.Series([0.5], index=pd.to_datetime(["2023-12-15"]))
        repo.upsert_capital_gains(_IID, s)
        r = _rows(repo)[0]
        assert r["amount"] == 0.5
        assert r["date"] == dt.date(2023, 12, 15)


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
