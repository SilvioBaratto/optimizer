"""Unit tests for FactorRepository.

Tests use the shared ``db_session`` SQLite fixture from conftest.py.
``bulk_upsert_scores`` mocks ``_upsert`` because pg_insert is not
compatible with SQLite.
"""

from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from app.models.factor import FactorScore, FactorValidationReport
from app.repositories.factor_repository import FactorRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_score(
    ticker: str = "AAPL",
    score_date: datetime.date = datetime.date(2024, 1, 15),
    factor_type: str = "momentum",
    factor_group: str = "technical",
    raw_score: float = 1.0,
) -> FactorScore:
    return FactorScore(
        ticker=ticker,
        score_date=score_date,
        factor_type=factor_type,
        factor_group=factor_group,
        raw_score=raw_score,
    )


def _make_report(
    report_date: datetime.date = datetime.date(2024, 1, 15),
    factor_type: str | None = "momentum",
    validation_type: str = "ic",
    ic_mean: float = 0.05,
) -> FactorValidationReport:
    return FactorValidationReport(
        report_date=report_date,
        factor_type=factor_type,
        validation_type=validation_type,
        ic_mean=ic_mean,
    )


# ---------------------------------------------------------------------------
# get_scores_by_ticker
# ---------------------------------------------------------------------------


class TestGetScoresByTicker:
    def test_returns_empty_list_when_no_data(self, db_session: Session) -> None:
        repo = FactorRepository(db_session)

        result = repo.get_scores_by_ticker("AAPL")

        assert result == []

    def test_filters_by_ticker(self, db_session: Session) -> None:
        aapl = _make_score(ticker="AAPL")
        msft = _make_score(ticker="MSFT", factor_type="value")
        db_session.add_all([aapl, msft])
        db_session.flush()
        repo = FactorRepository(db_session)

        result = repo.get_scores_by_ticker("AAPL")

        assert len(result) == 1
        assert result[0].ticker == "AAPL"

    def test_filters_by_start_date(self, db_session: Session) -> None:
        early = _make_score(score_date=datetime.date(2024, 1, 1))
        late = _make_score(score_date=datetime.date(2024, 6, 1), factor_type="value")
        db_session.add_all([early, late])
        db_session.flush()
        repo = FactorRepository(db_session)

        result = repo.get_scores_by_ticker("AAPL", start_date=datetime.date(2024, 3, 1))

        assert len(result) == 1
        assert result[0].score_date == datetime.date(2024, 6, 1)

    def test_filters_by_end_date(self, db_session: Session) -> None:
        early = _make_score(score_date=datetime.date(2024, 1, 1))
        late = _make_score(score_date=datetime.date(2024, 6, 1), factor_type="value")
        db_session.add_all([early, late])
        db_session.flush()
        repo = FactorRepository(db_session)

        result = repo.get_scores_by_ticker("AAPL", end_date=datetime.date(2024, 3, 1))

        assert len(result) == 1
        assert result[0].score_date == datetime.date(2024, 1, 1)

    def test_filters_by_start_and_end_date(self, db_session: Session) -> None:
        dates = [
            datetime.date(2024, 1, 1),
            datetime.date(2024, 4, 1),
            datetime.date(2024, 9, 1),
        ]
        factor_types = ["momentum", "value", "quality"]
        for d, ft in zip(dates, factor_types):
            db_session.add(_make_score(score_date=d, factor_type=ft))
        db_session.flush()
        repo = FactorRepository(db_session)

        result = repo.get_scores_by_ticker(
            "AAPL",
            start_date=datetime.date(2024, 2, 1),
            end_date=datetime.date(2024, 8, 1),
        )

        assert len(result) == 1
        assert result[0].score_date == datetime.date(2024, 4, 1)


# ---------------------------------------------------------------------------
# get_scores_by_date
# ---------------------------------------------------------------------------


class TestGetScoresByDate:
    def test_returns_empty_list_when_no_data(self, db_session: Session) -> None:
        repo = FactorRepository(db_session)

        result = repo.get_scores_by_date(datetime.date(2024, 1, 15))

        assert result == []

    def test_returns_all_scores_for_date(self, db_session: Session) -> None:
        target = datetime.date(2024, 1, 15)
        other = datetime.date(2024, 2, 1)
        db_session.add(_make_score(ticker="AAPL", score_date=target))
        db_session.add(_make_score(ticker="MSFT", score_date=target, factor_type="value"))
        db_session.add(_make_score(ticker="GOOG", score_date=other, factor_type="quality"))
        db_session.flush()
        repo = FactorRepository(db_session)

        result = repo.get_scores_by_date(target)

        assert len(result) == 2
        tickers = {r.ticker for r in result}
        assert tickers == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# bulk_upsert_scores
# ---------------------------------------------------------------------------


class TestBulkUpsertScores:
    def test_mocks_upsert_and_verifies_call_args(self, db_session: Session) -> None:
        repo = FactorRepository(db_session)
        scores = [
            {
                "ticker": "AAPL",
                "score_date": datetime.date(2024, 1, 15),
                "factor_type": "momentum",
                "factor_group": "technical",
                "raw_score": 1.5,
            }
        ]

        with patch.object(repo, "_upsert", return_value=1) as mock_upsert:
            result = repo.bulk_upsert_scores(scores)

        mock_upsert.assert_called_once_with(
            FactorScore,
            scores,
            constraint_name="uq_factor_scores_ticker_date_type",
        )
        assert result == 1

    def test_returns_zero_for_empty_list(self, db_session: Session) -> None:
        repo = FactorRepository(db_session)

        with patch.object(repo, "_upsert", return_value=0) as mock_upsert:
            result = repo.bulk_upsert_scores([])

        mock_upsert.assert_called_once_with(
            FactorScore,
            [],
            constraint_name="uq_factor_scores_ticker_date_type",
        )
        assert result == 0


# ---------------------------------------------------------------------------
# get_validation_reports
# ---------------------------------------------------------------------------


class TestGetValidationReports:
    def test_returns_empty_list_when_no_data(self, db_session: Session) -> None:
        repo = FactorRepository(db_session)

        result = repo.get_validation_reports()

        assert result == []

    def test_filters_by_factor_type(self, db_session: Session) -> None:
        db_session.add(_make_report(factor_type="momentum", validation_type="ic"))
        db_session.add(_make_report(factor_type="value", validation_type="oos"))
        db_session.flush()
        repo = FactorRepository(db_session)

        result = repo.get_validation_reports(factor_type="momentum")

        assert len(result) == 1
        assert result[0].factor_type == "momentum"

    def test_filters_by_validation_type(self, db_session: Session) -> None:
        db_session.add(_make_report(factor_type="momentum", validation_type="ic"))
        db_session.add(_make_report(factor_type="value", validation_type="oos"))
        db_session.flush()
        repo = FactorRepository(db_session)

        result = repo.get_validation_reports(validation_type="oos")

        assert len(result) == 1
        assert result[0].validation_type == "oos"

    def test_filters_by_both_factor_type_and_validation_type(
        self, db_session: Session
    ) -> None:
        db_session.add(_make_report(factor_type="momentum", validation_type="ic"))
        db_session.add(_make_report(factor_type="momentum", validation_type="oos"))
        db_session.add(_make_report(factor_type="value", validation_type="ic"))
        db_session.flush()
        repo = FactorRepository(db_session)

        result = repo.get_validation_reports(factor_type="momentum", validation_type="ic")

        assert len(result) == 1
        assert result[0].factor_type == "momentum"
        assert result[0].validation_type == "ic"


# ---------------------------------------------------------------------------
# save_validation_report
# ---------------------------------------------------------------------------


class TestSaveValidationReport:
    def test_creates_and_returns_report(self, db_session: Session) -> None:
        repo = FactorRepository(db_session)
        data = {
            "report_date": datetime.date(2024, 1, 15),
            "factor_type": "momentum",
            "validation_type": "ic",
            "ic_mean": 0.07,
            "ic_std": 0.03,
            "icir": 2.3,
            "t_stat": 3.1,
            "p_value": 0.002,
        }

        report = repo.save_validation_report(data)

        assert report.id is not None
        assert report.factor_type == "momentum"
        assert report.validation_type == "ic"

    def test_returns_report_with_correct_fields(self, db_session: Session) -> None:
        repo = FactorRepository(db_session)
        data = {
            "report_date": datetime.date(2024, 3, 1),
            "factor_type": "value",
            "validation_type": "oos",
            "ic_mean": 0.04,
            "ic_std": 0.02,
            "icir": 2.0,
            "t_stat": 2.8,
            "p_value": 0.005,
            "vif": 1.2,
        }

        report = repo.save_validation_report(data)

        assert report.report_date == datetime.date(2024, 3, 1)
        assert report.ic_mean == pytest.approx(0.04)
        assert report.vif == pytest.approx(1.2)
