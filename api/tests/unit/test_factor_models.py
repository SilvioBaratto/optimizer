"""Tests for FactorScore and FactorValidationReport SQLAlchemy models.

Covers:
- Table creation via Base.metadata.create_all()
- Successful row insertion with all required fields
- Unique constraint enforcement on factor_scores (ticker, score_date, factor_type)
- Unique constraint enforcement on factor_validation_reports
  (report_date, factor_type, validation_type)
- Nullable fields on both models
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models._shared import Base
from app.models.factors.factor import FactorScore, FactorValidationReport


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture
def session(engine) -> Session:
    connection = engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    sess = SessionLocal()
    sess.begin_nested()
    yield sess
    sess.close()
    transaction.rollback()
    connection.close()


class TestTableCreation:
    """Base.metadata.create_all() must succeed for both tables."""

    def test_factor_scores_table_exists(self, engine) -> None:
        table_names = engine.dialect.get_table_names(engine.connect())
        assert "factor_scores" in table_names

    def test_factor_validation_reports_table_exists(self, engine) -> None:
        table_names = engine.dialect.get_table_names(engine.connect())
        assert "factor_validation_reports" in table_names


class TestFactorScoreInsertion:
    """FactorScore rows can be inserted with required and optional fields."""

    def test_inserts_row_with_required_fields(self, session: Session) -> None:
        score = FactorScore(
            ticker="AAPL",
            score_date=datetime.date(2024, 1, 15),
            factor_type="momentum",
            factor_group="momentum",
            raw_score=1.23,
        )
        session.add(score)
        session.flush()

        result = session.query(FactorScore).filter_by(ticker="AAPL").first()
        assert result is not None
        assert result.ticker == "AAPL"
        assert result.raw_score == 1.23

    def test_inserts_row_with_nullable_standardized_and_composite_scores(
        self, session: Session
    ) -> None:
        score = FactorScore(
            ticker="MSFT",
            score_date=datetime.date(2024, 1, 15),
            factor_type="quality",
            factor_group="quality",
            raw_score=0.75,
            standardized_score=0.82,
            composite_score=0.79,
        )
        session.add(score)
        session.flush()

        result = session.query(FactorScore).filter_by(ticker="MSFT").first()
        assert result is not None
        assert result.standardized_score == pytest.approx(0.82)
        assert result.composite_score == pytest.approx(0.79)

    def test_standardized_and_composite_score_nullable(self, session: Session) -> None:
        score = FactorScore(
            ticker="GOOGL",
            score_date=datetime.date(2024, 1, 15),
            factor_type="value",
            factor_group="value",
            raw_score=-0.5,
        )
        session.add(score)
        session.flush()

        result = session.query(FactorScore).filter_by(ticker="GOOGL").first()
        assert result is not None
        assert result.standardized_score is None
        assert result.composite_score is None


class TestFactorScoreUniqueConstraint:
    """Duplicate (ticker, score_date, factor_type) must raise IntegrityError."""

    def test_raises_integrity_error_on_duplicate_ticker_date_factor(
        self, session: Session
    ) -> None:
        score_a = FactorScore(
            ticker="TSLA",
            score_date=datetime.date(2024, 2, 1),
            factor_type="momentum",
            factor_group="momentum",
            raw_score=1.0,
        )
        score_b = FactorScore(
            ticker="TSLA",
            score_date=datetime.date(2024, 2, 1),
            factor_type="momentum",
            factor_group="momentum",
            raw_score=2.0,
        )
        session.add(score_a)
        session.flush()

        session.add(score_b)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_allows_same_ticker_different_factor_type(self, session: Session) -> None:
        score_a = FactorScore(
            ticker="NVDA",
            score_date=datetime.date(2024, 2, 1),
            factor_type="momentum",
            factor_group="momentum",
            raw_score=1.0,
        )
        score_b = FactorScore(
            ticker="NVDA",
            score_date=datetime.date(2024, 2, 1),
            factor_type="quality",
            factor_group="quality",
            raw_score=0.5,
        )
        session.add_all([score_a, score_b])
        session.flush()  # must not raise

        count = session.query(FactorScore).filter_by(ticker="NVDA").count()
        assert count == 2

    def test_allows_same_factor_type_different_date(self, session: Session) -> None:
        score_a = FactorScore(
            ticker="META",
            score_date=datetime.date(2024, 1, 1),
            factor_type="value",
            factor_group="value",
            raw_score=0.3,
        )
        score_b = FactorScore(
            ticker="META",
            score_date=datetime.date(2024, 2, 1),
            factor_type="value",
            factor_group="value",
            raw_score=0.4,
        )
        session.add_all([score_a, score_b])
        session.flush()  # must not raise

        count = session.query(FactorScore).filter_by(ticker="META").count()
        assert count == 2


class TestFactorValidationReportInsertion:
    """FactorValidationReport rows can be inserted with required and optional fields."""

    def test_inserts_row_with_required_fields(self, session: Session) -> None:
        report = FactorValidationReport(
            report_date=datetime.date(2024, 3, 1),
            validation_type="in_sample",
        )
        session.add(report)
        session.flush()

        result = (
            session.query(FactorValidationReport)
            .filter_by(validation_type="in_sample")
            .first()
        )
        assert result is not None
        assert result.report_date == datetime.date(2024, 3, 1)

    def test_inserts_aggregate_report_with_null_factor_type(
        self, session: Session
    ) -> None:
        report = FactorValidationReport(
            report_date=datetime.date(2024, 3, 2),
            factor_type=None,
            validation_type="out_of_sample",
            ic_mean=0.05,
            ic_std=0.02,
            icir=2.5,
        )
        session.add(report)
        session.flush()

        result = (
            session.query(FactorValidationReport)
            .filter_by(report_date=datetime.date(2024, 3, 2))
            .first()
        )
        assert result is not None
        assert result.factor_type is None
        assert result.icir == pytest.approx(2.5)

    def test_inserts_row_with_all_optional_fields(self, session: Session) -> None:
        report = FactorValidationReport(
            report_date=datetime.date(2024, 3, 3),
            factor_type="momentum",
            validation_type="in_sample",
            ic_mean=0.04,
            ic_std=0.01,
            icir=4.0,
            t_stat=3.5,
            p_value=0.0005,
            vif=1.2,
            details={"factors": ["momentum"], "observations": 252},
        )
        session.add(report)
        session.flush()

        result = (
            session.query(FactorValidationReport)
            .filter_by(factor_type="momentum", report_date=datetime.date(2024, 3, 3))
            .first()
        )
        assert result is not None
        assert result.details == {"factors": ["momentum"], "observations": 252}
        assert result.p_value == pytest.approx(0.0005)

    def test_all_metric_fields_are_nullable(self, session: Session) -> None:
        report = FactorValidationReport(
            report_date=datetime.date(2024, 3, 4),
            factor_type="quality",
            validation_type="out_of_sample",
        )
        session.add(report)
        session.flush()

        result = (
            session.query(FactorValidationReport)
            .filter_by(factor_type="quality", report_date=datetime.date(2024, 3, 4))
            .first()
        )
        assert result is not None
        assert result.ic_mean is None
        assert result.ic_std is None
        assert result.icir is None
        assert result.t_stat is None
        assert result.p_value is None
        assert result.vif is None
        assert result.details is None


class TestFactorValidationReportUniqueConstraint:
    """Duplicate (report_date, factor_type, validation_type) must raise IntegrityError."""

    def test_raises_integrity_error_on_duplicate_report_date_factor_validation(
        self, session: Session
    ) -> None:
        report_a = FactorValidationReport(
            report_date=datetime.date(2024, 4, 1),
            factor_type="value",
            validation_type="in_sample",
            ic_mean=0.03,
        )
        report_b = FactorValidationReport(
            report_date=datetime.date(2024, 4, 1),
            factor_type="value",
            validation_type="in_sample",
            ic_mean=0.04,
        )
        session.add(report_a)
        session.flush()

        session.add(report_b)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_allows_same_date_and_factor_different_validation_type(
        self, session: Session
    ) -> None:
        report_a = FactorValidationReport(
            report_date=datetime.date(2024, 4, 2),
            factor_type="growth",
            validation_type="in_sample",
        )
        report_b = FactorValidationReport(
            report_date=datetime.date(2024, 4, 2),
            factor_type="growth",
            validation_type="out_of_sample",
        )
        session.add_all([report_a, report_b])
        session.flush()  # must not raise

        count = (
            session.query(FactorValidationReport)
            .filter_by(factor_type="growth", report_date=datetime.date(2024, 4, 2))
            .count()
        )
        assert count == 2

    def test_allows_same_factor_and_validation_different_date(
        self, session: Session
    ) -> None:
        report_a = FactorValidationReport(
            report_date=datetime.date(2024, 4, 3),
            factor_type="quality",
            validation_type="in_sample",
        )
        report_b = FactorValidationReport(
            report_date=datetime.date(2024, 5, 3),
            factor_type="quality",
            validation_type="in_sample",
        )
        session.add_all([report_a, report_b])
        session.flush()  # must not raise

        count = (
            session.query(FactorValidationReport)
            .filter_by(factor_type="quality")
            .count()
        )
        assert count == 2
