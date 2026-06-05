"""DB seed builder for the factors domain.

Seeds one ``FactorScore`` and one (aggregate) ``FactorValidationReport``.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

from sqlalchemy.orm import Session

from app.models.factors.factor import FactorScore, FactorValidationReport
from tests._fixtures._helpers import add_and_flush

_SEED_DATE = date(2024, 1, 1)


class FactorsSeed(NamedTuple):
    score: FactorScore
    report: FactorValidationReport


def _make_score(ticker: str, factor_type: str) -> FactorScore:
    return FactorScore(
        ticker=ticker,
        score_date=_SEED_DATE,
        factor_type=factor_type,
        factor_group="momentum",
        raw_score=1.25,
        standardized_score=0.8,
        composite_score=0.5,
    )


def _make_report(factor_type: str) -> FactorValidationReport:
    return FactorValidationReport(
        report_date=_SEED_DATE,
        factor_type=factor_type,
        validation_type="is",
        ic_mean=0.03,
        icir=0.6,
        details={"n_periods": 36},
    )


def seed_factors(
    session: Session,
    *,
    ticker: str = "AAPL",
    factor_type: str = "momentum_12_1",
) -> FactorsSeed:
    score = add_and_flush(session, _make_score(ticker, factor_type))
    report = add_and_flush(session, _make_report(factor_type))
    return FactorsSeed(score=score, report=report)
