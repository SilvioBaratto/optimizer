"""Tests for MacroRegimeRepository.upsert_regime_classification (issue #530)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.macro.macro_regime import MacroCalibration
from app.repositories.macro.macro_regime_repository import MacroRegimeRepository


def _row(session: Session, country: str) -> MacroCalibration | None:
    return (
        session.query(MacroCalibration)
        .filter(MacroCalibration.country == country)
        .one_or_none()
    )


class TestUpsertRegimeClassification:
    """Persist rule-based MacroRegime classifier output to macro_calibrations."""

    def test_when_country_row_missing_inserts_with_regime_only(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_regime_classification(country="US", regime="expansion")
        db_session.flush()

        row = _row(db_session, "US")
        assert row is not None
        assert row.regime_classification == "expansion"

    def test_when_country_row_exists_updates_regime_and_preserves_baml_fields(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        existing = MacroCalibration(
            country="US",
            phase="MID_EXPANSION",
            delta=2.5,
            tau=0.01,
            confidence=0.8,
            rationale="initial",
            macro_summary="summary",
        )
        db_session.add(existing)
        db_session.flush()

        repo.upsert_regime_classification(country="US", regime="recession")
        db_session.flush()
        db_session.refresh(existing)

        assert existing.regime_classification == "recession"
        assert existing.phase == "MID_EXPANSION"
        assert existing.delta == 2.5
        assert existing.tau == 0.01
        assert existing.confidence == 0.8
        assert existing.rationale == "initial"
        assert existing.macro_summary == "summary"

    def test_when_called_twice_overwrites_with_latest_regime(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_regime_classification(country="US", regime="recovery")
        db_session.flush()
        repo.upsert_regime_classification(country="US", regime="slowdown")
        db_session.flush()

        row = _row(db_session, "US")
        assert row is not None
        assert row.regime_classification == "slowdown"

    def test_when_two_countries_each_get_own_row(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_regime_classification(country="US", regime="expansion")
        repo.upsert_regime_classification(country="EU", regime="recession")
        db_session.flush()

        us = _row(db_session, "US")
        eu = _row(db_session, "EU")
        assert us is not None and us.regime_classification == "expansion"
        assert eu is not None and eu.regime_classification == "recession"
