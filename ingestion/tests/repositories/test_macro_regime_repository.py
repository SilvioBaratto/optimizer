"""Tests for MacroRegimeRepository.upsert_regime_classification (issue #530)."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.models.macro.macro_regime import MacroCalibration, MacroNews, MacroNewsTheme
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

    def test_when_called_repeatedly_then_exactly_one_row_persists(
        self, db_session: Session
    ) -> None:
        """R1/§5.4 (T1.2): the write must converge to a single row on re-run
        via INSERT ... ON CONFLICT DO UPDATE on uq_macro_calibration_country,
        never a duplicate — the at-least-once reclaim contract.
        """
        repo = MacroRegimeRepository(db_session)
        for regime in ("expansion", "slowdown", "recession"):
            repo.upsert_regime_classification(country="US", regime=regime)
            db_session.flush()

        rows = (
            db_session.query(MacroCalibration)
            .filter(MacroCalibration.country == "US")
            .all()
        )
        assert len(rows) == 1
        assert rows[0].regime_classification == "recession"


def _news_row(news_id: str, themes: str) -> dict:
    return {"id": uuid.uuid4(), "news_id": news_id, "title": "t", "themes": themes}


def _themes_of(session: Session, news_id: str) -> set[str]:
    # Query the child table directly rather than the parent's relationship
    # collection: upsert_macro_news adds children via session.add (not
    # theme_entries.append), so the in-session collection stays stale while the
    # DB rows are correct — a fresh reader (as in production) sees them.
    parent = session.query(MacroNews).filter(MacroNews.news_id == news_id).one()
    themes = (
        session.query(MacroNewsTheme).filter(MacroNewsTheme.news_id == parent.id).all()
    )
    return {t.theme for t in themes}


class TestUpsertMacroNewsThemes:
    """T1.4 / §5.4: theme children converge on re-run and on a changed set."""

    def test_when_rerun_with_same_themes_then_no_duplicate_children(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news([_news_row("n1", "rates,inflation")])
        db_session.flush()
        repo.upsert_macro_news([_news_row("n1", "rates,inflation")])
        db_session.flush()

        parents = db_session.query(MacroNews).filter(MacroNews.news_id == "n1").all()
        assert len(parents) == 1
        assert _themes_of(db_session, "n1") == {"rates", "inflation"}

    def test_when_theme_set_changes_then_stale_themes_are_removed(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news([_news_row("n2", "rates,inflation")])
        db_session.flush()
        repo.upsert_macro_news([_news_row("n2", "rates,growth")])
        db_session.flush()

        # inflation dropped, growth added, rates kept — no stale child left behind
        assert _themes_of(db_session, "n2") == {"rates", "growth"}
