"""Tests for MacroRegimeRepository macro-news theme upserts."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from portopt_db.models.macro.macro_regime import (
    MacroNews,
    MacroNewsTheme,
)
from portopt_db.repositories.macro.macro_regime_repository import MacroRegimeRepository


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
