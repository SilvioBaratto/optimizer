"""Tests for MacroRegimeRepository macro-news theme and bond-yield upserts."""

from __future__ import annotations

import datetime
import uuid

import pytest
from sqlalchemy.orm import Session

from portopt_db import coerce
from portopt_db.models.macro.macro_regime import (
    BondYield,
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
    """Verify theme children converge on re-run and on a changed set."""

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


def _body_row(news_id: str, snippet: str, full_content: str | None) -> dict:
    return {
        "id": uuid.uuid4(),
        "news_id": news_id,
        "title": "t",
        "snippet": snippet,
        "full_content": full_content,
    }


def _stored(session: Session, news_id: str) -> MacroNews:
    session.expire_all()
    return session.query(MacroNews).filter(MacroNews.news_id == news_id).one()


class TestUpsertMacroNewsFullContent:
    """A snippet-only re-fetch must not erase a body scraped earlier."""

    def test_when_snippet_only_rerun_then_scraped_body_is_preserved(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news([_body_row("b1", "old snippet", "scraped body")])
        db_session.flush()
        repo.upsert_macro_news([_body_row("b1", "new snippet", None)])
        db_session.flush()

        stored = _stored(db_session, "b1")
        assert stored.full_content == "scraped body"
        assert stored.snippet == "new snippet"

    def test_when_rerun_carries_a_body_then_body_is_replaced(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news([_body_row("b2", "s", "old body")])
        db_session.flush()
        repo.upsert_macro_news([_body_row("b2", "s", "fresh body")])
        db_session.flush()

        assert _stored(db_session, "b2").full_content == "fresh body"

    def test_when_new_article_is_snippet_only_then_body_is_null(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        count = repo.upsert_macro_news([_body_row("b3", "s", None)])
        db_session.flush()

        assert count == 1
        assert _stored(db_session, "b3").full_content is None

    def test_when_batch_mixes_bodies_and_snippets_then_each_row_is_handled(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_macro_news(
            [_body_row("b4", "s", "kept body"), _body_row("b5", "s", "old body")]
        )
        db_session.flush()

        count = repo.upsert_macro_news(
            [_body_row("b4", "s2", None), _body_row("b5", "s2", "fresh body")]
        )
        db_session.flush()

        assert count == 2
        assert _stored(db_session, "b4").full_content == "kept body"
        assert _stored(db_session, "b5").full_content == "fresh body"


class TestUpsertBondYieldsReferenceDate:
    def test_when_quote_still_dated_last_december_then_stored_in_previous_year(
        self, db_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TE prints 'Dec/31' without a year; read on 2 Jan it is last year's."""

        class _FrozenDate(datetime.date):
            @classmethod
            def today(cls) -> datetime.date:
                return datetime.date(2027, 1, 2)

        monkeypatch.setattr(coerce, "date", _FrozenDate)
        repo = MacroRegimeRepository(db_session)

        repo.upsert_bond_yields(
            "USA_BOND_DATE", {"10Y": {"yield": 4.0, "date": "Dec/31"}}
        )
        db_session.flush()

        row = db_session.query(BondYield).filter(BondYield.country == "USA_BOND_DATE")
        assert row.one().reference_date == datetime.date(2026, 12, 31)
