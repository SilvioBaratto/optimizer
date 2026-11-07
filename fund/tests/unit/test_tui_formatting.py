"""What the TUI shows and exports keeps the precision and structure of its source."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from _fund_fakes import PORTFOLIO_ID, threaded_sqlite_engine
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy.orm import sessionmaker
from textual.widgets import ContentSwitcher

from fund.tui.app import ShellApp
from fund.tui.views import AskView, _format_caps


def test_the_position_cap_keeps_its_decimals() -> None:
    """A 7.5% name cap reads 7.5%, not the rounded 8% an adviser might act on."""
    caps = _format_caps({"universe_filters": {"max_position_cap": 0.075}})

    assert "name ≤7.5%" in caps


def test_a_whole_percent_cap_keeps_one_decimal_like_the_drawdown_tiers() -> None:
    """The two kinds of figure share one format on the same line."""
    caps = _format_caps(
        {"nu2": 0.15, "universe_filters": {"max_position_cap": 0.1}},
    )

    assert "ceiling ≤15.0%" in caps
    assert "name ≤10.0%" in caps


def test_a_drawdown_tier_keeps_two_decimals_when_it_has_them() -> None:
    """A 1.5x stop of a 7.5% ceiling is 11.25%; one decimal would misstate it."""
    caps = _format_caps({"nu1": 0.075, "nu3": 0.1125})

    assert "warn ≤7.5%" in caps
    assert "stop ≤11.25%" in caps


@pytest.fixture
def shell(tmp_path):
    """A shell over an empty SQLite database, focused on the test portfolio."""
    engine = threaded_sqlite_engine(tmp_path / "fmt.db")
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @contextmanager
    def get_session():
        """Yield a session on the shared engine, closed on exit."""
        session = session_local()
        try:
            yield session
        finally:
            session.close()

    app = ShellApp(
        PORTFOLIO_ID,
        session_factory=get_session,
        persistence=SimpleNamespace(
            saver=MemorySaver(), store=InMemoryStore(), pool=None
        ),
        model_factory=lambda: None,
        poll_interval=30.0,
    )
    yield app
    engine.dispose()


@pytest.mark.asyncio
async def test_an_ask_export_quotes_model_text_so_it_cannot_restructure_the_file(
    shell, tmp_path
) -> None:
    """A heading or bullet inside an answer stays inside its speaker's block."""
    async with shell.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        shell.query_one(ContentSwitcher).current = "view-ask"
        shell._refresh()
        await pilot.pause()
        view = shell.query_one("#view-ask", AskView)
        view.append_log("you", "how are we doing?")
        view.append_log("ask", "Summary\n\n# Injected heading\n- injected bullet\n---")

        path = view.export_markdown(str(tmp_path / "ask.md"))

    lines = open(path, encoding="utf-8").read().splitlines()  # noqa: SIM115, PTH123
    assert lines[0] == "# Ask conversation"
    structural = [
        line for line in lines[1:] if line and not line.startswith((">", "**"))
    ]
    assert structural == []
    assert "> # Injected heading" in lines
    assert "> - injected bullet" in lines
    assert "> how are we doing?" in lines
