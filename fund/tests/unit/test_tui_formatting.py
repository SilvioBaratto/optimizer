"""What the TUI shows and exports keeps the precision and structure of its source."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from _fund_fakes import PORTFOLIO_ID, threaded_sqlite_engine
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy.orm import sessionmaker
from textual.widgets import ContentSwitcher, Static

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


_DIGEST = (
    "Market digest for 2026-11-06 (US). "
    "Macro: 2 series — BAA10Y 1.57 as of 2026-11-05 (delta 0.05), "
    "VIXCLS 28.18 as of 2026-11-05 (delta -11.57). "
    "Markets: US: GC=F, ^DJI. News themes: business_cycle (13), sector_rotation (5)."
)


def _brief_with(events):
    """A brief that carries only the given market journal events."""
    from datetime import date

    from fund import observe
    from fund.attribution import AttributionResult

    attribution = AttributionResult(
        total_return=0.0,
        contributions={},
        residual=0.0,
        factor_names=(),
        insufficient_history=True,
    )
    return observe.DailyBrief(
        portfolio_id=PORTFOLIO_ID,
        asof=date(2026, 11, 6),
        prev_day=None,
        next_day=None,
        performance=None,
        attribution=attribution,
        market_events=events,
        economist_read=None,
        economist_read_date=None,
        deliberation_run_id=None,
        fund_events=(),
    )


def _happened(detail: str) -> list[str]:
    """The lines of the "what happened" section of a rendered brief."""
    lines = detail.splitlines()
    start = lines.index("WHAT HAPPENED (market_journal)") + 1
    end = lines.index("ECONOMIST READ")
    return [line for line in lines[start:end] if line]


def test_the_daily_brief_shows_the_market_digest_as_short_lines() -> None:
    """The digest is not one paragraph the width of the screen."""
    from fund.tui.views import _render_brief_detail

    detail = _render_brief_detail(
        _brief_with(({"as_of": "2026-11-06", "narrative": _DIGEST},))
    )

    lines = _happened(detail)
    assert len(lines) >= 6
    assert max(len(line) for line in lines) <= 100
    assert any("BAA10Y" in line and "1.57" in line for line in lines)
    assert any("VIXCLS" in line and "Δ -11.57" in line for line in lines)
    assert lines[0].startswith("  • Market digest")


def test_the_daily_brief_leaves_other_journal_notes_as_they_were() -> None:
    """A note that is not a digest keeps its one-line bullet and its date."""
    from fund.tui.views import _render_brief_detail

    detail = _render_brief_detail(
        _brief_with(({"as_of": "2026-10-06", "narrative": "CPI < expected"},))
    )

    assert _happened(detail) == ["  • CPI < expected  (2026-10-06)"]


@pytest.mark.asyncio
async def test_the_ask_log_formats_the_answer_and_spaces_the_turns(shell) -> None:
    """Bold marks and bullets are laid out, and each turn is a separate block."""
    async with shell.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        shell.query_one(ContentSwitcher).current = "view-ask"
        shell._refresh()
        await pilot.pause()
        view = shell.query_one("#view-ask", AskView)
        view.append_log("you", "how are we doing?")
        view.append_log("ask", "## Status\n**Healthy** overall\n- one\n- two")
        await pilot.pause()

        shown = str(view.query_one("#ask-log", Static).render())

    assert "**" not in shown
    assert "##" not in shown
    assert "• one" in shown
    assert "you: how are we doing?\n\nask: Status\nHealthy overall" in shown
