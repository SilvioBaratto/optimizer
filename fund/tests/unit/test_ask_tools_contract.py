"""Exact contract of the Ask agent's eleven fund-data tools.

Validates ``fund/src/fund/agents/ask_agent.py`` (``build_ask_tools`` and the tools it
returns): the order, names, descriptions and argument schemas the model sees; that
every tool opens one session, asks Postgres for a read-only transaction before it
reads, rolls back after it, and never commits; that a conversation about one
portfolio refuses every other portfolio's data with one exact message; how ids,
prefixes, limits, run transcripts and dates are resolved; and the exact ``error:``
text of each refusal. ``test_ask_agent.py`` checks the same tools by behaviour; this
file pins the details a refactor could drop silently. No requirement id is
registered for the Ask view yet.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from fund import observe
from fund.agents import ask_agent
from fund.tui import ask as ask_module

PID = uuid.UUID("11111111-2222-3333-4444-555555555555")
OTHER = uuid.UUID("99999999-9999-9999-9999-999999999999")
RUN_ID = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
SAVER = MemorySaver()
REFUSED = (
    f"error: this conversation is about portfolio {PID}; switch the header to that "
    "portfolio to ask about another"
)
NOTHING_FOCUSED = (
    "error: no portfolio is focused; pass portfolio_id (see list_portfolios)"
)
READ_ONLY = "execute:SET TRANSACTION READ ONLY"
_BLANK_ID = {"portfolio_id": {"default": "", "title": "Portfolio Id", "type": "string"}}
# The order the model is offered the tools in, with the description and the
# argument schema it reads for each. The descriptions are pinned without the
# docstring's continuation indent: Python 3.12 keeps it and 3.13 strips it at compile
# time, so the test compares them through ``inspect.cleandoc``.
TOOL_CONTRACT: list[tuple[str, str, dict[str, Any]]] = [
    (
        "fund_snapshot",
        "Summarize the fund now: portfolios, pending gates, the focused portfolio.",
        {},
    ),
    ("list_portfolios", "List every portfolio id the fund manages.", {}),
    (
        "portfolio_state",
        "Show a portfolio's drift, holdings count and run status.\n\n"
        "A blank portfolio_id means the focused portfolio.",
        _BLANK_ID,
    ),
    (
        "performance",
        "Show a portfolio's returns, risk and excess return.\n\n"
        "A blank portfolio_id means the focused portfolio.",
        _BLANK_ID,
    ),
    (
        "holdings",
        "Show a portfolio's holdings with value, one-day change and P&L.\n\n"
        "A blank portfolio_id means the focused portfolio.",
        _BLANK_ID,
    ),
    (
        "mandate",
        "Show a portfolio's mandate (limits, eligible tickers) and its warnings.",
        _BLANK_ID,
    ),
    (
        "pending_gates",
        "List the decision gates waiting for a human to approve or reject.",
        {},
    ),
    (
        "list_runs",
        "List a portfolio's most recent decision runs with their status and date.",
        {
            **_BLANK_ID,
            "limit": {"default": 10, "title": "Limit", "type": "integer"},
        },
    ),
    (
        "run_transcript",
        "Read what the agents said and decided in one run (id from list_runs).",
        {"run_id": {"title": "Run Id", "type": "string"}},
    ),
    (
        "daily_brief",
        "Show the daily brief for a day: P&L, attribution, market digest, actions.\n\n"
        "The day is YYYY-MM-DD; blank is the latest.",
        {**_BLANK_ID, "day": {"default": "", "title": "Day", "type": "string"}},
    ),
    (
        "audit_trail",
        "Show the newest audit entries: who decided or did what, and when.\n\n"
        "A blank portfolio_id means the focused portfolio, or all when none is "
        "focused.",
        {
            **_BLANK_ID,
            "limit": {"default": 30, "title": "Limit", "type": "integer"},
        },
    ),
]
# The tools that take a portfolio id, and the readers each one calls with it.
PORTFOLIO_TOOLS = [
    ("portfolio_state", ["portfolio_state"]),
    ("performance", ["performance_summary"]),
    ("holdings", ["holdings_summary"]),
    ("mandate", ["get_mandate", "mandate_warnings"]),
    ("list_runs", ["list_portfolio_runs"]),
    ("daily_brief", ["daily_brief"]),
    ("audit_trail", ["audit_timeline"]),
]
EVERY_TOOL = [
    ("fund_snapshot", {}, ["build_context"]),
    ("list_portfolios", {}, []),
    *[(name, {}, readers) for name, readers in PORTFOLIO_TOOLS],
    ("pending_gates", {}, ["pending_hitl"]),
    ("run_transcript", {"run_id": str(RUN_ID)}, ["load_run_transcript"]),
]
_READERS = (
    "portfolio_state",
    "performance_summary",
    "holdings_summary",
    "get_mandate",
    "mandate_warnings",
    "list_portfolio_runs",
    "daily_brief",
    "audit_timeline",
    "pending_hitl",
    "load_run_transcript",
)
# What a reader returns unless a test says otherwise: a value each tool can render.
_DEFAULT_RESULTS: dict[str, Any] = {
    "build_context": "snapshot",
    "pending_hitl": [],
    "audit_timeline": [],
    "load_run_transcript": [],
}


class _Session:
    """A session stand-in that logs what each read does to it, and refuses commits."""

    def __init__(self, events: list[str], dialect: str | None) -> None:
        self._events = events
        self._dialect = dialect

    def __getattr__(self, name: str) -> Any:
        if name == "get_bind" and self._dialect is not None:
            dialect = SimpleNamespace(name=self._dialect)
            return lambda: SimpleNamespace(dialect=dialect)
        raise AttributeError(name)

    def execute(self, statement: Any) -> None:
        """Record the statement text."""
        self._events.append(f"execute:{statement}")

    def rollback(self) -> None:
        """Record the rollback."""
        self._events.append("rollback")

    def commit(self) -> None:
        """A read-only tool must never get here."""
        raise AssertionError("a read-only tool committed")


class _Unreachable:
    """A session factory whose connection fails before a session exists."""

    def __enter__(self) -> Any:
        raise RuntimeError("no connection")

    def __exit__(self, *exc: object) -> bool:
        return False


def _factory(events: list[str], dialect: str | None = "postgresql") -> Any:
    @contextmanager
    def session_factory() -> Iterator[_Session]:
        """Open a logging session and log its close."""
        events.append("open")
        try:
            yield _Session(events, dialect)
        finally:
            events.append("close")

    return session_factory


def _readers(monkeypatch: pytest.MonkeyPatch, events: list[str], **results: Any):
    """Replace every reader the tools call with one that logs its call."""
    calls: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    answers = {**_DEFAULT_RESULTS, **results}

    def reader(name: str) -> Any:
        """A stand-in for one ``observe`` reader."""

        def fake(*args: Any, **kwargs: Any) -> Any:
            """Log the call and return the canned result for this reader."""
            events.append(f"work:{name}")
            calls[name] = (args[1:], kwargs)
            return answers.get(name, {"reader": name})

        return fake

    for name in _READERS:
        monkeypatch.setattr(observe, name, reader(name))
    portfolios = answers.get("list_portfolios", [PID, OTHER])
    monkeypatch.setattr(observe, "list_portfolios", lambda session: list(portfolios))
    monkeypatch.setattr(ask_module, "build_context", reader("build_context"))
    return calls


def _run_owned_by(monkeypatch: pytest.MonkeyPatch, owner: Any) -> None:
    import fund.audit.repository as repository

    class FakeRuns:
        """Hands back one run of ``owner`` for ``RUN_ID`` and nothing else."""

        def __init__(self, session: Any) -> None:
            self.session = session

        def get_run(self, run_id: uuid.UUID) -> Any:
            """The run, or ``None`` for any other id."""
            if run_id != RUN_ID:
                return None
            return SimpleNamespace(id=run_id, portfolio_id=owner)

    monkeypatch.setattr(repository, "AgentRunRepository", FakeRuns)


def _tools(events: list[str], focus: uuid.UUID | None = PID, **kwargs: Any):
    return {
        t.name: t
        for t in ask_agent.build_ask_tools(
            session_factory=_factory(events, **kwargs), saver=SAVER, focus=focus
        )
    }


def _portfolio_read(call: tuple[tuple[Any, ...], dict[str, Any]]) -> Any:
    """The portfolio a reader was asked about, positional or by keyword."""
    args, kwargs = call
    return kwargs.get("portfolio_id", args[-1] if args else None)


# ----------------------------------------------------------------- what the model sees


def test_the_tools_come_in_a_fixed_order_with_their_exact_descriptions() -> None:
    """Names, order, descriptions and argument schemas are what the model reads."""
    tools = ask_agent.build_ask_tools(
        session_factory=_factory([]), saver=SAVER, focus=PID
    )

    seen = [(t.name, inspect.cleandoc(t.description), t.args) for t in tools]

    assert seen == TOOL_CONTRACT
    assert {type(t).__name__ for t in tools} == {"StructuredTool"}


# ------------------------------------------------------------------- read-only reads


@pytest.mark.parametrize(("name", "args", "work"), EVERY_TOOL)
def test_every_tool_reads_inside_one_read_only_rolled_back_session(
    monkeypatch, name, args, work
) -> None:
    """Read-only is declared before the read, and the session is rolled back after."""
    events: list[str] = []
    _readers(monkeypatch, events)
    _run_owned_by(monkeypatch, PID)

    text = _tools(events)[name].invoke(args)

    assert not text.startswith("error:"), text
    assert events == [
        "open",
        READ_ONLY,
        *[f"work:{reader}" for reader in work],
        "rollback",
        "close",
    ]


@pytest.mark.parametrize(("name", "args", "work"), EVERY_TOOL)
def test_a_session_that_is_not_postgres_is_read_without_the_statement(
    monkeypatch, name, args, work
) -> None:
    """SQLite has no read-only transactions, so nothing is executed there."""
    events: list[str] = []
    _readers(monkeypatch, events)
    _run_owned_by(monkeypatch, PID)

    _tools(events, dialect="sqlite")[name].invoke(args)

    assert events == [
        "open",
        *[f"work:{reader}" for reader in work],
        "rollback",
        "close",
    ]


def test_a_session_without_a_bind_is_read_without_the_statement(monkeypatch) -> None:
    """A session with no ``get_bind`` is read as it is."""
    events: list[str] = []
    _readers(monkeypatch, events)

    _tools(events, dialect=None)["portfolio_state"].invoke({})

    assert events == ["open", "work:portfolio_state", "rollback", "close"]


def test_a_failing_read_still_rolls_back_and_reports_the_error(monkeypatch) -> None:
    """The error becomes one ``error:`` line, after the rollback."""
    events: list[str] = []
    _readers(monkeypatch, events)

    def boom(*args: Any, **kwargs: Any) -> Any:
        """Fail like a lost connection."""
        events.append("work:boom")
        raise RuntimeError("db down")

    monkeypatch.setattr(observe, "performance_summary", boom)

    text = _tools(events)["performance"].invoke({})

    assert text == "error: db down"
    assert events == ["open", READ_ONLY, "work:boom", "rollback", "close"]


def test_a_refused_read_only_statement_stops_the_read(monkeypatch) -> None:
    """If Postgres will not go read-only, nothing is read and the session is undone."""
    events: list[str] = []
    _readers(monkeypatch, events)

    def refuse(self: Any, statement: Any) -> None:
        """Fail the way a server that refuses the statement does."""
        events.append("execute:refused")
        raise RuntimeError("cannot set read only")

    monkeypatch.setattr(_Session, "execute", refuse)

    text = _tools(events)["portfolio_state"].invoke({})

    assert text == "error: cannot set read only"
    assert events == ["open", "execute:refused", "rollback", "close"]


def test_a_session_that_cannot_be_opened_is_reported_as_text() -> None:
    """A factory that fails on enter gives an ``error:`` line, not an exception."""
    tools = {
        t.name: t
        for t in ask_agent.build_ask_tools(
            session_factory=_Unreachable, saver=SAVER, focus=PID
        )
    }

    assert tools["holdings"].invoke({}) == "error: no connection"


# ---------------------------------------------------- one portfolio per conversation


@pytest.mark.parametrize(("name", "work"), PORTFOLIO_TOOLS)
def test_a_focused_conversation_refuses_another_portfolio_by_id(
    monkeypatch, name, work
) -> None:
    """Another portfolio's full id is refused before any of its data is read."""
    events: list[str] = []
    _readers(monkeypatch, events)

    text = _tools(events)[name].invoke({"portfolio_id": str(OTHER)})

    assert text == REFUSED
    assert events == ["open", READ_ONLY, "rollback", "close"]


@pytest.mark.parametrize(("name", "work"), PORTFOLIO_TOOLS)
def test_a_focused_conversation_refuses_another_portfolio_by_prefix(
    monkeypatch, name, work
) -> None:
    """A prefix that resolves to another portfolio is refused the same way."""
    events: list[str] = []
    _readers(monkeypatch, events)

    text = _tools(events)[name].invoke({"portfolio_id": " 9999 "})

    assert text == REFUSED
    assert events == ["open", READ_ONLY, "rollback", "close"]


@pytest.mark.parametrize(("name", "work"), PORTFOLIO_TOOLS)
def test_a_focused_conversation_reads_its_own_portfolio_by_id(
    monkeypatch, name, work
) -> None:
    """The focused portfolio's own id, in any case, is read."""
    events: list[str] = []
    calls = _readers(monkeypatch, events)

    _tools(events)[name].invoke({"portfolio_id": str(PID).upper()})

    assert [_portfolio_read(calls[reader]) for reader in work] == [PID] * len(work)


@pytest.mark.parametrize(("name", "work"), PORTFOLIO_TOOLS)
def test_a_blank_id_means_the_focused_portfolio(monkeypatch, name, work) -> None:
    """With nothing named, each tool reads the portfolio on screen."""
    events: list[str] = []
    calls = _readers(monkeypatch, events)

    _tools(events)[name].invoke({})

    assert [_portfolio_read(calls[reader]) for reader in work] == [PID] * len(work)


def test_a_focused_conversation_cannot_open_another_portfolios_run(
    monkeypatch,
) -> None:
    """A run of another portfolio is refused and its transcript is never loaded."""
    events: list[str] = []
    _readers(monkeypatch, events)
    _run_owned_by(monkeypatch, OTHER)

    text = _tools(events)["run_transcript"].invoke({"run_id": str(RUN_ID)})

    assert text == REFUSED
    assert events == ["open", READ_ONLY, "rollback", "close"]


def test_a_run_without_an_owner_can_be_opened_in_a_focused_conversation(
    monkeypatch,
) -> None:
    """Only a run that names another portfolio is refused."""
    events: list[str] = []
    _readers(monkeypatch, events)
    _run_owned_by(monkeypatch, None)

    assert _tools(events)["run_transcript"].invoke({"run_id": str(RUN_ID)}) == "[]"


def test_a_focused_conversation_lists_and_snapshots_only_its_portfolio(
    monkeypatch,
) -> None:
    """The list, the gates and the snapshot are all scoped to the focus."""
    events: list[str] = []
    gates = [
        SimpleNamespace(portfolio_id=PID, run_id="mine"),
        SimpleNamespace(portfolio_id=OTHER, run_id="theirs"),
    ]
    calls = _readers(
        monkeypatch, events, pending_hitl=gates, build_context="line one\nline two"
    )
    tools = _tools(events)

    assert tools["list_portfolios"].invoke({}) == f'["{PID}"]'
    assert json.loads(tools["pending_gates"].invoke({})) == [
        {"portfolio_id": str(PID), "run_id": "mine"}
    ]
    assert tools["fund_snapshot"].invoke({}) == '["line one", "line two"]'
    assert calls["pending_hitl"] == ((SAVER, None), {})
    assert calls["build_context"] == ((SAVER, PID, ""), {})


# ------------------------------------------------------------ the fund-wide view


def test_the_fund_wide_view_reads_and_lists_every_portfolio(monkeypatch) -> None:
    """With nothing focused, any portfolio can be read and all are listed."""
    events: list[str] = []
    gates = [
        SimpleNamespace(portfolio_id=PID, run_id="mine"),
        SimpleNamespace(portfolio_id=OTHER, run_id="theirs"),
    ]
    calls = _readers(monkeypatch, events, pending_hitl=gates)
    tools = _tools(events, focus=None)

    tools["performance"].invoke({"portfolio_id": str(OTHER)})
    tools["fund_snapshot"].invoke({})

    assert calls["performance_summary"][0] == (OTHER,)
    assert tools["list_portfolios"].invoke({}) == f'["{PID}", "{OTHER}"]'
    assert len(json.loads(tools["pending_gates"].invoke({}))) == 2
    assert calls["build_context"] == ((SAVER, None, ""), {})


@pytest.mark.parametrize(
    "name", [name for name, _ in PORTFOLIO_TOOLS if name != "audit_trail"]
)
def test_a_blank_id_with_nothing_focused_asks_for_one(monkeypatch, name) -> None:
    """There is no portfolio to default to, so the tool says how to name one."""
    events: list[str] = []
    _readers(monkeypatch, events)

    text = _tools(events, focus=None)[name].invoke({"portfolio_id": "  "})

    assert text == NOTHING_FOCUSED
    assert not [event for event in events if event.startswith("work:")]


def test_an_ambiguous_prefix_asks_for_more_characters(monkeypatch) -> None:
    """Two portfolios share the prefix, so neither is guessed."""
    events: list[str] = []
    first = uuid.UUID("abcd0000-0000-0000-0000-000000000001")
    second = uuid.UUID("abcd0000-0000-0000-0000-000000000002")
    _readers(monkeypatch, events, list_portfolios=[first, second])

    text = _tools(events, focus=None)["holdings"].invoke({"portfolio_id": " ABCD "})

    assert text == "error: 'abcd' matches several portfolios; give more characters"


def test_an_unknown_prefix_points_at_the_portfolio_list(monkeypatch) -> None:
    """A prefix that matches nothing names the tool that lists the ids."""
    events: list[str] = []
    _readers(monkeypatch, events)

    text = _tools(events, focus=None)["holdings"].invoke({"portfolio_id": "ffff"})

    assert text == "error: no portfolio matches 'ffff' (see list_portfolios)"


def test_a_unique_prefix_in_the_fund_wide_view_reads_that_portfolio(
    monkeypatch,
) -> None:
    """The first characters the cockpit shows are enough to name a portfolio."""
    events: list[str] = []
    calls = _readers(monkeypatch, events)

    _tools(events, focus=None)["holdings"].invoke({"portfolio_id": "9999"})

    assert calls["holdings_summary"][0] == (OTHER,)


# ------------------------------------------------------------------ bounds and parsing


@pytest.mark.parametrize(
    ("asked", "sent"), [(-5, 1), (0, 1), (1, 1), (10, 10), (50, 50), (51, 50)]
)
def test_list_runs_bounds_its_limit(monkeypatch, asked, sent) -> None:
    """At least one run and at most fifty are asked for."""
    events: list[str] = []
    calls = _readers(monkeypatch, events, list_portfolio_runs=[])

    _tools(events)["list_runs"].invoke({"limit": asked})

    assert calls["list_portfolio_runs"] == ((PID,), {"limit": sent})


@pytest.mark.parametrize(
    ("asked", "kept"), [(-1, 1), (0, 1), (3, 3), (30, 30), (100, 100), (101, 100)]
)
def test_audit_trail_cuts_to_its_bounded_limit(monkeypatch, asked, kept) -> None:
    """At least one row and at most a hundred are returned, in the order read."""
    events: list[str] = []
    calls = _readers(monkeypatch, events, audit_timeline=list(range(150)))

    text = _tools(events)["audit_trail"].invoke({"limit": asked})

    assert json.loads(text) == list(range(kept))
    assert calls["audit_timeline"] == ((), {"portfolio_id": PID})


def test_the_audit_trail_of_the_fund_wide_view_reads_everything(monkeypatch) -> None:
    """With nothing focused and no id, the audit trail is not scoped."""
    events: list[str] = []
    calls = _readers(monkeypatch, events)

    _tools(events, focus=None)["audit_trail"].invoke({})

    assert calls["audit_timeline"] == ((), {"portfolio_id": None})


def test_a_run_transcript_keeps_the_last_forty_entries_cut_to_600_chars(
    monkeypatch,
) -> None:
    """Older entries and long texts are cut so one run cannot flood the context."""
    events: list[str] = []
    entries = [
        SimpleNamespace(agent=f"a{i}", step=f"s{i}", text=f"{i}:" + "x" * 700)
        for i in range(50)
    ]
    _readers(monkeypatch, events, load_run_transcript=entries)
    _run_owned_by(monkeypatch, PID)
    monkeypatch.setattr(ask_agent, "_TOOL_TEXT_LIMIT", 10**9)

    text = _tools(events)["run_transcript"].invoke({"run_id": f"  {RUN_ID}  "})

    rows = json.loads(text)
    assert len(rows) == 40
    assert rows[0] == {"agent": "a10", "step": "s10", "text": ("10:" + "x" * 700)[:600]}
    assert rows[-1]["agent"] == "a49"
    assert {len(row["text"]) for row in rows} == {600}


def test_an_unknown_run_is_named_in_the_error(monkeypatch) -> None:
    """A well-formed id with no run behind it says so."""
    events: list[str] = []
    _readers(monkeypatch, events)
    _run_owned_by(monkeypatch, PID)
    missing = str(uuid.UUID(int=7))

    text = _tools(events)["run_transcript"].invoke({"run_id": missing})

    assert text == f"error: no run {missing!r}"


def test_a_malformed_run_id_is_reported_as_text(monkeypatch) -> None:
    """The parse error of a made-up id reaches the model as one line."""
    events: list[str] = []
    _readers(monkeypatch, events)

    text = _tools(events)["run_transcript"].invoke({"run_id": "not-a-uuid"})

    assert text == "error: badly formed hexadecimal UUID string"


@pytest.mark.parametrize(
    ("day", "asof"), [("", None), ("   ", None), ("2026-11-09", dt.date(2026, 11, 9))]
)
def test_the_daily_brief_reads_the_day_asked_or_the_latest(
    monkeypatch, day, asof
) -> None:
    """A blank day is the latest brief; a date is parsed as ISO."""
    events: list[str] = []
    calls = _readers(monkeypatch, events)

    _tools(events)["daily_brief"].invoke({"day": day})

    assert calls["daily_brief"] == ((SAVER, PID), {"asof": asof})


def test_a_badly_written_day_is_reported_as_text(monkeypatch) -> None:
    """A date in another format is refused with the parser's own words."""
    events: list[str] = []
    _readers(monkeypatch, events)

    text = _tools(events)["daily_brief"].invoke({"day": "11/09/2026"})

    assert text == "error: Invalid isoformat string: '11/09/2026'"


def test_the_mandate_tool_returns_the_mandate_and_its_warnings(monkeypatch) -> None:
    """One read returns both parts under fixed keys."""
    events: list[str] = []
    _readers(monkeypatch, events, get_mandate={"cap": 0.1}, mandate_warnings=["w"])

    text = _tools(events)["mandate"].invoke({})

    assert json.loads(text) == {"mandate": {"cap": 0.1}, "warnings": ["w"]}


def test_a_long_result_is_cut_with_a_note_of_how_much_was_left_out(
    monkeypatch,
) -> None:
    """The cut keeps the first characters and counts the rest."""
    events: list[str] = []
    _readers(monkeypatch, events, performance_summary="y" * 7000)

    text = _tools(events)["performance"].invoke({})

    full = json.dumps("y" * 7000)
    assert text == f"{full[:6000]}\n… (truncated, {len(full) - 6000} more characters)"
