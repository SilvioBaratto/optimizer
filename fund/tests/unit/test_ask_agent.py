"""The Ask agent reads fund data, docs and source, never writes, and can be wiped."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field, PrivateAttr

from fund.agents import ask_agent
from fund.agents.ask_agent import (
    AskTimeoutError,
    ask_thread_id,
    build_ask_agent,
    clear_ask_state,
    run_ask_turn,
    stage_ask_workspace,
)
from fund.config import FundConfig

PID = uuid.UUID("11111111-2222-3333-4444-555555555555")


# --------------------------------------------------------------------------- fakes


class ScriptedAskModel(BaseChatModel):
    """A network-free chat model that plays back a script, one reply per call."""

    script: list[Any] = Field(default_factory=list)
    _calls: int = PrivateAttr(default=0)
    _seen: list[int] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-ask"

    @property
    def seen(self) -> list[int]:
        """How many messages each model call was given."""
        return self._seen

    def bind_tools(self, tools: Any, **_: Any) -> ScriptedAskModel:
        """Tool binding is a no-op: the script already says which tools to call."""
        return self

    def _generate(self, messages, stop=None, run_manager=None, **_):
        self._seen.append(len(messages))
        reply = (
            self.script[self._calls]
            if self._calls < len(self.script)
            else AIMessage(content="(done)")
        )
        self._calls += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    """A model reply that asks to run one tool."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


@contextmanager
def _no_session():
    yield SimpleNamespace(rollback=lambda: None)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small fake checkout with source, docs, and files that must never be staged."""
    root = tmp_path / "repo"
    (root / "fund/src/fund/tui").mkdir(parents=True)
    (root / "fund/src/fund/observe.py").write_text("def hello():\n    return 'hi'\n")
    (root / "fund/src/fund/tui/app.py").write_text("# app\n")
    (root / "fund/src/fund/__pycache__").mkdir()
    (root / "fund/src/fund/__pycache__/observe.cpython-312.pyc").write_bytes(b"\x00")
    (root / "fund/src/fund/.env").write_text("SECRET=1\n")
    (root / "fund/src/fund/key.pem").write_text("-----BEGIN-----\n")
    (root / "fund/src/fund/secrets").mkdir()
    (root / "fund/src/fund/secrets/token.txt").write_text("hunter2\n")
    (root / "fund/tests/unit").mkdir(parents=True)
    (root / "fund/tests/unit/test_x.py").write_text("def test_x(): ...\n")
    (root / "README.md").write_text("# Optimizer\n")
    (root / "optimizerwiki").mkdir()
    (root / "optimizerwiki/index.md").write_text("# Wiki\n")
    (root / "optimizerwiki/.obsidian").mkdir()
    (root / "optimizerwiki/.obsidian/app.json").write_text("{}")
    return root


@pytest.fixture
def runtime(tmp_path: Path) -> Path:
    """An empty runtime directory the workspace is staged under."""
    return tmp_path / "runtime"


def _staged(repo: Path, runtime: Path) -> Path:
    return stage_ask_workspace(repo_root=repo, runtime_root=runtime)


# --------------------------------------------------------------- staged workspace


def test_the_workspace_holds_source_and_docs(repo, runtime) -> None:
    """The agent's files are the fund source, the README and the wiki."""
    root = _staged(repo, runtime)

    assert (root / "code/fund/observe.py").read_text().startswith("def hello")
    assert (root / "code/fund/tui/app.py").exists()
    assert (root / "docs/README.md").exists()
    assert (root / "docs/optimizerwiki/index.md").exists()


def test_secrets_caches_tests_and_binaries_are_never_staged(repo, runtime) -> None:
    """Only readable text types are copied, and secret-looking paths are skipped."""
    root = _staged(repo, runtime)
    staged = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}

    assert not any(name.endswith((".env", ".pem", ".pyc", ".json")) for name in staged)
    assert not any("secrets" in name or "__pycache__" in name for name in staged)
    assert not any(name.startswith("code/fund/../") for name in staged)
    assert not any("test_x" in name for name in staged)
    assert not any(".obsidian" in name for name in staged)


def test_staging_twice_does_not_rewrite_an_unchanged_workspace(repo, runtime) -> None:
    """A question does not copy the whole tree again when nothing changed."""
    root = _staged(repo, runtime)
    target = root / "code/fund/observe.py"
    before = target.stat().st_mtime_ns

    again = _staged(repo, runtime)

    assert again == root
    assert target.stat().st_mtime_ns == before


def test_a_changed_or_deleted_source_file_is_reflected(repo, runtime) -> None:
    """The agent never reads stale code: edits and removals show on the next stage."""
    root = _staged(repo, runtime)
    (repo / "fund/src/fund/observe.py").write_text("def hello():\n    return 'bye'\n")
    (repo / "fund/src/fund/tui/app.py").unlink()

    root = _staged(repo, runtime)

    assert "bye" in (root / "code/fund/observe.py").read_text()
    assert not (root / "code/fund/tui/app.py").exists()


# ------------------------------------------------- the agent and its file access


def _agent(repo, runtime, script, scope="fund", saver=None):
    model = ScriptedAskModel(script=script)
    agent = build_ask_agent(
        model=model,
        session_factory=_no_session,
        persistence=SimpleNamespace(
            saver=saver or MemorySaver(), store=InMemoryStore()
        ),
        scope=scope,
        focus=None,
        config=FundConfig(),
        repo_root=repo,
        runtime_root=runtime,
    )
    return model, agent


def _tool_results(agent, scope="fund") -> list[str]:
    state = agent.get_state({"configurable": {"thread_id": ask_thread_id(scope)}})
    return [m.content for m in state.values["messages"] if isinstance(m, ToolMessage)]


def test_the_agent_can_read_the_source_it_was_given(repo, runtime) -> None:
    """A file question is answered from the staged source, found by the agent itself."""
    _, agent = _agent(
        repo,
        runtime,
        [
            tool_call("read_file", {"file_path": "/code/fund/observe.py"}),
            AIMessage("ok"),
        ],
    )

    answer = run_ask_turn(agent, scope="fund", question="what does hello do?")

    assert answer == "ok"
    assert any("return 'hi'" in text for text in _tool_results(agent))


def test_the_agent_cannot_write_or_edit_a_file(repo, runtime) -> None:
    """Writes are refused, and the staged source is left exactly as it was."""
    root = _staged(repo, runtime)
    before = (root / "code/fund/observe.py").read_text()
    _, agent = _agent(
        repo,
        runtime,
        [
            tool_call("write_file", {"file_path": "/code/fund/new.py", "content": "x"}),
            tool_call(
                "edit_file",
                {
                    "file_path": "/code/fund/observe.py",
                    "old_string": "hi",
                    "new_string": "pwned",
                },
                "c2",
            ),
            AIMessage("done"),
        ],
    )

    run_ask_turn(agent, scope="fund", question="change the code")

    results = " ".join(_tool_results(agent)).lower()
    assert "permission" in results or "denied" in results
    assert not (root / "code/fund/new.py").exists()
    assert (root / "code/fund/observe.py").read_text() == before


def test_the_agent_cannot_reach_files_outside_its_workspace(repo, runtime) -> None:
    """Absolute paths and ``..`` do not escape; the host's secrets are out of reach."""
    outside = repo.parent / "secret.txt"
    outside.write_text("TOP-SECRET")
    _, agent = _agent(
        repo,
        runtime,
        [
            tool_call("read_file", {"file_path": "/run/secrets/anthropic_api_key"}),
            tool_call("read_file", {"file_path": "/etc/hosts"}, "c2"),
            tool_call("read_file", {"file_path": "/code/../../../secret.txt"}, "c3"),
            AIMessage("done"),
        ],
    )

    run_ask_turn(agent, scope="fund", question="read the keys")

    joined = " ".join(_tool_results(agent))
    assert "TOP-SECRET" not in joined
    assert "localhost" not in joined


def test_the_agent_has_no_way_to_run_a_command(repo, runtime) -> None:
    """There is no shell: an ``execute`` call gets no output."""
    _, agent = _agent(
        repo,
        runtime,
        [
            tool_call("execute", {"command": "echo pwned > pwned.txt"}),
            AIMessage("done"),
        ],
    )

    run_ask_turn(agent, scope="fund", question="run something")

    assert "pwned" not in " ".join(_tool_results(agent)).replace("echo pwned", "")
    assert not Path("pwned.txt").exists()


# ------------------------------------------------------------------ conversation


def test_a_follow_up_question_is_answered_with_the_earlier_turns_in_mind(
    repo, runtime
) -> None:
    """The second question reaches the model together with the first exchange."""
    model, agent = _agent(
        repo, runtime, [AIMessage("first answer"), AIMessage("second")]
    )

    run_ask_turn(agent, scope="fund", question="how are we doing?")
    run_ask_turn(agent, scope="fund", question="and the second one?")

    assert model.seen[1] > model.seen[0]


def test_each_scope_has_its_own_conversation(repo, runtime) -> None:
    """Another portfolio's questions never reach this portfolio's model."""
    saver = MemorySaver()
    _, agent_a = _agent(repo, runtime, [AIMessage("a1"), AIMessage("a2")], "A", saver)
    run_ask_turn(agent_a, scope="A", question="about A")
    _, agent_b = _agent(repo, runtime, [AIMessage("b1")], "B", saver)

    run_ask_turn(agent_b, scope="B", question="about B")

    state_b = agent_b.get_state({"configurable": {"thread_id": ask_thread_id("B")}})
    texts = [str(m.content) for m in state_b.values["messages"]]
    assert "about A" not in " ".join(texts)


def test_clearing_forgets_the_conversation_and_its_files(repo, runtime) -> None:
    """After a clear the agent has no memory and no saved history file remains."""
    saver = MemorySaver()
    model, agent = _agent(
        repo, runtime, [AIMessage("one"), AIMessage("two")], "A", saver
    )
    run_ask_turn(agent, scope="A", question="remember me")
    root = _staged(repo, runtime)
    offloaded = ask_agent.scope_scratch_dir(root, "A") / "conversation_history" / "x.md"
    offloaded.parent.mkdir(parents=True, exist_ok=True)
    offloaded.write_text("portfolio details")

    clear_ask_state(saver, "A", runtime_root=runtime)

    assert saver.get_tuple({"configurable": {"thread_id": ask_thread_id("A")}}) is None
    assert not offloaded.exists()
    run_ask_turn(agent, scope="A", question="who am I?")
    assert model.seen[-1] == model.seen[0]


def test_clearing_one_scope_leaves_another_alone(repo, runtime) -> None:
    """Wiping one portfolio's conversation does not touch another's."""
    saver = MemorySaver()
    _, agent_a = _agent(repo, runtime, [AIMessage("a")], "A", saver)
    _, agent_b = _agent(repo, runtime, [AIMessage("b")], "B", saver)
    run_ask_turn(agent_a, scope="A", question="qa")
    run_ask_turn(agent_b, scope="B", question="qb")

    clear_ask_state(saver, "A", runtime_root=runtime)

    assert (
        saver.get_tuple({"configurable": {"thread_id": ask_thread_id("B")}}) is not None
    )


# --------------------------------------------------------------- running a turn


class _FakeAgent:
    """Streams canned graph updates and records how it was called."""

    def __init__(self, updates, raises: Exception | None = None):
        self.updates = updates
        self.raises = raises
        self.config: dict[str, Any] = {}

    def stream(self, payload, config, stream_mode):
        """Yield the canned updates, then raise the configured error if any."""
        self.config = config
        assert stream_mode == "updates"
        yield from self.updates
        if self.raises:
            raise self.raises


def _ai(content="", calls=None):
    return AIMessage(content=content, tool_calls=calls or [])


def test_each_step_the_agent_takes_is_reported_in_plain_words() -> None:
    """The adviser sees what is being read or searched while the answer is built."""
    agent = _FakeAgent(
        [
            {
                "model": {
                    "messages": [
                        _ai(
                            "",
                            [
                                {
                                    "name": "read_file",
                                    "args": {"file_path": "/code/fund/observe.py"},
                                    "id": "1",
                                }
                            ],
                        )
                    ]
                }
            },
            {
                "model": {
                    "messages": [
                        _ai(
                            "",
                            [{"name": "grep", "args": {"pattern": "gate"}, "id": "2"}],
                        )
                    ]
                }
            },
            {
                "model": {
                    "messages": [
                        _ai("", [{"name": "performance", "args": {}, "id": "3"}])
                    ]
                }
            },
            {"model": {"messages": [_ai("The gate is fine.")]}},
        ]
    )
    steps: list[str] = []

    answer = run_ask_turn(agent, scope="fund", question="q", on_step=steps.append)

    assert answer == "The gate is fine."
    assert steps[0] == "reading /code/fund/observe.py"
    assert steps[1] == "searching for “gate”"
    assert steps[2] == "reading performance"


def test_the_turn_is_bound_to_its_scopes_thread_and_step_cap() -> None:
    """The conversation thread and the step limit are passed to the agent."""
    agent = _FakeAgent([{"model": {"messages": [_ai("hi")]}}])

    run_ask_turn(agent, scope="A", question="q", max_steps=10)

    assert agent.config["configurable"]["thread_id"] == ask_thread_id("A")
    assert agent.config["recursion_limit"] >= 10


def test_a_turn_that_runs_out_of_steps_says_so_instead_of_failing() -> None:
    """Hitting the step cap gives the adviser a plain message, not a traceback."""
    from langgraph.errors import GraphRecursionError

    agent = _FakeAgent([], raises=GraphRecursionError("limit"))

    answer = run_ask_turn(agent, scope="fund", question="q", max_steps=5)

    assert "narrower question" in answer


def test_a_turn_that_takes_too_long_stops_starting_new_steps() -> None:
    """Past the time limit the turn ends with a clear error."""
    agent = _FakeAgent(
        [
            {
                "model": {
                    "messages": [
                        _ai("", [{"name": "ls", "args": {"path": "/"}, "id": "1"}])
                    ]
                }
            },
            {"model": {"messages": [_ai("late")]}},
        ]
    )
    ticks = iter([0.0, 0.0, 500.0, 500.0, 500.0])

    with pytest.raises(AskTimeoutError):
        run_ask_turn(
            agent,
            scope="fund",
            question="q",
            timeout_seconds=100.0,
            clock=lambda: next(ticks),
        )


def test_the_agent_message_without_text_is_not_taken_as_the_answer() -> None:
    """A tool-calling step carries no answer; the last text message does."""
    agent = _FakeAgent(
        [
            {
                "model": {
                    "messages": [
                        _ai("thinking", [{"name": "ls", "args": {}, "id": "1"}])
                    ]
                }
            },
            {"tools": {"messages": [ToolMessage(content="x", tool_call_id="1")]}},
            {"model": {"messages": [_ai("final")]}},
        ]
    )

    assert run_ask_turn(agent, scope="fund", question="q") == "final"


# ------------------------------------------------------------------------- tools


def _tools(monkeypatch, **overrides):
    from fund import observe

    for name, fn in overrides.items():
        monkeypatch.setattr(observe, name, fn)
    return {
        t.name: t
        for t in ask_agent.build_ask_tools(
            session_factory=_no_session, saver=MemorySaver(), focus=PID
        )
    }


def test_the_fund_data_tools_are_exposed(monkeypatch) -> None:
    """The agent can look at portfolios, runs, performance, mandate and audit."""
    names = set(_tools(monkeypatch))

    assert {
        "fund_snapshot",
        "list_portfolios",
        "portfolio_state",
        "performance",
        "holdings",
        "mandate",
        "pending_gates",
        "list_runs",
        "run_transcript",
        "daily_brief",
        "audit_trail",
    } <= names


def test_a_tool_returns_readable_text_and_no_blank_means_the_focus(monkeypatch) -> None:
    """With no id given, the tool answers for the portfolio on screen."""
    seen: list[uuid.UUID] = []

    def fake_performance(session, portfolio_id, *, asof=None):
        """Record which portfolio was asked about and return a tiny summary."""
        seen.append(portfolio_id)
        return {"returns": {"1D": -0.0032}}

    tools = _tools(monkeypatch, performance_summary=fake_performance)

    text = tools["performance"].invoke({})

    assert seen == [PID]
    assert "-0.0032" in text


def test_a_short_portfolio_id_is_accepted(monkeypatch) -> None:
    """The first eight characters, as the cockpit shows them, name a portfolio."""
    seen: list[uuid.UUID] = []
    tools = _tools(
        monkeypatch,
        list_portfolios=lambda session: [PID],
        performance_summary=lambda session, pid, *, asof=None: seen.append(pid) or {},
    )

    tools["performance"].invoke({"portfolio_id": str(PID)[:8]})

    assert seen == [PID]


def test_an_unknown_portfolio_is_reported_not_raised(monkeypatch) -> None:
    """A made-up id gives the agent a message it can recover from."""
    tools = _tools(monkeypatch, list_portfolios=lambda session: [PID])

    text = tools["performance"].invoke({"portfolio_id": "ffffffff"})

    assert "no portfolio" in text.lower()


def test_a_failing_read_is_reported_as_text(monkeypatch) -> None:
    """A database error becomes an ``error:`` line the agent can read, not a crash."""

    def boom(session, portfolio_id, *, asof=None):
        """Fail the way a lost database connection does."""
        raise RuntimeError("db down")

    tools = _tools(monkeypatch, performance_summary=boom)

    assert tools["performance"].invoke({}).startswith("error:")


def test_a_very_long_result_is_cut_to_keep_the_context_small(monkeypatch) -> None:
    """Tool output is capped so one read cannot flood the model's context."""
    tools = _tools(
        monkeypatch,
        performance_summary=lambda session, pid, *, asof=None: {"x": "y" * 50_000},
    )

    text = tools["performance"].invoke({})

    assert len(text) < 10_000
    assert "truncated" in text


def test_the_tools_only_read(monkeypatch) -> None:
    """No tool commits, and each one rolls its session back."""
    rolled: list[bool] = []

    @contextmanager
    def session_factory():
        """Hand out a session that records rollbacks and refuses commits."""
        yield SimpleNamespace(
            rollback=lambda: rolled.append(True),
            commit=lambda: (_ for _ in ()).throw(AssertionError("commit")),
        )

    from fund import observe

    monkeypatch.setattr(observe, "list_portfolios", lambda session: [PID])
    tools = {
        t.name: t
        for t in ask_agent.build_ask_tools(
            session_factory=session_factory, saver=MemorySaver(), focus=PID
        )
    }

    tools["list_portfolios"].invoke({})

    assert rolled == [True]
