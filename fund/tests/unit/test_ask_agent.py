"""The Ask agent reads fund data, docs and source, never writes, and can be wiped."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _fund_fakes import NoToolsChatModel
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


class ScriptedAskModel(NoToolsChatModel):
    """A network-free chat model that plays back a script, one reply per call."""

    script: list[Any] = Field(default_factory=list)
    _calls: int = PrivateAttr(default=0)
    _seen: list[int] = PrivateAttr(default_factory=list)
    _texts: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-ask"

    @property
    def seen(self) -> list[int]:
        """How many messages each model call was given."""
        return self._seen

    @property
    def texts(self) -> list[str]:
        """The text of every message each model call was given, joined."""
        return self._texts

    def reset(self) -> None:
        """Start the script again from its first reply."""
        self._calls = 0

    def _generate(self, messages, stop=None, run_manager=None, **_):
        self._seen.append(len(messages))
        self._texts.append(" | ".join(str(m.content) for m in messages))
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
    assert "how are we doing?" in model.texts[1]


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
    assert agent.config["recursion_limit"] == 21


def test_a_turn_that_runs_out_of_steps_says_so_instead_of_failing() -> None:
    """Hitting the step cap gives the adviser a plain message, not a traceback."""
    from langgraph.errors import GraphRecursionError

    agent = _FakeAgent([], raises=GraphRecursionError("limit"))

    answer = run_ask_turn(agent, scope="fund", question="q", max_steps=5)

    assert "narrower question" in answer


def test_a_turn_that_takes_too_long_stops_starting_new_steps() -> None:
    """Past the time limit an unfinished turn ends with a clear error."""
    agent = _FakeAgent(
        [
            {
                "model": {
                    "messages": [
                        _ai("", [{"name": "ls", "args": {"path": "/"}, "id": "1"}])
                    ]
                }
            },
            {
                "model": {
                    "messages": [
                        _ai("", [{"name": "ls", "args": {"path": "/x"}, "id": "2"}])
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


# ------------------------------------------------------- review hardening: staging


def test_a_symlinked_file_is_never_staged(repo, runtime) -> None:
    """A link in the source tree is not followed, so it cannot copy a secret in."""
    secret = repo.parent / "outside-secret.txt"
    secret.write_text("TOP-SECRET")
    (repo / "fund/src/fund/linked.md").symlink_to(secret)

    root = _staged(repo, runtime)

    assert not (root / "code/fund/linked.md").exists()
    assert "TOP-SECRET" not in " ".join(
        p.read_text() for p in root.rglob("*") if p.is_file() and p.suffix == ".md"
    )


def test_a_broken_symlink_does_not_stop_staging(repo, runtime) -> None:
    """A dangling editor lock file in the tree must not make every question fail."""
    (repo / "fund/src/fund/.#observe.py").symlink_to(repo / "does-not-exist.py")

    root = _staged(repo, runtime)

    assert (root / "code/fund/observe.py").exists()


def test_a_symlinked_source_root_is_not_followed(repo, runtime) -> None:
    """A whole directory that is a link to somewhere else is skipped."""
    elsewhere = repo.parent / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "plan.md").write_text("outside")
    (repo / "optimizerwiki").rename(repo / "wiki-real")
    (repo / "optimizerwiki").symlink_to(elsewhere)

    root = _staged(repo, runtime)

    assert not (root / "docs/optimizerwiki/plan.md").exists()


@pytest.mark.parametrize(
    "name",
    [
        "credentials.yaml",
        "secrets.toml",
        "my_secret_notes.md",
        ".ENV.md",
        "Password.txt",
    ],
)
def test_files_with_secret_looking_names_are_not_staged(repo, runtime, name) -> None:
    """The suffix allowlist is not the only gate: telling names are skipped too."""
    (repo / "fund/src/fund" / name).write_text("value")

    root = _staged(repo, runtime)

    assert not (root / "code/fund" / name).exists()


@pytest.mark.parametrize(
    "folder", ["tests", "__pycache__", ".obsidian", "secrets", ".git", ".venv"]
)
def test_skipped_folders_hold_nothing_that_is_staged(repo, runtime, folder) -> None:
    """A readable file inside a skipped folder is still left out."""
    inside = repo / "fund/src/fund" / folder
    inside.mkdir(parents=True, exist_ok=True)
    (inside / "note.md").write_text("hidden")

    root = _staged(repo, runtime)

    assert not (root / "code/fund" / folder / "note.md").exists()


def test_a_restage_that_fails_halfway_is_redone_next_time(
    repo, runtime, monkeypatch
) -> None:
    """A half-copied workspace is never mistaken for a finished one."""
    import os

    target = repo / "fund/src/fund/observe.py"
    original = target.read_text()
    root = _staged(repo, runtime)
    before = target.stat()
    target.write_text("def changed():\n    return 2\n")

    def failing_copy(src, dst, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ask_agent.shutil, "copyfile", failing_copy)
    with pytest.raises(OSError):
        _staged(repo, runtime)
    monkeypatch.undo()
    # The sources go back to exactly what the finished stage saw, so only the marker can
    # tell a finished workspace from the half-copied one.
    target.write_text(original)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))

    root = _staged(repo, runtime)

    assert (root / "code/fund/observe.py").read_text() == original


def test_the_real_source_mapping_still_matches_the_real_tree() -> None:
    """The copy list names paths that exist and stages only readable text."""
    from fund.agents.backend import _REPO_ROOT

    files = ask_agent._source_files(_REPO_ROOT)
    names = [dest for dest, _ in files]

    assert len(files) > 100
    assert "docs/README.md" in names
    assert "code/fund/observe.py" in names
    assert all(Path(n).suffix in ask_agent._SUFFIXES for n in names)
    assert not any(
        part in ask_agent._SKIPPED_DIRS for n in names for part in Path(n).parts
    )


def test_the_workspace_folders_are_private_to_the_owner(repo, runtime) -> None:
    """Offloaded history holds portfolio details, so no other user may read it."""
    import stat

    _, agent = _agent(repo, runtime, [AIMessage("ok")], "A")
    run_ask_turn(agent, scope="A", question="hi")
    root = runtime / "ask-fs"

    for folder in (root, root / "asks", ask_agent.scope_scratch_dir(root, "A")):
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700


# ------------------------------------------------------ review hardening: the agent


def test_a_delegated_helper_cannot_run_past_its_call_cap(repo, runtime) -> None:
    """The ``task`` helper is bounded, so one turn cannot spend thousands of calls."""
    loop = [tool_call("ls", {"path": "/"}, f"c{i}") for i in range(60)]
    model, agent = _agent(
        repo,
        runtime,
        [
            tool_call(
                "task", {"description": "dig", "subagent_type": "general-purpose"}
            ),
            *loop,
        ],
    )

    run_ask_turn(agent, scope="fund", question="delegate", max_steps=3)

    assert len(model.seen) <= 14


def test_every_write_is_denied_not_only_under_code(repo, runtime) -> None:
    """The scratch folder is read-only to the agent too."""
    root = _staged(repo, runtime)
    _, agent = _agent(
        repo,
        runtime,
        [
            tool_call("write_file", {"file_path": "/notes.md", "content": "x"}),
            tool_call(
                "write_file",
                {"file_path": "/conversation_history/x.md", "content": "x"},
                "c2",
            ),
            AIMessage("done"),
        ],
    )

    run_ask_turn(agent, scope="fund", question="write")

    results = " ".join(_tool_results(agent)).lower()
    assert results.count("permission denied") == 2
    scratch = ask_agent.scope_scratch_dir(root, "fund")
    assert not (scratch / "notes.md").exists()
    assert not (scratch / "conversation_history/x.md").exists()


def test_a_command_is_refused_and_nothing_is_run(repo, runtime) -> None:
    """``execute`` is answered with a refusal and leaves no file behind."""
    _, agent = _agent(
        repo,
        runtime,
        [tool_call("execute", {"command": "touch ran.txt"}), AIMessage("done")],
    )

    run_ask_turn(agent, scope="fund", question="run it")

    assert "not available" in " ".join(_tool_results(agent)).lower()
    assert not list(runtime.rglob("ran.txt"))


@pytest.mark.parametrize(
    "path", ["/code/fund/a..b.py", "/code/fund/x\x00y.py", "/docs/..hidden"]
)
def test_an_odd_path_is_answered_with_an_error_not_a_crash(repo, runtime, path) -> None:
    """A model-chosen path the backend rejects ends in a message, not a failed turn."""
    _, agent = _agent(
        repo,
        runtime,
        [tool_call("read_file", {"file_path": path}), AIMessage("done")],
    )

    answer = run_ask_turn(agent, scope="fund", question="read it")

    assert answer == "done"
    assert any("error" in r.lower() for r in _tool_results(agent))


def test_a_finished_answer_is_kept_even_if_the_deadline_passed_meanwhile() -> None:
    """The last model call may end past the limit; its answer is still delivered."""
    agent = _FakeAgent([{"model": {"messages": [_ai("the complete answer")]}}])
    ticks = iter([0.0, 500.0, 500.0])

    answer = run_ask_turn(
        agent,
        scope="fund",
        question="q",
        timeout_seconds=100.0,
        clock=lambda: next(ticks),
    )

    assert answer == "the complete answer"


def test_a_message_with_text_and_a_tool_call_is_not_the_answer() -> None:
    """Thinking aloud before a tool call does not count as the final answer."""
    agent = _FakeAgent(
        [
            {
                "model": {
                    "messages": [
                        _ai(
                            "let me check",
                            [{"name": "ls", "args": {}, "id": "1"}],
                        )
                    ]
                }
            }
        ]
    )

    assert run_ask_turn(agent, scope="fund", question="q").startswith("(the agent")


def test_a_zero_step_cap_still_allows_one_round() -> None:
    """A mistaken ``0`` never turns into an invalid recursion limit."""
    agent = _FakeAgent([{"model": {"messages": [_ai("hi")]}}])

    run_ask_turn(agent, scope="fund", question="q", max_steps=0)

    assert agent.config["recursion_limit"] == 3


# ------------------------------------------------------- review hardening: clearing


def test_clearing_reports_a_scratch_folder_that_could_not_be_removed(
    repo, runtime
) -> None:
    """A clear that left portfolio details on disk raises, not claims success."""
    saver = MemorySaver()
    _, agent = _agent(repo, runtime, [AIMessage("a")], "A", saver)
    run_ask_turn(agent, scope="A", question="q")
    root = _staged(repo, runtime)
    scratch = ask_agent.scope_scratch_dir(root, "A")
    history = scratch / "conversation_history"
    history.mkdir(parents=True, exist_ok=True)
    (history / "session.md").write_text("portfolio details")
    history.chmod(0o555)
    try:
        with pytest.raises(OSError):
            clear_ask_state(saver, "A", runtime_root=runtime)
    finally:
        history.chmod(0o755)


def test_clearing_one_scope_leaves_the_other_scopes_files(repo, runtime) -> None:
    """Each scope's offloaded history is its own to delete."""
    saver = MemorySaver()
    _, agent_a = _agent(repo, runtime, [AIMessage("a")], "A", saver)
    _, agent_b = _agent(repo, runtime, [AIMessage("b")], "B", saver)
    run_ask_turn(agent_a, scope="A", question="qa")
    run_ask_turn(agent_b, scope="B", question="qb")
    root = _staged(repo, runtime)
    keep = ask_agent.scope_scratch_dir(root, "B") / "keep.md"
    keep.write_text("B's history")

    clear_ask_state(saver, "A", runtime_root=runtime)

    assert keep.exists()
    assert not ask_agent.scope_scratch_dir(root, "A").exists()


# ------------------------------------------------------- review hardening: the tools


def test_the_data_tools_pass_their_arguments_through(monkeypatch) -> None:
    """Each tool calls the right reader with the right portfolio and bounds."""
    calls: dict[str, tuple] = {}

    def recorder(name, result=None):
        """Build a stand-in reader that remembers how it was called."""

        def fake(*args, **kwargs):
            """Record the call and return a small result."""
            calls[name] = (args[1:], kwargs)
            return result if result is not None else {"ok": name}

        return fake

    tools = _tools(
        monkeypatch,
        list_portfolios=lambda session: [PID],
        portfolio_state=recorder("state"),
        holdings_summary=recorder("holdings", []),
        list_portfolio_runs=recorder("runs", []),
        audit_timeline=recorder("audit", []),
        pending_hitl=recorder("gates", []),
        get_mandate=recorder("mandate"),
        mandate_warnings=recorder("warnings", []),
    )

    tools["portfolio_state"].invoke({})
    tools["holdings"].invoke({})
    tools["list_runs"].invoke({"limit": 500})
    tools["audit_trail"].invoke({"limit": 500})
    tools["pending_gates"].invoke({})
    tools["mandate"].invoke({})

    assert calls["state"][0] == (PID,)
    assert calls["holdings"][0] == (PID,)
    assert calls["runs"][1] == {"limit": 50}
    assert calls["audit"][1] == {"portfolio_id": PID}
    assert calls["gates"][0][1:] == (None,)
    assert calls["mandate"][0] == (PID,)
    assert calls["warnings"][0] == (PID,)


def test_a_blank_audit_scope_with_no_focus_means_every_portfolio(monkeypatch) -> None:
    """With nothing focused, an empty id reads the whole fund's audit trail."""
    from fund import observe

    seen: list[Any] = []
    monkeypatch.setattr(
        observe,
        "audit_timeline",
        lambda session, *, portfolio_id=None, **kw: seen.append(portfolio_id) or [],
    )
    tools = {
        t.name: t
        for t in ask_agent.build_ask_tools(
            session_factory=_no_session, saver=MemorySaver(), focus=None
        )
    }

    tools["audit_trail"].invoke({"portfolio_id": "  "})

    assert seen == [None]


def test_an_unknown_run_and_a_bad_date_are_reported_as_text(monkeypatch) -> None:
    """Bad ids and dates the model invents come back as ``error:`` lines."""
    tools = _tools(monkeypatch)

    assert tools["run_transcript"].invoke({"run_id": "not-a-uuid"}).startswith("error:")
    assert tools["daily_brief"].invoke({"day": "11/09/2026"}).startswith("error:")


def test_enum_and_numpy_values_are_rendered_as_values() -> None:
    """Readers' enums and numpy numbers become their values, not ``{}`` or strings."""
    import enum

    import numpy as np

    class Mode(enum.Enum):
        """A plain, non-string enum."""

        ON = "on"

    text = ask_agent._render(
        {"mode": Mode.ON, "count": np.int64(5), "x": np.float64(0.5)}
    )

    assert '"mode": "on"' in text
    assert '"count": 5' in text
    assert '"x": 0.5' in text


def test_the_tools_ask_postgres_for_a_read_only_transaction() -> None:
    """On Postgres each read declares itself read-only, whatever the account can do."""
    executed: list[str] = []

    class FakeSession:
        """A session that looks like Postgres and records what it is asked to run."""

        def get_bind(self):
            """Report a Postgres dialect."""
            return SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        def execute(self, statement):
            """Remember the statement text."""
            executed.append(str(statement))

        def rollback(self):
            """Nothing to undo."""

    @contextmanager
    def session_factory():
        """Hand out the fake session."""
        yield FakeSession()

    from fund import observe

    observe_original = observe.list_portfolios
    observe.list_portfolios = lambda session: [PID]
    try:
        tools = {
            t.name: t
            for t in ask_agent.build_ask_tools(
                session_factory=session_factory, saver=MemorySaver(), focus=PID
            )
        }
        tools["list_portfolios"].invoke({})
    finally:
        observe.list_portfolios = observe_original

    assert executed == ["SET TRANSACTION READ ONLY"]


# ----------------------------------------------- findings left open in the first pass


def test_a_file_with_a_private_key_inside_is_not_staged(repo, runtime) -> None:
    """A secret value in an innocently named file still keeps the file out."""
    (repo / "fund/src/fund/notes.md").write_text(
        "deploy notes\n-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n"
    )
    (repo / "fund/src/fund/tokens_doc.md").write_text("AKIAABCDEFGHIJKLMNOP")
    (repo / "fund/src/fund/plain.md").write_text("nothing secret here")

    root = _staged(repo, runtime)

    assert not (root / "code/fund/notes.md").exists()
    assert (root / "code/fund/plain.md").exists()


def test_a_tampered_workspace_is_restored_on_the_next_question(repo, runtime) -> None:
    """A planted or changed file in the staged copy is not trusted just because the
    source did not change."""
    root = _staged(repo, runtime)
    (root / "code/fund/planted.md").write_text("planted")
    (root / "code/fund/observe.py").write_text("tampered")

    root = _staged(repo, runtime)

    assert not (root / "code/fund/planted.md").exists()
    assert "def hello" in (root / "code/fund/observe.py").read_text()


def test_a_conversation_about_one_portfolio_cannot_read_another(monkeypatch) -> None:
    """With a portfolio focused, tools refuse every other portfolio's data."""
    other = uuid.UUID("99999999-9999-9999-9999-999999999999")
    tools = _tools(monkeypatch, list_portfolios=lambda session: [PID, other])

    text = tools["performance"].invoke({"portfolio_id": str(other)})

    assert text.startswith("error:")
    assert "about portfolio" in text


def test_the_portfolio_list_of_a_focused_conversation_is_only_that_portfolio(
    monkeypatch,
) -> None:
    """Other portfolios' ids are not even listed."""
    other = uuid.UUID("99999999-9999-9999-9999-999999999999")
    tools = _tools(monkeypatch, list_portfolios=lambda session: [PID, other])

    assert tools["list_portfolios"].invoke({}) == f'["{PID}"]'


def test_pending_gates_of_a_focused_conversation_are_only_its_own(monkeypatch) -> None:
    """A gate that waits on another portfolio is not shown."""
    other = uuid.UUID("99999999-9999-9999-9999-999999999999")
    gates = [
        SimpleNamespace(portfolio_id=PID, run_id="mine"),
        SimpleNamespace(portfolio_id=other, run_id="theirs"),
    ]
    tools = _tools(monkeypatch, pending_hitl=lambda session, saver, pid=None: gates)

    text = tools["pending_gates"].invoke({})

    assert "mine" in text
    assert "theirs" not in text


def test_a_run_of_another_portfolio_cannot_be_opened(monkeypatch) -> None:
    """A run id from another portfolio is refused in a focused conversation."""
    other = uuid.UUID("99999999-9999-9999-9999-999999999999")
    run_id = uuid.uuid4()

    class FakeRepo:
        """Returns a run that belongs to a different portfolio."""

        def __init__(self, session):
            pass

        def get_run(self, rid):
            """Hand back a run of the other portfolio."""
            return SimpleNamespace(id=rid, portfolio_id=other)

    import fund.audit.repository as repository

    monkeypatch.setattr(repository, "AgentRunRepository", FakeRepo)
    tools = _tools(monkeypatch)

    text = tools["run_transcript"].invoke({"run_id": str(run_id)})

    assert text.startswith("error:")


def test_a_database_error_is_shown_without_its_statement_and_parameters() -> None:
    """The text that reaches the model, the screen and the history is one safe line."""
    from sqlalchemy.exc import DBAPIError

    class DriverError(Exception):
        """A stand-in for the driver's own error."""

    exc = DBAPIError(
        "SELECT secret FROM clients WHERE token = %s", ("hunter2",), DriverError("boom")
    )

    shown = ask_agent.describe_error(exc)

    assert "hunter2" not in shown
    assert "SELECT" not in shown
    assert "boom" in shown


def test_an_error_with_control_characters_or_no_message_is_shown_safely() -> None:
    """Control codes go, a blank message uses the error name, length is capped."""
    assert ask_agent.describe_error(ValueError("bad \x1b[2J input")) == "bad [2J input"
    assert ask_agent.describe_error(ValueError()) == "ValueError"
    assert len(ask_agent.describe_error(ValueError("x" * 5000))) < 400


def test_a_conversations_checkpoints_are_never_pruned(repo, runtime) -> None:
    """Old checkpoints hold what the latest one is built from, so they stay.

    Pruning them to save space was tried and wiped the agent's memory on Postgres: the
    latest checkpoint no longer carried the conversation. The thread only shrinks when
    the adviser clears it.
    """

    class NoPruneSaver(MemorySaver):
        """A saver that fails the test if anything tries to prune."""

        def prune(self, thread_ids, *, strategy="keep_latest"):
            """Pruning is never allowed."""
            raise AssertionError("the Ask thread must not be pruned")

    _, agent = _agent(
        repo, runtime, [AIMessage("a"), AIMessage("b")], "A", NoPruneSaver()
    )

    run_ask_turn(agent, scope="A", question="one")
    run_ask_turn(agent, scope="A", question="two")

    thread = {"configurable": {"thread_id": ask_thread_id("A")}}
    assert len(agent.get_state(thread).values["messages"]) == 4


def test_a_follow_up_reaches_the_model_with_the_earlier_words(repo, runtime) -> None:
    """The second call carries the first question and its answer, verbatim."""
    model, agent = _agent(
        repo, runtime, [AIMessage("first answer"), AIMessage("second")]
    )

    run_ask_turn(agent, scope="fund", question="how are we doing?")
    run_ask_turn(agent, scope="fund", question="and the second one?")

    second_call = model.texts[1]
    assert "how are we doing?" in second_call
    assert "first answer" in second_call
    assert "and the second one?" in second_call


def test_a_looping_model_is_stopped_after_exactly_the_step_cap(repo, runtime) -> None:
    """``max_steps`` model calls, then a plain message; the next turn still works."""
    loop = [tool_call("ls", {"path": "/"}, f"c{i}") for i in range(40)]
    model, agent = _agent(repo, runtime, loop)

    answer = run_ask_turn(agent, scope="fund", question="loop", max_steps=4)

    assert len(model.seen) == 4
    assert "narrower question" in answer
    model.script = [AIMessage("fine now")]
    model.reset()
    assert run_ask_turn(agent, scope="fund", question="again") == "fine now"


def test_a_slow_model_stops_the_turn_at_the_time_limit(repo, runtime) -> None:
    """Between steps, a turn past its limit raises instead of carrying on."""
    import time

    class SlowModel(ScriptedAskModel):
        """Takes a moment on every call."""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            time.sleep(0.4)
            return super()._generate(messages, stop, run_manager, **kwargs)

    model = SlowModel(
        script=[tool_call("ls", {"path": "/"}, f"c{i}") for i in range(10)]
    )
    agent = build_ask_agent(
        model=model,
        session_factory=_no_session,
        persistence=SimpleNamespace(saver=MemorySaver(), store=InMemoryStore()),
        scope="fund",
        focus=None,
        config=FundConfig(),
        repo_root=repo,
        runtime_root=runtime,
    )

    with pytest.raises(AskTimeoutError):
        run_ask_turn(agent, scope="fund", question="slow", timeout_seconds=0.5)


def test_compaction_offloads_history_into_the_scope_and_clearing_removes_it(
    repo, runtime
) -> None:
    """With a small compaction trigger, the evicted history lands in that scope's
    scratch folder, and a clear removes it without touching another scope."""
    config = FundConfig(summary_token_threshold=300, summary_messages_to_keep=2)
    saver = MemorySaver()
    model = ScriptedAskModel(
        script=[AIMessage(f"answer number {i} " + "word " * 80) for i in range(12)]
    )

    def build(scope):
        return build_ask_agent(
            model=model,
            session_factory=_no_session,
            persistence=SimpleNamespace(saver=saver, store=InMemoryStore()),
            scope=scope,
            focus=None,
            config=config,
            repo_root=repo,
            runtime_root=runtime,
        )

    agent_a, agent_b = build("A"), build("B")
    for index in range(5):
        run_ask_turn(agent_a, scope="A", question=f"question {index} " + "pad " * 80)
    run_ask_turn(agent_b, scope="B", question="only one")
    root = _staged(repo, runtime)
    offloaded = list(
        (ask_agent.scope_scratch_dir(root, "A") / "conversation_history").glob("*.md")
    )

    assert offloaded, "compaction never offloaded anything"
    clear_ask_state(saver, "A", runtime_root=runtime)
    assert not ask_agent.scope_scratch_dir(root, "A").exists()
    assert ask_agent.scope_scratch_dir(root, "B").exists()


def test_a_reply_made_of_text_blocks_keeps_its_paragraph_breaks() -> None:
    """Providers that return content blocks do not run paragraphs together."""
    blocks = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]

    assert ask_agent._text(blocks) == "first\n\nsecond"
    assert (
        ask_agent._text(["plain", {"type": "text", "text": "more"}]) == "plain\n\nmore"
    )


def test_each_step_phrase_covers_every_kind_of_tool() -> None:
    """The status line has plain words for listing, finding, planning and delegating."""
    assert ask_agent.describe_step("ls", {"path": "/docs"}) == "listing /docs"
    assert ask_agent.describe_step("ls", {}) == "listing /"
    assert ask_agent.describe_step("glob", {"pattern": "*.py"}) == "finding files *.py"
    assert ask_agent.describe_step("write_todos", {}) == "planning"
    assert ask_agent.describe_step("task", {}) == "asking a helper agent"
    assert "not allowed" in ask_agent.describe_step("write_file", {"file_path": "/x"})


def test_a_stream_chunk_that_is_not_a_dict_is_ignored() -> None:
    """Unexpected chunk shapes never break a turn."""
    assert list(ask_agent._update_messages(["not", "a", "dict"])) == []
    assert list(ask_agent._update_messages({"node": None})) == []
    assert list(ask_agent._update_messages({"node": {"messages": "text"}})) == []


def test_all_eleven_tools_work_through_the_real_agent(
    repo, runtime, monkeypatch
) -> None:
    """Every tool is callable by the model with its documented arguments."""
    from fund import observe

    quiet = lambda *a, **k: {"ok": True}  # noqa: E731
    for name in (
        "portfolio_state",
        "performance_summary",
        "holdings_summary",
        "get_mandate",
        "mandate_warnings",
        "list_portfolio_runs",
        "daily_brief",
    ):
        monkeypatch.setattr(observe, name, quiet)
    monkeypatch.setattr(observe, "audit_timeline", lambda *a, **k: [])
    monkeypatch.setattr(observe, "list_portfolios", lambda session: [PID])
    monkeypatch.setattr(observe, "pending_hitl", lambda session, saver, pid=None: [])
    names = [
        "fund_snapshot",
        "list_portfolios",
        "portfolio_state",
        "performance",
        "holdings",
        "mandate",
        "pending_gates",
        "list_runs",
        "daily_brief",
        "audit_trail",
    ]
    script = [tool_call(n, {}, f"c{i}") for i, n in enumerate(names)] + [
        AIMessage("ok")
    ]
    model = ScriptedAskModel(script=script)
    agent = build_ask_agent(
        model=model,
        session_factory=_no_session,
        persistence=SimpleNamespace(saver=MemorySaver(), store=InMemoryStore()),
        scope="fund",
        focus=PID,
        config=FundConfig(),
        repo_root=repo,
        runtime_root=runtime,
    )
    from fund.tui import ask as ask_module

    monkeypatch.setattr(ask_module, "build_context", lambda *a, **k: "FUND STATE")

    run_ask_turn(agent, scope="fund", question="use everything", max_steps=40)

    results = _tool_results(agent)
    assert len(results) == len(names)
    assert not [r for r in results if str(r).startswith("error:")]


def test_an_edit_that_keeps_the_file_size_is_still_picked_up(repo, runtime) -> None:
    """Same length, new content: the change is noticed through the modified time."""
    import os

    target = repo / "fund/src/fund/observe.py"
    root = _staged(repo, runtime)
    text = target.read_text()
    stat = target.stat()
    target.write_text(text.replace("hi", "ho"))
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

    root = _staged(repo, runtime)

    assert "return 'ho'" in (root / "code/fund/observe.py").read_text()


def test_a_tool_message_is_never_taken_as_the_answer() -> None:
    """Only the model's own text counts, not what a tool returned."""
    agent = _FakeAgent(
        [
            {
                "tools": {
                    "messages": [ToolMessage(content="tool output", tool_call_id="1")]
                }
            }
        ]
    )

    assert run_ask_turn(agent, scope="fund", question="q").startswith("(the agent")
