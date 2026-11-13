"""How ``run_fund`` and ``stream_fund`` close a round, call by call.

Validates the endings of ``fund/src/fund/agents/graph.py`` (``run_fund`` and
``stream_fund``) once the PM agent stops: a round that never reached the gate is
finalised ``incomplete`` with one of four reasons, a round a freeze landed on is
halted ``rejected``, and any other round is flipped ``paused`` at the gate. For each
ending it pins the exact audit calls and their order, what ``run_fund`` returns and
the exact terminal event ``stream_fund`` yields, and that a freeze is looked at only
once the gate was reached. The round's setup and the database are replaced by fakes;
``test_fund_run.py`` and ``test_stream.py`` drive the same endings end to end. No
requirement id is registered for the fund round yet.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.errors import GraphRecursionError

from fund.agents import graph
from fund.tui.stream import StepStatus, StreamEvent

PID = uuid.UUID("11111111-2222-3333-4444-555555555555")
RUN = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
WEIGHTS = {"AAA": 0.6, "BBB": 0.4}
PAYLOAD = {"action_requests": [{"name": "place_orders", "description": "buy AAA"}]}
CONSTRAINTS = SimpleNamespace(name="constraint set")
SESSION = SimpleNamespace(name="session")
SAVER = SimpleNamespace(name="checkpointer")
THREAD = {"configurable": {"thread_id": str(RUN)}}
HALTED = "halted: autonomy was frozen while the round was running"

# Each ungated ending: the PM's last message, whether the round cap was reached,
# the reason recorded and the text the stream shows.
UNGATED = [
    (
        SimpleNamespace(),
        True,
        "pm_round_cap",
        "delegation round cap reached before a ticket",
    ),
    (
        SimpleNamespace(response_metadata={"stop_reason": "max_tokens"}),
        False,
        "output_truncated",
        "the model's output limit cut off the PM's order call",
    ),
    (
        SimpleNamespace(invalid_tool_calls=[{"name": "place_orders"}]),
        False,
        "invalid_tool_call",
        "the PM's order call was malformed",
    ),
    (SimpleNamespace(), False, "no_order", "pipeline ended before the gate"),
]


class _Audit:
    """Logs the audit calls a round ending makes, in order."""

    def __init__(self, log: list[Any]) -> None:
        self._log = log

    def append_decision(self, run_id: uuid.UUID, **fields: Any) -> None:
        """Record a decision row."""
        self._log.append(("append_decision", run_id, fields))

    def finalize_run(self, run_id: uuid.UUID, **fields: Any) -> None:
        """Record a finalisation."""
        self._log.append(("finalize_run", run_id, fields))

    def mark_paused(self, run_id: uuid.UUID) -> None:
        """Record the flip to paused."""
        self._log.append(("mark_paused", run_id))


class _Agent:
    """Answers ``invoke`` with one result and ``stream`` with fixed chunks."""

    def __init__(self, result: Any = None, chunks: list[Any] | None = None) -> None:
        self.result = result
        self.chunks = chunks or []

    def invoke(self, seed_input: Any, *, config: Any) -> Any:
        """Return the result, or raise it when it is an exception."""
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    def stream(
        self, seed_input: Any, *, config: Any, stream_mode: str
    ) -> Iterator[Any]:
        """Yield the chunks; an exception among them is raised in its place."""
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


@pytest.fixture
def log() -> list[Any]:
    """The ordered calls the round ending makes."""
    return []


def _patch_round(
    monkeypatch: pytest.MonkeyPatch,
    log: list[Any],
    agent: _Agent,
    *,
    frozen: str | None = None,
) -> SimpleNamespace:
    """Replace the setup, the weight read and the freeze lookup with logging fakes."""
    setup = graph._FundRunSetup(
        agent=agent,
        thread_config=THREAD,
        run=SimpleNamespace(id=RUN),
        constraint_set=CONSTRAINTS,  # type: ignore[arg-type]
        seed_input={"messages": []},
        audit=_Audit(log),
        ctx=SimpleNamespace(portfolio_id=PID),
        first_round=False,
    )

    def weights(session: Any, run_id: uuid.UUID) -> dict[str, float]:
        """The optimizer weights of the run."""
        log.append(("optimizer_weights", session, run_id))
        return dict(WEIGHTS)

    def frozen_scope(session: Any, portfolio_id: uuid.UUID) -> str | None:
        """The frozen scope over the portfolio, if any."""
        log.append(("frozen_scope", session, portfolio_id))
        return frozen

    monkeypatch.setattr(graph, "_prepare_fund_run", lambda *a, **k: setup)
    monkeypatch.setattr(graph, "_optimizer_weights", weights)
    monkeypatch.setattr(graph, "_frozen_scope", frozen_scope)
    return setup


def _run(portfolio_id: uuid.UUID | str = PID) -> graph.FundRun:
    return graph.run_fund(
        object(),
        object(),  # type: ignore[arg-type]
        portfolio_id=portfolio_id,
        asof=None,  # type: ignore[arg-type]
        session=SESSION,
        checkpointer=SAVER,
        store=object(),
    )


def _stream(portfolio_id: uuid.UUID | str = PID) -> list[StreamEvent]:
    return list(
        graph.stream_fund(
            object(),
            object(),  # type: ignore[arg-type]
            portfolio_id=portfolio_id,
            asof=None,  # type: ignore[arg-type]
            session=SESSION,
            checkpointer=SAVER,
            store=object(),
        )
    )


def _summary(run: graph.FundRun, setup: SimpleNamespace) -> tuple[Any, ...]:
    """Every field of the handle, the uncompared ones checked by identity."""
    assert run.agent is setup.agent
    assert run.thread_config is setup.thread_config
    assert run.session is SESSION
    assert run.ctx is setup.ctx
    return (run.constraint_set, run.run_id, run.interrupt, run.weights, run.status)


def _incomplete(reason: str) -> list[Any]:
    return [
        (
            "append_decision",
            RUN,
            {
                "agent": "orchestrator",
                "step": "place_orders",
                "hitl_decision": {"decision": "incomplete", "reason": reason},
            },
        ),
        ("finalize_run", RUN, {"weights": {}, "status": "incomplete"}),
    ]


def _halted(scope: str) -> list[Any]:
    return [
        ("frozen_scope", SESSION, PID),
        (
            "append_decision",
            RUN,
            {
                "agent": "orchestrator",
                "step": "place_orders",
                "hitl_decision": {
                    "decision": "halted",
                    "reason": "frozen",
                    "scope": scope,
                },
            },
        ),
        ("finalize_run", RUN, {"weights": {}, "status": "rejected"}),
    ]


def _terminal(kind: str, index: int, text: str, payload: Any = None) -> StreamEvent:
    return StreamEvent(
        kind=kind,
        run_id=RUN,
        index=index,
        agent="orchestrator",
        step="place_orders",
        status=StepStatus.PENDING,
        text=text,
        payload=payload,
    )


def _gate_chunk(payload: Any = PAYLOAD) -> dict[str, Any]:
    return {"__interrupt__": [SimpleNamespace(value=payload)]}


def _model_chunk(message: Any, **state: Any) -> dict[str, Any]:
    return {"model": {"messages": [message], **state}}


# ---------------------------------------------------------------------- run_fund


def test_run_fund_pauses_at_the_gate_with_the_optimizer_weights(
    monkeypatch, log
) -> None:
    """The weights are read, the freeze is checked, then the run is flipped paused."""
    setup = _patch_round(monkeypatch, log, _Agent({**_gate_chunk(), "messages": []}))

    run = _run()

    assert _summary(run, setup) == (CONSTRAINTS, RUN, PAYLOAD, WEIGHTS, "paused")
    assert log == [
        ("optimizer_weights", SESSION, RUN),
        ("frozen_scope", SESSION, PID),
        ("mark_paused", RUN),
    ]


@pytest.mark.parametrize("scope", ["portfolio", "fund"])
def test_run_fund_halts_a_round_a_freeze_landed_on(monkeypatch, log, scope) -> None:
    """A gate reached under a freeze is rejected with no weights and no interrupt."""
    setup = _patch_round(monkeypatch, log, _Agent(_gate_chunk()), frozen=scope)

    run = _run()

    assert _summary(run, setup) == (CONSTRAINTS, RUN, None, {}, "rejected")
    assert log == [("optimizer_weights", SESSION, RUN), *_halted(scope)]


@pytest.mark.parametrize(("message", "capped", "reason", "text"), UNGATED)
def test_run_fund_finalises_an_ungated_round_incomplete_with_its_reason(
    monkeypatch, log, message, capped, reason, text
) -> None:
    """No gate: the reason is recorded and the weights read are still returned."""
    result = {"messages": [SimpleNamespace(), message], "pm_round_cap_reached": capped}
    setup = _patch_round(monkeypatch, log, _Agent(result))

    run = _run()

    assert _summary(run, setup) == (CONSTRAINTS, RUN, None, WEIGHTS, "incomplete")
    assert log == [("optimizer_weights", SESSION, RUN), *_incomplete(reason)]


def test_run_fund_does_not_look_for_a_freeze_when_no_gate_was_reached(
    monkeypatch, log
) -> None:
    """An ungated round ends incomplete even under a freeze."""
    setup = _patch_round(monkeypatch, log, _Agent({"messages": []}), frozen="fund")

    run = _run()

    assert _summary(run, setup) == (CONSTRAINTS, RUN, None, WEIGHTS, "incomplete")
    assert log == [("optimizer_weights", SESSION, RUN), *_incomplete("no_order")]


def test_run_fund_treats_a_gate_without_a_payload_dict_as_no_gate(
    monkeypatch, log
) -> None:
    """An interrupt whose value is not a dict opens no gate."""
    setup = _patch_round(monkeypatch, log, _Agent(_gate_chunk("not a dict")))

    run = _run()

    assert _summary(run, setup) == (CONSTRAINTS, RUN, None, WEIGHTS, "incomplete")
    assert log == [("optimizer_weights", SESSION, RUN), *_incomplete("no_order")]


def test_run_fund_reads_the_freeze_of_the_portfolio_given_as_text(
    monkeypatch, log
) -> None:
    """A portfolio id passed as a string is looked up as its UUID."""
    _patch_round(monkeypatch, log, _Agent(_gate_chunk()))

    assert _run(portfolio_id=str(PID)).status == "paused"
    assert ("frozen_scope", SESSION, PID) in log


def test_run_fund_ends_a_round_past_the_recursion_limit_incomplete(
    monkeypatch, log
) -> None:
    """The round cap is recorded and no weights are read."""
    setup = _patch_round(monkeypatch, log, _Agent(GraphRecursionError("cap")))

    run = _run()

    assert _summary(run, setup) == (CONSTRAINTS, RUN, None, {}, "incomplete")
    assert log == _incomplete("round_cap")


# ------------------------------------------------------------------- stream_fund


def _delegation_chunks() -> list[dict[str, Any]]:
    """A PM delegation to the economist and its report: one step event."""
    call = {"name": "task", "id": "c1", "args": {"subagent_type": "economist"}}
    report = SimpleNamespace(name="task", tool_call_id="c1", content="report")
    return [
        _model_chunk(SimpleNamespace(tool_calls=[call])),
        {"tools": {"messages": [report]}},
    ]


def test_stream_fund_pauses_at_the_gate_with_its_why(monkeypatch, log) -> None:
    """The last event is the paused gate, carrying the payload and its description."""
    _patch_round(
        monkeypatch, log, _Agent(chunks=[*_delegation_chunks(), _gate_chunk()])
    )

    events = _stream()

    assert events[0] == StreamEvent(
        kind="step",
        run_id=RUN,
        index=0,
        agent="economist",
        step="report",
        status=StepStatus.LIVE,
        text="report",
    )
    assert events[1:] == [_terminal("paused", 1, "buy AAA", PAYLOAD)]
    assert log == [("frozen_scope", SESSION, PID), ("mark_paused", RUN)]


def test_stream_fund_gives_a_gate_without_a_description_a_default_why(
    monkeypatch, log
) -> None:
    """A gate whose request says nothing still tells the adviser what it is for."""
    payload = {"action_requests": [{"name": "place_orders"}]}
    _patch_round(monkeypatch, log, _Agent(chunks=[_gate_chunk(payload)]))

    assert _stream() == [_terminal("paused", 0, "review before it executes", payload)]


@pytest.mark.parametrize("scope", ["portfolio", "fund"])
def test_stream_fund_halts_a_round_a_freeze_landed_on(monkeypatch, log, scope) -> None:
    """A gate reached under a freeze ends the stream incomplete, never paused."""
    _patch_round(
        monkeypatch,
        log,
        _Agent(chunks=[*_delegation_chunks(), _gate_chunk()]),
        frozen=scope,
    )

    events = _stream()

    assert events[1:] == [_terminal("incomplete", 1, HALTED)]
    assert log == _halted(scope)


@pytest.mark.parametrize(("message", "capped", "reason", "text"), UNGATED)
def test_stream_fund_ends_an_ungated_round_incomplete_with_its_reason(
    monkeypatch, log, message, capped, reason, text
) -> None:
    """No gate: the reason is recorded and the adviser is told why in words."""
    chunks = [_model_chunk(message, pm_round_cap_reached=capped)]
    _patch_round(monkeypatch, log, _Agent(chunks=chunks), frozen="fund")

    assert _stream() == [_terminal("incomplete", 0, text)]
    assert log == _incomplete(reason)


def test_stream_fund_judges_the_reason_by_the_last_pm_message(monkeypatch, log) -> None:
    """A later tools update does not replace the PM's last message."""
    truncated = SimpleNamespace(response_metadata={"stop_reason": "length"})
    chunks = [
        _model_chunk(truncated),
        {"tools": {"messages": [SimpleNamespace(name="other", content="x")]}},
    ]
    _patch_round(monkeypatch, log, _Agent(chunks=chunks))

    events = _stream()

    assert events[-1].text == "the model's output limit cut off the PM's order call"
    assert log == _incomplete("output_truncated")


def test_stream_fund_ends_a_round_past_the_recursion_limit_incomplete(
    monkeypatch, log
) -> None:
    """The round cap is recorded and the stream ends where the steps stopped."""
    chunks = [*_delegation_chunks(), GraphRecursionError("cap")]
    _patch_round(monkeypatch, log, _Agent(chunks=chunks))

    events = _stream()

    assert events[1:] == [
        _terminal("incomplete", 1, "round cap reached before a ticket")
    ]
    assert log == _incomplete("round_cap")


def test_stream_fund_never_reads_the_optimizer_weights(monkeypatch, log) -> None:
    """The stream reports the gate; the weights are read only by ``run_fund``."""
    _patch_round(monkeypatch, log, _Agent(chunks=[_gate_chunk()]))

    _stream(portfolio_id=str(PID))

    assert not [call for call in log if call[0] == "optimizer_weights"]
    assert ("frozen_scope", SESSION, PID) in log
