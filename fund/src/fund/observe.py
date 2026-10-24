"""``fund.observe`` — the shared, model-free read model (Phase 8, Task 5).

The single read model both frontends use: the CLI ``status``/``report`` commands and
the TUI's four panels are thin wrappers over these functions. It constructs **no**
LLM and imports **no** agent stack at module load — ``import fund.observe`` drags in
no ``deepagents`` / ``langchain`` / ``langgraph`` (asserted by a subprocess guard).
The repositories it reads through live under ``fund.audit`` (whose package
``__init__`` pulls the LangGraph persistence bootstrap), so they are imported lazily
**inside** the functions, mirroring ``fund.agents.graph``'s lazy builders — the bare
import stays clean, the call path still routes through the single-source repos.

Every function takes an injected sync ``Session`` (the caller — CLI/TUI — owns the
transaction; these read paths never ``commit``). Where a run's transcript or a
pending-HITL cross-check is needed, an already-bootstrapped LangGraph ``saver`` is
passed in and typed ``Any`` (no ``langgraph`` import at module load; the transcript
fold imports ``add_messages`` lazily).

Two sources feed the transcript (SPEC §4a): the verbatim PM/subagent turns live in
the checkpoint's ``messages`` channel (rebuilt read-only from the saver — no compiled
graph, no model), and the load-bearing structured facts live in ``agent_decisions``.
:func:`load_run_transcript` interleaves them; economist/risk narrative exists only in
``messages`` (they log no decision row), so the merge never assumes every agent has
one.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation-only import: ``fund.schemas.mandate`` is agent-stack-free, but with
    # ``from __future__ import annotations`` the return type is a string, so guarding
    # it under TYPE_CHECKING keeps the module's *runtime* imports minimal while still
    # advertising the SPEC §4a signature ``get_mandate -> PortfolioMandate | None``.
    from fund.attribution import AttributionResult
    from fund.performance import HoldingMark, PerformanceSummary
    from fund.schemas.mandate import PortfolioMandate

# LangGraph writes a pending dynamic interrupt to this reserved checkpoint channel
# (``langgraph.constants.INTERRUPT``, now private). Kept as a literal so importing
# this module never pulls in ``langgraph`` — the agent-stack-free invariant (SPEC R2).
_INTERRUPT_CHANNEL = "__interrupt__"

# ``agent_runs.status`` values at a live HITL gate — the same pair
# ``AgentRunRepository.list_paused_runs`` selects, so the drift monitor's
# already-awaiting-a-human guard covers exactly the runs a proposal may come from.
_LIVE_GATE_STATUSES = ("paused", "responded")


@dataclass(frozen=True)
class RunSummary:
    """One row for the history / status / HITL-queue panels."""

    run_id: uuid.UUID
    portfolio_id: uuid.UUID | None
    asof: dt.date
    status: str
    created_at: dt.datetime
    finished_at: dt.datetime | None
    n_weights: int
    awaiting_hitl: bool


@dataclass(frozen=True)
class TranscriptEntry:
    """A merged narrative/structured line, ordered within a run's transcript."""

    source: str  # "message" | "decision"
    agent: str
    step: str | None
    index: int
    text: str  # message content, or a rendered decision summary
    payload: dict[str, Any] | None  # decision JSON when present, else None


@dataclass(frozen=True)
class PortfolioState:
    """Panel 2: current vs target holdings, with L1 drift + best-effort metrics."""

    portfolio_id: uuid.UUID
    current: dict[str, float]
    target: dict[str, float]
    drift_l1: float
    metrics: dict[str, float]  # best-effort — empty when none persisted (never faked)
    # ``False`` when ``target`` is only an unapproved allocator proposal (or absent):
    # a never-invested book has nothing to drift from, so drift triggers must not
    # treat that proposal as the weights the portfolio is supposed to hold.
    target_approved: bool = True


@dataclass(frozen=True)
class ReprofileStatus:
    """Whether a portfolio's MiFID profile is due for renewal — annual-only (Q4).

    A derived, model-free stale marker over the append-only ``mifid_profiles`` table.
    Only the **annual** leg is wired: the active profile's age vs
    ``reprofile_interval_days`` (a portfolio never profiled is due too).
    ``"risk_drift"`` is a *reserved* reason — a persisted NAV/drawdown series now
    exists (``nav_history``, via :func:`performance_summary`), but wiring a
    drawdown-based reprofile trigger is out of scope here, so it is still never
    emitted.
    """

    portfolio_id: uuid.UUID
    due: bool
    # {"annual"} today; "risk_drift" reserved (NAV now persisted) but never emitted.
    reasons: tuple[str, ...]
    profiled_at: dt.datetime | None
    profile_age_days: int | None
    reprofile_interval_days: int


@dataclass(frozen=True)
class EventDigest:
    """Panel 5: recent global market digests + this portfolio's own rebalances.

    Model-free read model for the cockpit's Recent Events panel.
    ``market_events`` are the global ``market_journal`` rows in the trailing
    window (newest first); ``portfolio_events`` are this portfolio's
    ``portfolio_journal`` overlays at/before ``asof`` (newest first). Each entry
    is a small JSON-ready dict; both tuples are empty when nothing is recorded.
    """

    portfolio_id: uuid.UUID
    market_events: tuple[dict[str, Any], ...]
    portfolio_events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DailyBrief:
    """Per-day recap payload for one portfolio, stitched from five existing reads.

    Every section is independently optional: a day with no NAV mark, no journal row,
    or no run yields ``None`` / empty tuples (never fabricated), so the view can render
    an explicit "no data" line per section instead of crashing.

    Attributes:
        portfolio_id: The portfolio this brief is for.
        asof: The day the brief recaps (the upper date bound passed to every section).
        prev_day: The previous NAV-marked day (``←`` target), or None at the earliest
            mark.
        next_day: The next NAV-marked day (``→`` target), or None at the latest mark.
        performance: Multi-horizon returns incl. today's ``returns["1D"]`` /
            ``excess["1D"]``; None when the portfolio is unmarked on-or-before ``asof``.
        attribution: Per-factor contributions + idiosyncratic ``residual`` summing to
            ``total_return``; ``insufficient_history`` flags a too-short window.
        market_events: ``market_journal`` narrative rows in the brief's day window.
        economist_read: The economist narrative from the most recent run on-or-before
            ``asof`` (carried forward when the day had no run); None only when the
            portfolio has no run on-or-before ``asof``.
        economist_read_date: The run date ``economist_read`` came from, so the view can
            label a carried-forward read ``(from MM-DD)``; None when there is no read.
        deliberation_run_id: The run ``↵`` opens — the same most-recent-on-or-before
            run; None when the portfolio has no run on-or-before ``asof``.
        fund_events: ``portfolio_journal`` overlays at/before ``asof``, newest first.
    """

    portfolio_id: uuid.UUID
    asof: dt.date
    prev_day: dt.date | None
    next_day: dt.date | None
    performance: PerformanceSummary | None
    attribution: AttributionResult | None
    market_events: tuple[dict[str, Any], ...]
    economist_read: str | None
    economist_read_date: dt.date | None
    deliberation_run_id: uuid.UUID | None
    fund_events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AuditEntry:
    """One actor-attributed row of the Audit view's cross-portfolio timeline.

    The union of an ``agent_decisions`` row (``kind == "agent"``) and a
    ``human_audit_log`` row (``kind == "human"``) flattened to a common shape. Agent
    rows carry no provenance flags (``override_flag`` / ``suitability_flag`` are always
    ``False``); human rows carry the full provenance the detail pane renders.

    Attributes:
        when: The decision timestamp (``created_at``), the sort key.
        kind: ``"agent"`` or ``"human"`` — the actor-column split.
        actor: The agent name (agent rows) or the human actor (human rows).
        action: The decision "type": the step (agent) or the action (human).
        portfolio_id: The owning portfolio, or ``None`` for a fund-wide row.
        run_id: The run the decision belongs to, when applicable.
        prev_value: Prior value (human edits/overrides); ``None`` for agent rows.
        new_value: New value (human edits) or a compact agent-decision summary.
        source: Provenance of the row (``"agent_decisions"`` or a human source).
        idempotency_key: The human row's dedup key; ``None`` for agent rows.
        override_flag: ⚑ a guardrail edited past its MiFID band (human only).
        suitability_flag: ⚠ outside the suitability band (human only).
        target: The field, gate, or scope the human action touched (e.g. a re-arm's
            ``"fund"`` / ``"portfolio"`` scope); ``None`` for agent rows.
        rationale: The free-text reason recorded with the human action — the only
            place a fund re-arm's mandatory justification is stored; ``None`` for
            agent rows.
    """

    when: dt.datetime
    kind: str
    actor: str
    action: str
    portfolio_id: uuid.UUID | None
    run_id: uuid.UUID | None
    prev_value: Any
    new_value: Any
    source: str | None
    idempotency_key: str | None
    override_flag: bool
    suitability_flag: bool
    target: str | None = None
    rationale: str | None = None


def drift_l1(current: dict[str, float], target: dict[str, float]) -> float:
    """``Σ_i |current_i - target_i|`` over the union of tickers (D12).

    A ticker present on only one side counts as weight ``0.0`` on the other.
    """
    tickers = set(current) | set(target)
    return sum(abs(current.get(t, 0.0) - target.get(t, 0.0)) for t in tickers)


def list_portfolios(session: Any) -> list[uuid.UUID]:
    """Distinct portfolio ids known to the fund, sorted ascending.

    Unions the three places a portfolio surfaces — an ``agent_runs`` row (nullable
    fund-wide runs excluded), a ``portfolio_mandates`` row, or a ``mifid_profiles``
    row — then dedups and sorts. An empty fund yields ``[]``. The repos are imported
    lazily inside the body so ``import fund.observe`` stays agent-stack-free,
    mirroring :func:`get_mandate` / :func:`reprofile_status`.

    Args:
        session: An injected sync ``portopt_db`` session; not owned here (no commit).

    Returns:
        The sorted, de-duplicated portfolio ids; ``[]`` when the fund is empty.
    """
    from fund.audit.mandate_repository import MandateRepository
    from fund.audit.mifid_repository import MifidProfileRepository
    from fund.audit.repository import AgentRunRepository

    ids: set[uuid.UUID] = set()
    ids.update(AgentRunRepository(session).list_portfolio_ids())
    ids.update(MandateRepository(session).list_portfolio_ids())
    ids.update(MifidProfileRepository(session).list_portfolio_ids())
    return sorted(ids)


def list_portfolio_runs(
    session: Any, portfolio_id: uuid.UUID, *, limit: int | None = None
) -> list[RunSummary]:
    """Runs for a portfolio, newest first, as :class:`RunSummary` rows.

    ``awaiting_hitl`` here is the plain ``status == "paused"`` flag (the history
    panel); the HITL queue uses :func:`pending_hitl`, which additionally checks the
    checkpoint for a live interrupt.

    ``limit`` caps the rows read; callers that only need the latest run (the poll
    path's status badges) pass ``1`` so a long history is not loaded every tick.
    ``None`` returns the full history.
    """
    from fund.audit.repository import AgentRunRepository

    runs = AgentRunRepository(session).list_runs_for_portfolio(
        portfolio_id, limit=limit
    )
    return [_run_summary(run, awaiting_hitl=run.status == "paused") for run in runs]


def pending_hitl(
    session: Any, saver: Any, portfolio_id: uuid.UUID | None = None
) -> list[RunSummary]:
    """Runs awaiting a human decision, newest first.

    A row is returned only when its ``status`` is at a live gate — ``"paused"`` or
    ``"responded"`` (a barge-in that re-gated) — **and** its checkpoint still carries a
    pending interrupt (§4f): being marked at the gate is necessary but not sufficient —
    the ``saver`` cross-check keeps a stale/finished thread out of the Approve-able
    queue.
    """
    from fund.audit.repository import AgentRunRepository

    paused = AgentRunRepository(session).list_paused_runs(portfolio_id)
    return [
        _run_summary(run, awaiting_hitl=True)
        for run in paused
        if _has_pending_interrupt(saver, run.thread_id)
    ]


def load_run_transcript(session: Any, saver: Any, run: Any) -> list[TranscriptEntry]:
    """Merge the checkpoint's ``messages`` with the run's ``agent_decisions``.

    Each decision is anchored right after the message that triggered its agent — a
    ``task(subagent_type=...)`` delegation or a top-level tool call whose name
    matches the decision's ``agent``/``step``. Decisions with no matching message
    (an agent that logged a row without a PM-thread tool call) trail in
    ``decision_index`` order. Messages from agents that log no decision (economist,
    risk) simply appear on their own.
    """
    entries: list[TranscriptEntry] = []
    messages = _messages_for(saver, getattr(run, "thread_id", None))
    subagents = _subagent_by_call(messages)
    remaining = list(run.decisions)
    for index, message in enumerate(messages):
        entries.append(_message_entry(message, index, subagents))
        tokens = _message_tokens(message)
        if not tokens:
            continue
        for decision in [d for d in remaining if _decision_matches(d, tokens)]:
            entries.append(_decision_entry(decision))
            remaining.remove(decision)
    entries.extend(_decision_entry(decision) for decision in remaining)
    return entries


def portfolio_state(session: Any, portfolio_id: uuid.UUID) -> PortfolioState:
    """Current holdings vs target weights, with L1 drift and best-effort metrics.

    ``current`` is the paper book's **market-value** weights (shares × latest
    base-currency close, as :func:`holdings_summary` marks them), so a price move
    since the fill shows up as drift. The ``positions`` snapshot stores the weights
    the book was *built* to, which equal ``target`` right after every approval, so
    it is used only when the book cannot be marked (no paper ticket, an unpriced
    holding, or a base currency without FX coverage). ``target`` is the latest
    ``completed`` run's ``weights`` (fallback: the latest allocator
    ``optimize_portfolio`` proposal of a run still awaiting its HITL decision; an
    ended-without-approval run never supplies one). ``metrics`` is read from that
    same run's newest allocator decision (see :func:`run_metrics`), so a newer
    unapproved proposal's metrics never sit beside the approved target weights; it is
    ``{}`` when that decision carries no ``metrics`` block — they are never
    fabricated.
    """
    from fund.audit.positions_repository import PositionRepository
    from fund.audit.repository import AgentRunRepository

    current = _marked_current_weights(session, portfolio_id)
    if current is None:
        holdings = PositionRepository(session).get_holdings(portfolio_id)
        current = {p.ticker: p.weight for p in holdings}
    run_repo = AgentRunRepository(session)
    target_run_id, target, target_approved = _target_source(run_repo, portfolio_id)
    target_run = run_repo.get_run(target_run_id) if target_run_id is not None else None
    return PortfolioState(
        portfolio_id=portfolio_id,
        current=current,
        target=target,
        drift_l1=drift_l1(current, target),
        metrics=run_metrics(target_run) if target_run is not None else {},
        target_approved=target_approved,
    )


# Only bars this recent count when judging what the stored panel covers: an
# instrument with nothing newer is delisted/dormant and must not hold the bar back.
_DECISION_BAR_LOOKBACK_DAYS = 14
# Share of the recently-priced universe allowed to lag the chosen bar (a market closed
# for a holiday, a half-ingested day) before the bar is judged not yet fillable.
_DECISION_BAR_LAGGING_SHARE = 0.1


def default_decision_bar(session: Any, *, today: dt.date | None = None) -> dt.date:
    """The newest decision bar whose next close is already stored for the universe.

    ``place_orders`` fills at the first close strictly after the decision bar (D30),
    so a bar equal to the wall-clock date (or to the last stored bar) has nothing to
    fill at until a later ingest: an approval in that window finalises the run with no
    ticket. This picks the day before the bar that all but
    ``_DECISION_BAR_LAGGING_SHARE`` of the recently-priced instruments have reached, so
    the ticket fills at a close that exists now and uses no price after the decision.

    Args:
        session: An injected sync ``portopt_db`` session; not owned here (no commit).
        today: Upper bound for the bars considered (UTC date by default); also the
            fallback when no instrument has a recent bar.

    Returns:
        The decision bar, or ``today`` when no recent price data is stored.
    """
    from portopt_db.models.market_data.yfinance_data import PriceHistory
    from sqlalchemy import func, select

    day = today if today is not None else dt.datetime.now(dt.UTC).date()
    stmt = (
        select(func.max(PriceHistory.date))
        .where(
            PriceHistory.close.is_not(None),
            PriceHistory.date >= day - dt.timedelta(days=_DECISION_BAR_LOOKBACK_DAYS),
            PriceHistory.date <= day,
        )
        .group_by(PriceHistory.instrument_id)
    )
    latest: list[dt.date] = sorted(session.execute(stmt).scalars())
    if not latest:
        return day
    reached = latest[int(len(latest) * _DECISION_BAR_LAGGING_SHARE)]
    return reached - dt.timedelta(days=1)


def _marked_current_weights(
    session: Any, portfolio_id: uuid.UUID
) -> dict[str, float] | None:
    """Market-value weights of the portfolio's paper book, or ``None`` if unmarkable.

    ``None`` — rather than a partial map — when there is no paper ticket, any
    holding is unpriced, or the mandate's base currency has no FX coverage: weights
    renormalised over only the priced holdings would misstate the drift, so the
    caller falls back to the stored snapshot instead.

    Reads through :func:`fund.performance.marked_weights`, which loads only the bars
    since the book's fill: this sits on the TUI's 2-second poll path (once per
    portfolio, several times a tick), where :func:`holdings_summary`'s whole-history
    read — needed there for its trailing returns — would stall the event loop.
    """
    from fund.performance import FxCoverageError, marked_weights

    try:
        return marked_weights(
            portfolio_id,
            session=session,
            base_currency=_base_currency(session, portfolio_id),
        )
    except FxCoverageError:
        return None


def _base_currency(session: Any, portfolio_id: uuid.UUID) -> str:
    """The portfolio's mandate base currency; ``"EUR"`` (the seeded FX base) if none."""
    mandate = get_mandate(session, portfolio_id)
    return mandate.base_currency if mandate is not None else "EUR"


def interrupt_for(saver: Any, run: Any) -> dict[str, Any] | None:
    """The pending HITL interrupt payload for ``run``'s thread, or ``None`` (§4f).

    The value written to the ``__interrupt__`` channel (the same one
    :func:`pending_hitl` cross-checks) is a ``list[Interrupt]``; its first element's
    ``.value`` is the dict the gated tool raised — carrying ``action_requests`` (the
    ``place_orders`` name + proposed weights the adviser reviews) and
    ``review_configs``. The HITL-queue panel renders that dict when a paused run is
    selected. Returns ``None`` when the thread carries no live interrupt.

    Agent-stack-free by construction: a bare ``saver.get_tuple`` read of the
    checkpoint's pending writes, mirroring :func:`_has_pending_interrupt` — no
    ``langgraph`` import, no compiled graph, no model (SPEC R2). Accepts the same
    kind of run object as :func:`load_run_transcript` (an ORM row carrying
    ``thread_id``).
    """
    thread_id = getattr(run, "thread_id", None)
    if not thread_id:
        return None
    tup = saver.get_tuple(_thread_config(thread_id))
    if tup is None:
        return None
    for _task_id, channel, value in tup.pending_writes or []:
        if channel == _INTERRUPT_CHANNEL:
            return _interrupt_value(value)
    return None


def _interrupt_value(value: Any) -> dict[str, Any] | None:
    """Unwrap a checkpoint ``__interrupt__`` write to its dict payload.

    The stored value is a ``list[Interrupt]`` (each ``Interrupt`` exposing ``.value``);
    a bare ``Interrupt`` or a plain dict is tolerated too. Returns the first dict
    payload found, or ``None``.
    """
    items = value if isinstance(value, list | tuple) else [value]
    for item in items:
        inner = getattr(item, "value", item)
        if isinstance(inner, dict):
            return inner
    return None


def get_mandate(session: Any, portfolio_id: uuid.UUID) -> PortfolioMandate | None:
    """Rehydrate the pydantic ``PortfolioMandate`` from its JSON column, or ``None``.

    ``MandateRepository.get`` returns the DB row (JSON source of truth + scalar
    mirror); JSON→pydantic rehydration lives here, on the read side, so the
    state/report panels get the typed mandate.
    """
    from fund.audit.mandate_repository import MandateRepository
    from fund.schemas.mandate import PortfolioMandate

    row = MandateRepository(session).get(portfolio_id)
    if row is None:
        return None
    return PortfolioMandate.model_validate(row.mandate)


def performance_summary(
    session: Any,
    portfolio_id: uuid.UUID,
    *,
    asof: dt.date | None = None,
) -> PerformanceSummary:
    """Multi-horizon returns + risk metrics over the persisted NAV series.

    The read-model entry point for the (deferred) Performance view and the Phase-07
    daily brief: reads the ``nav_history`` series written by the daily NAV-mark step
    (cheap — no recompute of the FX-normalized book) and derives the metrics via
    :func:`fund.performance.portfolio_metrics`. An unmarked portfolio yields an
    all-``None`` summary rather than a fabricated one. The tracker is imported
    lazily (it pulls ``optimizer``/``skfolio`` — allowed, not the agent stack) so
    ``import fund.observe`` stays agent-stack-free. Builds no model, never commits.

    Args:
        session: An injected sync ``portopt_db`` session; not owned here.
        portfolio_id: The portfolio to summarize.
        asof: Upper date bound for the series; ``None`` uses all persisted points.

    Returns:
        A :class:`~fund.performance.PerformanceSummary`; all-``None`` fields when no
        NAV has been persisted for the portfolio.
    """
    from fund.audit.nav_history_repository import NavHistoryRepository
    from fund.performance import NavPoint, portfolio_metrics

    rows = NavHistoryRepository(session).get_navs(portfolio_id, end=asof)
    points = [
        NavPoint(
            date=row.date,
            nav=float(row.nav),
            benchmark=float(row.benchmark) if row.benchmark is not None else None,
        )
        for row in rows
    ]
    return portfolio_metrics(portfolio_id, points)


def holdings_summary(
    session: Any,
    portfolio_id: uuid.UUID,
    *,
    asof: dt.date | None = None,
) -> list[HoldingMark]:
    """Per-holding mark-to-market snapshots in the mandate base currency.

    The read-model entry point for the Performance view's holdings table: marks the
    portfolio's current paper book via :func:`fund.performance.holding_marks` — a
    compute-on-demand read of ``paper_orders`` / ``price_history`` / ``fx_rates``,
    mirroring :func:`performance_summary`. The base currency is read from the
    portfolio's mandate, falling back to ``"EUR"`` when it has none (the fund's only
    seeded FX base). An unmarked portfolio (no paper orders) yields ``[]`` rather than
    a fabricated row. The tracker is imported lazily (it pulls ``optimizer``/``skfolio``
    — allowed, not the agent stack) so ``import fund.observe`` stays agent-stack-free.
    Builds no model, never commits.

    Args:
        session: An injected sync ``portopt_db`` session; not owned here.
        portfolio_id: The portfolio whose current holdings to mark.
        asof: Mark-date upper bound; ``None`` uses the latest available prices.

    Returns:
        One :class:`~fund.performance.HoldingMark` per current holding (degraded to
        all-``None`` fields for an unpriced one); ``[]`` when the portfolio has no
        paper orders.

    Raises:
        FxCoverageError: The mandate holds foreign-currency instruments but no FX rate
            covers their currency under its base (propagated from ``holding_marks``).
    """
    from fund.performance import holding_marks

    return holding_marks(
        portfolio_id,
        session=session,
        base_currency=_base_currency(session, portfolio_id),
        asof=asof,
    )


def macro_attribution_summary(
    session: Any,
    portfolio_id: uuid.UUID,
    *,
    asof: dt.date | None = None,
    config: Any = None,
) -> AttributionResult:
    """Attribute the persisted NAV series' period return to macro factors.

    The read-model entry point for the Phase-07 daily brief: reads the ``nav_history``
    series (+ benchmark) written by the daily NAV-mark step and runs the macro
    attribution over it, mirroring :func:`performance_summary` (cheap — no recompute of
    the FX-normalized book). An unmarked portfolio yields a zeroed,
    ``insufficient_history`` result rather than a fabricated one. The engine is imported
    lazily (it pulls ``optimizer``/``skfolio`` — allowed, not the agent stack) so
    ``import fund.observe`` stays agent-stack-free. Builds no model, never commits.

    Args:
        session: An injected sync ``portopt_db`` session; not owned here.
        portfolio_id: The portfolio to attribute.
        asof: Upper date bound for the series; ``None`` uses all persisted points.
        config: Optional ``MacroAttributionConfig``; ``None`` uses the default preset.

    Returns:
        An :class:`~fund.attribution.AttributionResult` whose contributions + residual
        sum to the total period return.
    """
    from fund.attribution import attribution_from_nav_points
    from fund.audit.nav_history_repository import NavHistoryRepository
    from fund.performance import NavPoint

    rows = NavHistoryRepository(session).get_navs(portfolio_id, end=asof)
    # Forward-fill an interior benchmark gap: a partial re-mark can persist a null
    # benchmark mid-series and the upsert overwrites the prior non-null value, so the
    # benchmark return window stays contiguous and never spans a hole with a
    # multi-period return that biases the equity beta — the same forward-fill the
    # compute path applies to base prices. A leading null stays absent, never
    # fabricated before the benchmark's first observation.
    points: list[NavPoint] = []
    last_benchmark: float | None = None
    for row in rows:
        benchmark = (
            float(row.benchmark) if row.benchmark is not None else last_benchmark
        )
        if benchmark is not None:
            last_benchmark = benchmark
        points.append(NavPoint(date=row.date, nav=float(row.nav), benchmark=benchmark))
    return attribution_from_nav_points(points, session=session, config=config)


def reprofile_status(
    session: Any,
    portfolio_id: uuid.UUID,
    *,
    reprofile_interval_days: int = 365,
    now: dt.datetime | None = None,
) -> ReprofileStatus:
    """Whether ``portfolio_id``'s MiFID profile is due for renewal (annual-only, Q4).

    Reads the active (highest-``version``) profile via ``MifidProfileRepository``
    (lazy import, mirroring :func:`get_mandate` so ``import fund.observe`` stays
    agent-stack-free). ``profiled_at`` is that row's ``created_at``; a profile at
    least ``reprofile_interval_days`` old — or a portfolio never profiled — is
    ``due`` with reason ``"annual"``. ``now`` defaults to a tz-aware UTC clock and
    is compared against the tz-aware ``created_at`` (a naive ``created_at``, as
    SQLite yields on round-trip, is read as UTC). ``"risk_drift"`` is reserved and
    never emitted — no persisted drawdown/NAV feeds it. Builds no model, issues no
    ``commit``.
    """
    from fund.audit.mifid_repository import MifidProfileRepository

    now = now or dt.datetime.now(dt.timezone.utc)
    profile = MifidProfileRepository(session).get_active(portfolio_id)
    if profile is None:
        return ReprofileStatus(
            portfolio_id=portfolio_id,
            due=True,
            reasons=("annual",),
            profiled_at=None,
            profile_age_days=None,
            reprofile_interval_days=reprofile_interval_days,
        )
    profiled_at = _as_utc(profile.created_at)
    profile_age_days = (now - profiled_at).days
    due = profile_age_days >= reprofile_interval_days
    return ReprofileStatus(
        portfolio_id=portfolio_id,
        due=due,
        reasons=("annual",) if due else (),
        profiled_at=profiled_at,
        profile_age_days=profile_age_days,
        reprofile_interval_days=reprofile_interval_days,
    )


def recent_events(
    session: Any,
    portfolio_id: uuid.UUID,
    *,
    asof: dt.date | None = None,
    window_days: int = 14,
    limit: int = 10,
) -> EventDigest:
    """Recent global market digests + this portfolio's rebalances, newest first.

    ``market_events`` are the global ``market_journal`` rows in
    ``[asof - window_days, asof]``; ``portfolio_events`` are the last ``limit``
    ``portfolio_journal`` overlays at/before ``asof`` (no look-ahead). Repos are
    imported lazily so ``import fund.observe`` stays agent-stack-free (they are
    ``portopt_db`` reads; no agent stack either way). Builds no model, never
    ``commit``s.

    Args:
        session: An injected sync ``portopt_db`` session; not owned here.
        portfolio_id: The portfolio whose overlays to surface.
        asof: Upper bound; defaults to today (UTC). Events after it are excluded.
        window_days: Trailing span for the global market digests.
        limit: Maximum rows per section.

    Returns:
        An :class:`EventDigest`; both sections empty when nothing is recorded.
    """
    from portopt_db.repositories.market_data.market_journal_repository import (
        MarketJournalRepository,
    )
    from portopt_db.repositories.orders.portfolio_journal_repository import (
        PortfolioJournalRepository,
    )

    end = asof or dt.date.today()
    start = end - dt.timedelta(days=window_days)

    digests = MarketJournalRepository(session).get_range(start, end)
    market_events = tuple(
        {"as_of": row.as_of.isoformat(), "narrative": row.narrative}
        for row in reversed(digests)  # get_range is ascending → newest first
    )[:limit]

    overlays = PortfolioJournalRepository(session).list_for_portfolio(portfolio_id)
    portfolio_events = tuple(
        {
            "as_of": row.as_of.isoformat(),
            "allocation": row.allocation,
            "narrative": row.narrative,
        }
        for row in overlays  # list_for_portfolio is newest first
        if row.as_of <= end
    )[:limit]

    return EventDigest(
        portfolio_id=portfolio_id,
        market_events=market_events,
        portfolio_events=portfolio_events,
    )


def daily_brief(
    session: Any,
    saver: Any,
    portfolio_id: uuid.UUID,
    *,
    asof: dt.date | None = None,
) -> DailyBrief:
    """Assemble the five-section per-day recap for a ``(portfolio, day)``.

    Composition only: delegates to :func:`performance_summary`,
    :func:`macro_attribution_summary`, :func:`recent_events` (one-day window), and the
    most-recent-on-or-before economist read over :func:`load_run_transcript`. Stays
    agent-stack-free — ``saver`` is treated as opaque (no ``langgraph`` import) and
    every repo is imported from its submodule inside this body, never from the
    ``fund.audit`` package root. Builds no model, never ``commit``s.

    Args:
        session: An injected read-only sync session; this path never commits.
        saver: The bootstrapped LangGraph checkpointer, opaque; needed only for the
            economist read.
        portfolio_id: The portfolio to recap.
        asof: The day to recap; defaults to the latest marked day (else today).

    Returns:
        A :class:`DailyBrief` whose sections degrade to None/empty where a day lacks
        data — never a fabricated figure, never a raise.
    """
    from fund.attribution import MacroAttributionConfig
    from fund.audit.nav_history_repository import NavHistoryRepository

    day = asof or _latest_nav_date(session, portfolio_id) or dt.date.today()
    prev_day, next_day = _nav_neighbors(session, portfolio_id, day)
    # performance_summary never returns None — collapse its all-None unmarked sentinel
    # to None so the view shows an explicit "no data" line instead of empty horizons.
    marked = bool(NavHistoryRepository(session).get_navs(portfolio_id, end=day))
    performance = (
        performance_summary(session, portfolio_id, asof=day) if marked else None
    )
    # Betas come from the full history but only the latest bar is explained, so the
    # block beside the 1D P&L line is that day's decomposition, not the cumulative one.
    attribution = macro_attribution_summary(
        session,
        portfolio_id,
        asof=day,
        config=MacroAttributionConfig(attribution_bars=1),
    )
    digest = recent_events(session, portfolio_id, asof=day, window_days=1)
    read, read_date, run_id = _economist_read_asof(session, saver, portfolio_id, day)
    return DailyBrief(
        portfolio_id=portfolio_id,
        asof=day,
        prev_day=prev_day,
        next_day=next_day,
        performance=performance,
        attribution=attribution,
        market_events=digest.market_events,
        economist_read=read,
        economist_read_date=read_date,
        deliberation_run_id=run_id,
        fund_events=digest.portfolio_events,
    )


def _latest_nav_date(session: Any, portfolio_id: uuid.UUID) -> dt.date | None:
    """Return the newest NAV-marked day for ``portfolio_id``, else None."""
    from fund.audit.nav_history_repository import NavHistoryRepository

    return NavHistoryRepository(session).last_nav_date(portfolio_id)


def _nav_neighbors(
    session: Any, portfolio_id: uuid.UUID, day: dt.date
) -> tuple[dt.date | None, dt.date | None]:
    """Return the NAV-marked days adjacent to ``day`` as ``(prev, next)``.

    Each is the nearest mark strictly before / after ``day`` on the NAV trading grid —
    the ``←`` / ``→`` targets — or None at that end of the grid. ``day`` itself need not
    be marked.
    """
    from fund.audit.nav_history_repository import NavHistoryRepository

    dates = [row.date for row in NavHistoryRepository(session).get_navs(portfolio_id)]
    prev_day = max((d for d in dates if d < day), default=None)
    next_day = min((d for d in dates if d > day), default=None)
    return prev_day, next_day


def _economist_read_asof(
    session: Any, saver: Any, portfolio_id: uuid.UUID, day: dt.date
) -> tuple[str | None, dt.date | None, uuid.UUID | None]:
    """Return the carried-forward economist read as ``(text, run_date, run_id)``.

    Picks the fund run with the greatest ``asof <= day`` (tie-broken by ``created_at``
    so a same-day re-run supersedes — ``list_runs_for_portfolio`` sorts by
    ``created_at``, not ``asof``) and takes the last economist ``message`` turn of its
    transcript. Profiling runs share the portfolio's run history but have no economist,
    so they are skipped: a later re-profile must not shadow the last rebalance's read.
    All-None when the portfolio has no fund run on-or-before ``day``.
    """
    from fund.audit.repository import AgentRunRepository

    eligible = [
        run
        for run in AgentRunRepository(session).list_runs_for_portfolio(portfolio_id)
        if run.asof <= day and not _is_profiler_run(run)
    ]
    if not eligible:
        return None, None, None
    # list_runs_for_portfolio is newest-created first, so the first run at the greatest
    # asof is the tie-break winner — no created_at compare (dodges naive/aware mixing).
    max_asof = max(run.asof for run in eligible)
    chosen = next(run for run in eligible if run.asof == max_asof)
    economist = [
        entry
        for entry in load_run_transcript(session, saver, chosen)
        if entry.source == "message" and entry.agent == "economist"
    ]
    text = economist[-1].text if economist else None
    return text, chosen.asof, chosen.id


def _is_profiler_run(run: Any) -> bool:
    """Whether ``run`` is a MiFID profiling attempt rather than a rebalance round."""
    return (run.optimizer_config or {}).get("step") == "profiler"


def audit_timeline(
    session: Any,
    *,
    portfolio_id: uuid.UUID | None = None,
    kind: str | None = None,
    action: str | None = None,
    overrides_only: bool = False,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    limit: int = 200,
) -> list[AuditEntry]:
    """The fund-wide decision timeline: ``agent_decisions`` ∪ ``human_audit_log``.

    Merges the two audit sources into one actor-attributed, newest-first list the
    Audit view renders. Every filter is optional and composes: ``kind`` narrows to one
    actor side (``"agent"`` / ``"human"``); ``action`` matches a decision type (the
    agent step or the human action); ``overrides_only`` keeps only ⚑ override rows
    (human by construction, so agent rows drop out); ``portfolio_id`` + ``start`` /
    ``end`` bound the rows. Repos are imported lazily so ``import fund.observe`` stays
    agent-stack-free; this read opens no transaction and never commits.

    Args:
        session: An injected sync ``portopt_db`` session; not owned here.
        portfolio_id: Keep only this portfolio's rows (both sides).
        kind: ``"agent"`` or ``"human"`` to keep one side; ``None`` for both.
        action: Keep only this decision type (agent step / human action).
        overrides_only: Keep only ⚑ override rows (drops every agent row).
        start: Lower ``created_at`` bound (inclusive).
        end: Upper ``created_at`` bound (inclusive).
        limit: Maximum rows returned after the merge.

    Returns:
        The merged entries ordered by ``when`` descending, capped at ``limit``.
    """
    entries: list[AuditEntry] = []
    if kind != "human" and not overrides_only:
        entries.extend(
            _agent_audit_entries(
                session,
                portfolio_id=portfolio_id,
                step=action,
                start=start,
                end=end,
                limit=limit,
            )
        )
    if kind != "agent":
        entries.extend(
            _human_audit_entries(
                session,
                portfolio_id=portfolio_id,
                action=action,
                overrides_only=overrides_only,
                start=start,
                end=end,
                limit=limit,
            )
        )
    # Stable on purpose: same-instant rows keep the newest-step-first order the
    # repositories return, which a ``when``-only key cannot express.
    entries.sort(key=lambda entry: entry.when, reverse=True)
    return entries[:limit]


# --- internals --------------------------------------------------------------


def _agent_audit_entries(
    session: Any,
    *,
    portfolio_id: uuid.UUID | None,
    step: str | None,
    start: dt.datetime | None,
    end: dt.datetime | None,
    limit: int,
) -> list[AuditEntry]:
    """Agent decisions as :class:`AuditEntry` rows (``kind == "agent"``, no flags)."""
    from fund.audit.repository import AgentRunRepository

    rows = AgentRunRepository(session).list_decisions(
        portfolio_id=portfolio_id, step=step, start=start, end=end, limit=limit
    )
    return [
        AuditEntry(
            when=decision.created_at,
            kind="agent",
            actor=decision.agent,
            action=decision.step,
            portfolio_id=pid,
            run_id=decision.run_id,
            prev_value=None,
            new_value=decision.hitl_decision,
            source="agent_decisions",
            idempotency_key=None,
            override_flag=False,
            suitability_flag=False,
        )
        for decision, pid in rows
    ]


def _human_audit_entries(
    session: Any,
    *,
    portfolio_id: uuid.UUID | None,
    action: str | None,
    overrides_only: bool,
    start: dt.datetime | None,
    end: dt.datetime | None,
    limit: int,
) -> list[AuditEntry]:
    """Human actions as :class:`AuditEntry` rows (``kind == "human"``, with flags)."""
    from fund.audit.human_audit_repository import HumanAuditLogRepository

    rows = HumanAuditLogRepository(session).list_recent(
        portfolio_id=portfolio_id,
        action=action,
        overrides_only=overrides_only,
        start=start,
        end=end,
        limit=limit,
    )
    return [
        AuditEntry(
            when=row.created_at,
            kind="human",
            actor=row.actor,
            action=row.action,
            portfolio_id=row.portfolio_id,
            run_id=row.run_id,
            prev_value=row.prev_value,
            new_value=row.new_value,
            source=row.source,
            idempotency_key=row.idempotency_key,
            override_flag=row.override_flag,
            suitability_flag=row.suitability_flag,
            target=row.target,
            rationale=row.rationale,
        )
        for row in rows
    ]


def _as_utc(value: dt.datetime) -> dt.datetime:
    """Read a persisted timestamp as tz-aware UTC (SQLite drops tzinfo on round-trip).

    ``created_at`` is a ``DateTime(timezone=True)`` column stored UTC; Postgres
    returns it tz-aware, but the SQLite test harness reads it back naive. Treating a
    naive value as UTC keeps :func:`reprofile_status`'s subtraction from raising.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value


def _run_summary(run: Any, *, awaiting_hitl: bool) -> RunSummary:
    return RunSummary(
        run_id=run.id,
        portfolio_id=run.portfolio_id,
        asof=run.asof,
        status=run.status,
        created_at=run.created_at,
        finished_at=run.finished_at,
        n_weights=len(run.weights or {}),
        awaiting_hitl=awaiting_hitl,
    )


def _thread_config(thread_id: str) -> dict[str, Any]:
    """The top-level (``checkpoint_ns=""``) config the saver reads a run's thread by."""
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _has_pending_interrupt(saver: Any, thread_id: str | None) -> bool:
    """Whether ``thread_id``'s checkpoint carries a pending ``__interrupt__`` write."""
    if not thread_id:
        return False
    tup = saver.get_tuple(_thread_config(thread_id))
    if tup is None:
        return False
    return any(
        channel == _INTERRUPT_CHANNEL
        for (_task_id, channel, _value) in (tup.pending_writes or [])
    )


def _messages_for(saver: Any, thread_id: str | None) -> list[Any]:
    """The ``messages`` channel state at ``thread_id``'s latest checkpoint.

    deepagents keeps ``messages`` in a ``DeltaChannel``: a checkpoint stores the
    channel only as a ``_DeltaSnapshot`` every Nth update and otherwise omits it, so
    the state is the nearest snapshot (the history's ``seed``) with the ancestor
    writes folded on top. A plain list under ``channel_values`` (a non-delta graph)
    is returned as-is. Empty when the thread or the channel is absent.
    """
    if not thread_id:
        return []
    tup = saver.get_tuple(_thread_config(thread_id))
    if tup is None:
        return []
    channel_values = tup.checkpoint.get("channel_values", {})
    if "messages" in channel_values:
        return _stored_messages(channel_values["messages"])
    return _replay_messages(saver, tup.config)


def _stored_messages(stored: Any) -> list[Any]:
    """Unwrap a stored ``messages`` blob (a plain list or a ``_DeltaSnapshot``)."""
    # Matched by name: the snapshot type is langgraph-private, and importing it here
    # would break the agent-stack-free module import.
    if type(stored).__name__ == "_DeltaSnapshot":
        stored = stored.value
    return list(stored or [])


def _replay_messages(saver: Any, config: Any) -> list[Any]:
    """Rebuild ``messages`` from the saver's delta history (seed + ordered writes)."""
    history_for = getattr(saver, "get_delta_channel_history", None)
    if history_for is None:
        return []
    history = history_for(config=config, channels=["messages"]).get("messages") or {}
    state = _stored_messages(history["seed"]) if "seed" in history else []
    incoming: list[Any] = []
    for _task_id, _channel, write in history.get("writes", []):
        # A write is one message-like or a list of them; only lists flatten.
        incoming.extend(write if isinstance(write, list) else [write])
    if not incoming:
        return state
    from langgraph.graph.message import add_messages

    return list(add_messages(state, incoming))


def _message_entry(
    message: Any, index: int, subagents: dict[str, str] | None = None
) -> TranscriptEntry:
    return TranscriptEntry(
        source="message",
        agent=_message_agent(message, subagents),
        step=None,
        index=index,
        text=_message_text(message),
        payload=None,
    )


def _subagent_by_call(messages: list[Any]) -> dict[str, str]:
    """Map each ``task`` delegation's ``tool_call_id`` to the subagent it targets."""
    subagents: dict[str, str] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if not isinstance(call, dict) or call.get("name") != "task":
                continue
            args = call.get("args")
            subagent = args.get("subagent_type") if isinstance(args, dict) else None
            if call.get("id") and subagent:
                subagents[str(call["id"])] = str(subagent)
    return subagents


def _message_agent(message: Any, subagents: dict[str, str] | None = None) -> str:
    """The agent a message belongs to.

    A subagent's report never carries its own name in the PM thread: deepagents
    returns it as ``ToolMessage(name="task")``, tied to the delegating AI turn only
    by ``tool_call_id``. Such a result is attributed to the ``subagent_type`` of that
    delegation (via ``subagents``); any other message reports its ``name``, else its
    ``type``.
    """
    name = getattr(message, "name", None)
    if name == "task":
        subagent = (subagents or {}).get(str(getattr(message, "tool_call_id", "")))
        if subagent:
            return subagent
    if name:
        return str(name)
    return str(getattr(message, "type", None) or type(message).__name__)


def _message_text(message: Any) -> str:
    """Flatten a message's ``content`` (str or a list of content blocks)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


def _message_tokens(message: Any) -> set[str]:
    """Anchor tokens for a message: its tool-call names + any ``subagent_type`` arg."""
    tokens: set[str] = set()
    for call in getattr(message, "tool_calls", None) or []:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if name:
            tokens.add(str(name))
        args = call.get("args")
        if isinstance(args, dict):
            subagent = args.get("subagent_type")
            if subagent:
                tokens.add(str(subagent))
    return tokens


def _decision_matches(decision: Any, tokens: set[str]) -> bool:
    return decision.agent in tokens or decision.step in tokens


def _decision_entry(decision: Any) -> TranscriptEntry:
    payload: dict[str, Any] = {}
    if decision.constraint_set:
        payload["constraint_set"] = decision.constraint_set
    if decision.views:
        payload["views"] = decision.views
    if decision.hitl_decision:
        payload["hitl_decision"] = decision.hitl_decision
    weights = _weights_from_response(decision.llm_response)
    if weights:
        payload["weights"] = weights
    return TranscriptEntry(
        source="decision",
        agent=decision.agent,
        step=decision.step,
        index=decision.decision_index,
        text=f"{decision.agent} · {decision.step}",
        payload=payload or None,
    )


def _target_source(
    run_repo: Any, portfolio_id: uuid.UUID
) -> tuple[uuid.UUID | None, dict[str, float], bool]:
    """``(run_id, weights, approved)`` of the run supplying the target, else none.

    The target is the latest completed run's weights, else a pending one's;
    ``approved`` is ``True`` only for a ``completed`` run's weights, while a proposal
    from a run still at a live HITL gate is returned for display with
    ``approved=False``. The proposal fallback only reads such runs: one that ended
    rejected / incomplete / blocked / failed keeps its allocator row in the audit
    trail, but nobody adopted it, and treating it as the target would make a portfolio
    that holds nothing read as drift 1.0. Both reads stream newest-first and stop at
    the first hit, so a portfolio with a long history costs a handful of rows rather
    than the whole trail. ``(None, {}, False)`` when no run supplies a target.
    """
    for run in run_repo.iter_runs_for_portfolio(portfolio_id, status="completed"):
        if run.weights:
            return run.id, {str(k): float(v) for k, v in run.weights.items()}, True
    seen_runs: set[uuid.UUID] = set()
    for decision in run_repo.iter_optimizer_decisions(
        portfolio_id, run_statuses=_LIVE_GATE_STATUSES
    ):
        # Only a run's newest proposal counts, matching latest_optimizer_weights.
        if decision.run_id in seen_runs:
            continue
        seen_runs.add(decision.run_id)
        weights: dict[str, float] = run_repo.decode_weights(decision)
        if weights:
            return decision.run_id, weights, False
    return None, {}, False


def run_metrics(run: Any) -> dict[str, float]:
    """Best-effort metrics from ``run``'s own newest allocator proposal.

    The one reader of a run's persisted metrics: :attr:`PortfolioState.metrics` and
    ``fund report`` both use it, so neither borrows another run's numbers. A run with
    no metrics block yields ``{}`` (metrics are never fabricated). The newest proposal
    is authoritative even when it carries no metrics (an adviser edit does not), so an
    older proposal's figures never describe the edited weights.

    Args:
        run: An ``AgentRun`` row with its ``decisions`` loaded; read-only, no session
            is touched here.

    Returns:
        Metric name to value from the run's newest ``allocator`` /
        ``optimize_portfolio`` decision; ``{}`` when the run has no such decision or
        its payload is unparseable.
    """
    for decision in reversed(run.decisions):
        if decision.agent == "allocator" and decision.step == "optimize_portfolio":
            return _metrics_from_response(decision.llm_response)
    return {}


def _weights_from_response(raw: str | None) -> dict[str, Any] | None:
    parsed = _parse_json_object(raw)
    if parsed is None:
        return None
    weights = parsed.get("weights")
    return weights if isinstance(weights, dict) else None


def _metrics_from_response(raw: str | None) -> dict[str, float]:
    parsed = _parse_json_object(raw)
    if parsed is None:
        return {}
    metrics = parsed.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        str(k): float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _parse_json_object(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


__all__ = [
    "AuditEntry",
    "DailyBrief",
    "EventDigest",
    "PortfolioState",
    "ReprofileStatus",
    "RunSummary",
    "TranscriptEntry",
    "audit_timeline",
    "daily_brief",
    "default_decision_bar",
    "drift_l1",
    "get_mandate",
    "holdings_summary",
    "interrupt_for",
    "list_portfolio_runs",
    "list_portfolios",
    "load_run_transcript",
    "macro_attribution_summary",
    "pending_hitl",
    "performance_summary",
    "portfolio_state",
    "recent_events",
    "reprofile_status",
    "run_metrics",
]
