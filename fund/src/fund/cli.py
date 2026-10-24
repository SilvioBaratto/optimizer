"""Manual fund operating cycle: ``python -m fund.cli <command>``.

The complete headless human driver, mirroring ``ingestion/app/cli.py``. Every
command is a thin wrapper over :mod:`fund.observe` (the model-free read model),
:func:`fund.actions.apply_action` (the audited path for every human mutation, shared
with the TUI: it drives ``run_fund`` / ``run_profiler`` / ``resume_run`` and appends
the ``human_audit_log`` row), and the ``fund.audit`` repositories. The seven commands
span one cycle::

    mandate set|show   persist / display the per-portfolio mandate (no model)
    profile            drive the MiFID profiler to its approval gate
    run                drive a paper rebalance to the place_orders gate, detach
    approve | reject   resume a paused run — profiler save_profile or rebalance
                       place_orders gate (persist the profile / ticket, or discard)
    status             runs + statuses + awaiting-HITL flag (no model)
    report             tabular audit of one run (no model)

**Transactions:** each command owns its session — ``with get_session() as
session:`` — and commits (writes) or rolls back (reads); the repositories and
``apply_action`` never ``commit``, so an action and its audit row land together.
``run`` / ``profile`` / ``approve`` / ``reject`` also open the LangGraph persistence
via :func:`setup_langgraph` and **close its pool** in a ``finally`` (the caller owns
the pool's lifetime).

**Model:** ``profile`` / ``run`` / ``approve`` / ``reject`` build the chat model
lazily via :func:`build_primary`; a missing ``OLLAMA_API_KEY`` raises a clear
``RuntimeError`` surfaced as a non-zero exit. ``mandate`` / ``status`` / ``report``
build no model. Importing this module reads no environment (``load_project_env``
runs first, then the model/pool are built per command).
"""

from __future__ import annotations

# Load .env FIRST, before importing anything that reads settings at import time. An
# exported variable keeps precedence over the .env (see ``load_project_env``).
from fund.config import load_project_env

load_project_env()

import datetime as dt
import getpass
import logging
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any, NoReturn

import typer

from fund import observe
from fund.actions import apply_action
from fund.agents.model import build_primary
from fund.audit import (
    AgentRunRepository,
    OrderRepository,
    setup_langgraph,
)
from fund.config import settings
from fund.database import get_session
from fund.schemas.mandate import PortfolioMandate, RunTriggers

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Manual fund operating cycle (headless human driver).",
)
mandate_app = typer.Typer(
    add_completion=False, help="Per-portfolio mandate management."
)
app.add_typer(mandate_app, name="mandate")

# Provenance stamped on every audit row the CLI writes, so the Audit view tells a
# headless action from a TUI one.
_SOURCE = "cli"

# Module-level typer option (B008: a list-typed default must be a shared singleton,
# not an inline call); the gated-tool flag is identical for `mandate set` and `run`.
_INTERRUPT_ON = typer.Option(
    None, "--interrupt-on", help="Gated tool (repeatable); default place_orders."
)


# --- boot / exit / error helpers --------------------------------------------


def _boot() -> None:
    """Configure logging; the DB engine lazy-initialises on first session."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _exit(ok: bool) -> None:
    raise typer.Exit(code=0 if ok else 1)


def _fail(message: str) -> NoReturn:
    # typer.Exit subclasses RuntimeError, so a command that wraps this in
    # `except RuntimeError` must re-raise typer.Exit first or the handler re-reports
    # it as a second, empty "error: " line.
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _uuid(value: str, *, label: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        _fail(f"invalid {label}: {value!r}")


def _parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except (ValueError, TypeError):
        _fail(f"invalid date (expected YYYY-MM-DD): {value!r}")


def _build_model() -> Any:
    """Build the primary chat model, surfacing a missing key as a clean exit."""
    try:
        return build_primary(settings)
    except RuntimeError as exc:
        _fail(str(exc))


def _actor() -> str:
    """The OS user driving the CLI, recorded as the audit actor."""
    try:
        return getpass.getuser()
    except (OSError, KeyError, ImportError):
        return "human"


def _to_decimal(capital: float) -> Decimal:
    try:
        return Decimal(str(capital))
    except (InvalidOperation, ValueError):
        _fail(f"invalid capital: {capital!r}")


# --- mandate construction ---------------------------------------------------


def _build_mandate(
    portfolio_id: uuid.UUID,
    existing: PortfolioMandate | None,
    *,
    capital: float | None,
    base_currency: str | None,
    drift: float | None,
    benchmark: str | None,
    interrupt_on: list[str] | None,
) -> PortfolioMandate:
    """Merge CLI flags over any ``existing`` mandate into a new ``PortfolioMandate``."""
    if capital is not None:
        capital_val = _to_decimal(capital)
    elif existing is not None:
        capital_val = existing.capital
    else:
        _fail("--capital is required (no mandate persisted for this portfolio)")
    currency = base_currency or (existing.base_currency if existing else "EUR")
    drift_val = (
        drift
        if drift is not None
        else (existing.drift_l1_threshold if existing else 0.1)
    )
    benchmark_val = (
        benchmark
        if benchmark is not None
        else (existing.benchmark if existing else None)
    )
    gates = (
        tuple(interrupt_on)
        if interrupt_on
        else (existing.hitl_gates if existing else ("place_orders",))
    )
    triggers = existing.triggers if existing else RunTriggers(cron=True, drift=True)
    # The adviser saves these from the TUI and no CLI flag can express them, so a
    # rebuild that omitted them would silently erase the persisted source of truth.
    directives = existing.remembered_directives if existing else ()
    return PortfolioMandate(
        portfolio_id=str(portfolio_id),
        capital=capital_val,
        base_currency=currency,
        drift_l1_threshold=drift_val,
        hitl_gates=gates,
        triggers=triggers,
        benchmark=benchmark_val,
        remembered_directives=directives,
    )


def _save_mandate(
    session: Any,
    portfolio_id: uuid.UUID,
    existing: PortfolioMandate | None,
    mandate: PortfolioMandate,
) -> None:
    """Persist ``mandate`` through the audited action path (no commit)."""
    apply_action(
        "mandate_edit" if existing is not None else "new_portfolio",
        session=session,
        portfolio_id=portfolio_id,
        mandate=mandate,
        prev_value=existing.model_dump(mode="json") if existing is not None else None,
        actor=_actor(),
        source=_SOURCE,
    )


# --- mandate set|show -------------------------------------------------------


@mandate_app.command("set")
def mandate_set(
    portfolio_id: str = typer.Argument(..., help="Portfolio UUID."),
    capital: float | None = typer.Option(
        None, "--capital", help="Positive notional; required for a new mandate."
    ),
    base_currency: str | None = typer.Option(
        None, "--base-currency", help="ISO-4217 code; default EUR for a new mandate."
    ),
    drift: float | None = typer.Option(
        None, "--drift", help="L1 drift threshold (>0); default 0.1 for a new mandate."
    ),
    benchmark: str | None = typer.Option(
        None, "--benchmark", help="Tracking benchmark."
    ),
    interrupt_on: list[str] | None = _INTERRUPT_ON,
) -> None:
    """Upsert the per-portfolio mandate (one row per portfolio).

    Only the flags given are changed on an existing mandate; everything else
    (triggers, gates, benchmark, remembered directives) is carried over.
    """
    _boot()
    pid = _uuid(portfolio_id, label="portfolio id")
    with get_session() as session:
        existing = observe.get_mandate(session, pid)
        mandate = _build_mandate(
            pid,
            existing,
            capital=capital,
            base_currency=base_currency,
            drift=drift,
            benchmark=benchmark,
            interrupt_on=interrupt_on,
        )
        _save_mandate(session, pid, existing, mandate)
        session.commit()
    typer.echo(f"mandate saved for portfolio {pid}")
    _exit(True)


@mandate_app.command("show")
def mandate_show(
    portfolio_id: str = typer.Argument(..., help="Portfolio UUID."),
) -> None:
    """Display the stored mandate for a portfolio."""
    _boot()
    pid = _uuid(portfolio_id, label="portfolio id")
    with get_session() as session:
        mandate = observe.get_mandate(session, pid)
        session.rollback()
    if mandate is None:
        _fail(f"no mandate for portfolio {pid}")
    typer.echo(
        "\n".join(
            [
                f"portfolio:  {mandate.portfolio_id}",
                f"capital:    {mandate.capital} {mandate.base_currency}",
                f"drift_l1:   {mandate.drift_l1_threshold}",
                f"benchmark:  {mandate.benchmark or '-'}",
                f"hitl_gates: {', '.join(mandate.hitl_gates)}",
            ]
        )
    )
    _exit(True)


# --- profile ----------------------------------------------------------------


@app.command()
def profile(
    portfolio_id: str = typer.Argument(..., help="Portfolio UUID."),
    answers: str = typer.Option(
        ..., "--answers", help="Free-text MiFID questionnaire answers."
    ),
    base_currency: str | None = typer.Option(None, "--base-currency"),
    asof: str | None = typer.Option(None, "--asof", help="ISO date; default today."),
) -> None:
    """Drive the MiFID profiler to its adviser-approval gate, then detach."""
    _boot()
    pid = _uuid(portfolio_id, label="portfolio id")
    run_asof = _parse_date(asof) if asof else None
    model = _build_model()
    persistence = setup_langgraph(settings)
    try:
        with get_session() as session:
            result = apply_action(
                "profile",
                session=session,
                portfolio_id=pid,
                questionnaire=answers,
                model=model,
                checkpointer=persistence.saver,
                store=persistence.store,
                base_currency=base_currency,
                asof=run_asof,
                actor=_actor(),
                source=_SOURCE,
                config=settings,
            )
            session.commit()
    except typer.Exit:
        raise
    except (RuntimeError, ValueError, LookupError) as exc:
        _fail(str(exc))
    finally:
        persistence.pool.close()
    typer.echo(result.message)
    _exit(True)


# --- run --------------------------------------------------------------------


@app.command()
def run(
    portfolio_id: str = typer.Argument(..., help="Portfolio UUID."),
    asof: str = typer.Option(..., "--asof", help="Decision bar (ISO date)."),
    capital: float | None = typer.Option(None, "--capital"),
    base_currency: str | None = typer.Option(None, "--base-currency"),
    drift: float | None = typer.Option(None, "--drift"),
    benchmark: str | None = typer.Option(None, "--benchmark"),
    interrupt_on: list[str] | None = _INTERRUPT_ON,
) -> None:
    """Resolve the mandate (flags override/upsert), drive one paper rebalance to
    the ``place_orders`` gate, print the run id, and detach."""
    _boot()
    pid = _uuid(portfolio_id, label="portfolio id")
    run_asof = _parse_date(asof)
    flags_given = (
        capital is not None
        or base_currency is not None
        or drift is not None
        or benchmark is not None
        or bool(interrupt_on)
    )
    model = _build_model()
    persistence = setup_langgraph(settings)
    try:
        with get_session() as session:
            existing = observe.get_mandate(session, pid)
            if existing is None and capital is None:
                _fail(
                    f"no mandate for portfolio {pid}; run 'mandate set' first "
                    f"or pass --capital"
                )
            if existing is not None and not flags_given:
                mandate = existing
            else:
                mandate = _build_mandate(
                    pid,
                    existing,
                    capital=capital,
                    base_currency=base_currency,
                    drift=drift,
                    benchmark=benchmark,
                    interrupt_on=interrupt_on,
                )
                _save_mandate(session, pid, existing, mandate)
            result = apply_action(
                "run",
                session=session,
                portfolio_id=pid,
                mandate=mandate,
                asof=run_asof,
                model=model,
                checkpointer=persistence.saver,
                store=persistence.store,
                actor=_actor(),
                source=_SOURCE,
                config=settings,
            )
            session.commit()
    except typer.Exit:
        raise
    except (RuntimeError, ValueError, LookupError) as exc:
        _fail(str(exc))
    finally:
        persistence.pool.close()
    typer.echo(f"{result.message} (HITL gate)")
    _exit(True)


# --- approve / reject -------------------------------------------------------


def _resume(run_id: str, decision: str) -> None:
    """Shared driver for ``approve`` / ``reject``: rebuild-to-resume a paused run.

    ``resume_run`` dispatches on the run's gate — a profiler ``save_profile`` run
    resumes via ``resume_profiler``, a rebalance ``place_orders`` run via
    ``resume_fund`` — so one command drives either HITL gate.
    """
    _boot()
    rid = _uuid(run_id, label="run id")
    model = _build_model()
    persistence = setup_langgraph(settings)
    try:
        with get_session() as session:
            result = apply_action(
                decision,
                session=session,
                run_id=rid,
                checkpointer=persistence.saver,
                store=persistence.store,
                model=model,
                actor=_actor(),
                source=_SOURCE,
                # Resolving a run is terminal: keying on (run, decision), as the TUI
                # does, makes a repeat replay the first outcome instead of re-resuming.
                idempotency_key=f"{rid}:{decision}",
                config=settings,
            )
            session.commit()
            run = AgentRunRepository(session).get_run(rid)
            ticketed = decision != "approve" or (
                run is not None and run.status == "completed"
            )
    except typer.Exit:
        raise
    except (RuntimeError, ValueError, LookupError) as exc:
        _fail(str(exc))
    finally:
        persistence.pool.close()
    typer.echo(result.message)
    # An approve that placed no ticket is a failed approve for a script driving this.
    _exit(ticketed)


@app.command()
def approve(run_id: str = typer.Argument(..., help="Paused run UUID.")) -> None:
    """Approve a paused run: commit the paper ticket and finalise it completed."""
    _resume(run_id, "approve")


@app.command()
def reject(run_id: str = typer.Argument(..., help="Paused run UUID.")) -> None:
    """Reject a paused run: place no order and finalise it rejected."""
    _resume(run_id, "reject")


# --- status / report --------------------------------------------------------


@app.command()
def status(
    portfolio_id: str | None = typer.Argument(
        None, help="Portfolio UUID; omit for the global paused queue."
    ),
) -> None:
    """Show runs + statuses + awaiting-HITL flag (no model)."""
    _boot()
    lines: list[str] = []
    with get_session() as session:
        if portfolio_id is None:
            for r in AgentRunRepository(session).list_paused_runs():
                lines.append(
                    f"{r.id}\tpaused\tportfolio={r.portfolio_id}\tasof={r.asof}"
                )
        else:
            pid = _uuid(portfolio_id, label="portfolio id")
            for s in observe.list_portfolio_runs(session, pid):
                lines.append(
                    f"{s.run_id}\t{s.status}\tasof={s.asof}"
                    f"\tweights={s.n_weights}\thitl={s.awaiting_hitl}"
                )
        session.rollback()
    typer.echo("\n".join(lines) if lines else "(no runs)")
    _exit(True)


@app.command()
def report(run_id: str = typer.Argument(..., help="Run UUID.")) -> None:
    """Tabular audit of one run: mandate, weights, metrics, ticket, decisions."""
    _boot()
    rid = _uuid(run_id, label="run id")
    lines: list[str] = []
    with get_session() as session:
        run = AgentRunRepository(session).get_run(rid)
        if run is None:
            _fail(f"no run {rid}")
        lines.append(f"run {run.id}  status={run.status}  asof={run.asof}")
        if run.weights:
            lines.append("weights:")
            lines.extend(
                f"  {ticker}: {weight:.4f}"
                for ticker, weight in sorted(run.weights.items())
            )
        pid = run.portfolio_id
        if pid is not None:
            mandate = observe.get_mandate(session, pid)
            if mandate is not None:
                lines.append(
                    f"mandate: capital={mandate.capital} {mandate.base_currency} "
                    f"drift={mandate.drift_l1_threshold}"
                )
            state = observe.portfolio_state(session, pid)
            if state.metrics:
                lines.append(
                    "metrics: "
                    + ", ".join(
                        f"{k}={v:.4f}" for k, v in sorted(state.metrics.items())
                    )
                )
            ticket = OrderRepository(session).latest_for_portfolio(pid)
            if ticket is not None:
                lines.append(
                    f"ticket {ticket.id}  status={ticket.status}  "
                    f"notional={ticket.notional}"
                )
        lines.append("decisions:")
        for d in run.decisions:
            hitl = f"  hitl={d.hitl_decision}" if d.hitl_decision else ""
            lines.append(f"  [{d.decision_index}] {d.agent}/{d.step}{hitl}")
        session.rollback()
    typer.echo("\n".join(lines))
    _exit(True)


if __name__ == "__main__":
    app()
