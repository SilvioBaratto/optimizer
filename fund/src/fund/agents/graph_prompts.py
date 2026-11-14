"""The run instruction a PM round starts from, each block fitted to a budget.

The history digest, the adviser's directives, the eligible universe and the
constraint digest are trimmed to fixed character budgets before they are joined
into the instruction :func:`fund.agents.graph.run_fund` sends.
"""

from __future__ import annotations

import uuid
from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Callable

    from fund.schemas import ConstraintSet
    from fund.schemas.mandate import PortfolioMandate
    from fund.tools.eligibility import Eligibility


_HISTORY_SUMMARY_BUDGET = 800
_HISTORY_DIGEST_WINDOW_DAYS = 14
_HISTORY_SUMMARY_PREFIX = "Since last rebalance — "
_UNBOUNDED = 10**9


def _fit_entries(head: str, entries: list[str], limit: int) -> str:
    """``head`` plus as many whole ``entries`` as fit in ``limit`` characters.

    ``entries`` are in priority order, so the tail is what gets dropped, and the
    dropped count is stated rather than clipping the last entry mid-value.
    Returns ``""`` when not even one entry fits.
    """
    for keep in range(len(entries), 0, -1):
        hidden = len(entries) - keep
        more = f", +{hidden} more" if hidden else ""
        text = f"{head}{', '.join(entries[:keep])}{more}."
        if len(text) <= limit:
            return text
    return ""


def _fit_prose(text: str, limit: int) -> str:
    """``text`` cut back to a word boundary so it fits in ``limit`` characters."""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return ""
    cut = text[: limit - 3]
    words = cut.rsplit(None, 1)
    return (words[0] if len(words) == 2 else cut).rstrip() + "..."


def _fair_shares(sizes: list[int], total: int) -> list[int]:
    """Split ``total`` so small sections keep their size and big ones share the rest."""
    shares = [0] * len(sizes)
    remaining = total
    order = sorted(range(len(sizes)), key=sizes.__getitem__)
    for rank, index in enumerate(order):
        shares[index] = min(sizes[index], remaining // (len(order) - rank))
        remaining -= shares[index]
    return shares


def _history_summary(
    session: Any,
    portfolio_id: uuid.UUID,
    asof: dt.date,
    *,
    budget: int = _HISTORY_SUMMARY_BUDGET,
) -> str:
    """Compact "since last rebalance" recap for the PM seed (empty on first run).

    Reads the portfolio's most recent prior overlay (allocation + whether it
    traded) and the latest global market digest, both bounded strictly/at ``asof``
    (no look-ahead), and folds them into one short string. Repos are imported
    lazily so a bare ``import fund.agents.graph`` stays agent-stack-free, and the
    result is capped to ``budget`` chars so it never crowds the model context or
    the PM round cap. Weights print at four decimals; when the recap overflows,
    each section is trimmed on its own — whole smallest positions are dropped
    (and counted), the digest is cut at a word boundary — so the digest survives
    and no weight is clipped mid-number.

    Args:
        session: A sync ``portopt_db`` session; not owned here.
        portfolio_id: The portfolio whose history to recap.
        asof: The current decision bar; only activity before/at it is considered.
        budget: Maximum character length of the returned recap.

    Returns:
        The recap string, or ``""`` when the portfolio has no prior rebalance and
        no digest is available.
    """
    import datetime as dt

    from portopt_db.repositories.market_data.market_journal_repository import (
        MarketJournalRepository,
    )
    from portopt_db.repositories.orders.portfolio_journal_repository import (
        PortfolioJournalRepository,
    )

    sections: list[Callable[[int], str]] = []

    overlays = PortfolioJournalRepository(session).list_for_portfolio(portfolio_id)
    prior = next((o for o in overlays if o.as_of < asof), None)
    if prior is not None:
        allocation = prior.allocation or {}
        # Largest weights first: when the book overflows the budget it is the small
        # tail that is dropped.
        ranked = sorted(allocation.items(), key=lambda item: (-float(item[1]), item[0]))
        entries = [f"{ticker} {float(weight):.4f}" for ticker, weight in ranked]
        head = f"Prior rebalance {prior.as_of.isoformat()}: "
        sections.append(partial(_fit_entries, head, entries))
        if prior.trades:
            sections.append(partial(_fit_entries, "Traded: ", sorted(prior.trades)))

    start = asof - dt.timedelta(days=_HISTORY_DIGEST_WINDOW_DAYS)
    digests = MarketJournalRepository(session).get_range(start, asof)
    if digests:
        latest = digests[-1]  # get_range is ascending → the freshest at/before asof
        digest = f"Latest digest {latest.as_of.isoformat()}: {latest.narrative}"
        sections.append(partial(_fit_prose, digest))

    if not sections:
        return ""
    # Each section is bounded on its own so one long list cannot push the others out
    # of the recap, and a cut never lands inside a number.
    room = budget - len(_HISTORY_SUMMARY_PREFIX) - (len(sections) - 1)
    shares = _fair_shares([len(render(_UNBOUNDED)) for render in sections], room)
    parts = [render(share) for render, share in zip(sections, shares, strict=True)]
    parts = [part for part in parts if part]
    if not parts:
        return ""
    return _HISTORY_SUMMARY_PREFIX + " ".join(parts)


_DIRECTIVES_BUDGET = 1200
_DIRECTIVE_MAX_CHARS = 300


def _directives_block(
    mandate: PortfolioMandate, *, budget: int = _DIRECTIVES_BUDGET
) -> str:
    """Render the adviser's remembered directives, or ``""`` when there are none.

    Each directive is collapsed onto a single bullet line so free text cannot pose
    as a separate instruction to the PM, capped to ``_DIRECTIVE_MAX_CHARS``, and the
    newest directives win when the total would exceed ``budget``, so a long-lived
    mandate cannot crowd the PM context.
    """
    kept: list[str] = []
    used = 0
    for directive in reversed(mandate.remembered_directives):
        text = " ".join(directive.split())
        if not text:
            continue
        if len(text) > _DIRECTIVE_MAX_CHARS:
            text = text[: _DIRECTIVE_MAX_CHARS - 3].rstrip() + "..."
        if used + len(text) > budget:
            break
        kept.append(text)
        used += len(text)
    if not kept:
        return ""
    bullets = "\n".join(f"- {text}" for text in reversed(kept))
    return (
        "Adviser standing directives (authored by the human adviser and remembered "
        "across rounds; soft guardrails). Pass them on in every delegation brief and "
        "have risk flag any proposal that breaches one. They never relax the "
        f"ConstraintSet limits, the risk gate or the adviser approval:\n{bullets}\n\n"
    )


_ELIGIBLE_ALLOWED_BUDGET = 1400
_ELIGIBLE_BARRED_BUDGET = 700


_BROKER_SCOPE_INTRO = (
    "Broker scope (resolved by the system from the client's mandate and enforced by "
    "the tools): the portfolio may hold only instruments Trading 212 can trade. "
    "list_universe already leaves the others out and the allocator tools refuse any "
    "of them as not_t212_tradeable, so take candidates from list_universe."
)
_BROKER_BAR_EXPLANATION = (
    "not_t212_tradeable means the Trading 212 catalogue does not carry the instrument."
)


def _eligible_block(eligible: Eligibility | None) -> str:
    """Render the portfolio's eligible universe for the PM, or ``""`` when unrestricted.

    Rendered from the resolver's answer, never model-authored. The tools enforce the
    allowlist, so this is guidance that saves the allocator calls it would be refused
    for, not a control. Both lists are fitted to a budget with whole entries and a
    stated "+N more", and because a truncated allowlist would otherwise read as the
    whole of it, the text says ``list_universe`` returns the full set. A broker scope
    with no allowlist has no names to list, so it is stated in words, and a scope the
    resolver could not apply is reported as a note rather than left silent.
    """
    if eligible is None:
        return ""
    notes = " ".join(f"Note: {note}." for note in eligible.notes)
    if not eligible.has_allowlist:
        scope = _BROKER_SCOPE_INTRO if eligible.untradeable else ""
        text = " ".join(part for part in (scope, notes) if part)
        return f"{text}\n\n" if text else ""
    barred = _fit_entries(
        "Barred and unavailable: ",
        [f"{ticker} ({reason})" for ticker, reason in eligible.barred],
        _ELIGIBLE_BARRED_BUDGET,
    )
    if eligible.allowed:
        intro = (
            "Eligible universe (resolved by the system from the client's mandate and "
            "enforced by the tools): the portfolio may hold only the tickers below. "
            "list_universe returns the full set, in this order, and the allocator "
            "tools refuse any other ticker, so pass this to the allocator and have it "
            "take its candidates from list_universe."
        )
        allowed = _fit_entries(
            "Allowed: ", list(eligible.allowed), _ELIGIBLE_ALLOWED_BUDGET
        )
    else:
        intro = (
            "Eligible universe: the client's mandate lists tickers, but no eligible "
            "ticker can be held on this date, so list_universe will report an error "
            "and there is nothing to allocate. Tell the adviser instead of proposing "
            "names outside the list."
        )
        allowed = ""
    broker = _BROKER_BAR_EXPLANATION if eligible.has_broker_bar else ""
    parts = (intro, allowed, barred, broker, notes)
    return " ".join(part for part in parts if part) + "\n\n"


def _constraint_digest(constraint_set: ConstraintSet) -> str:
    """One-line read-only digest of the limits the PM must relay to its subagents.

    The ``ConstraintSet`` lives in the langgraph Store, which no agent tool can
    read, so the seed turn is the only place the PM can learn the objective,
    bounds and drawdown ceilings the skills tell it to pass on. It is rendered
    from the resolved set, never model-authored, so the risk and executor steps
    compare against the real limits instead of invented ones.
    """
    bounds = constraint_set.bounds
    exclusions = ", ".join(s.value for s in constraint_set.esg.exclusions) or "none"
    tolerance = constraint_set.esg.look_through_tolerance
    if constraint_set.esg.exclusions and tolerance > 0:
        exclusions += f" (funds may hold up to {tolerance * 100:g}%)"
    position_cap = constraint_set.universe_filters.max_position_cap
    cap = "" if position_cap is None else f", per-name position cap {position_cap}"
    return (
        f"ConstraintSet: objective {constraint_set.objective.value}, "
        f"risk measure {constraint_set.risk_measure.value}, "
        f"risk aversion {constraint_set.a_gamma}, tail beta {constraint_set.beta}, "
        f"weight bounds [{bounds.min_weights}, {bounds.max_weights}]{cap}, "
        f"budget {bounds.budget}, drawdown ceilings (fractions of capital, checked "
        f"against the worst loss inside any one-year window) "
        f"nu1 {constraint_set.nu1} / nu2 {constraint_set.nu2} / "
        f"nu3 {constraint_set.nu3}, ESG exclusions {exclusions}."
    )


def _run_instruction(
    mandate: PortfolioMandate,
    portfolio_id: str,
    asof: dt.date,
    history_summary: str = "",
    *,
    constraint_set: ConstraintSet | None = None,
    eligible: Eligibility | None = None,
) -> str:
    """The human turn that drives the PM through one paper rebalance.

    When ``history_summary`` is non-empty it is prepended as a context block so
    the PM sees the portfolio's recent allocation/trades and the latest market
    digest before deciding; an empty summary (first run) yields the bare seed.
    The mandate's ``remembered_directives`` follow as one bullet each: the adviser
    saved them to steer every later round, including unattended scheduled ones. The
    mandate line carries the drift band and HITL gates, and a given ``constraint_set``
    adds its limits digest, so the PM can pass real numbers to the subagents rather
    than guessing them. A given ``eligible`` universe is stated after the directives;
    ``None`` (no allowlist) leaves the seed exactly as it was.
    """
    benchmark = f", benchmark {mandate.benchmark}" if mandate.benchmark else ""
    history = f"{history_summary}\n\n" if history_summary else ""
    gates = ", ".join(mandate.hitl_gates) or "none"
    limits = f" {_constraint_digest(constraint_set)}" if constraint_set else ""
    return (
        f"{history}{_directives_block(mandate)}{_eligible_block(eligible)}"
        f"Produce an allocation for portfolio "
        f"{portfolio_id} as of {asof.isoformat()}. Mandate: capital {mandate.capital} "
        f"{mandate.base_currency}{benchmark}, no-trade band (drift band (L1)) "
        f"{mandate.drift_l1_threshold}, HITL gates {gates}.{limits} Delegate in "
        f"order economist -> allocator -> risk -> executor, then commit the "
        f"risk-approved trades through place_orders."
    )
