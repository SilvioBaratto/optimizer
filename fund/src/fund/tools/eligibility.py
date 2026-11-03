"""Eligibility resolver: which tickers a portfolio may hold on a run date, and why not.

A mandate's ``eligible_tickers`` is a hard allowlist. Several weighting tools and the
PM seed all need the same answer to "which of these may the portfolio hold", so it is
computed once per run (``RunContext.eligible``) rather than inside each tool call,
where repeated queries could answer differently within one run.

ESG exclusions are sectors. A single stock sits in one and is judged by it. An ETF has
no sector, so it is judged by look-through of its latest ``etf_sector_weights``
snapshot, in order:

1. a commodity fund passes (its stored weights describe collateral, not exposure);
2. a government or inflation-linked bond fund passes;
3. every other fund (equity, multi-asset, corporate / aggregate / high-yield / EM /
   unclassified bonds) needs a snapshot no older than
   ``FundConfig.esg_sector_weights_max_age_days`` at the decision bar, whose non-null
   weights sum to one within five points and whose labels all map to a GICS sector.
   Otherwise it is barred ``esg_unclassified``;
4. its exposure is the weight in excluded sectors; it is barred ``esg:<sector>``
   (the excluded sector with the largest weight) when that is strictly above the
   client's ``look_through_tolerance``. The default 0.0 is strict.

Honest consequence: under ESG exclusions most corporate and aggregate bond ETFs stay
barred, because their sector data is sparse (government 6 of 265, corporate 27 of 471,
aggregate 70 of 969 funds carry any). That is intended: no data cannot show a fund is
clean.

A mandate's ``tradeable_only`` additionally scopes the portfolio to what the Trading 212
catalogue carries, so a run that must be executable at the broker cannot propose a name
it cannot buy. ``instruments.is_t212_tradeable`` is a tri-state and only an explicit
False bars a name (``not_t212_tradeable``); NULL means the overlay never evaluated the
row, which must not read as a refusal. When no instrument has been evaluated at all the
restriction cannot be applied, so the universe stays whole and the result carries a
note instead of silently going empty. It composes with the allowlist (both must hold),
and with no allowlist it still engages: the result is then a refusal list over the
whole universe (``has_allowlist`` False) rather than an enumeration of it.

The resolver is deliberately **not** a bound tool and is not exported from
``fund.tools``: the model never chooses its own eligibility. The agent-stack-light
callers (``fund.agents.graph``, ``fund.actions``) import it lazily inside a function,
since the package pulls the ``optimizer`` stack in through ``fund.tools.universe``,
which ``fund.observe`` and ``fund.tui`` must stay free of.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from portopt_db.models.universe.universe import Instrument
from portopt_db.repositories.market_data.etf_metadata_repository import (
    ETFMetadataRepository,
)
from portopt_db.repositories.universe.universe_repository import UniverseRepository
from sqlalchemy import select

from fund.config import settings
from fund.tools.mandate import held_tickers
from fund.tools.universe import (
    UNCLASSIFIED,
    _resolve_sector,
    esg_barred,
    uninvestable_reasons,
)

if TYPE_CHECKING:
    from portopt_db.repositories.market_data.etf_metadata_repository import (
        SectorSnapshot,
    )
    from sqlalchemy.orm import Session

    from fund.schemas.constraint_set import ConstraintSet
    from fund.schemas.enums import GicsSector
    from fund.schemas.mandate import PortfolioMandate

NOT_IN_ALLOWLIST = "not_in_allowlist"
ESG_UNCLASSIFIED = "esg_unclassified"
NOT_T212_TRADEABLE = "not_t212_tradeable"
TRADEABLE_NOT_EVALUATED_NOTE = (
    "tradeable_only is set but no instrument has been checked against the "
    "Trading 212 catalogue yet, so the broker restriction is not applied"
)
_ETF = "ETF"
_COMMODITY = "commodity"
_FIXED_INCOME = "fixed_income"
_COMPANY_FREE_FI_SUBCLASSES = frozenset({"government", "inflation_linked"})

# A snapshot's non-null weights must sum to one within this much, since Yahoo rounds
# each sector and omits tiny ones; a bigger gap means a missing or double-counted slice.
_SNAPSHOT_SUM_TOLERANCE = 0.05

# Weights are sums of floats, so ``0.1 + 0.2`` must compare equal to a tolerance of
# ``0.3``; the profiler snaps band edges to the same nine decimals.
_SNAP_DECIMALS = 9


def _snap(x: float) -> float:
    return round(x, _SNAP_DECIMALS)


@dataclasses.dataclass(frozen=True)
class Eligibility:
    """The outcome of resolving a portfolio's allowlist on one run date.

    Attributes:
        allowed: Tickers the portfolio may hold, in mandate order. Empty, and
            meaningless, without an allowlist (:attr:`has_allowlist` False).
        barred: ``(ticker, reason)`` for every allowlisted ticker that may not be
            held, in mandate order. Reasons are stable strings: ``no_instrument``,
            ``delisted``, ``not_priced``, ``not_t212_tradeable``, ``esg:<gics
            sector>``, ``esg_unclassified``.
        has_allowlist: ``True`` when the mandate lists tickers, so only
            :attr:`allowed` may be held. ``False`` when the only restriction is the
            Trading 212 scope over an otherwise unrestricted universe: then nothing
            is enumerated and only the names in :attr:`untradeable` are refused.
        untradeable: Without an allowlist, every ticker the overlay flagged absent
            from the broker. Unused with one, where :attr:`barred` carries the same
            reason per allowlisted name.
        notes: Caveats the system raises about this resolution for the adviser and
            the PM seed, such as a broker scope that could not be applied.
    """

    allowed: tuple[str, ...]
    barred: tuple[tuple[str, str], ...]
    has_allowlist: bool = True
    untradeable: frozenset[str] = frozenset()
    notes: tuple[str, ...] = ()

    @property
    def allowed_set(self) -> frozenset[str]:
        """The allowed tickers as a set, for membership tests."""
        return frozenset(self.allowed)

    @property
    def has_broker_bar(self) -> bool:
        """Whether any allowlisted name is barred for not being on the broker.

        A property rather than a constant import so the renderers that must stay
        agent-stack-free can ask without importing this module.
        """
        return any(reason == NOT_T212_TRADEABLE for _, reason in self.barred)

    def reason_for(self, ticker: str) -> str | None:
        """Why ``ticker`` is barred, or ``None`` when it is not a barred name.

        A ticker outside the allowlist altogether is not barred by the resolver, so it
        also yields ``None``; use :meth:`offenders` to classify arbitrary tickers.
        """
        if not self.has_allowlist:
            return NOT_T212_TRADEABLE if ticker in self.untradeable else None
        for barred_ticker, reason in self.barred:
            if barred_ticker == ticker:
                return reason
        return None

    def offenders(self, tickers: Iterable[str]) -> dict[str, str]:
        """Map each of ``tickers`` that may not be held to the reason.

        Args:
            tickers: Tickers a tool is about to weight, in the caller's spelling; the
                comparison is exact, as the downstream price lookups are.

        Returns:
            ``{ticker: reason}`` in the order given: the resolver's reason for a
            barred allowlist name, ``not_in_allowlist`` for any other name. Without an
            allowlist only a name the broker does not carry is an offender
            (``not_t212_tradeable``). Empty when every ticker is allowed.
        """
        if not self.has_allowlist:
            return {
                ticker: NOT_T212_TRADEABLE
                for ticker in tickers
                if ticker in self.untradeable
            }
        allowed = self.allowed_set
        barred = dict(self.barred)
        return {
            ticker: barred.get(ticker, NOT_IN_ALLOWLIST)
            for ticker in tickers
            if ticker not in allowed
        }


def _normalise(allowlist: Iterable[str]) -> tuple[str, ...]:
    """Strip, upper-case and de-duplicate in order, dropping blanks.

    ``model_copy(update=...)`` skips the mandate's validators, so the resolver cannot
    trust the list it is handed.
    """
    return tuple(
        dict.fromkeys(
            ticker for raw in allowlist if (ticker := raw.strip().upper()) != ""
        )
    )


@dataclasses.dataclass(frozen=True)
class _InstrumentFacts:
    """What the look-through needs to know about an instrument, case-folded."""

    instrument_type: str
    asset_class: str
    fi_subclass: str

    @property
    def is_etf(self) -> bool:
        return self.instrument_type == _ETF

    @property
    def holds_no_company_exposure(self) -> bool:
        """Commodity and government-type bond funds cannot sit in an excluded sector.

        A commodity fund's stored sector weights describe its collateral, not what it
        owns (a gold fund can show a third of its book in basic materials), so they
        are not read. A government or inflation-linked bond fund holds sovereign paper.
        """
        if self.asset_class == _COMMODITY:
            return True
        return (
            self.asset_class == _FIXED_INCOME
            and self.fi_subclass in _COMPANY_FREE_FI_SUBCLASSES
        )


def _instrument_facts(
    session: Session, tickers: Iterable[str]
) -> dict[str, _InstrumentFacts]:
    """The classification of each ticker's instrument, in one query."""
    rows = session.execute(
        select(
            Instrument.yfinance_ticker,
            Instrument.instrument_type,
            Instrument.asset_class,
            Instrument.fi_subclass,
        ).where(Instrument.yfinance_ticker.in_(list(tickers)))
    ).all()
    facts: dict[str, _InstrumentFacts] = {}
    for ticker, instrument_type, asset_class, fi_subclass in rows:
        facts.setdefault(
            ticker,
            _InstrumentFacts(
                instrument_type=(instrument_type or "").strip().upper(),
                asset_class=(asset_class or "").strip().lower(),
                fi_subclass=(fi_subclass or "").strip().lower(),
            ),
        )
    return facts


def _stock_esg_reasons(
    session: Session, tickers: list[str], exclusions: frozenset[GicsSector]
) -> dict[str, str]:
    """ESG bars for single stocks, judged by their stored sector and failing closed.

    The client's exclusions are sectors, and a stock sits in exactly one. A fund's
    exposure is spread over its holdings, so it is not judged here.
    """
    if not tickers:
        return {}
    return {
        ticker: ESG_UNCLASSIFIED if sector == UNCLASSIFIED else f"esg:{sector}"
        for ticker, sector in esg_barred(session, tickers, exclusions).items()
    }


def _look_through(
    snapshot: SectorSnapshot | None,
    exclusions: frozenset[GicsSector],
    tolerance: float,
    *,
    asof: dt.date,
    max_age_days: int,
) -> str | None:
    """The ESG bar of one fund judged by its sector snapshot, or ``None`` if it passes.

    Fails closed (``esg_unclassified``) whenever the snapshot cannot show the fund is
    clean: absent, older than ``max_age_days`` at the decision bar, a label outside the
    eleven GICS sectors, or non-null weights that do not sum to one. A NULL weight is
    skipped rather than read as zero, so the completeness check is what catches a
    snapshot missing the sector that matters.
    """
    if snapshot is None or (asof - snapshot.as_of).days > max_age_days:
        return ESG_UNCLASSIFIED
    by_sector: dict[GicsSector, float] = {}
    total = 0.0
    for label, weight in snapshot.weights.items():
        if weight is None:
            continue
        sector = _resolve_sector(label)
        if sector is None:
            return ESG_UNCLASSIFIED
        total += weight
        by_sector[sector] = by_sector.get(sector, 0.0) + weight
    if _snap(abs(total - 1.0)) > _SNAPSHOT_SUM_TOLERANCE:
        return ESG_UNCLASSIFIED
    # A short position in an excluded sector must not offset a long one.
    exposed = {
        sector: max(weight, 0.0)
        for sector, weight in by_sector.items()
        if sector in exclusions
    }
    if _snap(sum(exposed.values())) <= _snap(tolerance):
        return None
    dominant = max(exposed, key=lambda sector: (exposed[sector], sector.value))
    return f"esg:{dominant.value}"


def _fund_esg_reasons(
    session: Session,
    tickers: list[str],
    exclusions: frozenset[GicsSector],
    tolerance: float,
    *,
    asof: dt.date,
    max_age_days: int,
) -> dict[str, str]:
    """ESG bars for funds, by the look-through of their latest sector snapshot.

    An ETF has no single sector, so it is judged by the share of its snapshot that
    sits in an excluded sector against the client's tolerance. The snapshots of all
    ``tickers`` are fetched in one query however many there are.
    """
    if not tickers:
        return {}
    snapshots = ETFMetadataRepository(session).get_latest_sector_weights(tickers)
    reasons: dict[str, str] = {}
    for ticker in tickers:
        reason = _look_through(
            snapshots.get(ticker),
            exclusions,
            tolerance,
            asof=asof,
            max_age_days=max_age_days,
        )
        if reason is not None:
            reasons[ticker] = reason
    return reasons


def _esg_reasons(
    session: Session,
    tickers: list[str],
    constraint_set: ConstraintSet | None,
    *,
    asof: dt.date,
    max_age_days: int,
) -> dict[str, str]:
    """ESG bars among the investable ``tickers``; empty with no exclusions to apply.

    ETFs are judged by look-through; anything else is screened as a single stock, so
    an instrument of an unrecognised type cannot slip past a hard exclusion.
    """
    if constraint_set is None or not tickers:
        return {}
    exclusions = frozenset(constraint_set.esg.exclusions)
    if not exclusions:
        return {}
    facts = _instrument_facts(session, tickers)
    funds = [t for t in tickers if t in facts and facts[t].is_etf]
    stocks = [t for t in tickers if t not in funds]
    judged_funds = [t for t in funds if not facts[t].holds_no_company_exposure]
    reasons = _stock_esg_reasons(session, stocks, exclusions)
    reasons.update(
        _fund_esg_reasons(
            session,
            judged_funds,
            exclusions,
            constraint_set.esg.look_through_tolerance,
            asof=asof,
            max_age_days=max_age_days,
        )
    )
    return reasons


def resolve_eligibility(
    session: Session,
    allowlist: Iterable[str],
    asof: dt.date,
    *,
    constraint_set: ConstraintSet | None = None,
    max_sector_age_days: int | None = None,
    tradeable_only: bool = False,
) -> Eligibility | None:
    """Resolve ``allowlist`` into the tickers that may be held on ``asof``.

    Args:
        session: A sync ``portopt_db`` session; the function does not own it.
        allowlist: The mandate's ``eligible_tickers``.
        asof: Decision bar; delistings and prices after it are ignored, matching
            :func:`fund.tools.universe.list_universe`.
        constraint_set: The client's active constraints. Its ESG exclusions bar
            allowlisted stocks in an excluded or unplaceable sector, failing closed
            like ``list_universe``, and ETFs by look-through (see the module
            docstring). ``None`` (or no exclusions) applies no ESG screen.
        max_sector_age_days: How old, at ``asof``, an ETF's sector snapshot may be
            before it no longer counts. ``None`` reads
            ``FundConfig.esg_sector_weights_max_age_days``.
        tradeable_only: Also bar every name the Trading 212 overlay flagged absent
            from the broker (``not_t212_tradeable``), see the module docstring.

    Returns:
        ``None`` when the allowlist is empty and ``tradeable_only`` is off: the
        portfolio is unrestricted and callers apply no check. With ``tradeable_only``
        and no allowlist, an :class:`Eligibility` with ``has_allowlist`` False that
        refuses only the untradeable names. Otherwise an :class:`Eligibility` whose
        ``allowed`` and ``barred`` follow the mandate's order. A name that fails the
        data checks keeps that reason and is not screened further, and a name the
        broker lacks is not ESG-screened. The data checks cost a constant number of
        queries however long the list is, the broker scope adds two more, the ETF
        look-through a constant number more, and the stock screen costs a few per
        screened ticker, which is why the result is resolved once per run.
    """
    tickers = _normalise(allowlist)
    if not tickers:
        return _broker_scope(session) if tradeable_only else None
    if max_sector_age_days is None:
        max_sector_age_days = settings.esg_sector_weights_max_age_days
    barred = uninvestable_reasons(session, tickers, asof)
    notes: tuple[str, ...] = ()
    if tradeable_only:
        refused = UniverseRepository(session).get_t212_untradeable_tickers(
            [t for t in tickers if t not in barred]
        )
        if refused is None:
            notes = (TRADEABLE_NOT_EVALUATED_NOTE,)
        else:
            barred.update(dict.fromkeys(refused, NOT_T212_TRADEABLE))
    barred.update(
        _esg_reasons(
            session,
            [t for t in tickers if t not in barred],
            constraint_set,
            asof=asof,
            max_age_days=max_sector_age_days,
        )
    )
    return Eligibility(
        allowed=tuple(t for t in tickers if t not in barred),
        barred=tuple((t, barred[t]) for t in tickers if t in barred),
        notes=notes,
    )


def _broker_scope(session: Session) -> Eligibility:
    """The refusal list of a ``tradeable_only`` mandate that has no allowlist.

    Enumerating every tradeable instrument would turn a universe of thousands into
    an allowlist, so only the names the broker lacks are collected. When no instrument
    has been evaluated nothing can be refused and the note says why.
    """
    refused = UniverseRepository(session).get_t212_untradeable_tickers()
    return Eligibility(
        allowed=(),
        barred=(),
        has_allowlist=False,
        untradeable=frozenset(refused or ()),
        notes=(TRADEABLE_NOT_EVALUATED_NOTE,) if refused is None else (),
    )


def judges_funds_by_look_through(eligible: Eligibility | None) -> bool:
    """Whether the resolver already judged the run's ETFs by sector look-through.

    Only an allowlist is screened that way. A broker scope alone enumerates no names,
    so funds stay with the stored-sector screen they would face without it.

    Args:
        eligible: The run's resolved eligibility, or ``None`` when unrestricted.
    """
    return eligible is not None and eligible.has_allowlist


def resolve_mandate_eligibility(
    session: Session,
    mandate: PortfolioMandate | None,
    asof: dt.date,
    *,
    constraint_set: ConstraintSet | None = None,
    max_sector_age_days: int | None = None,
) -> Eligibility | None:
    """Resolve the allowlist and broker scope of ``mandate``, if it sets either.

    The one entry point the places that bind a ``RunContext`` share, so a missing
    mandate (a run whose mandate row has vanished) reads as unrestricted everywhere.

    Args:
        session: A sync ``portopt_db`` session; the function does not own it.
        mandate: The portfolio's current mandate, or ``None`` when it has none.
        asof: Decision bar of the run.
        constraint_set: The run's active constraints, for the ESG screen.
        max_sector_age_days: See :func:`resolve_eligibility`.

    Returns:
        ``None`` without a mandate, or with neither an allowlist nor
        ``tradeable_only``; otherwise the resolved :class:`Eligibility`.
    """
    if mandate is None:
        return None
    return resolve_eligibility(
        session,
        mandate.eligible_tickers,
        asof,
        constraint_set=constraint_set,
        max_sector_age_days=max_sector_age_days,
        tradeable_only=mandate.tradeable_only,
    )


def not_eligible_violations(
    eligible: Eligibility | None, weights: Mapping[str, float]
) -> list[dict[str, str]]:
    """Violation records for the held names of ``weights`` the portfolio may not hold.

    The one record shape the risk gate (``risk_check``) and the cockpit read-out both
    report, so the two cannot disagree. Only a real holding counts: solver dust and a
    zero weight are not held, so a name the allocator dropped does not block the book.

    Args:
        eligible: The run's resolved eligibility, or ``None`` when unrestricted.
        weights: The ``{ticker: weight}`` vector being checked.

    Returns:
        One ``{type: "not_eligible", asset, reason}`` record per offending held ticker,
        in the order ``weights`` holds them; empty when unrestricted or all allowed.
    """
    if eligible is None:
        return []
    return [
        {"type": "not_eligible", "asset": ticker, "reason": reason}
        for ticker, reason in eligible.offenders(held_tickers(weights)).items()
    ]


__all__ = [
    "ESG_UNCLASSIFIED",
    "NOT_IN_ALLOWLIST",
    "NOT_T212_TRADEABLE",
    "TRADEABLE_NOT_EVALUATED_NOTE",
    "Eligibility",
    "judges_funds_by_look_through",
    "not_eligible_violations",
    "resolve_eligibility",
    "resolve_mandate_eligibility",
]
