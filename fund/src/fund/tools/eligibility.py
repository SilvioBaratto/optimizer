"""Eligibility resolver: which tickers a portfolio may hold on a run date, and why not.

A mandate's ``eligible_tickers`` is a hard allowlist. Several weighting tools and the
PM seed all need the same answer to "which of these may the portfolio hold", so it is
computed once per run (``RunContext.eligible``) rather than inside each tool call,
where repeated queries could answer differently within one run.

The resolver is deliberately **not** a bound tool and is not exported from
``fund.tools``: the model never chooses its own eligibility. Callers import it lazily
inside a function — the package pulls the ``optimizer`` stack in through
``fund.tools.universe``, which ``fund.observe`` and ``fund.tui`` must stay free of.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from portopt_db.models.universe.universe import Instrument
from sqlalchemy import select

from fund.tools.mandate import held_tickers
from fund.tools.universe import UNCLASSIFIED, esg_barred, uninvestable_reasons

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from fund.schemas.constraint_set import ConstraintSet
    from fund.schemas.enums import GicsSector
    from fund.schemas.mandate import PortfolioMandate

NOT_IN_ALLOWLIST = "not_in_allowlist"
ESG_UNCLASSIFIED = "esg_unclassified"
_ETF = "ETF"


@dataclasses.dataclass(frozen=True)
class Eligibility:
    """The outcome of resolving a portfolio's allowlist on one run date.

    Attributes:
        allowed: Tickers the portfolio may hold, in mandate order.
        barred: ``(ticker, reason)`` for every allowlisted ticker that may not be
            held, in mandate order. Reasons are stable strings: ``no_instrument``,
            ``delisted``, ``not_priced``, ``esg:<gics sector>``, ``esg_unclassified``.
    """

    allowed: tuple[str, ...]
    barred: tuple[tuple[str, str], ...]

    @property
    def allowed_set(self) -> frozenset[str]:
        """The allowed tickers as a set, for membership tests."""
        return frozenset(self.allowed)

    def reason_for(self, ticker: str) -> str | None:
        """Why ``ticker`` is barred, or ``None`` when it is not a barred name.

        A ticker outside the allowlist altogether is not barred by the resolver, so it
        also yields ``None``; use :meth:`offenders` to classify arbitrary tickers.
        """
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
            barred allowlist name, ``not_in_allowlist`` for any other name. Empty when
            every ticker is allowed.
        """
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


def _instrument_types(session: Session, tickers: Iterable[str]) -> dict[str, str]:
    """The upper-cased ``instrument_type`` of each ticker, ``""`` when unknown."""
    rows = session.execute(
        select(Instrument.yfinance_ticker, Instrument.instrument_type).where(
            Instrument.yfinance_ticker.in_(list(tickers))
        )
    ).all()
    types: dict[str, str] = {}
    for ticker, instrument_type in rows:
        types.setdefault(ticker, (instrument_type or "").strip().upper())
    return types


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


def _esg_reasons(
    session: Session, tickers: list[str], constraint_set: ConstraintSet | None
) -> dict[str, str]:
    """ESG bars among the investable ``tickers``; empty with no exclusions to apply.

    Anything that is not an ETF is screened as a single stock, so an instrument of an
    unrecognised type cannot slip past a hard exclusion.
    """
    if constraint_set is None or not tickers:
        return {}
    exclusions = frozenset(constraint_set.esg.exclusions)
    if not exclusions:
        return {}
    types = _instrument_types(session, tickers)
    stocks = [t for t in tickers if types.get(t) != _ETF]
    return _stock_esg_reasons(session, stocks, exclusions)


def resolve_eligibility(
    session: Session,
    allowlist: Iterable[str],
    asof: dt.date,
    *,
    constraint_set: ConstraintSet | None = None,
) -> Eligibility | None:
    """Resolve ``allowlist`` into the tickers that may be held on ``asof``.

    Args:
        session: A sync ``portopt_db`` session; the function does not own it.
        allowlist: The mandate's ``eligible_tickers``.
        asof: Decision bar; delistings and prices after it are ignored, matching
            :func:`fund.tools.universe.list_universe`.
        constraint_set: The client's active constraints. Its ESG exclusions bar
            allowlisted stocks in an excluded or unplaceable sector, failing closed
            like ``list_universe``. ``None`` (or no exclusions) applies no ESG screen.

    Returns:
        ``None`` when the allowlist is empty: the portfolio is unrestricted and
        callers apply no check. Otherwise an :class:`Eligibility` whose ``allowed``
        and ``barred`` follow the mandate's order. A name that fails the data checks
        keeps that reason and is not screened for ESG. The data checks cost a constant
        number of queries however long the list is; the ESG screen costs a few per
        screened ticker, which is why the result is resolved once per run.
    """
    tickers = _normalise(allowlist)
    if not tickers:
        return None
    barred = uninvestable_reasons(session, tickers, asof)
    barred.update(
        _esg_reasons(session, [t for t in tickers if t not in barred], constraint_set)
    )
    return Eligibility(
        allowed=tuple(t for t in tickers if t not in barred),
        barred=tuple((t, barred[t]) for t in tickers if t in barred),
    )


def resolve_mandate_eligibility(
    session: Session,
    mandate: PortfolioMandate | None,
    asof: dt.date,
    *,
    constraint_set: ConstraintSet | None = None,
) -> Eligibility | None:
    """Resolve the allowlist of ``mandate``, if the portfolio has one.

    The one entry point the places that bind a ``RunContext`` share, so a missing
    mandate (a run whose mandate row has vanished) reads as unrestricted everywhere.

    Args:
        session: A sync ``portopt_db`` session; the function does not own it.
        mandate: The portfolio's current mandate, or ``None`` when it has none.
        asof: Decision bar of the run.
        constraint_set: The run's active constraints, for the ESG screen.

    Returns:
        ``None`` without a mandate or an allowlist; otherwise the resolved
        :class:`Eligibility`.
    """
    if mandate is None:
        return None
    return resolve_eligibility(
        session, mandate.eligible_tickers, asof, constraint_set=constraint_set
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
    "Eligibility",
    "not_eligible_violations",
    "resolve_eligibility",
    "resolve_mandate_eligibility",
]
