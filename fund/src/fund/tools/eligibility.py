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
from collections.abc import Iterable
from typing import TYPE_CHECKING

from fund.tools.universe import uninvestable_reasons

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

NOT_IN_ALLOWLIST = "not_in_allowlist"


@dataclasses.dataclass(frozen=True)
class Eligibility:
    """The outcome of resolving a portfolio's allowlist on one run date.

    Attributes:
        allowed: Tickers the portfolio may hold, in mandate order.
        barred: ``(ticker, reason)`` for every allowlisted ticker that may not be
            held, in mandate order. Reasons are stable strings: ``no_instrument``,
            ``delisted``, ``not_priced``.
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


def resolve_eligibility(
    session: Session, allowlist: Iterable[str], asof: dt.date
) -> Eligibility | None:
    """Resolve ``allowlist`` into the tickers that may be held on ``asof``.

    Args:
        session: A sync ``portopt_db`` session; the function does not own it.
        allowlist: The mandate's ``eligible_tickers``.
        asof: Decision bar; delistings and prices after it are ignored, matching
            :func:`fund.tools.universe.list_universe`.

    Returns:
        ``None`` when the allowlist is empty: the portfolio is unrestricted and
        callers apply no check. Otherwise an :class:`Eligibility` whose ``allowed``
        and ``barred`` follow the mandate's order. Costs a constant number of queries
        however long the list is.
    """
    tickers = _normalise(allowlist)
    if not tickers:
        return None
    barred = uninvestable_reasons(session, tickers, asof)
    return Eligibility(
        allowed=tuple(t for t in tickers if t not in barred),
        barred=tuple((t, barred[t]) for t in tickers if t in barred),
    )


__all__ = ["NOT_IN_ALLOWLIST", "Eligibility", "resolve_eligibility"]
