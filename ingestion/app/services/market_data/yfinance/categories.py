"""One data category of the yfinance sweep: due check, fetch, store, report.

Every category the sweep refreshes goes through the same steps. A settled
category is skipped; otherwise it is fetched, an answer with nothing in it counts
as zero rows, and anything else is stored inside its own savepoint. A failure is
turned into an error string, so one bad category never costs the ticker its
others. A ``CategorySpec`` holds what differs between categories and
``run_category`` runs the steps.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from sqlalchemy.exc import StatementError

from app.services.market_data.yfinance.fetch_plan import FetchContext, FetchPlan

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

# Longest per-category error text kept in logs and the job's error rows.
_MAX_ERROR_CHARS = 300


def default_is_empty(payload: Any) -> bool:
    """Report whether a frame-shaped answer holds nothing to store.

    Args:
        payload: A DataFrame or Series from Yahoo, or ``None`` when it sent none.

    Returns:
        True for ``None`` and for a frame or series without rows.
    """
    return payload is None or bool(payload.empty)


def is_empty_collection(payload: Any) -> bool:
    """Report whether a list or dict answer holds nothing to store.

    Args:
        payload: A list or dict from Yahoo, or ``None`` when it sent none.

    Returns:
        True for ``None`` and for an empty list or dict.
    """
    return not payload


@dataclass(frozen=True)
class CategorySpec(Generic[T]):
    """What one data category of the sweep fetches, checks and stores.

    Attributes:
        name: Key of the category in ``counts``, its label in ``skipped`` and the
            prefix of its error.
        marker: Staleness key that decides whether the category is due.
        fetch: Requests the category's answer for the context's ticker.
        store: Writes a non-empty answer and returns the number of rows stored.
        is_empty: Tells an answer with nothing to store. Such an answer counts as
            zero rows and is never stored.
        transform: Reshapes a non-empty answer before it is stored; ``None``
            stores the answer as fetched. An answer the transform empties counts
            as zero rows.
    """

    name: str
    marker: str
    fetch: Callable[[FetchContext], T]
    store: Callable[[T, FetchContext], int]
    is_empty: Callable[[T], bool] = default_is_empty
    transform: Callable[[T], T] | None = None


@dataclass
class FetchResult:
    """What one ticker's sweep reports: rows stored, failures and skips.

    Attributes:
        counts: Rows stored per category, zero when there was nothing to store.
        errors: One ``"<label>: <message>"`` entry per failure, in sweep order.
        skipped: Labels of the categories left alone, in sweep order.
    """

    counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_response(self) -> dict[str, Any]:
        """Return the result in the shape ``fetch_and_store`` has always returned.

        Returns:
            ``{"counts": ..., "errors": ..., "skipped": ...}`` in that key order,
            every key present even when empty, holding this result's own dict and
            lists.
        """
        return {"counts": self.counts, "errors": self.errors, "skipped": self.skipped}


class SavepointSession(Protocol):
    """A database session that can open a savepoint inside its transaction."""

    def begin_nested(self) -> AbstractContextManager[Any]:
        """Open a savepoint that is rolled back when its block raises."""
        ...


def describe_statement_error(exc: StatementError) -> str:
    """Return the cause of a failed statement without the statement itself.

    SQLAlchemy's message embeds the whole SQL text, and a multi-row upsert of a
    5y price history runs to ~300 KB. Kept as is, one such error per category
    would flood the container log, the sweep's in-memory error list and the
    job's stored error rows.

    Args:
        exc: The failed-statement error raised by a repository write.

    Returns:
        ``"<DriverError>: <message>"`` on one line, cut to ``_MAX_ERROR_CHARS``.
    """
    cause: Any = exc.orig
    if cause is None:
        text = str(exc.args[0]) if exc.args else type(exc).__name__
    else:
        text = f"{type(cause).__name__}: {cause}"
    text = " ".join(text.split())
    if len(text) > _MAX_ERROR_CHARS:
        text = text[: _MAX_ERROR_CHARS - 3] + "..."
    return text


def write_in_savepoint(
    session: SavepointSession,
    write: Callable[..., R],
    /,
    *args: Any,
    **kwargs: Any,
) -> R:
    """Run one repository write inside its own savepoint.

    PostgreSQL aborts the whole transaction after any failed statement, and
    COMMIT of an aborted transaction silently becomes ROLLBACK. With one shared
    session, a single bad category would therefore fail every later category and
    discard the ticker's earlier writes, while ``counts`` still reported them. The
    savepoint rolls back only the write that raised; the exception still
    propagates to the caller.

    Args:
        session: Session whose transaction the write joins.
        write: Repository write to run.
        *args: Positional arguments for *write*.
        **kwargs: Keyword arguments for *write*.

    Returns:
        Whatever *write* returns (the number of rows it stored).

    Raises:
        RuntimeError: When the database rejected the statement. The message is
            the driver's cause, without the SQL text; the original error stays on
            ``__cause__``.
    """
    try:
        with session.begin_nested():
            return write(*args, **kwargs)
    except StatementError as exc:
        raise RuntimeError(describe_statement_error(exc)) from exc


def is_settled(
    plan: FetchPlan,
    marker: str,
    recently_empty: Callable[[str], bool] | None = None,
) -> bool:
    """Report whether the category behind *marker* can be left alone this sweep.

    A category is settled when its marker is inside its freshness window, or when
    the plan uses staleness and Yahoo recently answered the category empty: an
    empty answer stores no row, so without that memory a category Yahoo has
    nothing for would be requested on every run.

    Args:
        plan: Freshness decision of the ticker's sweep.
        marker: Staleness key of the category.
        recently_empty: Reports whether Yahoo recently answered the category
            behind a marker empty. Asked only when the plan uses staleness and the
            marker is past its window, because the sweep adds to that memory while
            it runs. ``None`` when no memory is kept.

    Returns:
        True when the category need not be fetched.

    Raises:
        KeyError: *marker* is not one the plan knows.
    """
    if not plan.is_due(marker):
        return True
    return plan.uses_staleness and recently_empty is not None and recently_empty(marker)


def run_category(
    spec: CategorySpec[T],
    ctx: FetchContext,
    plan: FetchPlan,
    result: FetchResult,
    *,
    session: SavepointSession,
    recently_empty: Callable[[str], bool] | None = None,
) -> None:
    """Refresh one category of a ticker and report the outcome into *result*.

    A settled category is listed in ``skipped``. Any other is fetched and counted:
    zero rows for an answer with nothing in it, else the rows its store wrote. A
    failure of the fetch, the transform or the store is reported in ``errors`` as
    ``"<name>: <message>"`` and leaves the category out of ``counts``; it never
    propagates.

    Args:
        spec: The category to refresh.
        ctx: Ticker and settings of the sweep.
        plan: Decides whether the category is due.
        result: Receives the count, the error or the skip.
        session: Session the store writes through. Its savepoint is rolled back
            when the store raises, so the ticker's other writes survive.
        recently_empty: Memory of empty answers; see ``is_settled``.

    Raises:
        KeyError: The spec's marker is not one the plan knows. A misspelt marker
            is a programming error, not a category Yahoo failed to serve.
    """
    if is_settled(plan, spec.marker, recently_empty):
        result.skipped.append(spec.name)
        return
    try:
        stored = _refresh(spec, ctx, session)
    except Exception as exc:
        result.errors.append(f"{spec.name}: {exc}")
        logger.warning("Failed %s for %s: %s", spec.name, ctx.yfinance_ticker, exc)
        return
    result.counts[spec.name] = stored


def _refresh(
    spec: CategorySpec[T], ctx: FetchContext, session: SavepointSession
) -> int:
    """Fetch the category and store a non-empty answer; return the rows stored."""
    payload = spec.fetch(ctx)
    if spec.is_empty(payload):
        return 0
    if spec.transform is not None:
        payload = spec.transform(payload)
        if spec.is_empty(payload):
            return 0
    return write_in_savepoint(session, spec.store, payload, ctx)
