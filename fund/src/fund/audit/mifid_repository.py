"""``MifidProfileRepository`` + Store helper — fund-side behavior for Fase 5.

The ``mifid_profiles`` *model* lives in ``portopt_db`` (shared schema, Alembic-
owned); its *behavior* lives here, mirroring the ``agent_runs`` / ``paper_orders``
model/repo split. Sits on ``portopt_db.repository``'s ``RepositoryBase`` and opens
no session of its own — the caller injects a sync session (D1). No ``commit``: the
caller owns the transaction boundary.

The suitability profile is **append-only**: :meth:`MifidProfileRepository.add_version`
never overwrites a prior assessment — it inserts the next ``version`` for the
portfolio (an amended assessment is a new row). ``version`` is derived from
``max(version) + 1`` scoped to the portfolio; the DB-level
``UNIQUE(portfolio_id, version)`` is the backstop should two writers ever race
(the second insert raises ``IntegrityError``, which the caller's transaction
rolls back — this repo never double-writes silently).

The Store helper (:func:`put_constraint_set` / :func:`resolve_constraint_set`) is
the cache side of the system of record: it writes the active ``ConstraintSet``
into the LangGraph store under namespace ``(portfolio_id,)`` / key ``store_key``,
so a Phase-4 ``ConstraintSetRef`` resolves back to the persisted profile. The
store handle is a ``BaseStore`` (``PostgresStore`` in prod, ``InMemoryStore`` in
tests); the import stays under ``TYPE_CHECKING`` so importing this module needs no
LangGraph runtime dependency.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from portopt_db.models import MifidProfile
from portopt_db.repository import RepositoryBase
from sqlalchemy import event, func, select

from fund.schemas import ConstraintSet, ConstraintSetRef

if TYPE_CHECKING:  # runtime only needs the store's put/get, not the class
    from langgraph.store.base import BaseStore
    from sqlalchemy.orm import Session, SessionTransaction

logger = logging.getLogger(__name__)


class MifidProfileRepository(RepositoryBase):
    """Append/read MiFID suitability profiles on an injected sync session."""

    def add_version(
        self,
        *,
        portfolio_id: uuid.UUID,
        questionnaire: dict[str, Any],
        constraint_set: dict[str, Any],
        suitability: dict[str, Any],
        store_key: str,
        status: str = "active",
    ) -> MifidProfile:
        """Insert the next version for the portfolio; return the flushed row.

        Append-only: the new ``version`` is ``max(version) + 1`` scoped to
        ``portfolio_id`` (``1`` when the portfolio has no prior profile), so an
        amended assessment never overwrites the previous one.

        Args:
            portfolio_id: The portfolio whose suitability is being assessed.
            questionnaire: Raw investor questionnaire answers as submitted.
            constraint_set: Serialised ConstraintSet derived from the assessment.
            suitability: Suitability classification output from the profiler agent.
            store_key: LangGraph store key under which the constraint set is cached.
            status: Lifecycle state of this version; defaults to ``"active"``.

        Returns:
            The newly inserted and refreshed profile row.
        """
        current_max = self.session.execute(
            select(func.max(MifidProfile.version)).where(
                MifidProfile.portfolio_id == portfolio_id
            )
        ).scalar_one_or_none()
        next_version = (current_max or 0) + 1
        profile = MifidProfile(
            portfolio_id=portfolio_id,
            version=next_version,
            questionnaire=questionnaire,
            constraint_set=constraint_set,
            suitability=suitability,
            store_key=store_key,
            status=status,
        )
        self.session.add(profile)
        self.session.flush()
        self.session.refresh(profile)
        return profile

    def get_active(self, portfolio_id: uuid.UUID) -> MifidProfile | None:
        """Return the newest (highest-``version``) profile for the portfolio.

        Latest-by-version is the active assessment under the append-only model.
        ``None`` when the portfolio has never been profiled.

        Args:
            portfolio_id: Portfolio whose current suitability profile to retrieve.

        Returns:
            The highest-version profile row, or ``None`` if none exists.
        """
        stmt = (
            select(MifidProfile)
            .where(MifidProfile.portfolio_id == portfolio_id)
            .order_by(MifidProfile.version.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def list_portfolio_ids(self) -> list[uuid.UUID]:
        """Distinct portfolio ids with at least one MiFID profile version.

        Collapses the append-only version rows to one id per portfolio. Order is
        unspecified; ``observe.list_portfolios`` dedups across sources and sorts.
        """
        stmt = select(MifidProfile.portfolio_id).distinct()
        return list(self.session.execute(stmt).scalars().all())


def put_constraint_set(
    store: BaseStore,
    constraint_set: ConstraintSet,
    *,
    store_key: str,
) -> ConstraintSetRef:
    """Cache the active ``ConstraintSet`` in the Store; return a resolvable ref.

    Writes ``constraint_set`` (JSON-dumped for portability across the Postgres /
    in-memory stores) under namespace ``(portfolio_id,)`` / key ``store_key``, and
    returns the :class:`ConstraintSetRef` a later decision resolves it by.

    Args:
        store: LangGraph store instance (Postgres in production, in-memory in tests).
        constraint_set: The active constraint set to persist.
        store_key: Key under which the set is stored; also embedded in the returned ref.

    Returns:
        A ref that ``resolve_constraint_set`` can use to retrieve the cached set.
    """
    store.put(
        (constraint_set.portfolio_id,),
        store_key,
        constraint_set.model_dump(mode="json"),
    )
    return ConstraintSetRef(
        portfolio_id=constraint_set.portfolio_id, store_key=store_key
    )


@dataclass
class _PendingPut:
    store: Any
    constraint_set: ConstraintSet
    store_key: str
    # The innermost SAVEPOINT open when the put was queued (``None`` at the root):
    # rolling that SAVEPOINT back, or any transaction enclosing it, voids the put.
    origin: SessionTransaction | None


class _CommitPublisher:
    """Per-session queue of Store writes that are released once the root commits.

    The Store has its own connection, so a put cannot join the session's database
    transaction: written eagerly, it outlives a rollback of the row it caches. This
    queue holds each put back until the root transaction has really committed.

    A SAVEPOINT release also fires ``after_commit`` while the outer transaction can
    still roll back, and there is no way to tell the two apart from that event
    alone. The flag therefore only counts when it survives to the end of the *root*
    transaction: every transaction end clears it, so a commit that is followed by
    a nested release, a rollback or a bare ``close()`` is never mistaken for one.
    """

    def __init__(self) -> None:
        self.pending: list[_PendingPut] = []
        self._committed = False

    def on_commit(self, session: Session) -> None:
        self._committed = True

    def on_soft_rollback(
        self, session: Session, previous_transaction: SessionTransaction
    ) -> None:
        self._committed = False
        if previous_transaction.parent is None:
            self.pending.clear()
            return
        self.pending = [
            put
            for put in self.pending
            if not _encloses(previous_transaction, put.origin)
        ]

    def on_transaction_end(
        self, session: Session, transaction: SessionTransaction
    ) -> None:
        committed, self._committed = self._committed, False
        if transaction.parent is not None:
            return
        pending, self.pending = self.pending, []
        if not committed:
            return
        for put in pending:
            try:
                put_constraint_set(
                    put.store, put.constraint_set, store_key=put.store_key
                )
            except Exception:
                # The database commit already happened and cannot be undone here;
                # raising would only make the caller believe it failed.
                logger.exception(
                    "committed, but caching the ConstraintSet for portfolio %s in "
                    "the Store failed; re-profile or re-run the profiler to heal it",
                    put.constraint_set.portfolio_id,
                )


def _encloses(
    transaction: SessionTransaction, inner: SessionTransaction | None
) -> bool:
    """Whether ``inner`` is ``transaction`` or nested anywhere beneath it."""
    while inner is not None:
        if inner is transaction:
            return True
        inner = inner.parent
    return False


_PUBLISHER_KEY = "fund.constraint_set_publisher"


def put_constraint_set_after_commit(
    session: Session,
    store: BaseStore | None,
    constraint_set: ConstraintSet,
    *,
    store_key: str,
) -> None:
    """Cache ``constraint_set`` in the Store only once ``session`` commits.

    The Store is a cache of the ``mifid_profiles`` row, so it must never get ahead
    of it: an approval whose database transaction is lost (commit failure, killed
    process) would otherwise leave a risk profile that ``fund run`` resolves with no
    record behind it. The put is queued and runs after the root transaction commits;
    a rollback of the session, or of the SAVEPOINT it was queued in, drops it. A
    ``None`` store is a no-op. A Store failure after the commit is logged, never
    raised.

    Args:
        session: The session whose commit makes the cached profile durable.
        store: LangGraph store instance, or ``None`` when there is nothing to cache.
        constraint_set: The active constraint set to publish.
        store_key: Key under which the set is stored.
    """
    if store is None:
        return
    publisher = session.info.get(_PUBLISHER_KEY)
    if publisher is None:
        publisher = _CommitPublisher()
        session.info[_PUBLISHER_KEY] = publisher
        event.listen(session, "after_commit", publisher.on_commit)
        event.listen(session, "after_soft_rollback", publisher.on_soft_rollback)
        event.listen(session, "after_transaction_end", publisher.on_transaction_end)
    origin = session.get_nested_transaction()
    # A cache miss is refilled on every resolve; one identical queued put is enough.
    if any(
        put.store is store
        and put.store_key == store_key
        and put.origin is origin
        and put.constraint_set == constraint_set
        for put in publisher.pending
    ):
        return
    publisher.pending.append(
        _PendingPut(
            store=store,
            constraint_set=constraint_set,
            store_key=store_key,
            origin=origin,
        )
    )


def resolve_constraint_set(
    store: BaseStore, ref: ConstraintSetRef, *, session: Session | None = None
) -> ConstraintSet | None:
    """Resolve a ``ConstraintSetRef`` back to its persisted ``ConstraintSet``.

    Reads the Store item at namespace ``(ref.portfolio_id,)`` / key
    ``ref.store_key``. When the Store has nothing and a ``session`` is given, the
    portfolio's active ``mifid_profiles`` row (the system of record) is used instead
    and the Store entry is re-seeded once ``session`` commits, so a lost or recreated
    cache never strands a profiled portfolio.

    Args:
        store: LangGraph store instance to query.
        ref: Reference produced by a prior ``put_constraint_set`` call.
        session: Optional session enabling the fallback to the profile row.

    Returns:
        The constraint set, or ``None`` if the reference is stale or missing (and,
        with a ``session``, the portfolio has no profile).
    """
    item = store.get((ref.portfolio_id,), ref.store_key)
    if item is not None:
        return ConstraintSet.model_validate(item.value)
    if session is None:
        return None
    try:
        portfolio_id = uuid.UUID(ref.portfolio_id)
    except ValueError:
        return None
    profile = MifidProfileRepository(session).get_active(portfolio_id)
    if profile is None:
        return None
    constraint_set = ConstraintSet.model_validate(profile.constraint_set)
    # Deferred like every other cache write: the row read here may belong to a
    # transaction that is still open and could yet roll back.
    put_constraint_set_after_commit(
        session, store, constraint_set, store_key=ref.store_key
    )
    return constraint_set


__all__ = [
    "MifidProfileRepository",
    "put_constraint_set",
    "put_constraint_set_after_commit",
    "resolve_constraint_set",
]
