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

import uuid
from typing import TYPE_CHECKING, Any

from portopt_db.models import MifidProfile
from portopt_db.repository import RepositoryBase
from sqlalchemy import func, select

from fund.schemas import ConstraintSet, ConstraintSetRef

if TYPE_CHECKING:  # runtime only needs the store's put/get, not the class
    from langgraph.store.base import BaseStore


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
        """
        stmt = (
            select(MifidProfile)
            .where(MifidProfile.portfolio_id == portfolio_id)
            .order_by(MifidProfile.version.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()


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
    """
    store.put(
        (constraint_set.portfolio_id,),
        store_key,
        constraint_set.model_dump(mode="json"),
    )
    return ConstraintSetRef(
        portfolio_id=constraint_set.portfolio_id, store_key=store_key
    )


def resolve_constraint_set(
    store: BaseStore, ref: ConstraintSetRef
) -> ConstraintSet | None:
    """Resolve a ``ConstraintSetRef`` back to its persisted ``ConstraintSet``.

    Reads the Store item at namespace ``(ref.portfolio_id,)`` / key
    ``ref.store_key``; ``None`` when nothing is cached for that reference.
    """
    item = store.get((ref.portfolio_id,), ref.store_key)
    if item is None:
        return None
    return ConstraintSet.model_validate(item.value)


__all__ = [
    "MifidProfileRepository",
    "put_constraint_set",
    "resolve_constraint_set",
]
