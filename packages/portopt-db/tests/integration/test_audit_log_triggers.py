"""The append-only guarantee of ``human_audit_log`` holds on a real PostgreSQL server.

The compliance trail is protected by a row-level ``BEFORE UPDATE OR DELETE`` trigger
and a statement-level ``BEFORE TRUNCATE`` trigger (revisions ``f1a2b3c4d5e6`` and
``77f236075b2e``). Until now a test only rendered their SQL offline; none ran a
mutation and watched the server refuse it.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from portopt_db.models.agent.human_audit_log import HumanAuditLog

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE human_audit_log SET rationale = 'rewritten'",
        "DELETE FROM human_audit_log",
        "TRUNCATE human_audit_log",
    ],
)
def test_mutating_the_audit_log_is_refused(pg_session: Session, statement: str) -> None:
    """UPDATE, DELETE and TRUNCATE each fail with the trigger's own message."""
    pg_session.add(HumanAuditLog(actor="human", action="approve"))
    pg_session.flush()

    with (
        pytest.raises(DBAPIError, match="append-only"),
        pg_session.begin_nested(),
    ):
        pg_session.execute(text(statement))


def test_inserting_into_the_audit_log_is_allowed(pg_session: Session) -> None:
    """The guard blocks mutation only; appending a row still works."""
    pg_session.add(HumanAuditLog(actor="human", action="approve"))
    pg_session.flush()

    count = pg_session.execute(
        text("SELECT count(*) FROM human_audit_log")
    ).scalar_one()
    assert count >= 1
