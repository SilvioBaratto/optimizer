"""Injected database configuration.

``portopt_db`` never imports an application ``settings`` object — the consumer
(ingestion, fund/, …) builds a ``DbConfig`` from its own config and passes it to
``DatabaseManager``. Keeps the package free of any app-config coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DbConfig:
    """Connection and QueuePool tunables passed to the SQLAlchemy engine.

    Attributes:
        url: SQLAlchemy database URL (e.g. ``postgresql+psycopg2://...``).
        echo: Log every SQL statement; expensive, for debugging only.
        pool_size: Number of persistent connections kept open in the pool.
        max_overflow: Extra connections allowed beyond ``pool_size`` under load.
        pool_timeout: Seconds to wait for a free connection before raising.
        pool_recycle: Seconds before a connection is replaced; prevents silent
            server-side closure of long-idle connections.
        pool_pre_ping: Issue a lightweight ``SELECT 1`` before each checkout to
            discard dead connections without surfacing errors to callers.
        pool_reset_on_return: Transaction state cleanup on connection return;
            ``"rollback"`` avoids accidentally committing leftover state.
        application_name: Label shown in ``pg_stat_activity`` for observability.
        connect_timeout: Seconds before a TCP connection attempt is abandoned.
        lock_timeout_ms: Server-side ``lock_timeout`` in milliseconds. A writer
            blocked behind another transaction's row lock fails after this long
            instead of holding a pool connection indefinitely. ``0`` (the
            PostgreSQL default) waits forever.
        statement_timeout_ms: Server-side ``statement_timeout`` in milliseconds;
            ``0`` disables it. Time spent waiting for locks counts toward it.
        idle_in_transaction_timeout_ms: Server-side
            ``idle_in_transaction_session_timeout`` in milliseconds; ``0``
            disables it. Leave off for workloads that keep one transaction open
            across network calls: the server kills the session mid-transaction
            and the whole unit of work is lost.
    """

    url: str
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800
    pool_pre_ping: bool = True
    pool_reset_on_return: str = "rollback"
    application_name: str = "portopt"
    connect_timeout: int = 10
    lock_timeout_ms: int = 0
    statement_timeout_ms: int = 0
    idle_in_transaction_timeout_ms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "lock_timeout_ms",
            "statement_timeout_ms",
            "idle_in_transaction_timeout_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0 (0 disables it)")

    def connect_args(self) -> dict[str, Any]:
        """Build psycopg2 ``connect()`` kwargs for this config.

        Returns:
            Mapping passed verbatim to ``psycopg2.connect``, including TCP
            keepalive settings that prevent NAT/firewall timeouts from silently
            dropping long-idle connections, and an ``options`` entry carrying
            every server-side timeout that is enabled (absent when all are 0).
        """
        args: dict[str, Any] = {
            "application_name": self.application_name,
            "connect_timeout": self.connect_timeout,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 3,
        }
        server_settings = (
            ("lock_timeout", self.lock_timeout_ms),
            ("statement_timeout", self.statement_timeout_ms),
            (
                "idle_in_transaction_session_timeout",
                self.idle_in_transaction_timeout_ms,
            ),
        )
        options = [f"-c {name}={int(ms)}" for name, ms in server_settings if ms]
        if options:
            args["options"] = " ".join(options)
        return args


__all__ = ["DbConfig"]
