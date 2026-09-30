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

    def connect_args(self) -> dict[str, Any]:
        """Build psycopg2 ``connect()`` kwargs for this config.

        Returns:
            Mapping passed verbatim to ``psycopg2.connect``, including TCP
            keepalive settings that prevent NAT/firewall timeouts from silently
            dropping long-idle connections.
        """
        return {
            "application_name": self.application_name,
            "connect_timeout": self.connect_timeout,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 3,
        }


__all__ = ["DbConfig"]
