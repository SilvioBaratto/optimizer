"""Injected database configuration.

``portopt_db`` never imports an application ``settings`` object — the consumer
(ingestion, a future fund/, …) builds a :class:`DbConfig` from its own config and
hands it to :class:`portopt_db.engine.DatabaseManager`. Keeps the package free of
any app-config coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DbConfig:
    """Connection + QueuePool tunables for the SQLAlchemy engine."""

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
        """psycopg2-specific connect args (application name + keepalives)."""
        return {
            "application_name": self.application_name,
            "connect_timeout": self.connect_timeout,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 3,
        }


__all__ = ["DbConfig"]
