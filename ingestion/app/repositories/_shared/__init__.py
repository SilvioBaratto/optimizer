"""Shared Repositories.

Cross-cutting repository infrastructure used by all domain repos.
"""

from portopt_db.repositories.database_admin import (
    APP_TABLES,
    DatabaseAdminRepository,
)

from app.repositories._shared.base import (
    BaseRepository,
    RepositoryBase,
    _get_table,
)

__all__ = [
    "APP_TABLES",
    "BaseRepository",
    "DatabaseAdminRepository",
    "RepositoryBase",
    "_get_table",
]
