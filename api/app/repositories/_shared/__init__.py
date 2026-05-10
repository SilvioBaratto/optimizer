"""Shared Repositories.

Cross-cutting repository infrastructure used by all domain repos.
"""

from app.repositories._shared.base import (
    BaseRepository,
    RepositoryBase,
    _get_table,
)
from app.repositories._shared.database_admin_repository import (
    APP_TABLES,
    DatabaseAdminRepository,
)

__all__ = [
    "APP_TABLES",
    "BaseRepository",
    "DatabaseAdminRepository",
    "RepositoryBase",
    "_get_table",
]
