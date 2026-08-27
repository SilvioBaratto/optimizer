"""Compatibility shim — moved to ``portopt_db.repository`` (P1 of the portopt-db
extraction). Re-exported here so existing ``app.repositories._shared`` imports
keep working until repositories move (P3); removed in P4.2.
"""

from portopt_db.repository import (
    BaseRepository,
    CreateSchemaType,
    ModelType,
    RepositoryBase,
    UpdateSchemaType,
    _get_table,
)

__all__ = [
    "BaseRepository",
    "CreateSchemaType",
    "ModelType",
    "RepositoryBase",
    "UpdateSchemaType",
    "_get_table",
]
