"""SQLAlchemy models — single source of truth for the shared schema.

Every model module MUST be imported here so ``Base.metadata`` is complete for
Alembic autogenerate and SQLite ``create_all`` in tests.
"""
