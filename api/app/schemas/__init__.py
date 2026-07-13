"""Schemas package.

Pydantic models for the ingestion daemon. With the HTTP layer gone these are no
longer request/response bodies — they are the typed argument objects the CLI and
the scheduler pass into service functions (``YFinanceFetchRequest``,
``MacroFetchRequest``, ``UniverseBuildRequest``, …) and the progress payloads
those services report back.

Import from the domain module directly (``app.schemas.macro.macro_regime``);
this package deliberately re-exports nothing.
"""
