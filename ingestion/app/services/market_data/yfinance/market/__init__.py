"""Module-level API clients (no BaseClient)."""

from .search import SearchClient
from .sectors import SectorsClient

__all__ = ["SearchClient", "SectorsClient"]
