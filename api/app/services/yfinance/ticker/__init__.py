"""Ticker-scoped sub-clients (extend BaseClient)."""

from .analysis import AnalysisClient
from .corporate_actions import CorporateActionsClient
from .financials import FinancialsClient
from .funds import FundsClient
from .holders import HoldersClient
from .metadata import MetadataClient

__all__ = [
    "AnalysisClient",
    "CorporateActionsClient",
    "FinancialsClient",
    "FundsClient",
    "HoldersClient",
    "MetadataClient",
]
