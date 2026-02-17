"""Pre-selection pipeline assembly."""

from optimizer.pre_selection._config import PreSelectionConfig
from optimizer.pre_selection._pipeline import build_preselection_pipeline

__all__ = [
    "PreSelectionConfig",
    "build_preselection_pipeline",
]
