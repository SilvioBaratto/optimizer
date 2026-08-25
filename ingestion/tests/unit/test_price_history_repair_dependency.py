"""Guard the scikit-learn runtime dependency that yfinance price-repair needs.

``YFinanceClient.fetch_history`` defaults ``repair=True``. yfinance's repair path
(``PriceHistory._reconstruct_intervals_batch``) does ``from sklearn.cluster import
DBSCAN``. If scikit-learn is dropped from ``ingestion/pyproject.toml`` the import
raises ``ModuleNotFoundError`` *at fetch time* — not at startup — and every ticker
yfinance tries to reconstruct comes back empty, so the universe build silently
drops ~22% of live stocks (disproportionately non-US). This test fails loudly the
moment that dependency goes missing, long before it reaches a scheduled run.

See scripts/debug_universe_scale.py for the investigation that traced the collapse
to this import.
"""

from __future__ import annotations

import importlib


def test_sklearn_cluster_dbscan_is_importable() -> None:
    """The exact symbol yfinance price-repair imports must be available."""
    module = importlib.import_module("sklearn.cluster")
    assert hasattr(module, "DBSCAN")
