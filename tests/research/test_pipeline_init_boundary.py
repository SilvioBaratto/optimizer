"""Issue #683: research.pipeline must not re-export cross-boundary symbols.

RED tests — these fail before the fix, pass after.
"""

from __future__ import annotations

import pytest


class TestCrossBoundarySymbolsAbsentFromPipelineAll:
    """Symbols owned by research.data / research.optimization must not appear
    in research.pipeline.__all__."""

    def test_assemble_all_not_in_all(self) -> None:
        from research import pipeline

        assert "assemble_all" not in pipeline.__all__

    def test_region_map_not_in_all(self) -> None:
        from research import pipeline

        assert "_REGION_MAP" not in pipeline.__all__

    def test_decide_rebalance_not_in_all(self) -> None:
        from research import pipeline

        assert "_decide_rebalance" not in pipeline.__all__

    def test_solve_with_retighten_not_in_all(self) -> None:
        from research import pipeline

        assert "_solve_with_retighten" not in pipeline.__all__


class TestCrossBoundarySymbolsNotImportableFromPipeline:
    """Explicit import of each removed symbol from research.pipeline must fail."""

    def test_assemble_all_raises_import_error(self) -> None:
        with pytest.raises(ImportError):
            from research.pipeline import assemble_all  # noqa: F401

    def test_region_map_raises_import_error(self) -> None:
        with pytest.raises(ImportError):
            from research.pipeline import _REGION_MAP  # noqa: F401

    def test_decide_rebalance_raises_import_error(self) -> None:
        with pytest.raises(ImportError):
            from research.pipeline import _decide_rebalance  # noqa: F401

    def test_solve_with_retighten_raises_import_error(self) -> None:
        with pytest.raises(ImportError):
            from research.pipeline import _solve_with_retighten  # noqa: F401


class TestCanonicalPathsStillWork:
    """Canonical modules still export these symbols after removal from pipeline."""

    def test_assemble_all_importable_from_data(self) -> None:
        from research.data import assemble_all

        assert callable(assemble_all)

    def test_region_map_importable_from_optimization(self) -> None:
        from research.optimization import _REGION_MAP

        assert isinstance(_REGION_MAP, dict)
        assert len(_REGION_MAP) > 0

    def test_decide_rebalance_importable_from_optimization(self) -> None:
        from research.optimization import _decide_rebalance

        assert callable(_decide_rebalance)

    def test_solve_with_retighten_importable_from_optimization(self) -> None:
        from research.optimization import _solve_with_retighten

        assert callable(_solve_with_retighten)
