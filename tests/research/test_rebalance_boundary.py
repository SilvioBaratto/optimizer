"""Cycle-3 §672 — verify and harden _rebalance.py boundary (issue #672).

Covers gaps NOT addressed in test_optimization_helpers.py or
test_rebalance_wiring.py:

- Threshold constant values are exactly 0.0 / 1.5
- Only permitted imports (optimizer.rebalancing, research.optimization._retighten,
  stdlib, numpy, pandas, logging)
- No circular import between _retighten and _rebalance
- All __all__ symbols in research/optimization/__init__ are importable
- _decide_rebalance when should_rebalance_hybrid returns False
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REBALANCE_PY = (
    Path(__file__).parent.parent.parent / "research" / "optimization" / "_rebalance.py"
)

_PERMITTED_INTERNAL = {
    "optimizer.rebalancing",
    "research.optimization._retighten",
}


def _collect_imports(source_path: Path) -> list[str]:
    """Return every top-level module referenced in ``import`` / ``from … import``."""
    tree = ast.parse(source_path.read_text())
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
    return modules


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestThresholdConstants:
    """Threshold constants are hard-coded tunable knobs; values must be exact."""

    def test_neg_threshold_is_zero(self) -> None:
        from research.optimization._rebalance import _HOCKEY_STICK_NEG_THRESHOLD

        assert _HOCKEY_STICK_NEG_THRESHOLD == 0.0

    def test_pos_threshold_is_1_5(self) -> None:
        from research.optimization._rebalance import _HOCKEY_STICK_POS_THRESHOLD

        assert _HOCKEY_STICK_POS_THRESHOLD == 1.5

    def test_neg_threshold_exported_via_init(self) -> None:
        import research.optimization as ro

        assert ro._HOCKEY_STICK_NEG_THRESHOLD == 0.0

    def test_pos_threshold_exported_via_init(self) -> None:
        import research.optimization as ro

        assert ro._HOCKEY_STICK_POS_THRESHOLD == 1.5


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------


class TestImportBoundary:
    """Only permitted modules may be imported by _rebalance.py."""

    def test_no_api_imports(self) -> None:
        modules = _collect_imports(_REBALANCE_PY)
        api_imports = [m for m in modules if m and m.startswith("api")]
        assert api_imports == [], f"Forbidden api imports found: {api_imports}"

    def test_no_research_data_imports(self) -> None:
        modules = _collect_imports(_REBALANCE_PY)
        data_imports = [m for m in modules if m and m.startswith("research.data")]
        assert data_imports == [], f"Forbidden research.data imports: {data_imports}"

    def test_no_research_pipeline_imports(self) -> None:
        modules = _collect_imports(_REBALANCE_PY)
        pipeline_imports = [
            m for m in modules if m and m.startswith("research.pipeline")
        ]
        assert pipeline_imports == [], (
            f"Forbidden research.pipeline imports: {pipeline_imports}"
        )

    def test_no_research_cli_imports(self) -> None:
        modules = _collect_imports(_REBALANCE_PY)
        cli_imports = [m for m in modules if m and m.startswith("research.cli")]
        assert cli_imports == [], f"Forbidden research.cli imports: {cli_imports}"

    def test_all_internal_imports_are_permitted(self) -> None:
        modules = _collect_imports(_REBALANCE_PY)
        non_stdlib = [
            m
            for m in modules
            if m
            and not m.startswith("__future__")
            and m not in ("logging", "numpy", "pandas", "np", "pd")
            and not m.startswith("numpy")
            and not m.startswith("pandas")
        ]
        for mod in non_stdlib:
            permitted = any(mod.startswith(p) for p in _PERMITTED_INTERNAL)
            assert permitted, (
                f"Module '{mod}' is not in the permitted import list: "
                f"{sorted(_PERMITTED_INTERNAL)}"
            )


# ---------------------------------------------------------------------------
# Circular import check
# ---------------------------------------------------------------------------


class TestNoCircularImport:
    """_retighten must not back-import from _rebalance."""

    def test_no_circular_import(self) -> None:
        # Verify that both modules are already importable (they are loaded as part of
        # the test session).  We do NOT evict them from sys.modules — doing so would
        # rebind module objects and break patch() targets in other tests that run
        # after this one.
        try:
            importlib.import_module("research.optimization._retighten")
            importlib.import_module("research.optimization._rebalance")
        except ImportError as exc:
            pytest.fail(f"Circular import detected: {exc}")

    def test_retighten_does_not_import_rebalance(self) -> None:
        retighten_path = _REBALANCE_PY.parent / "_retighten.py"
        modules = _collect_imports(retighten_path)
        rebalance_imports = [m for m in modules if m and "_rebalance" in m]
        assert rebalance_imports == [], (
            f"_retighten imports _rebalance: {rebalance_imports}"
        )


# ---------------------------------------------------------------------------
# __all__ completeness
# ---------------------------------------------------------------------------


class TestInitAllExportable:
    """Every symbol declared in __all__ must resolve at runtime."""

    def test_all_symbols_importable_from_package(self) -> None:
        import research.optimization as ro

        missing = [sym for sym in ro.__all__ if not hasattr(ro, sym)]
        assert missing == [], f"Symbols in __all__ not importable: {missing}"

    def test_rebalance_symbols_in_all(self) -> None:
        import research.optimization as ro

        for sym in (
            "_HOCKEY_STICK_NEG_THRESHOLD",
            "_HOCKEY_STICK_POS_THRESHOLD",
            "_decide_rebalance",
            "_hockey_stick_warn",
        ):
            assert sym in ro.__all__, f"'{sym}' missing from __all__"


# ---------------------------------------------------------------------------
# _decide_rebalance — False branch
# ---------------------------------------------------------------------------


class TestDecideRebalanceFalseBranch:
    """_decide_rebalance propagates a (False, reason) pair from
    should_rebalance_hybrid."""

    def test_when_should_rebalance_hybrid_returns_false(self) -> None:
        prev = np.array([0.5, 0.5])
        target = np.array([0.52, 0.48])
        current = cast(pd.Timestamp, pd.Timestamp("2025-04-01"))
        last_review = cast(pd.Timestamp, pd.Timestamp("2025-01-01"))

        with patch(
            "research.optimization._rebalance.should_rebalance_hybrid",
            return_value=(False, "within_threshold"),
        ):
            decision, reason = __import__(
                "research.optimization._rebalance",
                fromlist=["_decide_rebalance"],
            )._decide_rebalance(
                prev_weights=prev,
                target_weights=target,
                current_date=current,
                last_review_date=last_review,
            )

        assert decision is False
        assert reason == "within_threshold"

    def test_when_should_rebalance_hybrid_returns_false_no_calendar(self) -> None:
        """Ensure (False, 'not_review_date') is propagated unchanged."""
        prev = np.array([0.4, 0.6])
        target = np.array([0.4, 0.6])
        current = cast(pd.Timestamp, pd.Timestamp("2025-04-10"))  # Mid-quarter
        last_review = cast(pd.Timestamp, pd.Timestamp("2025-01-01"))

        with patch(
            "research.optimization._rebalance.should_rebalance_hybrid",
            return_value=(False, "not_review_date"),
        ):
            from research.optimization._rebalance import _decide_rebalance

            decision, reason = _decide_rebalance(
                prev_weights=prev,
                target_weights=target,
                current_date=current,
                last_review_date=last_review,
            )

        assert decision is False
        assert reason == "not_review_date"
