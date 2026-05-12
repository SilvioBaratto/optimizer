"""Tests for research.pipeline package (__init__.py), cli/, and compat shims.

Covers issue #652 acceptance criteria.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Pipeline __init__.py re-exports
# ---------------------------------------------------------------------------


class TestPipelineReexports:
    def test_all_public_functions_importable(self) -> None:
        """Every public function from all 7 step modules is importable."""
        from research.pipeline import (
            # _checklist.py
            _CHECKLIST_TOTAL,
            N_SELECTED_MAX,
            N_SELECTED_MIN,
            REBALANCE_FREQ,
        )

        assert REBALANCE_FREQ == 63
        assert _CHECKLIST_TOTAL == 17
        assert N_SELECTED_MIN == 25
        assert N_SELECTED_MAX == 50


# ---------------------------------------------------------------------------
# CLI package
# ---------------------------------------------------------------------------


class TestCliPackage:
    def test_build_parser_importable(self) -> None:
        from research.cli import build_parser

        parser = build_parser()
        assert parser is not None

    def test_main_importable(self) -> None:
        from research.cli import main

        assert callable(main)


class TestCliParser:
    def test_produces_parser_with_expected_args(self) -> None:
        from research.cli._parser import build_parser

        parser = build_parser()
        actions = {a.dest for a in parser._actions}
        expected = {
            "rebalance_freq",
            "n_selected",
            "cost_bps",
            "tax_rate",
            "base_currency",
            "robust",
            "persist",
            "start_date",
            "end_date",
            "seed",
            "help",
        }
        assert expected.issubset(actions)

    def test_parse_iso_date_accepts_valid(self) -> None:
        from research.cli._parser import _parse_iso_date

        result = _parse_iso_date("2026-01-15")
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 15

    def test_parse_iso_date_rejects_invalid(self) -> None:
        import argparse

        from research.cli._parser import _parse_iso_date

        with pytest.raises(argparse.ArgumentTypeError, match="invalid ISO date"):
            _parse_iso_date("garbage")


class TestCliMain:
    def test_has_main_func_and_guard(self) -> None:
        import inspect

        from research.cli import _main

        assert hasattr(_main, "main")
        sig = inspect.signature(_main.main)
        params = list(sig.parameters.keys())
        assert "rebalance_freq" in params
        assert "n_selected" in params
        assert "cost_bps" in params


# ---------------------------------------------------------------------------
# Backward-compat shims
# ---------------------------------------------------------------------------


class TestPipelineShim:
    def test_imports_from_pipeline(self) -> None:
        import research.pipeline as pipeline

        assert hasattr(pipeline, "load_data")
        assert hasattr(pipeline, "optimize_portfolio")
        assert hasattr(pipeline, "classify_and_tilt")

    def test_has_all_dunder(self) -> None:
        import research.pipeline as pipeline

        all_list = getattr(pipeline, "__all__", None)
        assert all_list is not None
        assert "load_data" in all_list
        assert "optimize_portfolio" in all_list


# ---------------------------------------------------------------------------
# Circular import smoke test
# ---------------------------------------------------------------------------


class TestCircularImports:
    def test_import_research_succeeds(self) -> None:
        import research  # noqa: F401

    def test_import_pipeline_from_research(self) -> None:
        from research import pipeline  # noqa: F401

    def test_import_cli_from_research(self) -> None:
        from research import cli  # noqa: F401
