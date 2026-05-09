"""Tests for ``research/_cli.py`` — issue #550 (9-flag CLI surface)."""

from __future__ import annotations

import argparse
from datetime import date

import pytest

from research._cli import _parse_iso_date, build_parser

# ---------------------------------------------------------------------------
# _parse_iso_date
# ---------------------------------------------------------------------------


class TestParseIsoDate:
    def test_when_valid_iso_string_returns_date(self) -> None:
        assert _parse_iso_date("2026-04-01") == date(2026, 4, 1)

    def test_when_garbage_string_raises_argument_type_error(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_iso_date("not-a-date")

    def test_when_us_format_raises_argument_type_error(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_iso_date("04/01/2026")

    def test_when_empty_string_raises_argument_type_error(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_iso_date("")


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_when_called_returns_argument_parser(self) -> None:
        assert isinstance(build_parser(), argparse.ArgumentParser)

    def test_when_no_args_defaults_match_spec(self) -> None:
        ns = build_parser().parse_args([])
        assert ns.tax_rate == 0.26
        assert ns.base_currency == "EUR"
        assert ns.robust is False
        assert ns.persist is False
        assert ns.start_date is None
        assert ns.end_date is None
        assert ns.seed == 42

    def test_when_existing_flags_default_passed_through(self) -> None:
        ns = build_parser().parse_args([])
        assert hasattr(ns, "rebalance_freq")
        assert hasattr(ns, "n_selected")
        assert hasattr(ns, "cost_bps")

    def test_when_robust_flag_set_value_true(self) -> None:
        ns = build_parser().parse_args(["--robust"])
        assert ns.robust is True

    def test_when_persist_flag_set_value_true(self) -> None:
        ns = build_parser().parse_args(["--persist"])
        assert ns.persist is True

    def test_when_tax_rate_supplied_value_returned(self) -> None:
        ns = build_parser().parse_args(["--tax-rate", "0.30"])
        assert ns.tax_rate == 0.30

    def test_when_base_currency_supplied_value_returned(self) -> None:
        ns = build_parser().parse_args(["--base-currency", "USD"])
        assert ns.base_currency == "USD"

    def test_when_start_date_supplied_returns_date_object(self) -> None:
        ns = build_parser().parse_args(["--start-date", "2024-01-01"])
        assert ns.start_date == date(2024, 1, 1)

    def test_when_end_date_supplied_returns_date_object(self) -> None:
        ns = build_parser().parse_args(["--end-date", "2026-04-01"])
        assert ns.end_date == date(2026, 4, 1)

    def test_when_seed_supplied_int_returned(self) -> None:
        ns = build_parser().parse_args(["--seed", "777"])
        assert ns.seed == 777

    def test_when_invalid_start_date_parser_exits(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--start-date", "garbage"])

    def test_when_help_listed_all_nine_flags_present(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        out = capsys.readouterr().out
        for flag in (
            "--rebalance-freq",
            "--n-selected",
            "--cost-bps",
            "--tax-rate",
            "--base-currency",
            "--robust",
            "--persist",
            "--start-date",
            "--end-date",
            "--seed",
        ):
            assert flag in out
