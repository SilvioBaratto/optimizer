"""Argparse surface for research pipeline (Cycle 5 §13).

Exposes the 9-flag CLI as a separate module so the parser is unit-testable
without importing the full pipeline.  The pipeline entry point calls
:func:`build_parser` then forwards the resulting :class:`argparse.Namespace`
to :func:`main`.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

# Re-imported to keep CLI defaults in lock-step with pipeline constants.
from research.pipeline._load import N_SELECTED, REBALANCE_FREQ

_DEFAULT_TAX_RATE: float = 0.26
_DEFAULT_COST_BPS: float = 10.0
_DEFAULT_BASE_CURRENCY: str = "EUR"
_DEFAULT_SEED: int = 42
_ISO_DATE_FMT: str = "%Y-%m-%d"


def _parse_iso_date(value: str) -> date:
    """Parse an ISO ``YYYY-MM-DD`` string; raise on invalid input."""
    try:
        return datetime.strptime(value, _ISO_DATE_FMT).date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid ISO date {value!r} (expected YYYY-MM-DD)"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the 9-flag argparse parser for the research pipeline."""
    parser = argparse.ArgumentParser(
        description="Stock selection research pipeline (Cycle 5 CLI)",
    )
    _add_existing_flags(parser)
    _add_new_flags(parser)
    return parser


def _add_existing_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rebalance-freq",
        type=int,
        default=REBALANCE_FREQ,
        help=f"Rebalancing frequency in trading days (default: {REBALANCE_FREQ})",
    )
    parser.add_argument(
        "--n-selected",
        type=int,
        default=N_SELECTED,
        help=f"Number of stocks to select (default: {N_SELECTED})",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=_DEFAULT_COST_BPS,
        help=f"Transaction cost in basis points (default: {_DEFAULT_COST_BPS})",
    )


def _add_new_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tax-rate",
        type=float,
        default=_DEFAULT_TAX_RATE,
        help=f"Capital-gains tax rate (default: {_DEFAULT_TAX_RATE})",
    )
    parser.add_argument(
        "--base-currency",
        type=str,
        default=_DEFAULT_BASE_CURRENCY,
        help=f"FX base currency (default: {_DEFAULT_BASE_CURRENCY})",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use robust MeanRisk with mu uncertainty set",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Write the final snapshot to the API DB on 17/17 PASS",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_iso_date,
        default=None,
        help="Slice prices >= this ISO YYYY-MM-DD (default: full history)",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_iso_date,
        default=None,
        help="Slice prices <= this ISO YYYY-MM-DD (default: full history)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"RNG seed for bootstrap uncertainty (default: {_DEFAULT_SEED})",
    )


__all__ = ["_parse_iso_date", "build_parser"]
