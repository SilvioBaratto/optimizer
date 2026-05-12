"""Entry point for ``python -m research.cli``."""

from __future__ import annotations

from research.cli._main import main
from research.cli._parser import build_parser

_args = build_parser().parse_args()
main(
    rebalance_freq=_args.rebalance_freq,
    n_selected=_args.n_selected,
    cost_bps=_args.cost_bps,
    tax_rate=_args.tax_rate,
    base_currency=_args.base_currency,
    robust=_args.robust,
    persist=_args.persist,
    start_date=_args.start_date,
    end_date=_args.end_date,
    seed=_args.seed,
)
