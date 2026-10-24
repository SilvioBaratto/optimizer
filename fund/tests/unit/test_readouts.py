"""Unit tests for the stdlib-only readout helpers (bar-meter, risk verdict)."""

from __future__ import annotations

import pytest

from fund import readouts_compute
from fund.tui.readouts import attribution_bar, render_readout, risk_verdict


def test_bar_meter_largest_magnitude_fills_the_width():
    """``|value| == scale`` is the largest contribution, so every cell fills."""
    assert attribution_bar(0.31, 0.31, width=10) == "█" * 10


def test_bar_meter_cell_count_is_proportional():
    """Half the scale at width 8 yields four full blocks (32 eighths)."""
    assert attribution_bar(0.5, 1.0, width=8) == "█" * 4


def test_bar_meter_renders_a_partial_eighth_cell():
    """One eighth of a single full cell renders the thinnest ramp glyph."""
    assert attribution_bar(1.0, 8.0, width=1) == "▏"


def test_bar_meter_zero_is_empty():
    """A zero contribution renders no cells."""
    assert attribution_bar(0.0, 0.42, width=10) == ""


def test_bar_meter_marks_negatives_with_a_sign_glyph():
    """A negative contribution carries a trailing sign marker, not colour alone."""
    bar = attribution_bar(-0.08, 0.42, width=10)
    assert bar.endswith(" (-)")
    assert bar[0] in "▏▎▍▌▋▊▉█"


def test_bar_meter_clamps_to_the_requested_width():
    """A value beyond the scale never overflows the requested width."""
    assert attribution_bar(5.0, 1.0, width=4) == "█" * 4


def test_bar_meter_nonpositive_scale_is_empty():
    """A non-positive scale yields an empty meter rather than dividing by zero."""
    assert attribution_bar(0.31, 0.0, width=10) == ""


@pytest.mark.parametrize(
    "kind",
    [
        "esg_exclusion",
        "unclassified_sector",
        "leveraged_instrument",
        "complex_instrument",
    ],
)
def test_risk_verdict_reads_a_mandate_screen_violation_as_a_breach(kind):
    """A forbidden holding must never render as an ok or advisory verdict."""
    verdict = risk_verdict([{"type": kind, "asset": "AAA"}])

    assert verdict.label == "breach"


def _results_compute() -> readouts_compute.ResultsCompute:
    """A recompute carrying realistic annualised, small-magnitude figures."""
    return readouts_compute.ResultsCompute(
        frontier=readouts_compute.FrontierPoint(
            mean=0.0412,
            standard_deviation=0.1286,
            sharpe_ratio=0.32,
            n_observations=250,
        ),
        frontier_error=None,
        backtest=readouts_compute.BacktestResult(
            mean=0.05,
            standard_deviation=0.11,
            sharpe_ratio=0.45,
            max_drawdown=0.0197,
            n_observations=120,
            n_folds=3,
            out_of_sample=True,
            window_start="2026-01-01",
            window_end="2026-06-30",
        ),
        backtest_error=None,
    )


def test_frontier_readout_labels_annualised_percentages():
    """A 4% return must not round to 0.00: mean and vol print as percentages and the
    line says the figures are annualised, so they read on the Performance scale."""
    text = render_readout(
        "allocator", "optimize_portfolio", {"weights": {"AAA": 1.0}}, _results_compute()
    )
    assert text is not None
    assert (
        "frontier point · mean +4.12% · vol 12.86% · sharpe 0.32 (annualised)" in text
    )


def test_backtest_readout_shows_drawdown_as_a_percentage():
    """Max drawdown is a bare fraction in the data; the read-out shows it as the
    percentage the Performance strip uses, under an annualised label."""
    text = render_readout(
        "allocator", "optimize_portfolio", {"weights": {"AAA": 1.0}}, _results_compute()
    )
    assert text is not None
    assert "backtest · sharpe 0.45 (annualised) · maxDD 1.97%" in text


def _bounds_constraint_set(**overrides):
    """A recorded ConstraintSet dump with small-magnitude bounds and coefficients."""
    from _fund_fakes import make_constraint_set

    dumped = make_constraint_set().model_dump(mode="json")
    dumped.update(overrides)
    return dumped


def test_bounds_read_out_keeps_sub_percent_weight_limits_distinct():
    """A 0.5% floor and a 7.5% cap must not round to 0.01 / 0.07 (or to zero)."""
    constraint_set = _bounds_constraint_set(
        bounds={"min_weights": 0.005, "max_weights": 0.075, "budget": 1.0}
    )
    text = render_readout(
        "allocator",
        "optimize_portfolio",
        {"constraint_set": constraint_set, "weights": {"AAA": 1.0}},
    )
    assert text is not None
    assert "bounds · min 0.50% · max 7.50% · budget 100.00%" in text


def test_regularisation_read_out_keeps_small_coefficients_visible():
    """l1/l2 of 0.001/0.0005 must not both print 0.00, which reads as unregularised."""
    constraint_set = _bounds_constraint_set(l1_coef=0.001, l2_coef=0.0005)
    text = render_readout(
        "allocator",
        "optimize_portfolio",
        {
            "constraint_set": constraint_set,
            "weights": {"AAA": 1.0},
            "optimizer_config": {"l1_coef": 0.001, "l2_coef": 0.0005},
        },
    )
    assert text is not None
    assert text.count("l1 0.001 · l2 0.0005") == 2  # requested and applied lines


def test_violation_lines_do_not_print_value_and_limit_identically():
    """A 10.04% weight over a 10% cap is a visible breach, not '0.10 (limit 0.10)'."""
    verdict = risk_verdict(
        [
            {"type": "max_weight", "asset": "AAA", "value": 0.1004, "limit": 0.10},
            {"type": "budget", "value": 0.9996, "target": 1.0},
        ]
    )
    assert verdict.lines == (
        "max_weight · AAA 10.04% (limit 10.00%)",
        "budget · 99.96% vs 100.00%",
    )
