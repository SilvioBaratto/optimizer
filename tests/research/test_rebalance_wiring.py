"""Cycle-3 §11 hybrid rebalance wiring tests (issue #539)."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from research import stock_selection_pipeline as ssp
from research._optimization import _decide_rebalance


class TestDecideRebalance:
    """`_decide_rebalance` returns the hybrid (decision, reason) pair."""

    def test_when_previous_weights_none_then_cold_start(self) -> None:
        decision, reason = _decide_rebalance(
            prev_weights=None,
            target_weights=np.array([0.5, 0.5]),
            current_date=pd.Timestamp("2025-04-01"),
            last_review_date=pd.Timestamp("2025-01-01"),
        )
        assert decision is False
        assert reason == "cold_start"

    def test_when_previous_weights_supplied_then_calls_should_rebalance_hybrid(
        self,
    ) -> None:
        prev = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])
        current = pd.Timestamp("2025-04-01")
        last_review = pd.Timestamp("2025-01-01")

        with patch(
            "research._optimization.should_rebalance_hybrid",
            return_value=(True, "threshold_met"),
        ) as mock_should:
            decision, reason = _decide_rebalance(
                prev_weights=prev,
                target_weights=target,
                current_date=current,
                last_review_date=last_review,
            )

        assert (decision, reason) == (True, "threshold_met")
        # Positional args: current_weights, target_weights, config,
        # current_date, last_review_date
        call_args = mock_should.call_args
        assert call_args is not None
        positional = call_args.args
        np.testing.assert_array_equal(positional[0], prev)
        np.testing.assert_array_equal(positional[1], target)
        # config arg is HybridRebalancingConfig
        from optimizer.rebalancing import HybridRebalancingConfig

        assert isinstance(positional[2], HybridRebalancingConfig)
        assert positional[3] == current
        assert positional[4] == last_review

    def test_when_cold_start_then_logger_info_recorded(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="research._optimization"):
            _decide_rebalance(
                prev_weights=None,
                target_weights=np.array([1.0]),
                current_date=pd.Timestamp("2025-04-01"),
                last_review_date=pd.Timestamp("2025-01-01"),
            )
        text = caplog.text.lower()
        assert "cold_start" in text or "cold start" in text


class TestMainRebalanceWiring:
    """`main` derives decision and attaches it to the result object."""

    def _setup_main_mocks(self, *, prev_weights: np.ndarray | None) -> Any:
        """Patch every dependency of main() that needs a real DB / network."""
        from types import SimpleNamespace

        idx = pd.bdate_range("2020-01-01", periods=5)
        weights_series = pd.Series([0.5, 0.5], index=["AAA", "BBB"])
        result_stub = SimpleNamespace(
            summary={"sharpe_ratio": 1.0},
            net_sharpe_ratio=None,
            weights=weights_series,
            net_returns=None,
            retighten_trace=[],
        )
        assembly_stub = SimpleNamespace(
            prices=pd.DataFrame(100.0, index=idx, columns=["AAA", "BBB"]),
            sector_mapping={"AAA": "Tech", "BBB": "Finance"},
            previous_weights=prev_weights,
        )
        return assembly_stub, result_stub

    def test_when_review_file_present_then_read_returns_file_date(
        self, tmp_path: Any
    ) -> None:
        review_file = tmp_path / "last_review_date.txt"
        review_file.write_text("2025-01-15\n")
        with patch.object(ssp, "_LAST_REVIEW_DATE_FILE", review_file):
            got = ssp._read_last_review_date(pd.Timestamp("2025-04-01"))
        assert got == pd.Timestamp("2025-01-15")

    def test_when_review_file_missing_then_read_returns_prior_quarter_end(
        self, tmp_path: Any
    ) -> None:
        review_file = tmp_path / "missing.txt"
        with patch.object(ssp, "_LAST_REVIEW_DATE_FILE", review_file):
            got = ssp._read_last_review_date(pd.Timestamp("2025-04-15"))
        # Prior quarter-end of Q2 2025 = end of Q1 2025
        assert got.normalize() == pd.Timestamp("2025-03-31")

    def test_when_write_called_then_file_contains_iso_date(self, tmp_path: Any) -> None:
        review_file = tmp_path / "last_review_date.txt"
        with patch.object(ssp, "_LAST_REVIEW_DATE_FILE", review_file):
            ssp._write_last_review_date(pd.Timestamp("2025-04-01"))
        assert review_file.read_text().strip() == "2025-04-01"

    def test_when_main_runs_then_result_has_rebalance_decision_tuple(
        self,
    ) -> None:
        from types import SimpleNamespace

        assembly_stub, result_stub = self._setup_main_mocks(
            prev_weights=np.array([0.5, 0.5])
        )

        with (
            patch.object(
                ssp,
                "load_data",
                return_value=(assembly_stub, {"AAA": "United States"}, None),
            ),
            patch.object(
                ssp, "screen_investable", return_value=pd.Index(["AAA", "BBB"])
            ),
            patch.object(
                ssp, "_materialise_clean_returns", return_value=pd.DataFrame()
            ),
            patch.object(
                ssp,
                "build_history",
                return_value=(
                    {},
                    pd.DataFrame(),
                    SimpleNamespace(succeeded_dates=1, total_dates=1),
                ),
            ),
            patch.object(ssp, "validate_is", return_value=None),
            patch.object(
                ssp,
                "validate_oos",
                return_value=SimpleNamespace(per_fold_ic=pd.DataFrame()),
            ),
            patch.object(ssp, "_check_factor_coverage", return_value=None),
            patch.object(
                ssp,
                "classify_and_tilt",
                return_value=(SimpleNamespace(value="expansion"), {}),
            ),
            patch.object(ssp, "optimize_portfolio", return_value=result_stub),
            patch.object(ssp, "report_performance", return_value=(0, [])),
            patch.object(ssp, "_apply_terminal_gate"),
            patch.object(
                ssp,
                "_decide_rebalance",
                return_value=(True, "threshold_met"),
            ) as mock_decide,
        ):
            ssp.main(n_selected=25)

        mock_decide.assert_called_once()
        assert hasattr(result_stub, "rebalance_decision")
        assert result_stub.rebalance_decision == (True, "threshold_met")
