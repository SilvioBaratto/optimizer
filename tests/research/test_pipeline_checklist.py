"""Tests for research.pipeline._checklist — Step 7b checklist + terminal gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# _rule
# ---------------------------------------------------------------------------


class TestRule:
    def test_when_passing_then_dict_with_pass_true(self) -> None:
        from research.pipeline._checklist import _rule

        result = _rule("test rule", ok=True, measured="0.05", target="≤ 10%")
        assert result["rule"] == "test rule"
        assert result["pass"] is True
        assert result["measured"] == "0.05"
        assert result["target"] == "≤ 10%"

    def test_when_failing_then_dict_with_pass_false(self) -> None:
        from research.pipeline._checklist import _rule

        result = _rule("another rule", ok=False, measured=0.15, target="< 10%")
        assert result["pass"] is False
        assert result["measured"] == 0.15


# ---------------------------------------------------------------------------
# _sector_weights / _country_weights / _sector_lookup
# ---------------------------------------------------------------------------


class TestSectorWeights:
    def test_aggregates_by_sector(self) -> None:
        from research.pipeline._checklist import _sector_weights

        all_weights = [("AAPL", 0.3), ("MSFT", 0.2), ("JPM", 0.5)]
        sector_mapping = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Finance"}
        result = _sector_weights(all_weights, sector_mapping)
        assert abs(result["Tech"] - 0.5) < 0.001
        assert abs(result["Finance"] - 0.5) < 0.001

    def test_unknown_ticker_goes_to_unknown(self) -> None:
        from research.pipeline._checklist import _sector_weights

        all_weights = [("ZZZ", 1.0)]
        result = _sector_weights(all_weights, {})
        assert result["Unknown"] == 1.0


class TestCountryWeights:
    def test_aggregates_by_country(self) -> None:
        from research.pipeline._checklist import _country_weights

        all_weights = [("AAPL", 0.6), ("VOD", 0.4)]
        country_map = {"AAPL": "United States", "VOD": "United Kingdom"}
        result = _country_weights(all_weights, country_map)
        assert abs(result["United States"] - 0.6) < 0.001
        assert abs(result["United Kingdom"] - 0.4) < 0.001


class TestSectorLookup:
    def test_sums_across_alternative_names(self) -> None:
        from research.pipeline._checklist import _sector_lookup

        sector_w = {"Health Care": 0.05, "Healthcare": 0.03}
        result = _sector_lookup(sector_w, "Health Care", "Healthcare")
        assert abs(result - 0.08) < 0.001

    def test_missing_name_returns_zero(self) -> None:
        from research.pipeline._checklist import _sector_lookup

        result = _sector_lookup({}, "Tech")
        assert result == 0.0


# ---------------------------------------------------------------------------
# _eval_metric_threshold
# ---------------------------------------------------------------------------


class TestEvalMetricThreshold:
    def test_when_value_passes_then_ok_true(self) -> None:
        from research.pipeline._checklist import _eval_metric_threshold

        metrics = {"Portfolio": {"Sharpe": 1.5}}
        result = _eval_metric_threshold(
            metrics,
            "Portfolio",
            "Sharpe",
            "Sharpe > 1.0",
            "> 1.0",
            pass_pred=lambda v: v > 1.0,
        )
        assert result["pass"] is True
        assert result["measured"] == "1.500"

    def test_when_value_fails_then_ok_false(self) -> None:
        from research.pipeline._checklist import _eval_metric_threshold

        metrics = {"Portfolio": {"Sharpe": 0.5}}
        result = _eval_metric_threshold(
            metrics,
            "Portfolio",
            "Sharpe",
            "Sharpe > 1.0",
            "> 1.0",
            pass_pred=lambda v: v > 1.0,
        )
        assert result["pass"] is False
        assert result["measured"] == "0.500"

    def test_when_nan_then_pass_false_measured_na(self) -> None:
        from research.pipeline._checklist import _eval_metric_threshold

        metrics: dict[str, dict[str, float]] = {"Portfolio": {"Sharpe": float("nan")}}
        result = _eval_metric_threshold(
            metrics,
            "Portfolio",
            "Sharpe",
            "Sharpe > 1.0",
            "> 1.0",
            pass_pred=lambda v: v > 1.0,
        )
        assert result["pass"] is False
        assert result["measured"] == "N/A"

    def test_when_metric_missing_then_pass_false(self) -> None:
        from research.pipeline._checklist import _eval_metric_threshold

        metrics: dict[str, dict[str, float]] = {}
        result = _eval_metric_threshold(
            metrics,
            "Portfolio",
            "Sharpe",
            "Sharpe > 1.0",
            "> 1.0",
            pass_pred=lambda v: v > 1.0,
        )
        assert result["pass"] is False
        assert result["measured"] == "N/A"


# ---------------------------------------------------------------------------
# _project_rule_for_json
# ---------------------------------------------------------------------------


class TestProjectRuleForJson:
    def test_converts_nan_measured_to_none(self) -> None:
        from research.pipeline._checklist import _project_rule_for_json

        rule = {
            "rule": "test",
            "pass": False,
            "measured": float("nan"),
            "target": "> 0",
        }
        result = _project_rule_for_json(rule)
        assert result["measured"] is None

    def test_preserves_finite_measured(self) -> None:
        from research.pipeline._checklist import _project_rule_for_json

        rule = {"rule": "test", "pass": True, "measured": 0.05, "target": "> 0"}
        result = _project_rule_for_json(rule)
        assert result["measured"] == 0.05


# ---------------------------------------------------------------------------
# write_metrics_json
# ---------------------------------------------------------------------------


class TestWriteMetricsJson:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        from research.pipeline._checklist import write_metrics_json

        metrics_by_label = {
            "Portfolio": {
                "Ann. Return": 0.12,
                "Ann. Vol": 0.15,
                "Sharpe (rf)": 0.8,
                "Sortino": 1.1,
                "Info Ratio": 0.6,
                "Downside Vol": 0.10,
                "Max Drawdown": -0.15,
            },
        }
        out = write_metrics_json(metrics_by_label, tmp_path)
        assert out.suffix == ".json"
        assert out.exists()
        content = out.read_text()
        assert "net_of_cost" in content
        assert "rf_assumption" in content

    def test_handles_missing_portfolio_label(self, tmp_path: Path) -> None:
        from research.pipeline._checklist import write_metrics_json

        metrics_by_label: dict[str, dict[str, float]] = {}
        out = write_metrics_json(metrics_by_label, tmp_path)
        assert out.exists()


# ---------------------------------------------------------------------------
# write_checklist_json
# ---------------------------------------------------------------------------


class TestWriteChecklistJson:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        from research.pipeline._checklist import write_checklist_json

        rules = [{"rule": "R1", "pass": True, "measured": "ok", "target": "yes"}]
        out = write_checklist_json(
            rules=rules,
            gross_metrics=None,
            net_metrics=None,
            after_tax_metrics=None,
            output_dir=tmp_path,
        )
        assert out.exists()
        content = out.read_text()
        assert "summary" in content


# ---------------------------------------------------------------------------
# write_weights_csv
# ---------------------------------------------------------------------------


class TestWriteWeightsCsv:
    def test_writes_csv_sorted(self, tmp_path: Path) -> None:
        from research.pipeline._checklist import write_weights_csv

        weights = pd.Series([0.3, 0.5, 0.2], index=["A", "B", "C"])
        out = write_weights_csv(weights, tmp_path)
        assert out.exists()
        df = pd.read_csv(out)
        assert df.iloc[0]["ticker"] == "B"
        assert abs(df.iloc[0]["weight"] - 0.5) < 0.01


# ---------------------------------------------------------------------------
# _render_failure_table
# ---------------------------------------------------------------------------


class TestRenderFailureTable:
    def test_does_not_raise(self) -> None:
        from research.pipeline._checklist import _render_failure_table

        failed = [{"rule": "R1", "target": "> 0", "measured": "-0.1"}]
        _render_failure_table(failed)  # smoke test


# ---------------------------------------------------------------------------
# _apply_terminal_gate
# ---------------------------------------------------------------------------


class TestApplyTerminalGate:
    def test_when_all_pass_then_exits_zero(self, tmp_path: Path) -> None:
        from research.pipeline._checklist import (
            _CHECKLIST_TOTAL,
            _apply_terminal_gate,
        )

        rules = [
            {"rule": f"R{i}", "pass": True, "measured": "ok", "target": "x"}
            for i in range(_CHECKLIST_TOTAL)
        ]
        weights = pd.Series([1.0], index=["A"])
        with pytest.raises(SystemExit) as exc:
            _apply_terminal_gate(
                rules=rules,
                weights=weights,
                output_dir=tmp_path,
            )
        assert exc.value.code == 0

    def test_when_one_fails_then_exits_one(self, tmp_path: Path) -> None:
        from research.pipeline._checklist import (
            _CHECKLIST_TOTAL,
            _apply_terminal_gate,
        )

        rules = [
            {"rule": f"R{i}", "pass": True, "measured": "ok", "target": "x"}
            for i in range(_CHECKLIST_TOTAL)
        ]
        rules[5]["pass"] = False
        rules[5]["measured"] = "bad"
        weights = pd.Series([1.0], index=["A"])
        with pytest.raises(SystemExit) as exc:
            _apply_terminal_gate(
                rules=rules,
                weights=weights,
                output_dir=tmp_path,
            )
        assert exc.value.code == 1

    def test_asserts_exactly_17_rules(self, tmp_path: Path) -> None:
        from research.pipeline._checklist import _apply_terminal_gate

        weights = pd.Series([1.0], index=["A"])
        with pytest.raises(AssertionError):
            _apply_terminal_gate(
                rules=[{"rule": "R1", "pass": True, "measured": "x", "target": "y"}],
                weights=weights,
                output_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# _validate_checklist (integration test with real data)
# ---------------------------------------------------------------------------


class TestValidateChecklist:
    def test_all_rules_pass_with_good_data(self) -> None:
        from research.pipeline._checklist import (
            _CHECKLIST_TOTAL,
            _validate_checklist,
        )

        # 16-stock portfolio satisfying all 17 rules:
        # top-4 < 30%, single ≤ 10%, min ≥ 2%, IT ≥ 10%,
        # Health Care ≥ 8%, all 11 sectors present, region ≤ 60%.
        tickers = [f"T{i}" for i in range(16)]
        sectors = {
            "T0": "Health Care",
            "T1": "Health Care",
            "T2": "Information Technology",
            "T3": "Information Technology",
            "T4": "Financials",
            "T5": "Financials",
            "T6": "Energy",
            "T7": "Materials",
            "T8": "Industrials",
            "T9": "Consumer Discretionary",
            "T10": "Consumer Staples",
            "T11": "Communication Services",
            "T12": "Communication Services",
            "T13": "Utilities",
            "T14": "Real Estate",
            "T15": "Real Estate",
        }
        n = len(tickers)
        all_weights = [(t, 1.0 / n) for t in tickers]
        # 1/16 = 6.25% each → top-4 = 25% < 30% ✓,
        # IT = 3*6.25 = 18.75% ≥ 10% ✓,
        # Health Care = 3*6.25 = 18.75% ≥ 8% ✓

        # Distribute: Americas ≤ 60%, Europe ≤ 60%, rest "Other"
        country_map = {}
        for i, t in enumerate(tickers):
            if i < 5:  # T0-T4 → Americas (31.25%)
                country_map[t] = "United States"
            elif i < 10:  # T5-T9 → Europe (31.25%)
                country_map[t] = "France"
            elif i < 13:  # T10-T12 → Other via Japan (18.75%)
                country_map[t] = "Japan"
            else:  # T13-T15 → Americas (18.75%) — total Americas = 50%
                country_map[t] = "United States"
        # Americas = 50%, Europe = 31.25%, Other = 18.75% — all ≤ 60% ✓

        metrics = {
            "Portfolio (after-tax)": {
                "Max Drawdown": -0.10,
                "Ann. Vol": 0.12,
                "Sharpe (rf)": 1.5,
                "Sortino": 1.8,
                "Info Ratio": 0.7,
                "Downside Vol": 0.06,
            },
            "SPY (benchmark)": {"Ann. Vol": 0.15},
        }

        idx = pd.date_range("2015-01-01", periods=252 * 10, freq="B")
        bench = pd.Series(
            np.random.default_rng(42).normal(0, 0.01, len(idx)), index=idx
        )
        net = pd.Series(np.random.default_rng(43).normal(0, 0.01, len(idx)), index=idx)
        after = pd.Series(
            np.random.default_rng(44).normal(0, 0.01, len(idx)), index=idx
        )

        rules = _validate_checklist(
            all_weights,
            sectors,
            country_map,
            metrics,
            benchmark_returns=bench,
            net_returns=net,
            after_tax_returns=after,
            cost_bps_actual=50.0,
            currency_map=dict.fromkeys(tickers, "EUR"),
        )
        assert len(rules) == _CHECKLIST_TOTAL
        passing = sum(1 for r in rules if r["pass"])
        assert passing == _CHECKLIST_TOTAL

    def test_returns_17_rules(self) -> None:
        from research.pipeline._checklist import (
            _CHECKLIST_TOTAL,
            _validate_checklist,
        )

        all_weights = [("A", 0.5), ("B", 0.5)]
        sector_mapping = {"A": "Tech", "B": "Finance"}
        country_map = {"A": "US", "B": "US"}
        metrics: dict[str, dict[str, float]] = {}
        currency_map = {"A": "EUR", "B": "EUR"}

        rules = _validate_checklist(
            all_weights,
            sector_mapping,
            country_map,
            metrics,
            benchmark_returns=None,
            net_returns=None,
            after_tax_returns=None,
            cost_bps_actual=None,
            currency_map=currency_map,
        )
        assert len(rules) == _CHECKLIST_TOTAL


# ---------------------------------------------------------------------------
# _render_checklist_table
# ---------------------------------------------------------------------------


class TestRenderChecklistTable:
    def test_does_not_raise(self) -> None:
        from research.pipeline._checklist import _render_checklist_table

        rules = [
            {"rule": "R1", "pass": True, "measured": "x", "target": "y"},
            {"rule": "R2", "pass": False, "measured": "z", "target": "w"},
        ]
        _render_checklist_table(rules)  # smoke test
