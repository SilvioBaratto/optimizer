"""Cycle 4 §10 17-rule checklist + terminal gate tests (issue #556).

Covers:
- Hand-built passing inputs → all 17 rows pass.
- Each rule independently failed by perturbing one input → that row's
  ``pass=False``.
- ``_apply_terminal_gate`` exit-code contract: 17/17 PASS → exit 0 +
  ``weights.csv`` written; any FAIL → exit 1, no ``weights.csv``.
- ``len(rules) != 17`` → ``AssertionError``.
- Currency-hedge advisory logged for foreign-currency exposure > 30%
  but does **not** flip checklist status.
- Returned rule list has stable order matching §10 numbering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import research.pipeline as ssp

# ---------------------------------------------------------------------------
# Baseline fixtures — values chosen so every rule passes.
# ---------------------------------------------------------------------------


_TICKERS = [f"T{i:02d}" for i in range(25)]
_W = 1.0 / 25  # uniform 4%

# 11 sectors (Yahoo Finance naming), 25 tickers distributed (max 3 per sector → 12%).
_SECTOR_DIST: list[tuple[str, int]] = [
    ("Energy", 2),
    ("Basic Materials", 2),
    ("Industrials", 3),
    ("Consumer Cyclical", 3),
    ("Consumer Defensive", 2),
    ("Healthcare", 2),
    ("Financial Services", 3),
    ("Technology", 3),
    ("Communication Services", 2),
    ("Utilities", 2),
    ("Real Estate", 1),
]


def _baseline_sector_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    cursor = 0
    for sector, n in _SECTOR_DIST:
        for j in range(n):
            mapping[_TICKERS[cursor + j]] = sector
        cursor += n
    return mapping


def _baseline_country_map() -> dict[str, str]:
    # 12 US (48% Americas), 8 EU (32% Europe), 5 JP (20% AP) — all ≤ 60%.
    out: dict[str, str] = {}
    for i, ticker in enumerate(_TICKERS):
        if i < 12:
            out[ticker] = "United States"
        elif i < 20:
            out[ticker] = "Germany"
        else:
            out[ticker] = "Japan"
    return out


def _baseline_weights() -> list[tuple[str, float]]:
    return [(t, _W) for t in _TICKERS]


def _baseline_metrics() -> dict[str, dict[str, float]]:
    base = {
        "Ann. Return": 0.10,
        "Ann. Vol": 0.10,
        "Sharpe (rf)": 1.5,
        "Sortino": 2.0,
        "Info Ratio": 0.7,
        "Downside Vol": 0.05,  # < 0.075 = 0.75 × 0.10
        "Max Drawdown": -0.18,  # > -0.22
    }
    return {
        "Portfolio (gross)": dict(base),
        "Portfolio": dict(base),
        "Portfolio (after-tax)": dict(base),
        "SPY (benchmark)": {**base, "Ann. Vol": 0.15},
    }


def _baseline_net_returns() -> pd.Series:
    idx = pd.bdate_range("2016-01-01", "2025-01-15")  # ~9 years
    return pd.Series(0.0, index=idx)


def _baseline_inputs() -> dict[str, Any]:
    return {
        "all_weights": _baseline_weights(),
        "sector_mapping": _baseline_sector_mapping(),
        "country_map": _baseline_country_map(),
        "metrics": _baseline_metrics(),
        "benchmark_returns": pd.Series(dtype=float),
        "net_returns": _baseline_net_returns(),
        "after_tax_returns": pd.Series(dtype=float),
        "cost_bps_actual": 50.0,
        "currency_map": dict.fromkeys(_TICKERS, "EUR"),
    }


def _validate(**overrides: Any) -> list[dict[str, Any]]:
    kwargs = _baseline_inputs()
    kwargs.update(overrides)
    return ssp._validate_checklist(
        kwargs.pop("all_weights"),
        kwargs.pop("sector_mapping"),
        kwargs.pop("country_map"),
        kwargs.pop("metrics"),
        benchmark_returns=kwargs["benchmark_returns"],
        net_returns=kwargs["net_returns"],
        after_tax_returns=kwargs["after_tax_returns"],
        cost_bps_actual=kwargs["cost_bps_actual"],
        currency_map=kwargs["currency_map"],
    )


# ---------------------------------------------------------------------------
# Baseline — all 17 rules pass
# ---------------------------------------------------------------------------


class TestBaselineAllPass:
    def test_when_baseline_inputs_then_all_17_rules_pass(self) -> None:
        rules = _validate()
        failed = [r for r in rules if not r["pass"]]
        assert failed == [], f"Expected 17/17 PASS; failures: {failed}"

    def test_when_baseline_inputs_then_rule_count_is_17(self) -> None:
        assert len(_validate()) == 17

    def test_when_called_then_each_row_has_required_keys(self) -> None:
        for r in _validate():
            assert set(r.keys()) == {"rule", "pass", "measured", "target"}


# ---------------------------------------------------------------------------
# Per-rule perturbations — exactly one rule fails per scenario.
# ---------------------------------------------------------------------------


def _perturbation_max_region() -> dict[str, Any]:
    # All 25 tickers in US → 100% Americas.
    return {"country_map": dict.fromkeys(_TICKERS, "United States")}


def _perturbation_max_sector() -> dict[str, Any]:
    # All 25 tickers Tech → max sector = 100%.
    return {"sector_mapping": dict.fromkeys(_TICKERS, "Technology")}


def _perturbation_hhi() -> dict[str, Any]:
    # Concentrated 5-stock portfolio → HHI = 5 × 0.20² = 0.20 > 0.12.
    weights = [(_TICKERS[i], 0.20) for i in range(5)]
    return {"all_weights": weights}


def _perturbation_top4() -> dict[str, Any]:
    # 8% × 4 = 32% top-4 → fails. Remaining 17 stocks get the rest.
    weights = [(_TICKERS[i], 0.08) for i in range(4)]
    rest = (1.0 - 4 * 0.08) / 21
    weights += [(_TICKERS[i], rest) for i in range(4, 25)]
    return {"all_weights": weights}


def _perturbation_health_care() -> dict[str, Any]:
    sectors = _baseline_sector_mapping()
    # Reassign Healthcare tickers to Financial Services.
    for t, s in list(sectors.items()):
        if s == "Healthcare":
            sectors[t] = "Financial Services"
    return {"sector_mapping": sectors}


def _perturbation_information_technology() -> dict[str, Any]:
    sectors = _baseline_sector_mapping()
    for t, s in list(sectors.items()):
        if s == "Technology":
            sectors[t] = "Financial Services"
    return {"sector_mapping": sectors}


def _perturbation_missing_gics() -> dict[str, Any]:
    sectors = _baseline_sector_mapping()
    # Drop 4 sectors to reduce from 11 to 7 (< min 8).
    dropped = {"Real Estate", "Utilities", "Consumer Defensive", "Basic Materials"}
    for t, s in list(sectors.items()):
        if s in dropped:
            sectors[t] = "Financial Services"
    return {"sector_mapping": sectors}


def _perturbation_max_weight() -> dict[str, Any]:
    weights = [(_TICKERS[0], 0.12)] + [(t, (1.0 - 0.12) / 24) for t in _TICKERS[1:]]
    return {"all_weights": weights}


def _perturbation_min_weight() -> dict[str, Any]:
    # One ticker at 1% (< 2%); others spread.
    weights = [(_TICKERS[0], 0.01)] + [(t, (1.0 - 0.01) / 24) for t in _TICKERS[1:]]
    return {"all_weights": weights}


def _perturb_metric(label: str, **changes: float) -> dict[str, Any]:
    metrics = _baseline_metrics()
    metrics[label] = {**metrics[label], **changes}
    return {"metrics": metrics}


_AT = "Portfolio (after-tax)"


_PERTURBATIONS: list[tuple[int, str, dict[str, Any]]] = [
    (0, "No single region > 60%", _perturbation_max_region()),
    (1, "No single sector > 15%", _perturbation_max_sector()),
    (2, "HHI < 0.12", _perturbation_hhi()),
    (3, "Top-4 holdings < 30%", _perturbation_top4()),
    (4, "Health Care exposure ≥ 8%", _perturbation_health_care()),
    (
        5,
        "Information Technology exposure ≥ 10%",
        _perturbation_information_technology(),
    ),
    (6, "At least 8/11 sectors present", _perturbation_missing_gics()),
    (7, "Single-stock cap ≤ 10%", _perturbation_max_weight()),
    (8, "Min position ≥ 2%", _perturbation_min_weight()),
    (9, "Max drawdown > -22%", _perturb_metric(_AT, **{"Max Drawdown": -0.30})),
    (10, "Vol ≤ benchmark vol", _perturb_metric(_AT, **{"Ann. Vol": 0.30})),
    (11, "Sharpe ∈ (1.0, 2.0)", _perturb_metric(_AT, **{"Sharpe (rf)": 0.5})),
    (12, "Sortino > 1.5", _perturb_metric(_AT, **{"Sortino": 1.0})),
    (13, "Info Ratio > 0.5", _perturb_metric(_AT, **{"Info Ratio": 0.2})),
    (
        14,
        "Downside vol < 75% x total vol",
        _perturb_metric(_AT, **{"Downside Vol": 0.09}),
    ),
    (15, "Total cost ≤ 100 bps", {"cost_bps_actual": 150.0}),
    (
        16,
        "OOS span ≥ 1.5 years",
        {
            "net_returns": pd.Series(
                0.0,
                index=pd.bdate_range("2024-01-01", periods=20),
            )
        },
    ),
]


class TestRulePerturbations:
    @pytest.mark.parametrize(
        ("idx", "expected_rule", "override"),
        _PERTURBATIONS,
        ids=[f"rule_{i:02d}_{rule}" for i, rule, _ in _PERTURBATIONS],
    )
    def test_when_input_perturbed_then_target_rule_fails(
        self,
        idx: int,
        expected_rule: str,
        override: dict[str, Any],
    ) -> None:
        rules = _validate(**override)
        assert rules[idx]["pass"] is False
        assert rules[idx]["rule"] == expected_rule


# ---------------------------------------------------------------------------
# Rule order — stable §10 numbering
# ---------------------------------------------------------------------------


_EXPECTED_ORDER = [
    "No single region > 60%",
    "No single sector > 15%",
    "HHI < 0.12",
    "Top-4 holdings < 30%",
    "Health Care exposure ≥ 8%",
    "Information Technology exposure ≥ 10%",
    "At least 8/11 sectors present",
    "Single-stock cap ≤ 10%",
    "Min position ≥ 2%",
    "Max drawdown > -22%",
    "Vol ≤ benchmark vol",
    "Sharpe ∈ (1.0, 2.0)",
    "Sortino > 1.5",
    "Info Ratio > 0.5",
    "Downside vol < 75% x total vol",
    "Total cost ≤ 100 bps",
    "OOS span ≥ 1.5 years",
]


class TestRuleOrder:
    def test_when_called_then_rule_names_in_section_10_order(self) -> None:
        rules = _validate()
        assert [r["rule"] for r in rules] == _EXPECTED_ORDER


# ---------------------------------------------------------------------------
# Currency-hedge advisory
# ---------------------------------------------------------------------------


class TestCurrencyHedgeAdvisory:
    def test_when_fx_exposure_above_30pct_then_warning_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 12 USD tickers × 4% = 48% > 30%.
        currency_map = dict.fromkeys(_TICKERS, "EUR")
        for t in _TICKERS[:12]:
            currency_map[t] = "USD"
        with caplog.at_level(
            logging.WARNING,
            logger="research.pipeline._checklist",
        ):
            rules = _validate(currency_map=currency_map)
        assert any(
            "USD" in rec.message and "hedging" in rec.message.lower()
            for rec in caplog.records
        )
        # Currency advisory MUST NOT alter rule count or pass/fail status.
        assert len(rules) == 17
        assert all(r["pass"] for r in rules)

    def test_when_fx_exposure_below_30pct_then_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        currency_map = dict.fromkeys(_TICKERS, "EUR")
        currency_map[_TICKERS[0]] = "USD"  # 4% only
        with caplog.at_level(
            logging.WARNING,
            logger="research.pipeline._checklist",
        ):
            _validate(currency_map=currency_map)
        assert not any("hedging" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Terminal-gate — exit-code contract
# ---------------------------------------------------------------------------


def _weights_series() -> pd.Series:
    return pd.Series([w for _, w in _baseline_weights()], index=_TICKERS)


class TestTerminalGate:
    def test_when_all_17_pass_then_exit_zero_and_weights_written(
        self, tmp_path: Path
    ) -> None:
        rules = _validate()
        weights = _weights_series()
        with pytest.raises(SystemExit) as exc:
            ssp._apply_terminal_gate(
                rules=rules,
                weights=weights,
                output_dir=tmp_path,
            )
        assert exc.value.code == 0
        assert (tmp_path / "weights.csv").exists()

    def test_when_any_rule_fails_then_exit_one(self, tmp_path: Path) -> None:
        rules = _validate(**_perturbation_hhi())
        weights = _weights_series()
        with pytest.raises(SystemExit) as exc:
            ssp._apply_terminal_gate(
                rules=rules,
                weights=weights,
                output_dir=tmp_path,
            )
        assert exc.value.code == 1

    def test_when_any_rule_fails_then_no_weights_csv(self, tmp_path: Path) -> None:
        rules = _validate(**_perturbation_hhi())
        weights = _weights_series()
        with pytest.raises(SystemExit):
            ssp._apply_terminal_gate(
                rules=rules,
                weights=weights,
                output_dir=tmp_path,
            )
        assert not (tmp_path / "weights.csv").exists()

    def test_when_rules_count_not_17_then_assertion_error(self, tmp_path: Path) -> None:
        short_rules = _validate()[:10]
        weights = _weights_series()
        with pytest.raises(AssertionError, match="17"):
            ssp._apply_terminal_gate(
                rules=short_rules,
                weights=weights,
                output_dir=tmp_path,
            )

    def test_when_rules_empty_then_assertion_error(self, tmp_path: Path) -> None:
        weights = _weights_series()
        with pytest.raises(AssertionError, match="17"):
            ssp._apply_terminal_gate(
                rules=[],
                weights=weights,
                output_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# Companion JSON writers — metrics.json + checklist.json
# ---------------------------------------------------------------------------


class TestArtefactWriters:
    def test_when_metrics_written_then_metrics_json_present(
        self, tmp_path: Path
    ) -> None:
        out = ssp.write_metrics_json(_baseline_metrics(), tmp_path)
        assert out == tmp_path / "metrics.json"
        assert out.exists()
        assert out.stat().st_size > 0

    def test_when_checklist_written_then_checklist_json_present(
        self, tmp_path: Path
    ) -> None:
        rules = _validate()
        metrics = _baseline_metrics()
        out = ssp.write_checklist_json(
            rules=rules,
            gross_metrics=metrics["Portfolio (gross)"],
            net_metrics=metrics["Portfolio"],
            after_tax_metrics=metrics["Portfolio (after-tax)"],
            output_dir=tmp_path,
        )
        assert out == tmp_path / "checklist.json"
        import json

        payload = json.loads(out.read_text())
        assert payload["summary"] == {"passed": 17, "total": 17}


# ---------------------------------------------------------------------------
# Type sanity
# ---------------------------------------------------------------------------


class TestTypeContract:
    def test_when_baseline_then_pass_field_is_bool(self) -> None:
        for r in _validate():
            assert isinstance(r["pass"], bool | np.bool_)

    def test_when_baseline_then_rule_field_is_str(self) -> None:
        for r in _validate():
            assert isinstance(r["rule"], str)
