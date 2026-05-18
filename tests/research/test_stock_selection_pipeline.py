"""Tests for research/stock_selection_pipeline.py wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    FactorOOSResult,
    FactorValidationConfig,
    FactorValidationReport,
)

# Module under test imports api.app.database; ensure it can be loaded.
pytest.importorskip("rich")

import research.pipeline as ssp
import research.pipeline._factors as _factors_module
import research.pipeline._optimize as _opt_module
import research.pipeline._regime as _regime_module
from research.optimization._config import _REGION_MAP


def _stub_factor_scores() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=24)
    tickers = [f"T{i:02d}" for i in range(5)]
    rng = np.random.default_rng(0)
    return {
        "value": pd.DataFrame(rng.normal(size=(24, 5)), index=dates, columns=tickers),
    }


def _stub_returns() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=24)
    tickers = [f"T{i:02d}" for i in range(5)]
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.normal(size=(24, 5)), index=dates, columns=tickers)


class _StubVifReport:
    vif_scores = None


class TestValidateIsConfig:
    """Issue #526: validate_is must wire spec-compliant FactorValidationConfig."""

    def test_when_validate_is_called_run_factor_validation_receives_spec_config(
        self,
    ) -> None:
        captured: dict[str, FactorValidationConfig] = {}

        def fake_run_validation(
            factor_scores_history: dict[str, pd.DataFrame],
            returns_history: pd.DataFrame,
            config: FactorValidationConfig | None = None,
        ) -> FactorValidationReport:
            captured["config"] = config  # type: ignore[assignment]
            return FactorValidationReport()

        with (
            patch.object(
                _factors_module,
                "run_factor_validation",
                side_effect=fake_run_validation,
            ),
            patch.object(
                _factors_module, "validate_factors", return_value=_StubVifReport()
            ),
        ):
            ssp.validate_is(_stub_factor_scores(), _stub_returns())

        cfg = captured["config"]
        assert cfg is not None
        assert cfg.newey_west_lags == 4
        assert cfg.fdr_alpha == pytest.approx(0.10)
        assert cfg.t_stat_threshold == pytest.approx(1.645)

    def test_when_validate_is_returns_report_object(self) -> None:
        with (
            patch.object(
                _factors_module,
                "run_factor_validation",
                return_value=FactorValidationReport(),
            ),
            patch.object(
                _factors_module, "validate_factors", return_value=_StubVifReport()
            ),
        ):
            report = ssp.validate_is(_stub_factor_scores(), _stub_returns())
        assert isinstance(report, FactorValidationReport)


def _empty_oos_result(n_folds: int) -> FactorOOSResult:
    factors = ["value"]
    if n_folds == 0:
        ic = pd.DataFrame(columns=factors)
        spread = pd.DataFrame(columns=factors)
    else:
        ic = pd.DataFrame(0.05, index=range(n_folds), columns=factors)
        spread = pd.DataFrame(0.01, index=range(n_folds), columns=factors)
    return FactorOOSResult(
        per_fold_ic=ic,
        per_fold_spread=spread,
        mean_oos_ic=pd.Series([0.05], index=factors),
        mean_oos_icir=pd.Series([1.5], index=factors),
        n_folds=n_folds,
    )


class TestValidateOosHardFail:
    """Issue #527: validate_oos must raise when n_folds == 0."""

    def test_when_n_folds_zero_raises_runtime_error(self) -> None:
        with (
            patch.object(
                _factors_module,
                "run_factor_oos_validation",
                return_value=_empty_oos_result(0),
            ),
            pytest.raises(RuntimeError, match="0 folds"),
        ):
            ssp.validate_oos(_stub_factor_scores(), _stub_returns())

    def test_when_n_folds_zero_message_includes_oos_config_params(self) -> None:
        with (
            patch.object(
                _factors_module,
                "run_factor_oos_validation",
                return_value=_empty_oos_result(0),
            ),
            pytest.raises(RuntimeError) as exc,
        ):
            ssp.validate_oos(_stub_factor_scores(), _stub_returns())
        msg = str(exc.value)
        assert f"train_periods={ssp.OOS_CONFIG.train_periods}" in msg
        assert f"val_periods={ssp.OOS_CONFIG.val_periods}" in msg
        assert f"step_periods={ssp.OOS_CONFIG.step_periods}" in msg

    def test_when_n_folds_positive_returns_result(self) -> None:
        stub = _empty_oos_result(3)
        with patch.object(
            _factors_module, "run_factor_oos_validation", return_value=stub
        ):
            result = ssp.validate_oos(_stub_factor_scores(), _stub_returns())
        assert result is stub
        assert result.n_folds == 3


class TestOptimizePortfolioRegimeWiring:
    """Issue #529: orchestrator must receive enabled regime config + macro_data."""

    def _stub_assembly(self) -> Any:
        from types import SimpleNamespace

        idx = pd.bdate_range("2020-01-01", periods=10)
        tickers = ["AAA", "BBB"]
        prices = pd.DataFrame(100.0, index=idx, columns=tickers)
        volumes = pd.DataFrame(1_000_000.0, index=idx, columns=tickers)
        return SimpleNamespace(
            prices=prices,
            volumes=volumes,
            macro_data=pd.DataFrame({"gdp_growth": [1.0]}, index=[idx[0]]),
            regime_data=pd.DataFrame({"pmi": [55.0]}, index=[idx[0]]),
            fundamentals=pd.DataFrame({"market_cap": [1e9, 2e9]}, index=tickers),
            analyst_data=pd.DataFrame(),
            insider_data=pd.DataFrame(),
            sector_mapping={"AAA": "Tech", "BBB": "Finance"},
            risk_free_rate=0.0,
            delisting_returns={},
            currency_map={"AAA": "USD", "BBB": "USD"},
            fx_rates=pd.DataFrame(),
        )

    def test_when_optimize_portfolio_called_orchestrator_receives_enabled_regime(
        self,
    ) -> None:
        from types import SimpleNamespace

        captured: dict[str, Any] = {}

        def fake_pipeline(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                summary={"sharpe_ratio": 1.0},
                net_sharpe_ratio=None,
                weights=pd.Series([0.5, 0.5], index=["AAA", "BBB"]),
            )

        with patch.object(
            _opt_module, "run_full_pipeline_with_selection", side_effect=fake_pipeline
        ):
            ssp.optimize_portfolio(
                assembly=self._stub_assembly(),
                investable=pd.Index(["AAA", "BBB"]),
                ic_history=None,
                n_selected=2,
            )

        assert "regime_config" in captured
        assert captured["regime_config"] is not None
        assert captured["regime_config"].enable is True
        assert "macro_data" in captured
        assert captured["macro_data"] is not None


class TestClassifyAndTiltCachesRegime:
    """Issue #530: classify_and_tilt persists rule-based regime to DB."""

    def test_when_regime_persistence_supplied_then_upsert_called_with_us(
        self,
    ) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from optimizer.factors import MacroRegime

        assembly = SimpleNamespace(
            macro_data=pd.DataFrame({"gdp_growth": [1.0]}),
            fred_data=pd.DataFrame(),
            te_observations=pd.DataFrame(),
            sentiment_data=pd.DataFrame(),
        )
        persistence = MagicMock()

        with patch.object(
            _regime_module,
            "classify_regime",
            return_value=MacroRegime.RECESSION,
        ):
            ssp.classify_and_tilt(
                assembly,  # type: ignore[arg-type]
                regime_persistence=persistence,
            )

        persistence.upsert_regime_classification.assert_called_once_with(
            country="US", regime="recession"
        )

    def test_when_db_manager_omitted_no_persistence_attempt(self) -> None:
        from types import SimpleNamespace

        from optimizer.factors import MacroRegime

        assembly = SimpleNamespace(
            macro_data=pd.DataFrame({"gdp_growth": [1.0]}),
            fred_data=pd.DataFrame(),
            te_observations=pd.DataFrame(),
            sentiment_data=pd.DataFrame(),
        )
        with patch.object(
            _regime_module, "classify_regime", return_value=MacroRegime.EXPANSION
        ):
            regime, _tilts = ssp.classify_and_tilt(assembly)
        assert regime == MacroRegime.EXPANSION


class TestSpecCompliantSelectionAndScoring:
    """Issue #531: optimize_portfolio wires Cycle-2 selection + IC decay defaults."""

    def _stub_assembly(self) -> Any:
        from types import SimpleNamespace

        idx = pd.bdate_range("2020-01-01", periods=10)
        tickers = ["AAA", "BBB"]
        prices = pd.DataFrame(100.0, index=idx, columns=tickers)
        volumes = pd.DataFrame(1_000_000.0, index=idx, columns=tickers)
        return SimpleNamespace(
            prices=prices,
            volumes=volumes,
            macro_data=pd.DataFrame({"gdp_growth": [1.0]}, index=[idx[0]]),
            regime_data=pd.DataFrame({"pmi": [55.0]}, index=[idx[0]]),
            fundamentals=pd.DataFrame({"market_cap": [1e9, 2e9]}, index=tickers),
            analyst_data=pd.DataFrame(),
            insider_data=pd.DataFrame(),
            sector_mapping={"AAA": "Tech", "BBB": "Finance"},
            risk_free_rate=0.0,
            delisting_returns={},
            currency_map={"AAA": "USD", "BBB": "USD"},
            fx_rates=pd.DataFrame(),
        )

    def _capture_kwargs(self, n_selected: int) -> dict[str, Any]:
        from types import SimpleNamespace

        captured: dict[str, Any] = {}

        def fake_pipeline(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                summary={"sharpe_ratio": 1.0},
                net_sharpe_ratio=None,
                weights=pd.Series([0.5, 0.5], index=["AAA", "BBB"]),
            )

        with patch.object(
            _opt_module, "run_full_pipeline_with_selection", side_effect=fake_pipeline
        ):
            ssp.optimize_portfolio(
                assembly=self._stub_assembly(),
                investable=pd.Index(["AAA", "BBB"]),
                ic_history=None,
                n_selected=n_selected,
            )
        return captured

    def test_when_optimize_portfolio_called_scoring_config_uses_icir_weighted(
        self,
    ) -> None:
        from optimizer.factors import CompositeMethod

        captured = self._capture_kwargs(n_selected=30)
        cfg = captured["scoring_config"]
        assert cfg.method == CompositeMethod.ICIR_WEIGHTED
        assert cfg.ic_decay_halflife == 0

    def test_when_optimize_portfolio_called_selection_config_matches_spec(
        self,
    ) -> None:
        from optimizer.factors import SelectionMethod

        captured = self._capture_kwargs(n_selected=42)
        sel = captured["selection_config"]
        assert sel.target_count == 42
        assert sel.method == SelectionMethod.FIXED_COUNT
        assert sel.buffer_fraction == pytest.approx(0.05)
        assert sel.sector_balance is True
        assert sel.max_per_sector == 4


class TestNSelectedRangeValidation:
    """Issue #531: main() must validate 25 <= n_selected <= 50."""

    def test_when_n_selected_below_25_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="n_selected"):
            ssp._validate_n_selected(24)

    def test_when_n_selected_above_50_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="n_selected"):
            ssp._validate_n_selected(51)

    def test_when_n_selected_25_accepted(self) -> None:
        ssp._validate_n_selected(25)

    def test_when_n_selected_50_accepted(self) -> None:
        ssp._validate_n_selected(50)


class TestSectorCoverageWarning:
    """Issue #531: warn when fewer than 11 GICS sectors represented."""

    def test_when_all_11_sectors_present_no_missing_returned(self) -> None:
        sectors = [
            "Energy",
            "Basic Materials",
            "Industrials",
            "Consumer Cyclical",
            "Consumer Defensive",
            "Healthcare",
            "Financial Services",
            "Technology",
            "Communication Services",
            "Utilities",
            "Real Estate",
        ]
        weights = pd.Series([1.0 / 11] * 11, index=[f"T{i}" for i in range(11)])
        sector_mapping = {f"T{i}": s for i, s in enumerate(sectors)}
        missing = ssp._missing_gics_sectors(weights, sector_mapping)
        assert missing == []

    def test_when_some_sectors_absent_returns_missing_list(self) -> None:
        weights = pd.Series([0.5, 0.5], index=["AAA", "BBB"])
        sector_mapping = {"AAA": "Energy", "BBB": "Financial Services"}
        missing = ssp._missing_gics_sectors(weights, sector_mapping)
        assert "Energy" not in missing
        assert "Financial Services" not in missing
        assert len(missing) == 9


# ---------------------------------------------------------------------------
# Cycle-3 §7.1 hard-constrained MeanRisk wiring (issue #536)
# ---------------------------------------------------------------------------


def _spec_assembly() -> Any:
    from types import SimpleNamespace

    idx = pd.bdate_range("2020-01-01", periods=10)
    tickers = ["AAA", "BBB"]
    prices = pd.DataFrame(100.0, index=idx, columns=tickers)
    volumes = pd.DataFrame(1_000_000.0, index=idx, columns=tickers)
    return SimpleNamespace(
        prices=prices,
        volumes=volumes,
        macro_data=pd.DataFrame({"gdp_growth": [1.0]}, index=[idx[0]]),
        regime_data=pd.DataFrame({"pmi": [55.0]}, index=[idx[0]]),
        fundamentals=pd.DataFrame({"market_cap": [1e9, 2e9]}, index=tickers),
        analyst_data=pd.DataFrame(),
        insider_data=pd.DataFrame(),
        sector_mapping={"AAA": "Healthcare", "BBB": "Technology"},
        risk_free_rate=0.0,
        delisting_returns={},
        currency_map={"AAA": "USD", "BBB": "USD"},
        fx_rates=pd.DataFrame(),
    )


def _capture_optimizer(
    *,
    n_selected: int = 30,
    cost_bps: float = 10.0,
    country_map: dict[str, str] | None = None,
    previous_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    from types import SimpleNamespace

    captured: dict[str, Any] = {}

    def fake_pipeline(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            summary={"sharpe_ratio": 1.0},
            net_sharpe_ratio=None,
            weights=pd.Series([0.5, 0.5], index=["AAA", "BBB"]),
        )

    with patch.object(
        _opt_module, "run_full_pipeline_with_selection", side_effect=fake_pipeline
    ):
        ssp.optimize_portfolio(
            assembly=_spec_assembly(),
            investable=pd.Index(["AAA", "BBB"]),
            ic_history=None,
            n_selected=n_selected,
            cost_bps=cost_bps,
            country_map=country_map,
            previous_weights=previous_weights,
        )
    return captured


class TestRegionMapModuleConstant:
    def test_when_imported_then_module_constant_exists(self) -> None:
        assert isinstance(_REGION_MAP, dict)
        assert _REGION_MAP["United States"] == "Americas"
        assert _REGION_MAP["Germany"] == "Europe"
        assert _REGION_MAP["Japan"] == "Asia-Pacific"


class TestHardConstrainedMeanRiskSpec:
    """§7.1 spec wiring of optimize_portfolio."""

    def test_when_called_then_objective_is_minimize_risk(self) -> None:
        from skfolio.optimization.convex._base import ObjectiveFunction

        captured = _capture_optimizer()
        opt = captured["optimizer"]
        assert opt.objective_function == ObjectiveFunction.MINIMIZE_RISK

    def test_when_called_then_max_weights_010(self) -> None:
        captured = _capture_optimizer()
        assert captured["optimizer"].max_weights == pytest.approx(0.10)

    def test_when_survivors_meet_target_then_min_weights_002(self) -> None:
        captured = _capture_optimizer(n_selected=2)
        assert captured["optimizer"].min_weights == pytest.approx(0.02)

    def test_when_survivors_below_target_then_min_weights_fallback(self) -> None:
        # 2 investable tickers, target 30 → fallback 1/(2*2) = 0.25
        captured = _capture_optimizer(n_selected=30)
        assert captured["optimizer"].min_weights == pytest.approx(1.0 / (2 * 2))

    def test_when_called_then_l2_coef_005(self) -> None:
        captured = _capture_optimizer()
        assert captured["optimizer"].l2_coef == pytest.approx(0.05)

    def test_when_called_then_max_sector_weight_015_via_constraint(self) -> None:
        captured = _capture_optimizer()
        constraints = captured["optimizer"].linear_constraints or []
        assert any("Healthcare <= 0.15" in c for c in constraints)
        assert any("Technology <= 0.15" in c for c in constraints)

    def test_when_called_then_transaction_costs_match_cost_bps(self) -> None:
        captured = _capture_optimizer(cost_bps=10.0)
        assert captured["optimizer"].transaction_costs == pytest.approx(10.0 / 1e4)

    def test_when_called_then_solver_clarabel(self) -> None:
        captured = _capture_optimizer()
        assert captured["optimizer"].solver == "CLARABEL"

    def test_when_called_then_solver_params_match_spec(self) -> None:
        captured = _capture_optimizer()
        params = captured["optimizer"].solver_params
        assert params["max_iter"] == 200_000
        assert params["tol_gap_abs"] == pytest.approx(1e-8)
        assert params["tol_gap_rel"] == pytest.approx(1e-8)

    def test_when_called_then_sector_floor_constraints_empty(self) -> None:
        captured = _capture_optimizer()
        constraints = captured["optimizer"].linear_constraints or []
        sector_floor_constraints = [c for c in constraints if " >= " in c and any(
            sec in c for sec in ("Energy", "Materials", "Industrial", "Cyclical",
                                "Defensive", "Healthcare", "Financial", "Technology",
                                "Communication", "Utilities", "Real")
        )]
        assert len(sector_floor_constraints) == 0

    def test_when_country_map_supplied_then_region_caps_in_constraints(self) -> None:
        captured = _capture_optimizer(
            country_map={"AAA": "United States", "BBB": "Germany"}
        )
        constraints = captured["optimizer"].linear_constraints or []
        assert any("Americas <= 0.6" in c for c in constraints)
        assert any("Europe <= 0.6" in c for c in constraints)

    def test_when_country_map_none_then_no_region_constraints(self) -> None:
        captured = _capture_optimizer(country_map=None)
        constraints = captured["optimizer"].linear_constraints or []
        assert not any("Americas" in c for c in constraints)

    def test_when_previous_weights_supplied_then_forwarded_to_optimizer(self) -> None:
        prev = np.array([0.5, 0.5])
        captured = _capture_optimizer(previous_weights=prev)
        assert captured["optimizer"].previous_weights is not None
        np.testing.assert_array_equal(captured["optimizer"].previous_weights, prev)

    def test_when_survivors_below_target_then_warning_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            _capture_optimizer(n_selected=30)
        msg = caplog.text.lower()
        assert "min_weights" in msg or "feasibility" in msg or "survivors" in msg


class TestValidateChecklistReusesRegionMap:
    """`_validate_checklist` consumes the module-level `_REGION_MAP`."""

    def test_when_validate_checklist_called_then_no_local_region_map(self) -> None:
        import inspect

        src = inspect.getsource(ssp._validate_checklist)
        # Old code defined `region_map = {` inside the function.
        assert "region_map = {" not in src


# ---------------------------------------------------------------------------
# Cycle-3 §8 walk-forward + metadata routing + hockey-stick wiring (issue #538)
# ---------------------------------------------------------------------------


class TestWalkForwardConfigSpec:
    """`optimize_portfolio` must wire WalkForwardConfig(train=756, test=63)."""

    def test_when_called_then_cv_config_uses_spec_values(self) -> None:
        captured = _capture_optimizer()
        cv = captured["cv_config"]
        assert cv.train_size == 252 * 3
        assert cv.test_size == 63
        assert cv.expend_train is False
        assert cv.purged_size == 5


class TestSklearnMetadataRouting:
    """`optimize_portfolio` must enable metadata routing once at entry."""

    def test_when_called_then_enable_metadata_routing_invoked(self) -> None:
        from types import SimpleNamespace

        def fake_pipeline(**kwargs: Any) -> Any:
            return SimpleNamespace(
                summary={"sharpe_ratio": 1.0},
                net_sharpe_ratio=None,
                weights=pd.Series([0.5, 0.5], index=["AAA", "BBB"]),
                net_returns=None,
            )

        with (
            patch.object(
                _opt_module,
                "run_full_pipeline_with_selection",
                side_effect=fake_pipeline,
            ),
            patch("sklearn.set_config") as mock_set_config,
        ):
            ssp.optimize_portfolio(
                assembly=_spec_assembly(),
                investable=pd.Index(["AAA", "BBB"]),
                ic_history=None,
                n_selected=2,
            )

        # called at least once with enable_metadata_routing=True
        kw_calls = [c for c in mock_set_config.call_args_list if c.kwargs]
        assert any(c.kwargs.get("enable_metadata_routing") is True for c in kw_calls)


class TestHockeyStickInvocation:
    """`optimize_portfolio` must invoke `_hockey_stick_warn` on result.net_returns."""

    def test_when_pipeline_returns_then_hockey_stick_warn_called(self) -> None:
        from types import SimpleNamespace

        oos_stub = pd.Series(
            [0.0, 0.0, 0.0],
            index=pd.bdate_range("2020-01-01", periods=3),
        )

        def fake_pipeline(**kwargs: Any) -> Any:
            return SimpleNamespace(
                summary={"sharpe_ratio": 1.0},
                net_sharpe_ratio=None,
                weights=pd.Series([0.5, 0.5], index=["AAA", "BBB"]),
                net_returns=oos_stub,
            )

        with (
            patch.object(
                _opt_module,
                "run_full_pipeline_with_selection",
                side_effect=fake_pipeline,
            ),
            patch.object(_opt_module, "_hockey_stick_warn") as mock_warn,
        ):
            ssp.optimize_portfolio(
                assembly=_spec_assembly(),
                investable=pd.Index(["AAA", "BBB"]),
                ic_history=None,
                n_selected=2,
            )

        mock_warn.assert_called_once()
        args, _ = mock_warn.call_args
        # First positional arg is the OOS return series we stubbed
        pd.testing.assert_series_equal(args[0], oos_stub)


class TestRetightenTraceWiring:
    """Issue #537: `optimize_portfolio` must surface a retighten trace."""

    def _wide_assembly(self, n_tickers: int = 8) -> Any:
        from types import SimpleNamespace

        idx = pd.bdate_range("2020-01-01", periods=10)
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
        prices = pd.DataFrame(100.0, index=idx, columns=tickers)
        volumes = pd.DataFrame(1_000_000.0, index=idx, columns=tickers)
        sectors = ["Healthcare", "Technology", "Financial Services", "Energy"]
        sector_mapping = {t: sectors[i % len(sectors)] for i, t in enumerate(tickers)}
        return SimpleNamespace(
            prices=prices,
            volumes=volumes,
            macro_data=pd.DataFrame({"gdp_growth": [1.0]}, index=[idx[0]]),
            regime_data=pd.DataFrame({"pmi": [55.0]}, index=[idx[0]]),
            fundamentals=pd.DataFrame({"market_cap": [1e9] * n_tickers}, index=tickers),
            analyst_data=pd.DataFrame(),
            insider_data=pd.DataFrame(),
            sector_mapping=sector_mapping,
            risk_free_rate=0.0,
            delisting_returns={},
            currency_map=dict.fromkeys(tickers, "USD"),
            fx_rates=pd.DataFrame(),
        )

    def test_when_universe_above_top4_then_retighten_disabled_trace_empty(
        self,
    ) -> None:
        from types import SimpleNamespace

        def fake_pipeline(**kwargs: Any) -> Any:
            return SimpleNamespace(
                summary={"sharpe_ratio": 1.0},
                net_sharpe_ratio=None,
                weights=pd.Series([0.125] * 8, index=[f"T{i:02d}" for i in range(8)]),
            )

        with patch.object(
            _opt_module,
            "run_full_pipeline_with_selection",
            side_effect=fake_pipeline,
        ):
            assembly = self._wide_assembly(n_tickers=8)
            result = ssp.optimize_portfolio(
                assembly=assembly,
                investable=pd.Index([f"T{i:02d}" for i in range(8)]),
                ic_history=None,
                n_selected=8,
            )

        assert result.retighten_trace == []

    def test_when_universe_at_or_below_top4_then_retighten_disabled(self) -> None:
        from types import SimpleNamespace

        def fake_pipeline(**kwargs: Any) -> Any:
            return SimpleNamespace(
                summary={"sharpe_ratio": 1.0},
                net_sharpe_ratio=None,
                weights=pd.Series([0.5, 0.5], index=["T00", "T01"]),
            )

        with patch.object(
            _opt_module,
            "run_full_pipeline_with_selection",
            side_effect=fake_pipeline,
        ):
            assembly = self._wide_assembly(n_tickers=2)
            result = ssp.optimize_portfolio(
                assembly=assembly,
                investable=pd.Index(["T00", "T01"]),
                ic_history=None,
                n_selected=2,
            )

        assert result.retighten_trace == []


class TestCountryCostsBps:
    """Issue #542: per-country transaction-cost dict + weighted helper."""

    def test_when_module_loaded_then_country_costs_bps_defined(self) -> None:
        assert isinstance(ssp.COUNTRY_COSTS_BPS, dict)
        assert isinstance(ssp._DEFAULT_COSTS, dict)
        for country in ("United Kingdom", "France", "Italy", "United States"):
            assert country in ssp.COUNTRY_COSTS_BPS
        for component in ("stamp", "spread", "fx"):
            assert component in ssp._DEFAULT_COSTS

    def test_when_uk_country_then_total_is_58_bps(self) -> None:
        weights = pd.Series([1.0], index=["UKX"])
        country_map = {"UKX": "United Kingdom"}
        result = ssp.compute_weighted_cost_bps(weights, country_map)
        assert result == pytest.approx(58.0)

    def test_when_france_country_then_total_is_48_bps(self) -> None:
        weights = pd.Series([1.0], index=["FRX"])
        country_map = {"FRX": "France"}
        result = ssp.compute_weighted_cost_bps(weights, country_map)
        assert result == pytest.approx(48.0)

    def test_when_italy_country_then_total_is_30_bps(self) -> None:
        weights = pd.Series([1.0], index=["ITX"])
        country_map = {"ITX": "Italy"}
        result = ssp.compute_weighted_cost_bps(weights, country_map)
        assert result == pytest.approx(30.0)

    def test_when_us_country_then_total_is_15_bps(self) -> None:
        weights = pd.Series([1.0], index=["USX"])
        country_map = {"USX": "United States"}
        result = ssp.compute_weighted_cost_bps(weights, country_map)
        assert result == pytest.approx(15.0)

    def test_when_unknown_country_then_default_18_bps_silently(self) -> None:
        weights = pd.Series([1.0], index=["XYZ"])
        country_map = {"XYZ": "Atlantis"}
        result = ssp.compute_weighted_cost_bps(weights, country_map)
        assert result == pytest.approx(18.0)

    def test_when_country_missing_from_map_then_default_18_bps(self) -> None:
        """Ticker absent from country_map falls through to default (e.g. None)."""
        weights = pd.Series([1.0], index=["GHOST"])
        result = ssp.compute_weighted_cost_bps(weights, {})
        assert result == pytest.approx(18.0)

    def test_when_mixed_countries_then_weighted_average_matches_manual(self) -> None:
        weights = pd.Series([0.5, 0.3, 0.2], index=["USX", "UKX", "ITX"], dtype=float)
        country_map = {
            "USX": "United States",
            "UKX": "United Kingdom",
            "ITX": "Italy",
        }
        # 0.5 * 15 + 0.3 * 58 + 0.2 * 30 = 7.5 + 17.4 + 6.0 = 30.9
        expected = 0.5 * 15.0 + 0.3 * 58.0 + 0.2 * 30.0
        result = ssp.compute_weighted_cost_bps(weights, country_map)
        assert result == pytest.approx(expected)

    def test_when_empty_weights_then_zero(self) -> None:
        empty = pd.Series([], index=pd.Index([], dtype=str), dtype=float)
        assert ssp.compute_weighted_cost_bps(empty, {}) == 0.0

    def test_when_nan_weights_then_zero(self) -> None:
        weights = pd.Series([float("nan"), float("nan")], index=["A", "B"])
        country_map = {"A": "United States", "B": "United Kingdom"}
        assert ssp.compute_weighted_cost_bps(weights, country_map) == 0.0

    def test_when_some_weights_nan_then_skipped(self) -> None:
        weights = pd.Series([1.0, float("nan")], index=["USX", "UKX"])
        country_map = {"USX": "United States", "UKX": "United Kingdom"}
        # NaN dropped → only USX contributes 1.0 * 15
        result = ssp.compute_weighted_cost_bps(weights, country_map)
        assert result == pytest.approx(15.0)

    def test_when_helper_called_pure_function_no_io(self) -> None:
        """Pure function: no DB, no FS, no logger calls."""
        import inspect

        src = inspect.getsource(ssp.compute_weighted_cost_bps)
        # No DB, file, or logging calls inside helper body
        assert "DatabaseManager" not in src
        assert "logger." not in src
        assert "console." not in src
        assert "open(" not in src


class TestSortinoHelper:
    """Issue #544: _sortino helper (Cycle 4 §9.3)."""

    def test_when_empty_series_then_zero(self) -> None:
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        rf = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        assert ssp._sortino(empty, rf) == 0.0

    def test_when_no_downside_returns_then_zero(self) -> None:
        idx = pd.bdate_range("2024-01-01", periods=5)
        all_positive = pd.Series([0.01] * 5, index=idx)
        rf = pd.Series([0.0] * 5, index=idx)
        # No returns < daily_rf → downside std = 0 → 0.0 per AC
        assert ssp._sortino(all_positive, rf) == 0.0

    def test_when_5_day_series_then_matches_hand_computed(self) -> None:
        """Sortino = ann_excess / downside_vol; downside is std on r < 0."""
        idx = pd.bdate_range("2024-01-01", periods=5)
        rets = pd.Series([0.01, -0.02, 0.005, -0.01, 0.02], index=idx)
        rf = pd.Series([0.0] * 5, index=idx)
        # Hand-compute:
        # excess = rets (rf=0)
        # ann_excess = (1+rets).prod()^(252/5) - 1
        # downside subset (r < 0): [-0.02, -0.01]
        # downside_vol = std([-0.02, -0.01], ddof=1) * sqrt(252)
        excess_arr = np.array([0.01, -0.02, 0.005, -0.01, 0.02])
        ann_excess = float((1.0 + excess_arr).prod() ** (252.0 / 5.0) - 1.0)
        downside = np.array([-0.02, -0.01])
        downside_std = float(downside.std(ddof=1))
        downside_vol = downside_std * np.sqrt(252.0)
        expected = ann_excess / downside_vol
        assert ssp._sortino(rets, rf) == pytest.approx(expected, rel=1e-9)


class TestDownsideVolHelper:
    """Issue #544: _downside_vol helper."""

    def test_when_empty_series_then_zero(self) -> None:
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        rf = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        assert ssp._downside_vol(empty, rf) == 0.0

    def test_when_all_positive_returns_then_zero(self) -> None:
        idx = pd.bdate_range("2024-01-01", periods=5)
        all_positive = pd.Series([0.01] * 5, index=idx)
        rf = pd.Series([0.0] * 5, index=idx)
        assert ssp._downside_vol(all_positive, rf) == 0.0

    def test_when_mixed_then_annualised_std_of_negative_excess(self) -> None:
        idx = pd.bdate_range("2024-01-01", periods=5)
        rets = pd.Series([0.01, -0.02, 0.005, -0.01, 0.02], index=idx)
        rf = pd.Series([0.0] * 5, index=idx)
        downside = np.array([-0.02, -0.01])
        expected = float(downside.std(ddof=1)) * np.sqrt(252.0)
        assert ssp._downside_vol(rets, rf) == pytest.approx(expected, rel=1e-9)


class TestInformationRatioHelper:
    """Issue #544: _information_ratio helper."""

    def test_when_empty_series_then_zero(self) -> None:
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        assert ssp._information_ratio(empty, empty) == 0.0

    def test_when_constant_active_return_then_zero(self) -> None:
        """Constant active return → std = 0 → IR = 0.0 per AC."""
        idx = pd.bdate_range("2024-01-01", periods=10)
        portfolio = pd.Series([0.001] * 10, index=idx)
        benchmark = pd.Series([0.0] * 10, index=idx)
        assert ssp._information_ratio(portfolio, benchmark) == 0.0

    def test_when_no_index_overlap_then_nan(self) -> None:
        idx_a = pd.bdate_range("2024-01-01", periods=5)
        idx_b = pd.bdate_range("2025-01-01", periods=5)
        portfolio = pd.Series([0.01] * 5, index=idx_a)
        benchmark = pd.Series([0.005] * 5, index=idx_b)
        result = ssp._information_ratio(portfolio, benchmark)
        assert np.isnan(result)

    def test_when_typical_active_then_matches_hand_computed(self) -> None:
        idx = pd.bdate_range("2024-01-01", periods=5)
        portfolio = pd.Series([0.02, -0.01, 0.015, 0.005, 0.0], index=idx)
        benchmark = pd.Series([0.01, -0.005, 0.01, 0.0, 0.005], index=idx)
        active = portfolio - benchmark
        expected = float(active.mean()) / float(active.std(ddof=1)) * np.sqrt(252.0)
        assert ssp._information_ratio(portfolio, benchmark) == pytest.approx(
            expected, rel=1e-9
        )


class TestWriteMetricsJson:
    """Issue #544: write_metrics_json writer."""

    def test_when_metrics_written_then_file_exists_with_correct_keys(
        self, tmp_path: Any
    ) -> None:
        import json

        metrics_by_label = {
            "Portfolio": {
                "Ann. Return": 0.10,
                "Ann. Vol": 0.15,
                "Sharpe (rf)": 0.8,
                "Sortino": 1.2,
                "Info Ratio": 0.6,
                "Downside Vol": 0.10,
                "Max Drawdown": -0.20,
            },
            "Portfolio (after-tax)": {
                "Ann. Return": 0.08,
                "Ann. Vol": 0.15,
                "Sharpe (rf)": 0.65,
                "Sortino": 1.0,
                "Info Ratio": 0.5,
                "Downside Vol": 0.10,
                "Max Drawdown": -0.22,
            },
        }
        out = ssp.write_metrics_json(metrics_by_label, tmp_path)
        assert out.exists()
        with out.open() as fh:
            data = json.load(fh)
        assert "net_of_cost" in data
        assert "after_tax" in data
        assert data["rf_assumption"] == "FRED DGS3MO daily forward-fill ÷ 252"
        for key in (
            "ann_return",
            "ann_vol",
            "sharpe",
            "sortino",
            "info_ratio",
            "downside_vol",
            "max_drawdown",
        ):
            assert key in data["net_of_cost"]
            assert key in data["after_tax"]

    def test_when_after_tax_missing_then_only_net_block(self, tmp_path: Any) -> None:
        import json

        metrics_by_label = {
            "Portfolio": {
                "Ann. Return": 0.10,
                "Ann. Vol": 0.15,
                "Sharpe (rf)": 0.8,
                "Sortino": 1.2,
                "Info Ratio": 0.6,
                "Downside Vol": 0.10,
                "Max Drawdown": -0.20,
            },
        }
        out = ssp.write_metrics_json(metrics_by_label, tmp_path)
        with out.open() as fh:
            data = json.load(fh)
        assert "net_of_cost" in data
        assert "after_tax" not in data

    def test_when_nan_then_serialised_as_null(self, tmp_path: Any) -> None:
        import json

        metrics_by_label = {
            "Portfolio": {
                "Ann. Return": float("nan"),
                "Ann. Vol": 0.15,
                "Sharpe (rf)": 0.8,
                "Sortino": 1.2,
                "Info Ratio": 0.6,
                "Downside Vol": 0.10,
                "Max Drawdown": -0.20,
            },
        }
        out = ssp.write_metrics_json(metrics_by_label, tmp_path)
        text = out.read_text()
        # `nan` is not valid JSON; helper must emit null
        assert "NaN" not in text
        assert "null" in text
        # Must round-trip with strict json
        json.loads(text)

    def test_when_output_dir_missing_then_created(self, tmp_path: Any) -> None:
        nested = tmp_path / "nested" / "dir"
        metrics_by_label = {
            "Portfolio": {
                "Ann. Return": 0.10,
                "Ann. Vol": 0.15,
                "Sharpe (rf)": 0.8,
                "Sortino": 1.2,
                "Info Ratio": 0.6,
                "Downside Vol": 0.10,
                "Max Drawdown": -0.20,
            },
        }
        out = ssp.write_metrics_json(metrics_by_label, nested)
        assert nested.exists()
        assert out.exists()


# ---------------------------------------------------------------------------
# Issue #545 — _validate_checklist returns 17 structured rule dicts (§10)
# ---------------------------------------------------------------------------


def _good_inputs() -> dict[str, Any]:
    """Build a passing scenario for every rule (boundary-relaxed)."""
    # 25 tickers, 4% each = 100% total, all in spec-passing buckets.
    tickers = [f"T{i:02d}" for i in range(25)]
    weights = [0.04] * 25
    all_weights = list(zip(tickers, weights, strict=True))
    # All 11 GICS Level-1 sectors covered; max sector = 12% (3 tickers × 4%).
    sectors = (
        ["Technology"] * 3  # 12%
        + ["Healthcare"] * 3  # 12%
        + ["Financial Services"] * 3  # 12%
        + ["Industrials"] * 2  # 8%
        + ["Consumer Cyclical"] * 2  # 8%
        + ["Communication Services"] * 2  # 8%
        + ["Consumer Defensive"] * 2  # 8%
        + ["Energy"] * 2  # 8%
        + ["Utilities"] * 2  # 8%
        + ["Real Estate"] * 2  # 8%
        + ["Basic Materials"] * 2  # 8%
    )
    assert len(sectors) == 25
    sector_mapping = dict(zip(tickers, sectors, strict=True))
    countries = (
        ["United States"] * 8
        + ["Japan"] * 5
        + ["United Kingdom"] * 4
        + ["France"] * 4
        + ["Australia"] * 2
        + ["South Africa"] * 2
    )
    assert len(countries) == 25
    country_map = dict(zip(tickers, countries, strict=True))
    currency_map = dict.fromkeys(tickers, "USD")
    metrics = {
        "Portfolio (after-tax)": {
            "Ann. Return": 0.10,
            "Ann. Vol": 0.12,
            "Sharpe (rf)": 1.5,
            "Sortino": 2.0,
            "Info Ratio": 0.7,
            "Downside Vol": 0.05,
            "Max Drawdown": -0.18,
        },
        "SPY (benchmark)": {
            "Ann. Return": 0.08,
            "Ann. Vol": 0.15,
            "Sharpe (rf)": 0.8,
            "Sortino": 1.0,
            "Info Ratio": float("nan"),
            "Downside Vol": 0.10,
            "Max Drawdown": -0.30,
        },
    }
    idx = pd.bdate_range("2018-01-01", "2026-04-01")
    benchmark_returns = pd.Series(0.0005, index=idx)
    net_returns = pd.Series(0.0006, index=idx)
    after_tax_returns = pd.Series(0.0005, index=idx)
    return {
        "all_weights": all_weights,
        "sector_mapping": sector_mapping,
        "country_map": country_map,
        "metrics": metrics,
        "benchmark_returns": benchmark_returns,
        "net_returns": net_returns,
        "after_tax_returns": after_tax_returns,
        "cost_bps_actual": 50.0,
        "currency_map": currency_map,
    }


class TestValidateChecklistStructure:
    """Issue #545: _validate_checklist must return list[dict] of length 17."""

    def test_when_called_then_returns_17_rules_in_order(self) -> None:
        kwargs = _good_inputs()
        rules = ssp._validate_checklist(**kwargs)
        assert isinstance(rules, list)
        assert len(rules) == 17
        for rule in rules:
            assert set(rule.keys()) == {"rule", "pass", "measured", "target"}

    def test_when_called_twice_then_rule_order_is_deterministic(self) -> None:
        kwargs = _good_inputs()
        a = [r["rule"] for r in ssp._validate_checklist(**kwargs)]
        b = [r["rule"] for r in ssp._validate_checklist(**kwargs)]
        assert a == b

    def test_when_passing_inputs_then_all_rules_pass(self) -> None:
        kwargs = _good_inputs()
        rules = ssp._validate_checklist(**kwargs)
        failing = [r for r in rules if not r["pass"]]
        assert failing == [], f"Expected all rules to pass, got: {failing}"


class TestValidateChecklistRules:
    """Per-rule pass/fail boundary tests."""

    def test_when_region_exceeds_60pct_then_region_rule_fails(self) -> None:
        kwargs = _good_inputs()
        # Concentrate everything in United States → North America > 60%
        kwargs["country_map"] = dict.fromkeys(kwargs["country_map"], "United States")
        rules = ssp._validate_checklist(**kwargs)
        region_rule = next(r for r in rules if "region" in r["rule"].lower())
        assert region_rule["pass"] is False

    def test_when_sector_exceeds_15pct_then_sector_rule_fails(self) -> None:
        kwargs = _good_inputs()
        # Push all weight into one sector
        kwargs["sector_mapping"] = dict.fromkeys(
            kwargs["sector_mapping"], "Technology"
        )
        rules = ssp._validate_checklist(**kwargs)
        sector_rule = next(
            r for r in rules if "sector" in r["rule"].lower() and "15" in r["target"]
        )
        assert sector_rule["pass"] is False

    def test_when_hhi_above_threshold_then_hhi_rule_fails(self) -> None:
        kwargs = _good_inputs()
        # 2 stocks at 50% each → HHI = 0.5
        kwargs["all_weights"] = [("T00", 0.5), ("T01", 0.5)]
        kwargs["sector_mapping"] = {
            "T00": "Technology",
            "T01": "Healthcare",
        }
        kwargs["country_map"] = {"T00": "United States", "T01": "United Kingdom"}
        kwargs["currency_map"] = {"T00": "USD", "T01": "GBP"}
        rules = ssp._validate_checklist(**kwargs)
        hhi_rule = next(r for r in rules if "HHI" in r["rule"])
        assert hhi_rule["pass"] is False

    def test_when_top4_above_30pct_then_top4_rule_fails(self) -> None:
        kwargs = _good_inputs()
        kwargs["all_weights"] = [
            ("T00", 0.10),
            ("T01", 0.10),
            ("T02", 0.10),
            ("T03", 0.10),  # top4 = 40%
        ] + [(f"T{i:02d}", 0.60 / 21) for i in range(4, 25)]
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "Top-4" in r["rule"])
        assert rule["pass"] is False

    def test_when_health_below_8pct_then_health_rule_fails(self) -> None:
        kwargs = _good_inputs()
        # Remove all Healthcare
        kwargs["sector_mapping"] = {
            t: ("Technology" if v in ("Healthcare", "Healthcare") else v)
            for t, v in kwargs["sector_mapping"].items()
        }
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "Health" in r["rule"])
        assert rule["pass"] is False

    def test_when_tech_below_10pct_then_tech_rule_fails(self) -> None:
        kwargs = _good_inputs()
        kwargs["sector_mapping"] = {
            t: ("Healthcare" if v == "Technology" else v)
            for t, v in kwargs["sector_mapping"].items()
        }
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "Technology" in r["rule"])
        assert rule["pass"] is False

    def test_when_fewer_than_eight_sectors_then_coverage_rule_fails(self) -> None:
        kwargs = _good_inputs()
        # Collapse to only 7 distinct sectors by mapping multiple to "Technology"
        collapsed_sectors = {
            "Technology": ["Technology", "Healthcare"],
            "Financial Services": ["Financial Services", "Industrials"],
            "Communication Services": ["Communication Services", "Consumer Cyclical"],
            "Energy": ["Energy", "Utilities"],
            "Real Estate": ["Real Estate"],
            "Consumer Defensive": ["Consumer Defensive"],
            "Basic Materials": ["Basic Materials"],
        }
        reverse_map = {}
        for target, sources in collapsed_sectors.items():
            for source in sources:
                reverse_map[source] = target

        kwargs["sector_mapping"] = {
            t: reverse_map.get(v, v)
            for t, v in kwargs["sector_mapping"].items()
        }
        rules = ssp._validate_checklist(**kwargs)
        rule = next(
            r for r in rules
            if "8" in r["target"] and "sector" in r["rule"].lower()
        )
        assert rule["pass"] is False

    def test_when_max_weight_above_10pct_then_single_stock_rule_fails(self) -> None:
        kwargs = _good_inputs()
        kwargs["all_weights"] = [("T00", 0.20)] + [
            (f"T{i:02d}", 0.80 / 24) for i in range(1, 25)
        ]
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "Single-stock" in r["rule"])
        assert rule["pass"] is False

    def test_when_min_weight_below_2pct_then_min_position_rule_fails(self) -> None:
        kwargs = _good_inputs()
        kwargs["all_weights"] = [
            (t, 0.01 if i == 24 else (0.99 / 24))
            for i, t in enumerate([f"T{j:02d}" for j in range(25)])
        ]
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "Min position" in r["rule"])
        assert rule["pass"] is False

    def test_when_max_dd_below_minus_22pct_then_dd_rule_fails(self) -> None:
        kwargs = _good_inputs()
        kwargs["metrics"]["Portfolio (after-tax)"]["Max Drawdown"] = -0.30
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "drawdown" in r["rule"].lower())
        assert rule["pass"] is False

    def test_when_vol_above_benchmark_vol_then_vol_rule_fails(self) -> None:
        kwargs = _good_inputs()
        kwargs["metrics"]["Portfolio (after-tax)"]["Ann. Vol"] = 0.20
        kwargs["metrics"]["SPY (benchmark)"]["Ann. Vol"] = 0.15
        rules = ssp._validate_checklist(**kwargs)
        rule = next(
            r
            for r in rules
            if "vol" in r["rule"].lower() and "benchmark" in r["rule"].lower()
        )
        assert rule["pass"] is False

    def test_when_sharpe_outside_range_then_sharpe_rule_fails(self) -> None:
        kwargs = _good_inputs()
        kwargs["metrics"]["Portfolio (after-tax)"]["Sharpe (rf)"] = 0.5
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "Sharpe" in r["rule"])
        assert rule["pass"] is False

    def test_when_sortino_below_15_then_sortino_rule_fails(self) -> None:
        kwargs = _good_inputs()
        kwargs["metrics"]["Portfolio (after-tax)"]["Sortino"] = 1.0
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "Sortino" in r["rule"])
        assert rule["pass"] is False

    def test_when_ir_below_05_then_ir_rule_fails(self) -> None:
        kwargs = _good_inputs()
        kwargs["metrics"]["Portfolio (after-tax)"]["Info Ratio"] = 0.3
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "Info Ratio" in r["rule"])
        assert rule["pass"] is False

    def test_when_downside_above_75pct_total_then_downside_rule_fails(self) -> None:
        kwargs = _good_inputs()
        # downside_vol > 0.75 × ann_vol → fail
        kwargs["metrics"]["Portfolio (after-tax)"]["Ann. Vol"] = 0.10
        kwargs["metrics"]["Portfolio (after-tax)"]["Downside Vol"] = 0.09
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "Downside" in r["rule"])
        assert rule["pass"] is False

    def test_when_cost_above_100bps_then_cost_rule_fails(self) -> None:
        kwargs = _good_inputs()
        kwargs["cost_bps_actual"] = 150.0
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "cost" in r["rule"].lower())
        assert rule["pass"] is False

    def test_when_oos_span_below_8_years_then_oos_rule_fails(self) -> None:
        kwargs = _good_inputs()
        idx = pd.bdate_range("2024-01-01", "2024-06-01")
        kwargs["net_returns"] = pd.Series(0.001, index=idx)
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "OOS" in r["rule"])
        assert rule["pass"] is False


class TestValidateChecklistMissingData:
    """NaN metric inputs → pass=False, measured='N/A'."""

    def test_when_sharpe_nan_then_pass_false_and_na(self) -> None:
        kwargs = _good_inputs()
        kwargs["metrics"]["Portfolio (after-tax)"]["Sharpe (rf)"] = float("nan")
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "Sharpe" in r["rule"])
        assert rule["pass"] is False
        assert rule["measured"] == "N/A"

    def test_when_sortino_nan_then_pass_false_and_na(self) -> None:
        kwargs = _good_inputs()
        kwargs["metrics"]["Portfolio (after-tax)"]["Sortino"] = float("nan")
        rules = ssp._validate_checklist(**kwargs)
        rule = next(r for r in rules if "Sortino" in r["rule"])
        assert rule["pass"] is False
        assert rule["measured"] == "N/A"

    def test_when_no_after_tax_metrics_then_perf_rules_fail_with_na(self) -> None:
        kwargs = _good_inputs()
        kwargs["metrics"].pop("Portfolio (after-tax)")
        rules = ssp._validate_checklist(**kwargs)
        sortino = next(r for r in rules if "Sortino" in r["rule"])
        assert sortino["pass"] is False
        assert sortino["measured"] == "N/A"


class TestValidateChecklistNoSideEffects:
    """No prints inside `_validate_checklist` (caller owns rendering)."""

    def test_when_called_then_no_console_print_in_source(self) -> None:
        import inspect

        src = inspect.getsource(ssp._validate_checklist)
        assert "console.print" not in src


# ---------------------------------------------------------------------------
# Issue #546 — checklist.json + weights.csv gate + exit-code contract (§10)
# ---------------------------------------------------------------------------


def _passing_rule(name: str = "rule") -> dict[str, Any]:
    return {"rule": name, "pass": True, "measured": "10.0%", "target": "≤ 15%"}


def _failing_rule(name: str = "rule") -> dict[str, Any]:
    return {"rule": name, "pass": False, "measured": "20.0%", "target": "≤ 15%"}


def _passing_rules() -> list[dict[str, Any]]:
    return [_passing_rule(f"r{i:02d}") for i in range(17)]


def _good_metrics() -> dict[str, dict[str, float]]:
    base = {
        "Ann. Return": 0.10,
        "Ann. Vol": 0.12,
        "Sharpe (rf)": 1.5,
        "Sortino": 2.0,
        "Info Ratio": 0.7,
        "Downside Vol": 0.05,
        "Max Drawdown": -0.18,
    }
    return {
        "Portfolio (gross)": dict(base),
        "Portfolio": dict(base),
        "Portfolio (after-tax)": dict(base),
    }


class TestWriteChecklistJson:
    """Issue #546: write_checklist_json structure + content."""

    def test_when_all_pass_then_summary_17_of_17(self, tmp_path: Any) -> None:
        import json

        rules = _passing_rules()
        out = ssp.write_checklist_json(
            rules=rules,
            gross_metrics=_good_metrics()["Portfolio (gross)"],
            net_metrics=_good_metrics()["Portfolio"],
            after_tax_metrics=_good_metrics()["Portfolio (after-tax)"],
            output_dir=tmp_path,
        )
        assert out.name == "checklist.json"
        with out.open() as fh:
            data = json.load(fh)
        assert data["summary"]["passed"] == 17
        assert data["summary"]["total"] == 17
        assert isinstance(data["rules"], list)
        assert len(data["rules"]) == 17
        assert {"gross", "net_of_cost", "after_tax"} == set(data["breakdown"].keys())

    def test_when_some_fail_then_summary_partial(self, tmp_path: Any) -> None:
        import json

        rules = _passing_rules()
        rules[0] = _failing_rule("r00")
        rules[5] = _failing_rule("r05")
        out = ssp.write_checklist_json(
            rules=rules,
            gross_metrics=_good_metrics()["Portfolio (gross)"],
            net_metrics=_good_metrics()["Portfolio"],
            after_tax_metrics=_good_metrics()["Portfolio (after-tax)"],
            output_dir=tmp_path,
        )
        with out.open() as fh:
            data = json.load(fh)
        assert data["summary"]["passed"] == 15
        assert data["summary"]["total"] == 17

    def test_when_nan_metric_then_null_in_json(self, tmp_path: Any) -> None:
        import json

        gross = _good_metrics()["Portfolio (gross)"]
        gross["Sharpe (rf)"] = float("nan")
        out = ssp.write_checklist_json(
            rules=_passing_rules(),
            gross_metrics=gross,
            net_metrics=_good_metrics()["Portfolio"],
            after_tax_metrics=_good_metrics()["Portfolio (after-tax)"],
            output_dir=tmp_path,
        )
        text = out.read_text()
        assert "NaN" not in text
        json.loads(text)  # strict round-trip

    def test_when_output_dir_missing_then_created(self, tmp_path: Any) -> None:
        nested = tmp_path / "nested" / "out"
        ssp.write_checklist_json(
            rules=_passing_rules(),
            gross_metrics=_good_metrics()["Portfolio (gross)"],
            net_metrics=_good_metrics()["Portfolio"],
            after_tax_metrics=_good_metrics()["Portfolio (after-tax)"],
            output_dir=nested,
        )
        assert nested.exists()


class TestWriteWeightsCsv:
    """Issue #546: write_weights_csv contract."""

    def test_when_called_then_csv_written_at_expected_path(self, tmp_path: Any) -> None:
        weights = pd.Series([0.10, 0.05, 0.20], index=["B", "C", "A"])
        out = ssp.write_weights_csv(weights, tmp_path)
        assert out == tmp_path / "weights.csv"
        assert out.exists()

    def test_when_called_then_columns_are_ticker_weight(self, tmp_path: Any) -> None:
        weights = pd.Series([0.10, 0.05, 0.20], index=["B", "C", "A"])
        out = ssp.write_weights_csv(weights, tmp_path)
        df = pd.read_csv(out)
        assert list(df.columns) == ["ticker", "weight"]

    def test_when_called_then_sorted_descending_by_weight(self, tmp_path: Any) -> None:
        weights = pd.Series([0.10, 0.05, 0.20], index=["B", "C", "A"])
        out = ssp.write_weights_csv(weights, tmp_path)
        df = pd.read_csv(out)
        assert list(df["ticker"]) == ["A", "B", "C"]
        assert list(df["weight"]) == [0.20, 0.10, 0.05]

    def test_when_output_dir_missing_then_created(self, tmp_path: Any) -> None:
        nested = tmp_path / "nested"
        weights = pd.Series([0.5, 0.5], index=["X", "Y"])
        ssp.write_weights_csv(weights, nested)
        assert nested.exists()


class TestApplyTerminalGate:
    """Issue #546: terminal-gate exit-code contract."""

    def test_when_all_17_pass_then_exit_0_and_weights_written(
        self, tmp_path: Any
    ) -> None:
        weights = pd.Series([0.5, 0.5], index=["X", "Y"])
        with pytest.raises(SystemExit) as exc:
            ssp._apply_terminal_gate(
                rules=_passing_rules(), weights=weights, output_dir=tmp_path
            )
        assert exc.value.code == 0
        assert (tmp_path / "weights.csv").exists()

    def test_when_any_fail_then_exit_1_and_no_weights(self, tmp_path: Any) -> None:
        rules = _passing_rules()
        rules[3] = _failing_rule("r03")
        weights = pd.Series([0.5, 0.5], index=["X", "Y"])
        with pytest.raises(SystemExit) as exc:
            ssp._apply_terminal_gate(rules=rules, weights=weights, output_dir=tmp_path)
        assert exc.value.code == 1
        assert not (tmp_path / "weights.csv").exists()


class TestReportPerformanceReturnContract:
    """Issue #546: report_performance returns (pass_count, rules)."""

    def test_when_called_then_signature_returns_tuple(self) -> None:
        import inspect

        src = inspect.getsource(ssp.report_performance)
        assert "return " in src

    def test_when_called_then_return_annotation_is_tuple(self) -> None:
        import inspect

        sig = inspect.signature(ssp.report_performance)
        # tuple[int, list[dict]] or tuple[int, list[dict[str, Any]]]
        assert "tuple" in str(sig.return_annotation).lower()
