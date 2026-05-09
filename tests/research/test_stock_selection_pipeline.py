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

from research import stock_selection_pipeline as ssp


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
            patch.object(ssp, "run_factor_validation", side_effect=fake_run_validation),
            patch.object(ssp, "validate_factors", return_value=_StubVifReport()),
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
                ssp, "run_factor_validation", return_value=FactorValidationReport()
            ),
            patch.object(ssp, "validate_factors", return_value=_StubVifReport()),
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
                ssp, "run_factor_oos_validation", return_value=_empty_oos_result(0)
            ),
            pytest.raises(RuntimeError, match="0 folds"),
        ):
            ssp.validate_oos(_stub_factor_scores(), _stub_returns())

    def test_when_n_folds_zero_message_includes_oos_config_params(self) -> None:
        with (
            patch.object(
                ssp, "run_factor_oos_validation", return_value=_empty_oos_result(0)
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
        with patch.object(ssp, "run_factor_oos_validation", return_value=stub):
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
            ssp, "run_full_pipeline_with_selection", side_effect=fake_pipeline
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

    def test_when_db_manager_supplied_repository_upsert_called_with_us(
        self,
    ) -> None:
        from types import SimpleNamespace

        from optimizer.factors import MacroRegime

        assembly = SimpleNamespace(
            macro_data=pd.DataFrame({"gdp_growth": [1.0]}),
            fred_data=pd.DataFrame(),
            te_observations=pd.DataFrame(),
            sentiment_data=pd.DataFrame(),
        )
        recorded: list[tuple[str, str]] = []

        class _StubRepo:
            def __init__(self, _session: Any) -> None:
                pass

            def upsert_regime_classification(self, country: str, regime: str) -> None:
                recorded.append((country, regime))

        from contextlib import contextmanager
        from unittest.mock import MagicMock

        class _StubDb:
            @contextmanager
            def get_session(self) -> Any:
                session = MagicMock()
                try:
                    yield session
                finally:
                    pass

        # Patch the late-imported MacroRegimeRepository inside the helper.
        import sys

        fake_module = type(sys)("app.repositories.macro_regime_repository")
        fake_module.MacroRegimeRepository = _StubRepo  # type: ignore[attr-defined]
        with (
            patch.dict(
                sys.modules,
                {"app.repositories.macro_regime_repository": fake_module},
            ),
            patch.object(ssp, "classify_regime", return_value=MacroRegime.RECESSION),
        ):
            ssp.classify_and_tilt(
                assembly,  # type: ignore[arg-type]
                db_manager=_StubDb(),  # type: ignore[arg-type]
            )

        assert recorded == [("US", "recession")]

    def test_when_db_manager_omitted_no_persistence_attempt(self) -> None:
        from types import SimpleNamespace

        from optimizer.factors import MacroRegime

        assembly = SimpleNamespace(
            macro_data=pd.DataFrame({"gdp_growth": [1.0]}),
            fred_data=pd.DataFrame(),
            te_observations=pd.DataFrame(),
            sentiment_data=pd.DataFrame(),
        )
        with patch.object(ssp, "classify_regime", return_value=MacroRegime.EXPANSION):
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
            ssp, "run_full_pipeline_with_selection", side_effect=fake_pipeline
        ):
            ssp.optimize_portfolio(
                assembly=self._stub_assembly(),
                investable=pd.Index(["AAA", "BBB"]),
                ic_history=None,
                n_selected=n_selected,
            )
        return captured

    def test_when_optimize_portfolio_called_scoring_config_uses_ic_decay_halflife_4(
        self,
    ) -> None:
        from optimizer.factors import CompositeMethod

        captured = self._capture_kwargs(n_selected=30)
        cfg = captured["scoring_config"]
        assert cfg.method == CompositeMethod.IC_WEIGHTED
        assert cfg.ic_decay_halflife == 4

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
        assert sel.max_per_sector == 8


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
            "Materials",
            "Industrials",
            "Consumer Discretionary",
            "Consumer Staples",
            "Health Care",
            "Financials",
            "Information Technology",
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
        sector_mapping = {"AAA": "Energy", "BBB": "Financials"}
        missing = ssp._missing_gics_sectors(weights, sector_mapping)
        assert "Energy" not in missing
        assert "Financials" not in missing
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
        ssp, "run_full_pipeline_with_selection", side_effect=fake_pipeline
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
        assert hasattr(ssp, "_REGION_MAP")
        assert isinstance(ssp._REGION_MAP, dict)
        assert ssp._REGION_MAP["United States"] == "Americas"
        assert ssp._REGION_MAP["Germany"] == "Europe"
        assert ssp._REGION_MAP["Japan"] == "Asia-Pacific"


class TestHardConstrainedMeanRiskSpec:
    """§7.1 spec wiring of optimize_portfolio."""

    def test_when_called_then_objective_is_maximize_ratio(self) -> None:
        from skfolio.optimization.convex._base import ObjectiveFunction

        captured = _capture_optimizer()
        opt = captured["optimizer"]
        assert opt.objective_function == ObjectiveFunction.MAXIMIZE_RATIO

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
        assert params["eps_abs"] == pytest.approx(1e-8)
        assert params["eps_rel"] == pytest.approx(1e-8)

    def test_when_called_then_min_sector_weights_floors_present(self) -> None:
        captured = _capture_optimizer()
        constraints = captured["optimizer"].linear_constraints or []
        assert any("Healthcare >= 0.08" in c for c in constraints)
        assert any("Technology >= 0.1" in c for c in constraints)

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
                ssp, "run_full_pipeline_with_selection", side_effect=fake_pipeline
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
                ssp, "run_full_pipeline_with_selection", side_effect=fake_pipeline
            ),
            patch.object(ssp, "_hockey_stick_warn") as mock_warn,
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
        sectors = ["Healthcare", "Technology", "Financials", "Energy"]
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

    def test_when_universe_above_top4_then_retighten_runs_and_trace_attached(
        self,
    ) -> None:
        from types import SimpleNamespace

        fake_trace = [{"attempt": 1, "top4": 0.25, "max_weights": 0.10}]

        def fake_pipeline(**kwargs: Any) -> Any:
            return SimpleNamespace(
                summary={"sharpe_ratio": 1.0},
                net_sharpe_ratio=None,
                weights=pd.Series([0.125] * 8, index=[f"T{i:02d}" for i in range(8)]),
            )

        with (
            patch.object(
                ssp, "run_full_pipeline_with_selection", side_effect=fake_pipeline
            ),
            patch.object(
                ssp,
                "_solve_with_retighten",
                return_value=("STUB_OPT", fake_trace),
            ) as mock_retighten,
        ):
            assembly = self._wide_assembly(n_tickers=8)
            result = ssp.optimize_portfolio(
                assembly=assembly,
                investable=pd.Index([f"T{i:02d}" for i in range(8)]),
                ic_history=None,
                n_selected=8,
            )

        assert mock_retighten.called
        assert result.retighten_trace == fake_trace

    def test_when_universe_at_or_below_top4_then_retighten_skipped(self) -> None:
        from types import SimpleNamespace

        def fake_pipeline(**kwargs: Any) -> Any:
            return SimpleNamespace(
                summary={"sharpe_ratio": 1.0},
                net_sharpe_ratio=None,
                weights=pd.Series([0.5, 0.5], index=["T00", "T01"]),
            )

        with (
            patch.object(
                ssp, "run_full_pipeline_with_selection", side_effect=fake_pipeline
            ),
            patch.object(ssp, "_solve_with_retighten") as mock_retighten,
        ):
            assembly = self._wide_assembly(n_tickers=2)
            result = ssp.optimize_portfolio(
                assembly=assembly,
                investable=pd.Index(["T00", "T01"]),
                ic_history=None,
                n_selected=2,
            )

        assert not mock_retighten.called
        assert result.retighten_trace == []
