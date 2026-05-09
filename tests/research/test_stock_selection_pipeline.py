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
