"""Tests for the Cycle-2 §4.4 factor coverage gate (issue #528)."""

from __future__ import annotations

import pandas as pd
import pytest

from optimizer.exceptions import FactorCoverageError
from optimizer.factors import FactorOOSResult, FactorValidationReport

pytest.importorskip("rich")

from research import stock_selection_pipeline as ssp


def _report(significant: list[str]) -> FactorValidationReport:
    return FactorValidationReport(significant_factors=list(significant))


def _oos(icir_by_factor: dict[str, float]) -> FactorOOSResult:
    factors = list(icir_by_factor)
    return FactorOOSResult(
        per_fold_ic=pd.DataFrame(0.05, index=range(3), columns=factors),
        per_fold_spread=pd.DataFrame(0.01, index=range(3), columns=factors),
        mean_oos_ic=pd.Series([0.05] * len(factors), index=factors),
        mean_oos_icir=pd.Series(icir_by_factor),
        n_folds=3,
    )


class TestCheckFactorCoverage:
    def test_when_five_factors_pass_both_no_raise(self) -> None:
        is_report = _report(["a", "b", "c", "d", "e"])
        oos_result = _oos({"a": 1.0, "b": 1.5, "c": 0.5, "d": 2.0, "e": 0.1})
        ssp._check_factor_coverage(is_report, oos_result)

    def test_when_three_factors_pass_both_raises_factor_coverage_error(self) -> None:
        is_report = _report(["a", "b", "c"])
        oos_result = _oos({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0})
        with pytest.raises(FactorCoverageError, match="Factor coverage gate failed"):
            ssp._check_factor_coverage(is_report, oos_result)

    def test_when_failing_message_lists_passing_and_partial_legs(self) -> None:
        # IS={a,b,c,d}, OOS+={a,e,f} → pass={a}, is_only={b,c,d}, oos_only={e,f}
        is_report = _report(["a", "b", "c", "d"])
        oos_result = _oos(
            {"a": 1.0, "b": -0.5, "c": -0.1, "d": -0.2, "e": 1.0, "f": 0.5}
        )
        with pytest.raises(FactorCoverageError) as exc:
            ssp._check_factor_coverage(is_report, oos_result)
        msg = str(exc.value)
        assert "Passing:" in msg and "a" in msg
        assert "IS-only" in msg
        for f in ("b", "c", "d"):
            assert f in msg
        assert "OOS-only" in msg
        for f in ("e", "f"):
            assert f in msg

    def test_when_min_factors_kwarg_overridden_threshold_changes(self) -> None:
        is_report = _report(["a", "b"])
        oos_result = _oos({"a": 1.0, "b": 1.0})
        # 2 passing, threshold 2 → no raise
        ssp._check_factor_coverage(is_report, oos_result, min_factors=2)
        with pytest.raises(FactorCoverageError):
            ssp._check_factor_coverage(is_report, oos_result, min_factors=3)
