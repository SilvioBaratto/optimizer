"""Pin ``_NAN_METRICS``, the NaN statistics of an under-sampled slice.

``run_regime_validation`` (short sub-periods and short regimes) and
``_compute_period_metrics`` (an empty slice) all report it, as copies of one
read-only constant.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from optimizer.validation import (
    RegimeValidationConfig,
    RegimeValidationResult,
    run_regime_validation,
)
from optimizer.validation._regime import _NAN_METRICS, _compute_period_metrics

_EXPANSION = {"pmi": 55.0, "spread_2s10s": 1.5, "hy_oas": 300.0}
_RECESSION = {"pmi": 46.0, "spread_2s10s": -0.5, "hy_oas": 600.0}
_BLOCKS = (_EXPANSION, _RECESSION, _EXPANSION)
_DAYS_PER_BLOCK = 6
# Longer than the whole sample, so no sub-period or regime reaches it.
_MIN_OBS = len(_BLOCKS) * _DAYS_PER_BLOCK + 1


def _short_blocks_result() -> RegimeValidationResult:
    """Run a validation whose every sub-period and regime is under the minimum."""
    macro_rows = [row for row in _BLOCKS for _ in range(_DAYS_PER_BLOCK)]
    days = pd.bdate_range("2024-01-02", periods=len(macro_rows))
    returns = pd.Series(np.linspace(-0.01, 0.01, len(macro_rows)), index=days)
    macro = pd.DataFrame(macro_rows, index=days)
    return run_regime_validation(
        returns, macro, RegimeValidationConfig(min_regime_obs=_MIN_OBS)
    )


def _metric_rows(frame: pd.DataFrame) -> list[dict[str, float]]:
    return frame[list(_NAN_METRICS)].to_dict("records")


def test_nan_metrics_dict_is_shared() -> None:
    """Short sub-periods, short regimes and an empty slice report one NaN dict."""
    result = _short_blocks_result()
    expected = dict(_NAN_METRICS)

    subperiod_rows = _metric_rows(result.per_subperiod_metrics)
    regime_rows = _metric_rows(result.per_regime_metrics)
    empty_slice = _compute_period_metrics(pd.Series(dtype=float), 252, 0.0)

    assert len(subperiod_rows) == 3
    assert len(regime_rows) == 2
    for row in [*subperiod_rows, *regime_rows, empty_slice]:
        np.testing.assert_equal(row, expected)
    assert list(empty_slice) == list(_NAN_METRICS)
    assert list(result.per_subperiod_metrics.columns[-4:]) == list(_NAN_METRICS)
    assert list(result.per_regime_metrics.columns[-4:]) == list(_NAN_METRICS)


def test_nan_metrics_constant_is_read_only_and_never_handed_out() -> None:
    """A caller that edits its NaN dict cannot change the shared constant."""
    handed_out = _compute_period_metrics(pd.Series(dtype=float), 252, 0.0)
    handed_out["sharpe"] = 1.0

    assert isinstance(_NAN_METRICS, Mapping)
    assert math.isnan(_NAN_METRICS["sharpe"])
    assert _compute_period_metrics(pd.Series(dtype=float), 252, 0.0) is not handed_out
    with pytest.raises(TypeError):
        _NAN_METRICS["sharpe"] = 0.0  # type: ignore[index]
