"""Validation tests for yfinance request schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.market_data.yfinance_data import (
    YFinanceFetchRequest,
    YFinanceSingleFetchRequest,
)


class TestYFinanceFetchRequestWorkers:
    def test_when_no_workers_provided_default_is_one(self):
        request = YFinanceFetchRequest()
        assert request.workers == 1

    def test_when_workers_eight_then_accepted(self):
        request = YFinanceFetchRequest(workers=8)
        assert request.workers == 8

    def test_when_workers_zero_then_validation_error_raised(self):
        with pytest.raises(ValidationError):
            YFinanceFetchRequest(workers=0)

    def test_when_workers_seventeen_then_validation_error_raised(self):
        with pytest.raises(ValidationError):
            YFinanceFetchRequest(workers=17)

    def test_when_legacy_max_workers_sent_then_silently_ignored_workers_default(self):
        request = YFinanceFetchRequest(max_workers=4)  # type: ignore[call-arg]
        assert request.workers == 1
        assert not hasattr(request, "max_workers")


_REQUESTS = [YFinanceFetchRequest, YFinanceSingleFetchRequest]


class TestPeriodValidation:
    """yfinance answers an unrecognised period with an empty frame rather than an
    error, so an unvalidated typo would read as a successful all-empty sweep."""

    @pytest.mark.parametrize("model", _REQUESTS)
    @pytest.mark.parametrize(
        "period",
        ["1d", "5d", "60d", "2wk", "1mo", "6mo", "1y", "5y", "10y", "ytd", "max"],
    )
    def test_when_period_is_valid_yfinance_period_then_accepted(
        self, model, period: str
    ):
        assert model(period=period).period == period

    @pytest.mark.parametrize("model", _REQUESTS)
    @pytest.mark.parametrize(
        "period",
        ["5yrs", "5", "y", "", "0y", "-1y", "5 y", "5y\n", "MAX", "1.5y", "5m"],
    )
    def test_when_period_is_not_a_yfinance_period_then_validation_error_raised(
        self, model, period: str
    ):
        with pytest.raises(ValidationError):
            model(period=period)

    @pytest.mark.parametrize("model", _REQUESTS)
    def test_when_no_period_provided_default_is_five_years(self, model):
        assert model().period == "5y"
