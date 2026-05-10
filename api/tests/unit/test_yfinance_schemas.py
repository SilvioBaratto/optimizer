"""Validation tests for yfinance request schemas (issue #573)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.market_data.yfinance_data import YFinanceFetchRequest


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
