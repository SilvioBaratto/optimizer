"""Whole result of ``MacroRegimeService.fetch_country`` for each source outcome.

Validates ``app/services/macro/macro_regime_service.py``
(``MacroRegimeService.fetch_country``): the exact ``counts`` mapping, key order
included, the exact ``errors`` list and the writes made, for each way the IlSole
and Trading Economics sources and their writes can end.
``test_macro_regime_service.py`` checks single fields of the same paths. No
requirement id is registered for the macro country fetch yet.
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from app.services.macro.macro_regime_service import MacroRegimeService

_FORECAST = {"last_inflation": 3.1, "gdp_growth_6m": 2.2, "timestamp": "t"}
_UNPARSED_FORECAST = {"last_inflation": None, "timestamp": "t"}
_INDICATORS = {"inflation_rate": {"value": 3.1}}
_YIELDS = {"10Y": {"yield": 4.2}}

_KEY_ORDER = [
    "ilsole_forecast",
    "ilsole_observations",
    "te_indicators",
    "te_observations",
    "bond_yields",
    "bond_yield_observations",
]

_NO_ILSOLE_ROW = (
    "ilsole_forecast: no data returned (page unreachable or table structure changed)"
)
_UNPARSED_ILSOLE_ROW = (
    "ilsole_forecast: row found but no forecast value parsed "
    "(page structure may have drifted)"
)


def _te_success(**extra: Any) -> dict[str, Any]:
    return {
        "status": "success",
        "indicators": _INDICATORS,
        "bond_yields": _YIELDS,
        **extra,
    }


def _service(
    forecast: Any = _FORECAST, te_answer: Any = None
) -> tuple[MacroRegimeService, MagicMock, MagicMock]:
    """Build a service over a mock repository; an exception answer is raised."""
    repo = MagicMock(name="repo")
    repo.upsert_economic_indicator.return_value = 1
    repo.upsert_economic_indicator_observation.return_value = 1
    repo.upsert_te_indicators.return_value = 3
    repo.upsert_te_observations.return_value = 3
    repo.upsert_bond_yields.return_value = 2
    repo.upsert_bond_yield_observations.return_value = 2
    ilsole = MagicMock(name="ilsole")
    te = MagicMock(name="te")
    if isinstance(forecast, Exception):
        ilsole.get_forecasts.side_effect = forecast
    else:
        ilsole.get_forecasts.return_value = forecast
    answer = _te_success() if te_answer is None else te_answer
    if isinstance(answer, Exception):
        te.get_country_indicators.side_effect = answer
    else:
        te.get_country_indicators.return_value = answer
    service = MacroRegimeService(
        repo=repo, ilsole_scraper=ilsole, te_scraper=te, fred_scraper=MagicMock()
    )
    return service, repo, te


def _counts(*values: int) -> dict[str, int]:
    return dict(zip(_KEY_ORDER, values, strict=True))


class TestWholeResult:
    """Every outcome reports all six counts, in one order, and its exact errors."""

    @pytest.mark.parametrize(
        ("forecast", "te_answer", "include_bonds", "country", "expected"),
        [
            pytest.param(
                _FORECAST,
                None,
                True,
                "USA",
                {"counts": _counts(1, 1, 3, 3, 2, 2), "errors": []},
                id="everything-stored",
            ),
            pytest.param(
                RuntimeError("boom"),
                {"status": "error"},
                True,
                "USA",
                {
                    "counts": _counts(0, 0, 0, 0, 0, 0),
                    "errors": [
                        "ilsole_forecast: boom",
                        "trading_economics: Unknown error",
                    ],
                },
                id="both-sources-fail",
            ),
            pytest.param(
                _FORECAST,
                {"status": "error", "error": "HTML parse failed"},
                True,
                "USA",
                {
                    "counts": _counts(1, 1, 0, 0, 0, 0),
                    "errors": ["trading_economics: HTML parse failed"],
                },
                id="te-reports-an-error",
            ),
            pytest.param(
                _UNPARSED_FORECAST,
                _te_success(indicators={}, bond_yields_error=None),
                True,
                "USA",
                {
                    "counts": _counts(0, 0, 0, 0, 2, 2),
                    "errors": [
                        _UNPARSED_ILSOLE_ROW,
                        "trading_economics_bonds: Unknown error",
                    ],
                },
                id="unparsed-row-no-indicators-bond-error-with-yields",
            ),
            pytest.param(
                None,
                _te_success(bond_yields_error="bond table not found"),
                False,
                "Atlantis",
                {"counts": _counts(0, 0, 3, 3, 0, 0), "errors": []},
                id="no-ilsole-table-bonds-not-requested",
            ),
            pytest.param(
                None,
                _te_success(bond_yields={}),
                True,
                "USA",
                {"counts": _counts(0, 0, 3, 3, 0, 0), "errors": [_NO_ILSOLE_ROW]},
                id="missing-ilsole-row-no-yields",
            ),
            pytest.param(
                _FORECAST,
                RuntimeError("connection refused"),
                True,
                "USA",
                {
                    "counts": _counts(1, 1, 0, 0, 0, 0),
                    "errors": ["trading_economics: connection refused"],
                },
                id="te-raises",
            ),
        ],
    )
    def test_the_result_is_exactly(
        self,
        forecast: Any,
        te_answer: Any,
        include_bonds: bool,
        country: str,
        expected: dict[str, Any],
    ) -> None:
        """Pin the whole result, including the order of the count keys."""
        service, _, _ = _service(forecast, te_answer)

        result = service.fetch_country(country, include_bonds=include_bonds)

        assert result == expected
        assert list(result) == ["counts", "errors"]
        assert list(result["counts"]) == _KEY_ORDER


class TestFailedWrites:
    """A failed write zeroes only the counts of the writes it rolled back."""

    def test_a_failed_forecast_write_is_an_ilsole_error(self) -> None:
        """The forecast counts stay zero and Trading Economics still runs."""
        service, repo, _ = _service()
        repo.upsert_economic_indicator_observation.side_effect = RuntimeError(
            "value too long"
        )

        result = service.fetch_country("USA")

        assert result == {
            "counts": _counts(0, 0, 3, 3, 2, 2),
            "errors": ["ilsole_forecast: value too long"],
        }

    def test_a_failed_indicator_write_zeroes_every_te_count(self) -> None:
        """The indicators failed before the bonds, so no bond write is tried."""
        service, repo, _ = _service()
        repo.upsert_te_observations.side_effect = RuntimeError("deadlock")

        result = service.fetch_country("USA")

        assert result == {
            "counts": _counts(1, 1, 0, 0, 0, 0),
            "errors": ["trading_economics: deadlock"],
        }
        repo.upsert_bond_yields.assert_not_called()

    def test_a_failed_bond_write_keeps_the_indicator_counts(self) -> None:
        """Indicators were stored in their own savepoint before the bonds failed."""
        service, repo, _ = _service()
        repo.upsert_bond_yield_observations.side_effect = RuntimeError("deadlock")

        result = service.fetch_country("USA")

        assert result == {
            "counts": _counts(1, 1, 3, 3, 0, 0),
            "errors": ["trading_economics: deadlock"],
        }
        assert list(result["counts"]) == _KEY_ORDER


class TestWrites:
    """What each source hands to the repository, and in which savepoints."""

    def test_each_source_is_written_with_todays_snapshot_date(self) -> None:
        """Three savepoints: the forecast, the indicators, then the bond yields."""
        service, repo, te = _service()
        before = datetime.date.today()

        service.fetch_country("Germany", include_bonds=True)

        after = datetime.date.today()
        te.get_country_indicators.assert_called_once_with("Germany", include_bonds=True)
        snapshot = repo.upsert_economic_indicator_observation.call_args.kwargs[
            "snapshot_date"
        ]
        assert snapshot in {before, after}
        assert repo.method_calls[:1] == [call.session.begin_nested()]
        assert repo.upsert_economic_indicator.call_args == call(
            country="Germany", data=_FORECAST
        )
        assert repo.upsert_economic_indicator_observation.call_args == call(
            country="Germany", snapshot_date=snapshot, data=_FORECAST
        )
        assert repo.upsert_te_indicators.call_args == call(
            country="Germany", indicators_dict=_INDICATORS
        )
        assert repo.upsert_te_observations.call_args == call(
            country="Germany", snapshot_date=snapshot, indicators_dict=_INDICATORS
        )
        assert repo.upsert_bond_yields.call_args == call(
            country="Germany", yields_dict=_YIELDS
        )
        assert repo.upsert_bond_yield_observations.call_args == call(
            country="Germany", snapshot_date=snapshot, yields_dict=_YIELDS
        )
        assert repo.session.begin_nested.call_count == 3
