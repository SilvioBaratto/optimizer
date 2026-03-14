"""Unit tests for macro regime data read endpoints.

All tests mock MacroRegimeRepository at the router's import site so no
real database is touched. The conftest-provided ``client`` fixture is used
for HTTP-level assertions.

Endpoints covered:
  GET /api/v1/macro-data/countries/{country}                -- get_country_summary
  GET /api/v1/macro-data/te-observations                    -- get_te_observations
  GET /api/v1/macro-data/fred/series                        -- get_fred_observations
  GET /api/v1/macro-data/news                               -- get_macro_news
  GET /api/v1/macro-data/news/themes                        -- get_macro_news_themes
  GET /api/v1/macro-data/fred/catalog                       -- get_fred_catalog
  GET /api/v1/macro-data/countries                          -- get_distinct_countries
  GET /api/v1/macro-data/economic-indicator-observations    -- get_economic_indicator_observations
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Module-level patch target
# ---------------------------------------------------------------------------

_REPO_CLS = "app.api.v1.macro_regime.MacroRegimeRepository"

BASE_URL = "/api/v1/macro-data"


# ---------------------------------------------------------------------------
# Mock object factories
# ---------------------------------------------------------------------------


def _make_te_obs(
    country: str = "USA",
    indicator_key: str = "manufacturing_pmi",
    value: float = 51.2,
) -> MagicMock:
    obj = MagicMock()
    obj.id = uuid.uuid4()
    obj.country = country
    obj.indicator_key = indicator_key
    obj.date = datetime.date(2024, 1, 15)
    obj.value = value
    obj.created_at = datetime.datetime(2024, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)
    obj.updated_at = datetime.datetime(2024, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)
    return obj


def _make_fred_obs(
    series_id: str = "T10Y2Y",
    obs_date: datetime.date = datetime.date(2024, 1, 10),
    value: float = 0.42,
) -> MagicMock:
    obj = MagicMock()
    obj.id = uuid.uuid4()
    obj.series_id = series_id
    obj.date = obs_date
    obj.value = value
    obj.created_at = datetime.datetime(2024, 1, 10, 12, 0, tzinfo=datetime.timezone.utc)
    obj.updated_at = datetime.datetime(2024, 1, 10, 12, 0, tzinfo=datetime.timezone.utc)
    return obj


def _make_macro_news(
    news_id: str = "abc123",
    title: str = "Fed raises rates",
    themes: str = "monetary_policy",
) -> MagicMock:
    obj = MagicMock()
    obj.id = uuid.uuid4()
    obj.news_id = news_id
    obj.title = title
    obj.publisher = "Reuters"
    obj.link = "https://example.com/news"
    obj.publish_time = datetime.datetime(
        2024, 1, 15, 12, 0, tzinfo=datetime.timezone.utc
    )
    obj.source_ticker = "^TNX"
    obj.source_query = None
    obj.themes = themes
    obj.snippet = "The Federal Reserve..."
    obj.full_content = None
    obj.created_at = datetime.datetime(2024, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)
    obj.updated_at = datetime.datetime(2024, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)
    return obj


def _make_econ_obs(
    country: str = "USA",
    obs_date: datetime.date = datetime.date(2024, 1, 15),
) -> MagicMock:
    obj = MagicMock()
    obj.id = uuid.uuid4()
    obj.country = country
    obj.date = obs_date
    obj.last_inflation = 3.1
    obj.inflation_6m = 2.8
    obj.inflation_10y_avg = 2.5
    obj.gdp_growth_6m = 1.4
    obj.earnings_12m = 8.2
    obj.eps_expected_12m = 7.9
    obj.peg_ratio = 1.8
    obj.lt_rate_forecast = 4.2
    obj.reference_date = datetime.date(2024, 1, 1)
    obj.created_at = datetime.datetime(2024, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)
    obj.updated_at = datetime.datetime(2024, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)
    return obj


def _make_mock_repo(**method_overrides: Any) -> MagicMock:
    """Return a MagicMock repo with sensible empty defaults."""
    repo = MagicMock()
    repo.get_country_summary.return_value = {
        "economic_indicators": [],
        "te_indicators": [],
        "bond_yields": [],
    }
    repo.get_te_observations.return_value = []
    repo.get_bond_yield_observations.return_value = []
    repo.get_fred_observations.return_value = []
    repo.get_macro_news.return_value = []
    repo.get_distinct_countries.return_value = []
    repo.get_economic_indicator_observations.return_value = []
    for attr, val in method_overrides.items():
        getattr(repo, attr).return_value = val
    return repo


def _patch_repo(mock_repo: MagicMock):
    """Patch MacroRegimeRepository so every instantiation returns mock_repo."""
    return patch(_REPO_CLS, return_value=mock_repo)


# ===========================================================================
# GET /api/v1/macro-data/countries/{country}
# ===========================================================================


class TestGetCountrySummary:
    URL = f"{BASE_URL}/countries/{{country}}"

    def test_returns_200_with_empty_data(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            resp = client.get(self.URL.format(country="USA"))

        assert resp.status_code == 200
        body = resp.json()
        assert body["country"] == "USA"
        assert body["economic_indicators"] == []
        assert body["te_indicators"] == []
        assert body["bond_yields"] == []

    def test_unknown_country_returns_200_with_empty_lists(
        self, client: TestClient
    ) -> None:
        """Current implementation does NOT raise 404 for unknown countries."""
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            resp = client.get(self.URL.format(country="Atlantis"))

        assert resp.status_code == 200
        assert resp.json()["country"] == "Atlantis"

    def test_country_forwarded_verbatim_to_repo(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(self.URL.format(country="Germany"))

        mock_repo.get_country_summary.assert_called_once_with("Germany")


# ===========================================================================
# GET /api/v1/macro-data/te-observations
# ===========================================================================


class TestGetTeObservations:
    URL = f"{BASE_URL}/te-observations"

    def test_no_filters_returns_all(self, client: TestClient) -> None:
        obs = [_make_te_obs("USA"), _make_te_obs("UK", "gdp_growth", 0.3)]
        mock_repo = _make_mock_repo(get_te_observations=obs)

        with _patch_repo(mock_repo):
            resp = client.get(self.URL)

        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_country_filter_forwarded(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo(get_te_observations=[_make_te_obs("USA")])
        with _patch_repo(mock_repo):
            client.get(self.URL, params={"country": "USA"})

        call_kw = mock_repo.get_te_observations.call_args.kwargs
        assert call_kw["country"] == "USA"

    def test_indicator_keys_filter_forwarded(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(
                self.URL,
                params={"indicator_keys": ["manufacturing_pmi", "gdp_growth"]},
            )

        call_kw = mock_repo.get_te_observations.call_args.kwargs
        assert call_kw["indicator_keys"] == ["manufacturing_pmi", "gdp_growth"]

    def test_limit_slices_result(self, client: TestClient) -> None:
        obs_list = [_make_te_obs("USA", f"ind_{i}") for i in range(10)]
        mock_repo = _make_mock_repo(get_te_observations=obs_list)

        with _patch_repo(mock_repo):
            resp = client.get(self.URL, params={"limit": 3})

        assert len(resp.json()) == 3

    def test_date_range_forwarded(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(
                self.URL,
                params={"start_date": "2024-01-01", "end_date": "2024-03-31"},
            )

        call_kw = mock_repo.get_te_observations.call_args.kwargs
        assert call_kw["start_date"] == datetime.date(2024, 1, 1)
        assert call_kw["end_date"] == datetime.date(2024, 3, 31)


# ===========================================================================
# GET /api/v1/macro-data/fred/series
# ===========================================================================


class TestGetFredObservations:
    URL = f"{BASE_URL}/fred/series"

    def test_no_filter_returns_all(self, client: TestClient) -> None:
        obs = [_make_fred_obs("T10Y2Y"), _make_fred_obs("VIXCLS")]
        mock_repo = _make_mock_repo(get_fred_observations=obs)

        with _patch_repo(mock_repo):
            resp = client.get(self.URL)

        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_series_id_filter_forwarded(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(self.URL, params={"series_id": "T10Y2Y"})

        call_kw = mock_repo.get_fred_observations.call_args.kwargs
        assert call_kw["series_id"] == "T10Y2Y"

    def test_start_and_end_date_forwarded(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(
                self.URL,
                params={"start_date": "2024-01-01", "end_date": "2024-06-30"},
            )

        call_kw = mock_repo.get_fred_observations.call_args.kwargs
        assert call_kw["start_date"] == datetime.date(2024, 1, 1)
        assert call_kw["end_date"] == datetime.date(2024, 6, 30)

    def test_default_limit_is_500(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(self.URL)

        call_kw = mock_repo.get_fred_observations.call_args.kwargs
        assert call_kw["limit"] == 500

    def test_custom_limit_forwarded(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(self.URL, params={"limit": 50})

        call_kw = mock_repo.get_fred_observations.call_args.kwargs
        assert call_kw["limit"] == 50

    def test_response_contains_series_id_and_date(self, client: TestClient) -> None:
        obs = _make_fred_obs("T10Y2Y", datetime.date(2024, 3, 1), 0.25)
        mock_repo = _make_mock_repo(get_fred_observations=[obs])

        with _patch_repo(mock_repo):
            resp = client.get(self.URL)

        item = resp.json()[0]
        assert item["series_id"] == "T10Y2Y"
        assert item["date"] == "2024-03-01"
        assert item["value"] == pytest.approx(0.25)

    def test_unknown_series_id_returns_400(self, client: TestClient) -> None:
        """GET /fred/series?series_id=NONEXISTENT must return 400, not 200 []."""
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            resp = client.get(self.URL, params={"series_id": "NONEXISTENT"})

        assert resp.status_code == 400
        mock_repo.get_fred_observations.assert_not_called()

    def test_unknown_series_id_detail_names_the_bad_id(
        self, client: TestClient
    ) -> None:
        """Error detail must identify the rejected series_id."""
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            resp = client.get(self.URL, params={"series_id": "BOGUS_SERIES"})

        body = resp.json()
        message = body.get("detail") or body["error"]["message"]
        assert "BOGUS_SERIES" in message

    def test_known_series_id_bypasses_400(self, client: TestClient) -> None:
        """A valid catalog ID must pass validation and reach the repo."""
        obs = [_make_fred_obs("T10Y2Y")]
        mock_repo = _make_mock_repo(get_fred_observations=obs)
        with _patch_repo(mock_repo):
            resp = client.get(self.URL, params={"series_id": "T10Y2Y"})

        assert resp.status_code == 200
        mock_repo.get_fred_observations.assert_called_once()

    def test_no_series_id_skips_validation(self, client: TestClient) -> None:
        """Omitting series_id must not trigger catalog validation."""
        mock_repo = _make_mock_repo(get_fred_observations=[])
        with _patch_repo(mock_repo):
            resp = client.get(self.URL)

        assert resp.status_code == 200
        mock_repo.get_fred_observations.assert_called_once()


# ===========================================================================
# GET /api/v1/macro-data/news  -- theme exact-match filtering
# ===========================================================================


class TestGetMacroNews:
    URL = f"{BASE_URL}/news"

    def test_no_theme_filter_returns_all(self, client: TestClient) -> None:
        news_list = [
            _make_macro_news("id1", themes="monetary_policy"),
            _make_macro_news("id2", themes="yield_curve"),
        ]
        mock_repo = _make_mock_repo(get_macro_news=news_list)

        with _patch_repo(mock_repo):
            resp = client.get(self.URL)

        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_theme_forwarded_verbatim(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(self.URL, params={"theme": "monetary_policy"})

        call_kw = mock_repo.get_macro_news.call_args.kwargs
        assert call_kw["theme"] == "monetary_policy"

    def test_theme_policy_not_expanded_to_monetary_policy(
        self, client: TestClient
    ) -> None:
        """Route passes theme='policy' literally -- no substring expansion."""
        mock_repo = _make_mock_repo(get_macro_news=[])
        with _patch_repo(mock_repo):
            client.get(self.URL, params={"theme": "policy"})

        call_kw = mock_repo.get_macro_news.call_args.kwargs
        assert call_kw["theme"] == "policy"

    def test_start_date_converted_to_datetime(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(self.URL, params={"start_date": "2024-01-01"})

        start_dt = mock_repo.get_macro_news.call_args.kwargs["start_date"]
        assert start_dt.date() == datetime.date(2024, 1, 1)
        assert start_dt.hour == 0 and start_dt.minute == 0

    def test_end_date_converted_to_end_of_day(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(self.URL, params={"end_date": "2024-03-31"})

        end_dt = mock_repo.get_macro_news.call_args.kwargs["end_date"]
        assert end_dt.date() == datetime.date(2024, 3, 31)
        assert end_dt.hour == 23

    def test_default_limit_is_50(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(self.URL)

        call_kw = mock_repo.get_macro_news.call_args.kwargs
        assert call_kw["limit"] == 50


# ===========================================================================
# GET /api/v1/macro-data/news/themes
# ===========================================================================


class TestGetMacroNewsThemes:
    URL = f"{BASE_URL}/news/themes"

    def test_returns_200_with_list(self, client: TestClient) -> None:
        resp = client.get(self.URL)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_each_item_has_value_and_label(self, client: TestClient) -> None:
        resp = client.get(self.URL)
        for item in resp.json():
            assert "value" in item
            assert "label" in item

    def test_all_macro_theme_values_present(self, client: TestClient) -> None:
        from app.services.yfinance.news.macro_news import MacroTheme

        resp = client.get(self.URL)
        returned = {item["value"] for item in resp.json()}
        expected = {t.value for t in MacroTheme}
        assert returned == expected

    def test_exactly_nine_themes(self, client: TestClient) -> None:
        resp = client.get(self.URL)
        assert len(resp.json()) == 9

    @pytest.mark.parametrize(
        "value,expected_label",
        [
            ("monetary_policy", "Monetary Policy"),
            ("yield_curve", "Yield Curve"),
            ("business_cycle", "Business Cycle"),
            ("sector_rotation", "Sector Rotation"),
            ("volatility_risk", "Volatility Risk"),
            ("commodity_inflation", "Commodity Inflation"),
            ("geographic_allocation", "Geographic Allocation"),
            ("growth_indicators", "Growth Indicators"),
            ("credit_conditions", "Credit Conditions"),
        ],
    )
    def test_label_is_title_cased(
        self, client: TestClient, value: str, expected_label: str
    ) -> None:
        resp = client.get(self.URL)
        item = next((i for i in resp.json() if i["value"] == value), None)
        assert item is not None, f"Theme '{value}' not found"
        assert item["label"] == expected_label


# ===========================================================================
# GET /api/v1/macro-data/fred/catalog
# ===========================================================================


class TestGetFredCatalog:
    URL = f"{BASE_URL}/fred/catalog"

    def test_returns_200_with_list(self, client: TestClient) -> None:
        resp = client.get(self.URL)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_each_item_has_series_id_description_group(
        self, client: TestClient
    ) -> None:
        resp = client.get(self.URL)
        for item in resp.json():
            assert "series_id" in item
            assert "description" in item
            assert "group" in item

    def test_total_count_matches_fred_series(self, client: TestClient) -> None:
        from app.services.scrapers.fred_scraper import FRED_SERIES

        resp = client.get(self.URL)
        assert len(resp.json()) == len(FRED_SERIES)

    def test_known_series_ids_present(self, client: TestClient) -> None:
        resp = client.get(self.URL)
        series_ids = {item["series_id"] for item in resp.json()}
        assert "T10Y2Y" in series_ids
        assert "VIXCLS" in series_ids

    def test_spread_series_group(self, client: TestClient) -> None:
        from app.services.scrapers.fred_scraper import FRED_SPREAD_SERIES

        resp = client.get(self.URL)
        by_id = {item["series_id"]: item for item in resp.json()}
        for sid in FRED_SPREAD_SERIES:
            assert by_id[sid]["group"] == "spreads"

    def test_volatility_series_group(self, client: TestClient) -> None:
        from app.services.scrapers.fred_scraper import FRED_VOLATILITY_SERIES

        resp = client.get(self.URL)
        by_id = {item["series_id"]: item for item in resp.json()}
        for sid in FRED_VOLATILITY_SERIES:
            assert by_id[sid]["group"] == "volatility"

    def test_cli_series_group(self, client: TestClient) -> None:
        from app.services.scrapers.fred_scraper import FRED_CLI_SERIES

        resp = client.get(self.URL)
        by_id = {item["series_id"]: item for item in resp.json()}
        for sid in FRED_CLI_SERIES:
            assert by_id[sid]["group"] == "cli"

    def test_recession_series_group(self, client: TestClient) -> None:
        from app.services.scrapers.fred_scraper import FRED_RECESSION_SERIES

        resp = client.get(self.URL)
        by_id = {item["series_id"]: item for item in resp.json()}
        for sid in FRED_RECESSION_SERIES:
            assert by_id[sid]["group"] == "recession"


# ===========================================================================
# GET /api/v1/macro-data/countries  -- distinct country list
# ===========================================================================


class TestGetDistinctCountries:
    URL = f"{BASE_URL}/countries"

    def test_returns_200(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo(get_distinct_countries=["UK", "USA"])
        with _patch_repo(mock_repo):
            resp = client.get(self.URL)

        assert resp.status_code == 200

    def test_returns_list_of_strings(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo(get_distinct_countries=["Germany", "USA"])
        with _patch_repo(mock_repo):
            resp = client.get(self.URL)

        body = resp.json()
        assert isinstance(body, list)
        assert all(isinstance(c, str) for c in body)

    def test_returns_expected_countries(self, client: TestClient) -> None:
        expected = ["France", "Germany", "UK", "USA"]
        mock_repo = _make_mock_repo(get_distinct_countries=expected)
        with _patch_repo(mock_repo):
            resp = client.get(self.URL)

        assert resp.json() == expected

    def test_empty_when_no_data(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo(get_distinct_countries=[])
        with _patch_repo(mock_repo):
            resp = client.get(self.URL)

        assert resp.json() == []

    def test_repo_method_called(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(self.URL)

        mock_repo.get_distinct_countries.assert_called_once()


# ===========================================================================
# GET /api/v1/macro-data/economic-indicator-observations
# ===========================================================================


class TestGetEconomicIndicatorObservations:
    URL = f"{BASE_URL}/economic-indicator-observations"

    def test_no_filters_returns_all(self, client: TestClient) -> None:
        obs = [_make_econ_obs("USA"), _make_econ_obs("UK")]
        mock_repo = _make_mock_repo(get_economic_indicator_observations=obs)
        with _patch_repo(mock_repo):
            resp = client.get(self.URL)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_country_filter_forwarded(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo(
            get_economic_indicator_observations=[_make_econ_obs("USA")]
        )
        with _patch_repo(mock_repo):
            client.get(self.URL, params={"country": "USA"})
        call_kw = mock_repo.get_economic_indicator_observations.call_args.kwargs
        assert call_kw["country"] == "USA"

    def test_date_range_forwarded(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            client.get(
                self.URL,
                params={"start_date": "2024-01-01", "end_date": "2024-03-31"},
            )
        call_kw = mock_repo.get_economic_indicator_observations.call_args.kwargs
        assert call_kw["start_date"] == datetime.date(2024, 1, 1)
        assert call_kw["end_date"] == datetime.date(2024, 3, 31)

    def test_limit_slices_result(self, client: TestClient) -> None:
        obs_list = [
            _make_econ_obs("USA", datetime.date(2024, 1, i + 1)) for i in range(10)
        ]
        mock_repo = _make_mock_repo(get_economic_indicator_observations=obs_list)
        with _patch_repo(mock_repo):
            resp = client.get(self.URL, params={"limit": 3})
        assert len(resp.json()) == 3

    def test_default_limit_is_500(self, client: TestClient) -> None:
        obs_list = [_make_econ_obs("USA") for _ in range(5)]
        mock_repo = _make_mock_repo(get_economic_indicator_observations=obs_list)
        with _patch_repo(mock_repo):
            resp = client.get(self.URL)
        assert len(resp.json()) == 5

    def test_no_columns_returns_all_forecast_fields(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo(
            get_economic_indicator_observations=[_make_econ_obs()]
        )
        with _patch_repo(mock_repo):
            resp = client.get(self.URL)
        item = resp.json()[0]
        for col in (
            "last_inflation",
            "inflation_6m",
            "inflation_10y_avg",
            "gdp_growth_6m",
            "earnings_12m",
            "eps_expected_12m",
            "peg_ratio",
            "lt_rate_forecast",
        ):
            assert col in item

    def test_columns_filter_single_column(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo(
            get_economic_indicator_observations=[_make_econ_obs()]
        )
        with _patch_repo(mock_repo):
            resp = client.get(self.URL, params={"columns": "gdp_growth_6m"})
        assert resp.status_code == 200
        item = resp.json()[0]
        assert "gdp_growth_6m" in item
        assert item["gdp_growth_6m"] == pytest.approx(1.4)
        assert "country" in item
        assert "date" in item

    def test_columns_filter_multiple_columns(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo(
            get_economic_indicator_observations=[_make_econ_obs()]
        )
        with _patch_repo(mock_repo):
            resp = client.get(
                self.URL,
                params={"columns": ["gdp_growth_6m", "last_inflation"]},
            )
        assert resp.status_code == 200
        item = resp.json()[0]
        assert "gdp_growth_6m" in item
        assert "last_inflation" in item

    def test_invalid_column_returns_422(self, client: TestClient) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            resp = client.get(self.URL, params={"columns": "nonexistent_col"})
        assert resp.status_code == 422
        body = resp.json()
        message = body.get("detail") or body.get("error", {}).get("message", "")
        assert "nonexistent_col" in message

    def test_mixed_valid_invalid_columns_returns_422(
        self, client: TestClient
    ) -> None:
        mock_repo = _make_mock_repo()
        with _patch_repo(mock_repo):
            resp = client.get(
                self.URL,
                params={"columns": ["gdp_growth_6m", "bad_col"]},
            )
        assert resp.status_code == 422

    def test_empty_result_with_columns_returns_empty_list(
        self, client: TestClient
    ) -> None:
        mock_repo = _make_mock_repo(get_economic_indicator_observations=[])
        with _patch_repo(mock_repo):
            resp = client.get(self.URL, params={"columns": "gdp_growth_6m"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_identity_fields_always_present_when_columns_filtered(
        self, client: TestClient
    ) -> None:
        mock_repo = _make_mock_repo(
            get_economic_indicator_observations=[_make_econ_obs()]
        )
        with _patch_repo(mock_repo):
            resp = client.get(self.URL, params={"columns": "peg_ratio"})
        item = resp.json()[0]
        for field in ("id", "country", "date", "reference_date"):
            assert field in item, f"Identity field '{field}' missing"

    def test_response_contains_correct_values(self, client: TestClient) -> None:
        obs = _make_econ_obs("USA", datetime.date(2024, 3, 1))
        mock_repo = _make_mock_repo(get_economic_indicator_observations=[obs])
        with _patch_repo(mock_repo):
            resp = client.get(self.URL)
        item = resp.json()[0]
        assert item["country"] == "USA"
        assert item["date"] == "2024-03-01"
        assert item["last_inflation"] == pytest.approx(3.1)
