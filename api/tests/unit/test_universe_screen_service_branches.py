"""Branch-coverage tests for universe_screening_service.

Targets the missed lines:
  144  — _assemble_statements non-empty path (returns real DataFrame)
  168  — _hysteresis inner fn: req_field is not None → returns HysteresisConfig(...)
  173  — _region inner fn: value is not None → returns ExchangeRegion(value)
  243  — _run_market_cap_screens: "market_cap" not in fundamentals → early return
  257  — _run_market_cap_screens: "exchange" not in columns → mcap_percentile fallback
  279-282 — _compute_mcap_percentile_pass: group smaller than min_size → thresh = 0.0
  294-297 — _compute_mcap_percentile_pass: current_members not None → surviving branch
  308  — _run_liquidity_screens: empty close_df → early return
  330  — _run_price_screen: "current_price" not in columns → early return
  346  — _run_history_screens: empty close_df → early return
  363-371 — _run_statements_screen: statements.empty → early return
  433  — _resolve_current_members: tickers provided → returns pd.Index
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.schemas.universe.universe_screen import (
    HysteresisRequest,
    ScreenPreset,
    UniverseScreenRequest,
)
from optimizer.universe._config import InvestabilityScreenConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_returning(rows: list) -> MagicMock:
    session = MagicMock()
    result = MagicMock()
    result.all.return_value = rows
    session.execute.return_value = result
    return session


def _fundamentals_with_market_cap(*tickers: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_cap": [1e9] * len(tickers),
            "current_price": [100.0] * len(tickers),
            "exchange": ["NYSE"] * len(tickers),
        },
        index=pd.Index(list(tickers)),
    )


def _fundamentals_without_column(col: str, *tickers: str) -> pd.DataFrame:
    df = _fundamentals_with_market_cap(*tickers)
    return df.drop(columns=[col])


# ---------------------------------------------------------------------------
# L144 — _assemble_statements non-empty path
# ---------------------------------------------------------------------------


class TestAssembleStatementsNonEmpty:
    """When the DB returns rows, _assemble_statements must return a real DataFrame."""

    def test_non_empty_rows_returns_dataframe(self) -> None:
        from app.services.universe.universe_screening_service import (
            _assemble_statements,
        )

        rows = [("AAPL", "annual", "2023-12-31"), ("AAPL", "quarterly", "2024-03-31")]
        session = _session_returning(rows)

        df = _assemble_statements(session)

        assert not df.empty
        assert list(df.columns) == ["ticker", "period_type", "period_date"]
        assert len(df) == 2


# ---------------------------------------------------------------------------
# L168/L173 — _apply_overrides inner helpers (_hysteresis and _region)
# ---------------------------------------------------------------------------


class TestApplyOverridesInnerHelpers:
    """_apply_overrides must thread non-None override fields through correctly."""

    def test_hysteresis_override_applied_when_req_field_not_none(self) -> None:
        """When request.market_cap is set, the returned config uses those values."""
        from app.services.universe.universe_screening_service import _build_config

        request = UniverseScreenRequest(
            market_cap=HysteresisRequest(entry=2e9, exit=1e9)
        )
        config = _build_config(request)

        assert config.market_cap.entry == pytest.approx(2e9)
        assert config.market_cap.exit_ == pytest.approx(1e9)

    def test_exchange_region_override_applied_when_not_none(self) -> None:
        """When exchange_region is set, _region returns ExchangeRegion(value)."""
        from app.services.universe.universe_screening_service import _build_config

        request = UniverseScreenRequest(exchange_region="europe")
        config = _build_config(request)

        assert config.exchange_region.value == "europe"


# ---------------------------------------------------------------------------
# L243 — _run_market_cap_screens: "market_cap" column absent → early return
# ---------------------------------------------------------------------------


class TestRunMarketCapScreensNoColumn:
    """When fundamentals lacks 'market_cap', the screen is silently skipped."""

    def test_missing_market_cap_column_leaves_result_unchanged(self) -> None:
        from app.services.universe.universe_screening_service import (
            _SCREEN_NAMES,
            _run_market_cap_screens,
        )

        fundamentals = pd.DataFrame(
            {"current_price": [50.0, 60.0]}, index=pd.Index(["AAPL", "MSFT"])
        )
        config = InvestabilityScreenConfig()
        results: dict[str, set[str]] = {name: set() for name in _SCREEN_NAMES}
        original_market_cap = set(results["market_cap"])

        _run_market_cap_screens(fundamentals, config, None, results)

        # Both sets stay empty — function returned early without writing
        assert results["market_cap"] == original_market_cap
        assert results["mcap_percentile"] == set()


# ---------------------------------------------------------------------------
# L257 — _run_market_cap_screens: "exchange" column absent → percentile fallback
# ---------------------------------------------------------------------------


class TestRunMarketCapScreensNoExchange:
    """When 'exchange' is absent from fundamentals, mcap_percentile is set to all tickers."""

    def test_missing_exchange_column_sets_all_tickers_as_passing(self) -> None:
        from app.services.universe.universe_screening_service import (
            _SCREEN_NAMES,
            _run_market_cap_screens,
        )

        fundamentals = pd.DataFrame(
            {"market_cap": [1e9, 2e9]}, index=pd.Index(["AAPL", "MSFT"])
        )
        config = InvestabilityScreenConfig()
        results: dict[str, set[str]] = {name: set() for name in _SCREEN_NAMES}

        _run_market_cap_screens(fundamentals, config, None, results)

        # mcap_percentile fallback: all tickers in the index pass
        assert "AAPL" in results["mcap_percentile"]
        assert "MSFT" in results["mcap_percentile"]


# ---------------------------------------------------------------------------
# L279-282 — _compute_mcap_percentile_pass: small group → thresh = 0.0
# ---------------------------------------------------------------------------


class TestComputeMcapPercentileSmallGroup:
    """Groups smaller than min_size (10) receive zero entry/exit thresholds."""

    def test_small_group_gets_zero_threshold_so_all_pass(self) -> None:
        from app.services.universe.universe_screening_service import (
            _compute_mcap_percentile_pass,
        )

        # Only 3 tickers in one exchange — below min_size of 10
        mcap = pd.Series(
            {"AAPL": 1e9, "MSFT": 2e9, "GOOG": 3e9},
        )
        exchange_mapping = pd.Series(
            {"AAPL": "NYSE", "MSFT": "NYSE", "GOOG": "NYSE"},
        )
        config = InvestabilityScreenConfig()

        passing = _compute_mcap_percentile_pass(mcap, exchange_mapping, config, None)

        # With threshold = 0.0, all tickers whose mcap >= 0 pass
        assert set(passing.tolist()) == {"AAPL", "MSFT", "GOOG"}


# ---------------------------------------------------------------------------
# L294-297 — _compute_mcap_percentile_pass: current_members not None → surviving
# ---------------------------------------------------------------------------


class TestComputeMcapPercentileWithCurrentMembers:
    """When current_members is provided, surviving members using exit threshold are kept."""

    def test_current_member_above_exit_thresh_survives(self) -> None:
        from app.services.universe.universe_screening_service import (
            _compute_mcap_percentile_pass,
        )

        # Build 12 tickers so group size >= 10
        tickers = [f"T{i:02d}" for i in range(12)]
        mcaps = {t: float(1_000_000_000 * (i + 1)) for i, t in enumerate(tickers)}
        mcap = pd.Series(mcaps)
        exchange_mapping = pd.Series(dict.fromkeys(tickers, "NYSE"))
        current_members = pd.Index(tickers[:3])  # first 3 are current members
        config = InvestabilityScreenConfig()

        passing = _compute_mcap_percentile_pass(
            mcap, exchange_mapping, config, current_members
        )

        # Surviving members with mcap >= exit threshold should remain
        assert isinstance(passing, pd.Index)
        assert len(passing) > 0


# ---------------------------------------------------------------------------
# L308 — _run_liquidity_screens: empty close_df → early return
# ---------------------------------------------------------------------------


class TestRunLiquidityScreensEmptyData:
    """When price/volume DataFrames are empty, liquidity screen results stay empty."""

    def test_empty_close_df_skips_liquidity_screens(self) -> None:
        from app.services.universe.universe_screening_service import (
            _SCREEN_NAMES,
            _run_liquidity_screens,
        )

        config = InvestabilityScreenConfig()
        results: dict[str, set[str]] = {name: set() for name in _SCREEN_NAMES}

        _run_liquidity_screens(pd.DataFrame(), pd.DataFrame(), config, None, results)

        assert results["addv_12m"] == set()
        assert results["addv_3m"] == set()
        assert results["trading_frequency"] == set()


# ---------------------------------------------------------------------------
# L330 — _run_price_screen: "current_price" column absent → early return
# ---------------------------------------------------------------------------


class TestRunPriceScreenMissingColumn:
    """When fundamentals lacks 'current_price', price screen is silently skipped."""

    def test_missing_current_price_column_leaves_price_result_empty(self) -> None:
        from app.services.universe.universe_screening_service import (
            _SCREEN_NAMES,
            _run_price_screen,
        )

        fundamentals = pd.DataFrame(
            {"market_cap": [1e9]}, index=pd.Index(["AAPL"])
        )
        config = InvestabilityScreenConfig()
        results: dict[str, set[str]] = {name: set() for name in _SCREEN_NAMES}

        _run_price_screen(fundamentals, config, None, results)

        assert results["price"] == set()


# ---------------------------------------------------------------------------
# L346 — _run_history_screens: empty close_df → early return
# ---------------------------------------------------------------------------


class TestRunHistoryScreensEmptyData:
    """When close_df is empty, history/IPO screens stay empty."""

    def test_empty_close_df_skips_history_screens(self) -> None:
        from app.services.universe.universe_screening_service import (
            _SCREEN_NAMES,
            _run_history_screens,
        )

        config = InvestabilityScreenConfig()
        results: dict[str, set[str]] = {name: set() for name in _SCREEN_NAMES}

        _run_history_screens(pd.DataFrame(), config, results)

        assert results["trading_history"] == set()
        assert results["ipo_seasoning"] == set()


# ---------------------------------------------------------------------------
# L363-371 — _run_statements_screen: statements.empty → early return
# ---------------------------------------------------------------------------


class TestRunStatementsScreenEmpty:
    """When statements DataFrame is empty, financial_statements result stays empty."""

    def test_empty_statements_skips_financial_statements_screen(self) -> None:
        from app.services.universe.universe_screening_service import (
            _SCREEN_NAMES,
            _run_statements_screen,
        )

        config = InvestabilityScreenConfig()
        results: dict[str, set[str]] = {name: set() for name in _SCREEN_NAMES}

        _run_statements_screen(pd.DataFrame(), config, results)

        assert results["financial_statements"] == set()

    def test_non_empty_statements_populates_screen_result(self) -> None:
        """Complement: non-empty statements must populate financial_statements."""
        from app.services.universe.universe_screening_service import (
            _SCREEN_NAMES,
            _run_statements_screen,
        )

        statements = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 5,
                "period_type": ["annual"] * 5,
                "period_date": pd.date_range("2019-01-01", periods=5, freq="YE"),
            }
        )
        config = InvestabilityScreenConfig(min_annual_reports=2)
        results: dict[str, set[str]] = {name: set() for name in _SCREEN_NAMES}

        _run_statements_screen(statements, config, results)

        assert "AAPL" in results["financial_statements"]


# ---------------------------------------------------------------------------
# L433 — _resolve_current_members: tickers provided → returns pd.Index
# ---------------------------------------------------------------------------


class TestResolveCurrentMembers:
    """_resolve_current_members converts a non-empty list to pd.Index."""

    def test_none_returns_none(self) -> None:
        from app.services.universe.universe_screening_service import (
            _resolve_current_members,
        )

        assert _resolve_current_members(None) is None

    def test_empty_list_returns_none(self) -> None:
        from app.services.universe.universe_screening_service import (
            _resolve_current_members,
        )

        assert _resolve_current_members([]) is None

    def test_non_empty_list_returns_index(self) -> None:
        from app.services.universe.universe_screening_service import (
            _resolve_current_members,
        )

        result = _resolve_current_members(["AAPL", "MSFT"])

        assert isinstance(result, pd.Index)
        assert "AAPL" in result
        assert "MSFT" in result


# ---------------------------------------------------------------------------
# run_universe_screen — preset path (uses _PRESET_FACTORIES)
# ---------------------------------------------------------------------------


class TestRunUniverseScreenWithPreset:
    """Using a preset must return a non-error response dict."""

    def test_preset_developed_markets_with_empty_db_returns_empty_result(self) -> None:
        from app.services.universe.universe_screening_service import run_universe_screen

        session = _session_returning([])  # empty DB
        request = UniverseScreenRequest(preset=ScreenPreset.DEVELOPED_MARKETS)

        result = run_universe_screen(request, session)

        assert result == {
            "passing_tickers": [],
            "total_screened": 0,
            "diagnostics": {},
        }
