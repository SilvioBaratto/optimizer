"""Unit tests for app.services.factors.factor_compute_service — gap-fill paths.

Existing tests (tests/unit/test_factor_compute_service.py) cover:
  - run_factor_compute happy path, upsert_called, empty factor_df → 0, progress.

This file adds:
  - _load_fundamentals: no instruments found → empty DataFrame
  - _load_fundamentals: instruments found but no financial statements → empty DataFrame
  - _load_prices: no instruments found → empty DataFrame
  - _build_score_rows: NaN raw_score rows are filtered out (no dict entry)
  - _build_score_rows: empty factor_df → empty list
  - _resolve_group: unknown factor column → returns "unknown"
  - _resolve_group: known factor column → returns non-empty string (not "unknown")

No DB, no real optimizer computation.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_MODULE = "app.services.factors.factor_compute_service"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> MagicMock:
    return MagicMock()


def _make_instrument(instrument_id: str = "uuid-001") -> MagicMock:
    inst = MagicMock()
    inst.id = instrument_id
    return inst


# ---------------------------------------------------------------------------
# _load_fundamentals
# ---------------------------------------------------------------------------


class TestLoadFundamentals:
    def test_when_no_instrument_found_then_returns_empty_dataframe(self) -> None:
        from app.services.factors.factor_compute_service import _load_fundamentals
        from optimizer.factors import PublicationLagConfig

        session = _make_session()

        with patch(
            f"{_MODULE}._find_instrument",
            return_value=None,
        ):
            result = _load_fundamentals(
                session,
                tickers=["AAPL", "MSFT"],
                as_of_date=datetime.date(2024, 1, 1),
                lag_config=PublicationLagConfig(),
            )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_no_financial_statements_then_returns_empty_dataframe(self) -> None:
        from app.services.factors.factor_compute_service import _load_fundamentals
        from optimizer.factors import PublicationLagConfig

        session = _make_session()
        yf_repo = MagicMock()
        yf_repo.get_financial_statements.return_value = []

        with (
            patch(f"{_MODULE}._find_instrument", return_value=_make_instrument()),
            patch(f"{_MODULE}.YFinanceRepository", return_value=yf_repo),
        ):
            result = _load_fundamentals(
                session,
                tickers=["AAPL"],
                as_of_date=datetime.date(2024, 1, 1),
                lag_config=PublicationLagConfig(),
            )

        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# _load_prices
# ---------------------------------------------------------------------------


class TestLoadPrices:
    def test_when_no_instrument_found_then_returns_empty_dataframe(self) -> None:
        from app.services.factors.factor_compute_service import _load_prices

        session = _make_session()

        with patch(f"{_MODULE}._find_instrument", return_value=None):
            result = _load_prices(
                session,
                tickers=["AAPL", "MSFT"],
                start_date=datetime.date(2023, 1, 1),
                end_date=datetime.date(2024, 1, 1),
            )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_price_history_empty_then_returns_empty_dataframe(self) -> None:
        from app.services.factors.factor_compute_service import _load_prices

        session = _make_session()
        yf_repo = MagicMock()
        yf_repo.get_price_history.return_value = []

        with (
            patch(f"{_MODULE}._find_instrument", return_value=_make_instrument()),
            patch(f"{_MODULE}.YFinanceRepository", return_value=yf_repo),
        ):
            result = _load_prices(
                session,
                tickers=["AAPL"],
                start_date=datetime.date(2023, 1, 1),
                end_date=datetime.date(2024, 1, 1),
            )

        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# _build_score_rows
# ---------------------------------------------------------------------------


class TestBuildScoreRows:
    def test_when_empty_factor_df_then_returns_empty_list(self) -> None:
        from app.services.factors.factor_compute_service import _build_score_rows

        result = _build_score_rows(pd.DataFrame(), as_of_date=datetime.date(2024, 1, 1))

        assert result == []

    def test_when_nan_raw_score_then_row_is_excluded(self) -> None:
        from app.services.factors.factor_compute_service import _build_score_rows

        factor_df = pd.DataFrame(
            {"book_to_price": [float("nan"), 0.5]},
            index=["AAPL", "MSFT"],
        )
        rows = _build_score_rows(factor_df, as_of_date=datetime.date(2024, 1, 1))

        tickers_in_rows = [r["ticker"] for r in rows]
        assert "AAPL" not in tickers_in_rows
        assert "MSFT" in tickers_in_rows

    def test_when_valid_scores_then_row_contains_expected_keys(self) -> None:
        from app.services.factors.factor_compute_service import _build_score_rows

        factor_df = pd.DataFrame(
            {"book_to_price": [0.5]},
            index=["AAPL"],
        )
        rows = _build_score_rows(factor_df, as_of_date=datetime.date(2024, 1, 1))

        assert len(rows) == 1
        row = rows[0]
        assert row["ticker"] == "AAPL"
        assert row["score_date"] == datetime.date(2024, 1, 1)
        assert row["raw_score"] == pytest.approx(0.5)
        assert row["standardized_score"] is None
        assert row["composite_score"] is None


# ---------------------------------------------------------------------------
# _resolve_group
# ---------------------------------------------------------------------------


class TestResolveGroup:
    def test_when_unknown_factor_column_then_returns_unknown(self) -> None:
        from app.services.factors.factor_compute_service import _resolve_group

        result = _resolve_group("not_a_real_factor", {})

        assert result == "unknown"

    def test_when_key_error_in_mapping_then_returns_unknown(self) -> None:
        from app.services.factors.factor_compute_service import _resolve_group

        # A valid FactorType key but deliberately missing from the mapping dict
        from optimizer.factors._config import FactorType

        # Pick any known FactorType and pass empty mapping
        any_factor = next(iter(FactorType))
        result = _resolve_group(any_factor.value, {})

        assert result == "unknown"

    def test_when_known_factor_and_full_mapping_then_returns_non_unknown(self) -> None:
        """With the real FACTOR_GROUP_MAPPING, a known factor resolves."""
        from app.services.factors.factor_compute_service import _resolve_group
        from optimizer.factors._config import FACTOR_GROUP_MAPPING, FactorType

        any_factor = next(iter(FactorType))
        result = _resolve_group(any_factor.value, FACTOR_GROUP_MAPPING)

        assert result != "unknown"
        assert isinstance(result, str)
        assert len(result) > 0
