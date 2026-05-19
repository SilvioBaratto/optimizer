"""End-to-end fixture-based unit tests for T212 normalizer (issue #778)."""

from __future__ import annotations

import logging

import pytest

from app.schemas.portfolio import PositionFlag
from app.services.portfolio.t212_position_normalizer import (
    normalize_live_positions,
)

from ._fixtures import (
    _FIXTURE_ACCOUNT_INFO_GBP,
    AS_OF,
    make_client,
    make_fx_provider,
    seed_instruments,
)


def _run_default(db_session, **overrides):
    """Helper: seed DB, build client, run orchestrator."""
    seed_instruments(db_session)
    client = make_client(**overrides)
    fx = make_fx_provider()
    return normalize_live_positions(
        client, db_session, as_of_date=AS_OF, fx_provider=fx
    )


def _position_by_t212(result, t212_ticker: str):
    matches = [p for p in result.positions if p.t212_ticker == t212_ticker]
    assert matches, f"position {t212_ticker} not in result"
    return matches[0]


class TestGbxLondonPosition:
    def test_gbx_london_position_normalizes_pence_to_pounds_then_to_eur(
        self, db_session
    ) -> None:
        result = _run_default(db_session)
        pos = _position_by_t212(result, "VOD_l_EQ")

        assert pos.currency == "GBP"
        assert pos.price_native == pytest.approx(75.0, rel=1e-9)
        assert pos.eur_value == pytest.approx(100.0 * 75.0 * 1.17, rel=1e-9)


class TestEurParisPosition:
    def test_eur_paris_position_skips_fx_call(self, db_session) -> None:
        seed_instruments(db_session)
        positions = [
            {
                "ticker": "AIR_p_EQ",
                "quantity": 1.0,
                "currentPrice": 150.0,
                "currencyCode": "EUR",
                "ppl": 0.0,
                "fxPpl": 0.0,
            }
        ]
        client = make_client(positions=positions, account_cash={"invested": 150.0})
        fx = make_fx_provider()

        normalize_live_positions(client, db_session, as_of_date=AS_OF, fx_provider=fx)

        assert fx.get_rate_to_base.call_count == 0


class TestParisPreferredShare:
    def test_paris_preferred_share_resolves_to_pa_suffix(self, db_session) -> None:
        result = _run_default(db_session)
        pos = _position_by_t212(result, "AIRp_pp_EQ")

        assert pos.ticker.endswith(".PA")


class TestUsdUsPosition:
    def test_usd_us_position_converts_via_fx(self, db_session) -> None:
        result = _run_default(db_session)
        pos = _position_by_t212(result, "AAPL_US_EQ")

        assert pos.currency == "USD"
        assert pos.eur_value == pytest.approx(10.0 * 180.0 * 0.92, rel=1e-9)


class TestUnmappedTicker:
    def test_unmapped_ticker_falls_back_to_suffix_resolver_and_flags_unmapped(
        self, db_session
    ) -> None:
        result = _run_default(db_session)
        pos = _position_by_t212(result, "FAKE_XX_EQ")

        assert PositionFlag.UNMAPPED in pos.flags
        assert pos.ticker != "FAKE_XX_EQ"


class TestMissingFx:
    def test_missing_fx_rate_flags_fx_missing_and_zeroes_eur_value(
        self, db_session
    ) -> None:
        result = _run_default(db_session)
        pos = _position_by_t212(result, "FAKE_XX_EQ")

        assert PositionFlag.FX_MISSING in pos.flags
        assert pos.eur_value == 0.0


class TestReconciliationWithinThreshold:
    def test_reconciliation_within_threshold_does_not_flag(self, db_session) -> None:
        """sum(eur_value) = invested * 1.005 → within 1.5% gate."""
        seed_instruments(db_session)
        positions = [
            {
                "ticker": "AIR_p_EQ",
                "quantity": 10.0,
                "currentPrice": 100.0,
                "currencyCode": "EUR",
            }
        ]
        # eur_value = 1000. invested = 1000 / 1.005 ≈ 995.02 → delta = 0.005 < 0.015
        cash = {"invested": 995.025}
        client = make_client(positions=positions, account_cash=cash)
        fx = make_fx_provider()

        result = normalize_live_positions(
            client, db_session, as_of_date=AS_OF, fx_provider=fx
        )

        assert result.reconciliation_ok is True
        assert all(
            PositionFlag.RECONCILIATION_MISMATCH not in p.flags
            for p in result.positions
        )


class TestReconciliationOutsideThreshold:
    def test_reconciliation_outside_threshold_flags_all_positions(
        self, db_session
    ) -> None:
        """sum(eur_value) = invested * 1.05 → outside 1.5% gate."""
        seed_instruments(db_session)
        positions = [
            {
                "ticker": "AIR_p_EQ",
                "quantity": 10.0,
                "currentPrice": 100.0,
                "currencyCode": "EUR",
            },
            {
                "ticker": "SAP_d_EQ",
                "quantity": 5.0,
                "currentPrice": 200.0,
                "currencyCode": "EUR",
            },
        ]
        # total = 1000 + 1000 = 2000. invested = 2000 / 1.05 ≈ 1904.76
        cash = {"invested": 1904.76}
        client = make_client(positions=positions, account_cash=cash)
        fx = make_fx_provider()

        result = normalize_live_positions(
            client, db_session, as_of_date=AS_OF, fx_provider=fx
        )

        assert result.reconciliation_ok is False
        assert all(
            PositionFlag.RECONCILIATION_MISMATCH in p.flags for p in result.positions
        )


class TestNonEurAccount:
    def test_non_eur_account_skips_reconciliation(self, db_session) -> None:
        result = _run_default(
            db_session,
            account_info=_FIXTURE_ACCOUNT_INFO_GBP,
        )

        assert result.base_currency == "GBP"
        assert result.reconciliation_ok is True
        assert result.reconciliation_delta_pct is None


class TestWeightsSumToOne:
    def test_eur_weights_sum_to_one_when_no_fx_missing(self, db_session) -> None:
        """All-EUR portfolio: weights sum to exactly 1.0."""
        seed_instruments(db_session)
        positions = [
            {
                "ticker": "AIR_p_EQ",
                "quantity": 10.0,
                "currentPrice": 100.0,
                "currencyCode": "EUR",
            },
            {
                "ticker": "SAP_d_EQ",
                "quantity": 5.0,
                "currentPrice": 200.0,
                "currencyCode": "EUR",
            },
            {
                "ticker": "ENI_m_EQ",
                "quantity": 100.0,
                "currentPrice": 15.0,
                "currencyCode": "EUR",
            },
        ]
        client = make_client(positions=positions, account_cash={"invested": 3500.0})
        fx = make_fx_provider()

        result = normalize_live_positions(
            client, db_session, as_of_date=AS_OF, fx_provider=fx
        )

        total_weight = sum(p.eur_weight for p in result.positions)
        assert total_weight == pytest.approx(1.0, abs=1e-9)


class TestEmptyPortfolio:
    def test_empty_position_list_returns_empty_result(self, db_session) -> None:
        client = make_client(positions=[], account_cash={"invested": 0.0})
        fx = make_fx_provider()

        result = normalize_live_positions(
            client, db_session, as_of_date=AS_OF, fx_provider=fx
        )

        assert result.positions == []
        assert result.reconciliation_ok is True
        assert result.reconciliation_delta_pct == 0.0


class TestCredentialsNeverLeak:
    def test_credentials_never_appear_in_logs_or_result(
        self, db_session, caplog
    ) -> None:
        seed_instruments(db_session)
        client = make_client()
        client.api_key = "SECRET_API_KEY_42"
        client.api_secret = "SECRET_API_SECRET_42"  # noqa: S105
        fx = make_fx_provider()

        with caplog.at_level(logging.DEBUG):
            result = normalize_live_positions(
                client, db_session, as_of_date=AS_OF, fx_provider=fx
            )

        full_log = " ".join(record.getMessage() for record in caplog.records)
        assert "SECRET_API_KEY_42" not in full_log
        assert "SECRET_API_SECRET_42" not in full_log
        assert "SECRET_API_KEY_42" not in repr(result)
        assert "SECRET_API_SECRET_42" not in repr(result)
