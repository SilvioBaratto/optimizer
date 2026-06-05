"""Empty-data and exception-path tests for broker_sync_service (issue #825).

Covers paths NOT already tested in tests/unit/test_broker_sync_service.py:
  - empty positions list (positions_synced == 0, no errors)
  - position-sync exception captured in result.errors
  - account-sync exception captured in result.errors
  - order-fetch exception captured in result.errors
  - no errors → result.errors is None
  - on_progress callback receives expected step names
  - result is json-safe (BrokerSyncResult as dict)
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from app.services.portfolio.broker_sync_service import BrokerSyncResult, sync_portfolio
from tests._fixtures.assertions import assert_json_safe

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _empty_client() -> MagicMock:
    """T212 client that returns empty collections everywhere."""
    client = MagicMock()
    client.get_portfolio_positions.return_value = []
    client.get_account_cash.return_value = {"total": 0.0, "free": 0.0, "invested": 0.0}
    client.get_account_info.return_value = {"currencyCode": "EUR"}
    client.get_all_order_history.return_value = []
    client.get_all_dividend_history.return_value = []
    return client


def _make_repo() -> MagicMock:
    repo = MagicMock()
    repo.upsert_positions.return_value = 0
    repo.delete_stale_positions.return_value = 0
    repo.add_event_idempotent.return_value = None
    return repo


def _run(
    client: MagicMock,
    repo: MagicMock | None = None,
    portfolio_id: uuid.UUID | None = None,
    on_progress=None,
) -> BrokerSyncResult:
    portfolio_id = portfolio_id or uuid.uuid4()
    session = MagicMock()
    with patch(
        "app.services.portfolio.broker_sync_service.PortfolioRepository"
    ) as MockRepo:
        MockRepo.return_value = repo or _make_repo()
        return sync_portfolio(client, portfolio_id, session, on_progress=on_progress)


# ---------------------------------------------------------------------------
# Empty-data paths
# ---------------------------------------------------------------------------


class TestWhenEmptyPositionsThenSyncedIsZero:
    def test_when_no_positions_then_positions_synced_is_zero(self) -> None:
        result = _run(_empty_client())
        assert result.positions_synced == 0

    def test_when_no_positions_then_positions_removed_is_zero(self) -> None:
        result = _run(_empty_client())
        assert result.positions_removed == 0

    def test_when_no_positions_then_errors_is_none(self) -> None:
        result = _run(_empty_client())
        assert result.errors is None

    def test_when_no_orders_then_orders_fetched_is_zero(self) -> None:
        result = _run(_empty_client())
        assert result.orders_fetched == 0

    def test_when_no_orders_then_orders_inserted_is_zero(self) -> None:
        result = _run(_empty_client())
        assert result.orders_inserted == 0

    def test_when_no_dividends_then_dividends_fetched_is_zero(self) -> None:
        result = _run(_empty_client())
        assert result.dividends_fetched == 0

    def test_when_no_dividends_then_dividends_inserted_is_zero(self) -> None:
        result = _run(_empty_client())
        assert result.dividends_inserted == 0

    def test_when_empty_sync_then_account_synced_true(self) -> None:
        result = _run(_empty_client())
        assert result.account_synced is True

    def test_when_result_converted_to_dict_then_json_safe(self) -> None:
        result = _run(_empty_client())
        payload = {
            "positions_synced": result.positions_synced,
            "positions_removed": result.positions_removed,
            "account_synced": result.account_synced,
            "orders_fetched": result.orders_fetched,
            "orders_inserted": result.orders_inserted,
            "dividends_fetched": result.dividends_fetched,
            "dividends_inserted": result.dividends_inserted,
            "errors": result.errors,
        }
        assert_json_safe(payload)


# ---------------------------------------------------------------------------
# Exception-path tests
# ---------------------------------------------------------------------------


class TestWhenPositionSyncRaisesErrorCaptured:
    def test_when_get_portfolio_positions_raises_then_error_in_result(self) -> None:
        client = _empty_client()
        client.get_portfolio_positions.side_effect = RuntimeError("connection refused")
        result = _run(client)

        assert result.errors is not None
        assert any("Position sync failed" in e for e in result.errors)

    def test_when_position_fetch_raises_then_account_still_synced(self) -> None:
        client = _empty_client()
        client.get_portfolio_positions.side_effect = OSError("network error")
        result = _run(client)

        assert result.account_synced is True

    def test_when_position_fetch_raises_then_positions_synced_zero(self) -> None:
        client = _empty_client()
        client.get_portfolio_positions.side_effect = OSError("network error")
        result = _run(client)

        assert result.positions_synced == 0


class TestWhenAccountSyncRaisesErrorCaptured:
    def test_when_get_account_cash_raises_then_error_in_result(self) -> None:
        client = _empty_client()
        client.get_account_cash.side_effect = RuntimeError("auth failed")
        result = _run(client)

        assert result.errors is not None
        assert any("Account sync failed" in e for e in result.errors)

    def test_when_account_cash_raises_then_account_synced_false(self) -> None:
        client = _empty_client()
        client.get_account_cash.side_effect = RuntimeError("timeout")
        result = _run(client)

        assert result.account_synced is False

    def test_when_account_info_raises_then_error_captured(self) -> None:
        client = _empty_client()
        client.get_account_info.side_effect = RuntimeError("rate limited")
        result = _run(client)

        assert result.errors is not None
        assert any("Account sync failed" in e for e in result.errors)


class TestWhenOrderFetchRaisesErrorCaptured:
    def test_when_get_all_order_history_raises_then_error_in_result(self) -> None:
        client = _empty_client()
        client.get_all_order_history.side_effect = RuntimeError("timeout")
        result = _run(client)

        assert result.errors is not None
        assert any("Order history fetch failed" in e for e in result.errors)

    def test_when_order_fetch_raises_then_orders_fetched_zero(self) -> None:
        client = _empty_client()
        client.get_all_order_history.side_effect = RuntimeError("timeout")
        result = _run(client)

        assert result.orders_fetched == 0
        assert result.orders_inserted == 0


# ---------------------------------------------------------------------------
# No-error path
# ---------------------------------------------------------------------------


class TestWhenNoErrorsThenErrorsFieldIsNone:
    def test_when_all_steps_succeed_then_errors_is_none(self) -> None:
        result = _run(_empty_client())
        assert result.errors is None


# ---------------------------------------------------------------------------
# on_progress callback
# ---------------------------------------------------------------------------


class TestWhenOnProgressProvidedThenCalledWithSteps:
    def test_when_on_progress_provided_then_called_four_times(self) -> None:
        progress_calls: list[dict] = []

        def capture(**kwargs):
            progress_calls.append(kwargs)

        _run(_empty_client(), on_progress=capture)
        assert len(progress_calls) == 5  # 0,1,2,3,4

    def test_when_on_progress_provided_then_step_names_correct(self) -> None:
        steps: list[str] = []

        def capture(**kwargs):
            extra = kwargs.get("extra", {})
            if "step" in extra:
                steps.append(extra["step"])

        _run(_empty_client(), on_progress=capture)
        assert "positions" in steps
        assert "account" in steps
        assert "orders" in steps
        assert "dividends" in steps
        assert "done" in steps
