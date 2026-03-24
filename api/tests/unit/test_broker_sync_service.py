"""Unit tests for sync_portfolio() — broker order deduplication (issue #310)."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.services.broker_sync_service import BrokerSyncResult, sync_portfolio


def _make_order(
    order_id: str,
    status: str = "FILLED",
    order_type: str = "BUY",
    ticker: str = "AAPL_US_EQ",
) -> dict:
    return {
        "id": order_id,
        "status": status,
        "type": order_type,
        "ticker": ticker,
        "filledQuantity": 5.0,
        "fillPrice": 180.0,
    }


def _make_t212_client(orders: list[dict]) -> MagicMock:
    client = MagicMock()
    client.get_portfolio_positions.return_value = []
    client.get_account_cash.return_value = {"total": 0, "free": 0, "invested": 0}
    client.get_account_info.return_value = {"currencyCode": "EUR"}
    client.get_all_order_history.return_value = orders
    client.get_all_dividend_history.return_value = []
    return client


def _make_repo_mock(inserted_return: object = "sentinel") -> MagicMock:
    repo = MagicMock()
    repo.upsert_positions.return_value = 0
    repo.delete_stale_positions.return_value = 0
    if inserted_return == "sentinel":
        repo.add_event_idempotent.return_value = MagicMock()  # non-None = inserted
    else:
        repo.add_event_idempotent.return_value = inserted_return
    return repo


def _run_sync(
    orders: list[dict],
    portfolio_id: uuid.UUID | None = None,
    repo_mock: MagicMock | None = None,
) -> tuple[BrokerSyncResult, MagicMock]:
    portfolio_id = portfolio_id or uuid.uuid4()
    session = MagicMock()
    client = _make_t212_client(orders)

    with patch("app.services.broker_sync_service.PortfolioRepository") as MockRepo:
        repo = repo_mock or _make_repo_mock()
        MockRepo.return_value = repo
        result = sync_portfolio(client, portfolio_id, session)

    return result, repo


class TestSyncPortfolioDeduplication:
    def test_calls_add_event_idempotent_with_external_id(self) -> None:
        portfolio_id = uuid.uuid4()
        result, repo = _run_sync([_make_order("order-001")], portfolio_id)

        repo.add_event_idempotent.assert_called_once_with(
            event_type="trade",
            title="Bought 5.0 AAPL_US_EQ",
            portfolio_id=portfolio_id,
            description="Fill price: 180.0",
            external_id="order-001",
            metadata={"order_id": "order-001", "ticker": "AAPL_US_EQ"},
        )

    def test_orders_inserted_incremented_on_new_event(self) -> None:
        result, _ = _run_sync([_make_order("order-001")])
        assert result.orders_inserted == 1

    def test_duplicate_order_not_counted_in_orders_inserted(self) -> None:
        """When add_event_idempotent returns None (conflict), orders_inserted stays 0."""
        repo = _make_repo_mock(inserted_return=None)
        result, _ = _run_sync([_make_order("order-999")], repo_mock=repo)

        assert result.orders_inserted == 0
        assert result.orders_fetched == 1

    def test_orders_fetched_counts_all_regardless_of_dedup(self) -> None:
        orders = [_make_order("order-001"), _make_order("order-002")]
        result, _ = _run_sync(orders)
        assert result.orders_fetched == 2

    def test_only_filled_orders_passed_to_add_event_idempotent(self) -> None:
        orders = [
            _make_order("order-001", status="FILLED"),
            _make_order("order-002", status="CANCELLED"),
            _make_order("order-003", status="PENDING"),
        ]
        result, repo = _run_sync(orders)
        assert repo.add_event_idempotent.call_count == 1
        assert result.orders_fetched == 3

    def test_sell_order_produces_sold_title(self) -> None:
        result, repo = _run_sync([_make_order("order-sell-1", order_type="SELL")])
        call_kwargs = repo.add_event_idempotent.call_args.kwargs
        assert call_kwargs["title"].startswith("Sold")

    def test_buy_order_produces_bought_title(self) -> None:
        result, repo = _run_sync([_make_order("order-buy-1", order_type="BUY")])
        call_kwargs = repo.add_event_idempotent.call_args.kwargs
        assert call_kwargs["title"].startswith("Bought")

    def test_order_without_id_uses_none_external_id(self) -> None:
        """Orders with no id field pass external_id=None (non-broker fallback)."""
        order = {"status": "FILLED", "type": "BUY", "ticker": "TSLA", "filledQuantity": 1}
        _, repo = _run_sync([order])
        call_kwargs = repo.add_event_idempotent.call_args.kwargs
        assert call_kwargs["external_id"] is None

    def test_multiple_new_orders_all_counted(self) -> None:
        orders = [_make_order(f"order-{i}") for i in range(5)]
        result, _ = _run_sync(orders)
        assert result.orders_inserted == 5

    def test_mixed_new_and_duplicate_orders(self) -> None:
        """Two orders: first inserts (returns event), second is duplicate (returns None)."""
        orders = [_make_order("new-001"), _make_order("dup-002")]
        repo = _make_repo_mock()
        # First call inserts, second is a duplicate
        repo.add_event_idempotent.side_effect = [MagicMock(), None]

        result, _ = _run_sync(orders, repo_mock=repo)
        assert result.orders_inserted == 1
        assert result.orders_fetched == 2


# ---------------------------------------------------------------------------
# Dividend persistence helpers
# ---------------------------------------------------------------------------


def _make_dividend(
    reference: str,
    ticker: str = "AAPL_US_EQ",
    amount: float = 1.25,
    paid_on: str = "2024-06-15T10:00:00Z",
) -> dict:
    return {
        "reference": reference,
        "ticker": ticker,
        "quantity": 10.0,
        "amount": amount,
        "amountInEuro": amount * 0.93,
        "type": "ORDINARY",
        "appointedDate": "2024-06-01",
        "paidOn": paid_on,
    }


def _make_t212_client_with_dividends(dividends: list[dict]) -> MagicMock:
    client = MagicMock()
    client.get_portfolio_positions.return_value = []
    client.get_account_cash.return_value = {"total": 0, "free": 0, "invested": 0}
    client.get_account_info.return_value = {"currencyCode": "EUR"}
    client.get_all_order_history.return_value = []
    client.get_all_dividend_history.return_value = dividends
    return client


def _run_sync_dividends(
    dividends: list[dict],
    portfolio_id: uuid.UUID | None = None,
    repo_mock: MagicMock | None = None,
) -> tuple[BrokerSyncResult, MagicMock]:
    portfolio_id = portfolio_id or uuid.uuid4()
    session = MagicMock()
    client = _make_t212_client_with_dividends(dividends)

    with patch("app.services.broker_sync_service.PortfolioRepository") as MockRepo:
        repo = repo_mock or _make_repo_mock()
        MockRepo.return_value = repo
        result = sync_portfolio(client, portfolio_id, session)

    return result, repo


class TestSyncPortfolioDividends:
    def test_calls_add_event_idempotent_with_reference_as_external_id(self) -> None:
        portfolio_id = uuid.uuid4()
        result, repo = _run_sync_dividends(
            [_make_dividend("div-ref-001")], portfolio_id,
        )

        repo.add_event_idempotent.assert_called_once_with(
            event_type="dividend",
            title="Dividend 1.25 AAPL_US_EQ",
            portfolio_id=portfolio_id,
            description="Paid: 2024-06-15T10:00:00Z",
            external_id="div-ref-001",
            metadata={
                "reference": "div-ref-001",
                "ticker": "AAPL_US_EQ",
                "amount": 1.25,
                "amount_in_euro": 1.25 * 0.93,
                "quantity": 10.0,
                "type": "ORDINARY",
                "appointed_date": "2024-06-01",
                "paid_on": "2024-06-15T10:00:00Z",
            },
        )

    def test_dividends_inserted_incremented_on_new_event(self) -> None:
        result, _ = _run_sync_dividends([_make_dividend("div-ref-001")])
        assert result.dividends_inserted == 1

    def test_duplicate_dividend_not_counted_in_dividends_inserted(self) -> None:
        """When add_event_idempotent returns None (conflict), dividends_inserted stays 0."""
        repo = _make_repo_mock(inserted_return=None)
        result, _ = _run_sync_dividends([_make_dividend("div-ref-dup")], repo_mock=repo)

        assert result.dividends_inserted == 0
        assert result.dividends_fetched == 1

    def test_dividends_fetched_counts_all_regardless_of_dedup(self) -> None:
        dividends = [_make_dividend("div-001"), _make_dividend("div-002")]
        result, _ = _run_sync_dividends(dividends)
        assert result.dividends_fetched == 2

    def test_multiple_new_dividends_all_counted(self) -> None:
        dividends = [_make_dividend(f"div-{i}") for i in range(5)]
        result, _ = _run_sync_dividends(dividends)
        assert result.dividends_inserted == 5

    def test_mixed_new_and_duplicate_dividends(self) -> None:
        """First dividend inserts, second is a duplicate."""
        dividends = [_make_dividend("div-new"), _make_dividend("div-dup")]
        repo = _make_repo_mock()
        repo.add_event_idempotent.side_effect = [MagicMock(), None]

        result, _ = _run_sync_dividends(dividends, repo_mock=repo)
        assert result.dividends_inserted == 1
        assert result.dividends_fetched == 2

    def test_only_latest_20_dividends_processed(self) -> None:
        """More than 20 dividends in history — only the first 20 are processed."""
        dividends = [_make_dividend(f"div-{i}") for i in range(25)]
        result, repo = _run_sync_dividends(dividends)

        assert result.dividends_fetched == 25
        assert repo.add_event_idempotent.call_count == 20

    def test_dividend_without_reference_uses_none_external_id(self) -> None:
        """Dividends with no reference field pass external_id=None (no dedup)."""
        div = {
            "ticker": "MSFT_US_EQ",
            "amount": 0.75,
            "amountInEuro": 0.70,
            "quantity": 5.0,
            "type": "ORDINARY",
            "appointedDate": "2024-05-01",
            "paidOn": "2024-05-15T10:00:00Z",
        }
        _, repo = _run_sync_dividends([div])
        call_kwargs = repo.add_event_idempotent.call_args.kwargs
        assert call_kwargs["external_id"] is None

    def test_dividend_missing_amount_is_skipped(self) -> None:
        """Dividends with no amount key are silently skipped (malformed payload)."""
        div = {"reference": "div-no-amount", "ticker": "MSFT_US_EQ"}
        result, repo = _run_sync_dividends([div])

        repo.add_event_idempotent.assert_not_called()
        assert result.dividends_fetched == 1
        assert result.dividends_inserted == 0

    def test_dividend_with_zero_amount_is_persisted(self) -> None:
        """A zero amount is a valid event (e.g. stock dividend pending settlement)."""
        div = _make_dividend("div-zero", amount=0.0)
        result, repo = _run_sync_dividends([div])

        assert repo.add_event_idempotent.call_count == 1
        call_kwargs = repo.add_event_idempotent.call_args.kwargs
        assert "Dividend 0.0" in call_kwargs["title"]

    def test_dividend_missing_paid_on_produces_none_description(self) -> None:
        """Dividends with no paidOn field produce description=None."""
        div = {
            "reference": "div-no-date",
            "ticker": "AAPL_US_EQ",
            "amount": 1.0,
        }
        _, repo = _run_sync_dividends([div])
        call_kwargs = repo.add_event_idempotent.call_args.kwargs
        assert call_kwargs["description"] is None

    def test_dividend_fetch_error_appended_to_errors(self) -> None:
        """A failing get_all_dividend_history is captured in result.errors."""
        session = MagicMock()
        client = _make_t212_client_with_dividends([])
        client.get_all_dividend_history.side_effect = RuntimeError("timeout")

        with patch("app.services.broker_sync_service.PortfolioRepository") as MockRepo:
            MockRepo.return_value = _make_repo_mock()
            result = sync_portfolio(client, uuid.uuid4(), session)

        assert result.errors is not None
        assert any("Dividend history fetch failed" in e for e in result.errors)
        assert result.dividends_fetched == 0
        assert result.dividends_inserted == 0

    def test_dividends_inserted_field_present_on_result(self) -> None:
        """Smoke test: BrokerSyncResult exposes dividends_inserted with default 0."""
        r = BrokerSyncResult()
        assert r.dividends_inserted == 0
