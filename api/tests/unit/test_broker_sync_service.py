from types import SimpleNamespace

from app.services import broker_sync_service


class DummyClient:
    def get_portfolio_positions(self):
        return []

    def get_account_cash(self):
        return {"total": 100, "free": 50, "invested": 50}

    def get_account_info(self):
        return {"currencyCode": "USD"}

    def get_all_order_history(self):
        return []

    def get_all_dividend_history(self):
        return [
            {
                "id": "div-1",
                "ticker": "AAPL",
                "amount": 1.23,
                "currencyCode": "USD",
                "exDividendDate": "2026-03-01",
            }
        ]


class DummyRepo:
    instances = []

    def __init__(self, session):
        self.events = []
        DummyRepo.instances.append(self)

    def upsert_positions(self, portfolio_id, position_rows, synced_at):
        return 0

    def delete_stale_positions(self, portfolio_id, current_tickers):
        return 0

    def upsert_account_snapshot(self, portfolio_id, cash_data, synced_at):
        return None

    def add_event(self, **kwargs):
        self.events.append(kwargs)


class DummySession:
    def commit(self):
        return None


def test_sync_portfolio_persists_dividend_events(monkeypatch):
    monkeypatch.setattr(broker_sync_service, "PortfolioRepository", DummyRepo)
    monkeypatch.setattr(
        broker_sync_service,
        "_map_t212_ticker_to_yfinance",
        lambda ticker, session: None,
    )

    DummyRepo.instances.clear()

    result = broker_sync_service.sync_portfolio(
        client=DummyClient(),
        portfolio_id="portfolio-1",
        session=DummySession(),
    )

    repo = DummyRepo.instances[0]

    assert result.dividends_fetched == 1
    dividend_events = [e for e in repo.events if e["event_type"] == "dividend"]
    assert len(dividend_events) == 1
    assert dividend_events[0]["title"] == "Dividend AAPL"
    assert dividend_events[0]["metadata"]["dividend_id"] == "div-1"
