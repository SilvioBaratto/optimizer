"""T4 — ETF screen + builder branch.

The equity screens don't apply to funds, so ETFs run their own pipeline
(classification upstream + the history bar) and are deduped to one listing per
ISIN. The builder tags every instrument with its asset class and routes ETFs
through the ETF pipeline while the STOCK path is unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.universe.trading212.builder import UniverseBuilder
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.trading212.filters.etf_screen import dedup_etfs_by_isin

CFG = UniverseBuilderConfig()


class TestDedup:
    def test_keeps_preferred_exchange_per_isin(self) -> None:
        pref = ("Deutsche Börse Xetra", "Borsa Italiana")
        candidates = [
            ("Borsa Italiana", {"isin": "IE1", "shortName": "A"}),
            ("Deutsche Börse Xetra", {"isin": "IE1", "shortName": "A"}),
            ("Borsa Italiana", {"isin": "IE2", "shortName": "B"}),
        ]
        out = dedup_etfs_by_isin(candidates, pref)
        by_isin = {i["isin"]: ex for ex, i in out}
        assert by_isin["IE1"] == "Deutsche Börse Xetra"
        assert by_isin["IE2"] == "Borsa Italiana"
        assert len(out) == 2

    def test_keeps_candidates_without_isin(self) -> None:
        out = dedup_etfs_by_isin([("Xetra", {"shortName": "NOISIN"})], ("Xetra",))
        assert len(out) == 1


def _builder(**kw) -> UniverseBuilder:
    return UniverseBuilder(
        config=CFG,
        api_client=MagicMock(),
        ticker_mapper=MagicMock(),
        repository=MagicMock(),
        **kw,
    )


class TestPrepareExchangeETFs:
    def test_selects_fi_and_multiasset_dedups_and_excludes(self) -> None:
        exchanges = [
            {"name": "Deutsche Börse Xetra", "workingSchedules": [{"id": 1}]},
            {"name": "Borsa Italiana", "workingSchedules": [{"id": 2}]},
        ]
        instruments = [
            {
                "type": "ETF",
                "name": "iShares Global Aggregate Bond",
                "isin": "IE1",
                "workingScheduleId": 1,
                "shortName": "AGGH",
            },
            # same ISIN on Milan -> deduped away (Xetra preferred)
            {
                "type": "ETF",
                "name": "iShares Global Aggregate Bond",
                "isin": "IE1",
                "workingScheduleId": 2,
                "shortName": "AGGH",
            },
            # equity ETF -> excluded
            {
                "type": "ETF",
                "name": "Vanguard FTSE All-World",
                "isin": "IE2",
                "workingScheduleId": 1,
                "shortName": "VWCE",
            },
            # leveraged -> excluded
            {
                "type": "ETF",
                "name": "Leverage Shares 3x Long Treasury Bond",
                "isin": "IE3",
                "workingScheduleId": 1,
                "shortName": "3TYL",
            },
            # stock -> not an ETF candidate
            {
                "type": "STOCK",
                "name": "Apple",
                "workingScheduleId": 1,
                "shortName": "AAPL",
            },
        ]
        b = _builder()
        b._build_schedule_mappings(exchanges, instruments)
        result = b._prepare_exchange_etfs(exchanges)

        flat = [(ex["name"], i) for ex, insts in result for i in insts]
        assert len(flat) == 1
        exch, inst = flat[0]
        assert inst["isin"] == "IE1"
        assert exch == "Deutsche Börse Xetra"


class TestProcessSingleInstrumentTagging:
    def test_etf_is_tagged_fixed_income(self) -> None:
        b = _builder()
        b.ticker_mapper.discover.return_value = "JAGA.DE"

        inst = {
            "type": "ETF",
            "name": "JPMorgan Global Aggregate Bond",
            "shortName": "JAGA",
            "isin": "IE9",
            "currencyCode": "EUR",
        }
        data, status, _ = b._process_single_instrument(inst, "Deutsche Börse Xetra")

        assert status == "passed"
        assert data is not None
        assert data["assetClass"] == "fixed_income"
        assert data["fiSubclass"] == "aggregate"

    def test_stock_is_tagged_equity(self) -> None:
        b = _builder()
        b.ticker_mapper.discover.return_value = "AAPL"

        inst = {"type": "STOCK", "name": "Apple Inc", "shortName": "AAPL"}
        data, status, _ = b._process_single_instrument(inst, "NASDAQ")

        assert status == "passed"
        assert data is not None
        assert data["assetClass"] == "equity"


class TestBuildEndToEnd:
    def test_build_processes_stocks_and_etfs_tagged(self) -> None:
        """Full build(): stocks saved equity, FI ETFs saved fixed_income, equity
        ETFs excluded, each type through its own pipeline."""
        exchanges = [
            {"name": "NASDAQ", "workingSchedules": [{"id": 1}]},
            {"name": "Deutsche Börse Xetra", "workingSchedules": [{"id": 2}]},
        ]
        instruments = [
            {
                "type": "STOCK",
                "name": "Apple",
                "shortName": "AAPL",
                "ticker": "AAPL_US",
                "workingScheduleId": 1,
            },
            {
                "type": "ETF",
                "name": "JPMorgan Global Aggregate Bond",
                "shortName": "JAGA",
                "ticker": "JAGA",
                "isin": "IE9",
                "workingScheduleId": 2,
            },
            # equity ETF -> excluded by the classifier
            {
                "type": "ETF",
                "name": "Vanguard FTSE All-World",
                "shortName": "VWCE",
                "ticker": "VWCE",
                "isin": "IE8",
                "workingScheduleId": 2,
            },
        ]
        api = MagicMock()
        api.get_exchanges.return_value = exchanges
        api.get_instruments.return_value = instruments

        saved: list[dict] = []
        repo = MagicMock()
        ex_dto = MagicMock()
        ex_dto.id = "x"
        repo.save_exchange.return_value = ex_dto
        repo.get_active_tickers.return_value = set()
        repo.save_instruments_batch.side_effect = lambda processed, exchange_id: (
            saved.extend(processed) or len(processed)
        )

        mapper = MagicMock()
        mapper.discover.side_effect = lambda short_name, exchange: short_name

        b = UniverseBuilder(
            config=CFG,
            api_client=api,
            ticker_mapper=mapper,
            repository=repo,
        )
        result = b.build()

        by_ac = {d["ticker"]: d["assetClass"] for d in saved}
        assert by_ac.get("AAPL_US") == "equity"
        assert by_ac.get("JAGA") == "fixed_income"
        assert "VWCE" not in by_ac  # equity ETF excluded by classification
        assert result.instruments_saved == 2

    def test_etf_pass_does_not_delist_stocks_on_shared_exchange(self) -> None:
        """Review-critical regression: NASDAQ carries both a stock and a bond ETF;
        the type-scoped delisting reconciliation must not mark the stock delisted
        during the ETF pass."""

        class _FakeRepo:
            def __init__(self) -> None:
                self.store: dict[tuple, set] = {}
                self.delisted: list[str] = []

            def save_exchange(self, ex_data):
                m = MagicMock()
                m.id = ex_data["name"]
                return m

            def get_active_tickers(self, exchange_id, instrument_type=None):
                if instrument_type is None:
                    out: set = set()
                    for (e, _t), v in self.store.items():
                        if e == exchange_id:
                            out |= v
                    return out
                return set(self.store.get((exchange_id, instrument_type), set()))

            def save_instruments_batch(self, processed, exchange_id):
                for d in processed:
                    self.store.setdefault((exchange_id, d.get("type")), set()).add(
                        d.get("ticker", "")
                    )
                return len(processed)

            def mark_delisted(self, ticker, exchange_id, delisted_at):
                self.delisted.append(ticker)
                for v in self.store.values():
                    v.discard(ticker)
                return True

        exchanges = [{"name": "NASDAQ", "workingSchedules": [{"id": 1}]}]
        instruments = [
            {
                "type": "STOCK",
                "name": "Apple",
                "shortName": "AAPL",
                "ticker": "AAPL_US",
                "workingScheduleId": 1,
            },
            {
                "type": "ETF",
                "name": "iShares Core Aggregate Bond",
                "shortName": "AGG",
                "ticker": "AGG_US",
                "isin": "US1",
                "workingScheduleId": 1,
            },
        ]
        api = MagicMock()
        api.get_exchanges.return_value = exchanges
        api.get_instruments.return_value = instruments
        mapper = MagicMock()
        mapper.discover.side_effect = lambda short_name, exchange: short_name
        repo = _FakeRepo()

        b = UniverseBuilder(
            config=CFG,
            api_client=api,
            ticker_mapper=mapper,
            repository=repo,
        )
        b.build()

        assert "AAPL_US" not in repo.delisted  # stock survived the ETF pass
        assert repo.store[("NASDAQ", "STOCK")] == {"AAPL_US"}
