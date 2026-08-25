"""T4 — ETF screen + builder branch.

The equity screens don't apply to funds, so ETFs run their own pipeline
(AUM + liquidity + history) and are deduped to one listing per ISIN. The builder
tags every instrument with its asset class and routes ETFs through the ETF
pipeline while the STOCK path is unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.universe.trading212.builder import UniverseBuilder
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.trading212.filters.etf_screen import (
    AUMFilter,
    dedup_etfs_by_isin,
)

CFG = UniverseBuilderConfig()


class TestAUMFilter:
    def test_passes_above_floor(self) -> None:
        ok, _ = AUMFilter(CFG).filter({"totalAssets": 2e8}, "X")
        assert ok is True

    def test_rejects_below_floor(self) -> None:
        ok, reason = AUMFilter(CFG).filter({"totalAssets": 5e7}, "X")
        assert ok is False and "AUM" in reason

    def test_passes_when_aum_unknown(self) -> None:
        # Yahoo omits totalAssets for many valid bond ETFs — unknown must pass.
        ok, reason = AUMFilter(CFG).filter({"foo": 1}, "X")
        assert ok is True and "unknown" in reason


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
        filter_pipeline=MagicMock(),
        etf_filter_pipeline=MagicMock(),
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
    def test_etf_is_tagged_and_routed_to_etf_pipeline(self) -> None:
        etf_pipeline = MagicMock()
        etf_pipeline.apply.return_value = (True, "ok")
        stock_pipeline = MagicMock()
        b = _builder()
        b.filter_pipeline = stock_pipeline
        b.etf_filter_pipeline = etf_pipeline
        b.ticker_mapper.discover.return_value = "JAGA.DE"

        inst = {
            "type": "ETF",
            "name": "JPMorgan Global Aggregate Bond",
            "shortName": "JAGA",
            "isin": "IE9",
            "currencyCode": "EUR",
        }
        with patch.object(b, "_fetch_filter_data", return_value={"totalAssets": 1e9}):
            data, status, _ = b._process_single_instrument(inst, "Deutsche Börse Xetra")

        assert status == "passed"
        assert data is not None
        assert data["assetClass"] == "fixed_income"
        assert data["fiSubclass"] == "aggregate"
        etf_pipeline.apply.assert_called_once()
        stock_pipeline.apply.assert_not_called()

    def test_stock_is_tagged_equity_and_uses_stock_pipeline(self) -> None:
        stock_pipeline = MagicMock()
        stock_pipeline.apply.return_value = (True, "ok")
        b = _builder()
        b.filter_pipeline = stock_pipeline
        b.etf_filter_pipeline = MagicMock()
        b.ticker_mapper.discover.return_value = "AAPL"

        inst = {"type": "STOCK", "name": "Apple Inc", "shortName": "AAPL"}
        with patch.object(b, "_fetch_filter_data", return_value={"marketCap": 1e12}):
            data, status, _ = b._process_single_instrument(inst, "NASDAQ")

        assert status == "passed"
        assert data is not None
        assert data["assetClass"] == "equity"
        stock_pipeline.apply.assert_called_once()
