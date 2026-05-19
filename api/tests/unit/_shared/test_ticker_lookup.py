"""Tests for app.services._shared._ticker_lookup.lookup_yf_ticker (issue #773)."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestLookupYfTicker:
    def test_when_t212_ticker_found_then_yfinance_ticker_returned(self) -> None:
        from app.services._shared._ticker_lookup import lookup_yf_ticker

        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = "VOD.L"

        result = lookup_yf_ticker("VOD_l_EQ", session)

        assert result == "VOD.L"
        session.execute.assert_called_once()

    def test_when_t212_ticker_missing_then_none_returned(self) -> None:
        from app.services._shared._ticker_lookup import lookup_yf_ticker

        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = None

        result = lookup_yf_ticker("UNKNOWN_EQ", session)

        assert result is None

    def test_when_empty_string_then_none_returned(self) -> None:
        from app.services._shared._ticker_lookup import lookup_yf_ticker

        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = None

        result = lookup_yf_ticker("", session)

        assert result is None

    def test_when_called_then_query_uses_instrument_ticker_filter(self) -> None:
        """The lookup must query Instrument.yfinance_ticker WHERE Instrument.ticker = t212_ticker."""
        from app.services._shared._ticker_lookup import lookup_yf_ticker

        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = "AAPL"

        lookup_yf_ticker("AAPL_US_EQ", session)

        stmt = session.execute.call_args.args[0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "yfinance_ticker" in compiled
        assert "AAPL_US_EQ" in compiled


class TestBrokerSyncUsesSharedLookup:
    def test_broker_sync_module_does_not_define_local_mapper(self) -> None:
        """The local _map_t212_ticker_to_yfinance must be fully removed."""
        import app.services.portfolio.broker_sync_service as mod

        assert not hasattr(mod, "_map_t212_ticker_to_yfinance")

    def test_broker_sync_imports_shared_lookup_yf_ticker(self) -> None:
        from app.services._shared._ticker_lookup import lookup_yf_ticker
        from app.services.portfolio import broker_sync_service as mod

        assert mod.lookup_yf_ticker is lookup_yf_ticker
