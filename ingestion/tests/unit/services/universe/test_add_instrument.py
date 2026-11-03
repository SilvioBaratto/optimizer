"""``add_instrument``: validate one Yahoo ticker and upsert its instrument row.

Yahoo is mocked at the ``YFinanceClient`` seam (``fetch_info`` / ``fetch_history``)
and the database is the SQLite test engine, so the service runs end to end with
no network. The row it writes must look like one the universe build would store,
so the weekly build, the price sweep and the fund read it without special cases.
"""

from __future__ import annotations

import datetime as dt
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pandas as pd
import pytest
from portopt_db.models.universe.universe import Exchange, Instrument

from app.services.universe.add_instrument import (
    AddInstrumentError,
    AddInstrumentResult,
    add_instrument,
)

XETRA = "Deutsche Börse Xetra"
LSE = "London Stock Exchange"


def _etf_info(**overrides: Any) -> dict[str, Any]:
    info = {
        "quoteType": "ETF",
        "exchange": "GER",
        "currency": "EUR",
        "longName": "Vanguard FTSE All-World UCITS ETF USD Acc",
        "shortName": "VANGUARD FTSE ALL-WORLD U",
    }
    info.update(overrides)
    return info


def _history(rows: int = 20) -> pd.DataFrame:
    index = pd.date_range("2026-10-01", periods=rows, freq="B")
    return pd.DataFrame({"Close": [100.0 + i for i in range(rows)]}, index=index)


def _client(info: dict[str, Any] | None, history: pd.DataFrame | None = None):
    client = MagicMock()
    client.fetch_info.return_value = info
    client.fetch_history.return_value = _history() if history is None else history
    return client


def _exchange(session, name: str = XETRA) -> Exchange:
    row = session.query(Exchange).filter_by(name=name).one_or_none()
    if row is None:
        row = Exchange(name=name)
        session.add(row)
        session.flush()
    return row


def _row(session, ticker: str, exchange: Exchange) -> Instrument:
    return (
        session.query(Instrument)
        .filter_by(ticker=ticker, exchange_id=exchange.id)
        .execution_options(populate_existing=True)
        .one()
    )


def _add(session, client, **kwargs: Any) -> AddInstrumentResult:
    kwargs.setdefault("yahoo_ticker", "VWCE.DE")
    kwargs.setdefault("asset_class", "equity")
    return add_instrument(session, client, **kwargs)


class TestCreatesTheRow:
    """A valid ETF or stock becomes one instrument row shaped like a build row."""

    def test_etf_row_carries_yahoo_listing_data(self, db_session) -> None:
        """Type, class, currency, names and the Yahoo ticker are stored."""
        xetra = _exchange(db_session)

        result = _add(db_session, _client(_etf_info()))

        row = _row(db_session, "VWCE.DE", xetra)
        assert result.created is True
        assert result.instrument_id == row.id
        assert result.warnings == []
        assert row.instrument_type == "ETF"
        assert row.asset_class == "equity"
        assert row.yfinance_ticker == "VWCE.DE"
        assert row.currency_code == "EUR"
        assert row.name == "Vanguard FTSE All-World UCITS ETF USD Acc"
        assert row.short_name == "VANGUARD FTSE ALL-WORLD U"
        assert row.delisted_at is None

    def test_stock_defaults_to_equity(self, db_session) -> None:
        """A STOCK needs no asset class."""
        borsa = _exchange(db_session, "Borsa Italiana")
        info = _etf_info(quoteType="EQUITY", exchange="MIL", longName="Eni SpA")

        result = add_instrument(
            db_session, _client(info), yahoo_ticker="ENI.MI", asset_class=None
        )

        row = _row(db_session, "ENI.MI", borsa)
        assert result.created is True
        assert row.instrument_type == "STOCK"
        assert row.asset_class == "equity"

    def test_currency_is_stored_raw(self, db_session) -> None:
        """Pence stay pence: the price unit is the raw Yahoo listing currency."""
        lse = _exchange(db_session, LSE)
        info = _etf_info(exchange="LSE", currency="GBp")

        _add(db_session, _client(info), yahoo_ticker="EGLN.L", asset_class="commodity")

        assert _row(db_session, "EGLN.L", lse).currency_code == "GBp"

    def test_ticker_is_stripped_and_uppercased(self, db_session) -> None:
        """The stored ticker and the Yahoo request use the normalised symbol."""
        xetra = _exchange(db_session)
        client = _client(_etf_info())

        _add(db_session, client, yahoo_ticker="  vwce.de ")

        assert _row(db_session, "VWCE.DE", xetra).yfinance_ticker == "VWCE.DE"
        assert client.fetch_info.call_args.args[0] == "VWCE.DE"
        assert client.fetch_history.call_args.args[0] == "VWCE.DE"

    def test_fixed_income_options_are_stored(self, db_session) -> None:
        """fi_subclass and duration_bucket go with a fixed_income class."""
        xetra = _exchange(db_session)

        _add(
            db_session,
            _client(_etf_info()),
            yahoo_ticker="IEGA.DE",
            asset_class="fixed_income",
            fi_subclass="government",
            duration_bucket="long",
        )

        row = _row(db_session, "IEGA.DE", xetra)
        assert (row.fi_subclass, row.duration_bucket) == ("government", "long")

    def test_t212_ticker_isin_and_name_overrides_are_stored(self, db_session) -> None:
        """Operator-supplied fields are written to the row."""
        xetra = _exchange(db_session)

        _add(
            db_session,
            _client(_etf_info()),
            t212_ticker="VWCEd_EQ",
            isin="IE00BK5BQT80",
            name="Vanguard All-World",
        )

        row = _row(db_session, "VWCE.DE", xetra)
        assert row.t212_ticker == "VWCEd_EQ"
        assert row.isin == "IE00BK5BQT80"
        assert row.name == "Vanguard All-World"

    def test_history_is_probed_and_the_session_is_not_committed(
        self, db_session, monkeypatch
    ) -> None:
        """Price history is fetched to validate, and the caller owns the commit."""
        _exchange(db_session)
        client = _client(_etf_info())
        commits: list[bool] = []
        monkeypatch.setattr(db_session, "commit", lambda: commits.append(True))

        _add(db_session, client)

        client.fetch_history.assert_called_once()
        assert commits == []


class TestIdempotency:
    """Re-running with the same arguments converges on one row."""

    def test_second_call_updates_the_same_row(self, db_session) -> None:
        """created flips to False, the id is stable, no duplicate appears."""
        xetra = _exchange(db_session)
        first = _add(db_session, _client(_etf_info()))

        second = _add(db_session, _client(_etf_info()))

        assert (first.created, second.created) == (True, False)
        assert second.instrument_id == first.instrument_id
        assert db_session.query(Instrument).filter_by(exchange_id=xetra.id).count() == 1

    def test_a_delisted_row_is_revived(self, db_session) -> None:
        """Adding an instrument the build had delisted brings it back."""
        xetra = _exchange(db_session)
        _add(db_session, _client(_etf_info()))
        row = _row(db_session, "VWCE.DE", xetra)
        row.delisted_at = dt.date(2026, 10, 1)
        row.delisting_return = -0.1
        db_session.flush()

        result = _add(db_session, _client(_etf_info()))

        row = _row(db_session, "VWCE.DE", xetra)
        assert result.created is False
        assert row.delisted_at is None
        assert row.delisting_return is None

    def test_rerun_without_name_isin_or_t212_keeps_the_stored_values(
        self, db_session
    ) -> None:
        """A missing optional value never erases one already stored."""
        xetra = _exchange(db_session)
        _add(
            db_session,
            _client(_etf_info(longName="First Name")),
            name="Operator Name",
            isin="IE00BK5BQT80",
            t212_ticker="VWCEd_EQ",
        )

        _add(db_session, _client(_etf_info(longName=None)))

        row = _row(db_session, "VWCE.DE", xetra)
        assert row.name == "Operator Name"
        assert row.isin == "IE00BK5BQT80"
        assert row.t212_ticker == "VWCEd_EQ"

    def test_rerun_with_a_corrected_asset_class_changes_it(self, db_session) -> None:
        """The operator can fix a classification by running the command again."""
        xetra = _exchange(db_session)
        _add(db_session, _client(_etf_info()), asset_class="equity")

        _add(db_session, _client(_etf_info()), asset_class="commodity")

        assert _row(db_session, "VWCE.DE", xetra).asset_class == "commodity"


class TestRefusals:
    """Each invalid input raises a clear error and writes nothing."""

    def _assert_nothing_written(self, db_session) -> None:
        assert db_session.query(Instrument).count() == 0

    def test_etf_without_asset_class_is_refused(self, db_session) -> None:
        """No guessing: an ETF must be classified by the operator."""
        _exchange(db_session)

        with pytest.raises(AddInstrumentError, match="asset.class"):
            add_instrument(
                db_session,
                _client(_etf_info()),
                yahoo_ticker="VWCE.DE",
                asset_class=None,
            )

        self._assert_nothing_written(db_session)

    def test_unknown_asset_class_lists_the_choices(self, db_session) -> None:
        """An invalid class names the supported values."""
        _exchange(db_session)

        with pytest.raises(AddInstrumentError, match="commodity"):
            _add(db_session, _client(_etf_info()), asset_class="crypto")

    @pytest.mark.parametrize("asset_class", ["equity", "multi_asset", "commodity"])
    def test_fi_options_only_go_with_fixed_income(
        self, db_session, asset_class: str
    ) -> None:
        """fi_subclass and duration_bucket are refused on any other class."""
        _exchange(db_session)

        with pytest.raises(AddInstrumentError, match="fixed_income"):
            _add(
                db_session,
                _client(_etf_info()),
                asset_class=asset_class,
                fi_subclass="government",
            )
        with pytest.raises(AddInstrumentError, match="fixed_income"):
            _add(
                db_session,
                _client(_etf_info()),
                asset_class=asset_class,
                duration_bucket="short",
            )

    def test_unknown_fi_values_list_the_choices(self, db_session) -> None:
        """Invalid sub-class and bucket values name the supported ones."""
        _exchange(db_session)

        with pytest.raises(AddInstrumentError, match="inflation_linked"):
            _add(
                db_session,
                _client(_etf_info()),
                asset_class="fixed_income",
                fi_subclass="junk",
            )
        with pytest.raises(AddInstrumentError, match="intermediate"):
            _add(
                db_session,
                _client(_etf_info()),
                asset_class="fixed_income",
                duration_bucket="forever",
            )

    @pytest.mark.parametrize(
        "quote_type", ["MUTUALFUND", "CRYPTOCURRENCY", "INDEX", ""]
    )
    def test_other_quote_types_are_refused(self, db_session, quote_type: str) -> None:
        """Only EQUITY and ETF listings can be added."""
        _exchange(db_session)

        with pytest.raises(AddInstrumentError, match="EQUITY"):
            _add(db_session, _client(_etf_info(quoteType=quote_type)))

        self._assert_nothing_written(db_session)

    def test_yahoo_without_a_profile_is_refused(self, db_session) -> None:
        """No info dict means the ticker is unknown or Yahoo did not answer."""
        _exchange(db_session)

        with pytest.raises(AddInstrumentError, match="VWCE.DE"):
            _add(db_session, _client(None))

        self._assert_nothing_written(db_session)

    def test_unmapped_exchange_code_lists_supported_venues(self, db_session) -> None:
        """A venue outside the covered set is an error naming the covered ones."""
        _exchange(db_session)

        with pytest.raises(AddInstrumentError) as excinfo:
            _add(db_session, _client(_etf_info(exchange="FKA")))

        message = str(excinfo.value)
        assert "FKA" in message
        assert XETRA in message
        assert "--exchange" in message
        self._assert_nothing_written(db_session)

    def test_missing_exchange_code_is_refused(self, db_session) -> None:
        """Yahoo omitting the venue cannot be resolved silently."""
        _exchange(db_session)

        with pytest.raises(AddInstrumentError, match="exchange"):
            _add(db_session, _client(_etf_info(exchange=None)))

    def test_exchange_missing_from_the_database_is_never_created(
        self, db_session
    ) -> None:
        """The service reads exchanges; it does not invent one."""
        with pytest.raises(AddInstrumentError, match=XETRA):
            _add(db_session, _client(_etf_info()))

        assert db_session.query(Exchange).count() == 0
        self._assert_nothing_written(db_session)

    def test_empty_history_is_refused(self, db_session) -> None:
        """A listing with no price bars is not investable."""
        _exchange(db_session)
        client = _client(_etf_info())
        client.fetch_history.return_value = None

        with pytest.raises(AddInstrumentError, match="price history"):
            _add(db_session, client)

        self._assert_nothing_written(db_session)

    def test_empty_dataframe_history_is_refused(self, db_session) -> None:
        """An empty frame counts as no history too."""
        _exchange(db_session)

        with pytest.raises(AddInstrumentError, match="price history"):
            _add(db_session, _client(_etf_info(), history=_history(rows=0)))

        self._assert_nothing_written(db_session)


class TestExchangeOverride:
    """``exchange`` wins over the Yahoo code."""

    def test_override_resolves_an_unmapped_code(self, db_session) -> None:
        """An explicit venue lets a ticker on an unmapped Yahoo code in."""
        lse = _exchange(db_session, LSE)

        _add(
            db_session,
            _client(_etf_info(exchange="FKA", currency="GBp")),
            yahoo_ticker="EGLN.L",
            asset_class="commodity",
            exchange=LSE,
        )

        assert _row(db_session, "EGLN.L", lse).asset_class == "commodity"

    def test_override_beats_the_mapped_code(self, db_session) -> None:
        """The operator's venue is used even when Yahoo's code maps elsewhere."""
        _exchange(db_session, XETRA)
        lse = _exchange(db_session, LSE)

        _add(db_session, _client(_etf_info(exchange="GER")), exchange=LSE)

        assert _row(db_session, "VWCE.DE", lse) is not None

    def test_unknown_override_lists_supported_venues(self, db_session) -> None:
        """A venue name not in the database is an error naming the covered ones."""
        _exchange(db_session)

        with pytest.raises(AddInstrumentError, match=XETRA):
            _add(db_session, _client(_etf_info()), exchange="Narnia Exchange")


class TestDuplicateIsinWarning:
    """Two listings of one ISIN are allowed but flagged, naming the other."""

    def test_warns_and_names_the_other_instrument(self, db_session) -> None:
        """The IBCI.L / IBCI.AS case: same ISIN, different listing."""
        xetra = _exchange(db_session)
        db_session.add(
            Instrument(
                ticker="IBCI.AS",
                short_name="IBCI",
                yfinance_ticker="IBCI.AS",
                isin="IE00B0M62X26",
                exchange_id=_exchange(db_session, "Euronext Amsterdam").id,
            )
        )
        db_session.flush()

        result = _add(
            db_session,
            _client(_etf_info()),
            yahoo_ticker="IBCI.L",
            isin="IE00B0M62X26",
        )

        assert result.created is True
        assert len(result.warnings) == 1
        assert "IBCI.AS" in result.warnings[0]
        assert _row(db_session, "IBCI.L", xetra).isin == "IE00B0M62X26"

    def test_no_warning_when_the_isin_is_unique(self, db_session) -> None:
        """A distinct ISIN, or none, raises no warning."""
        _exchange(db_session)

        assert (
            _add(db_session, _client(_etf_info()), isin="IE00BK5BQT80").warnings == []
        )
        assert _add(db_session, _client(_etf_info())).warnings == []

    def test_rerun_does_not_warn_about_itself(self, db_session) -> None:
        """The row being updated is not its own duplicate."""
        _exchange(db_session)
        _add(db_session, _client(_etf_info()), isin="IE00BK5BQT80")

        result = _add(db_session, _client(_etf_info()), isin="IE00BK5BQT80")

        assert result.warnings == []


class TestMissingCurrency:
    """Yahoo omitting the currency is stored as unknown, with a warning."""

    def test_warns_and_stores_none(self, db_session) -> None:
        """Absent currency does not block the add."""
        xetra = _exchange(db_session)

        result = _add(db_session, _client(_etf_info(currency=None)))

        assert _row(db_session, "VWCE.DE", xetra).currency_code is None
        assert any("currency" in w for w in result.warnings)


def test_result_is_a_frozen_value_object() -> None:
    """``AddInstrumentResult`` cannot be mutated after the call returns."""
    result = AddInstrumentResult(instrument_id=uuid4(), created=True)
    with pytest.raises(AttributeError):
        result.created = False  # type: ignore[misc]
    assert result.warnings == []
