"""Tests for issue #606: ScreenerClient.screen_etfs_predefined convenience method.

Validates the canonical 4-name ETF allowlist and routes through Cycle 1's
branched ``screen()`` so ``yf.screen`` receives ``count=size`` (string-query
branch), not ``size=size``.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytest.importorskip("yfinance")

from app.services.market_data.yfinance.market.screener import ScreenerClient
from app.services.market_data.yfinance.protocols import ScreenerClientProtocol

_VALID_NAMES = (
    "top_etfs_us",
    "top_performing_etfs",
    "technology_etfs",
    "bond_etfs",
)


def _build_client() -> ScreenerClient:
    rate_limiter = MagicMock(name="rate_limiter")
    circuit_breaker = MagicMock(name="circuit_breaker")
    circuit_breaker.check.return_value = None
    return ScreenerClient(
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
        default_max_retries=1,
    )


def _patch_yf_screen(df: pd.DataFrame | None = None):
    if df is None:
        df = pd.DataFrame({"symbol": ["SPY"]})
    return patch(
        "app.services.market_data.yfinance.market.screener.yf.screen",
        return_value=df,
    )


@pytest.mark.parametrize("name", _VALID_NAMES)
def test_when_valid_etf_name_then_yf_screen_invoked_with_count_size(name: str) -> None:
    client = _build_client()
    with _patch_yf_screen() as mock_screen:
        client.screen_etfs_predefined(name, size=10)
    kwargs = mock_screen.call_args.kwargs
    assert kwargs.get("count") == 10
    assert "size" not in kwargs
    assert mock_screen.call_args.args[0] == name


def test_when_invalid_name_then_value_error_lists_canonical_names() -> None:
    client = _build_client()
    with pytest.raises(ValueError) as excinfo:
        client.screen_etfs_predefined("not_a_real_screener")
    msg = str(excinfo.value)
    for name in _VALID_NAMES:
        assert name in msg


def test_when_invalid_name_then_yf_screen_not_called() -> None:
    client = _build_client()
    with (
        patch(
            "app.services.market_data.yfinance.market.screener.yf.screen"
        ) as mock_screen,
        pytest.raises(ValueError),
    ):
        client.screen_etfs_predefined("most_actives")
    mock_screen.assert_not_called()


def test_when_name_case_differs_then_value_error_raised() -> None:
    client = _build_client()
    with pytest.raises(ValueError):
        client.screen_etfs_predefined("Top_ETFs_US")


def test_when_offset_passed_then_forwarded_to_yf_screen() -> None:
    client = _build_client()
    with _patch_yf_screen() as mock_screen:
        client.screen_etfs_predefined("top_etfs_us", size=5, offset=12)
    assert mock_screen.call_args.kwargs.get("offset") == 12


def test_when_sort_field_and_sort_asc_passed_then_forwarded() -> None:
    client = _build_client()
    with _patch_yf_screen() as mock_screen:
        client.screen_etfs_predefined(
            "bond_etfs",
            size=5,
            sort_field="percentchange",
            sort_asc=False,
        )
    kwargs = mock_screen.call_args.kwargs
    assert kwargs.get("sortField") == "percentchange"
    assert kwargs.get("sortAsc") is False


def test_when_default_size_used_then_count_is_25() -> None:
    client = _build_client()
    with _patch_yf_screen() as mock_screen:
        client.screen_etfs_predefined("technology_etfs")
    assert mock_screen.call_args.kwargs.get("count") == 25


def test_when_screener_client_built_then_protocol_is_satisfied() -> None:
    assert isinstance(_build_client(), ScreenerClientProtocol)


def test_when_inspecting_protocol_then_screen_etfs_predefined_declared() -> None:
    assert hasattr(ScreenerClientProtocol, "screen_etfs_predefined")


def test_when_inspecting_signatures_then_impl_matches_protocol() -> None:
    impl = inspect.signature(ScreenerClient.screen_etfs_predefined)
    proto = inspect.signature(ScreenerClientProtocol.screen_etfs_predefined)
    assert impl.parameters.keys() == proto.parameters.keys()
    for name, impl_param in impl.parameters.items():
        assert impl_param.default == proto.parameters[name].default, name


def test_when_screen_etfs_predefined_returns_dataframe_then_passed_through() -> None:
    df = pd.DataFrame({"symbol": ["SPY", "VOO"]})
    client = _build_client()
    with _patch_yf_screen(df=df):
        result = client.screen_etfs_predefined("top_etfs_us", size=2)
    pd.testing.assert_frame_equal(result, df)
