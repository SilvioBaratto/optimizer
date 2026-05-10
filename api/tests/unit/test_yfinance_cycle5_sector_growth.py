"""Tests for issue #605: SectorIndustryClient adds 1.3.0 surface methods.

Five new methods delegate via existing ``_fetch_sector_attr`` /
``_fetch_industry_attr`` helpers and route through the standard resilience
chain. DataFrame-returning methods skip empty results.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytest.importorskip("yfinance")

from app.services.market_data.yfinance.market.sector_industry import (
    SectorIndustryClient,
)
from app.services.market_data.yfinance.protocols import SectorIndustryClientProtocol


def _build_client() -> SectorIndustryClient:
    rate_limiter = MagicMock(name="rate_limiter")
    circuit_breaker = MagicMock(name="circuit_breaker")
    circuit_breaker.check.return_value = None
    return SectorIndustryClient(
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
        default_max_retries=1,
    )


def _patch_sector(**attrs: Any):
    instance = MagicMock(name="yf.Sector.instance")
    drop = attrs.pop("drop", ())
    for attr, value in attrs.items():
        setattr(instance, attr, value)
    for attr in drop:
        delattr(instance, attr)
    return patch(
        "app.services.market_data.yfinance.market.sector_industry.yf.Sector",
        return_value=instance,
    )


def _patch_industry(**attrs: Any):
    instance = MagicMock(name="yf.Industry.instance")
    drop = attrs.pop("drop", ())
    for attr, value in attrs.items():
        setattr(instance, attr, value)
    for attr in drop:
        delattr(instance, attr)
    return patch(
        "app.services.market_data.yfinance.market.sector_industry.yf.Industry",
        return_value=instance,
    )


# -----------------------------------------------------------------------------
# Sector — happy path
# -----------------------------------------------------------------------------


def test_when_fetch_sector_industries_called_then_yf_sector_industries_returned() -> (
    None
):
    df = pd.DataFrame({"key": ["software"], "name": ["Software"]})
    client = _build_client()
    with _patch_sector(industries=df) as mock_cls:
        result = client.fetch_sector_industries("technology")

    mock_cls.assert_called_once_with("technology")
    pd.testing.assert_frame_equal(result, df)


def test_when_fetch_sector_research_reports_called_then_yf_sector_attribute_returned() -> (
    None
):
    df = pd.DataFrame({"title": ["Tech Outlook"]})
    client = _build_client()
    with _patch_sector(research_reports=df) as mock_cls:
        result = client.fetch_sector_research_reports("technology")

    mock_cls.assert_called_once_with("technology")
    pd.testing.assert_frame_equal(result, df)


def test_when_fetch_sector_top_growth_companies_called_then_yf_sector_attribute_returned() -> (
    None
):
    df = pd.DataFrame({"symbol": ["AAPL"], "growth": [0.25]})
    client = _build_client()
    with _patch_sector(top_growth_companies=df) as mock_cls:
        result = client.fetch_sector_top_growth_companies("technology")

    mock_cls.assert_called_once_with("technology")
    pd.testing.assert_frame_equal(result, df)


# -----------------------------------------------------------------------------
# Industry — happy path
# -----------------------------------------------------------------------------


def test_when_fetch_industry_research_reports_called_then_yf_industry_attribute_returned() -> (
    None
):
    df = pd.DataFrame({"title": ["Software Outlook"]})
    client = _build_client()
    with _patch_industry(research_reports=df) as mock_cls:
        result = client.fetch_industry_research_reports("software-infrastructure")

    mock_cls.assert_called_once_with("software-infrastructure")
    pd.testing.assert_frame_equal(result, df)


def test_when_fetch_industry_top_growth_companies_called_then_yf_industry_attribute_returned() -> (
    None
):
    df = pd.DataFrame({"symbol": ["MSFT"], "growth": [0.30]})
    client = _build_client()
    with _patch_industry(top_growth_companies=df) as mock_cls:
        result = client.fetch_industry_top_growth_companies("software-infrastructure")

    mock_cls.assert_called_once_with("software-infrastructure")
    pd.testing.assert_frame_equal(result, df)


# -----------------------------------------------------------------------------
# Missing-attribute fallback returns None
# -----------------------------------------------------------------------------


def test_when_sector_lacks_industries_attr_then_returns_none() -> None:
    client = _build_client()
    with _patch_sector(drop=("industries",)):
        assert client.fetch_sector_industries("technology") is None


def test_when_sector_lacks_research_reports_attr_then_returns_none() -> None:
    client = _build_client()
    with _patch_sector(drop=("research_reports",)):
        assert client.fetch_sector_research_reports("technology") is None


def test_when_sector_lacks_top_growth_companies_attr_then_returns_none() -> None:
    client = _build_client()
    with _patch_sector(drop=("top_growth_companies",)):
        assert client.fetch_sector_top_growth_companies("technology") is None


def test_when_industry_lacks_research_reports_attr_then_returns_none() -> None:
    client = _build_client()
    with _patch_industry(drop=("research_reports",)):
        assert client.fetch_industry_research_reports("software-infrastructure") is None


def test_when_industry_lacks_top_growth_companies_attr_then_returns_none() -> None:
    client = _build_client()
    with _patch_industry(drop=("top_growth_companies",)):
        assert (
            client.fetch_industry_top_growth_companies("software-infrastructure")
            is None
        )


# -----------------------------------------------------------------------------
# Empty DataFrame guard (stricter is_valid for DataFrame methods)
# -----------------------------------------------------------------------------


def test_when_sector_industries_returns_empty_dataframe_then_none_returned() -> None:
    client = _build_client()
    with _patch_sector(industries=pd.DataFrame()):
        assert client.fetch_sector_industries("technology") is None


def test_when_industry_top_growth_companies_returns_empty_dataframe_then_none_returned() -> (
    None
):
    client = _build_client()
    with _patch_industry(top_growth_companies=pd.DataFrame()):
        assert (
            client.fetch_industry_top_growth_companies("software-infrastructure")
            is None
        )


# -----------------------------------------------------------------------------
# Protocol + regression
# -----------------------------------------------------------------------------


def test_when_sector_industry_client_built_then_protocol_is_satisfied() -> None:
    assert isinstance(_build_client(), SectorIndustryClientProtocol)


def test_when_inspecting_protocol_then_five_new_methods_declared() -> None:
    expected = {
        "fetch_sector_industries",
        "fetch_sector_research_reports",
        "fetch_industry_research_reports",
        "fetch_sector_top_growth_companies",
        "fetch_industry_top_growth_companies",
    }
    methods = {m for m in dir(SectorIndustryClientProtocol) if not m.startswith("_")}
    assert expected <= methods


def test_when_existing_fetch_sector_overview_called_then_still_returns_dict() -> None:
    overview = {"key": "technology", "name": "Technology"}
    client = _build_client()
    with _patch_sector(overview=overview):
        assert client.fetch_sector_overview("technology") == overview


def test_when_existing_fetch_industry_top_companies_called_then_still_returns_df() -> (
    None
):
    df = pd.DataFrame({"symbol": ["AAPL"]})
    client = _build_client()
    with _patch_industry(top_companies=df):
        result = client.fetch_industry_top_companies("software-infrastructure")
    pd.testing.assert_frame_equal(result, df)


def test_when_new_methods_signatures_match_protocol() -> None:
    impl_methods = (
        "fetch_sector_industries",
        "fetch_sector_research_reports",
        "fetch_industry_research_reports",
        "fetch_sector_top_growth_companies",
        "fetch_industry_top_growth_companies",
    )
    for name in impl_methods:
        impl_sig = inspect.signature(getattr(SectorIndustryClient, name))
        proto_sig = inspect.signature(getattr(SectorIndustryClientProtocol, name))
        assert impl_sig.parameters.keys() == proto_sig.parameters.keys(), name
