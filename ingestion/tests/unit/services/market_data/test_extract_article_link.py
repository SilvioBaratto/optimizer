"""The one rule every news path shares for a Yahoo item's link.

Validates ``app/services/market_data/yfinance/news/_links.py``
(``extract_article_link`` and the ``content_link`` fallback). The cases come
from the four copies the helper replaced: the yfinance sweep, ``NewsClient``,
``CountryNewsFetcher`` and ``MacroNewsFetcher``. ``test_article_link_sites.py``
pins what each path does with it. No requirement id is registered for link
selection yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from app.services.market_data.yfinance.news._links import (
    content_link,
    extract_article_link,
)

_URL = "https://example.com/story"


def _refuse(item: Mapping[str, Any], content: Mapping[str, Any]) -> str | None:
    raise AssertionError("the fallback must not be consulted")


def _recording(answer: str | None) -> tuple[list[tuple[Any, Any]], Any]:
    calls: list[tuple[Any, Any]] = []

    def fallback(item: Mapping[str, Any], content: Mapping[str, Any]) -> str | None:
        """Record the arguments and give the fixed answer."""
        calls.append((item, content))
        return answer

    return calls, fallback


def test_extract_article_link_reads_the_nested_canonical_url() -> None:
    """A ``canonicalUrl`` object under ``content`` gives its ``url``."""
    item = {"content": {"canonicalUrl": {"url": _URL}}}

    assert extract_article_link(item) == _URL


def test_extract_article_link_reads_the_canonical_url_of_a_flat_item() -> None:
    """A legacy item without ``content`` is read as its own content."""
    assert extract_article_link({"canonicalUrl": {"url": _URL}}) == _URL


def test_extract_article_link_prefers_the_canonical_url_to_the_fallback() -> None:
    """The canonical object wins, so no path's fallback is consulted."""
    item = {
        "link": "https://example.com/item",
        "content": {
            "canonicalUrl": {"url": _URL},
            "previewUrl": "https://example.com/preview",
            "link": "https://example.com/content",
        },
    }

    assert extract_article_link(item, _refuse) == _URL


def test_extract_article_link_gives_empty_text_for_a_canonical_object_without_url() -> (
    None
):
    """An object without ``url`` names no page, and the fallback is not consulted."""
    item = {"content": {"canonicalUrl": {}, "link": "https://example.com/content"}}

    assert extract_article_link(item, _refuse) == ""


def test_extract_article_link_ignores_a_nested_click_through_url() -> None:
    """``clickThroughUrl`` is not a canonical URL; only the fallback can answer."""
    item = {
        "content": {
            "clickThroughUrl": {"url": "https://click.example.com/story"},
            "link": "https://example.com/content",
        }
    }

    assert extract_article_link(item) is None
    assert extract_article_link(item, content_link) == "https://example.com/content"


def test_extract_article_link_hands_a_string_canonical_url_to_the_fallback() -> None:
    """Only an object counts as canonical; a string is left to the fallback."""
    item = {"content": {"canonicalUrl": _URL}}
    calls, fallback = _recording("https://example.com/fallback")

    assert extract_article_link(item, fallback) == "https://example.com/fallback"
    assert calls == [(item, item["content"])]


def test_extract_article_link_passes_a_flat_item_as_its_own_content() -> None:
    """The fallback of a flat item sees the item twice."""
    item = {"link": _URL}
    calls, fallback = _recording(_URL)

    extract_article_link(item, fallback)

    assert calls[0][0] is item
    assert calls[0][1] is item


@pytest.mark.parametrize("answer", [None, "", _URL])
def test_extract_article_link_returns_the_fallback_answer_as_is(
    answer: str | None,
) -> None:
    """Whatever the fallback answers is the link, falsy answers included."""
    _, fallback = _recording(answer)

    assert extract_article_link({"content": {}}, fallback) == answer


@pytest.mark.parametrize(
    "item", [{}, {"content": {}}, {"content": {"canonicalUrl": None}}]
)
def test_extract_article_link_is_none_when_missing_without_a_fallback(
    item: dict[str, Any],
) -> None:
    """Without a canonical object and without a fallback there is no link."""
    assert extract_article_link(item) is None


@pytest.mark.parametrize(
    "url", ["ftp://files.example.com/story", "/relative/story", "javascript:void(0)"]
)
def test_extract_article_link_keeps_a_non_http_url(url: str) -> None:
    """No scheme check here: the scraper refuses what it cannot fetch."""
    assert extract_article_link({"content": {"canonicalUrl": {"url": url}}}) == url


def test_content_link_reads_the_link_of_the_content() -> None:
    """The country and macro fallback reads ``link`` from the content only."""
    item = {"link": "https://example.com/item", "content": {"link": _URL}}

    assert content_link(item, item["content"]) == _URL


def test_content_link_is_empty_text_when_the_content_has_none() -> None:
    """Those paths store ``""`` for a missing link, and hash it into the id."""
    item = {"link": "https://example.com/item", "content": {}}

    assert content_link(item, item["content"]) == ""
