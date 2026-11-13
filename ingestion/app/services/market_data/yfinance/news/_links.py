"""The page a Yahoo news item links to.

Yahoo serves a news item either flat (the legacy shape) or with its fields
nested under ``content``. Every news path takes a nested ``canonicalUrl`` object
first. Where there is none they look in different places, and some store or hash
what they find, so each path passes its own fallback rather than sharing one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar, overload

_Link = TypeVar("_Link", bound=str | None)

LinkFallback = Callable[[Mapping[str, Any], Mapping[str, Any]], _Link]
"""Picks a link from ``(item, content)`` when the item has no canonical object."""


@overload
def extract_article_link(item: Mapping[str, Any]) -> str | None: ...


@overload
def extract_article_link(
    item: Mapping[str, Any], fallback: LinkFallback[_Link]
) -> str | _Link: ...


def extract_article_link(
    item: Mapping[str, Any], fallback: LinkFallback[_Link] | None = None
) -> str | _Link | None:
    """Return the URL a Yahoo news item links to.

    The URL is not checked: a non-http one is returned as it is, and the scraper
    refuses what it cannot fetch.

    Args:
        item: A feed or search item, flat or with its fields under ``content``.
        fallback: Consulted only when the content carries no ``canonicalUrl``
            object (a string, ``None`` or no key at all), with the item and its
            content; its answer is returned unchanged.

    Returns:
        The canonical object's ``url``, or ``""`` when the object has none.
        Otherwise the fallback's answer, or ``None`` without a fallback.
    """
    content = item.get("content", item)
    canonical = content.get("canonicalUrl")
    if isinstance(canonical, dict):
        return canonical.get("url", "")
    return None if fallback is None else fallback(item, content)


def content_link(item: Mapping[str, Any], content: Mapping[str, Any]) -> str:
    """Read ``link`` from an item's content, ``""`` when it has none.

    The country and macro news paths use this fallback. They ignore a top-level
    ``link`` beside a nested ``content``, and they store ``""`` for a missing link
    because the macro path hashes it into the article's id.

    Args:
        item: The news item; not consulted.
        content: The item's content, or the item itself when it is flat.

    Returns:
        The content's ``link``, or ``""``.
    """
    return content.get("link", "")
