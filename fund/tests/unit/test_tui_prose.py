"""Prose the cockpit shows is laid out so it can be read, without changing its words."""

from __future__ import annotations

from fund.tui.prose import format_answer, format_ask_log, format_market_digest

DIGEST = (
    "Market digest for 2026-11-06 (US). "
    "Macro: 3 series — BAA10Y 1.57 as of 2026-11-05 (delta 0.05), "
    "DEULOLITOAASTSAM 101.00657438227 as of 2026-09-01 (delta 0.103212), "
    "VIXCLS 28.18 as of 2026-11-05 (delta -11.57). "
    "Markets: US: GC=F, ^DJI, ^GSPC; EU: ^STOXX50E. "
    "News themes: business_cycle (13), commodity_inflation (3)."
)


def test_a_market_digest_reads_one_series_per_line() -> None:
    """Three macro series become three short lines, not one wrapped paragraph."""
    lines = format_market_digest(DIGEST)

    assert lines is not None
    series = [
        line for line in lines if line.lstrip().startswith(("BAA10Y", "DEU", "VIX"))
    ]
    assert len(series) == 3
    assert max(len(line) for line in lines) <= 90


def test_a_digest_series_keeps_its_value_date_and_signed_change() -> None:
    """Every figure of the source line survives, with the change signed."""
    lines = format_market_digest(DIGEST)

    assert lines is not None
    baa = next(line for line in lines if "BAA10Y" in line)
    assert "1.57" in baa
    assert "as of 2026-11-05" in baa
    assert "Δ +0.05" in baa
    vix = next(line for line in lines if "VIXCLS" in line)
    assert "Δ -11.57" in vix


def test_a_long_series_value_is_rounded_to_four_decimals() -> None:
    """101.00657438227 reads 101.0066; nothing a reader acts on is lost."""
    lines = format_market_digest(DIGEST)

    assert lines is not None
    deu = next(line for line in lines if "DEULOLITOAASTSAM" in line)
    assert "101.0066" in deu
    assert "101.00657" not in deu


def test_the_digest_columns_line_up() -> None:
    """The date and the change start at the same place on every series line."""
    lines = format_market_digest(DIGEST)

    assert lines is not None
    series = [line for line in lines if "as of" in line]
    assert len(series) == 3
    assert len({line.index("as of") for line in series}) == 1
    assert len({line.index("Δ") for line in series}) == 1


def test_markets_and_news_themes_get_their_own_lines() -> None:
    """Each market and the theme counts are shown on separate, labelled lines."""
    lines = format_market_digest(DIGEST)

    assert lines is not None
    text = "\n".join(lines)
    assert "US: GC=F, ^DJI, ^GSPC" in text
    assert "EU: ^STOXX50E" in text
    assert "business_cycle 13" in text
    assert "commodity_inflation 3" in text


def test_a_long_theme_list_wraps_between_themes() -> None:
    """Six themes do not make one screen-wide line, and none is cut in two."""
    names = ("business_cycle", "commodity_inflation", "credit_conditions")
    names += ("growth_indicators", "sector_rotation", "volatility_risk")
    themes = ", ".join(f"{name} ({i + 1})" for i, name in enumerate(names))
    digest = DIGEST.replace("business_cycle (13), commodity_inflation (3)", themes)

    lines = format_market_digest(digest)

    assert lines is not None
    start = lines.index("  NEWS THEMES") + 1
    rows = lines[start:]
    assert len(rows) >= 2
    assert max(len(row) for row in rows) <= 80
    for i, name in enumerate(names):
        assert any(f"{name} {i + 1}" in row for row in rows)


def test_the_digest_header_names_the_day_and_region() -> None:
    """The first line says which day and which market the digest is about."""
    lines = format_market_digest(DIGEST)

    assert lines is not None
    assert "2026-11-06" in lines[0]
    assert "US" in lines[0]


def test_text_that_is_not_a_digest_is_not_touched() -> None:
    """A journal note or a model sentence gets no digest layout."""
    assert format_market_digest("CPI < expected") is None
    assert format_market_digest("rebalance ✓") is None
    assert format_market_digest("") is None


def test_a_digest_with_an_unreadable_series_is_left_as_written() -> None:
    """If one entry does not parse, the whole line stays as the source wrote it."""
    broken = DIGEST.replace("BAA10Y 1.57 as of 2026-11-05 (delta 0.05)", "BAA10Y")

    assert format_market_digest(broken) is None


def test_a_digest_without_macro_data_still_reads() -> None:
    """The no-data sentences the builder writes are shown as they are."""
    digest = (
        "Market digest for 2026-11-06 (US). Macro: no series available. "
        "Markets: no summaries available. News: no recent articles."
    )

    lines = format_market_digest(digest)

    assert lines is not None
    text = "\n".join(lines)
    assert "no series available" in text
    assert "no summaries available" in text
    assert "no recent articles" in text


def test_an_answer_with_no_formatting_marks_is_unchanged() -> None:
    """Plain prose, brackets included, reaches the reader word for word."""
    answer = "Holding 3 names [AAPL, MSFT]. Run is ok — see [/policy]."

    assert format_answer(answer).plain == answer


def test_bold_marks_become_bold_text() -> None:
    """``**net worth**`` is shown as bold text, without the asterisks."""
    content = format_answer("Your **net worth** is up")

    assert content.plain == "Your net worth is up"
    assert any("bold" in str(span.style) for span in content.spans)


def test_a_heading_loses_its_hash_marks_and_is_bold() -> None:
    """``## Summary`` reads as a bold line, not as literal hash marks."""
    content = format_answer("## Summary\nAll good")

    assert content.plain == "Summary\nAll good"
    assert any("bold" in str(span.style) for span in content.spans)


def test_dash_lists_become_bullets() -> None:
    """``- a`` and ``* b`` are laid out as bullets, keeping their indent."""
    content = format_answer("- first\n  - nested\n* second")

    assert content.plain == "• first\n  • nested\n• second"


def test_a_numbered_list_is_left_as_written() -> None:
    """``1. x`` already reads fine in a terminal."""
    assert format_answer("1. buy\n2. hold").plain == "1. buy\n2. hold"


def test_an_unmatched_bold_mark_stays_literal() -> None:
    """A stray ``**`` is not guessed at."""
    assert format_answer("5 ** 2 equals 25").plain == "5 ** 2 equals 25"


def test_inline_code_marks_are_dropped_but_the_code_is_kept() -> None:
    """``run`` shows as run, styled, with no backticks."""
    content = format_answer("use `fund status` to check")

    assert content.plain == "use fund status to check"
    assert content.spans


def test_a_code_block_keeps_its_lines_untouched() -> None:
    """Inside a fenced block, ``- x`` and ``**y**`` are code, not formatting."""
    content = format_answer("run:\n```\n- keep **this**\n```\ndone")

    assert content.plain == "run:\n- keep **this**\ndone"


def test_an_empty_answer_renders_as_empty_text() -> None:
    """No text in, no text out, and no error."""
    assert format_answer("").plain == ""


def test_a_rule_line_becomes_a_visible_divider() -> None:
    """``---`` between sections is drawn as a line, not three dashes."""
    plain = format_answer("one\n---\ntwo").plain

    assert plain.splitlines()[0] == "one"
    assert plain.splitlines()[2] == "two"
    assert set(plain.splitlines()[1]) == {"─"}


def test_the_conversation_has_a_blank_line_between_turns() -> None:
    """A question and its answer no longer run together."""
    content = format_ask_log([("you", "how are we doing?"), ("ask", "all good")])

    assert content.plain == "you: how are we doing?\n\nask: all good"


def test_a_single_plain_turn_reads_exactly_as_before() -> None:
    """One plain answer is still ``ask: <answer>``, nothing added."""
    assert format_ask_log([("ask", "fine")]).plain == "ask: fine"


def test_the_speaker_labels_are_bold() -> None:
    """Who spoke stands out from what was said."""
    content = format_ask_log([("you", "hi"), ("ask", "hello")])

    bold_text = [
        content.plain[span.start : span.end]
        for span in content.spans
        if "bold" in str(span.style)
    ]
    assert "you" in bold_text
    assert "ask" in bold_text


def test_your_own_question_is_never_reformatted() -> None:
    """Only the model's answer is read as formatted text, not what you typed."""
    content = format_ask_log([("you", "what is **this** and # that")])

    assert content.plain == "you: what is **this** and # that"


def test_terminal_control_sequences_in_an_answer_are_removed() -> None:
    """Escape codes a model echoes from untrusted text never reach the terminal."""
    hostile = "ok \x1b]52;c;QUFB\x07 \x1b]0;pwned\x07 \x1b[2J \x9b31m done"

    plain = format_answer(hostile).plain

    assert "\x1b" not in plain
    assert "\x9b" not in plain
    assert "\x07" not in plain
    assert plain.startswith("ok ")
    assert plain.endswith(" done")


def test_your_own_typed_text_is_stripped_of_control_codes_too() -> None:
    """What you typed is shown as typed, minus anything a terminal would obey."""
    content = format_ask_log([("you", "hello \x1b[2J there")])

    assert content.plain == "you: hello [2J there"


def test_newlines_and_tabs_are_kept_when_control_codes_are_removed() -> None:
    """Layout characters survive; only terminal-affecting ones go."""
    assert format_answer("a\tb\nc").plain == "a\tb\nc"


def test_windows_line_endings_are_shown_as_plain_line_breaks() -> None:
    """A model's CRLF text reads as ordinary lines (the carriage return is dropped)."""
    assert format_answer("one\r\ntwo\r\n").plain == "one\ntwo\n"


def test_trailing_whitespace_and_blank_lines_are_kept() -> None:
    """Plain text with no marks comes back exactly, paragraph breaks included."""
    text = "first paragraph  \n\nsecond paragraph\n"

    assert format_answer(text).plain == text


def test_a_very_long_message_is_cut_with_a_marker() -> None:
    """A message past the size limit is shortened, not rendered in full."""
    plain = format_answer("x" * 100_000).plain

    assert len(plain) < 25_000
    assert "more characters" in plain


def test_a_heading_followed_by_endless_spaces_renders_in_a_blink() -> None:
    """A degenerate model line cannot make the regex take seconds."""
    import time

    started = time.monotonic()
    format_answer("# a" + " " * 4000 + "b")

    assert time.monotonic() - started < 0.5


def test_thousands_of_unclosed_bold_marks_render_in_a_blink() -> None:
    """Unbalanced ``**`` runs stay linear, not quadratic."""
    import time

    started = time.monotonic()
    format_answer("**a " * 20_000)

    assert time.monotonic() - started < 1.5


def test_a_heading_with_bold_text_and_closing_hashes_is_clean() -> None:
    """``## **x** ##`` reads as a bold ``x``."""
    content = format_answer("## **x** ##")

    assert content.plain == "x"
    assert any("bold" in str(span.style) for span in content.spans)


def test_a_tiny_series_value_is_shown_as_written() -> None:
    """Rounding to four decimals never turns a real value into zero."""
    digest = DIGEST.replace("BAA10Y 1.57", "BAA10Y 0.00001234")

    lines = format_market_digest(digest)

    assert lines is not None
    assert any("0.00001234" in line for line in lines)


def test_a_digest_with_text_before_the_sections_is_left_as_written() -> None:
    """Unexpected extra text means the layout cannot be trusted, so none is applied."""
    digest = DIGEST.replace("(US). ", "(US). Preamble nobody expected. ")

    assert format_market_digest(digest) is None


def test_a_gigantic_digest_is_not_parsed() -> None:
    """A huge machine line is returned unchanged quickly instead of being scanned."""
    import time

    started = time.monotonic()
    huge = "Market digest for 2026-11-06 (US). " + "Macro: " * 8000
    result = format_market_digest(huge)

    assert result is None
    assert time.monotonic() - started < 0.5


def test_a_news_line_that_is_not_a_theme_list_is_shown_as_written() -> None:
    """Whatever the builder writes after ``News themes:`` is never dropped."""
    digest = DIGEST.replace("business_cycle (13), commodity_inflation (3)", "odd text")

    lines = format_market_digest(digest)

    assert lines is not None
    assert any(line.strip() == "odd text" for line in lines)


def test_inline_code_is_underlined() -> None:
    """Code is set apart with an underline, not just stripped of its backticks."""
    content = format_answer("run `fund status` now")

    assert any("underline" in str(span.style) for span in content.spans)


def test_the_series_value_column_ends_where_the_date_column_starts() -> None:
    """Values are right-aligned so the figures line up under each other."""
    lines = format_market_digest(DIGEST)

    assert lines is not None
    series = [line for line in lines if "as of" in line]
    ends = {line.index("as of") for line in series}
    assert len(ends) == 1
