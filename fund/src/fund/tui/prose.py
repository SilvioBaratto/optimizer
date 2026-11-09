"""Layout for the prose the cockpit shows: market digests and the model's answers.

Both arrive as text written for a program or a model, not for a reader: the market
digest is one deterministic line of about 150 words, and an answer carries markdown
marks. These pure functions lay them out for the screen without changing a word, and
fall back to the original text whenever it does not have the expected shape.

Import-light on purpose (stdlib and ``textual.content`` only), like the rest of
``fund.tui``: nothing here may pull the agent stack or the optimizer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from textual.content import Content

from fund.tui.safe_text import clip, strip_control

_DIGEST = re.compile(
    r"^Market digest for (?P<day>\d{4}-\d{2}-\d{2}) \((?P<region>[^)]*)\)\. "
    r"(?P<body>.+)$",
    re.DOTALL,
)
# A section ends at the first period that is followed by the next section's name (or
# the end), so the decimal points inside the series values never end one early.
_SECTION = re.compile(
    r"(?P<name>Macro|Markets|News themes|News): (?P<text>.*?)\."
    r"(?= (?:Macro|Markets|News themes|News): |$)",
    re.DOTALL,
)
_MACRO_BODY = re.compile(r"^(?P<count>\d+) series — (?P<series>.+)$", re.DOTALL)
_SERIES = re.compile(
    r"^(?P<sid>\S+) (?P<latest>\S+)"
    r"(?: as of (?P<day>\d{4}-\d{2}-\d{2}))?"
    r"(?: \(delta (?P<delta>\S+)\))?$"
)
_THEME = re.compile(r"^(?P<name>\S+) \((?P<count>\d+)\)$")
_AS_OF_WIDTH = len("as of 0000-00-00")
_THEME_JOIN = " · "
_THEMES_WIDTH = 72

_RULE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>.*)$")
_BULLET = re.compile(r"^(?P<indent>\s*)[-*+]\s+(?P<text>.*)$")
_FENCE = re.compile(r"^\s*```")
_DIVIDER = "─" * 24
# How far past an opening ``**`` or backtick a closer is looked for. A bound, so a model
# line of thousands of unclosed marks costs a fixed amount per mark, not the rest of the
# line.
_MARK_WINDOW = 1_000
# A machine-written digest is a few kilobytes; text far beyond that is not one.
_MAX_DIGEST_CHARS = 20_000


def _number(text: str) -> str:
    """Round a figure to four decimals and drop trailing zeros; keep other text."""
    try:
        value = float(text)
    except ValueError:
        return text
    if value != 0 and round(value, 4) == 0:
        return text
    shown = f"{round(value, 4):.4f}".rstrip("0").rstrip(".")
    return "0" if shown in ("", "-0") else shown


def _signed(text: str) -> str:
    """Like :func:`_number`, with an explicit ``+`` on a positive change."""
    shown = _number(text)
    return f"+{shown}" if shown[:1].isdigit() and shown != "0" else shown


def _macro_lines(count: str, series: str) -> list[str] | None:
    """One aligned line per macro series, or ``None`` if any entry is unreadable."""
    rows = []
    for entry in series.split(", "):
        match = _SERIES.match(entry)
        if match is None:
            return None
        rows.append(match)
    sid_width = max(len(row["sid"]) for row in rows)
    values = [_number(row["latest"]) for row in rows]
    value_width = max(len(value) for value in values)
    lines = [f"  MACRO · {count} series"]
    for row, value in zip(rows, values, strict=True):
        as_of = f"as of {row['day']}" if row["day"] else ""
        change = f"  Δ {_signed(row['delta'])}" if row["delta"] is not None else ""
        lines.append(
            f"    {row['sid']:<{sid_width}}  {value:>{value_width}}  "
            f"{as_of:<{_AS_OF_WIDTH}}{change}".rstrip()
        )
    return lines


def _themes_lines(text: str) -> list[str]:
    """``name (n), name (n)`` as ``name n · name n`` packed into short lines.

    Anything that is not a theme list is returned as one line, as written.
    """
    matches = [_THEME.match(part) for part in text.split(", ")]
    if any(match is None for match in matches):
        return [text]
    lines: list[str] = []
    current = ""
    for match in matches:
        if match is None:
            continue
        item = f"{match['name']} {match['count']}"
        if current and len(current) + len(_THEME_JOIN) + len(item) > _THEMES_WIDTH:
            lines.append(current)
            current = item
        else:
            current = f"{current}{_THEME_JOIN}{item}" if current else item
    lines.append(current)
    return lines


def format_market_digest(narrative: str) -> list[str] | None:
    """Lay the one-line market digest out as short labelled lines.

    Args:
        narrative: A ``market_journal`` narrative as the ingestion step writes it:
            ``Market digest for <day> (<region>). Macro: ... Markets: ... News ...``.

    Returns:
        The digest as lines (a header, then Macro with one aligned series per line,
        Markets with one market per line, and the news themes), or ``None`` when the
        text is not a digest of exactly that shape, so the caller shows it unchanged.
    """
    if len(narrative) > _MAX_DIGEST_CHARS:
        return None
    head = _DIGEST.match(narrative.strip())
    if head is None:
        return None
    body = head["body"]
    sections = list(_SECTION.finditer(body))
    rebuilt = " ".join(f"{m['name']}: {m['text']}." for m in sections)
    if not sections or rebuilt != body:
        return None

    lines = [f"Market digest · {head['day']} ({head['region']})"]
    for section in sections:
        name, text = section["name"], section["text"]
        if name == "Macro":
            macro = _MACRO_BODY.match(text)
            if macro is None:
                lines.append(f"  MACRO · {text}")
                continue
            macro_lines = _macro_lines(macro["count"], macro["series"])
            if macro_lines is None:
                return None
            lines.extend(macro_lines)
        elif name == "Markets":
            if text.startswith("no "):
                lines.append(f"  MARKETS · {text}")
                continue
            lines.append("  MARKETS")
            lines.extend(f"    {market}" for market in text.split("; "))
        elif name == "News themes":
            lines.append("  NEWS THEMES")
            lines.extend(f"    {row}" for row in _themes_lines(text))
        else:
            lines.append(f"  NEWS · {text}")
    return lines


def _bold_end(text: str, start: int) -> int:
    """Index of the ``**`` closing the bold run opened at ``start``, or ``-1``.

    The run needs a non-space right after the opener and right before the closer, and
    the closer is searched for within :data:`_MARK_WINDOW` characters.
    """
    if text[start + 2 : start + 3].strip() == "":
        return -1
    limit = min(len(text), start + 2 + _MARK_WINDOW)
    closer = text.find("**", start + 3, limit)
    while closer != -1:
        if not text[closer - 1].isspace():
            return closer
        closer = text.find("**", closer + 1, limit)
    return -1


def _inline_parts(text: str) -> list[str | tuple[str, str]]:
    """Split one line into plain text, bold runs and code runs, in a single pass."""
    parts: list[str | tuple[str, str]] = []
    plain_from = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "*" and text.startswith("**", index):
            closer = _bold_end(text, index)
            if closer != -1:
                if index > plain_from:
                    parts.append(text[plain_from:index])
                parts.append((text[index + 2 : closer], "bold"))
                index = plain_from = closer + 2
                continue
        elif char == "`":
            closer = text.find("`", index + 1, index + 1 + _MARK_WINDOW)
            if closer > index + 1:
                if index > plain_from:
                    parts.append(text[plain_from:index])
                parts.append((text[index + 1 : closer], "underline"))
                index = plain_from = closer + 1
                continue
        index += 1
    if plain_from < len(text):
        parts.append(text[plain_from:])
    return parts


def _plain(parts: list[str | tuple[str, str]]) -> str:
    """The text of a list of inline parts, without their styles."""
    return "".join(part if isinstance(part, str) else part[0] for part in parts)


def format_answer(text: str) -> Content:
    """Render a model answer's light markdown as styled text.

    Headings become bold lines, ``**x**`` bold text, dash and star lists bullets,
    ``---`` a divider and `` `x` `` underlined text; the marks themselves are dropped.
    Anything else, brackets and a stray ``**`` included, is shown exactly as written,
    and the text is never read as Textual markup.

    Args:
        text: The answer as the model wrote it.

    Returns:
        The styled content; its ``plain`` text equals ``text`` when the answer holds no
        formatting marks, no terminal control characters (removed, as is the carriage
        return of a Windows line ending) and is within the message size limit (longer
        text is cut with a note).
    """
    parts: list[str | tuple[str, str] | Content] = []
    in_code = False
    emitted = False
    for line in clip(strip_control(text)).split("\n"):
        if _FENCE.match(line):
            in_code = not in_code
            continue
        if emitted:
            parts.append("\n")
        emitted = True
        if in_code:
            parts.append(line)
            continue
        heading = _HEADING.match(line)
        bullet = _BULLET.match(line)
        if _RULE.match(line):
            parts.append(_DIVIDER)
        elif heading is not None:
            title = heading["text"].rstrip().rstrip("#").rstrip()
            parts.append((_plain(_inline_parts(title)), "bold"))
        elif bullet is not None:
            parts.append(f"{bullet['indent']}• ")
            parts.extend(_inline_parts(bullet["text"]))
        else:
            parts.extend(_inline_parts(line))
    return Content.assemble(*parts)


def format_ask_log(turns: Sequence[tuple[str, str]]) -> Content:
    """Lay the Ask conversation out with a bold speaker and a blank line per turn.

    Args:
        turns: ``(speaker, text)`` pairs, oldest first. Only the ``ask`` speaker's
            text is read as formatted; what the adviser typed is shown as typed.

    Returns:
        The whole log as one styled content.
    """
    parts: list[str | tuple[str, str] | Content] = []
    for index, (speaker, text) in enumerate(turns):
        if index:
            parts.append("\n\n")
        parts.append((strip_control(speaker), "bold"))
        parts.append(": ")
        parts.append(
            format_answer(text) if speaker == "ask" else clip(strip_control(str(text)))
        )
    return Content.assemble(*parts)
