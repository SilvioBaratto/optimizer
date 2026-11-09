"""Make untrusted text safe to show in the cockpit and to keep.

Text a model writes can echo whatever it read: a news headline, a file, a database row.
Textual does not parse it as markup here, but a raw escape sequence in it is written to
the terminal as it is, where it can retitle the window, write to the clipboard, clear
the screen or fake what is shown. And an answer of any length is rendered and saved, so
a runaway one slows every redraw. These helpers close both, for everything the Ask
conversation shows, saves or exports.

Stdlib only, like the rest of ``fund.tui`` that must stay agent-stack-free.
"""

from __future__ import annotations

import re

__all__ = [
    "MAX_MESSAGE_CHARS",
    "clip",
    "defang_markdown",
    "describe_error",
    "strip_control",
]

# Every C0 and C1 control character except the line feed and the tab, which are layout.
# That includes ESC (every escape sequence starts with it), the single-character C1
# introducers (0x9b is a CSI), BEL, and the carriage return, so CRLF text reads as
# plain lines.
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")
# Far more than any answer needs; the point is a bound, not a style.
MAX_MESSAGE_CHARS = 20_000
# An error shown to the adviser or the model: one line, never a wall of text.
_ERROR_CHARS = 300
_IMAGE = re.compile(r"!\[(?P<alt>[^\]\n]*)\]\([^)\n]*\)")


def strip_control(text: str) -> str:
    """Remove terminal control characters, keeping line feeds and tabs."""
    return _CONTROL.sub("", text)


def clip(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Shorten ``text`` to ``limit`` characters, saying how much was cut."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n… ({len(text) - limit} more characters not shown)"


def defang_markdown(text: str) -> str:
    """Replace embedded markdown images by their description.

    An exported conversation is opened in a markdown previewer, and a model that was
    steered by untrusted text can write ``![](https://host/?d=<data>)``: previewing the
    file would then fetch that address. Ordinary links are left as they are.
    """
    return _IMAGE.sub(lambda m: f"[image: {m['alt']}]" if m["alt"] else "[image]", text)


def describe_error(exc: BaseException) -> str:
    """One safe line describing an error, for the screen, the model and the status bar.

    A driver error's own text embeds the SQL statement and its bound parameters, and a
    provider error may echo request details, so only the first line is kept, control
    characters are removed and the length is capped.
    """
    from portopt_db.engine import summarize_db_error

    text = strip_control(summarize_db_error(exc)).strip()
    return clip(text or type(exc).__name__, _ERROR_CHARS)
