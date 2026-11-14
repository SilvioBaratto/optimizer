"""The base class every ``ShellApp`` mixin declares.

At runtime it is ``object``, so each mixin stays a plain class and ``ShellApp``
supplies Textual's ``App`` itself. A type checker sees ``App[None]`` instead, so
the Textual API a mixin calls on ``self`` is still checked now that the methods
live outside ``ShellApp``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App

    ShellHost = App[None]
else:
    ShellHost = object
