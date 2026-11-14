"""The base class every ``ShellApp`` mixin declares.

At runtime it is ``object``, so each mixin is a plain class and ``ShellApp``
supplies Textual's ``App`` itself. A type checker sees ``App[None]`` plus the
``ShellApp`` state and the methods the mixins call on one another, so a mixin's
use of ``self`` is checked against what ``ShellApp`` and the other mixins define.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as dt
    import threading
    import uuid
    from collections.abc import Callable
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import AbstractContextManager
    from typing import Any

    from sqlalchemy.orm import Session
    from textual.app import App
    from textual.worker import Worker

    from fund.config import FundConfig
    from fund.tui._mixins.attention import _AttentionKey
    from fund.tui._mixins.drive import _QueuedDirective
    from fund.tui.views import DeliberationView

    class ShellHost(App[None]):
        """The ``ShellApp`` state and cross-mixin methods, declared for the checker."""

        _portfolio_id: uuid.UUID | None
        _session_factory: Callable[[], AbstractContextManager[Session]]
        _persistence: Any
        _model_factory: Callable[[], Any]
        _fallback_factory: Callable[[], Any] | None
        _config: FundConfig
        last_status: str
        last_worker: Worker[None] | None
        _portfolio_ids: list[uuid.UUID]
        _runstate: str
        _streaming: bool
        _stream_resume: threading.Event
        _round_lock: threading.Lock
        _ask_lock: threading.Lock
        _mandate_lock: threading.Lock
        _ask_pending: int
        _ask_clearing: bool
        _ask_history_retry_at: float
        _ask_io: ThreadPoolExecutor
        _queued_directive: _QueuedDirective | None
        _seen_attention: set[_AttentionKey]
        _awaiting_count: int
        _attention_degraded: bool
        _logged_attention_failures: set[str]
        last_attention: list[str]
        _attention_read_inflight: bool

        def _refresh(self) -> None: ...
        def _set_status(self, message: str) -> None: ...
        def _after_action(self, message: str) -> None: ...
        def _build_fallback(self) -> Any | None: ...
        def _populate_picker(self) -> None: ...
        def _refresh_runstate(self) -> None: ...
        def _run_state_glyph(self) -> str: ...
        def _short_label(self, portfolio_id: uuid.UUID) -> str: ...
        def _ask_approve_gate(self, run_id: uuid.UUID) -> str | None: ...
        def _launch_run(
            self,
            portfolio_id: uuid.UUID,
            asof: dt.date | None,
            view: DeliberationView | None = None,
        ) -> Worker[None] | None: ...

else:
    ShellHost = object
