"""T3.0 — the shared tool envelope: ``{ok,data}`` / ``{ok:false,error}`` contract.

The envelope is the load-bearing invariant of the whole Fase-3 backbone: a tool
NEVER raises across its boundary, and never hands a raw DataFrame back. These
tests pin both, plus the small ``ok``/``err``/``summarize_frame`` helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fund.tools._base import err, ok, session_scope, summarize_frame, tool_envelope


class TestResultConstructors:
    def test_ok_wraps_payload(self) -> None:
        assert ok({"weights": [0.5, 0.5]}) == {
            "ok": True,
            "data": {"weights": [0.5, 0.5]},
        }

    def test_err_carries_message(self) -> None:
        assert err("unknown ticker XYZ") == {
            "ok": False,
            "error": "unknown ticker XYZ",
        }


class TestToolEnvelope:
    def test_plain_return_is_wrapped_in_ok(self) -> None:
        @tool_envelope
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == {"ok": True, "data": 5}

    def test_existing_ok_dict_passes_through_unchanged(self) -> None:
        @tool_envelope
        def already_enveloped() -> dict[str, object]:
            return {"ok": True, "data": {"n": 1}}

        assert already_enveloped() == {"ok": True, "data": {"n": 1}}

    def test_existing_error_dict_passes_through(self) -> None:
        @tool_envelope
        def guarded() -> dict[str, object]:
            return err("bad column")

        assert guarded() == {"ok": False, "error": "bad column"}

    def test_exception_becomes_error_envelope_never_raises(self) -> None:
        @tool_envelope
        def boom() -> None:
            raise ValueError("kaboom")

        result = boom()
        assert result["ok"] is False
        assert "kaboom" in result["error"]
        assert "ValueError" in result["error"]

    def test_mapping_without_ok_key_is_wrapped(self) -> None:
        # A dict return that isn't already an envelope (no "ok" key) is wrapped by
        # ok(), so it lands under data rather than passing through verbatim.
        @tool_envelope
        def payload() -> dict[str, int]:
            return {"n": 1}

        assert payload() == {"ok": True, "data": {"n": 1}}

    def test_preserves_wrapped_function_identity(self) -> None:
        @tool_envelope
        def named_tool() -> int:
            """A docstring worth keeping."""
            return 1

        assert named_tool.__name__ == "named_tool"
        assert named_tool.__doc__ == "A docstring worth keeping."


class TestSummarizeFrame:
    def test_reports_shape_columns_and_coverage_not_raw_frame(self) -> None:
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {"AAA": [0.1, 0.2, np.nan], "BBB": [0.0, 0.0, 0.0]}, index=idx
        )

        summary = summarize_frame(df, name="returns")

        assert summary["name"] == "returns"
        assert summary["rows"] == 3
        assert summary["cols"] == 2
        assert summary["columns"] == ["AAA", "BBB"]
        assert summary["na_counts"] == {"AAA": 1, "BBB": 0}
        # 1 NaN of 6 cells → coverage 5/6.
        assert summary["coverage"] == 5 / 6
        assert summary["index_start"].startswith("2020-01-01")
        assert summary["index_end"].startswith("2020-01-03")
        # No raw frame leaked into the summary.
        assert not any(isinstance(v, pd.DataFrame) for v in summary.values())

    def test_empty_frame_is_serialisable(self) -> None:
        summary = summarize_frame(pd.DataFrame(), name="empty")
        assert summary["rows"] == 0
        assert summary["cols"] == 0
        assert summary["columns"] == []

    def test_non_datetime_index_omits_span(self) -> None:
        # A frame without a DatetimeIndex still summarises, but reports no span.
        df = pd.DataFrame({"AAA": [0.1, 0.2]}, index=["a", "b"])

        summary = summarize_frame(df, name="plain")

        assert summary["rows"] == 2
        assert "index_start" not in summary
        assert "index_end" not in summary

    def test_default_name_is_frame(self) -> None:
        summary = summarize_frame(pd.DataFrame({"AAA": [1.0]}))

        assert summary["name"] == "frame"


class TestSessionScope:
    def test_yields_manager_session_and_closes(self) -> None:
        events: list[str] = []

        class _FakeSession:
            def close(self) -> None:
                events.append("close")

        class _FakeManager:
            def __init__(self) -> None:
                self._session = _FakeSession()

            def get_session(self):
                from contextlib import contextmanager

                @contextmanager
                def _cm():
                    events.append("open")
                    try:
                        yield self._session
                    finally:
                        self._session.close()

                return _cm()

        manager = _FakeManager()
        with session_scope(manager) as session:  # type: ignore[arg-type]
            events.append("use")
            assert isinstance(session, _FakeSession)

        assert events == ["open", "use", "close"]
