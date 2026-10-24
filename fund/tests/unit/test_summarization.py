"""Contract tests for ``fund.agents.summarization.summarization_override``.

The override must change only what the operator pinned (the trigger and how much
recent history survives); it must not silently alter how the summary itself is built.
deepagents' own factory passes ``trim_tokens_to_summarize=None`` (no trimming), while
the raw ``SummarizationMiddleware`` constructor defaults to a 4000-token trim, which
would summarise only the tail of the evicted history.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from _fund_fakes import ScriptedFundModel
from deepagents.backends import StateBackend

from fund.agents.summarization import summarization_override
from fund.config import settings


def _profiled_model() -> ScriptedFundModel:
    """Build a model whose profile exposes a context window, like hosted models."""
    return ScriptedFundModel(profile={"max_input_tokens": 128_000})


def _pinned() -> Any:
    """Return a config with the operator's summary trigger pinned."""
    return dataclasses.replace(
        settings, summary_token_threshold=24000, summary_messages_to_keep=4
    )


def _build(model: Any) -> Any:
    """Build the single override middleware for ``model`` under the pinned config."""
    (middleware,) = summarization_override(model, StateBackend(), _pinned())
    return middleware


def test_no_threshold_means_no_override() -> None:
    """Without a pinned threshold deepagents' model-aware default stays in force."""
    config = dataclasses.replace(settings, summary_token_threshold=None)

    assert summarization_override(ScriptedFundModel(), StateBackend(), config) == []


def test_override_pins_the_trigger_and_the_kept_messages() -> None:
    """The operator's trigger and keep settings reach the middleware verbatim."""
    middleware = _build(ScriptedFundModel())

    assert middleware._lc_helper.trigger == ("tokens", 24000)
    assert middleware._lc_helper.keep == ("messages", 4)


def test_override_does_not_trim_the_history_it_summarises() -> None:
    """The summary input is untrimmed, matching deepagents' own factory."""
    # A pinned 24k trigger summarising only the last ~4k tokens would drop the
    # earlier economist/allocator/risk outputs from the PM's summary.
    assert _build(ScriptedFundModel())._lc_helper.trim_tokens_to_summarize is None


def test_arg_truncation_fires_at_the_pinned_trigger_for_a_profiled_model() -> None:
    """A profiled model's fraction-based truncation trigger is rebased on the pin."""
    # deepagents' fraction-based truncation trigger (0.85 of the window) sits far
    # above a pinned token trigger and would never fire before compaction does.
    middleware = _build(_profiled_model())

    assert middleware._truncate_args_trigger == ("tokens", 24000)


def test_arg_truncation_keeps_deepagents_default_for_an_unprofiled_model() -> None:
    """Without a profile deepagents' message-count truncation trigger is untouched."""
    middleware = _build(ScriptedFundModel())

    assert middleware._truncate_args_trigger == ("messages", 20)
