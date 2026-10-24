"""Operator-tunable context bounding for the deepagents summarization middleware.

deepagents compacts an agent's transcript once it crosses a built-in trigger: a
fraction of the model's context window when the model profile exposes one, else a
fixed 170000 tokens. A small-context model (a local 32k Ollama model, a capped Groq
tier) overflows long before that fixed trigger, so ``FUND_SUMMARY_TOKEN_THRESHOLD``
and ``FUND_SUMMARY_MESSAGES_TO_KEEP`` let the operator replace it. Left unset, the
built-in model-aware default stays in force.

The deepagents stack is imported inside the builder so a bare import of this module
stays agent-stack-free (mirrors :mod:`fund.agents.graph`).
"""

from __future__ import annotations

from typing import Any

from fund.config import FundConfig

__all__ = ["summarization_override"]


def summarization_override(model: Any, backend: Any, config: FundConfig) -> list[Any]:
    """Build the summarization middleware override for one agent, if configured.

    Each agent needs its own instance: deepagents compiles the PM, every declarative
    subagent and the general-purpose subagent with independent middleware stacks, and
    a declarative subagent inherits none of its parent's middleware.

    Args:
        model: The chat model the agent runs on; it also writes the summaries.
        backend: The agent's backend, where evicted history is offloaded.
        config: Run config; ``summary_token_threshold`` of ``None`` means no override.

    Returns:
        ``[]`` when no threshold is configured (deepagents keeps its model-aware
        default), else a single ``SummarizationMiddleware`` that replaces the default
        one by name. Splat it into the agent's ``middleware``.
    """
    if config.summary_token_threshold is None:
        return []

    from deepagents.middleware.summarization import (
        SummarizationMiddleware,
        TruncateArgsSettings,
        compute_summarization_defaults,
    )
    from langchain.agents.middleware.summarization import ContextSize

    trigger: ContextSize = ("tokens", config.summary_token_threshold)
    truncate_args: TruncateArgsSettings = compute_summarization_defaults(model)[
        "truncate_args_settings"
    ]
    # deepagents clips large tool-call arguments at the same fraction of the window
    # that triggers compaction. A pinned token trigger sits far below that fraction,
    # so the inherited one would never fire before compaction does.
    current = truncate_args.get("trigger")
    if current is not None and current[0] == "fraction":
        truncate_args = {**truncate_args, "trigger": trigger}
    return [
        SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=trigger,
            keep=("messages", config.summary_messages_to_keep),
            # The raw constructor defaults to a 4000-token trim, which would
            # summarise only the tail of the evicted history; deepagents' own
            # factory passes None.
            trim_tokens_to_summarize=None,
            truncate_args_settings=truncate_args,
        )
    ]
