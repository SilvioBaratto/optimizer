"""Runtime configuration for the ``fund`` deep-agent bridge.

Pins the Phase-0 verified contract (SPEC.md §8, decisions D1–D4/D10/D22). The
shape mirrors ``portopt_db.config.DbConfig``: a frozen dataclass of primitives
(serialisable, hashable) plus a ``load_config`` factory that sources secrets from
the environment.

Pinned constants are dataclass defaults, so a bare ``import fund.config`` yields
the verified values with no environment set. Secrets (``DATABASE_URL``,
``OLLAMA_API_KEY``, ``FRED_API_KEY``, ``SSL_CERT_FILE``) are read from ``env`` and
stay ``None`` when absent — import must never fail in CI, so *required*-secret
validation happens at use-time (pool/model construction), not here.

Deliberately imports nothing from ``optimizer`` (config must be cheap to import);
the psycopg ``dict_row`` callable is imported lazily inside ``langgraph_pool_kwargs``
so reading config in a test does not drag in the DB driver.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FundConfig:
    """Frozen runtime contract for a fund run (SPEC §8).

    Fields hold only primitives/tuples (serialisable). The two non-primitive
    outputs the runtime needs — the psycopg pool kwargs (carries a callable) and
    the deepagents ``interrupt_on`` map — are built by methods, not stored.
    """

    # --- secrets / env (D4 env vars); None when unset so import never fails ---
    database_url: str | None = None
    ollama_api_key: str | None = None
    fred_api_key: str | None = None
    ssl_cert_file: str | None = None

    # --- D1 concurrency: sync sessions + sync agent.invoke ---
    db_sync: bool = True

    # --- D2 agent backend: virtual_mode + read-only Store preload of theory ---
    agent_virtual_mode: bool = True
    preload_theory: bool = True

    # --- D3 LangGraph persistence: dedicated `langgraph` schema via search_path ---
    langgraph_schema: str = "langgraph"
    pool_autocommit: bool = True
    pool_prepare_threshold: int = 0

    # --- D4 model: DeepSeek on Ollama Cloud, single-provider fallback ---
    ollama_base_url: str = "https://ollama.com"
    primary_model: str = "deepseek-v4.1-flash:cloud"
    fallback_model: str = "deepseek-v4-pro:cloud"
    model_temperature: float = 0.0
    # Force the non-thinking "flash" route: thinking mode rejects tool_choice and
    # deepagents depends on reliable tool-calling (SPEC D4).
    model_reasoning: bool = False
    # `json_schema` fell through in probing; `function_calling` round-tripped.
    structured_output_method: str = "function_calling"

    # --- D22 guards: LOW explicit recursion_limit + PM round cap (library
    # default 9999 is NOT a safety bound). ---
    recursion_limit: int = 50
    max_pm_rounds: int = 10

    # --- D10 HITL: tools gated behind interrupt_on (checkpointer required) ---
    interrupt_on: tuple[str, ...] = field(default=("place_orders",))

    # --- Fase-5 persistence: Store key the active ConstraintSet is cached under
    # (namespace = (portfolio_id,)) so a Phase-4 ``ConstraintSetRef`` resolves. ---
    constraint_set_store_key: str = "constraint_set"

    def langgraph_pool_kwargs(self) -> dict[str, Any]:
        """psycopg ``ConnectionPool(kwargs=...)`` for the LangGraph pool (D3).

        ``.setup()`` needs ``autocommit=True`` (else ``CREATE INDEX CONCURRENTLY``
        fails) and ``row_factory=dict_row``; ``search_path`` redirects all
        unqualified DDL into the dedicated schema so nothing leaks into Alembic's
        ``public``.
        """
        from psycopg.rows import dict_row

        return {
            "options": f"-c search_path={self.langgraph_schema}",
            "autocommit": self.pool_autocommit,
            "row_factory": dict_row,
            "prepare_threshold": self.pool_prepare_threshold,
        }

    def interrupt_on_map(self) -> dict[str, bool]:
        """deepagents ``interrupt_on={tool: True}`` map from the gated-tool tuple."""
        return dict.fromkeys(self.interrupt_on, True)


def load_config(
    *,
    env: Mapping[str, str] | None = None,
    load_dotenv_file: bool = True,
) -> FundConfig:
    """Build a :class:`FundConfig`, sourcing secrets from ``env``.

    Args:
        env: Environment mapping to read secrets from. Defaults to
            ``os.environ``. Inject a dict in tests for determinism.
        load_dotenv_file: When ``True`` and ``env`` is not injected, load a local
            ``.env`` first (no-op if the file is absent).

    Returns:
        A frozen config with the SPEC §8 constants and any secrets present in
        ``env``. Missing secrets stay ``None`` — validated at use-time.
    """
    if env is None:
        if load_dotenv_file:
            from dotenv import load_dotenv

            load_dotenv()
        env = os.environ

    return FundConfig(
        database_url=env.get("DATABASE_URL"),
        ollama_api_key=env.get("OLLAMA_API_KEY"),
        fred_api_key=env.get("FRED_API_KEY"),
        ssl_cert_file=env.get("SSL_CERT_FILE"),
    )


# Module-level singleton: cheap to import, secrets validated when used.
settings: FundConfig = load_config()

__all__ = ["FundConfig", "load_config", "settings"]
