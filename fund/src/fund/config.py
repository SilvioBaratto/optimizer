"""Runtime configuration for the ``fund`` deep-agent bridge.

Pins the Phase-0 verified contract (SPEC.md §8, decisions D1–D4/D10/D22). The
shape mirrors ``portopt_db.config.DbConfig``: a frozen dataclass of primitives
(serialisable, hashable) plus a ``load_config`` factory that sources secrets from
the environment.

Pinned constants are dataclass defaults, so a bare ``import fund.config`` yields
the verified values with no environment set. Secrets (``DATABASE_URL``,
``OLLAMA_API_KEY``, ``FRED_API_KEY``, ``SSL_CERT_FILE``, plus every per-provider
key/endpoint behind ``LLM_PROVIDER``) are read from ``env`` and stay ``None`` when
absent — import must never fail in CI, so *required*-secret validation happens at
use-time (pool/model construction), not here.

Deliberately imports nothing from ``optimizer`` (config must be cheap to import);
the psycopg ``dict_row`` callable is imported lazily inside ``langgraph_pool_kwargs``
so reading config in a test does not drag in the DB driver.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
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
    # `llm_provider` selects the builder in the provider registry
    # (`fund.agents.model._BUILDERS`); the env-free default keeps today's
    # DeepSeek-on-Ollama-Cloud path byte-for-byte (SPEC "Switchable LLM Backends").
    llm_provider: str = "ollama"
    ollama_base_url: str = "https://ollama.com"
    primary_model: str = "deepseek-v4.1-flash:cloud"
    fallback_model: str = "deepseek-v4-pro:cloud"
    model_temperature: float = 0.0
    # Force the non-thinking "flash" route: thinking mode rejects tool_choice and
    # deepagents depends on reliable tool-calling (SPEC D4).
    model_reasoning: bool = False
    # `json_schema` fell through in probing; `function_calling` round-tripped.
    structured_output_method: str = "function_calling"

    # --- D4 provider registry: per-provider keys/endpoints. Each stays None (or
    # its cloud default) when unset so a bare import never fails; auth is
    # validated at build time in each builder, never here. `ollama_api_key` +
    # `ollama_base_url` live above (the current default provider). ---
    # openai / openrouter (langchain-openai; ChatOpenAI)
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openrouter_api_key: str | None = None
    # anthropic / google / groq (a single cloud key each)
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None
    # nvidia — hosted (NVIDIA_API_KEY) vs self-hosted NIM (NVIDIA_BASE_URL set)
    nvidia_api_key: str | None = None
    nvidia_base_url: str | None = None
    # huggingface — cloud only in this delivery; `hf_mode == "local"` is deferred
    hf_api_token: str | None = None
    hf_mode: str = "cloud"
    # aws (Bedrock Converse) — boto3 also reads AWS_*/IAM role; region_name here
    aws_region: str | None = None
    # microsoft (Azure OpenAI) — the four Azure env vars, wired at build time
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_deployment_name: str | None = None

    # --- D22 guards: LOW explicit recursion_limit + PM round cap (library
    # default 9999 is NOT a safety bound). ---
    recursion_limit: int = 50
    max_pm_rounds: int = 10

    # --- D10 HITL: tools gated behind interrupt_on (checkpointer required) ---
    interrupt_on: tuple[str, ...] = field(default=("place_orders",))

    # --- Phase-5 persistence: Store key the active ConstraintSet is cached under
    # (namespace = (portfolio_id,)) so a Phase-4 ``ConstraintSetRef`` resolves. ---
    constraint_set_store_key: str = "constraint_set"

    # --- Phase-9 daemon: scheduler cadences + heartbeat-lease + drain (OQ3). ---
    # Cron uses the weekday NAME `sat`; a bare `0` fires Monday under APScheduler
    # ``from_crontab`` (0=Mon..6=Sun).
    fund_rebalance_cron: str = "0 3 * * sat"
    fund_drift_interval_seconds: int = 900
    fund_heartbeat_cadence_seconds: int = 30
    fund_orphan_timeout_seconds: int = 300
    fund_shutdown_drain_timeout_seconds: int = 30

    # --- Phase-9 agent hardening: annual re-profiling marker + summarization pin
    # (mirrors the deepagents built-in defaults so the daemon can override them). ---
    reprofile_interval_days: int = 365
    summary_token_threshold: int = 170000
    summary_messages_to_keep: int = 6

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


def _read_secret(env: Mapping[str, str], name: str) -> str | None:
    """Resolve a secret from a docker-secret file first, then an inline env var.

    Reads the path in ``<name>_FILE`` (e.g. ``/run/secrets/ollama_api_key``) and
    returns its stripped content when non-empty — the file-based docker-secret
    path always wins. An empty/whitespace-only or unreadable secret file falls
    through to the inline ``<name>`` value, then to ``None``. Compose renders
    every declared secret file (empty placeholder when unset), so an empty file
    must mean "unset", not ``""``.

    Note: only true secrets are routed through this. ``SSL_CERT_FILE`` is *not* —
    despite its ``_FILE`` suffix it is itself the CA-bundle path, read verbatim.
    """
    file_path = env.get(f"{name}_FILE")
    if file_path:
        try:
            content = Path(file_path).read_text(encoding="utf-8").strip()
        except OSError:
            content = ""
        if content:
            return content
    return env.get(name) or None


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

            # override=True: the project .env is authoritative over ambient shell
            # pollution. Notably a conda-activated env exports SSL_CERT_FILE to a
            # stock cacert.pem lacking the corporate CA; without override that
            # shadows the correct .env value and breaks TLS to Ollama Cloud.
            load_dotenv(override=True)
        env = os.environ

    defaults = FundConfig()
    return FundConfig(
        database_url=env.get("DATABASE_URL"),
        ollama_api_key=_read_secret(env, "OLLAMA_API_KEY"),
        fred_api_key=_read_secret(env, "FRED_API_KEY"),
        # NOT a secret file: SSL_CERT_FILE is itself the CA-bundle path (verbatim).
        ssl_cert_file=env.get("SSL_CERT_FILE"),
        # D4 provider registry: LLM_PROVIDER + shared model ids + Ollama base_url;
        # missing → dataclass default (env-free load stays on DeepSeek/Ollama Cloud).
        llm_provider=env.get("LLM_PROVIDER", defaults.llm_provider),
        primary_model=env.get("FUND_PRIMARY_MODEL", defaults.primary_model),
        fallback_model=env.get("FUND_FALLBACK_MODEL", defaults.fallback_model),
        ollama_base_url=env.get("OLLAMA_BASE_URL", defaults.ollama_base_url),
        # Per-provider secret keys → _read_secret (file-based docker secret wins);
        # non-secret endpoints/urls/region stay plain env.get (absent → None).
        openai_api_key=_read_secret(env, "OPENAI_API_KEY"),
        openai_base_url=env.get("OPENAI_BASE_URL"),
        openrouter_api_key=_read_secret(env, "OPENROUTER_API_KEY"),
        anthropic_api_key=_read_secret(env, "ANTHROPIC_API_KEY"),
        google_api_key=_read_secret(env, "GOOGLE_API_KEY"),
        groq_api_key=_read_secret(env, "GROQ_API_KEY"),
        nvidia_api_key=_read_secret(env, "NVIDIA_API_KEY"),
        nvidia_base_url=env.get("NVIDIA_BASE_URL"),
        hf_api_token=_read_secret(env, "HUGGINGFACEHUB_API_TOKEN"),
        hf_mode=env.get("FUND_HF_MODE", defaults.hf_mode),
        aws_region=env.get("AWS_REGION"),
        azure_openai_api_key=_read_secret(env, "AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=env.get("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_version=env.get("OPENAI_API_VERSION"),
        azure_openai_deployment_name=env.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
        # Phase-9 daemon knobs: `FUND_*` aliases; missing → dataclass default.
        fund_rebalance_cron=env.get(
            "FUND_REBALANCE_CRON", defaults.fund_rebalance_cron
        ),
        fund_drift_interval_seconds=int(
            env.get("FUND_DRIFT_INTERVAL_SECONDS", defaults.fund_drift_interval_seconds)
        ),
        fund_heartbeat_cadence_seconds=int(
            env.get(
                "FUND_HEARTBEAT_CADENCE_SECONDS",
                defaults.fund_heartbeat_cadence_seconds,
            )
        ),
        fund_orphan_timeout_seconds=int(
            env.get("FUND_ORPHAN_TIMEOUT_SECONDS", defaults.fund_orphan_timeout_seconds)
        ),
        fund_shutdown_drain_timeout_seconds=int(
            env.get(
                "FUND_SHUTDOWN_DRAIN_TIMEOUT_SECONDS",
                defaults.fund_shutdown_drain_timeout_seconds,
            )
        ),
        reprofile_interval_days=int(
            env.get("FUND_REPROFILE_INTERVAL_DAYS", defaults.reprofile_interval_days)
        ),
        summary_token_threshold=int(
            env.get("FUND_SUMMARY_TOKEN_THRESHOLD", defaults.summary_token_threshold)
        ),
        summary_messages_to_keep=int(
            env.get("FUND_SUMMARY_MESSAGES_TO_KEEP", defaults.summary_messages_to_keep)
        ),
    )


# Module-level singleton: cheap to import, secrets validated when used.
settings: FundConfig = load_config()

__all__ = ["FundConfig", "load_config", "settings"]
