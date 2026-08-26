"""Ingestion-daemon configuration using Pydantic Settings v2."""

import os
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Secret fields overridden by mounted Docker-compose secret files at runtime.
# The container mounts each at ``$PORTOPT_SECRETS_DIR/<field>`` (default
# ``/run/secrets``); a present file wins over env/.env and is stripped.
_SECRET_FILE_FIELDS = (
    "trading_212_api_key",
    "trading_212_secret_key",
    "fred_api_key",
    "openai_api_key",
    "anthropic_api_key",
)


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Project Information
    project_name: str = "Optimizer Ingestion Daemon"
    version: str = "1.0.0"

    # Database Configuration - Local PostgreSQL via Docker
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:54320/optimizer_db",
        alias="DATABASE_URL",
    )

    # Pool Configuration - Standard settings for local PostgreSQL
    database_pool_size: int = Field(default=5)
    database_max_overflow: int = Field(default=10)
    database_pool_timeout: int = Field(default=30)
    database_pool_recycle: int = Field(default=3600)
    database_pool_pre_ping: bool = Field(default=True)
    database_echo: bool = Field(default=False)
    database_pool_reset_on_return: str = Field(default="rollback")

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # Monitoring — the worker serves Prometheus metrics on this port, and the
    # container healthcheck probes the same endpoint. Disabling metrics
    # therefore also removes the healthcheck target.
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9000, alias="METRICS_PORT")

    # Trading212 API
    trading_212_api_key: str = Field(default="", alias="TRADING_212_API_KEY")
    trading_212_secret_key: str = Field(default="", alias="TRADING_212_SECRET_KEY")
    trading_212_mode: str = Field(default="live", alias="TRADING_212_MODE")

    # FRED API
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")

    # LLM provider — cloud-only (openai or anthropic); local models are not
    # supported. Keys/models flow into BAML at call time via a ClientRegistry.
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-latest", alias="ANTHROPIC_MODEL"
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_llm_provider(cls, v: object) -> object:
        """Accept case-insensitive 'openai'/'anthropic'; reject local/unknown."""
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized not in ("openai", "anthropic"):
                raise ValueError(
                    "LLM_PROVIDER must be 'openai' or 'anthropic' (cloud only)"
                )
            return normalized
        return v

    # Scheduler — cron expressions (5-field: min hour dom month dow)
    scheduler_daily_pipeline_cron: str = Field(
        default="0 7 * * *",
        alias="SCHEDULER_DAILY_PIPELINE_CRON",
    )
    scheduler_midday_news_cron: str = Field(
        default="0 14 * * *",
        alias="SCHEDULER_MIDDAY_NEWS_CRON",
    )
    scheduler_universe_build_cron: str = Field(
        default="0 2 * * 0",
        alias="SCHEDULER_UNIVERSE_BUILD_CRON",
        description=(
            "Trading 212 universe rebuild. Must precede "
            "SCHEDULER_WEEKLY_REFETCH_CRON so the yfinance rebuild fetches the "
            "fresh instrument set."
        ),
    )
    scheduler_weekly_refetch_cron: str = Field(
        default="0 3 * * 0",
        alias="SCHEDULER_WEEKLY_REFETCH_CRON",
    )
    scheduler_fred_monthly_cron: str = Field(
        default="0 8 1 * *",
        alias="SCHEDULER_FRED_MONTHLY_CRON",
    )
    scheduler_news_refresh_interval_minutes: int = Field(
        default=30,
        alias="SCHEDULER_NEWS_REFRESH_INTERVAL_MIN",
    )
    scheduler_misfire_grace_time_seconds: int = Field(
        default=3600,
        alias="SCHEDULER_MISFIRE_GRACE_TIME_SECONDS",
    )
    # Liveness reaper (issues #585-#590): heartbeat cadence written by each
    # background worker, and the staleness threshold past which the orphan
    # reaper marks a row failed.
    scheduler_heartbeat_cadence_seconds: int = Field(
        default=30,
        ge=1,
        alias="SCHEDULER_HEARTBEAT_CADENCE_SECONDS",
    )
    scheduler_orphan_heartbeat_timeout_seconds: int = Field(
        default=300,
        ge=1,
        alias="SCHEDULER_ORPHAN_HEARTBEAT_TIMEOUT_SECONDS",
    )
    scheduler_orphan_strategy: str = Field(
        default="fail",
        alias="SCHEDULER_ORPHAN_STRATEGY",
        description=(
            "How the orphan reaper handles a dead-worker job (R3/§5.3). "
            "'fail' marks it failed (the next cron re-runs it). 'reclaim' "
            "additionally re-dispatches the step immediately (at-least-once "
            "self-healing). Keep 'fail' until the idempotent-upsert audit is "
            "signed off — reclaiming a non-idempotent job duplicates rows."
        ),
    )
    scheduler_orphan_max_reclaim_attempts: int = Field(
        default=3,
        ge=1,
        alias="SCHEDULER_ORPHAN_MAX_RECLAIM_ATTEMPTS",
        description=(
            "Cap on RECLAIM re-dispatches for one job lineage, so a job whose "
            "worker keeps dying is not re-dispatched forever."
        ),
    )

    @field_validator("scheduler_orphan_strategy", mode="before")
    @classmethod
    def _normalize_orphan_strategy(cls, v: object) -> object:
        """Accept case-insensitive 'fail'/'reclaim'; reject anything else."""
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized not in ("fail", "reclaim"):
                raise ValueError(
                    "SCHEDULER_ORPHAN_STRATEGY must be 'fail' or 'reclaim'"
                )
            return normalized
        return v

    scheduler_shutdown_drain_timeout_seconds: int = Field(
        default=30,
        ge=1,
        alias="SCHEDULER_SHUTDOWN_DRAIN_TIMEOUT_SECONDS",
        description=(
            "On SIGTERM the daemon stops claiming new jobs and waits up to this "
            "many seconds for in-flight steps to finish before a forced exit. "
            "Abandoned work is safe to re-run (fetch writes are idempotent "
            "upserts). Size the container stop_grace_period / "
            "terminationGracePeriodSeconds above this value so Docker/K8s does "
            "not SIGKILL mid-drain; raise both to drain long fetches cleanly."
        ),
    )

    # yfinance fetch settings
    yfinance_request_timeout_seconds: int = Field(
        default=30,
        alias="YFINANCE_REQUEST_TIMEOUT_SECONDS",
        description=(
            "Wall-clock timeout (seconds) applied to each yfinance network call. "
            "A timed-out call raises TimeoutError and is recorded as a per-ticker "
            "error; the job continues to the next ticker/category."
        ),
    )
    yfinance_fetch_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        alias="YFINANCE_FETCH_WORKERS",
        description=(
            "Number of parallel workers used by the scheduler's daily yfinance "
            "step. Must be in [1, 16]. Setting 1 uses the serial path."
        ),
    )

    # Notifications
    notification_webhook_url: str | None = Field(
        default=None,
        alias="NOTIFICATION_WEBHOOK_URL",
    )

    # Environment detection helpers
    debug: bool = Field(default=False)
    environment: str = Field(default="development")

    @model_validator(mode="after")
    def _apply_docker_secret_files(self) -> "Settings":
        """Override secret fields from mounted Docker-compose secret files.

        In the container each secret is mounted at ``$PORTOPT_SECRETS_DIR/<field>``
        (default ``/run/secrets``). A present file wins over env/.env (the compose
        secret is source of truth), value stripped. Missing files leave the
        env/default value, so an unset secret stays "" (T212-absent skip intact).
        """
        secrets_dir = Path(os.getenv("PORTOPT_SECRETS_DIR", "/run/secrets"))
        for field in _SECRET_FILE_FIELDS:
            secret_path = secrets_dir / field
            if secret_path.is_file():
                setattr(self, field, secret_path.read_text(encoding="utf-8").strip())
        return self

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development" or self.debug


# Create global settings instance
settings = Settings()
