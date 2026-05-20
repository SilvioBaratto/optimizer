"""Application configuration using Pydantic Settings v2"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Project Information
    project_name: str = "FastAPI Template"
    version: str = "1.0.0"

    # API Configuration
    api_v1_str: str = "/api/v1"

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
    cache_ttl_default: int = Field(default=300)
    cache_ttl_users: int = Field(default=600)
    cache_ttl_leagues: int = Field(default=1800)

    # CORS
    cors_origins: str = Field(default="http://localhost:4200,http://localhost:4300")

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list"""
        if not self.cors_origins or self.cors_origins.strip() == "":
            return [
                "http://localhost:4200",
                "http://localhost:4300",
                "http://127.0.0.1:4200",
            ]
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]

    # Security (optional for production)

    # Rate Limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_path: str = Field(default="/metrics")

    # Trading212 API
    trading_212_api_key: str = Field(default="", alias="TRADING_212_API_KEY")
    trading_212_secret_key: str = Field(default="", alias="TRADING_212_SECRET_KEY")
    trading_212_mode: str = Field(default="live", alias="TRADING_212_MODE")

    # FRED API
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")

    # Benchmarks / reference indices
    default_benchmark_ticker: str = Field(
        default="SPY",
        alias="DEFAULT_BENCHMARK_TICKER",
    )
    benchmark_tickers: list[str] = Field(
        default=[
            "SPY",
            "QQQ",
            "IWM",
            "EFA",
            "EEM",
            "AGG",
            "VGK",
            "VWO",
            "TLT",
            "GLD",
            "URTH",
            "VBINX",
        ],
        alias="BENCHMARK_TICKERS",
        description=(
            "Reference-index tickers required in the DB. "
            "Bootstrap on startup + refreshed by daily/weekly scheduler. "
            "Override via comma-separated env var (e.g. 'SPY,QQQ,IWM')."
        ),
    )
    scheduler_benchmark_stale_days: int = Field(
        default=2,
        alias="SCHEDULER_BENCHMARK_STALE_DAYS",
        description=(
            "Bootstrap re-seeds benchmarks whose latest price is older "
            "than this many days. Higher = fewer cold-start fetches."
        ),
    )

    @field_validator("benchmark_tickers", mode="before")
    @classmethod
    def _split_benchmark_tickers(cls, v: object) -> object:
        """Allow comma-separated env strings: BENCHMARK_TICKERS=SPY,QQQ,IWM."""
        if isinstance(v, str):
            return [t.strip().upper() for t in v.split(",") if t.strip()]
        if isinstance(v, list):
            return [str(t).strip().upper() for t in v if str(t).strip()]
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
        alias="SCHEDULER_HEARTBEAT_CADENCE_SECONDS",
    )
    scheduler_orphan_heartbeat_timeout_seconds: int = Field(
        default=300,
        alias="SCHEDULER_ORPHAN_HEARTBEAT_TIMEOUT_SECONDS",
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

    # Portfolio drift — stale-price freshness threshold (issue #794, used by #795)
    stale_price_threshold_hours: int = Field(
        default=48,
        alias="STALE_PRICE_THRESHOLD_HOURS",
        description=(
            "Age (hours) past which a position's last-tick timestamp is "
            "considered stale and triggers a STALE_PRICE diagnostic flag."
        ),
    )

    # Notifications
    notification_webhook_url: str | None = Field(
        default=None,
        alias="NOTIFICATION_WEBHOOK_URL",
    )

    # Performance
    connection_timeout: int = Field(default=10)
    read_timeout: int = Field(default=30)

    # Environment detection helpers
    debug: bool = Field(default=False)
    environment: str = Field(default="development")

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development" or self.debug

    @property
    def is_staging(self) -> bool:
        """Check if running in staging"""
        return self.environment == "staging"

    @property
    def is_production_like(self) -> bool:
        """Check if running in production or staging mode"""
        return self.environment in ("production", "staging")


# Create global settings instance
settings = Settings()
