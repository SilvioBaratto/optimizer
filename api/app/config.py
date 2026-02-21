"""Application configuration using Pydantic Settings v2"""

from pydantic import Field
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
    trading_212_mode: str = Field(default="live", alias="TRADING_212_MODE")

    # Performance
    connection_timeout: int = Field(default=10)
    read_timeout: int = Field(default=30)

    # Environment detection helpers
    debug: bool = Field(default=True)
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
