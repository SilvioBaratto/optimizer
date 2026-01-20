"""Database configuration using Pydantic Settings v2"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Database settings with environment variable support"""

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Database Configuration
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5433/optimizer_db",
        alias="DATABASE_URL"
    )

    # Connection Pool Settings
    database_pool_size: int = Field(default=5)
    database_max_overflow: int = Field(default=5)
    database_pool_timeout: int = Field(default=10)
    database_pool_recycle: int = Field(default=300)
    database_pool_pre_ping: bool = Field(default=True)
    database_echo: bool = Field(default=False)
    database_command_timeout: int = Field(default=30)
    database_pool_reset_on_return: str = Field(default="rollback")

    # Logging
    log_level: str = Field(default="INFO")

    @property
    def is_local_database(self) -> bool:
        """Check if using local PostgreSQL database"""
        return "localhost" in self.database_url or "127.0.0.1" in self.database_url


# Create global settings instance
settings = Settings()
