"""Application configuration using Pydantic Settings v2"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=["../.env.dev", "../.env.staging", "../.env.prod"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Project Information
    project_name: str = "API"
    version: str = "1.0.0"
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    debug: bool = Field(default=False, alias="DEBUG")
    
    # API Configuration
    api_v1_str: str = "/api/v1"
    
    # Database Configuration - Optimized for DuplicatePreparedStatementError Prevention
    # 
    # RECOMMENDED: Use Transaction Mode (port 6543) for maximum stability
    # Example Transaction Mode URL: postgresql+asyncpg://postgres.xxx:password@aws-0-region.pooler.supabase.com:6543/postgres
    # Example Session Mode URL:    postgresql+asyncpg://postgres.xxx:password@aws-0-region.pooler.supabase.com:5432/postgres
    #
    database_url: str = Field(alias="SUPABASE_DB_URL")
    
    # Pool Configuration - Conservative settings for Supabase Micro (15 pool limit)
    database_pool_size: int = Field(default=15)  # Reduced from 8 - safer for multiple instances
    database_max_overflow: int = Field(default=3)  # Reduced from 5 - total = 8 connections max
    database_pool_timeout: int = Field(default=10)  # Increased for better reliability
    database_pool_recycle: int = Field(default=300)  # 15 minutes - faster recycling
    database_pool_pre_ping: bool = Field(default=True)  # Connection health checking
    database_echo: bool = Field(default=False)
    database_command_timeout: int = Field(default=30)  # Increased for stability
    
    # CRITICAL: Prepared Statement Configuration (fixes DuplicatePreparedStatementError)
    database_statement_cache_size: int = Field(default=0)  # DISABLED by default - prevents conflicts
    database_enable_prepared_statements: bool = Field(default=False)  # Safer default
    
    # Connection Resilience - Optimized for cloud environments
    database_pool_reset_on_return: str = Field(default="rollback")  # Clean connection state
    database_connection_ping_interval: int = Field(default=60)  # TCP keepalive interval
    
    # Supabase
    supabase_url: str = Field(alias="SUPABASE_URL")
    supabase_key: str = Field(alias="SUPABASE_KEY")
    supabase_jwt_secret: Optional[str] = Field(default=None, alias="SUPABASE_JWT_SECRET")
    
    # Development-only authentication bypass
    dev_auth_bypass: bool = Field(default=False, alias="DEV_AUTH_BYPASS")
    supabase_service_role_key: Optional[str] = Field(default=None, alias="SUPABASE_SERVICE_ROLE_KEY")
    
    # Redis
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    cache_ttl_default: int = Field(default=300)
    cache_ttl_users: int = Field(default=600)
    cache_ttl_leagues: int = Field(default=1800)
    
    # CORS
    cors_origins: str = Field(
        default="http://localhost:4200,http://localhost:4300"
    )
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list"""
        if not self.cors_origins or self.cors_origins.strip() == "":
            return [
                "http://localhost:4200",
                "http://localhost:4300",
                "http://127.0.0.1:4200"
            ]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    # Security
    swagger_user: Optional[str] = Field(default=None, alias="SWAGGER_USER")
    swagger_pass: Optional[str] = Field(default=None, alias="SWAGGER_PASS")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    
    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_path: str = Field(default="/metrics")
    
    # Performance
    connection_timeout: int = Field(default=10)
    read_timeout: int = Field(default=30)
    
    @property
    def async_database_url(self) -> str:
        """Convert database URL to async format"""
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self.database_url
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
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