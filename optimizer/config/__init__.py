"""
Configuration Layer - Externalized configuration for all components.

This module provides dataclass-based configuration objects that:
- Externalize magic numbers and thresholds
- Enable easy testing with different configurations
- Support environment-based configuration
- Provide sensible defaults with override capability

Usage:
    from config import QualityFilterConfig, PortfolioConfig

    # Use defaults
    config = QualityFilterConfig()

    # Override specific values
    config = QualityFilterConfig(min_sharpe_ratio=0.7)

    # Load from environment
    config = QualityFilterConfig.from_env()
"""

from optimizer.config.quality_filter_config import QualityFilterConfig
from optimizer.config.portfolio_config import PortfolioConfig
from optimizer.config.optimizer_config import OptimizerConfig
from optimizer.config.universe_builder_config import UniverseBuilderConfig

__all__ = [
    "QualityFilterConfig",
    "PortfolioConfig",
    "OptimizerConfig",
    "UniverseBuilderConfig",
]
