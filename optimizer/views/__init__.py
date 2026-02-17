"""View integration frameworks (Black-Litterman, Entropy Pooling, Opinion Pooling)."""

from optimizer.views._config import (
    BlackLittermanConfig,
    EntropyPoolingConfig,
    OpinionPoolingConfig,
    ViewUncertaintyMethod,
)
from optimizer.views._factory import (
    build_black_litterman,
    build_entropy_pooling,
    build_opinion_pooling,
)

__all__ = [
    "BlackLittermanConfig",
    "EntropyPoolingConfig",
    "OpinionPoolingConfig",
    "ViewUncertaintyMethod",
    "build_black_litterman",
    "build_entropy_pooling",
    "build_opinion_pooling",
]
