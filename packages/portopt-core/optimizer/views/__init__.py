"""View integration frameworks (Black-Litterman, Entropy Pooling, Opinion Pooling).

Also exposes DB-agnostic adapters that turn analyst/estimate snapshots
(``analyst_price_targets``, ``analyst_recommendations``) into Black-Litterman
views + Idzorek confidences (:mod:`optimizer.views._builder`).
"""

from optimizer.views._builder import (
    AnalystSignal,
    PriceTargetStatistic,
    build_analyst_bl_views,
    build_black_litterman_config_from_signals,
    implied_return_from_price_target,
    recommendation_confidence,
)
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
from optimizer.views._uncertainty import calibrate_omega_from_track_record

__all__ = [
    "AnalystSignal",
    "BlackLittermanConfig",
    "EntropyPoolingConfig",
    "OpinionPoolingConfig",
    "PriceTargetStatistic",
    "ViewUncertaintyMethod",
    "build_analyst_bl_views",
    "build_black_litterman",
    "build_black_litterman_config_from_signals",
    "build_entropy_pooling",
    "build_opinion_pooling",
    "calibrate_omega_from_track_record",
    "implied_return_from_price_target",
    "recommendation_confidence",
]
