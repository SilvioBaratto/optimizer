from typing import Optional
import numpy as np
from sqlalchemy import select, desc

from optimizer.database.database import database_manager
from optimizer.database.models.signal_distribution import SignalDistribution
from baml_client.types import SignalType


class SignalClassifier:
    """
    Percentile-based signal classifier with distribution tracking.
    """

    def __init__(self, validation_interval: int = 50, save_interval: int = 100):
        """
        Initialize signal classifier.
        """
        self.z_score_history: list[float] = []
        self.validation_interval = validation_interval
        self.save_interval = save_interval
        self.saved_distribution: Optional[SignalDistribution] = None

        # Load previous distribution for cumulative tracking
        load_saved_distribution(self)

    def classify(self, composite_z: float) -> SignalType:
        """
        Classify signal using empirical percentile thresholds.
        """
        # Track z-score before classification
        self.z_score_history.append(composite_z)

        # Classify using appropriate method
        signal_type = self._classify_with_percentiles(composite_z)

        # Periodic save
        if len(self.z_score_history) % self.save_interval == 0:
            save_distribution_snapshot(
                self,
                universe_description="Mathematical Signal Calculator - Periodic Snapshot",
            )

        return signal_type

    def _classify_with_percentiles(self, composite_z: float) -> SignalType:
        """Classify using percentile thresholds with fallback."""
        # Option 1: Use saved distribution (bootstraps quickly)
        if self.saved_distribution is not None:
            p20 = self.saved_distribution.p20
            p40 = self.saved_distribution.p40
            p60 = self.saved_distribution.p60
            p80 = self.saved_distribution.p80

        # Option 2: Use recent history (100+ stocks)
        elif len(self.z_score_history) >= 100:
            recent_history = np.array(self.z_score_history[-500:])
            p20 = float(np.percentile(recent_history, 20))
            p40 = float(np.percentile(recent_history, 40))
            p60 = float(np.percentile(recent_history, 60))
            p80 = float(np.percentile(recent_history, 80))

        # Option 3: Fallback to fixed thresholds
        else:
            return self._classify_with_fixed_thresholds(composite_z)

        # Classify using percentile thresholds
        if composite_z >= p80:
            return SignalType.LARGE_GAIN
        elif composite_z >= p60:
            return SignalType.SMALL_GAIN
        elif composite_z >= p40:
            return SignalType.NEUTRAL
        elif composite_z >= p20:
            return SignalType.SMALL_DECLINE
        else:
            return SignalType.LARGE_DECLINE

    def _classify_with_fixed_thresholds(self, composite_z: float) -> SignalType:
        """Fallback classification using theoretical N(0,1) quintiles."""
        if composite_z >= 0.84:
            return SignalType.LARGE_GAIN
        elif composite_z >= 0.25:
            return SignalType.SMALL_GAIN
        elif composite_z >= -0.25:
            return SignalType.NEUTRAL
        elif composite_z >= -0.84:
            return SignalType.SMALL_DECLINE
        else:
            return SignalType.LARGE_DECLINE

    def finalize_run(self, universe_description: Optional[str] = None) -> None:
        """
        Save final distribution snapshot to database.

        Call this at the end of each batch/run to ensure cumulative tracking.

        Args:
            universe_description: Optional description of stock universe
        """
        if len(self.z_score_history) == 0:
            return

        save_distribution_snapshot(
            self,
            universe_description=universe_description
            or f"Signal Run - {len(self.z_score_history)} stocks",
            force_save=True,
        )


def load_saved_distribution(classifier: SignalClassifier) -> None:
    """
    Load latest valid distribution from database.
    """
    try:
        with database_manager.get_session() as session:
            query = (
                select(SignalDistribution)
                .where(
                    SignalDistribution.is_active == True,
                    SignalDistribution.sample_size >= 50,
                )
                .order_by(desc(SignalDistribution.distribution_date))
                .limit(1)
            )

            result = session.execute(query)
            saved_dist = result.scalar_one_or_none()

            if saved_dist:
                classifier.saved_distribution = saved_dist

                # Load previous z-scores for cumulative tracking
                if saved_dist.raw_zscores:
                    raw_data = saved_dist.raw_zscores
                    if isinstance(raw_data, dict) and "scores" in raw_data:
                        classifier.z_score_history = list(raw_data["scores"])

    except Exception:
        pass


def save_distribution_snapshot(
    classifier: SignalClassifier,
    universe_description: Optional[str] = None,
    force_save: bool = False,
) -> None:
    """
    Save current z-score distribution to database.
    """
    if not force_save and len(classifier.z_score_history) < 50:
        return

    try:
        distribution = SignalDistribution.from_zscore_history(
            zscores=classifier.z_score_history,
            universe_description=universe_description or "Mathematical Signal Calculator Run",
            store_raw_zscores=True,  # Enable cumulative tracking
        )

        with database_manager.get_session() as session:
            # Deactivate previous distributions
            stmt = select(SignalDistribution).where(SignalDistribution.is_active == True)
            previous_dists = session.execute(stmt).scalars().all()
            for prev_dist in previous_dists:
                prev_dist.is_active = False

            # Add new distribution
            session.add(distribution)
            session.commit()

            # Update classifier
            if distribution.is_valid:
                classifier.saved_distribution = distribution

    except Exception:
        pass
