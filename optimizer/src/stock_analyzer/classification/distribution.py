"""
Distribution Tracking and Signal Classification
===============================================

Implements percentile-based signal classification with cumulative distribution tracking.

Approach:
1. Track composite z-scores across stock universe
2. Calculate empirical percentile thresholds (p20, p40, p60, p80)
3. Classify signals based on cross-sectional ranking
4. Persist distribution to database for cumulative tracking

Benefits:
- Self-corrects for distribution drift (mean ≠ 0, std ≠ 1)
- Maintains equal-sized buckets (20% each) regardless of regime
- Aligns with institutional cross-sectional methodology
"""

from typing import Optional
import logging
import numpy as np
from scipy import stats
from sqlalchemy import select, desc

from app.database import database_manager
from app.models.signal_distribution import SignalDistribution
from baml_client.types import SignalType

logger = logging.getLogger(__name__)


class SignalClassifier:
    """
    Percentile-based signal classifier with distribution tracking.

    Three-tier classification strategy:
    1. Saved distribution (database): Bootstrap from previous run (50+ stocks)
    2. Empirical percentiles (rolling): Use recent history (100+ stocks)
    3. Fixed thresholds (fallback): Theoretical N(0,1) quintiles
    """

    def __init__(self, validation_interval: int = 50, save_interval: int = 100):
        """
        Initialize signal classifier.

        Args:
            validation_interval: Log distribution validation every N stocks
            save_interval: Save distribution snapshot every N stocks
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

        Args:
            composite_z: Composite z-score after factor combination and macro adjustments

        Returns:
            SignalType based on percentile classification
        """
        # Track z-score before classification
        self.z_score_history.append(composite_z)

        # Classify using appropriate method
        signal_type = self._classify_with_percentiles(composite_z)

        # Periodic validation
        if len(self.z_score_history) % self.validation_interval == 0:
            self._log_distribution_validation()

        # Periodic save
        if len(self.z_score_history) % self.save_interval == 0:
            save_distribution_snapshot(
                self, universe_description="Mathematical Signal Calculator - Periodic Snapshot"
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

            logger.debug(
                f"Using saved distribution thresholds: "
                f"p20={p20:.2f}, p40={p40:.2f}, p60={p60:.2f}, p80={p80:.2f}"
            )

        # Option 2: Use recent history (100+ stocks)
        elif len(self.z_score_history) >= 100:
            recent_history = np.array(self.z_score_history[-500:])
            p20 = float(np.percentile(recent_history, 20))
            p40 = float(np.percentile(recent_history, 40))
            p60 = float(np.percentile(recent_history, 60))
            p80 = float(np.percentile(recent_history, 80))

            logger.debug(
                f"Using empirical percentiles from {len(recent_history)} stocks: "
                f"p20={p20:.2f}, p40={p40:.2f}, p60={p60:.2f}, p80={p80:.2f}"
            )

        # Option 3: Fallback to fixed thresholds
        else:
            logger.debug(
                f"Insufficient history ({len(self.z_score_history)} < 100 stocks). "
                f"Falling back to fixed thresholds."
            )
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

    def _log_distribution_validation(self) -> None:
        """Log empirical distribution validation."""
        if len(self.z_score_history) < 30:
            return

        z_scores = np.array(self.z_score_history)
        n = len(z_scores)

        # Calculate empirical statistics
        mean = np.mean(z_scores)
        std = np.std(z_scores, ddof=1)
        median = np.median(z_scores)

        # Calculate empirical percentiles
        p20 = np.percentile(z_scores, 20)
        p40 = np.percentile(z_scores, 40)
        p60 = np.percentile(z_scores, 60)
        p80 = np.percentile(z_scores, 80)

        # Theoretical percentiles
        theory_p20 = stats.norm.ppf(0.20)
        theory_p40 = stats.norm.ppf(0.40)
        theory_p60 = stats.norm.ppf(0.60)
        theory_p80 = stats.norm.ppf(0.80)

        # Calculate bucket sizes
        large_gain_pct = (z_scores >= p80).sum() / n * 100
        small_gain_pct = ((z_scores >= p60) & (z_scores < p80)).sum() / n * 100
        neutral_pct = ((z_scores >= p40) & (z_scores < p60)).sum() / n * 100
        small_decline_pct = ((z_scores >= p20) & (z_scores < p40)).sum() / n * 100
        large_decline_pct = (z_scores < p20).sum() / n * 100

        # Log validation
        logger.info("=" * 80)
        logger.info(f"📊 Z-SCORE DISTRIBUTION VALIDATION (n={n} stocks)")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Empirical Statistics:")
        logger.info(f"  Mean:   {mean:+.3f} (expected: 0.000)")
        logger.info(f"  Std:    {std:.3f} (expected: 1.000)")
        logger.info(f"  Median: {median:+.3f} (expected: 0.000)")
        logger.info("")
        logger.info("Percentile Comparison:")
        logger.info(
            f"  20th: {p20:+.3f} vs {theory_p20:+.3f} (diff: {abs(p20 - theory_p20):.3f})"
        )
        logger.info(
            f"  40th: {p40:+.3f} vs {theory_p40:+.3f} (diff: {abs(p40 - theory_p40):.3f})"
        )
        logger.info(
            f"  60th: {p60:+.3f} vs {theory_p60:+.3f} (diff: {abs(p60 - theory_p60):.3f})"
        )
        logger.info(
            f"  80th: {p80:+.3f} vs {theory_p80:+.3f} (diff: {abs(p80 - theory_p80):.3f})"
        )
        logger.info("")
        logger.info("Signal Bucket Distribution:")
        logger.info(f"  LARGE_GAIN:    {large_gain_pct:5.1f}% (expected: 20.0%)")
        logger.info(f"  SMALL_GAIN:    {small_gain_pct:5.1f}% (expected: 20.0%)")
        logger.info(f"  NEUTRAL:       {neutral_pct:5.1f}% (expected: 20.0%)")
        logger.info(f"  SMALL_DECLINE: {small_decline_pct:5.1f}% (expected: 20.0%)")
        logger.info(f"  LARGE_DECLINE: {large_decline_pct:5.1f}% (expected: 20.0%)")
        logger.info("")

        # Warnings
        if abs(mean) > 0.3:
            logger.warning(f"⚠️  Mean deviation: {mean:+.3f} (expected ~0)")
        if abs(std - 1.0) > 0.3:
            logger.warning(f"⚠️  Std deviation: {std:.3f} (expected ~1)")

        bucket_pcts = [
            large_gain_pct,
            small_gain_pct,
            neutral_pct,
            small_decline_pct,
            large_decline_pct,
        ]
        max_deviation = max(abs(pct - 20.0) for pct in bucket_pcts)
        if max_deviation > 5.0:
            logger.warning(
                f"⚠️  Bucket imbalance: Max deviation from 20% is {max_deviation:.1f}%"
            )
        else:
            logger.info("✅ Distribution validation PASSED: Buckets balanced within ±5%")

        logger.info("=" * 80)

    def finalize_run(self, universe_description: Optional[str] = None) -> None:
        """
        Save final distribution snapshot to database.

        Call this at the end of each batch/run to ensure cumulative tracking.

        Args:
            universe_description: Optional description of stock universe
        """
        if len(self.z_score_history) == 0:
            logger.warning("No z-scores to save. Skipping distribution save.")
            return

        logger.info(
            f"Finalizing run: Saving distribution with {len(self.z_score_history)} z-scores..."
        )

        save_distribution_snapshot(
            self,
            universe_description=universe_description
            or f"Signal Run - {len(self.z_score_history)} stocks",
            force_save=True,
        )

        logger.info("✅ Run finalized. Distribution saved to database.")


def load_saved_distribution(classifier: SignalClassifier) -> None:
    """
    Load latest valid distribution from database.

    Loads both distribution parameters AND z-score history for cumulative tracking.

    Args:
        classifier: SignalClassifier instance to update
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
                logger.info(
                    f"Loaded saved distribution: n={saved_dist.sample_size}, "
                    f"μ={saved_dist.mean:.3f}, σ={saved_dist.std:.3f}, "
                    f"date={saved_dist.distribution_date.date()}"
                )

                # Load previous z-scores for cumulative tracking
                if saved_dist.raw_zscores:
                    raw_data = saved_dist.raw_zscores
                    if isinstance(raw_data, dict) and 'scores' in raw_data:
                        classifier.z_score_history = list(raw_data['scores'])
                        logger.info(
                            f"Loaded {len(classifier.z_score_history)} previous z-scores "
                            f"for cumulative tracking"
                        )
            else:
                logger.info(
                    "No saved distribution found. Will use fixed thresholds "
                    "until sufficient history is built (100+ stocks)."
                )

    except Exception as e:
        logger.warning(f"Error loading saved distribution: {e}. Using fixed thresholds.")


def save_distribution_snapshot(
    classifier: SignalClassifier,
    universe_description: Optional[str] = None,
    force_save: bool = False,
) -> None:
    """
    Save current z-score distribution to database.

    Args:
        classifier: SignalClassifier instance with z-score history
        universe_description: Optional description of stock universe
        force_save: Force save even if below minimum sample size
    """
    if not force_save and len(classifier.z_score_history) < 50:
        logger.debug(
            f"Skipping distribution save: insufficient sample size "
            f"({len(classifier.z_score_history)} < 50)"
        )
        return

    try:
        distribution = SignalDistribution.from_zscore_history(
            zscores=classifier.z_score_history,
            universe_description=universe_description
            or "Mathematical Signal Calculator Run",
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

            logger.info(
                f"Saved distribution snapshot: n={distribution.sample_size}, "
                f"μ={distribution.mean:.3f}, σ={distribution.std:.3f}, "
                f"valid={distribution.is_valid}"
            )

            # Update classifier
            if distribution.is_valid:
                classifier.saved_distribution = distribution

    except Exception as e:
        logger.error(f"Error saving distribution snapshot: {e}")
        import traceback

        logger.debug(traceback.format_exc())
