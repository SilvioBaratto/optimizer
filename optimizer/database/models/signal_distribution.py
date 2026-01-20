"""
Signal Distribution Models
===========================
SQLAlchemy models for storing z-score distribution statistics and percentile thresholds.

Used by mathematical signal calculator to:
- Bootstrap classification from saved distributions (avoid needing 100+ stock history)
- Track distribution drift over time
- Ensure consistent signal classification across runs
- Validate new signals against historical reference distributions
"""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Float, DateTime, Text, Index, func
from sqlalchemy.dialects.postgresql import UUID, JSONB

from optimizer.database.models.base import Base, TimestampMixin


class SignalDistribution(Base, TimestampMixin):
    """
    Stores z-score distribution statistics for signal classification.

    Institutional Approach (Section 5.2.1):
    - Capture cross-sectional distribution of composite z-scores
    - Store quintile thresholds (20/40/60/80 percentiles)
    - Enable percentile-based signal classification
    - Track distribution quality and validity

    Usage Pattern:
    1. After analyzing N stocks, save distribution snapshot
    2. On next run, load latest distribution to classify new signals
    3. Periodically refresh distribution (e.g., weekly) to capture market changes
    4. Validate that new distributions are statistically similar
    """

    __tablename__ = "signal_distributions"

    # Primary Key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="UUID primary key",
    )

    # Distribution Metadata
    distribution_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        default=func.now(),
        comment="Date when this distribution was calculated",
    )

    sample_size: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Number of stocks in this distribution sample"
    )

    universe_description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Description of stock universe (e.g., 'S&P 500', 'European Large Cap', 'Trading212 Universe')",
    )

    # Distribution Statistics
    mean: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Mean of composite z-scores (should be ~0 for proper standardization)",
    )

    std: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Standard deviation of composite z-scores (should be ~1 for proper standardization)",
    )

    median: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Median of composite z-scores (should be ~0 for symmetric distribution)",
    )

    skewness: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Skewness of distribution (0 = symmetric, positive = right-skewed)",
    )

    kurtosis: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Kurtosis of distribution (3 = normal, higher = fat tails)",
    )

    # Quintile Thresholds (Industry Standard: 20% Buckets)
    # These are the empirical z-score cutoffs for signal classification
    p20: Mapped[float] = mapped_column(
        Float, nullable=False, comment="20th percentile threshold (bottom 20%)"
    )

    p40: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="40th percentile threshold (separates SMALL_DECLINE from NEUTRAL)",
    )

    p60: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="60th percentile threshold (separates NEUTRAL from SMALL_GAIN)",
    )

    p80: Mapped[float] = mapped_column(
        Float, nullable=False, comment="80th percentile threshold (top 20%)"
    )

    # Theoretical Comparison (for validation)
    # For N(0,1): p20=-0.84, p40=-0.25, p60=+0.25, p80=+0.84
    p20_deviation: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Deviation from theoretical p20 (-0.84)"
    )

    p40_deviation: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Deviation from theoretical p40 (-0.25)"
    )

    p60_deviation: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Deviation from theoretical p60 (+0.25)"
    )

    p80_deviation: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Deviation from theoretical p80 (+0.84)"
    )

    # Bucket Counts (Validation: should all be ~20%)
    large_gain_pct: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Percentage of stocks classified as LARGE_GAIN (expected: 20%)",
    )

    small_gain_pct: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Percentage of stocks classified as SMALL_GAIN (expected: 20%)",
    )

    neutral_pct: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Percentage of stocks classified as NEUTRAL (expected: 20%)",
    )

    small_decline_pct: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Percentage of stocks classified as SMALL_DECLINE (expected: 20%)",
    )

    large_decline_pct: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Percentage of stocks classified as LARGE_DECLINE (expected: 20%)",
    )

    max_bucket_deviation: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Maximum deviation from 20% across all buckets (should be < 5%)",
    )

    # Quality Flags
    is_valid: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        index=True,
        comment="Whether this distribution passes validation criteria",
    )

    is_active: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        index=True,
        comment="Whether this distribution should be used for classification (latest valid one)",
    )

    validation_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Notes on validation results (warnings, issues, etc.)",
    )

    # Advanced Statistics (optional, for monitoring)
    min_zscore: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Minimum z-score in sample"
    )

    max_zscore: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Maximum z-score in sample"
    )

    # Raw Distribution Data (optional: store full histogram for analysis)
    raw_zscores: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Optional: store raw z-score array or histogram bins for detailed analysis",
    )

    # Indexes
    __table_args__ = (
        # Query for latest active distribution
        Index("idx_signal_distributions_active_date", "is_active", "distribution_date"),
        # Query for valid distributions
        Index("idx_signal_distributions_valid", "is_valid"),
        # Time-series analysis
        Index("idx_signal_distributions_date", "distribution_date"),
        # Sample size filter (e.g., only use distributions with n >= 100)
        Index("idx_signal_distributions_sample_size", "sample_size"),
    )

    def __repr__(self) -> str:
        return (
            f"<SignalDistribution(id={self.id}, "
            f"date={self.distribution_date.date()}, "
            f"n={self.sample_size}, "
            f"μ={self.mean:.3f}, "
            f"σ={self.std:.3f}, "
            f"active={self.is_active})>"
        )

    def to_dict(self) -> dict:
        """Return dictionary representation for API responses."""
        return {
            "id": str(self.id),
            "distribution_date": self.distribution_date.isoformat(),
            "sample_size": self.sample_size,
            "universe_description": self.universe_description,
            # Statistics
            "statistics": {
                "mean": self.mean,
                "std": self.std,
                "median": self.median,
                "skewness": self.skewness,
                "kurtosis": self.kurtosis,
                "min": self.min_zscore,
                "max": self.max_zscore,
            },
            # Thresholds
            "thresholds": {
                "p20": self.p20,
                "p40": self.p40,
                "p60": self.p60,
                "p80": self.p80,
            },
            # Deviations from theory
            "theory_deviations": {
                "p20_deviation": self.p20_deviation,
                "p40_deviation": self.p40_deviation,
                "p60_deviation": self.p60_deviation,
                "p80_deviation": self.p80_deviation,
            },
            # Bucket distribution
            "bucket_distribution": {
                "large_gain_pct": self.large_gain_pct,
                "small_gain_pct": self.small_gain_pct,
                "neutral_pct": self.neutral_pct,
                "small_decline_pct": self.small_decline_pct,
                "large_decline_pct": self.large_decline_pct,
                "max_deviation": self.max_bucket_deviation,
            },
            # Flags
            "is_valid": self.is_valid,
            "is_active": self.is_active,
            "validation_notes": self.validation_notes,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_zscore_history(
        cls,
        zscores: list[float],
        universe_description: Optional[str] = None,
        store_raw_zscores: bool = True,
    ) -> "SignalDistribution":
        """
        Create SignalDistribution from a list of z-scores.

        Args:
            zscores: List of composite z-scores
            universe_description: Optional description of stock universe
            store_raw_zscores: Whether to store raw z-scores in database (for cumulative tracking)

        Returns:
            SignalDistribution instance (not yet saved to database)
        """
        import numpy as np
        from scipy import stats as scipy_stats

        z_array = np.array(zscores)
        n = len(z_array)

        # Calculate statistics (may be NaN if input contains NaN values)
        # Note: PostgreSQL Float columns support NaN, so we keep them as-is
        mean = float(np.mean(z_array))
        std = float(np.std(z_array, ddof=1))
        median = float(np.median(z_array))
        skewness = float(scipy_stats.skew(z_array))
        kurtosis = float(scipy_stats.kurtosis(z_array))

        # Calculate percentiles
        p20 = float(np.percentile(z_array, 20))
        p40 = float(np.percentile(z_array, 40))
        p60 = float(np.percentile(z_array, 60))
        p80 = float(np.percentile(z_array, 80))

        # Theoretical values for N(0,1)
        from scipy.stats import norm

        theory_p20 = norm.ppf(0.20)  # ≈ -0.84
        theory_p40 = norm.ppf(0.40)  # ≈ -0.25
        theory_p60 = norm.ppf(0.60)  # ≈ +0.25
        theory_p80 = norm.ppf(0.80)  # ≈ +0.84

        # Calculate deviations (convert to Python float to avoid np.float64 in SQL)
        p20_deviation = float(abs(p20 - theory_p20))
        p40_deviation = float(abs(p40 - theory_p40))
        p60_deviation = float(abs(p60 - theory_p60))
        p80_deviation = float(abs(p80 - theory_p80))

        # Calculate bucket percentages
        large_gain_pct = float((z_array >= p80).sum() / n * 100)
        small_gain_pct = float(((z_array >= p60) & (z_array < p80)).sum() / n * 100)
        neutral_pct = float(((z_array >= p40) & (z_array < p60)).sum() / n * 100)
        small_decline_pct = float(((z_array >= p20) & (z_array < p40)).sum() / n * 100)
        large_decline_pct = float((z_array < p20).sum() / n * 100)

        bucket_pcts = [
            large_gain_pct,
            small_gain_pct,
            neutral_pct,
            small_decline_pct,
            large_decline_pct,
        ]
        max_bucket_deviation = float(max(abs(pct - 20.0) for pct in bucket_pcts))

        # Validation
        is_valid = True
        validation_notes = []

        if np.isnan(mean):
            is_valid = False
            validation_notes.append("Mean is undefined (likely NaN in input data)")
        elif abs(mean) > 0.5:
            is_valid = False
            validation_notes.append(f"Mean deviation: {mean:.3f} (expected ~0)")

        if np.isnan(std):
            is_valid = False
            validation_notes.append("Std is undefined (likely NaN in input data)")
        elif abs(std - 1.0) > 0.5:
            is_valid = False
            validation_notes.append(f"Std deviation: {std:.3f} (expected ~1)")

        if not np.isnan(max_bucket_deviation) and max_bucket_deviation > 10.0:
            is_valid = False
            validation_notes.append(
                f"Bucket imbalance: {max_bucket_deviation:.1f}% (expected < 5%)"
            )

        if n < 50:
            is_valid = False
            validation_notes.append(f"Insufficient sample size: {n} (need >= 50)")

        # Optionally store raw z-scores for cumulative tracking
        raw_zscores_data = None
        if store_raw_zscores:
            # Store as list in JSONB (more efficient than full array for large datasets)
            # Limit to last 5000 scores to avoid database bloat
            max_stored = 5000
            if n > max_stored:
                # Keep most recent scores
                stored_zscores = zscores[-max_stored:]
            else:
                stored_zscores = zscores

            # Sanitize NaN/Inf values for JSON (JSON standard doesn't support NaN/Infinity)
            def sanitize_for_json(val):
                """Convert NaN/Infinity to None for valid JSON."""
                try:
                    f = float(val)
                    return None if (np.isnan(f) or np.isinf(f)) else f
                except (TypeError, ValueError):
                    return None

            raw_zscores_data = {
                "scores": [sanitize_for_json(z) for z in stored_zscores],
                "total_count": n,
                "truncated": n > max_stored,
            }

        # Create instance
        return cls(
            distribution_date=datetime.now(),
            sample_size=n,
            universe_description=universe_description,
            mean=mean,
            std=std,
            median=median,
            skewness=skewness,
            kurtosis=kurtosis,
            p20=p20,
            p40=p40,
            p60=p60,
            p80=p80,
            p20_deviation=p20_deviation,
            p40_deviation=p40_deviation,
            p60_deviation=p60_deviation,
            p80_deviation=p80_deviation,
            large_gain_pct=large_gain_pct,
            small_gain_pct=small_gain_pct,
            neutral_pct=neutral_pct,
            small_decline_pct=small_decline_pct,
            large_decline_pct=large_decline_pct,
            max_bucket_deviation=max_bucket_deviation,
            is_valid=is_valid,
            is_active=True,
            validation_notes="; ".join(validation_notes) if validation_notes else None,
            min_zscore=float(np.min(z_array)),
            max_zscore=float(np.max(z_array)),
            raw_zscores=raw_zscores_data,
        )
