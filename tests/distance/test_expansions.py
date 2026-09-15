"""Tests for the expanded distance config/factory surface (skfolio 1.0.6)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from skfolio.distance import (
    CovarianceDistance,
    DistanceCorrelation,
    MutualInformation,
    PearsonDistance,
)
from skfolio.distance import (
    NBinsMethod as SkNBinsMethod,
)
from skfolio.moments import LedoitWolf

from optimizer.distance import (
    DistanceConfig,
    DistanceEstimatorType,
    NBinsMethod,
    build_distance,
)
from optimizer.exceptions import ConfigurationError


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    data = rng.normal(loc=0.0004, scale=0.011, size=(220, 6))
    return pd.DataFrame(data, columns=[f"A{i:02d}" for i in range(6)])


class TestNBinsMethodEnum:
    def test_when_listed_then_matches_skfolio_values(self) -> None:
        assert {m.value for m in NBinsMethod} == {m.value for m in SkNBinsMethod}


class TestAbsolutePower:
    def test_when_default_then_absolute_false_power_one(self) -> None:
        cfg = DistanceConfig()
        assert cfg.absolute is False
        assert cfg.power == 1.0

    def test_when_pearson_absolute_power_then_forwarded(self) -> None:
        cfg = DistanceConfig.for_pearson(absolute=True, power=2.0)
        est = build_distance(cfg)
        assert isinstance(est, PearsonDistance)
        assert est.absolute is True
        assert est.power == 2.0

    def test_when_absolute_on_non_corr_family_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="correlation/covariance"):
            DistanceConfig(
                estimator=DistanceEstimatorType.DISTANCE_CORRELATION,
                absolute=True,
            )

    def test_when_power_on_mi_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="correlation/covariance"):
            DistanceConfig(
                estimator=DistanceEstimatorType.MUTUAL_INFORMATION,
                power=2.0,
            )

    def test_when_power_non_positive_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="power must be"):
            DistanceConfig.for_pearson(power=0.0)

    def test_when_absolute_true_then_negative_corr_becomes_close(
        self, returns: pd.DataFrame
    ) -> None:
        # Build an anti-correlated pair and check absolute collapses distance.
        x = returns.copy()
        x["A05"] = -x["A00"]
        plain = build_distance(DistanceConfig.for_pearson()).fit(x)
        abso = build_distance(DistanceConfig.for_pearson(absolute=True)).fit(x)
        i, j = x.columns.get_loc("A00"), x.columns.get_loc("A05")
        assert plain.distance_[i, j] > abso.distance_[i, j]


class TestCovarianceEstimatorKwarg:
    def test_when_supplied_for_covariance_then_forwarded(
        self, returns: pd.DataFrame
    ) -> None:
        lw = LedoitWolf()
        est = build_distance(DistanceConfig.for_covariance(), covariance_estimator=lw)
        assert isinstance(est, CovarianceDistance)
        assert est.covariance_estimator is lw
        est.fit(returns)
        assert est.distance_.shape == (returns.shape[1],) * 2

    def test_when_supplied_for_non_covariance_then_raises(self) -> None:
        with pytest.raises(ValueError, match="COVARIANCE"):
            build_distance(
                DistanceConfig.for_pearson(), covariance_estimator=LedoitWolf()
            )

    def test_when_covariance_default_then_none_estimator(self) -> None:
        est = build_distance(DistanceConfig.for_covariance())
        assert isinstance(est, CovarianceDistance)
        assert est.covariance_estimator is None


class TestDistanceCorrelationThreshold:
    def test_when_threshold_set_then_forwarded(self) -> None:
        cfg = DistanceConfig.for_distance_correlation(threshold=0.3)
        est = build_distance(cfg)
        assert isinstance(est, DistanceCorrelation)
        assert est.threshold == 0.3

    def test_when_threshold_none_then_skfolio_default(self) -> None:
        est = build_distance(DistanceConfig.for_distance_correlation())
        assert isinstance(est, DistanceCorrelation)
        assert est.threshold == DistanceCorrelation().threshold

    def test_when_threshold_out_of_range_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match=r"\[0, 1\]"):
            DistanceConfig.for_distance_correlation(threshold=1.5)

    def test_when_threshold_on_non_dcorr_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="DISTANCE_CORRELATION"):
            DistanceConfig(estimator=DistanceEstimatorType.PEARSON, threshold=0.5)


class TestMutualInformationKnobs:
    def test_when_n_bins_method_set_then_forwarded(self) -> None:
        cfg = DistanceConfig.for_mutual_information(
            n_bins=None, n_bins_method=NBinsMethod.KNUTH
        )
        est = build_distance(cfg)
        assert isinstance(est, MutualInformation)
        assert est.n_bins_method == SkNBinsMethod.KNUTH
        assert est.n_bins is None

    def test_when_normalize_set_then_forwarded(self) -> None:
        cfg = DistanceConfig.for_mutual_information(normalize=False)
        est = build_distance(cfg)
        assert isinstance(est, MutualInformation)
        assert est.normalize is False

    def test_when_normalize_unset_then_skfolio_default(self) -> None:
        est = build_distance(DistanceConfig.for_mutual_information())
        assert isinstance(est, MutualInformation)
        assert est.normalize == MutualInformation().normalize

    def test_when_n_bins_method_on_non_mi_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="MUTUAL_INFORMATION"):
            DistanceConfig(
                estimator=DistanceEstimatorType.PEARSON,
                n_bins_method=NBinsMethod.KNUTH,
            )

    def test_when_normalize_on_non_mi_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="MUTUAL_INFORMATION"):
            DistanceConfig(estimator=DistanceEstimatorType.KENDALL, normalize=True)

    def test_when_n_bins_non_positive_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="n_bins must be"):
            DistanceConfig.for_mutual_information(n_bins=0)

    def test_when_fitted_with_knuth_then_square_matrix(
        self, returns: pd.DataFrame
    ) -> None:
        cfg = DistanceConfig.for_mutual_information(
            n_bins=None, n_bins_method=NBinsMethod.KNUTH
        )
        est = build_distance(cfg)
        est.fit(returns)
        assert est.distance_.shape == (returns.shape[1],) * 2


class TestSerialisation:
    def test_when_asdict_then_only_primitives_and_enums(self) -> None:
        cfg = DistanceConfig.for_pearson(absolute=True, power=1.5)
        d = dataclasses.asdict(cfg)
        assert d["absolute"] is True
        assert d["power"] == 1.5
        assert isinstance(d["estimator"], DistanceEstimatorType)

    def test_when_frozen_then_immutable(self) -> None:
        cfg = DistanceConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.power = 2.0  # type: ignore[misc]


class TestCodependenceAttr:
    def test_when_fitted_then_codependence_present(self, returns: pd.DataFrame) -> None:
        est = build_distance(DistanceConfig.for_pearson()).fit(returns)
        assert est.codependence_.shape == (returns.shape[1],) * 2


class TestDbDataCompatibility:
    """Lock the input contract exercised by real ingestion data.

    The DB serves 5y ragged history (NaN gaps, unequal-length series) and
    ``Numeric`` columns as ``Decimal``. These tests characterise what the
    skfolio distance estimators require so the builder's documented contract
    cannot silently drift.
    """

    @pytest.mark.parametrize(
        "estimator_type",
        list(DistanceEstimatorType),
    )
    def test_when_nan_present_then_fit_raises(
        self, estimator_type: DistanceEstimatorType, returns: pd.DataFrame
    ) -> None:
        # A late-listed asset (NaN gap) must be cleaned upstream: every
        # skfolio distance estimator rejects NaN via sklearn validation.
        if estimator_type == DistanceEstimatorType.MUTUAL_INFORMATION:
            cfg = DistanceConfig.for_mutual_information(n_bins=8)
        else:
            cfg = DistanceConfig(estimator=estimator_type)
        ragged = returns.copy()
        ragged.iloc[:40, ragged.columns.get_loc("A05")] = np.nan
        est = build_distance(cfg)
        with pytest.raises(ValueError, match="NaN"):
            est.fit(ragged)

    def test_when_covariance_default_fitted_then_gerber(
        self, returns: pd.DataFrame
    ) -> None:
        # Documents the skfolio default: GerberCovariance (outlier-robust),
        # not EmpiricalCovariance.
        from skfolio.moments import GerberCovariance

        est = build_distance(DistanceConfig.for_covariance()).fit(returns)
        assert isinstance(est.covariance_estimator_, GerberCovariance)

    def test_when_decimal_cast_to_float_then_fit_succeeds(
        self, returns: pd.DataFrame
    ) -> None:
        # Numeric->Decimal columns are not numpy-friendly; float-cast is the
        # reader's job. Once cast, fitting works.
        from decimal import Decimal

        decimal_col = returns["A00"].map(lambda v: Decimal(str(v)))
        as_float = decimal_col.astype(float)
        x = returns.copy()
        x["A00"] = as_float
        est = build_distance(DistanceConfig.for_pearson()).fit(x)
        assert est.distance_.shape == (x.shape[1],) * 2
