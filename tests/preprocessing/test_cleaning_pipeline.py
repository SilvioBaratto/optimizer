"""Tests for make_cleaning_pipeline factory."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from optimizer.preprocessing import (
    CleaningConfig,
    DataValidator,
    ImputerStrategy,
    OutlierTreater,
    RegressionImputer,
    SectorImputer,
    make_cleaning_pipeline,
)


@pytest.fixture()
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    common = rng.standard_normal(150)
    df = pd.DataFrame(
        {
            "A": 0.8 * common + 0.2 * rng.standard_normal(150),
            "B": 0.7 * common + 0.3 * rng.standard_normal(150),
            "C": 0.6 * common + 0.4 * rng.standard_normal(150),
        },
        index=pd.date_range("2020-01-01", periods=150),
    )
    df.iloc[10, 0] = np.inf  # data error
    df.iloc[20, 1] = np.nan  # missing
    return df


class TestConfig:
    def test_frozen(self) -> None:
        cfg = CleaningConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.validate = False  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = CleaningConfig()
        assert cfg.validate is True
        assert cfg.treat_outliers is True
        assert cfg.imputer is ImputerStrategy.NONE


class TestFactory:
    def test_default_pipeline_steps(self) -> None:
        pipe = make_cleaning_pipeline()
        assert isinstance(pipe, Pipeline)
        names = [n for n, _ in pipe.steps]
        assert names == ["validate", "outliers"]

    def test_validate_step_type(self) -> None:
        pipe = make_cleaning_pipeline()
        assert isinstance(pipe.named_steps["validate"], DataValidator)
        assert isinstance(pipe.named_steps["outliers"], OutlierTreater)

    def test_sector_imputer_step(self) -> None:
        cfg = CleaningConfig(imputer=ImputerStrategy.SECTOR)
        mapping = {"A": "S", "B": "S", "C": "S"}
        pipe = make_cleaning_pipeline(cfg, sector_mapping=mapping)
        step = pipe.named_steps["impute"]
        assert isinstance(step, SectorImputer)
        assert step.sector_mapping == mapping

    def test_regression_imputer_step(self) -> None:
        cfg = CleaningConfig(
            imputer=ImputerStrategy.REGRESSION, n_neighbors=2, min_train_periods=20
        )
        pipe = make_cleaning_pipeline(cfg)
        step = pipe.named_steps["impute"]
        assert isinstance(step, RegressionImputer)
        assert step.n_neighbors == 2

    def test_pipeline_cleans_returns(self, returns: pd.DataFrame) -> None:
        cfg = CleaningConfig(
            imputer=ImputerStrategy.REGRESSION, n_neighbors=2, min_train_periods=20
        )
        pipe = make_cleaning_pipeline(cfg)
        out = pipe.fit_transform(returns)
        # inf coerced away and NaN imputed
        assert not np.isinf(out.to_numpy()).any()
        assert not out.isna().any().any()

    def test_thresholds_forwarded(self) -> None:
        cfg = CleaningConfig(
            max_abs_return=5.0, winsorize_threshold=2.5, remove_threshold=8.0
        )
        pipe = make_cleaning_pipeline(cfg)
        assert pipe.named_steps["validate"].max_abs_return == 5.0
        assert pipe.named_steps["outliers"].winsorize_threshold == 2.5
        assert pipe.named_steps["outliers"].remove_threshold == 8.0

    def test_empty_pipeline_raises(self) -> None:
        cfg = CleaningConfig(
            validate=False, treat_outliers=False, imputer=ImputerStrategy.NONE
        )
        with pytest.raises(ValueError, match="empty"):
            make_cleaning_pipeline(cfg)

    def test_only_imputer_step(self) -> None:
        cfg = CleaningConfig(
            validate=False, treat_outliers=False, imputer=ImputerStrategy.SECTOR
        )
        pipe = make_cleaning_pipeline(cfg)
        assert [n for n, _ in pipe.steps] == ["impute"]

    def test_default_pipeline_handles_decimal_returns(self) -> None:
        # Regression: DB Decimal (object dtype) returns fed to the default
        # validate -> outliers pipeline previously crashed at OutlierTreater.
        rng = np.random.default_rng(7)
        floats = rng.normal(scale=0.02, size=(120, 3))
        df = pd.DataFrame(
            [[Decimal(str(v)) for v in row] for row in floats],
            columns=["A", "B", "C"],
            index=pd.date_range("2020-01-01", periods=120),
        )
        assert (df.dtypes == "object").all()
        out = make_cleaning_pipeline().fit_transform(df)
        assert (out.dtypes == np.float64).all()
        assert not np.isinf(out.to_numpy()).any()
