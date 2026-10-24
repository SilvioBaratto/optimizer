"""Tests for make_cleaning_pipeline factory."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from optimizer.exceptions import DataError
from optimizer.preprocessing import (
    CleaningConfig,
    DataValidator,
    ImputerStrategy,
    OutlierTreater,
    RegressionImputer,
    SectorImputer,
    apply_delisting_returns,
    delisting_protection_mask,
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


class TestConfigValidation:
    @pytest.mark.parametrize("max_abs_return", [0.0, -1.0, float("nan")])
    def test_non_positive_max_abs_return_raises(self, max_abs_return: float) -> None:
        with pytest.raises(ValueError, match="max_abs_return"):
            CleaningConfig(max_abs_return=max_abs_return)

    @pytest.mark.parametrize(("winsorize", "remove"), [(5.0, 3.0), (3.0, 3.0)])
    def test_winsorize_not_below_remove_raises(
        self, winsorize: float, remove: float
    ) -> None:
        with pytest.raises(ValueError, match="winsorize_threshold"):
            CleaningConfig(winsorize_threshold=winsorize, remove_threshold=remove)

    @pytest.mark.parametrize("winsorize", [0.0, -1.0])
    def test_non_positive_winsorize_threshold_raises(self, winsorize: float) -> None:
        with pytest.raises(ValueError, match="winsorize_threshold"):
            CleaningConfig(winsorize_threshold=winsorize)

    def test_negative_n_neighbors_raises(self) -> None:
        with pytest.raises(ValueError, match="n_neighbors"):
            CleaningConfig(n_neighbors=-1)

    @pytest.mark.parametrize(("n_neighbors", "min_train"), [(5, 6), (5, 3), (0, 1)])
    def test_min_train_periods_must_exceed_regression_parameters(
        self, n_neighbors: int, min_train: int
    ) -> None:
        with pytest.raises(ValueError, match="min_train_periods"):
            CleaningConfig(n_neighbors=n_neighbors, min_train_periods=min_train)

    def test_boundary_values_accepted(self) -> None:
        cfg = CleaningConfig(n_neighbors=5, min_train_periods=7)
        assert cfg.min_train_periods == 7

    def test_inverted_thresholds_are_not_silently_accepted_by_factory(self) -> None:
        with pytest.raises(ValueError, match="winsorize_threshold"):
            make_cleaning_pipeline(
                CleaningConfig(winsorize_threshold=5.0, remove_threshold=3.0)
            )


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

    @pytest.mark.parametrize(
        "imputer", [ImputerStrategy.SECTOR, ImputerStrategy.REGRESSION]
    )
    def test_row_missing_for_every_asset_is_left_missing(
        self, returns: pd.DataFrame, imputer: ImputerStrategy
    ) -> None:
        # The standalone cleaning pipeline keeps a no-observation row as NaN;
        # only the pre-selection pipeline insists on a fillable frame.
        returns.iloc[60, :] = np.nan
        cfg = CleaningConfig(imputer=imputer, n_neighbors=2, min_train_periods=20)

        out = make_cleaning_pipeline(cfg).fit_transform(returns)

        assert out.iloc[60].isna().all()
        assert out.drop(out.index[60]).notna().all().all()

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


class TestStructuralGaps:
    """A cell before an asset's first or after its last observation is not missing.

    It marks the time before listing or after delisting; filling it invents a
    return history the asset never had and defeats every later history-length
    check.
    """

    @staticmethod
    def _panel() -> pd.DataFrame:
        rng = np.random.default_rng(5)
        common = rng.normal(0.0, 0.01, 500)
        df = pd.DataFrame(
            {n: common + rng.normal(0.0, 0.005, 500) for n in "ABCDEF"},
            index=pd.bdate_range("2020-01-01", periods=500),
        )
        df.loc[df.index[:400], "E"] = np.nan  # lists on day 400
        df.loc[df.index[300:], "F"] = np.nan  # delisted after day 300
        df.loc[df.index[100], "A"] = np.nan  # an ordinary missing day
        return df

    @pytest.mark.parametrize(
        "imputer", [ImputerStrategy.SECTOR, ImputerStrategy.REGRESSION]
    )
    def test_pre_listing_and_post_delisting_cells_stay_missing(
        self, imputer: ImputerStrategy
    ) -> None:
        cfg = CleaningConfig(imputer=imputer, n_neighbors=2, min_train_periods=20)
        mapping = dict.fromkeys("ABCDEF", "S")

        out = make_cleaning_pipeline(cfg, sector_mapping=mapping).fit_transform(
            self._panel()
        )

        assert out["E"].iloc[:400].isna().all()
        assert out["E"].iloc[400:].notna().all()
        assert out["F"].iloc[300:].isna().all()
        assert out["F"].iloc[:300].notna().all()

    @pytest.mark.parametrize(
        "imputer", [ImputerStrategy.SECTOR, ImputerStrategy.REGRESSION]
    )
    def test_interior_gaps_are_still_filled(self, imputer: ImputerStrategy) -> None:
        cfg = CleaningConfig(imputer=imputer, n_neighbors=2, min_train_periods=20)

        out = make_cleaning_pipeline(cfg).fit_transform(self._panel())

        assert out["A"].notna().all()


class TestTransformTimeColumns:
    """Column-aware transformers align by label and never relabel positionally."""

    @pytest.fixture()
    def frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        scales = {"A": 0.005, "B": 0.01, "C": 0.02, "D": 0.03}
        return pd.DataFrame(
            {name: rng.normal(0.0, sd, 120) for name, sd in scales.items()},
            index=pd.date_range("2021-01-01", periods=120),
        )

    makers = pytest.mark.parametrize(
        "make",
        [DataValidator, OutlierTreater, SectorImputer, RegressionImputer],
        ids=["validator", "outliers", "sector_imputer", "regression_imputer"],
    )

    @makers
    def test_reordered_columns_are_realigned_by_label(
        self, frame: pd.DataFrame, make: type
    ) -> None:
        fitted = make().fit(frame)
        out = fitted.transform(frame[["D", "C", "B", "A"]])
        assert list(out.columns) == list(frame.columns)
        pd.testing.assert_frame_equal(out, fitted.transform(frame))

    @makers
    def test_pandas_output_does_not_scramble_reordered_columns(
        self, frame: pd.DataFrame, make: type
    ) -> None:
        pipe = Pipeline([("step", make())]).set_output(transform="pandas")
        pipe.fit(frame)
        out = pipe.transform(frame[["D", "C", "B", "A"]])
        pd.testing.assert_frame_equal(out, pipe.transform(frame))

    @makers
    def test_missing_column_raises(self, frame: pd.DataFrame, make: type) -> None:
        fitted = make().fit(frame)
        with pytest.raises(DataError, match=r"missing.*'C'"):
            fitted.transform(frame[["A", "B", "D"]])

    @makers
    def test_unseen_column_raises(self, frame: pd.DataFrame, make: type) -> None:
        fitted = make().fit(frame[["A", "B", "C"]])
        with pytest.raises(DataError, match=r"unexpected.*'D'"):
            fitted.transform(frame[["A", "B", "D"]])

    @makers
    def test_same_width_different_universe_is_not_relabelled(
        self, frame: pd.DataFrame, make: type
    ) -> None:
        pipe = Pipeline([("step", make())]).set_output(transform="pandas")
        pipe.fit(frame[["A", "B", "C"]])
        with pytest.raises(DataError):
            pipe.transform(frame[["A", "B", "D"]])

    @makers
    def test_feature_names_out_defaults_to_fit_names(
        self, frame: pd.DataFrame, make: type
    ) -> None:
        fitted = make().fit(frame)
        np.testing.assert_array_equal(fitted.get_feature_names_out(), frame.columns)
        np.testing.assert_array_equal(
            fitted.get_feature_names_out(list(frame.columns)), frame.columns
        )

    @makers
    def test_feature_names_out_rejects_names_unlike_fit(
        self, frame: pd.DataFrame, make: type
    ) -> None:
        # Echoing the fit names for a different ``input_features`` would let
        # sklearn relabel foreign columns positionally.
        fitted = make().fit(frame)
        with pytest.raises(ValueError, match="input_features"):
            fitted.get_feature_names_out(["D", "C", "B", "A"])
        with pytest.raises(ValueError, match="input_features"):
            fitted.get_feature_names_out(["A", "B", "C"])


class TestOutlierProtection:
    @staticmethod
    def _delisted_panel() -> pd.DataFrame:
        rng = np.random.default_rng(11)
        df = pd.DataFrame(
            rng.normal(0.0, 0.015, size=(500, 4)),
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=500, freq="B"),
        )
        df.iloc[300:, df.columns.get_loc("D")] = np.nan
        return df

    @pytest.mark.parametrize("imputer", [ImputerStrategy.NONE, ImputerStrategy.SECTOR])
    def test_delisting_terminal_return_survives_cleaning(
        self, imputer: ImputerStrategy
    ) -> None:
        raw = self._delisted_panel()
        delisting = {"D": -0.30}
        mask = delisting_protection_mask(raw, delisting)
        applied = apply_delisting_returns(raw, delisting)

        pipe = make_cleaning_pipeline(
            CleaningConfig(imputer=imputer), outlier_protection_mask=mask
        )
        out = pipe.fit_transform(applied)

        terminal = raw["D"].last_valid_index()
        assert out.at[terminal, "D"] == pytest.approx(-0.30)

    def test_mask_is_forwarded_to_outlier_treater(self) -> None:
        mask = pd.DataFrame(False, index=pd.RangeIndex(3), columns=["A"], dtype=bool)
        pipe = make_cleaning_pipeline(outlier_protection_mask=mask)
        assert pipe.named_steps["outliers"].protected_mask is mask

    def test_no_mask_protects_nothing_by_default(self) -> None:
        pipe = make_cleaning_pipeline()
        assert pipe.named_steps["outliers"].protected_mask is None
