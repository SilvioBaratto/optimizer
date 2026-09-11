"""Tests for the cross-sectional transformer config + factory."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from skfolio.preprocessing import (
    CSGaussianRankScaler,
    CSPercentileRankScaler,
    CSStandardScaler,
    CSTanhShrinker,
    CSWinsorizer,
)

from optimizer.preprocessing import (
    CSTransformerConfig,
    CSTransformerType,
    make_cs_transformer,
)


@pytest.fixture()
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(40, 12)))


class TestConfig:
    def test_default_is_standard(self) -> None:
        assert CSTransformerConfig().transformer is CSTransformerType.STANDARD

    def test_frozen(self) -> None:
        cfg = CSTransformerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.min_group_size = 2  # type: ignore[misc]

    def test_hashable(self) -> None:
        assert hash(CSTransformerConfig()) == hash(CSTransformerConfig())


class TestFactory:
    def test_none_builds_standard(self) -> None:
        assert isinstance(make_cs_transformer(), CSStandardScaler)

    @pytest.mark.parametrize(
        ("kind", "cls"),
        [
            (CSTransformerType.STANDARD, CSStandardScaler),
            (CSTransformerType.WINSORIZER, CSWinsorizer),
            (CSTransformerType.GAUSSIAN_RANK, CSGaussianRankScaler),
            (CSTransformerType.PERCENTILE_RANK, CSPercentileRankScaler),
            (CSTransformerType.TANH_SHRINKER, CSTanhShrinker),
        ],
    )
    def test_dispatch(self, kind: CSTransformerType, cls: type) -> None:
        t = make_cs_transformer(CSTransformerConfig(transformer=kind))
        assert isinstance(t, cls)

    def test_winsorizer_params_forwarded(self) -> None:
        t = make_cs_transformer(
            CSTransformerConfig(
                transformer=CSTransformerType.WINSORIZER, low=0.05, high=0.95
            )
        )
        assert t.low == 0.05
        assert t.high == 0.95

    def test_gaussian_scale_forwarded(self) -> None:
        t = make_cs_transformer(
            CSTransformerConfig(
                transformer=CSTransformerType.GAUSSIAN_RANK, scale=False
            )
        )
        assert t.scale is False

    def test_min_group_size_forwarded(self) -> None:
        t = make_cs_transformer(
            CSTransformerConfig(
                transformer=CSTransformerType.PERCENTILE_RANK, min_group_size=3
            )
        )
        assert t.min_group_size == 3

    def test_tanh_knee_forwarded(self) -> None:
        t = make_cs_transformer(
            CSTransformerConfig(transformer=CSTransformerType.TANH_SHRINKER, knee=1.5)
        )
        assert t.knee == 1.5

    def test_built_transformer_runs(self, panel: pd.DataFrame) -> None:
        t = make_cs_transformer(
            CSTransformerConfig(transformer=CSTransformerType.WINSORIZER)
        )
        out = t.fit_transform(panel)
        assert np.asarray(out).shape == panel.shape
