"""Tests for tuning configs."""

from __future__ import annotations

import math

import pytest

from optimizer.scoring import ScorerConfig
from optimizer.tuning import GridSearchConfig, RandomizedSearchConfig
from optimizer.validation import WalkForwardConfig


class TestGridSearchConfig:
    def test_defaults(self) -> None:
        cfg = GridSearchConfig()
        assert isinstance(cfg.cv_config, WalkForwardConfig)
        assert isinstance(cfg.scorer_config, ScorerConfig)
        assert cfg.n_jobs is None
        assert cfg.return_train_score is False
        assert math.isnan(cfg.error_score)  # type: ignore[arg-type]
        assert cfg.refit is True
        assert cfg.verbose == 0

    def test_error_score_raise_allowed(self) -> None:
        cfg = GridSearchConfig(error_score="raise")
        assert cfg.error_score == "raise"

    def test_error_score_invalid_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="error_score"):
            GridSearchConfig(error_score="boom")

    def test_error_score_invalid_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="error_score"):
            GridSearchConfig(error_score=object())  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        cfg = GridSearchConfig()
        with pytest.raises(AttributeError):
            cfg.n_jobs = 4  # type: ignore[misc]

    def test_for_quick_search(self) -> None:
        cfg = GridSearchConfig.for_quick_search()
        assert cfg.cv_config.test_size == 21  # monthly
        assert cfg.n_jobs == -1

    def test_for_thorough_search(self) -> None:
        cfg = GridSearchConfig.for_thorough_search()
        assert cfg.cv_config.expend_train is True
        assert cfg.n_jobs == -1
        assert cfg.return_train_score is True


class TestRandomizedSearchConfig:
    def test_defaults(self) -> None:
        cfg = RandomizedSearchConfig()
        assert cfg.n_iter == 50
        assert isinstance(cfg.cv_config, WalkForwardConfig)
        assert cfg.random_state is None

    def test_frozen(self) -> None:
        cfg = RandomizedSearchConfig()
        with pytest.raises(AttributeError):
            cfg.n_iter = 10  # type: ignore[misc]

    def test_for_quick_search(self) -> None:
        cfg = RandomizedSearchConfig.for_quick_search(n_iter=30)
        assert cfg.n_iter == 30
        assert cfg.random_state == 42
        assert cfg.n_jobs == -1

    def test_for_thorough_search(self) -> None:
        cfg = RandomizedSearchConfig.for_thorough_search(n_iter=200)
        assert cfg.n_iter == 200
        assert cfg.cv_config.expend_train is True
        assert cfg.return_train_score is True

    def test_defaults_new_fields(self) -> None:
        cfg = RandomizedSearchConfig()
        assert math.isnan(cfg.error_score)  # type: ignore[arg-type]
        assert cfg.refit is True
        assert cfg.verbose == 0

    def test_n_iter_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_iter"):
            RandomizedSearchConfig(n_iter=0)

    def test_error_score_invalid_rejected(self) -> None:
        with pytest.raises(ValueError, match="error_score"):
            RandomizedSearchConfig(error_score="nope")
