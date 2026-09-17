"""Task 4 — ``fund.schemas.views`` pins the Black-Litterman view schema.

``View`` / ``ViewSet`` are pure serialisable data: constructing one imports **no**
``optimizer`` code — rendering onto ``BlackLittermanConfig`` is deferred to the
method-local import inside ``ViewSet.to_black_litterman_config``. These tests
assert validation (accept absolute + relative views, reject bad confidence), the
skfolio view-string rendering (fixed-point, no scientific notation), the JSON
round-trip invariant, that construction stays optimizer-free, and that the mapped
``BlackLittermanConfig`` is accepted by its own ``__post_init__``
(len(views) == len(confidences)).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pydantic
import pytest
from optimizer.moments import CovEstimatorType, MomentEstimationConfig, MuEstimatorType
from optimizer.views import BlackLittermanConfig, ViewUncertaintyMethod

from fund.schemas.enums import MomentsEstimator
from fund.schemas.views import View, ViewKind, ViewSet


def _abs_view(**overrides: object) -> View:
    kwargs: dict[str, object] = {
        "target": "AAPL",
        "expected_return": 0.0123,
        "confidence": 0.5,
    }
    kwargs.update(overrides)
    return View(**kwargs)  # type: ignore[arg-type]


def _view_set(**overrides: object) -> ViewSet:
    kwargs: dict[str, object] = {
        "views": (
            _abs_view(),
            _abs_view(target="MSFT", relative_to="GOOG", expected_return=0.02),
        ),
    }
    kwargs.update(overrides)
    return ViewSet(**kwargs)  # type: ignore[arg-type]


# -- validation --------------------------------------------------------------


def test_absolute_view_is_accepted():
    v = _abs_view()
    assert v.target == "AAPL"
    assert v.relative_to is None  # absolute
    assert v.kind is ViewKind.ASSET  # default
    assert v.confidence == 0.5


def test_relative_view_is_accepted():
    v = _abs_view(relative_to="MSFT", expected_return=0.02)
    assert v.relative_to == "MSFT"


def test_factor_kind_is_accepted():
    v = _abs_view(kind=ViewKind.FACTOR, target="momentum")
    assert v.kind is ViewKind.FACTOR


def test_view_and_view_set_are_frozen():
    v = _abs_view()
    with pytest.raises(pydantic.ValidationError):
        v.confidence = 0.9  # type: ignore[misc]
    vs = _view_set()
    with pytest.raises(pydantic.ValidationError):
        vs.views = ()  # type: ignore[misc]


def test_models_are_hashable():
    # frozen=True keeps the schema hashable (matches the optimizer ethos).
    assert hash(_view_set()) == hash(_view_set())


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.5, -1.0, 2.0])
def test_rejects_confidence_outside_unit_interval(bad_confidence: float):
    with pytest.raises(pydantic.ValidationError):
        _abs_view(confidence=bad_confidence)


@pytest.mark.parametrize("edge_confidence", [0.0, 1.0])
def test_accepts_confidence_at_unit_interval_edges(edge_confidence: float):
    assert _abs_view(confidence=edge_confidence).confidence == edge_confidence


# -- skfolio view-string rendering -------------------------------------------


def test_absolute_view_renders_to_skfolio_string():
    assert _abs_view(target="AAPL", expected_return=0.0123).to_view_string() == (
        "AAPL == 0.012300"
    )


def test_relative_view_renders_to_skfolio_string():
    v = _abs_view(target="AAPL", relative_to="MSFT", expected_return=0.02)
    assert v.to_view_string() == "AAPL - MSFT == 0.020000"


def test_view_string_is_fixed_point_not_scientific():
    # A tiny per-period return must not leak scientific notation to skfolio's
    # parser (mirrors optimizer.views._builder's fixed-point convention).
    rendered = _abs_view(expected_return=0.00001).to_view_string()
    assert "e" not in rendered.lower()
    assert rendered == "AAPL == 0.000010"


def test_negative_expected_return_renders_bearish_view():
    assert _abs_view(expected_return=-0.05).to_view_string() == "AAPL == -0.050000"


# -- JSON round-trip ---------------------------------------------------------


def test_json_round_trip():
    vs = _view_set(moments_estimator=MomentsEstimator.EW)
    assert ViewSet.model_validate(vs.model_dump(mode="json")) == vs


def test_json_round_trip_with_default_prior_selector():
    vs = _view_set()
    assert vs.moments_estimator is None
    assert ViewSet.model_validate(vs.model_dump(mode="json")) == vs


# -- mapping onto BlackLittermanConfig ---------------------------------------


def test_to_black_litterman_config_returns_config():
    assert isinstance(_view_set().to_black_litterman_config(), BlackLittermanConfig)


def test_maps_views_to_tuple_of_rendered_strings():
    cfg = _view_set().to_black_litterman_config()
    assert isinstance(cfg.views, tuple)
    assert cfg.views == ("AAPL == 0.012300", "MSFT - GOOG == 0.020000")


def test_maps_confidences_aligned_and_in_unit_interval():
    vs = _view_set(
        views=(
            _abs_view(confidence=0.2),
            _abs_view(target="MSFT", confidence=0.8),
        )
    )
    cfg = vs.to_black_litterman_config()
    assert isinstance(cfg.view_confidences, tuple)
    assert cfg.view_confidences == (0.2, 0.8)
    assert len(cfg.view_confidences) == len(cfg.views)
    assert all(0.0 <= c <= 1.0 for c in cfg.view_confidences)


def test_uses_idzorek_uncertainty_method():
    cfg = _view_set().to_black_litterman_config()
    assert cfg.uncertainty_method is ViewUncertaintyMethod.IDZOREK


def test_default_prior_is_equilibrium_ledoitwolf():
    cfg = _view_set().to_black_litterman_config()
    assert isinstance(cfg.prior_config, MomentEstimationConfig)
    assert cfg.prior_config == MomentEstimationConfig.for_equilibrium_ledoitwolf()
    assert cfg.prior_config.mu_estimator is MuEstimatorType.EQUILIBRIUM
    assert cfg.prior_config.cov_estimator is CovEstimatorType.LEDOIT_WOLF


@pytest.mark.parametrize(
    ("selector", "expected_cov"),
    [
        (MomentsEstimator.LEDOIT_WOLF, CovEstimatorType.LEDOIT_WOLF),
        (MomentsEstimator.EMPIRICAL, CovEstimatorType.EMPIRICAL),
        (MomentsEstimator.EW, CovEstimatorType.EW),
    ],
)
def test_prior_selector_maps_to_covariance_estimator(
    selector: MomentsEstimator, expected_cov: CovEstimatorType
):
    cfg = _view_set(moments_estimator=selector).to_black_litterman_config()
    assert cfg.prior_config is not None
    # BL always pairs the chosen covariance with EquilibriumMu.
    assert cfg.prior_config.mu_estimator is MuEstimatorType.EQUILIBRIUM
    assert cfg.prior_config.cov_estimator is expected_cov


def test_post_init_accepts_the_mapped_config():
    # BlackLittermanConfig.__post_init__ rejects a length mismatch; a clean
    # construction (no raise) proves len(views) == len(view_confidences).
    cfg = _view_set().to_black_litterman_config()
    assert len(cfg.views) == len(cfg.view_confidences)


def test_prior_selector_covers_every_moments_estimator_member():
    # The cov-estimator map is total: every fund MomentsEstimator maps without
    # a KeyError (guards accidental enum rename drift).
    for member in MomentsEstimator:
        cfg = _view_set(moments_estimator=member).to_black_litterman_config()
        assert cfg.prior_config is not None


def test_construction_does_not_import_optimizer():
    # The optimizer import is method-local (inside to_black_litterman_config);
    # merely building a ViewSet must not drag optimizer into sys.modules.
    code = textwrap.dedent(
        """
        import sys
        from fund.schemas.views import View, ViewSet
        ViewSet(
            views=(
                View(target="AAPL", expected_return=0.01, confidence=0.5),
                View(target="MSFT", relative_to="GOOG",
                     expected_return=0.02, confidence=0.3),
            ),
        )
        leaked = sorted(m for m in sys.modules if m.split(".")[0] == "optimizer")
        assert not leaked, leaked
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
