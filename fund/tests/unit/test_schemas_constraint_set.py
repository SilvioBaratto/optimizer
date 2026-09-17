"""Task 3 — ``fund.schemas.constraint_set`` pins the MiFID risk-profile schema.

``ConstraintSet`` (+ nested ``Bounds`` / ``EsgPolicy`` / ``UniverseFilters``) is
pure serialisable data: constructing one imports **no** ``optimizer`` code — the
mapping onto ``MeanRiskConfig`` is deferred to the method-local import inside
``to_mean_risk_config`` (covered by ``test_constraint_mapping.py``). These tests
assert validation (accept good, reject bad), the JSON round-trip invariant, and
that construction stays optimizer-free.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pydantic
import pytest

from fund.schemas.constraint_set import (
    Bounds,
    ConstraintSet,
    EsgPolicy,
    UniverseFilters,
)
from fund.schemas.enums import (
    GicsSector,
    Horizon,
    MomentsEstimator,
    ObjectiveChoice,
    RiskMeasureChoice,
    UncertaintyLevel,
)


def _valid_constraint_set(**overrides: object) -> ConstraintSet:
    kwargs: dict[str, object] = {
        "portfolio_id": "pf-001",
        "base_currency": "EUR",
        "a_gamma": 2.0,
        "objective": ObjectiveChoice.GROWTH,
        "risk_measure": RiskMeasureChoice.CVAR,
        "beta": 0.95,
        "nu1": 0.05,
        "nu2": 0.10,
        "nu3": 0.20,
        "horizon": Horizon.LONG,
    }
    kwargs.update(overrides)
    return ConstraintSet(**kwargs)  # type: ignore[arg-type]


def test_valid_input_is_accepted():
    cs = _valid_constraint_set()
    assert cs.portfolio_id == "pf-001"
    assert cs.base_currency == "EUR"
    assert cs.a_gamma == 2.0
    assert cs.objective is ObjectiveChoice.GROWTH
    assert cs.risk_measure is RiskMeasureChoice.CVAR
    assert cs.beta == 0.95


def test_nested_defaults_are_long_only_fully_invested():
    cs = _valid_constraint_set()
    assert cs.bounds == Bounds()
    assert cs.bounds.min_weights == 0.0  # long-only (D17)
    assert cs.bounds.max_weights == 1.0
    assert cs.bounds.budget == 1.0  # fully invested
    assert cs.esg == EsgPolicy()
    assert cs.universe_filters == UniverseFilters()
    assert cs.moments_estimator is MomentsEstimator.LEDOIT_WOLF  # D23 default
    assert cs.uncertainty_level is UncertaintyLevel.NONE  # D35 default
    assert cs.cardinality is None
    assert cs.risk_free_rate is None


def test_every_model_is_frozen():
    cs = _valid_constraint_set()
    with pytest.raises(pydantic.ValidationError):
        cs.a_gamma = 1.0  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        cs.bounds.min_weights = 0.5  # type: ignore[misc]


def test_models_are_hashable():
    # frozen=True keeps the schema hashable (matches the optimizer ethos).
    assert hash(_valid_constraint_set()) == hash(_valid_constraint_set())


@pytest.mark.parametrize("bad_currency", ["eur", "EURO", "EU", "E1R", "US$"])
def test_rejects_non_iso_currency(bad_currency: str):
    with pytest.raises(pydantic.ValidationError):
        _valid_constraint_set(base_currency=bad_currency)


@pytest.mark.parametrize("bad_gamma", [0.0, -1.0])
def test_rejects_non_positive_a_gamma(bad_gamma: float):
    with pytest.raises(pydantic.ValidationError):
        _valid_constraint_set(a_gamma=bad_gamma)


@pytest.mark.parametrize("bad_beta", [0.0, 1.0, -0.1, 1.5])
def test_rejects_beta_outside_open_unit_interval(bad_beta: float):
    with pytest.raises(pydantic.ValidationError):
        _valid_constraint_set(beta=bad_beta)


def test_rejects_negative_min_weights_long_only():
    with pytest.raises(pydantic.ValidationError):
        _valid_constraint_set(bounds=Bounds(min_weights=-0.01))


@pytest.mark.parametrize("bad_cardinality", [0, -5])
def test_rejects_non_positive_cardinality(bad_cardinality: int):
    with pytest.raises(pydantic.ValidationError):
        _valid_constraint_set(cardinality=bad_cardinality)


@pytest.mark.parametrize("bad_coef", [-0.01, -1.0])
def test_rejects_negative_regularisation(bad_coef: float):
    with pytest.raises(pydantic.ValidationError):
        _valid_constraint_set(l1_coef=bad_coef)
    with pytest.raises(pydantic.ValidationError):
        _valid_constraint_set(l2_coef=bad_coef)


@pytest.mark.parametrize("bad_nu", [-0.01, 1.01])
def test_rejects_nu_tiers_outside_capital_fraction(bad_nu: float):
    # nu1/nu2/nu3 are drawdown ceilings as fractions of capital: [0, 1].
    with pytest.raises(pydantic.ValidationError):
        _valid_constraint_set(nu1=bad_nu)


def test_rejects_unknown_gics_sector_in_esg_exclusions():
    with pytest.raises(pydantic.ValidationError):
        EsgPolicy(exclusions=("not_a_sector",))  # type: ignore[arg-type]


def test_accepts_known_gics_sector_in_esg_exclusions():
    esg = EsgPolicy(exclusions=(GicsSector.ENERGY, GicsSector.UTILITIES))
    assert GicsSector.ENERGY in esg.exclusions


def test_json_round_trip():
    cs = _valid_constraint_set(
        esg=EsgPolicy(
            exclusions=(GicsSector.ENERGY,),
            min_taxonomy=0.3,
            pai_flags=("carbon",),
        ),
        universe_filters=UniverseFilters(no_complex=True, no_leverage=True),
        bounds=Bounds(min_weights=0.01, max_weights=0.1),
        cardinality=25,
        l1_coef=0.01,
        l2_coef=0.02,
        risk_free_rate=0.03,
    )
    assert ConstraintSet.model_validate(cs.model_dump(mode="json")) == cs


def test_construction_does_not_import_optimizer():
    # The optimizer import is method-local (inside to_mean_risk_config); merely
    # building a ConstraintSet must not drag optimizer into sys.modules.
    code = textwrap.dedent(
        """
        import sys
        from fund.schemas.constraint_set import ConstraintSet
        from fund.schemas.enums import ObjectiveChoice, RiskMeasureChoice, Horizon
        ConstraintSet(
            portfolio_id="pf-001", base_currency="EUR", a_gamma=2.0,
            objective=ObjectiveChoice.GROWTH, risk_measure=RiskMeasureChoice.CVAR,
            nu1=0.05, nu2=0.10, nu3=0.20, horizon=Horizon.LONG,
        )
        leaked = sorted(m for m in sys.modules if m.split(".")[0] == "optimizer")
        assert not leaked, leaked
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
