"""Task 3 — ``ConstraintSet.to_mean_risk_config`` maps MiFID knobs onto the optimizer.

fund is the bridge, so this is the one place a schema imports ``optimizer``
(lazily, inside the method). These tests assert the mapping field-by-field, that
``_OBJECTIVE_MAP`` / ``_RISK_MEASURE_MAP`` are **total** (every fund enum member
maps — no silent default / ``KeyError``) and drift-free (each mapped string is a
real optimizer enum value), and that ``build_mean_risk`` accepts the result.

Design decision (2026-09-17, user sign-off): ``beta`` maps onto ``cvar_beta`` /
``cdar_beta`` so a CVaR/CDaR profile honours the client's chosen tail confidence
(D34 "β da tolleranza"), rather than silently defaulting to 0.95.
"""

from __future__ import annotations

import pytest
from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
    build_mean_risk,
)
from skfolio.optimization import MeanRisk

from fund.schemas.constraint_set import (
    _OBJECTIVE_MAP,
    _RISK_MEASURE_MAP,
    Bounds,
    ConstraintSet,
)
from fund.schemas.enums import Horizon, ObjectiveChoice, RiskMeasureChoice


def _cs(**overrides: object) -> ConstraintSet:
    kwargs: dict[str, object] = {
        "portfolio_id": "pf-001",
        "base_currency": "EUR",
        "a_gamma": 2.5,
        "objective": ObjectiveChoice.MAX,
        "risk_measure": RiskMeasureChoice.CVAR,
        "beta": 0.99,
        "nu1": 0.05,
        "nu2": 0.10,
        "nu3": 0.20,
        "horizon": Horizon.LONG,
    }
    kwargs.update(overrides)
    return ConstraintSet(**kwargs)  # type: ignore[arg-type]


def test_returns_a_mean_risk_config():
    assert isinstance(_cs().to_mean_risk_config(), MeanRiskConfig)


def test_scalar_fields_map_field_by_field():
    cs = _cs(
        a_gamma=3.0,
        bounds=Bounds(min_weights=0.01, max_weights=0.2, budget=1.0),
        cardinality=30,
        l1_coef=0.01,
        l2_coef=0.02,
        risk_free_rate=0.03,
    )
    cfg = cs.to_mean_risk_config()
    assert cfg.risk_aversion == 3.0  # a_gamma → risk_aversion
    assert cfg.min_weights == 0.01
    assert cfg.max_weights == 0.2
    assert cfg.budget == 1.0
    assert cfg.cardinality == 30
    assert cfg.l1_coef == 0.01
    assert cfg.l2_coef == 0.02
    assert cfg.risk_free_rate == 0.03


def test_beta_maps_onto_both_tail_betas():
    cfg = _cs(beta=0.99).to_mean_risk_config()
    assert cfg.cvar_beta == 0.99
    assert cfg.cdar_beta == 0.99


def test_risk_free_rate_none_maps_to_zero():
    # MeanRiskConfig.risk_free_rate is a plain float (default 0.0), not optional.
    cfg = _cs(risk_free_rate=None).to_mean_risk_config()
    assert cfg.risk_free_rate == 0.0


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (ObjectiveChoice.PROTECTION, ObjectiveFunctionType.MINIMIZE_RISK),
        (ObjectiveChoice.INCOME, ObjectiveFunctionType.MAXIMIZE_UTILITY),
        (ObjectiveChoice.GROWTH, ObjectiveFunctionType.MAXIMIZE_UTILITY),
        (ObjectiveChoice.MAX, ObjectiveFunctionType.MAXIMIZE_RATIO),
    ],
)
def test_objective_mapping(choice: ObjectiveChoice, expected: ObjectiveFunctionType):
    cfg = _cs(objective=choice).to_mean_risk_config()
    assert cfg.objective is expected


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (RiskMeasureChoice.VARIANCE, RiskMeasureType.VARIANCE),
        (RiskMeasureChoice.SEMI_VARIANCE, RiskMeasureType.SEMI_VARIANCE),
        (RiskMeasureChoice.CVAR, RiskMeasureType.CVAR),
        (RiskMeasureChoice.CDAR, RiskMeasureType.CDAR),
        (RiskMeasureChoice.MAX_DRAWDOWN, RiskMeasureType.MAX_DRAWDOWN),
    ],
)
def test_risk_measure_mapping(choice: RiskMeasureChoice, expected: RiskMeasureType):
    cfg = _cs(risk_measure=choice).to_mean_risk_config()
    assert cfg.risk_measure is expected


def test_objective_map_is_total_and_drift_free():
    for member in ObjectiveChoice:
        assert member in _OBJECTIVE_MAP, f"unmapped {member}"
        # str value must resolve to a real optimizer enum (guards rename drift).
        ObjectiveFunctionType(_OBJECTIVE_MAP[member])


def test_risk_measure_map_is_total_and_drift_free():
    for member in RiskMeasureChoice:
        assert member in _RISK_MEASURE_MAP, f"unmapped {member}"
        RiskMeasureType(_RISK_MEASURE_MAP[member])


def test_build_mean_risk_accepts_the_mapped_config():
    for objective in ObjectiveChoice:
        for measure in RiskMeasureChoice:
            cfg = _cs(objective=objective, risk_measure=measure).to_mean_risk_config()
            estimator = build_mean_risk(cfg)
            assert isinstance(estimator, MeanRisk)
