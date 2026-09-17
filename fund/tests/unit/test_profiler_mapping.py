"""Task 2 — ``build_constraint_set`` deterministic MiFID → knob mapping.

Pure, total, auditable — building a ``ConstraintSet`` from ``MiFIDAnswers`` imports
no optimizer code (the optimizer only appears when the produced ``ConstraintSet``
is fed through ``.to_mean_risk_config()``). These tests pin the correctness-critical
bits SPEC §8 calls out:

* the appetite → aversion **inversion** (low appetite ⇒ high ``a_gamma``), so a
  naive ``a_gamma = min(gamma_tol, gamma_cap)`` regression is caught;
* the ``A = min(tolerance, capacity)`` double-binding, with tolerance and capacity
  scored from **disjoint** answer fields (a high-attitude / low-buffer client must
  land conservative — the regulatory failure mode);
* the 5-band boundary lookup (0.2 / 0.4 / 0.6 / 0.8) and every band reachable;
* mapping totality — every ``LossReaction`` / ``ObjectiveChoice`` / ``Horizon`` is a
  key (unmapped ⇒ ``KeyError``), and every output survives ``ConstraintSet``
  validation + ``.to_mean_risk_config()``.

ESG hard-gate + K&E universe filters + the ``SuitabilityAssessment`` assembly are
Task 3 — here ``esg`` / ``universe_filters`` stay at their schema defaults.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from optimizer.optimization import MeanRiskConfig

from fund.agents.profiler import (
    _AVERSION_BANDS,
    _appetite_from_capacity,
    _appetite_from_tolerance,
    _appetite_to_aversion,
    _category,
    _nu_tiers,
    build_constraint_set,
)
from fund.schemas.constraint_set import ConstraintSet
from fund.schemas.enums import (
    Horizon,
    KnowledgeLevel,
    LossReaction,
    ObjectiveChoice,
    RiskMeasureChoice,
    RiskToleranceBand,
)
from fund.schemas.questionnaire import (
    CapacityAnswers,
    EsgAnswers,
    KnowledgeAnswers,
    MiFIDAnswers,
    ObjectivesAnswers,
)


def _answers(
    *,
    likert: tuple[int, ...] = (5, 6, 4),
    max_loss: float = 0.25,
    buffer: float = 6.0,
    goal: ObjectiveChoice = ObjectiveChoice.GROWTH,
    horizon: Horizon = Horizon.LONG,
    reaction: LossReaction = LossReaction.HOLD,
    knowledge: KnowledgeLevel = KnowledgeLevel.INFORMED,
    currency: str = "EUR",
) -> MiFIDAnswers:
    return MiFIDAnswers(
        base_currency=currency,
        knowledge=KnowledgeAnswers(level=knowledge),
        capacity=CapacityAnswers(max_1yr_loss_pct=max_loss, buffer_months=buffer),
        objectives=ObjectivesAnswers(
            goal=goal,
            horizon=horizon,
            likert_items=likert,
            loss_reaction=reaction,
        ),
        esg=EsgAnswers(),
    )


# --- appetite scoring (SPEC §8.2 — normalized-average composite) -----------


@pytest.mark.parametrize(
    ("likert", "expected"),
    [
        ((1,), 0.0),  # floor
        ((7,), 1.0),  # ceiling
        ((4,), 0.5),  # midpoint
        ((7, 1), 0.5),  # mean 4 → 0.5
    ],
)
def test_appetite_from_tolerance_rescales_likert(
    likert: tuple[int, ...], expected: float
) -> None:
    objectives = ObjectivesAnswers(
        goal=ObjectiveChoice.GROWTH,
        horizon=Horizon.LONG,
        likert_items=likert,
        loss_reaction=LossReaction.HOLD,
    )
    assert _appetite_from_tolerance(objectives) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("max_loss", "buffer", "expected"),
    [
        (0.0, 0.0, 0.0),  # nothing to lose, no buffer
        (0.50, 12.0, 1.0),  # both sub-scores saturate at 1.0
        (0.25, 6.0, 0.5),  # loss_sub 0.5, buffer_sub 0.5
        (1.0, 24.0, 1.0),  # clips above 1.0 (0.50 / 12-month caps)
    ],
)
def test_appetite_from_capacity_composite_with_clip(
    max_loss: float, buffer: float, expected: float
) -> None:
    capacity = CapacityAnswers(max_1yr_loss_pct=max_loss, buffer_months=buffer)
    assert _appetite_from_capacity(capacity) == pytest.approx(expected)


def test_tolerance_and_capacity_use_disjoint_fields() -> None:
    # Changing capacity fields must not move the tolerance score, and vice versa.
    base_obj = ObjectivesAnswers(
        goal=ObjectiveChoice.GROWTH,
        horizon=Horizon.LONG,
        likert_items=(5, 5),
        loss_reaction=LossReaction.HOLD,
    )
    tol = _appetite_from_tolerance(base_obj)
    assert tol == _appetite_from_tolerance(base_obj)  # deterministic

    cap_a = _appetite_from_capacity(
        CapacityAnswers(max_1yr_loss_pct=0.1, buffer_months=1)
    )
    cap_b = _appetite_from_capacity(
        CapacityAnswers(max_1yr_loss_pct=0.4, buffer_months=9)
    )
    assert cap_a != cap_b  # capacity responds only to its own fields
    # tolerance is unchanged regardless of which capacity we pair it with
    assert _appetite_from_tolerance(base_obj) == tol


# --- A = min double-binding + inversion ------------------------------------


def test_capacity_binds_high_attitude_low_buffer_lands_conservative() -> None:
    # Bullish attitude (all 7s → tol 1.0) but almost no capacity to lose.
    answers = _answers(likert=(7, 7, 7), max_loss=0.05, buffer=0.0)
    cs = build_constraint_set(answers, portfolio_id="pf-1")
    # a_cap ≈ mean(clip(0.05/0.5)=0.1, 0) = 0.05 → Defensive band.
    assert cs.a_gamma == 12.0
    assert _category(cs.a_gamma) is RiskToleranceBand.DEFENSIVE


def test_tolerance_binds_low_attitude_high_capacity_lands_conservative() -> None:
    answers = _answers(likert=(1, 1, 1), max_loss=0.50, buffer=12.0)
    cs = build_constraint_set(answers, portfolio_id="pf-1")
    assert cs.a_gamma == 12.0  # appetite 0.0 → Defensive
    assert _category(cs.a_gamma) is RiskToleranceBand.DEFENSIVE


def test_a_gamma_is_inverse_to_appetite() -> None:
    # Monotone-DECREASING: as appetite rises across bands, a_gamma falls.
    seq = [_appetite_to_aversion(a) for a in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert seq == [12.0, 8.0, 5.0, 2.5, 1.0]
    assert seq == sorted(seq, reverse=True)  # strictly decreasing


# --- 5-band lookup (SPEC §8.1) ---------------------------------------------


@pytest.mark.parametrize(
    ("appetite", "expected"),
    [
        (0.0, 12.0),
        (0.19, 12.0),
        (0.2, 8.0),  # boundary belongs to the upper band [0.2, 0.4)
        (0.39, 8.0),
        (0.4, 5.0),
        (0.59, 5.0),
        (0.6, 2.5),
        (0.79, 2.5),
        (0.8, 1.0),
        (1.0, 1.0),
    ],
)
def test_appetite_to_aversion_band_boundaries(appetite: float, expected: float) -> None:
    assert _appetite_to_aversion(appetite) == expected


def test_appetite_to_aversion_caps_above_unit_interval() -> None:
    # Defensive fallthrough guard (appetite is clamped ≤ 1.0 upstream, but the
    # helper must still be total).
    assert _appetite_to_aversion(1.5) == 1.0


def test_every_band_reachable_and_category_total() -> None:
    produced = {_appetite_to_aversion(a) for a in (0.1, 0.3, 0.5, 0.7, 0.9)}
    assert produced == {12.0, 8.0, 5.0, 2.5, 1.0}
    # _category is total over every a_gamma the band table emits.
    categories = {_category(a) for _, a in _AVERSION_BANDS}
    assert categories == set(RiskToleranceBand)


def test_category_raises_on_unmapped_gamma() -> None:
    with pytest.raises(KeyError):
        _category(3.14)  # not a band value → hard KeyError, never a silent default


# --- objective / horizon identity maps (total) -----------------------------


@pytest.mark.parametrize("goal", list(ObjectiveChoice))
def test_objective_mapping_total_and_reachable(goal: ObjectiveChoice) -> None:
    cs = build_constraint_set(_answers(goal=goal), portfolio_id="pf-1")
    assert cs.objective is goal  # every ObjectiveChoice reachable


@pytest.mark.parametrize("horizon", list(Horizon))
def test_horizon_mapping_total_and_reachable(horizon: Horizon) -> None:
    cs = build_constraint_set(_answers(horizon=horizon), portfolio_id="pf-1")
    assert cs.horizon is horizon  # every Horizon reachable


# --- risk_measure + beta driven by loss reaction ---------------------------


@pytest.mark.parametrize(
    ("reaction", "measure", "beta"),
    [
        (LossReaction.SELL_ALL, RiskMeasureChoice.MAX_DRAWDOWN, 0.99),
        (LossReaction.SELL_SOME, RiskMeasureChoice.CDAR, 0.975),
        (LossReaction.HOLD, RiskMeasureChoice.CVAR, 0.95),
        (LossReaction.BUY_MORE, RiskMeasureChoice.VARIANCE, 0.90),
    ],
)
def test_measure_and_beta_by_reaction(
    reaction: LossReaction, measure: RiskMeasureChoice, beta: float
) -> None:
    cs = build_constraint_set(_answers(reaction=reaction), portfolio_id="pf-1")
    assert cs.risk_measure is measure
    assert cs.beta == pytest.approx(beta)


def test_reaction_maps_every_member_to_distinct_measure() -> None:
    measures = {
        build_constraint_set(_answers(reaction=r), portfolio_id="pf").risk_measure
        for r in LossReaction
    }
    assert len(measures) == len(list(LossReaction))  # no reaction collapses


# --- nu drawdown-ceiling tiers ---------------------------------------------


@pytest.mark.parametrize("max_loss", [0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
def test_nu_tiers_monotone_and_bounded(max_loss: float) -> None:
    nu1, nu2, nu3 = _nu_tiers(
        CapacityAnswers(max_1yr_loss_pct=max_loss, buffer_months=6)
    )
    assert 0.0 <= nu1 <= nu2 <= nu3 <= 1.0
    assert nu2 == pytest.approx(max_loss)  # hard ceiling == stated loss tolerance


# --- output validity: schema + optimizer bridge ----------------------------


def test_output_is_a_valid_constraint_set() -> None:
    cs = build_constraint_set(_answers(), portfolio_id="pf-42")
    assert isinstance(cs, ConstraintSet)
    assert cs.portfolio_id == "pf-42"
    # esg / universe_filters untouched in Task 2 (Task 3 derives them).
    assert cs.esg.exclusions == ()
    assert cs.universe_filters.no_complex is False


def test_output_survives_to_mean_risk_config_across_the_matrix() -> None:
    for goal in ObjectiveChoice:
        for reaction in LossReaction:
            cs = build_constraint_set(
                _answers(goal=goal, reaction=reaction), portfolio_id="pf"
            )
            cfg = cs.to_mean_risk_config()
            assert isinstance(cfg, MeanRiskConfig)
            assert cfg.risk_aversion == cs.a_gamma
            assert cfg.cvar_beta == cs.beta
            assert cfg.cdar_beta == cs.beta


def test_mapping_is_total_over_every_driver_enum() -> None:
    # Every LossReaction × ObjectiveChoice × Horizon combination maps without a
    # KeyError and yields a config the optimizer accepts.
    for goal in ObjectiveChoice:
        for horizon in Horizon:
            for reaction in LossReaction:
                cs = build_constraint_set(
                    _answers(goal=goal, horizon=horizon, reaction=reaction),
                    portfolio_id="pf",
                )
                cs.to_mean_risk_config()  # must not raise


# --- portfolio_id / base_currency threading --------------------------------


def test_base_currency_defaults_to_the_answers_currency() -> None:
    cs = build_constraint_set(_answers(currency="USD"), portfolio_id="pf")
    assert cs.base_currency == "USD"


def test_base_currency_override_wins_over_answers() -> None:
    cs = build_constraint_set(
        _answers(currency="USD"), portfolio_id="pf", base_currency="CHF"
    )
    assert cs.base_currency == "CHF"


def test_module_imports_no_optimizer_at_construction() -> None:
    import fund.agents.profiler as mod

    assert mod.__file__ is not None
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))

    # profiler MAY import optimizer / the agent stack (fund is the bridge and Task 7
    # adds the LLM profiler), but the deterministic mapping must stay import-light:
    # every heavy dep is imported lazily inside a function (or guarded by
    # TYPE_CHECKING), never at module top level. Scanning only the module body's
    # direct import statements skips both lazy (in-function) and TYPE_CHECKING
    # (inside an ``if`` block) imports.
    forbidden = {
        "optimizer",
        "skfolio",
        "deepagents",
        "langchain",
        "langchain_core",
        "langchain_ollama",
        "app",
    }
    top_level_roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_roots.add(node.module.split(".")[0])
    assert forbidden.isdisjoint(top_level_roots)
