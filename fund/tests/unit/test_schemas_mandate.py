"""``fund.schemas.mandate`` pins the operational mandate.

``PortfolioMandate`` + nested ``RunTriggers`` are pure serialisable data — no
``optimizer`` import. This module establishes the pydantic **v2**
``frozen=True`` pattern; the venv resolves pydantic
2.13.4 (``python -c "import pydantic; pydantic.VERSION"`` → ``2.13.4``), so the
v2 ``ConfigDict`` / ``model_dump`` surface is available.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pydantic
import pytest

from fund.schemas.mandate import PortfolioMandate, RunTriggers


def _valid_mandate() -> PortfolioMandate:
    return PortfolioMandate(
        portfolio_id="pf-001",
        capital=Decimal("100000.00"),
        base_currency="EUR",
        drift_l1_threshold=0.05,
        triggers=RunTriggers(cron=True, drift=True),
    )


def test_pydantic_major_is_two():
    assert pydantic.VERSION.startswith("2")


def test_valid_input_is_accepted():
    m = _valid_mandate()
    assert m.portfolio_id == "pf-001"
    assert m.capital == Decimal("100000.00")
    assert m.base_currency == "EUR"
    assert m.drift_l1_threshold == 0.05
    assert m.triggers.cron is True
    assert m.triggers.drift is True


def test_hitl_gates_defaults_to_place_orders():
    assert _valid_mandate().hitl_gates == ("place_orders",)


def test_benchmark_defaults_to_none():
    assert _valid_mandate().benchmark is None


def test_remembered_directives_defaults_to_empty_tuple():
    assert _valid_mandate().remembered_directives == ()


def test_old_serialised_mandate_validates_remembered_directives_to_empty():
    # A mandate serialised before the field existed carries no key; it must
    # rehydrate to the empty default rather than raising.
    legacy = _valid_mandate().model_dump(mode="json")
    legacy.pop("remembered_directives", None)
    assert "remembered_directives" not in legacy

    rehydrated = PortfolioMandate.model_validate(legacy)
    assert rehydrated.remembered_directives == ()


def test_remembered_directives_round_trips_non_empty():
    m = PortfolioMandate(
        portfolio_id="pf-001",
        capital=Decimal("1000"),
        base_currency="EUR",
        drift_l1_threshold=0.05,
        triggers=RunTriggers(cron=True, drift=False),
        remembered_directives=("hold cash heavy", "avoid energy"),
    )
    round_tripped = PortfolioMandate.model_validate(m.model_dump(mode="json"))
    assert round_tripped.remembered_directives == ("hold cash heavy", "avoid energy")
    assert round_tripped == m


def test_remembered_directives_is_frozen():
    m = _valid_mandate()
    with pytest.raises(pydantic.ValidationError):
        m.remembered_directives = ("x",)  # type: ignore[misc]


def test_benchmark_can_be_set():
    m = PortfolioMandate(
        portfolio_id="pf-001",
        capital=Decimal("1000"),
        base_currency="USD",
        drift_l1_threshold=0.1,
        triggers=RunTriggers(cron=False, drift=True),
        benchmark="^GSPC",
    )
    assert m.benchmark == "^GSPC"


def _mandate_with_benchmark(benchmark: str) -> PortfolioMandate:
    return PortfolioMandate(
        portfolio_id="pf-001",
        capital=Decimal("1000"),
        base_currency="USD",
        drift_l1_threshold=0.1,
        triggers=RunTriggers(cron=False, drift=True),
        benchmark=benchmark,
    )


def test_benchmark_longer_than_the_mirrored_column_is_rejected_by_the_schema():
    from portopt_db.models import PortfolioMandate as PortfolioMandateRow

    width = PortfolioMandateRow.__table__.c.benchmark.type.length

    assert _mandate_with_benchmark("S" * width).benchmark == "S" * width
    with pytest.raises(pydantic.ValidationError):
        _mandate_with_benchmark("S" * (width + 1))


def test_both_models_are_frozen():
    m = _valid_mandate()
    with pytest.raises(pydantic.ValidationError):
        m.capital = Decimal("1")  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        m.triggers.cron = False  # type: ignore[misc]


@pytest.mark.parametrize("bad_currency", ["eur", "EURO", "EU", "E1R", "US$"])
def test_rejects_non_iso_currency(bad_currency: str):
    with pytest.raises(pydantic.ValidationError):
        PortfolioMandate(
            portfolio_id="pf-001",
            capital=Decimal("1000"),
            base_currency=bad_currency,
            drift_l1_threshold=0.05,
            triggers=RunTriggers(cron=True, drift=True),
        )


@pytest.mark.parametrize("bad_capital", [Decimal("0"), Decimal("-1")])
def test_rejects_non_positive_capital(bad_capital: Decimal):
    with pytest.raises(pydantic.ValidationError):
        PortfolioMandate(
            portfolio_id="pf-001",
            capital=bad_capital,
            base_currency="EUR",
            drift_l1_threshold=0.05,
            triggers=RunTriggers(cron=True, drift=True),
        )


@pytest.mark.parametrize(
    "bad_threshold", [0.0, -0.01, 2.0, 5.0, float("nan"), float("inf")]
)
def test_rejects_drift_threshold_outside_the_l1_range(bad_threshold: float):
    with pytest.raises(pydantic.ValidationError):
        PortfolioMandate(
            portfolio_id="pf-001",
            capital=Decimal("1000"),
            base_currency="EUR",
            drift_l1_threshold=bad_threshold,
            triggers=RunTriggers(cron=True, drift=True),
        )


@pytest.mark.parametrize(
    "bad_threshold", [2.0, 2.01, 5.0, 10.0, float("inf"), float("nan")]
)
def test_rejects_drift_threshold_the_l1_drift_can_never_reach(bad_threshold: float):
    # L1 drift between two long-only portfolios is at most 2, so a percent typed as a
    # fraction (5 for 5%) would validate yet leave the drift trigger permanently inert.
    with pytest.raises(pydantic.ValidationError):
        PortfolioMandate(
            portfolio_id="pf-001",
            capital=Decimal("1000"),
            base_currency="EUR",
            drift_l1_threshold=bad_threshold,
            triggers=RunTriggers(cron=True, drift=True),
        )


@pytest.mark.parametrize("threshold", [0.0001, 0.05, 1.99])
def test_accepts_drift_threshold_inside_the_l1_range(threshold: float):
    m = PortfolioMandate(
        portfolio_id="pf-001",
        capital=Decimal("1000"),
        base_currency="EUR",
        drift_l1_threshold=threshold,
        triggers=RunTriggers(cron=True, drift=True),
    )
    assert m.drift_l1_threshold == threshold


def test_json_round_trip():
    m = PortfolioMandate(
        portfolio_id="pf-001",
        capital=Decimal("100000.00"),
        base_currency="EUR",
        drift_l1_threshold=0.05,
        hitl_gates=("place_orders", "rebalance"),
        triggers=RunTriggers(cron=True, drift=False),
        benchmark="^STOXX50E",
    )
    assert PortfolioMandate.model_validate(m.model_dump(mode="json")) == m


def test_module_imports_nothing_forbidden():
    from fund.schemas import mandate

    text = Path(mandate.__file__).read_text(encoding="utf-8")
    for forbidden in ("optimizer", "deepagents", "langgraph", "app", "skfolio"):
        assert f"import {forbidden}" not in text
        assert f"from {forbidden}" not in text


def _stored_payload(**overrides) -> dict:
    return {**_valid_mandate().model_dump(mode="json"), **overrides}


def test_from_stored_clamps_a_legacy_threshold_above_the_l1_bound():
    # A mandate saved before the cap (a percent-style "5") must still load: the
    # daemon sweeps and the TUI read every stored row and one bad row cannot stop them.
    loaded = PortfolioMandate.from_stored(_stored_payload(drift_l1_threshold=5.0))

    # Just under the open L1 bound: what a threshold that can never be crossed was.
    assert loaded.drift_l1_threshold == pytest.approx(2.0)
    assert loaded.drift_l1_threshold < 2.0
    assert loaded.capital == Decimal("100000.00")  # nothing else is touched


def test_from_stored_leaves_an_in_range_threshold_alone():
    payload = _stored_payload(drift_l1_threshold=0.3)

    assert PortfolioMandate.from_stored(payload).drift_l1_threshold == 0.3


def test_from_stored_does_not_mutate_its_input():
    payload = _stored_payload(drift_l1_threshold=5.0)

    PortfolioMandate.from_stored(payload)

    assert payload["drift_l1_threshold"] == 5.0


def test_from_stored_still_rejects_what_it_cannot_repair():
    with pytest.raises(pydantic.ValidationError):
        PortfolioMandate.from_stored(_stored_payload(capital="-1"))
    with pytest.raises(pydantic.ValidationError):
        PortfolioMandate.from_stored(_stored_payload(drift_l1_threshold=-0.5))


def test_constructor_still_rejects_a_threshold_above_the_l1_bound():
    # The clamp is a read-side repair only; new input stays strict.
    with pytest.raises(pydantic.ValidationError):
        PortfolioMandate.model_validate(_stored_payload(drift_l1_threshold=5.0))
