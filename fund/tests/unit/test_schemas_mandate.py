"""Task 2 — ``fund.schemas.mandate`` pins the operational mandate (D13).

``PortfolioMandate`` + nested ``RunTriggers`` are pure serialisable data — no
``optimizer`` import. This module also establishes the pydantic **v2**
``frozen=True`` pattern the rest of Phase 4 copies; the venv resolves pydantic
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
    # Task 2 verifies the installed major before the pattern is reused five times.
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


@pytest.mark.parametrize("bad_threshold", [0.0, -0.01])
def test_rejects_non_positive_drift_threshold(bad_threshold: float):
    with pytest.raises(pydantic.ValidationError):
        PortfolioMandate(
            portfolio_id="pf-001",
            capital=Decimal("1000"),
            base_currency="EUR",
            drift_l1_threshold=bad_threshold,
            triggers=RunTriggers(cron=True, drift=True),
        )


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
