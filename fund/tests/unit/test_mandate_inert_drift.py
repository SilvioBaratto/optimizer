"""A legacy percent-style drift threshold is surfaced, not silently kept dead.

A mandate stored with ``drift_l1_threshold = 5`` (meaning 5%) is clamped on load to
just under the L1 bound of 2, so the drift trigger can never fire; the next save used
to persist that value with only a log line. Re-reading it as 5% would switch on a
trigger that has been inert, so the clamp stays and the problem is reported: in the
mandate-save warnings, in ``fund mandate show``, and in the Posture view.
"""

from __future__ import annotations

import math
import uuid
from decimal import Decimal

import pytest
from _fund_fakes import PORTFOLIO_ID, make_mandate
from portopt_db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from typer.testing import CliRunner

import fund.cli as cli
from fund import observe
from fund.actions import apply_action
from fund.audit.mandate_repository import MandateRepository
from fund.schemas.mandate import PortfolioMandate, RunTriggers
from fund.tui.views import _render_posture_detail

_CLAMP = math.nextafter(PortfolioMandate.MAX_DRIFT_L1, 0.0)


def _stored(**overrides) -> dict:
    """A stored mandate payload with ``overrides`` applied as raw JSON."""
    return {**make_mandate().model_dump(mode="json"), **overrides}


class TestSchemaReportsAClampedThreshold:
    """The loaded mandate says whether its threshold was clamped."""

    def test_a_percent_style_value_loads_clamped_and_flagged(self) -> None:
        """5 loads as the clamp (never as 5%) and reports the trigger as inert."""
        loaded = PortfolioMandate.from_stored(_stored(drift_l1_threshold=5.0))

        assert loaded.drift_l1_threshold == _CLAMP
        assert loaded.drift_threshold_clamped is True

    def test_a_usable_threshold_is_not_flagged(self) -> None:
        """An ordinary threshold has nothing to report."""
        loaded = PortfolioMandate.from_stored(_stored(drift_l1_threshold=0.3))

        assert loaded.drift_threshold_clamped is False
        assert loaded.stored_value_warnings() == []

    def test_the_flag_survives_a_re_save_of_the_clamped_value(self) -> None:
        """A mandate rebuilt from the clamped value is still flagged."""
        resaved = PortfolioMandate.model_validate(
            PortfolioMandate.from_stored(_stored(drift_l1_threshold=5.0)).model_dump(
                mode="json"
            )
        )

        assert resaved.drift_threshold_clamped is True

    def test_the_warning_names_the_dead_trigger_and_the_fix(self) -> None:
        """The text says the trigger never fires and gives a fraction to use."""
        loaded = PortfolioMandate.from_stored(_stored(drift_l1_threshold=5.0))

        (warning,) = loaded.stored_value_warnings()

        assert "never fire" in warning
        assert "0.05" in warning

    def test_a_threshold_set_just_below_the_clamp_is_not_the_clamp(self) -> None:
        """Only the clamp value itself is treated as the dead sentinel."""
        mandate = PortfolioMandate(
            portfolio_id=str(PORTFOLIO_ID),
            capital=Decimal("1000"),
            base_currency="EUR",
            drift_l1_threshold=1.5,
            triggers=RunTriggers(cron=True, drift=True),
        )

        assert mandate.drift_threshold_clamped is False


@pytest.fixture
def session():
    """A private in-memory session where commits do not leak between tests."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _legacy_row(session, pid: uuid.UUID = PORTFOLIO_ID) -> None:
    """Store a mandate whose JSON holds the legacy percent-style threshold."""
    row = MandateRepository(session).upsert(make_mandate(pid))
    row.mandate = {**row.mandate, "drift_l1_threshold": 5.0}
    row.drift_l1_threshold = 5.0
    session.flush()


class TestSaveWarnings:
    """A mandate save reports a clamped threshold in its warnings."""

    @pytest.mark.parametrize("action", ["mandate_edit", "new_portfolio"])
    def test_saving_a_clamped_mandate_warns(self, session, action) -> None:
        """The result carries the same warning list the CLI and TUI print."""
        clamped = PortfolioMandate.from_stored(_stored(drift_l1_threshold=5.0))

        result = apply_action(
            action, session=session, portfolio_id=PORTFOLIO_ID, mandate=clamped
        )

        warnings = (result.detail or {}).get("warnings", [])
        assert [w for w in warnings if "drift" in w and "never fire" in w]
        assert "warning:" in result.message

    def test_saving_a_usable_threshold_does_not_warn(self, session) -> None:
        """A normal mandate keeps its message clean."""
        result = apply_action(
            "mandate_edit",
            session=session,
            portfolio_id=PORTFOLIO_ID,
            mandate=make_mandate(),
        )

        assert "never fire" not in result.message


class TestObserveReadsTheWarnings:
    """``observe.mandate_warnings`` is the model-free read the TUI uses."""

    def test_a_legacy_row_reports_its_warning(self, session) -> None:
        """The stored 5 is reported through the read model."""
        _legacy_row(session)

        (warning,) = observe.mandate_warnings(session, PORTFOLIO_ID)

        assert "never fire" in warning

    def test_a_healthy_row_reports_nothing(self, session) -> None:
        """No warning for a usable threshold."""
        MandateRepository(session).upsert(make_mandate())

        assert observe.mandate_warnings(session, PORTFOLIO_ID) == []

    def test_an_unknown_portfolio_reports_nothing(self, session) -> None:
        """No mandate, nothing to warn about."""
        assert observe.mandate_warnings(session, uuid.uuid4()) == []


class TestPostureShowsTheWarning:
    """The Posture detail pane surfaces a dead drift trigger."""

    def test_a_portfolio_with_no_runs_still_shows_it(self, session) -> None:
        """The empty-state pane carries the warning too."""
        _legacy_row(session)

        text = _render_posture_detail(session, PORTFOLIO_ID)

        assert "no runs yet" in text
        assert "never fire" in text

    def test_a_healthy_portfolio_shows_no_warning(self, session) -> None:
        """The pane is unchanged for a usable threshold."""
        MandateRepository(session).upsert(make_mandate())

        assert "never fire" not in _render_posture_detail(session, PORTFOLIO_ID)


@pytest.fixture
def cli_env(monkeypatch):
    """The CLI wired onto an isolated in-memory DB (see ``test_cli``)."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    from contextlib import contextmanager

    @contextmanager
    def _get_session():
        """Hand each command a fresh session on the shared engine."""
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(cli, "get_session", _get_session)
    with _get_session() as db:
        _legacy_row(db)
        db.commit()
    yield CliRunner()
    engine.dispose()


class TestCliReportsTheWarning:
    """``fund mandate show`` and ``set`` print the warning."""

    def test_show_prints_the_warning(self, cli_env) -> None:
        """The stored 5 is flagged where the adviser reads the mandate."""
        res = cli_env.invoke(cli.app, ["mandate", "show", str(PORTFOLIO_ID)])

        assert res.exit_code == 0, res.output
        assert "warning:" in res.output
        assert "never fire" in res.output

    def test_set_without_a_new_drift_repeats_the_warning(self, cli_env) -> None:
        """Saving another field carries the dead value over and says so."""
        res = cli_env.invoke(
            cli.app, ["mandate", "set", str(PORTFOLIO_ID), "--capital", "5000"]
        )

        assert res.exit_code == 0, res.output
        assert "never fire" in res.output

    def test_set_with_a_real_drift_clears_the_warning(self, cli_env) -> None:
        """Passing a usable fraction replaces the clamp and silences the warning."""
        res = cli_env.invoke(
            cli.app, ["mandate", "set", str(PORTFOLIO_ID), "--drift", "0.05"]
        )
        shown = cli_env.invoke(cli.app, ["mandate", "show", str(PORTFOLIO_ID)])

        assert res.exit_code == 0, res.output
        assert "never fire" not in res.output
        assert "never fire" not in shown.output
