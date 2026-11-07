"""Operational mandate for a fund run — pure serialisable data.

``PortfolioMandate`` carries the *how it runs* knobs (capital, base currency,
drift threshold, HITL gates, run triggers, optional benchmark) that the
agents read once per run. It is deliberately data-only: no ``optimizer`` /
``deepagents`` / ``app`` import, so constructing a mandate stays cheap and
optimizer-free (the MiFID→knob mapping lives in ``ConstraintSet`` / ``ViewSet``).

This module establishes the pydantic v2 ``frozen=True`` pattern
(``model_config = ConfigDict(frozen=True)``) reused by the other schemas: frozen
models are hashable + serialisable, matching the optimizer's frozen-dataclass
ethos. ``Decimal`` capital round-trips through JSON as a string.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Annotated, Any, ClassVar, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

logger = logging.getLogger(__name__)

_MAX_TICKER_LEN = 32
_MAX_ELIGIBLE_TICKERS = 100


def _dedupe_in_order(tickers: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(tickers))


# ``max_length`` on the tuple caps the entry count, not each ticker's width, so the
# per-ticker constraint lives on the item type. The width matches ``benchmark``; the
# count keeps the PM seed block bounded.
_EligibleTickers = Annotated[
    tuple[
        Annotated[
            str,
            StringConstraints(
                strip_whitespace=True,
                to_upper=True,
                min_length=1,
                max_length=_MAX_TICKER_LEN,
            ),
        ],
        ...,
    ],
    Field(max_length=_MAX_ELIGIBLE_TICKERS),
    AfterValidator(_dedupe_in_order),
]


class RunTriggers(BaseModel):
    """What causes a run to fire: a cron schedule and/or a drift breach."""

    model_config = ConfigDict(frozen=True)

    cron: bool
    drift: bool


class PortfolioMandate(BaseModel):
    """Per-portfolio operational mandate.

    Fields hold only primitives/tuples plus the nested frozen ``RunTriggers`` —
    fully serialisable. ``capital`` is strictly positive and ``drift_l1_threshold``
    lies in the open interval ``(0, MAX_DRIFT_L1)`` so the drift trigger can fire;
    ``base_currency`` is an ISO-4217 alpha code. ``hitl_gates`` lists the tool names
    gated behind a human approval and defaults to the single ``place_orders`` gate.
    ``remembered_directives`` carries free-text co-pilot steers the adviser chose to
    persist across rounds; every run renders them into the PM's instruction, and the
    mandate JSON is their source of truth (no scalar mirror, no migration).
    ``eligible_tickers`` is the hard allowlist of instruments the portfolio may hold:
    normalised to upper-case, de-duplicated in order, and empty meaning unrestricted.
    ``model_copy(update=...)`` skips validators, so consumers normalise again.
    ``tradeable_only`` additionally scopes the portfolio to instruments the Trading 212
    catalogue carries, for runs that must be executable at the broker. It composes with
    ``eligible_tickers`` (both must hold) and, being JSON-only like the allowlist, needs
    no migration: a mandate saved before the field existed reads as ``False``.
    """

    model_config = ConfigDict(frozen=True)

    # Two disjoint fully-invested books sit at L1 distance 2, the most
    # ``observe.drift_l1`` can report. The drift monitor breaches only on
    # ``drift_l1 > threshold``, so a threshold of 2 or more is never crossed and
    # silently disables the drift trigger.
    MAX_DRIFT_L1: ClassVar[float] = 2.0
    # What ``from_stored`` clamps an over-bound stored threshold to: the largest value
    # the schema accepts. It is also the value a re-save of the loaded mandate
    # persists, so it identifies a threshold that is still the dead legacy one.
    CLAMPED_DRIFT_L1: ClassVar[float] = math.nextafter(MAX_DRIFT_L1, 0.0)

    portfolio_id: str
    capital: Decimal = Field(gt=0)
    base_currency: str = Field(pattern=r"^[A-Z]{3}$")
    drift_l1_threshold: float = Field(gt=0, lt=MAX_DRIFT_L1, allow_inf_nan=False)
    hitl_gates: tuple[str, ...] = ("place_orders",)
    triggers: RunTriggers
    # Mirrored into portfolio_mandates.benchmark (VARCHAR(32)); PostgreSQL rejects a
    # longer value at flush, so reject it here as a clean validation error instead.
    benchmark: str | None = Field(default=None, max_length=32)
    remembered_directives: tuple[str, ...] = ()
    eligible_tickers: _EligibleTickers = ()
    tradeable_only: bool = False

    @property
    def drift_threshold_clamped(self) -> bool:
        """Whether the drift threshold is the clamp of a legacy over-bound value.

        Such a threshold sits at :attr:`CLAMPED_DRIFT_L1`, so the drift trigger can
        never fire. It is read from the value rather than remembered at load time, so
        it stays true after a save persists the clamped value, until a real threshold
        replaces it.
        """
        return self.drift_l1_threshold >= self.CLAMPED_DRIFT_L1

    def stored_value_warnings(self) -> list[str]:
        """Describe stored values that silently disable part of the mandate.

        The clamp is not re-read as a percentage: a stored ``5`` may never have been
        meant as 5%, and turning on a trigger that has been inert is the adviser's
        decision, not the loader's.

        Returns:
            One sentence per problem, empty for a healthy mandate.
        """
        if not self.drift_threshold_clamped:
            return []
        return [
            "the drift threshold is clamped just under the L1 bound of "
            f"{self.MAX_DRIFT_L1:g} (a legacy percent-style value such as 5 was "
            "stored), so the drift trigger can never fire; set a fraction, e.g. "
            "0.05 for 5% (fund mandate set --drift 0.05)"
        ]

    @classmethod
    def from_stored(cls, payload: Mapping[str, Any]) -> Self:
        """Rehydrate a persisted mandate, repairing a legacy over-bound threshold.

        Rows saved before ``drift_l1_threshold`` was capped can hold a percent-style
        value such as ``5``. Strict validation would make every reader of the stored
        row (the daemon sweeps, the TUI) fail on it, so the threshold is clamped to
        the largest value the schema accepts, :attr:`CLAMPED_DRIFT_L1` --
        behaviourally what the larger value always was, a trigger that can never fire
        -- and the next save persists the repaired value. The clamp is reported by
        :attr:`drift_threshold_clamped` and :meth:`stored_value_warnings`, never
        re-read as a percentage. Anything else that fails validation still raises.

        Args:
            payload: The mandate JSON column. Not mutated.

        Returns:
            The validated mandate.

        Raises:
            pydantic.ValidationError: ``payload`` violates the schema in a way the
                clamp does not repair.
        """
        threshold = payload.get("drift_l1_threshold")
        if (
            isinstance(threshold, int | float)
            and not isinstance(threshold, bool)
            and threshold >= cls.MAX_DRIFT_L1
        ):
            clamped = cls.CLAMPED_DRIFT_L1
            logger.warning(
                "mandate %s: stored drift_l1_threshold %s exceeds the L1 bound; "
                "clamped to %s",
                payload.get("portfolio_id"),
                threshold,
                clamped,
            )
            payload = {**payload, "drift_l1_threshold": clamped}
        return cls.model_validate(payload)


__all__ = ["PortfolioMandate", "RunTriggers"]
