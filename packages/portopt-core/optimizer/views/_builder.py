"""DB-agnostic adapters: analyst/estimate signals -> Black-Litterman views.

The optimizer library never imports the ingestion database (guarded by
``tests/test_no_portopt_db_import.py``).  These helpers therefore consume plain
Python primitives — ``float`` / ``Decimal`` prices and integer vote counts — that
a caller maps from DB rows:

* ``analyst_price_targets`` (``current`` / ``mean`` / ``median`` / ``low`` /
  ``high``, all SQL ``Numeric`` → Python ``Decimal``) → an **absolute expected
  return view** ``target / current - 1`` per ticker;
* ``analyst_recommendations`` (``strong_buy`` / ``buy`` / ``hold`` / ``sell`` /
  ``strong_sell`` vote counts) → an **Idzorek confidence** in ``[0, 1]`` derived
  from consensus agreement (dispersion of the votes).

The output is the ``(views, confidences)`` tuple of skfolio view *strings*
(``"AAPL == 0.0123"``) plus aligned confidences that
:class:`~optimizer.views._config.BlackLittermanConfig` consumes — mirroring the
existing ``optimizer.factors.build_factor_bl_views`` bridge, but sourced from raw
analyst snapshots instead of composite factor z-scores.

Scale gotcha
------------
Analyst price targets are **12-month** figures, but Black-Litterman expects a view
on the *same periodicity as the returns passed to* ``fit`` (typically daily linear
returns from :func:`skfolio.preprocessing.prices_to_returns`).  Pass
``horizon_periods`` (e.g. ``252`` for a 12-month target against daily returns) to
de-annualise the implied return; the default ``1.0`` assumes the caller already
supplies a per-period figure, so a raw 12-month target would otherwise be injected
as a single-period view and blow up the posterior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.views._config import BlackLittermanConfig, ViewUncertaintyMethod

# Numeric score per recommendation bucket, best (5.0) to worst (1.0).
_RECOMMENDATION_SCORES = (5.0, 4.0, 3.0, 2.0, 1.0)
# Maximum attainable variance of the bucket scores: all mass split evenly across
# the two extreme buckets, i.e. ((5-3)^2 + (1-3)^2) / 2 = 4.0.
_MAX_RECOMMENDATION_VARIANCE = 4.0


class PriceTargetStatistic(str, Enum):
    """Which ``analyst_price_targets`` aggregate drives the return view."""

    MEAN = "mean"
    MEDIAN = "median"
    LOW = "low"
    HIGH = "high"


def _to_float(value: float | Decimal | None) -> float | None:
    """Cast a possibly-``Decimal`` DB value to ``float`` (``None`` passthrough)."""
    return None if value is None else float(value)


@dataclass(frozen=True)
class AnalystSignal:
    """Per-ticker analyst snapshot, mapped from DB rows by the caller.

    Prices accept SQL ``Numeric`` values (Python ``Decimal``) and are coerced to
    ``float`` on construction (the "cast before numpy" rule).  Vote counts accept
    ``None`` and are coerced to ``0``.

    Parameters
    ----------
    current_price : float or Decimal or None
        Latest price (``analyst_price_targets.current`` or the profile's
        ``current_price``).  Required to compute an implied return.
    target_mean, target_median, target_low, target_high : float or Decimal or None
        12-month price-target aggregates (``analyst_price_targets``).
    num_analysts : int or None
        Coverage count (``number_of_analyst_opinions``); metadata only.
    strong_buy, buy, hold, sell, strong_sell : int
        Trailing recommendation vote counts (``analyst_recommendations``).
    """

    current_price: float | Decimal | None = None
    target_mean: float | Decimal | None = None
    target_median: float | Decimal | None = None
    target_low: float | Decimal | None = None
    target_high: float | Decimal | None = None
    num_analysts: int | None = None
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "current_price",
            "target_mean",
            "target_median",
            "target_low",
            "target_high",
        ):
            object.__setattr__(self, field_name, _to_float(getattr(self, field_name)))
        for field_name in ("strong_buy", "buy", "hold", "sell", "strong_sell"):
            object.__setattr__(self, field_name, int(getattr(self, field_name) or 0))

    def target(self, statistic: PriceTargetStatistic) -> float | None:
        """Return the price-target aggregate selected by *statistic*."""
        value = {
            PriceTargetStatistic.MEAN: self.target_mean,
            PriceTargetStatistic.MEDIAN: self.target_median,
            PriceTargetStatistic.LOW: self.target_low,
            PriceTargetStatistic.HIGH: self.target_high,
        }[statistic]
        return None if value is None else float(value)

    @property
    def total_votes(self) -> int:
        """Total recommendation votes across all five buckets."""
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell


def implied_return_from_price_target(
    current_price: float | Decimal,
    target_price: float | Decimal,
    *,
    horizon_periods: float = 1.0,
    compounding: bool = True,
) -> float:
    """Implied expected return from an analyst price target.

    Computes ``target / current - 1`` and de-annualises it over
    ``horizon_periods`` return periods so it matches the periodicity of the
    returns fed to Black-Litterman ``fit`` (see the module scale gotcha).

    Parameters
    ----------
    current_price, target_price : float or Decimal
        Latest price and the (12-month) target.  Both must be strictly positive.
    horizon_periods : float, default 1.0
        Number of return periods until the target is expected to realise
        (e.g. ``252`` for a 12-month target against daily returns).  ``1.0``
        returns the raw cumulative implied return unchanged.
    compounding : bool, default True
        When de-annualising, ``True`` uses geometric compounding
        ``(1 + total) ** (1 / horizon) - 1``; ``False`` uses the simple
        ``total / horizon``.

    Returns
    -------
    float
        The per-period implied expected return.
    """
    current: float = float(current_price)
    target: float = float(target_price)
    if current <= 0.0:
        raise DataError(f"current_price must be strictly positive, got {current}")
    if target <= 0.0:
        raise DataError(f"target_price must be strictly positive, got {target}")
    if horizon_periods <= 0.0:
        raise ConfigurationError(
            f"horizon_periods must be strictly positive, got {horizon_periods}"
        )
    total = target / current - 1.0
    if horizon_periods == 1.0:
        return total
    if compounding:
        return float((1.0 + total) ** (1.0 / horizon_periods)) - 1.0
    return total / horizon_periods


def recommendation_confidence(
    signal: AnalystSignal,
    *,
    min_votes: int = 1,
    cap: float = 1.0,
) -> float:
    """Map recommendation vote counts to an Idzorek confidence in ``[0, cap]``.

    Confidence reflects analyst *agreement*: it is ``1 - dispersion`` of the
    bucket scores (strong_buy=5 ... strong_sell=1).  Unanimous votes → ``cap``;
    a maximal split between the two extreme buckets → ``0``.  The ``cap`` mirrors
    :attr:`FactorIntegrationConfig.view_confidence_cap` — Idzorek confidence
    ``1.0`` forces the posterior onto the view exactly (extreme concentration),
    so callers typically cap at ``0.25``-``0.50`` to blend with the prior.

    Raises
    ------
    DataError
        If fewer than ``min_votes`` total votes are present.
    ConfigurationError
        If ``cap`` is outside ``(0, 1]``.
    """
    if not (0.0 < cap <= 1.0):
        raise ConfigurationError(f"cap must be in (0, 1], got {cap}")
    total = signal.total_votes
    if total < min_votes:
        raise DataError(
            f"recommendation_confidence needs >= {min_votes} votes, got {total}"
        )
    counts = (
        signal.strong_buy,
        signal.buy,
        signal.hold,
        signal.sell,
        signal.strong_sell,
    )
    weighted = list(zip(counts, _RECOMMENDATION_SCORES, strict=True))
    mean = sum(c * s for c, s in weighted) / total
    variance = sum(c * (s - mean) ** 2 for c, s in weighted) / total
    agreement = 1.0 - variance / _MAX_RECOMMENDATION_VARIANCE
    agreement = min(1.0, max(0.0, agreement))
    return agreement * cap


def build_analyst_bl_views(
    signals: Mapping[str, AnalystSignal],
    *,
    statistic: PriceTargetStatistic = PriceTargetStatistic.MEAN,
    horizon_periods: float = 1.0,
    compounding: bool = True,
    with_confidence: bool = False,
    min_votes: int = 1,
    confidence_cap: float = 0.5,
    precision: int = 6,
    skip_incomplete: bool = True,
) -> tuple[tuple[str, ...], tuple[float, ...] | None]:
    """Build Black-Litterman absolute views from analyst price targets.

    Iterates *signals* in insertion order and, for each ticker with a usable
    current price and target, emits a view string ``"<ticker> == <return>"``.
    When ``with_confidence`` is set, an aligned Idzorek confidence derived from
    the ticker's recommendation votes is produced too (see
    :func:`recommendation_confidence`).

    Parameters
    ----------
    signals : Mapping[str, AnalystSignal]
        Ticker → analyst snapshot.  Order defines the view/confidence order.
    statistic : PriceTargetStatistic, default MEAN
        Which price-target aggregate drives the view.
    horizon_periods, compounding : see :func:`implied_return_from_price_target`.
    with_confidence : bool, default False
        Emit aligned Idzorek confidences from recommendation votes.
    min_votes : int, default 1
        Minimum recommendation votes required when ``with_confidence`` is set.
    confidence_cap : float, default 0.5
        Upper bound for the emitted confidences.
    precision : int, default 6
        Decimal places used to format each view's return (fixed-point, so no
        scientific notation reaches skfolio's parser).  Use a larger value for
        small per-period returns (e.g. ``8`` for daily).
    skip_incomplete : bool, default True
        Skip tickers missing a target/current price (or, under
        ``with_confidence``, enough votes) rather than raising.

    Returns
    -------
    tuple[tuple[str, ...], tuple[float, ...] or None]
        ``(views, confidences)``; ``confidences`` is ``None`` when
        ``with_confidence`` is ``False``.

    Raises
    ------
    DataError
        If no usable view could be generated, or if ``skip_incomplete`` is
        ``False`` and a ticker is incomplete.
    ConfigurationError
        If ``precision`` is negative.
    """
    if precision < 0:
        raise ConfigurationError(f"precision must be non-negative, got {precision}")

    views: list[str] = []
    confidences: list[float] = []
    for ticker, signal in signals.items():
        target = signal.target(statistic)
        current = signal.current_price
        if target is None or current is None or current <= 0.0 or target <= 0.0:
            if skip_incomplete:
                continue
            raise DataError(
                f"{ticker}: missing or non-positive current/{statistic.value} target"
            )
        if with_confidence and signal.total_votes < min_votes:
            if skip_incomplete:
                continue
            raise DataError(
                f"{ticker}: needs >= {min_votes} recommendation votes for confidence"
            )
        implied = implied_return_from_price_target(
            current,
            target,
            horizon_periods=horizon_periods,
            compounding=compounding,
        )
        views.append(f"{ticker} == {implied:.{precision}f}")
        if with_confidence:
            confidences.append(
                recommendation_confidence(
                    signal, min_votes=min_votes, cap=confidence_cap
                )
            )

    if not views:
        raise DataError(
            "no usable analyst views could be generated from the given signals"
        )
    return tuple(views), (tuple(confidences) if with_confidence else None)


def build_black_litterman_config_from_signals(
    signals: Mapping[str, AnalystSignal],
    *,
    statistic: PriceTargetStatistic = PriceTargetStatistic.MEAN,
    horizon_periods: float = 1.0,
    compounding: bool = True,
    with_confidence: bool = False,
    min_votes: int = 1,
    confidence_cap: float = 0.5,
    precision: int = 6,
    skip_incomplete: bool = True,
    **bl_kwargs: object,
) -> BlackLittermanConfig:
    """Build a :class:`BlackLittermanConfig` directly from analyst *signals*.

    A thin convenience over :func:`build_analyst_bl_views`: when
    ``with_confidence`` is set, the config uses the Idzorek uncertainty method
    with the generated confidences; otherwise it uses the He-Litterman default.
    Extra ``bl_kwargs`` (e.g. ``tau``, ``groups``, ``prior_config``) pass through
    to the config, but ``views`` / ``uncertainty_method`` / ``view_confidences``
    must not be supplied (they are derived here).
    """
    views, confidences = build_analyst_bl_views(
        signals,
        statistic=statistic,
        horizon_periods=horizon_periods,
        compounding=compounding,
        with_confidence=with_confidence,
        min_votes=min_votes,
        confidence_cap=confidence_cap,
        precision=precision,
        skip_incomplete=skip_incomplete,
    )
    if confidences is not None:
        return BlackLittermanConfig(
            views=views,
            uncertainty_method=ViewUncertaintyMethod.IDZOREK,
            view_confidences=confidences,
            **bl_kwargs,  # type: ignore[arg-type]
        )
    return BlackLittermanConfig(views=views, **bl_kwargs)  # type: ignore[arg-type]
