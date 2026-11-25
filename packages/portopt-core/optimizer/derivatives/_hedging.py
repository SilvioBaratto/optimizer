"""Monte Carlo simulation of a discretely rebalanced delta hedge.

Black-Scholes prices an option as the cost of a self-financing replicating
portfolio that is rebalanced continuously. Hedging only at ``n`` dates leaves a
residual error on every path; this module measures it. A hedger sells one option at
the Black-Scholes price for a *hedge* volatility, then holds the Black-Scholes delta
in shares, financed at the risk-free rate, while the underlying follows a
geometric Brownian motion with a *true* volatility and a real-world drift.

Two facts follow and are what the simulator is for. With the true volatility used
for hedging, the standard deviation of the final P&L shrinks like ``1 / sqrt(n)``
(Boyle and Emanuel, 1980). With a wrong hedge volatility the expected P&L is close
to ``vega * (hedge - true)`` volatility, to first order, because the option was sold
at the wrong price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from optimizer.derivatives._black_scholes import (
    black_scholes_greeks,
    black_scholes_price,
)
from optimizer.derivatives._config import DeltaHedgeConfig, OptionType
from optimizer.derivatives._validation import FloatArray, scalar_contract

_BPS_PER_UNIT: Final = 10_000.0


@dataclass(frozen=True)
class DeltaHedgeResult:
    """Outcome of a delta-hedging simulation.

    Attributes:
        pnl: Terminal profit and loss of the hedger on each path, in money units
            at expiry; shape ``(n_paths,)`` and read-only.
        mean_pnl: Average of ``pnl``.
        std_pnl: Sample standard deviation of ``pnl``.
        premium: Black-Scholes price at the hedge volatility received for the
            option at inception.
        n_rebalances: Number of hedge dates used.
    """

    pnl: FloatArray
    mean_pnl: float
    std_pnl: float
    premium: float
    n_rebalances: int


def _gbm_step(
    prices: FloatArray,
    drift: float,
    volatility: float,
    dt: float,
    shocks: FloatArray,
) -> FloatArray:
    """Advance prices one step of an exact geometric Brownian motion."""
    log_return = (drift - 0.5 * volatility**2) * dt + volatility * math.sqrt(
        dt
    ) * shocks
    return np.asarray(prices * np.exp(log_return), dtype=np.float64)


def _bs_delta(
    prices: FloatArray,
    strike: float,
    remaining: float,
    config: DeltaHedgeConfig,
    kind: OptionType,
) -> FloatArray:
    """Return the Black-Scholes delta, at the hedge volatility, of every path."""
    greeks = black_scholes_greeks(
        prices, strike, remaining, config.rate, config.hedge_volatility, kind
    )
    return np.asarray(greeks.delta, dtype=np.float64)


def _hedged_pnl(
    spot: float,
    strike: float,
    maturity: float,
    kind: OptionType,
    config: DeltaHedgeConfig,
    premium: float,
) -> FloatArray:
    """Run the hedge on every path and return the terminal P&L."""
    dt = maturity / config.n_rebalances
    growth = math.exp(config.rate * dt)
    cost_rate = config.transaction_cost_bps / _BPS_PER_UNIT
    rng = np.random.default_rng(config.random_state)
    prices = np.full(config.n_paths, spot)
    shares = np.zeros(config.n_paths)
    cash = np.full(config.n_paths, premium)
    for step in range(config.n_rebalances):
        remaining = (config.n_rebalances - step) * dt
        target = _bs_delta(prices, strike, remaining, config, kind)
        traded_value = np.abs(target - shares) * prices
        cash = (cash - (target - shares) * prices - cost_rate * traded_value) * growth
        shares = target
        shocks = rng.standard_normal(config.n_paths)
        prices = _gbm_step(prices, config.drift, config.true_volatility, dt, shocks)
    return np.asarray(
        cash + shares * prices - kind.payoff(prices, strike), dtype=np.float64
    )


def simulate_delta_hedge(
    spot: float,
    strike: float,
    maturity: float,
    option_type: OptionType | str,
    config: DeltaHedgeConfig,
) -> DeltaHedgeResult:
    """Simulate selling one option and delta-hedging it until expiry.

    The hedger receives the Black-Scholes price at ``config.hedge_volatility`` and
    holds that model's delta in shares from inception, rebalancing at
    ``config.n_rebalances`` equally spaced dates (the first at inception), with
    cash earning ``config.rate``. Every trade, the initial hedge included, pays
    ``transaction_cost_bps / 1e4`` of the value traded. The position is not
    liquidated at expiry: the P&L is cash plus the shares at the final price minus
    the option payoff.

    Args:
        spot: Underlying price at inception, strictly positive.
        strike: Strike price, strictly positive.
        maturity: Time to expiry in years, strictly positive.
        option_type: ``OptionType.CALL``, ``OptionType.PUT`` or its string value.
        config: Simulation settings.

    Returns:
        The per-path P&L with its mean, standard deviation and the premium.

    Raises:
        ConfigurationError: If the contract is invalid or not a scalar, or the
            option type is unknown.
    """
    kind = OptionType.parse(option_type)
    contract = scalar_contract(
        spot,
        strike,
        maturity,
        config.rate,
        config.hedge_volatility,
        positive_maturity=True,
    )
    premium = float(
        black_scholes_price(
            contract.spot,
            contract.strike,
            contract.maturity,
            config.rate,
            config.hedge_volatility,
            kind,
        )
    )
    pnl = _hedged_pnl(
        contract.spot, contract.strike, contract.maturity, kind, config, premium
    )
    pnl.setflags(write=False)
    return DeltaHedgeResult(
        pnl=pnl,
        mean_pnl=float(pnl.mean()),
        std_pnl=float(pnl.std(ddof=1)),
        premium=premium,
        n_rebalances=config.n_rebalances,
    )
