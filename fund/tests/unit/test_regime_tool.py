"""``get_regime_stats`` tool: the numbers behind a regime read.

The economist's ``macro-regime-read`` skill interprets turbulence, the absorption
ratio and a two-state mixture probability; it must never invent them. These tests pin
the tool as a pure function of a seeded SQLite price panel:

* turbulence is the Mahalanobis distance of the latest return from the window's
  history, reported with its percentile inside the window;
* the absorption ratio is the share of variance carried by the top fifth of the
  principal components, and rises with cross-asset correlation;
* the mixture probability is the posterior of the high-variance state at the latest
  observation;
* it respects the ``asof`` bound, the ``window`` bound, and degrades to
  ``{ok: false}`` / ``mixture: None`` instead of raising or fabricating.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument

from fund.tools.regime import get_regime_stats

_START = dt.date(2022, 1, 1)
_N_DAYS = 400
_TICKERS = ["AAA", "BBB", "CCC"]


def _asof(n_days: int = _N_DAYS) -> dt.date:
    return _START + dt.timedelta(days=n_days - 1)


def _seed(db_session, returns: dict[str, np.ndarray]) -> None:
    """Seed one close series per ticker from its simple-return path."""
    for ticker, path in returns.items():
        exchange = Exchange(name=f"EX-{ticker}")
        db_session.add(exchange)
        db_session.flush()
        instrument = Instrument(
            ticker=ticker,
            short_name=ticker,
            exchange_id=exchange.id,
            instrument_type="STOCK",
            asset_class="equity",
            yfinance_ticker=ticker,
        )
        db_session.add(instrument)
        db_session.flush()
        closes = 100.0 * np.cumprod(1.0 + path)
        for i, close in enumerate(closes):
            db_session.add(
                PriceHistory(
                    instrument_id=instrument.id,
                    date=_START + dt.timedelta(days=i),
                    close=float(close),
                    volume=1000,
                )
            )
    db_session.flush()


def _independent(
    seed: int = 1, n_days: int = _N_DAYS, sigma: np.ndarray | float = 0.01
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=(n_days, len(_TICKERS))) * sigma
    return {t: noise[:, i] for i, t in enumerate(_TICKERS)}


def _one_factor(seed: int = 2) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, 0.01, size=_N_DAYS)
    return {t: factor + rng.normal(0.0, 0.001, size=_N_DAYS) for t in _TICKERS}


def _regime_shift(calm_end: bool, seed: int = 3) -> dict[str, np.ndarray]:
    """Calm at 0.5% daily vol with a 60-day 3% vol cluster, at the end or mid-sample."""
    sigma = np.full(_N_DAYS, 0.005)
    cluster = slice(_N_DAYS - 60, _N_DAYS) if not calm_end else slice(150, 210)
    sigma[cluster] = 0.03
    return _independent(seed=seed, sigma=sigma[:, None])


def test_returns_all_three_statistics(db_session) -> None:
    """A healthy panel yields turbulence, absorption ratio and mixture blocks."""
    _seed(db_session, _independent())

    result = get_regime_stats(db_session, _asof(), _TICKERS)

    assert result["ok"] is True
    data = result["data"]
    assert data["assets"] == _TICKERS
    assert data["missing"] == []
    assert data["n_observations"] == _N_DAYS - 1
    assert 0.0 < data["absorption_ratio"]["value"] <= 1.0
    assert 0.0 <= data["mixture"]["stress_probability"] <= 1.0
    assert data["turbulence"]["latest"] >= 0.0


def test_a_shock_on_the_last_day_is_the_most_turbulent_observation(db_session) -> None:
    """A -8% day across every asset tops its own window's turbulence ranking."""
    shocked = _independent()
    for path in shocked.values():
        path[-1] = -0.08
    _seed(db_session, shocked)

    data = get_regime_stats(db_session, _asof(), _TICKERS)["data"]

    assert data["turbulence"]["percentile"] == pytest.approx(1.0)
    assert data["turbulence"]["date"] == _asof().isoformat()


def test_a_quiet_last_day_is_not_turbulent(db_session) -> None:
    """A near-zero return after normal noise ranks low in turbulence."""
    quiet = _independent()
    for path in quiet.values():
        path[-1] = 0.0
    _seed(db_session, quiet)

    data = get_regime_stats(db_session, _asof(), _TICKERS)["data"]

    assert data["turbulence"]["percentile"] < 0.2


def test_absorption_ratio_rises_with_cross_asset_correlation(db_session) -> None:
    """One common factor concentrates variance in a single component."""
    _seed(db_session, _one_factor())

    correlated = get_regime_stats(db_session, _asof(), _TICKERS)["data"]

    assert correlated["absorption_ratio"]["n_components"] == 1
    assert correlated["absorption_ratio"]["value"] > 0.95


def test_absorption_ratio_is_low_for_independent_assets(db_session) -> None:
    """Independent assets spread variance across components (ratio near 1/3)."""
    _seed(db_session, _independent())

    data = get_regime_stats(db_session, _asof(), _TICKERS)["data"]

    assert data["absorption_ratio"]["value"] < 0.6


def test_absorption_ratio_reports_its_change_over_the_prior_window(db_session) -> None:
    """The prior-window ratio makes "rising" readable without a second call."""
    _seed(db_session, _independent())

    ratio = get_regime_stats(db_session, _asof(), _TICKERS)["data"]["absorption_ratio"]

    assert ratio["prior"] is not None
    assert ratio["change"] == pytest.approx(ratio["value"] - ratio["prior"])


def test_mixture_flags_stress_when_the_sample_ends_in_a_high_vol_cluster(
    db_session,
) -> None:
    """The latest observation sits in the high-variance state."""
    _seed(db_session, _regime_shift(calm_end=False))

    mixture = get_regime_stats(db_session, _asof(), _TICKERS)["data"]["mixture"]

    assert mixture["stress_probability"] > 0.8
    assert mixture["stress_weight"] < 0.5


def test_mixture_is_calm_when_the_high_vol_cluster_is_behind_us(db_session) -> None:
    """The latest observation sits in the low-variance state."""
    _seed(db_session, _regime_shift(calm_end=True))

    mixture = get_regime_stats(db_session, _asof(), _TICKERS)["data"]["mixture"]

    assert mixture["stress_probability"] < 0.2


def test_respects_the_asof_upper_bound(db_session) -> None:
    """Observations after ``asof`` never enter the window."""
    _seed(db_session, _independent())
    asof = _asof(300)

    data = get_regime_stats(db_session, asof, _TICKERS)["data"]

    assert data["asof"] == asof.isoformat()
    assert data["window_end"] == asof.isoformat()
    assert data["n_observations"] == 299


def test_window_bounds_the_observations(db_session) -> None:
    """Only the trailing ``window`` returns are used."""
    _seed(db_session, _independent())

    data = get_regime_stats(db_session, _asof(), _TICKERS, window=100)["data"]

    assert data["n_observations"] == 100


def test_unpriced_tickers_are_flagged_not_raised(db_session) -> None:
    """A ticker with no price history is reported in ``missing``."""
    _seed(db_session, _independent())

    data = get_regime_stats(db_session, _asof(), [*_TICKERS, "ZZZ"])["data"]

    assert data["missing"] == ["ZZZ"]
    assert data["assets"] == _TICKERS


def test_mixture_is_withheld_when_the_sample_is_too_thin(db_session) -> None:
    """Four assets over 70 days cannot support a full-covariance two-state fit."""
    rng = np.random.default_rng(5)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    _seed(db_session, {t: rng.normal(0.0, 0.01, size=70) for t in tickers})

    data = get_regime_stats(db_session, _asof(70), tickers)["data"]

    assert data["mixture"] is None
    assert data["turbulence"]["latest"] >= 0.0


def test_is_deterministic(db_session) -> None:
    """Identical data and arguments give identical output, mixture included."""
    _seed(db_session, _regime_shift(calm_end=False))

    first = get_regime_stats(db_session, _asof(), _TICKERS)
    second = get_regime_stats(db_session, _asof(), _TICKERS)

    assert first == second


def test_empty_tickers_is_an_error(db_session) -> None:
    """No tickers is a caller error, not an empty result."""
    result = get_regime_stats(db_session, _asof(), [])

    assert result["ok"] is False


def test_a_single_priced_asset_is_an_error(db_session) -> None:
    """Cross-asset statistics need at least two priced assets."""
    _seed(db_session, {"AAA": _independent()["AAA"]})

    result = get_regime_stats(db_session, _asof(), ["AAA", "ZZZ"])

    assert result["ok"] is False
    assert "two" in result["error"]


def test_too_few_observations_is_an_error(db_session) -> None:
    """Under 60 returns the statistics would be noise."""
    _seed(db_session, _independent(n_days=30))

    result = get_regime_stats(db_session, _asof(30), _TICKERS)

    assert result["ok"] is False
    assert "observations" in result["error"]


def test_bad_asof_is_an_error_not_raised(db_session) -> None:
    """A malformed date degrades through the envelope."""
    result = get_regime_stats(db_session, "not-a-date", _TICKERS)

    assert result["ok"] is False
