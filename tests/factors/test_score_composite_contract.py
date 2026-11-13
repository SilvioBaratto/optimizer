"""Pin every branch of the composite macro-regime score.

``_score_composite`` scores PMI, the 2s10s spread, HY OAS and sentiment +1, 0
or -1 against their thresholds and maps the sum to a regime. The tables below
fix that mapping for every combination of component scores, each indicator at
and around both of its thresholds, inverted thresholds (where both conditions
hold at once), missing and non-finite readings, and the UNKNOWN fallback.
"""

from __future__ import annotations

import itertools
import logging
import math

import pandas as pd
import pytest

from optimizer.factors import (
    MacroRegime,
    RegimeThresholdConfig,
    classify_regime,
    classify_regime_composite,
)
from optimizer.factors._regime import _score_composite

_LOGGER = "optimizer.factors._regime"
_DEFAULT = RegimeThresholdConfig()

_EXPANSION = MacroRegime.EXPANSION
_SLOWDOWN = MacroRegime.SLOWDOWN
_RECOVERY = MacroRegime.RECOVERY
_RECESSION = MacroRegime.RECESSION

# Readings that score +1, 0 and -1 against the default thresholds.
_PMI = {1: 55.0, 0: 50.0, -1: 45.0}
_SPREAD = {1: 1.5, 0: 0.5, -1: -0.5}
_HY = {1: 300.0, 0: 400.0, -1: 600.0}
_SENTIMENT = {1: 0.5, 0: 0.0, -1: -0.5}

# The regime of each score sum, without and with a non-zero sentiment score.
_THREE_INDICATOR_REGIME = {
    3: _EXPANSION,
    2: _EXPANSION,
    1: _EXPANSION,
    0: _SLOWDOWN,
    -1: _RECOVERY,
    -2: _RECESSION,
    -3: _RECESSION,
}
_WITH_SENTIMENT_REGIME = {
    4: _EXPANSION,
    3: _EXPANSION,
    2: _EXPANSION,
    1: _EXPANSION,
    0: _SLOWDOWN,
    -1: _RECOVERY,
    -2: _RECOVERY,
    -3: _RECESSION,
    -4: _RECESSION,
}

_SCORES = (1, 0, -1)


def _score(
    pmi: float | None = None,
    spread: float | None = None,
    hy: float | None = None,
    sent: float | None = None,
    thresholds: RegimeThresholdConfig = _DEFAULT,
) -> MacroRegime:
    return _score_composite(
        pmi=pmi, spread=spread, hy=hy, sent=sent, thresholds=thresholds
    )


@pytest.mark.parametrize(
    ("pmi", "spread", "hy", "sent"), itertools.product(_SCORES, repeat=4)
)
def test_every_combination_of_component_scores(
    pmi: int, spread: int, hy: int, sent: int
) -> None:
    """All 81 score combinations map through the two literal ladders."""
    total = pmi + spread + hy
    expected = (
        _WITH_SENTIMENT_REGIME[total + sent] if sent else _THREE_INDICATOR_REGIME[total]
    )

    regime = _score(_PMI[pmi], _SPREAD[spread], _HY[hy], _SENTIMENT[sent])

    assert regime is expected


# Each indicator alone (PMI neutral for sentiment, which needs a partner):
# a reading strictly beyond a threshold scores, a reading on it does not.
_BOUNDARIES = {
    "pmi above expansion": ({"pmi": 52.000001}, _EXPANSION),
    "pmi on expansion": ({"pmi": 52.0}, _SLOWDOWN),
    "pmi between": ({"pmi": 50.0}, _SLOWDOWN),
    "pmi on contraction": ({"pmi": 48.0}, _SLOWDOWN),
    "pmi below contraction": ({"pmi": 47.999999}, _RECOVERY),
    "spread above steep": ({"spread": 1.000001}, _EXPANSION),
    "spread on steep": ({"spread": 1.0}, _SLOWDOWN),
    "spread on inversion": ({"spread": 0.0}, _SLOWDOWN),
    "spread below inversion": ({"spread": -0.000001}, _RECOVERY),
    "hy below risk-on": ({"hy": 349.999}, _EXPANSION),
    "hy on risk-on": ({"hy": 350.0}, _SLOWDOWN),
    "hy on risk-off": ({"hy": 500.0}, _SLOWDOWN),
    "hy above risk-off": ({"hy": 500.001}, _RECOVERY),
    "sentiment above positive": ({"pmi": 50.0, "sent": 0.300001}, _EXPANSION),
    "sentiment on positive": ({"pmi": 50.0, "sent": 0.3}, _SLOWDOWN),
    "sentiment on negative": ({"pmi": 50.0, "sent": -0.3}, _SLOWDOWN),
    "sentiment below negative": ({"pmi": 50.0, "sent": -0.300001}, _RECOVERY),
}


@pytest.mark.parametrize(
    ("readings", "expected"), _BOUNDARIES.values(), ids=_BOUNDARIES.keys()
)
def test_thresholds_are_strict(
    readings: dict[str, float], expected: MacroRegime
) -> None:
    """A reading equal to a threshold is neutral; only beyond it scores."""
    assert _score(**readings) is expected


_INVERTED = RegimeThresholdConfig(
    pmi_expansion=48.0,
    pmi_contraction=52.0,
    spread_2s10s_steep=0.0,
    spread_2s10s_inversion=1.0,
    hy_oas_risk_on=500.0,
    hy_oas_risk_off=350.0,
    sentiment_positive=-0.3,
    sentiment_negative=0.3,
)

# With each band inverted a reading inside it meets both conditions; the
# favourable one is tested first and wins, and outside it the plain rule holds.
# No reading is neutral any more, so PMI 50 (+1) partners the sentiment rows.
_INVERTED_READINGS = {
    "pmi inside": ({"pmi": 50.0}, _EXPANSION),
    "pmi above both": ({"pmi": 55.0}, _EXPANSION),
    "pmi below both": ({"pmi": 45.0}, _RECOVERY),
    "pmi on the lower": ({"pmi": 48.0}, _RECOVERY),
    "spread inside": ({"spread": 0.5}, _EXPANSION),
    "spread on the lower": ({"spread": 0.0}, _RECOVERY),
    "hy inside": ({"hy": 400.0}, _EXPANSION),
    "hy above both": ({"hy": 600.0}, _RECOVERY),
    "hy on the upper": ({"hy": 500.0}, _RECOVERY),
    "hy below both": ({"hy": 300.0}, _EXPANSION),
    "sentiment inside": ({"pmi": 50.0, "sent": 0.0}, _EXPANSION),
    "sentiment beyond the lower": ({"pmi": 50.0, "sent": -0.5}, _SLOWDOWN),
}


@pytest.mark.parametrize(
    ("readings", "expected"),
    _INVERTED_READINGS.values(),
    ids=_INVERTED_READINGS.keys(),
)
def test_inverted_thresholds_favour_the_positive_score(
    readings: dict[str, float], expected: MacroRegime
) -> None:
    """Inverted thresholds resolve an overlap in favour of the +1 score."""
    assert _score(**readings, thresholds=_INVERTED) is expected


_EQUAL = RegimeThresholdConfig(
    pmi_expansion=50.0,
    pmi_contraction=50.0,
    spread_2s10s_steep=0.5,
    spread_2s10s_inversion=0.5,
    hy_oas_risk_on=400.0,
    hy_oas_risk_off=400.0,
    sentiment_positive=0.0,
    sentiment_negative=0.0,
)


@pytest.mark.parametrize(
    ("readings", "expected"),
    [
        ({"pmi": 50.0, "spread": 0.5, "hy": 400.0, "sent": 0.0}, _SLOWDOWN),
        ({"pmi": 50.1, "spread": 0.6, "hy": 399.0, "sent": 0.1}, _EXPANSION),
        ({"pmi": 49.9, "spread": 0.4, "hy": 401.0, "sent": -0.1}, _RECESSION),
    ],
)
def test_equal_thresholds_leave_only_the_point_itself_neutral(
    readings: dict[str, float], expected: MacroRegime
) -> None:
    """With both thresholds equal, any reading off the point scores."""
    assert _score(**readings, thresholds=_EQUAL) is expected


_MISSING = {
    "pmi only": ({"pmi": 55.0}, _EXPANSION),
    "spread only": ({"spread": -0.5}, _RECOVERY),
    "hy only": ({"hy": 300.0}, _EXPANSION),
    "pmi and hy": ({"pmi": 55.0, "hy": 300.0}, _EXPANSION),
    "spread and hy": ({"spread": -0.5, "hy": 600.0}, _RECESSION),
    "pmi and spread with sentiment": (
        {"pmi": 45.0, "spread": -0.5, "sent": -0.5},
        _RECESSION,
    ),
    "hy with opposite sentiment": ({"hy": 600.0, "sent": 0.5}, _SLOWDOWN),
}


@pytest.mark.parametrize(
    ("readings", "expected"), _MISSING.values(), ids=_MISSING.keys()
)
def test_missing_indicators_are_neutral(
    readings: dict[str, float], expected: MacroRegime
) -> None:
    """An indicator passed as None scores 0, as if it sat between thresholds."""
    assert _score(**readings) is expected


@pytest.mark.parametrize(
    ("readings", "expected"),
    [
        ({"pmi": math.nan, "spread": 1.5}, _EXPANSION),
        ({"pmi": math.nan, "spread": math.nan, "hy": math.nan}, _SLOWDOWN),
        ({"pmi": 50.0, "sent": math.nan}, _SLOWDOWN),
        ({"pmi": math.inf}, _EXPANSION),
        ({"pmi": -math.inf}, _RECOVERY),
        ({"hy": math.inf}, _RECOVERY),
        ({"hy": -math.inf}, _EXPANSION),
        ({"spread": -0.0, "sent": math.inf}, _EXPANSION),
    ],
)
def test_non_finite_readings(readings: dict[str, float], expected: MacroRegime) -> None:
    """NaN fails every comparison and scores 0; infinities score like any reading."""
    assert _score(**readings) is expected


@pytest.mark.parametrize("sent", [None, -0.5, 0.0, 0.5])
def test_no_composite_reading_is_unknown_and_logged(
    sent: float | None, caplog: pytest.LogCaptureFixture
) -> None:
    """Without PMI, spread and HY the result is UNKNOWN, whatever the sentiment."""
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        regime = _score(sent=sent)

    assert regime is MacroRegime.UNKNOWN
    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        (
            _LOGGER,
            logging.WARNING,
            "classify_regime_composite: no observation for pmi, spread_2s10s "
            "or hy_oas; returning MacroRegime.UNKNOWN with neutral tilts.",
        )
    ]


def test_a_scored_regime_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Only the UNKNOWN fallback logs."""
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        _score(pmi=55.0, spread=1.5, hy=300.0, sent=0.5)

    assert caplog.records == []


def test_classifiers_score_the_latest_readings_with_the_given_thresholds() -> None:
    """Both public entry points score the latest readings, by default thresholds."""
    macro = pd.DataFrame(
        {
            "pmi": [45.0, 50.0],
            "spread_2s10s": [-0.5, math.nan],
            "hy_oas": [600.0, 400.0],
            "sentiment": [math.nan, math.nan],
        }
    )

    assert classify_regime_composite(macro) is _RECOVERY
    assert classify_regime(macro) is _RECOVERY
    assert classify_regime_composite(macro, _INVERTED) is _EXPANSION
    assert classify_regime(macro, _INVERTED) is _EXPANSION
