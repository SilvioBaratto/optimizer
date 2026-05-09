"""Identity smoke tests for CS-transformer re-exports.

Ensures the ``optimizer.preprocessing`` re-exports are the same class
objects as the upstream ``skfolio.preprocessing`` symbols. Catches
silent breakage when skfolio renames or relocates a class at the pin
boundary.
"""

from __future__ import annotations

import skfolio.preprocessing as sk

import optimizer.preprocessing as op


def test_when_cs_gaussian_rank_scaler_imported_then_matches_skfolio() -> None:
    assert op.CSGaussianRankScaler is sk.CSGaussianRankScaler


def test_when_cs_percentile_rank_scaler_imported_then_matches_skfolio() -> None:
    assert op.CSPercentileRankScaler is sk.CSPercentileRankScaler


def test_when_cs_standard_scaler_imported_then_matches_skfolio() -> None:
    assert op.CSStandardScaler is sk.CSStandardScaler


def test_when_cs_tanh_shrinker_imported_then_matches_skfolio() -> None:
    assert op.CSTanhShrinker is sk.CSTanhShrinker


def test_when_cs_winsorizer_imported_then_matches_skfolio() -> None:
    assert op.CSWinsorizer is sk.CSWinsorizer
