"""Unit tests for the news sentiment pipeline (issue #13)."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.sentiment import (
    _ALPHA_MAX,
    _ALPHA_MIN,
    adjust_idzorek_alpha,
    compute_sentiment_signal,
    fetch_news_sentiment,
)
from app.services.view_generation import adjust_view_confidences
from baml_client.types import AssetView, NewsSentimentOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_view(
    asset: str = "AAPL",
    direction: int = 1,
    confidence: float = 0.6,
) -> AssetView:
    return AssetView(
        asset=asset,
        direction=direction,
        magnitude_bps=200.0,
        confidence=confidence,
        reasoning="test",
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _series(scores: list[float], ages_days: list[float]) -> pd.Series:
    """Build a sentiment Series with publish times ``ages_days`` ago."""
    now = _now()
    dates = [now - timedelta(days=d) for d in ages_days]
    return pd.Series(scores, index=pd.DatetimeIndex(dates), dtype=float)


# ===========================================================================
# TestComputeSentimentSignal
# ===========================================================================


class TestComputeSentimentSignal:
    def test_empty_series_returns_zero(self) -> None:
        assert compute_sentiment_signal(pd.Series(dtype=float)) == 0.0

    def test_single_fresh_article_bullish(self) -> None:
        s = _series([0.8], [0.0])  # published just now
        signal = compute_sentiment_signal(s, half_life_days=5.0)
        # weight ≈ exp(0) = 1.0
        assert pytest.approx(signal, abs=1e-3) == 0.8

    def test_single_fresh_article_bearish(self) -> None:
        s = _series([-0.6], [0.0])
        signal = compute_sentiment_signal(s, half_life_days=5.0)
        assert signal < 0.0
        assert pytest.approx(signal, abs=1e-3) == -0.6

    def test_half_life_decay(self) -> None:
        """Article published exactly half_life_days ago should have weight 0.5."""
        half_life = 5.0
        s = _series([1.0], [half_life])
        signal = compute_sentiment_signal(s, half_life_days=half_life)
        assert pytest.approx(signal, abs=1e-3) == 0.5

    def test_old_news_diminishes(self) -> None:
        """Recent news outweighs identical old news."""
        half_life = 5.0
        fresh = _series([1.0], [0.0])
        old = _series([1.0], [30.0])  # 30 days old
        assert compute_sentiment_signal(fresh, half_life) > compute_sentiment_signal(old, half_life)

    def test_multiple_articles_summed(self) -> None:
        """Sum of two fresh articles with opposite signs cancels out."""
        s = _series([1.0, -1.0], [0.0, 0.0])
        signal = compute_sentiment_signal(s, half_life_days=5.0)
        assert pytest.approx(signal, abs=1e-6) == 0.0

    def test_zero_signal_all_neutral(self) -> None:
        s = _series([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        assert compute_sentiment_signal(s) == 0.0

    def test_naive_datetimes_treated_as_utc(self) -> None:
        """Naive datetimes should not raise and produce a valid signal."""
        now_naive = datetime.utcnow()
        s = pd.Series([0.5], index=pd.DatetimeIndex([now_naive]))
        signal = compute_sentiment_signal(s, half_life_days=5.0)
        assert pytest.approx(signal, abs=1e-3) == 0.5

    def test_invalid_half_life_raises(self) -> None:
        s = _series([0.5], [0.0])
        with pytest.raises(ValueError):
            compute_sentiment_signal(s, half_life_days=0.0)

    def test_default_half_life_is_five_days(self) -> None:
        s = _series([1.0], [5.0])
        signal = compute_sentiment_signal(s)  # default half_life_days=5.0
        assert pytest.approx(signal, abs=1e-3) == 0.5


# ===========================================================================
# TestAdjustIdzorekAlpha
# ===========================================================================


class TestAdjustIdzorekAlpha:
    # --- Acceptance criteria from issue #13 ---

    def test_positive_sentiment_positive_view_increases_alpha(self) -> None:
        """Aligned bullish news + bullish view → alpha goes up."""
        alpha = adjust_idzorek_alpha(0.5, sentiment_signal=0.4, view_direction=1)
        assert alpha > 0.5

    def test_negative_sentiment_positive_view_decreases_alpha(self) -> None:
        """Conflicting bearish news + bullish view → alpha goes down."""
        alpha = adjust_idzorek_alpha(0.5, sentiment_signal=-0.4, view_direction=1)
        assert alpha < 0.5

    def test_zero_sentiment_unchanged(self) -> None:
        """Zero sentiment signal → alpha unchanged (after clamping)."""
        base = 0.65
        alpha = adjust_idzorek_alpha(base, sentiment_signal=0.0, view_direction=1)
        assert alpha == pytest.approx(base, abs=1e-9)

    def test_always_clamped_above_min(self) -> None:
        """Even extreme conflicting signal → alpha ≥ 0.01."""
        alpha = adjust_idzorek_alpha(0.5, sentiment_signal=-100.0, view_direction=1)
        assert alpha == pytest.approx(_ALPHA_MIN)

    def test_always_clamped_below_max(self) -> None:
        """Even extreme aligned signal → alpha ≤ 0.99."""
        alpha = adjust_idzorek_alpha(0.5, sentiment_signal=100.0, view_direction=1)
        assert alpha == pytest.approx(_ALPHA_MAX)

    # --- Directional symmetry ---

    def test_bearish_news_confirms_bearish_view(self) -> None:
        """Bearish news + bearish view (direction=-1) should boost alpha."""
        alpha = adjust_idzorek_alpha(0.5, sentiment_signal=-0.4, view_direction=-1)
        assert alpha > 0.5

    def test_bullish_news_conflicts_bearish_view(self) -> None:
        """Bullish news + bearish view (direction=-1) should reduce alpha."""
        alpha = adjust_idzorek_alpha(0.5, sentiment_signal=0.4, view_direction=-1)
        assert alpha < 0.5

    # --- Magnitude scaling ---

    def test_stronger_alignment_gives_higher_alpha(self) -> None:
        alpha_weak = adjust_idzorek_alpha(0.5, sentiment_signal=0.2, view_direction=1)
        alpha_strong = adjust_idzorek_alpha(0.5, sentiment_signal=0.8, view_direction=1)
        assert alpha_strong > alpha_weak

    def test_stronger_conflict_gives_lower_alpha(self) -> None:
        alpha_weak = adjust_idzorek_alpha(0.5, sentiment_signal=-0.2, view_direction=1)
        alpha_strong = adjust_idzorek_alpha(0.5, sentiment_signal=-0.8, view_direction=1)
        assert alpha_strong < alpha_weak

    # --- Edge values ---

    def test_base_at_min_still_at_min(self) -> None:
        alpha = adjust_idzorek_alpha(0.0, sentiment_signal=-0.5, view_direction=1)
        assert alpha == pytest.approx(_ALPHA_MIN)

    def test_base_at_max_still_clamped(self) -> None:
        alpha = adjust_idzorek_alpha(1.0, sentiment_signal=0.5, view_direction=1)
        assert alpha == pytest.approx(_ALPHA_MAX)


# ===========================================================================
# TestFetchNewsSentiment
# ===========================================================================


class TestFetchNewsSentiment:
    def _make_news_row(
        self,
        title: str,
        age_days: float = 1.0,
    ) -> MagicMock:
        row = MagicMock()
        row.title = title
        row.publish_time = _now() - timedelta(days=age_days)
        return row

    def test_unknown_ticker_returns_empty_series(self) -> None:
        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = None

        result = fetch_news_sentiment(session, "UNKN")

        assert isinstance(result, pd.Series)
        assert result.empty

    def test_no_news_returns_empty_series(self) -> None:
        session = MagicMock()
        instrument_id = uuid.uuid4()
        # First call returns instrument id; second returns empty news list
        session.execute.return_value.scalar_one_or_none.return_value = instrument_id
        session.execute.return_value.scalars.return_value.all.return_value = []

        result = fetch_news_sentiment(session, "AAPL")

        assert isinstance(result, pd.Series)
        assert result.empty

    @patch("app.services.sentiment.b.ScoreNewsSentiment")
    def test_returns_series_with_scores(self, mock_score: MagicMock) -> None:
        mock_score.return_value = NewsSentimentOutput(
            scores=[0.8, -0.3],
            reasoning="test",
        )

        session = MagicMock()
        instrument_id = uuid.uuid4()

        rows = [
            self._make_news_row("Apple reports record earnings", age_days=1.0),
            self._make_news_row("Apple faces antitrust probe", age_days=3.0),
        ]

        # Simulate two execute() calls: first for instrument, second for news
        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = instrument_id
        mock_exec.scalars.return_value.all.return_value = rows
        session.execute.return_value = mock_exec

        result = fetch_news_sentiment(session, "AAPL", lookback_days=30)

        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert result.iloc[0] == pytest.approx(0.8)
        assert result.iloc[1] == pytest.approx(-0.3)

    @patch("app.services.sentiment.b.ScoreNewsSentiment")
    def test_mismatched_score_count_padded(self, mock_score: MagicMock) -> None:
        """If BAML returns fewer scores than articles, pad with zeros."""
        mock_score.return_value = NewsSentimentOutput(
            scores=[0.5],  # Only 1 score for 2 articles
            reasoning="test",
        )

        session = MagicMock()
        instrument_id = uuid.uuid4()
        rows = [
            self._make_news_row("Article 1", age_days=1.0),
            self._make_news_row("Article 2", age_days=2.0),
        ]

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = instrument_id
        mock_exec.scalars.return_value.all.return_value = rows
        session.execute.return_value = mock_exec

        result = fetch_news_sentiment(session, "AAPL")

        assert len(result) == 2
        assert result.iloc[0] == pytest.approx(0.5)
        assert result.iloc[1] == pytest.approx(0.0)  # padded

    @patch("app.services.sentiment.b.ScoreNewsSentiment")
    def test_mismatched_score_count_truncated(self, mock_score: MagicMock) -> None:
        """If BAML returns more scores than articles, truncate."""
        mock_score.return_value = NewsSentimentOutput(
            scores=[0.5, 0.7, 0.9],  # 3 scores for 1 article
            reasoning="test",
        )

        session = MagicMock()
        instrument_id = uuid.uuid4()
        rows = [self._make_news_row("Article 1", age_days=1.0)]

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = instrument_id
        mock_exec.scalars.return_value.all.return_value = rows
        session.execute.return_value = mock_exec

        result = fetch_news_sentiment(session, "AAPL")

        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(0.5)

    @patch("app.services.sentiment.b.ScoreNewsSentiment")
    def test_baml_failure_returns_empty_series(self, mock_score: MagicMock) -> None:
        """If BAML call throws, return empty Series gracefully."""
        mock_score.side_effect = RuntimeError("LLM timeout")

        session = MagicMock()
        instrument_id = uuid.uuid4()
        rows = [self._make_news_row("Some news", age_days=1.0)]

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = instrument_id
        mock_exec.scalars.return_value.all.return_value = rows
        session.execute.return_value = mock_exec

        result = fetch_news_sentiment(session, "AAPL")

        assert result.empty

    @patch("app.services.sentiment.b.ScoreNewsSentiment")
    def test_none_publish_time_handled(self, mock_score: MagicMock) -> None:
        """News rows with NULL publish_time fall back to current time."""
        mock_score.return_value = NewsSentimentOutput(
            scores=[0.5],
            reasoning="test",
        )

        session = MagicMock()
        instrument_id = uuid.uuid4()

        row = MagicMock()
        row.title = "Some news"
        row.publish_time = None  # NULL in DB

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = instrument_id
        mock_exec.scalars.return_value.all.return_value = [row]
        session.execute.return_value = mock_exec

        result = fetch_news_sentiment(session, "AAPL")

        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(0.5)


# ===========================================================================
# TestAdjustViewConfidences
# ===========================================================================


class TestAdjustViewConfidences:
    def test_empty_views_returns_empty(self) -> None:
        result = adjust_view_confidences([], {"AAPL": 0.5})
        assert result == []

    def test_empty_sentiment_map_leaves_views_unchanged(self) -> None:
        views = [_make_view("AAPL", direction=1, confidence=0.6)]
        result = adjust_view_confidences(views, {})
        assert result[0].confidence == pytest.approx(0.6)

    def test_positive_sentiment_boosts_bullish_view(self) -> None:
        views = [_make_view("AAPL", direction=1, confidence=0.5)]
        result = adjust_view_confidences(views, {"AAPL": 0.4})
        assert result[0].confidence > 0.5

    def test_negative_sentiment_reduces_bullish_view(self) -> None:
        views = [_make_view("AAPL", direction=1, confidence=0.5)]
        result = adjust_view_confidences(views, {"AAPL": -0.4})
        assert result[0].confidence < 0.5

    def test_missing_ticker_in_map_leaves_view_unchanged(self) -> None:
        views = [_make_view("MSFT", direction=1, confidence=0.7)]
        result = adjust_view_confidences(views, {"AAPL": 0.8})
        assert result[0].confidence == pytest.approx(0.7)

    def test_original_views_not_mutated(self) -> None:
        """adjust_view_confidences must not mutate the input list."""
        views = [_make_view("AAPL", direction=1, confidence=0.6)]
        original_confidence = views[0].confidence
        adjust_view_confidences(views, {"AAPL": 0.8})
        assert views[0].confidence == pytest.approx(original_confidence)

    def test_multiple_views_adjusted_independently(self) -> None:
        views = [
            _make_view("AAPL", direction=1, confidence=0.5),
            _make_view("MSFT", direction=-1, confidence=0.5),
        ]
        sentiment_map = {
            "AAPL": 0.4,   # bullish signal, bullish view → boost
            "MSFT": -0.4,  # bearish signal, bearish view → boost
        }
        result = adjust_view_confidences(views, sentiment_map)
        assert result[0].confidence > 0.5  # AAPL boosted
        assert result[1].confidence > 0.5  # MSFT boosted (confirmed)

    def test_other_view_fields_preserved(self) -> None:
        views = [_make_view("AAPL", direction=1, confidence=0.6)]
        result = adjust_view_confidences(views, {"AAPL": 0.3})
        assert result[0].asset == "AAPL"
        assert result[0].direction == 1
        assert result[0].magnitude_bps == pytest.approx(200.0)
        assert result[0].reasoning == "test"

    def test_result_is_new_list_not_in_place(self) -> None:
        views = [_make_view("AAPL", direction=1, confidence=0.5)]
        result = adjust_view_confidences(views, {"AAPL": 0.5})
        assert result is not views
        assert result[0] is not views[0]

    def test_clamped_to_min_when_extreme_conflict(self) -> None:
        views = [_make_view("AAPL", direction=1, confidence=0.5)]
        result = adjust_view_confidences(views, {"AAPL": -100.0})
        assert result[0].confidence == pytest.approx(_ALPHA_MIN)

    def test_clamped_to_max_when_extreme_alignment(self) -> None:
        views = [_make_view("AAPL", direction=1, confidence=0.5)]
        result = adjust_view_confidences(views, {"AAPL": 100.0})
        assert result[0].confidence == pytest.approx(_ALPHA_MAX)


# ===========================================================================
# Integration: compute_sentiment_signal + adjust_idzorek_alpha
# ===========================================================================


class TestSentimentPipelineIntegration:
    """End-to-end flow: series → signal → adjusted alpha."""

    def test_aligned_recent_news_boosts_alpha(self) -> None:
        """Bullish news from today + bullish view → alpha rises."""
        series = _series([0.7, 0.5], [0.5, 2.0])  # recent bullish news
        signal = compute_sentiment_signal(series, half_life_days=5.0)
        alpha = adjust_idzorek_alpha(0.5, signal, view_direction=1)
        assert alpha > 0.5

    def test_conflicting_recent_news_lowers_alpha(self) -> None:
        """Bearish news from today + bullish view → alpha drops."""
        series = _series([-0.7, -0.5], [0.5, 2.0])  # recent bearish news
        signal = compute_sentiment_signal(series, half_life_days=5.0)
        alpha = adjust_idzorek_alpha(0.5, signal, view_direction=1)
        assert alpha < 0.5

    def test_old_news_has_negligible_effect(self) -> None:
        """News from 60 days ago (half_life=5) contributes almost nothing."""
        series_old = _series([1.0], [60.0])
        signal_old = compute_sentiment_signal(series_old, half_life_days=5.0)
        # exp(-ln(2)/5 * 60) ≈ exp(-8.3) ≈ 0.00025
        expected_weight = math.exp(-math.log(2) / 5.0 * 60.0)
        assert signal_old == pytest.approx(expected_weight, rel=1e-3)
        # Should produce minimal alpha change
        alpha = adjust_idzorek_alpha(0.5, signal_old, view_direction=1)
        assert abs(alpha - 0.5) < 0.01  # nearly unchanged

    def test_zero_net_signal_leaves_alpha_unchanged(self) -> None:
        """Equal bullish and bearish news cancel → alpha stays the same."""
        series = _series([1.0, -1.0], [0.0, 0.0])
        signal = compute_sentiment_signal(series, half_life_days=5.0)
        assert signal == pytest.approx(0.0, abs=1e-9)
        alpha = adjust_idzorek_alpha(0.5, signal, view_direction=1)
        assert alpha == pytest.approx(0.5, abs=1e-9)
