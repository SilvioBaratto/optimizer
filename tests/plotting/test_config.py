"""Tests for the shared chart theme and its use by the figure builders."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.plotting import (
    ChartTheme,
    correlation_heatmap,
    efficient_frontier_figure,
)


class TestChartTheme:
    """``ChartTheme`` validates its fields and is immutable."""

    def test_defaults_are_the_documented_ones(self) -> None:
        """The default theme is a 500 pixel high white chart of automatic width."""
        theme = ChartTheme()
        assert theme.template == "plotly_white"
        assert theme.height == 500
        assert theme.width is None

    def test_a_theme_cannot_be_changed_after_creation(self) -> None:
        """The theme is shared between figures, so it is frozen."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            ChartTheme().height = 300  # type: ignore[misc]

    @pytest.mark.parametrize("height", [0, -5])
    def test_a_height_below_one_is_rejected(self, height: int) -> None:
        """A chart needs at least one pixel of height."""
        with pytest.raises(ConfigurationError, match="height"):
            ChartTheme(height=height)

    @pytest.mark.parametrize("width", [0, -1])
    def test_a_width_below_one_is_rejected(self, width: int) -> None:
        """An explicit width needs at least one pixel."""
        with pytest.raises(ConfigurationError, match="width"):
            ChartTheme(width=width)

    def test_an_unknown_template_is_rejected(self) -> None:
        """Only templates plotly knows can be applied."""
        with pytest.raises(ConfigurationError, match="template"):
            ChartTheme(template="no-such-template")


class TestThemeApplication:
    """The builders apply the theme they are given, or the default."""

    @staticmethod
    def _frontier(theme: ChartTheme | None):
        """Build a minimal frontier figure with the given theme."""
        return efficient_frontier_figure([0.1, 0.2], [0.05, 0.08], theme=theme)

    def test_the_default_theme_is_used_when_none_is_given(self) -> None:
        """Without a theme the figure is 500 pixels high on the white template."""
        figure = self._frontier(None)
        assert figure.layout.height == 500
        assert figure.layout.width is None
        assert figure.layout.template == pio.templates["plotly_white"]

    def test_a_custom_theme_reaches_the_layout(self) -> None:
        """Template, height and width are copied onto the figure layout."""
        theme = ChartTheme(template="plotly_dark", height=320, width=640)
        figure = self._frontier(theme)
        assert figure.layout.height == 320
        assert figure.layout.width == 640
        assert figure.layout.template == pio.templates["plotly_dark"]

    def test_every_builder_accepts_the_same_theme(self) -> None:
        """The heatmap honours the theme just like the frontier."""
        frame = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 3)))
        figure = correlation_heatmap(frame, theme=ChartTheme(height=250))
        assert figure.layout.height == 250

    def test_a_theme_of_the_wrong_type_is_rejected(self) -> None:
        """A string is not a theme: the mistake is reported, not ignored."""
        with pytest.raises(ConfigurationError, match="ChartTheme"):
            self._frontier("plotly_dark")  # type: ignore[arg-type]
