"""Chart appearance shared by every figure builder."""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio

from optimizer.exceptions import ConfigurationError


@dataclass(frozen=True)
class ChartTheme:
    """Immutable look of a chart: template and size.

    Attributes:
        template: Name of a registered plotly template, for example
            ``"plotly_white"`` or ``"plotly_dark"``.
        height: Figure height in pixels.
        width: Figure width in pixels, or ``None`` to let the page decide.
    """

    template: str = "plotly_white"
    height: int = 500
    width: int | None = None

    def __post_init__(self) -> None:
        """Reject a template plotly does not know and a size below one pixel."""
        if self.template not in pio.templates:
            known = ", ".join(sorted(pio.templates))
            msg = f"template must be one of {known}, got {self.template!r}"
            raise ConfigurationError(msg)
        if self.height < 1:
            msg = f"height must be at least 1 pixel, got {self.height}"
            raise ConfigurationError(msg)
        if self.width is not None and self.width < 1:
            msg = f"width must be at least 1 pixel or None, got {self.width}"
            raise ConfigurationError(msg)

    def apply(self, figure: go.Figure) -> None:
        """Set the template, height and width on ``figure`` in place.

        Args:
            figure: Figure whose layout is updated.
        """
        figure.update_layout(
            template=self.template, height=self.height, width=self.width
        )


def resolve_theme(theme: ChartTheme | None) -> ChartTheme:
    """Return ``theme``, or the default theme when it is ``None``.

    Args:
        theme: Theme chosen by the caller, if any.

    Returns:
        The theme to apply.

    Raises:
        ConfigurationError: If ``theme`` is neither ``None`` nor a ``ChartTheme``.
    """
    if theme is None:
        return ChartTheme()
    if not isinstance(theme, ChartTheme):
        msg = f"theme must be a ChartTheme or None, got {type(theme).__name__}"
        raise ConfigurationError(msg)
    return theme
