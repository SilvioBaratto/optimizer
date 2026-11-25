"""Chart appearance shared by every figure builder."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import plotly.graph_objects as go
import plotly.io as pio

from optimizer.exceptions import ConfigurationError
from optimizer.plotting._validation import require_integer

# plotly rejects a layout height or width below 10 pixels, so a smaller theme would
# only fail later, inside every figure builder that applies it.
_MIN_PIXELS: Final = 10


class CorrelationOrder(str, Enum):
    """How the assets of a correlation heatmap are ordered.

    Attributes:
        NONE: Keep the column order of the input.
        CLUSTER: Reorder by hierarchical clustering so similar assets sit together.
    """

    NONE = "none"
    CLUSTER = "cluster"


@dataclass(frozen=True)
class ChartTheme:
    """Immutable look of a chart: template and size.

    Attributes:
        template: Name of a registered plotly template, for example
            ``"plotly_white"`` or ``"plotly_dark"``.
        height: Figure height in whole pixels, at least 10 (plotly's minimum).
        width: Figure width in whole pixels, at least 10, or ``None`` to let the page
            decide.
    """

    template: str = "plotly_white"
    height: int = 500
    width: int | None = None

    def __post_init__(self) -> None:
        """Reject an unknown template and a size plotly cannot draw."""
        if self.template not in pio.templates:
            known = ", ".join(sorted(pio.templates))
            msg = f"template must be one of {known}, got {self.template!r}"
            raise ConfigurationError(msg)
        require_integer("height", self.height, _MIN_PIXELS)
        if self.width is not None:
            require_integer("width", self.width, _MIN_PIXELS)

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
