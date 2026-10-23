"""`SelectComplete` variant that runs on the raw, un-cleaned return frame."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from skfolio.pre_selection import SelectComplete

from optimizer.exceptions import DataError
from optimizer.pre_selection._guarded import _SmallUniverseGuard


class RawSelectComplete(_SmallUniverseGuard, SelectComplete):
    """`SelectComplete` that tolerates infinities in the frame it inspects.

    Completeness is a statement about *missing* observations, so only ``NaN``
    marks a gap.  An infinite return is a present-but-invalid observation that
    the validation step turns into ``NaN`` later; it must not count as a gap and
    skfolio's own ``fit`` would otherwise raise on it.  Running this selector
    before any cleaning keeps cleaning-created ``NaN`` from being mistaken for a
    late listing or an early expiry.

    For the same reason ``drop_assets_with_internal_nan`` sees only the gaps the
    caller supplied: a cell that cleaning blanks later is imputed, not counted
    as an internal gap.

    Only a `pandas.DataFrame` is accepted.  As the first pipeline step this
    selector is where asset labels enter, and a sector mapping and every later
    step rely on them; a bare array would silently lose them.
    """

    def fit(self, X: Any, y: None = None) -> RawSelectComplete:
        """Select assets with no missing first/last (or any, if configured) value.

        Args:
            X: Returns of the assets, shape (n_observations, n_assets).
            y: Ignored; present for sklearn API compatibility.

        Returns:
            The fitted selector.

        Raises:
            DataError: If X is not a DataFrame.
        """
        self._require_frame(X)
        super().fit(X.replace([np.inf, -np.inf], 0.0), y)
        return self

    def transform(self, X: Any) -> Any:
        """Select the kept assets; a frame with no assets is returned unchanged.

        Args:
            X: Returns of the assets with the columns seen in fit.

        Returns:
            ``X`` restricted to the selected assets.

        Raises:
            DataError: If X is not a DataFrame.
        """
        self._require_frame(X)
        return super().transform(X)

    @staticmethod
    def _require_frame(X: Any) -> None:
        if not isinstance(X, pd.DataFrame):
            raise DataError(
                f"RawSelectComplete requires a pandas DataFrame, got {type(X).__name__}"
            )
