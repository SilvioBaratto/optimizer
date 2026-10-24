"""skfolio selectors that pass a too-small universe straight through.

skfolio's pre-selection transformers assume a usable universe: pairwise
selectors need two assets and sklearn's validation rejects a frame with no
columns.  A universe can legitimately shrink to one asset, or to none, after an
earlier step drops assets, and that is a valid outcome to return rather than an
error to raise.  Each guarded selector below behaves exactly like its skfolio
parent once there are enough assets and keeps every asset otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skfolio.pre_selection import (
    DropCorrelated,
    DropZeroVariance,
    SelectKExtremes,
    SelectNonDominated,
    SelectNonExpiring,
)


class _SmallUniverseGuard:
    """Mixin that skips the wrapped selector when too few assets are present.

    Must precede the skfolio selector in the bases so its ``fit`` / ``transform``
    run first.  Subclasses set ``_min_assets`` to the smallest universe the
    parent can fit on.
    """

    _min_assets: int = 1

    def fit(self, X: Any, y: None = None) -> Any:
        """Fit the parent selector, or keep every asset on a tiny universe.

        Args:
            X: Returns of the assets, shape (n_observations, n_assets).
            y: Ignored; present for sklearn API compatibility.

        Returns:
            The fitted selector.
        """
        n_assets = int(np.shape(X)[1])
        if n_assets < self._min_assets:
            # Record what sklearn's feature-name / feature-count checks read at
            # transform time, since the parent's validation never ran.
            self.n_features_in_ = n_assets
            columns = getattr(X, "columns", None)
            if columns is not None and all(isinstance(c, str) for c in columns):
                self.feature_names_in_ = np.asarray(columns, dtype=object)
            self.to_keep_ = np.ones(n_assets, dtype=bool)
            return self
        super().fit(X, y)  # type: ignore[misc]
        return self

    def transform(self, X: Any) -> Any:
        """Select the kept assets; a frame with no assets is returned unchanged.

        Args:
            X: Returns of the assets with the columns seen in fit.

        Returns:
            ``X`` restricted to the selected assets.
        """
        if int(np.shape(X)[1]) == 0:
            return X
        return super().transform(X)  # type: ignore[misc]


class GuardedDropZeroVariance(_SmallUniverseGuard, DropZeroVariance):
    """`DropZeroVariance` that tolerates an empty universe."""


class GuardedDropCorrelated(_SmallUniverseGuard, DropCorrelated):
    """`DropCorrelated` that tolerates fewer than two assets."""

    _min_assets = 2


class GuardedSelectKExtremes(_SmallUniverseGuard, SelectKExtremes):
    """`SelectKExtremes` that tolerates an empty universe."""


class GuardedSelectNonDominated(_SmallUniverseGuard, SelectNonDominated):
    """`SelectNonDominated` that tolerates an empty universe."""


class GuardedSelectNonExpiring(_SmallUniverseGuard, SelectNonExpiring):
    """`SelectNonExpiring` that tolerates an empty universe."""
