"""`SectorImputer` variant that refuses rows it has no peer observation for."""

from __future__ import annotations

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from optimizer.exceptions import DataError
from optimizer.preprocessing._imputation import SectorImputer

_MAX_DATES_SHOWN = 5


class StrictSectorImputer(SectorImputer):
    """`SectorImputer` that raises on a row in which every asset is missing.

    The imputer fills a gap from the other assets on the same date, so such a
    row would stay ``NaN`` and the next selector would reject it with an error
    naming neither the date nor the imputer.  `SectorImputer` itself keeps its
    documented behaviour of returning the row as ``NaN``, since it is also used
    standalone and inside the cleaning pipeline; only the pre-selection
    pipeline, whose selectors cannot take ``NaN``, needs the check.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN with leave-one-out sector averages.

        Args:
            X: Asset return DataFrame; must have the same columns seen in fit.

        Returns:
            DataFrame with the same shape as X, missing values filled.

        Raises:
            DataError: If X is not a DataFrame, or if some row is missing for
                every asset, since there is then no peer observation to
                impute from.
        """
        check_is_fitted(self)
        frame = self._validate_input(X)
        self._raise_on_unfillable_rows(frame)
        return super().transform(frame)

    @staticmethod
    def _raise_on_unfillable_rows(X: pd.DataFrame) -> None:
        # ``all`` over zero columns is vacuously true and would flag every row
        # of an empty selection, which has nothing to impute.
        if X.shape[1] == 0:
            return
        empty = X.index[X.isna().all(axis=1)]
        if len(empty) == 0:
            return
        shown = ", ".join(str(label) for label in empty[:_MAX_DATES_SHOWN])
        more = (
            f" (+{len(empty) - _MAX_DATES_SHOWN} more)"
            if len(empty) > _MAX_DATES_SHOWN
            else ""
        )
        raise DataError(
            f"Pre-selection imputation cannot fill {len(empty)} row(s) in which "
            f"every asset is missing: {shown}{more}. No peer observation exists "
            "to impute from; drop these dates before fitting."
        )
