"""Transform-time column alignment shared by the column-aware transformers.

``DataValidator``, ``OutlierTreater``, ``SectorImputer`` and
``RegressionImputer`` expose ``get_feature_names_out`` so sklearn's
``set_output(transform="pandas")`` can label their result.  sklearn assigns
those names *by position*, so a frame whose columns arrive reordered or
substituted at ``transform`` would come back mislabelled — and, being the first
pipeline steps, they also hide the problem from skfolio's later feature-name
checks.  Each transformer therefore re-aligns its input to the fit columns here
before doing any work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer.exceptions import DataError

__all__ = ["_align_to_fit_columns"]


def _align_to_fit_columns(
    X: pd.DataFrame, feature_names_in: np.ndarray, who: str
) -> pd.DataFrame:
    """Return X with its columns in the order seen during fit.

    Args:
        X: Frame passed to ``transform``.
        feature_names_in: Column names recorded by ``fit``.
        who: Caller name embedded in the error message for diagnostics.

    Returns:
        X unchanged when the columns already match, else X reordered by label.

    Raises:
        DataError: If X is missing a fit column, carries a column unseen in
            fit, or repeats a label.
    """
    expected = list(feature_names_in)
    actual = list(X.columns)
    if actual == expected:
        return X

    expected_set, actual_set = set(expected), set(actual)
    missing = [c for c in expected if c not in actual_set]
    unexpected = [c for c in actual if c not in expected_set]
    duplicated = len(actual) != len(actual_set)
    if missing or unexpected or duplicated:
        problems = []
        if missing:
            problems.append(f"missing {missing}")
        if unexpected:
            problems.append(f"unexpected {unexpected}")
        if duplicated:
            problems.append("duplicated labels")
        raise DataError(
            f"{who}: columns at transform differ from those seen in fit "
            f"({'; '.join(problems)})"
        )
    return X.reindex(columns=expected)
