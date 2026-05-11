"""Return preprocessing and after-tax computation for the research pipeline.

Re-exports from ``_preprocessing.py`` and ``_tax.py`` — the canonical
locations for FX price conversion, return preprocessing pipeline, and
after-tax return reconstruction.
"""

from research.returns._preprocessing import (
    apply_fx_to_prices,
    build_return_preprocessing_pipeline,
)
from research.returns._tax import DEFAULT_TAX_RATE, compute_after_tax_returns

__all__ = [
    "DEFAULT_TAX_RATE",
    "apply_fx_to_prices",
    "build_return_preprocessing_pipeline",
    "compute_after_tax_returns",
]
