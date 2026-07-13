"""Macro Services."""

from app.services.macro.macro_calibration import (
    CalibrationResult,
    classify_macro_regime,
    run_bulk_calibrate,
)
from app.services.macro.macro_news_summary import (
    CountrySummaryResult,
    run_news_summarize,
)
from app.services.macro.macro_regime_service import (
    MacroRegimeService,
    run_bulk_fred_fetch,
    run_bulk_macro_fetch,
    run_macro_news_fetch,
)

__all__ = [
    "CalibrationResult",
    "CountrySummaryResult",
    "MacroRegimeService",
    "classify_macro_regime",
    "run_bulk_calibrate",
    "run_bulk_fred_fetch",
    "run_bulk_macro_fetch",
    "run_macro_news_fetch",
    "run_news_summarize",
]
