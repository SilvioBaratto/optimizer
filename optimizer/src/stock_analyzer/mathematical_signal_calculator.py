"""
Mathematical Signal Calculator - Backward Compatibility Layer
=============================================================

This module maintains backward compatibility with existing code by re-exporting
the refactored MathematicalSignalCalculator class from the core module.

**REFACTORED**: The original ~3000-line file has been refactored into a modular
structure for better maintainability:

    stock_analyzer/
    ├── core/                   # Main orchestrator
    │   └── signal_calculator.py
    ├── data/                   # Data fetching (yfinance, database)
    │   └── fetchers.py
    ├── technical/              # Technical indicators and metrics
    │   ├── indicators.py
    │   └── metrics.py
    ├── factors/                # Factor calculations (value, momentum, quality, growth)
    │   └── calculators.py
    ├── adjustments/            # Macro and risk adjustments
    │   ├── macro.py
    │   └── risk.py
    └── classification/         # Signal classification and scoring
        ├── distribution.py
        └── scoring.py

All functionality remains the same. Existing imports continue to work:

    from src.stock_analyzer.mathematical_signal_calculator import MathematicalSignalCalculator

For new code, you can import directly from the core module:

    from src.stock_analyzer.core import MathematicalSignalCalculator
"""

# Re-export MathematicalSignalCalculator for backward compatibility
from .core.signal_calculator import MathematicalSignalCalculator

__all__ = ['MathematicalSignalCalculator']


# Maintain test harness for backward compatibility
if __name__ == "__main__":
    """Test mathematical signal calculator"""
    import asyncio
    import logging
    from app.database import init_db
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)

    async def test_signal():
        print("=" * 80)
        print("MATHEMATICAL SIGNAL CALCULATOR TEST - NO LLM COSTS!")
        print("=" * 80)

        test_ticker = "AAPL"

        # Initialize database
        load_dotenv()
        init_db()

        # Test calculator
        calculator = MathematicalSignalCalculator()

        print(f"\nGenerating signal for {test_ticker}...")
        signal = await calculator.generate_signal(test_ticker)

        if signal:
            print(f"\n[SUCCESS] Signal generated:")
            print(f"  Signal Type:     {signal.signal_type.value}")
            print(f"  Confidence:      {signal.confidence_level.value}")
            print(f"  Data Quality:    {signal.data_quality_score:.2f}")
            print(f"  Close Price:     ${signal.close_price:.2f}")
            print(f"\n[SIGNAL DRIVERS]")
            print(f"  Technical:       {signal.signal_drivers.technical_score:+.2f}")
            print(f"  Valuation:       {signal.signal_drivers.valuation_score:+.2f}")
            print(f"  Momentum:        {signal.signal_drivers.momentum_score:+.2f}")
            print(f"  Quality:         {signal.signal_drivers.quality_score:+.2f}")
            print(f"\n[ANALYSIS]")
            print(f"  {signal.analysis_notes}")
            print(f"\n[COST] $0.00 (no LLM calls)")
        else:
            print(f"  Failed to generate signal")

        print("\n" + "=" * 80)

    asyncio.run(test_signal())
