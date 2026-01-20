"""
Views Module - Black-Litterman view generation and matrix construction.

Provides:
- ViewGeneratorImpl: Generate views from signals using BAML
- ViewMatrixBuilder: Construct P, Q, Omega matrices from views

Usage:
    from src.black_litterman.views import ViewGeneratorImpl, ViewMatrixBuilder

    generator = ViewGeneratorImpl(signal_repo, macro_repo)
    views = await generator.generate(signals, macro_regime)

    builder = ViewMatrixBuilder()
    P, Q, Omega = builder.construct(views, universe_tickers)
"""

from optimizer.src.black_litterman.views.generator import ViewGeneratorImpl
from optimizer.src.black_litterman.views.matrix_builder import ViewMatrixBuilder

__all__ = [
    "ViewGeneratorImpl",
    "ViewMatrixBuilder",
]
