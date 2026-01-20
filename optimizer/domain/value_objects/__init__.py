"""
Value Objects - Immutable domain types.

Value objects are immutable and defined by their attributes rather than identity.
They are used to encapsulate domain concepts and ensure type safety.
"""

from optimizer.domain.value_objects.weights import PortfolioWeights
from optimizer.domain.value_objects.covariance import CovarianceMatrix

__all__ = [
    "PortfolioWeights",
    "CovarianceMatrix",
]
