"""
Enumerations for Type-Safe String Constants
============================================
"""

from enum import Enum


class PriorType(str, Enum):
    """
    Prior return estimation methods for Black-Litterman model.
    """
    MARKET = "market"
    EQUAL = "equal"


class OmegaMethod(str, Enum):
    """
    View uncertainty matrix calculation methods for Black-Litterman.

    The omega matrix (Ω) represents the uncertainty in investor views.
    Higher uncertainty = less weight given to views.
    """
    DEFAULT = "default"
    IDZOREK = "idzorek"


class FixMethod(str, Enum):
    """
    Methods for fixing non-positive semidefinite covariance matrices.

    Covariance matrices must be positive semidefinite (PSD) for optimization.
    Numerical errors or insufficient data can create non-PSD matrices.
    """
    SPECTRAL = "spectral"
    DIAG = "diag"
