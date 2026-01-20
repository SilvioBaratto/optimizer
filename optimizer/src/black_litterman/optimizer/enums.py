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
    """

    DEFAULT = "default"
    IDZOREK = "idzorek"


class FixMethod(str, Enum):
    """
    Methods for fixing non-positive semidefinite covariance matrices.
    """

    SPECTRAL = "spectral"
    DIAG = "diag"
