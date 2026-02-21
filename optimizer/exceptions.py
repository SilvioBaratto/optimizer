"""Custom exception hierarchy for the optimizer library."""


class OptimizerError(Exception):
    """Base exception for all optimizer library errors."""


class ConfigurationError(OptimizerError):
    """Invalid configuration parameters or missing required arguments."""


class DataError(OptimizerError):
    """Invalid input data: wrong type, shape, or alignment."""


class ConvergenceError(OptimizerError):
    """An iterative algorithm failed to converge."""


class ValidationError(OptimizerError):
    """Post-optimization validation failure."""


class OptimizationError(OptimizerError):
    """Portfolio optimization solver failure."""
