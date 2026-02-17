"""Portfolio optimization library built on skfolio and scikit-learn.

Modules
-------
preprocessing
    Custom sklearn-compatible transformers for return data cleaning.
pre_selection
    Pipeline assembly and asset pre-selection.
moments
    Moment estimation and prior construction.
views
    View integration frameworks (Black-Litterman, Entropy Pooling,
    Opinion Pooling).
optimization
    Portfolio optimization models (Mean-Risk, Risk Budgeting,
    Maximum Diversification, HRP, HERC, NCO, Benchmark Tracking,
    naive baselines, and ensemble stacking).
synthetic
    Synthetic data generation, vine copula models, and conditional
    stress testing.
validation
    Model selection and cross-validation (Walk-Forward, Combinatorial
    Purged CV, Multiple Randomized CV).
scoring
    Performance scoring for model selection and hyperparameter tuning.
tuning
    Hyperparameter tuning with temporal cross-validation
    (GridSearchCV, RandomizedSearchCV).
rebalancing
    Rebalancing frameworks (calendar-based, threshold-based,
    turnover computation, transaction cost estimation).
"""

__version__ = "0.1.0"
