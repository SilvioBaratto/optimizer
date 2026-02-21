# Validation

Cross-validation strategies for portfolio model selection.

## Methods

- **WalkForwardConfig** -- Rolling or expanding window temporal CV
- **CPCVConfig** -- Combinatorial Purged Cross-Validation
- **MultipleRandomizedCVConfig** -- Multiple Randomized CV

## Usage

```python
from optimizer.validation import WalkForwardConfig, run_cross_val

cv_config = WalkForwardConfig.for_quarterly_rolling()
cv_result = run_cross_val(returns, optimizer, cv_config)
```

`run_cross_val()` defaults to quarterly rolling walk-forward when no `cv` is passed.
