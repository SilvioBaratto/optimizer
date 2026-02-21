# View Integration

Frameworks for incorporating investor views into portfolio construction.

## Methods

- **Black-Litterman** -- Equilibrium-based view integration
- **Entropy Pooling** -- Non-parametric view blending (mean, variance, correlation, skew, kurtosis, CVaR)
- **Opinion Pooling** -- Expert estimator combination

## Omega Calibration

```python
from optimizer.views import calibrate_omega_from_track_record

omega = calibrate_omega_from_track_record(view_history, return_history)
```

Requires at least 5 aligned observations for reliable calibration.
