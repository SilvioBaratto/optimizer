-- Deterministic synthetic price fixture for the GitHub Actions smoke workflow.
--
-- Loaded into the optimizer_db after `alembic upgrade head` and before the
-- workflow invokes POST /api/v1/optimize and POST /api/v1/backtest, so those
-- endpoints have enough price data to compute returns, covariance, and
-- equity curves on a fresh CI database.
--
-- Shape:
--   - 1 synthetic exchange:       SMOKE (not a real venue)
--   - 5 synthetic instruments:    SM1…SM5 (yfinance_ticker = SM1…SM5)
--   - ~260 business-day rows/ticker covering 2023-01-02 → 2023-12-29
--
-- Each ticker uses a different sinusoidal phase + linear trend + amplitude to
-- keep pairwise correlations below the pre-selection DropCorrelated threshold
-- (default 0.95) so the optimizer receives a well-conditioned covariance.
--
-- All inserts are guarded by ON CONFLICT DO NOTHING so the fixture is safe to
-- re-run on the same container (e.g. a retried CI job).

BEGIN;

-- ---------------------------------------------------------------------------
-- Exchange
-- ---------------------------------------------------------------------------
INSERT INTO exchanges (id, name, created_at, updated_at)
VALUES (
    '11111111-1111-1111-1111-111111111111',
    'SMOKE',
    now(),
    now()
)
ON CONFLICT (name) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Instruments (SM1…SM5)
-- ---------------------------------------------------------------------------
INSERT INTO instruments (
    id, ticker, short_name, name, isin, instrument_type,
    currency_code, yfinance_ticker, exchange_id, created_at, updated_at
)
SELECT
    md5(t.tk)::uuid,
    t.tk,
    t.tk,
    'Synthetic ' || t.tk,
    'US0000000SM' || substring(t.tk, 3, 1),
    'STOCK',
    'USD',
    t.tk,
    '11111111-1111-1111-1111-111111111111',
    now(),
    now()
FROM (VALUES ('SM1'), ('SM2'), ('SM3'), ('SM4'), ('SM5')) AS t(tk)
ON CONFLICT ON CONSTRAINT uq_instrument_ticker_exchange DO NOTHING;

-- ---------------------------------------------------------------------------
-- Price history (daily, weekdays only, one business year)
-- ---------------------------------------------------------------------------
INSERT INTO price_history (
    id, instrument_id, date, close, created_at, updated_at
)
SELECT
    gen_random_uuid(),
    i.id,
    d.dt,
    ROUND((
        100.0
        + t.amplitude * sin(extract(doy from d.dt) / 20.0 + t.phase)
        + extract(doy from d.dt) * t.trend
    )::numeric, 6),
    now(),
    now()
FROM generate_series(
    '2023-01-02'::date,
    '2023-12-29'::date,
    '1 day'::interval
) AS d(dt)
CROSS JOIN (VALUES
    ('SM1', 10.0, 0.0,  0.05),
    ('SM2',  7.5, 1.0,  0.03),
    ('SM3', 12.0, 2.1,  0.07),
    ('SM4',  9.0, 3.2, -0.02),
    ('SM5', 11.0, 4.4,  0.04)
) AS t(tk, amplitude, phase, trend)
JOIN instruments i
    ON i.yfinance_ticker = t.tk
WHERE extract(isodow FROM d.dt) < 6  -- Mon–Fri only
ON CONFLICT ON CONSTRAINT uq_price_history_instrument_date DO NOTHING;

COMMIT;
