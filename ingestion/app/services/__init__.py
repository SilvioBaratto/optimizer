"""Services package.

Business logic for the ingestion daemon. Each service fetches from one external
source and writes to Postgres through a repository:

- ``market_data`` — yfinance prices, fundamentals, holders, news; reference-index seeding
- ``macro``       — FRED, Il Sole 24 Ore, Trading Economics scrapers; LLM news summary + regime calibration
- ``universe``    — Trading 212 instrument universe build
- ``jobs``        — APScheduler wiring + ``BackgroundJobService`` lifecycle
- ``infrastructure`` — circuit breaker, rate limiter, retry, TTL cache
- ``_shared``     — progress callbacks, price fetch, sector resolution, trading calendar, webhooks
"""
