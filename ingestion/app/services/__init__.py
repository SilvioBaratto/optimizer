"""Services package.

Business logic for the ingestion daemon. Each service fetches from one external
source and writes to Postgres through a repository:

- ``market_data`` — yfinance prices, fundamentals, holders, news; FX, calendars, market structure/summary, options
- ``macro``       — FRED, Il Sole 24 Ore, Trading Economics scrapers; macro news scraping
- ``universe``    — yfinance-Screener instrument universe build + optional Trading 212 annotation
- ``jobs``        — APScheduler wiring + ``BackgroundJobService`` lifecycle
- ``infrastructure`` — circuit breaker, rate limiter, retry, TTL cache
- ``_shared``     — progress callbacks, price fetch, sector resolution, trading calendar, webhooks
"""
