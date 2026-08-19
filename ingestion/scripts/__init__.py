"""
Scripts Package
===============

This directory contains utility scripts for development and operations.

Common Scripts
--------------
scripts/
    __init__.py           # This file
    seed_database.py      # Populate database with initial/test data
    create_superuser.py   # Create admin user via CLI
    run_migrations.py     # Run database migrations programmatically
    healthcheck.py        # Health check script for Docker/K8s
    cleanup.py            # Database/cache cleanup utilities

Usage Examples
--------------

Create Superuser:
    python -m scripts.create_superuser --email admin@example.com

Seed Database:
    python -m scripts.seed_database --env development

Run Health Check:
    python -m scripts.healthcheck

Example Script: sync_portfolio.py
-----------------------------------
```python
#!/usr/bin/env python3
\"\"\"Sync portfolio from broker via command line.\"\"\"

import argparse
from app.services.infrastructure.database_manager import DatabaseManager
from app.services.portfolio.broker_sync_service import sync_portfolio
from app.repositories.portfolio.portfolio_repository import PortfolioRepository


def main():
    parser = argparse.ArgumentParser(description="Sync portfolio from broker")
    parser.add_argument("--portfolio-id", required=True, help="Portfolio UUID")
    args = parser.parse_args()

    db = DatabaseManager()
    with db.get_session() as session:
        result = sync_portfolio(session, uuid=args.portfolio_id)
        if result.success:
            print(f"Portfolio synced: {result.holdings_count} holdings")
        else:
            print(f"Sync failed: {result.error}")


if __name__ == "__main__":
    main()
```

Example Script: seed_database.py
--------------------------------
```python
#!/usr/bin/env python3
\"\"\"Seed database with initial data.\"\"\"

from app.services.infrastructure.database_manager import DatabaseManager
from app.models.portfolio.portfolio import Portfolio


def seed():
    \"\"\"Populate database with seed data.\"\"\"
    db = DatabaseManager()
    with db.get_session() as session:
        # Add seed data here
        print("Database seeded successfully!")


if __name__ == "__main__":
    seed()
```
"""
