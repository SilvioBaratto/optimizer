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

Example Script: create_superuser.py
-----------------------------------
```python
#!/usr/bin/env python3
\"\"\"Create a superuser via command line.\"\"\"

import asyncio
import argparse
import getpass

from app.database import database_manager
from app.services.user_service import UserService
from app.schemas.user import UserCreate


async def create_superuser(email: str, password: str):
    \"\"\"Create a new superuser.\"\"\"
    async with database_manager.get_session() as session:
        service = UserService(session)

        user_data = UserCreate(
            email=email,
            password=password,
            is_superuser=True,
            is_active=True
        )

        user = await service.create(user_data)
        print(f"Superuser created: {user.email}")


def main():
    parser = argparse.ArgumentParser(description="Create superuser")
    parser.add_argument("--email", required=True, help="User email")
    args = parser.parse_args()

    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")

    if password != confirm:
        print("Passwords do not match!")
        return

    asyncio.run(create_superuser(args.email, password))


if __name__ == "__main__":
    main()
```

Example Script: seed_database.py
--------------------------------
```python
#!/usr/bin/env python3
\"\"\"Seed database with initial data.\"\"\"

import asyncio
from app.database import database_manager
from app.models.user import User


async def seed():
    \"\"\"Populate database with seed data.\"\"\"
    async with database_manager.get_session() as session:
        # Add seed data here
        print("Database seeded successfully!")


if __name__ == "__main__":
    asyncio.run(seed())
```
"""
