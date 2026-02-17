"""
API Package
===========

This package contains all API-related modules organized by version.

Directory Structure
-------------------
api/
├── __init__.py          # This file - package initialization
└── v1/                  # API version 1
    ├── __init__.py      # V1 package initialization
    ├── router.py        # Main router that includes all route modules
    └── <domain>.py      # Domain-specific route modules (e.g., users.py, items.py)

Best Practices
--------------
1. **API Versioning**: Always version your API (v1, v2, etc.) to allow
   backward-compatible changes and gradual migrations.

2. **Router Organization**: Keep the main router.py clean - it should only
   import and include routers from domain-specific modules.

3. **Domain Separation**: Create separate files for each domain/resource
   (users.py, items.py, auth.py, etc.) to maintain single responsibility.

4. **Consistent Naming**: Use plural nouns for resource endpoints
   (e.g., /users, /items, not /user, /item).

Example Usage
-------------
In main.py:
    from app.api.v1.router import api_router
    app.include_router(api_router, prefix="/api/v1")

Adding a new domain (e.g., users):
    1. Create api/v1/users.py with a router
    2. Import and include in api/v1/router.py
"""
