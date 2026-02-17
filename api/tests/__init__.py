"""
Test Suite
==========

This package contains all tests for the FastAPI application.

Test Structure
--------------
tests/
    __init__.py           # This file
    conftest.py           # Shared fixtures (database, client, auth)
    unit/                 # Unit tests (isolated, fast)
        __init__.py
        test_services.py
        test_repositories.py
        test_utils.py
    integration/          # Integration tests (database, external APIs)
        __init__.py
        test_api_endpoints.py
        test_database.py
    e2e/                  # End-to-end tests (full workflows)
        __init__.py
        test_user_flows.py

Running Tests
-------------
    # Run all tests
    pytest

    # Run with coverage
    pytest --cov=app --cov-report=html

    # Run specific test file
    pytest tests/unit/test_services.py

    # Run tests matching pattern
    pytest -k "test_user"

    # Run with verbose output
    pytest -v

    # Run parallel (requires pytest-xdist)
    pytest -n auto
"""
