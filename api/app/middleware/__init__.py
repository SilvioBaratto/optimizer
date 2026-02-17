"""
Middleware Package
==================

This package contains custom ASGI/Starlette middleware for request processing.

File Structure
--------------
middleware/
├── __init__.py       # This file - exports middleware classes
├── logging.py        # Request/response logging middleware
├── security.py       # Security headers middleware
└── rate_limiting.py  # Rate limiting middleware

Creating Custom Middleware
--------------------------
FastAPI supports two middleware patterns:

1. **ASGI Middleware** (recommended for low-level control):
```python
# middleware/custom.py
from starlette.types import ASGIApp, Scope, Receive, Send

class CustomMiddleware:
    \"\"\"Custom ASGI middleware example.\"\"\"

    def __init__(self, app: ASGIApp, some_config: str = "default"):
        self.app = app
        self.some_config = some_config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only process HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Pre-processing (before request)
        # ... your logic here ...

        # Process the request
        await self.app(scope, receive, send)

        # Post-processing (after response)
        # ... your logic here ...
```

2. **BaseHTTPMiddleware** (simpler but with limitations):
```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class SimpleMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Pre-processing
        # ... your logic ...

        response = await call_next(request)

        # Post-processing
        response.headers["X-Custom-Header"] = "value"
        return response
```

Registering Middleware
----------------------
In main.py or your application factory:
```python
from app.middleware.logging import LoggingMiddleware
from app.middleware.security import SecurityHeadersMiddleware
from app.middleware.rate_limiting import RateLimitingMiddleware

# Order matters! Last added = first executed
app.add_middleware(LoggingMiddleware)        # Executes 3rd
app.add_middleware(SecurityHeadersMiddleware) # Executes 2nd
app.add_middleware(RateLimitingMiddleware)   # Executes 1st
```

Best Practices
--------------
1. **Keep Middleware Focused**: Each middleware should do one thing well
2. **Consider Performance**: Middleware runs on every request - keep it fast
3. **Handle Errors Gracefully**: Don't let middleware errors crash requests
4. **Skip Non-HTTP**: Always check scope["type"] for ASGI middleware
5. **Order Matters**: Add middleware in reverse order of desired execution
6. **Use Typing**: Import types from starlette.types for better IDE support
7. **Avoid BaseHTTPMiddleware for Streaming**: It buffers responses fully
"""

from app.middleware.logging import LoggingMiddleware
from app.middleware.security import SecurityHeadersMiddleware
from app.middleware.rate_limiting import RateLimitingMiddleware

__all__ = [
    "LoggingMiddleware",
    "SecurityHeadersMiddleware",
    "RateLimitingMiddleware",
]
