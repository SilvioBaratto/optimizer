"""Rate limiting middleware for API protection"""

import time
import logging
from typing import Dict, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware for basic protection"""
    
    def __init__(self, app, requests: int = 100, window: int = 60):
        super().__init__(app)
        self.requests = requests
        self.window = window
        self.client_requests: Dict[str, List[float]] = {}
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting in development
        if settings.is_development:
            return await call_next(request)
        
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/metrics"]:
            return await call_next(request)
        
        current_time = time.time()
        
        # Clean old entries for this client
        if client_ip in self.client_requests:
            self.client_requests[client_ip] = [
                timestamp for timestamp in self.client_requests[client_ip]
                if timestamp > current_time - self.window
            ]
        else:
            self.client_requests[client_ip] = []
        
        # Check rate limit
        request_count = len(self.client_requests[client_ip])
        
        if request_count >= self.requests:
            logger.warning(
                f"Rate limit exceeded for client {client_ip}: {request_count} requests in {self.window}s"
            )
            
            # Calculate retry after time
            oldest_request = min(self.client_requests[client_ip])
            retry_after = int(oldest_request + self.window - current_time) + 1
            
            return Response(
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Maximum {self.requests} requests per {self.window} seconds.",
                        "retry_after": retry_after
                    }
                },
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(oldest_request + self.window))
                },
                media_type="application/json"
            )
        
        # Record this request
        self.client_requests[client_ip].append(current_time)
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers to response
        remaining = max(0, self.requests - len(self.client_requests[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window))
        
        return response
    
    def cleanup_old_entries(self) -> None:
        """Cleanup old entries from all clients to prevent memory leaks"""
        current_time = time.time()
        
        for client_ip in list(self.client_requests.keys()):
            # Filter out old requests
            self.client_requests[client_ip] = [
                timestamp for timestamp in self.client_requests[client_ip]
                if timestamp > current_time - self.window
            ]
            
            # Remove client if no recent requests
            if not self.client_requests[client_ip]:
                del self.client_requests[client_ip]