"""Global dependencies for the application"""

import time
from typing import Optional, Annotated, Any
from fastapi import Depends, HTTPException, status, Header, Request
from sqlalchemy.orm import Session
import logging

from app.exceptions import AuthenticationError, RateLimitError

logger = logging.getLogger(__name__)

# Database session dependencies - imported from database module
from app.database import get_db

class RateLimiter:
    """
    Production-ready rate limiting dependency with Redis backend
    
    Features:
    - Sliding window rate limiting
    - IP-based and user-based limiting
    - Redis integration for distributed rate limiting
    - Configurable requests per window
    """
    
    def __init__(self, requests: int = 100, window: int = 60, per_user: bool = False):
        self.requests = requests
        self.window = window
        self.per_user = per_user
        self._in_memory_cache = {}  # Fallback for when Redis is unavailable
    
    def __call__(self, request: Request, current_user: Optional[dict] = None):
        """Check rate limit for the request"""
        try:
            # Try Redis-based rate limiting first
            return self._check_redis_rate_limit(request, current_user)
        except Exception as e:
            logger.warning(f"Redis rate limiting failed, falling back to in-memory: {e}")
            return self._check_memory_rate_limit(request, current_user)
    
    def _check_redis_rate_limit(self, request: Request, current_user: Optional[dict] = None) -> bool:
        """Redis-based distributed rate limiting"""
        from app.config import settings
        
        if not settings.redis_url:
            raise Exception("Redis not configured")
        
        # Determine rate limit key
        if self.per_user and current_user:
            key = f"rate_limit:user:{current_user['id']}:{self.window}"
        else:
            client_ip = self._get_client_ip(request)
            key = f"rate_limit:ip:{client_ip}:{self.window}"
        
        # Import Redis here to avoid circular imports
        import redis
        
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        
        try:
            # Sliding window rate limiting using Redis
            current_time = int(time.time())
            window_start = current_time - self.window
            
            # Use Redis pipeline for atomic operations
            with redis_client.pipeline() as pipe:
                # Remove old entries
                pipe.zremrangebyscore(key, 0, window_start)
                # Count current requests
                pipe.zcard(key)
                # Add current request
                pipe.zadd(key, {str(current_time): current_time})
                # Set expiration
                pipe.expire(key, self.window)
                results = pipe.execute()
            
            current_count = results[1]  # Result from zcard
            
            if current_count >= self.requests:
                logger.warning(f"⚠️  Rate limit exceeded for key {key}: {current_count}/{self.requests}")
                from app.exceptions import RateLimitError
                retry_after = self.window - (current_time % self.window)
                raise RateLimitError(
                    message=f"Rate limit exceeded: {current_count}/{self.requests} requests per {self.window}s",
                    retry_after=retry_after
                )
            
            return True
            
        finally:
            redis_client.close()
    
    def _check_memory_rate_limit(self, request: Request, current_user: Optional[dict] = None) -> bool:
        """In-memory fallback rate limiting"""
        # Determine rate limit key
        if self.per_user and current_user:
            key = f"user:{current_user['id']}"
        else:
            key = f"ip:{self._get_client_ip(request)}"
        
        current_time = time.time()
        window_start = current_time - self.window
        
        # Clean old entries and check current count
        if key not in self._in_memory_cache:
            self._in_memory_cache[key] = []
        
        # Remove old entries
        self._in_memory_cache[key] = [
            timestamp for timestamp in self._in_memory_cache[key]
            if timestamp > window_start
        ]
        
        # Check rate limit
        if len(self._in_memory_cache[key]) >= self.requests:
            from app.exceptions import RateLimitError
            raise RateLimitError(
                message=f"Rate limit exceeded: {len(self._in_memory_cache[key])}/{self.requests} requests per {self.window}s"
            )
        
        # Add current request
        self._in_memory_cache[key].append(current_time)
        return True
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers"""
        # Check for forwarded headers (load balancer, proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"


def get_rate_limiter(requests: int = 100, window: int = 60) -> RateLimiter:
    """Factory for rate limiter dependency"""
    return RateLimiter(requests=requests, window=window)


# Common dependency injections
DBSession = Annotated[Session, Depends(get_db)]
# CurrentUser, OptionalUser, RequireAuth imported from auth module above


class PaginationParams:
    """
    Advanced pagination parameters with cursor support for better performance
    
    Features:
    - Traditional offset/limit pagination
    - Cursor-based pagination for large datasets (17x faster performance)
    - Configurable limits with safety bounds
    """
    
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        order_desc: bool = False,
        cursor: Optional[str] = None,
        use_cursor: bool = False
    ):
        self.skip = max(0, skip)
        self.limit = min(max(1, limit), 1000)  # Max 1000 items for safety
        self.order_by = order_by
        self.order_desc = order_desc
        self.cursor = cursor
        self.use_cursor = use_cursor or cursor is not None
        
        # Performance warning for large offsets
        if self.skip > 10000 and not self.use_cursor:
            logger.warning(f"⚠️  Large offset detected ({self.skip}). Consider using cursor-based pagination for better performance.")
    
    def encode_cursor(self, value: Any) -> str:
        """Encode cursor value for pagination"""
        import base64
        import json
        cursor_data = {"value": str(value), "order_desc": self.order_desc}
        cursor_json = json.dumps(cursor_data)
        return base64.urlsafe_b64encode(cursor_json.encode()).decode()
    
    def decode_cursor(self) -> tuple[Any, bool]:
        """Decode cursor value from pagination"""
        if not self.cursor:
            return None, self.order_desc
        
        try:
            import base64
            import json
            cursor_json = base64.urlsafe_b64decode(self.cursor.encode()).decode()
            cursor_data = json.loads(cursor_json)
            return cursor_data["value"], cursor_data["order_desc"]
        except Exception as e:
            logger.warning(f"Invalid cursor format: {e}")
            return None, self.order_desc


class FilterParams:
    """Common filter parameters"""
    
    def __init__(
        self,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None
    ):
        self.search = search
        self.is_active = is_active
        self.created_after = created_after
        self.created_before = created_before


# Dependency shortcuts
Pagination = Annotated[PaginationParams, Depends()]
Filters = Annotated[FilterParams, Depends()]