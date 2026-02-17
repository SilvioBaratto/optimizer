"""Centralized exception handling for the API"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
from typing import Optional, Any, Dict, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class BaseAPIException(Exception):
    """Base exception class for API errors"""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(BaseAPIException):
    """Raised when business logic validation fails"""

    def __init__(self, message: str, errors: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details={"errors": errors or {}},
        )


class AuthenticationError(BaseAPIException):
    """Raised when authentication fails"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_ERROR",
        )


class AuthorizationError(BaseAPIException):
    """Raised when user lacks permission"""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="AUTHORIZATION_ERROR",
        )


class NotFoundError(BaseAPIException):
    """Raised when a resource is not found"""

    def __init__(self, resource: str, identifier: Optional[Any] = None):
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} with id '{identifier}' not found"
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
        )


class ConflictError(BaseAPIException):
    """Raised when there's a conflict with existing data"""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT_ERROR",
        )


class RateLimitError(BaseAPIException):
    """Raised when rate limit is exceeded"""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_ERROR",
            details={"retry_after": retry_after} if retry_after else {},
        )


class ExternalServiceError(BaseAPIException):
    """Raised when an external service fails"""

    def __init__(self, service: str, message: str):
        super().__init__(
            message=f"{service}: {message}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service},
        )


class DatabaseError(BaseAPIException):
    """Raised when database operations fail"""

    def __init__(self, message: str = "Database operation failed"):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="DATABASE_ERROR",
        )


class DatabaseTimeoutError(DatabaseError):
    """Raised when database operations timeout"""

    def __init__(
        self,
        message: str = "Database operation timed out",
        timeout_duration: Optional[float] = None,
    ):
        super().__init__(message)
        self.error_code = "DATABASE_TIMEOUT_ERROR"
        if timeout_duration:
            self.details = {"timeout_duration": timeout_duration}


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""

    def __init__(
        self,
        message: str = "Database connection failed",
        connection_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_code = "DATABASE_CONNECTION_ERROR"
        if connection_info:
            self.details = {"connection_info": connection_info}


# Enhanced Authentication Exceptions


class TokenValidationError(AuthenticationError):
    """Raised when JWT token validation fails"""

    def __init__(
        self,
        message: str = "Token validation failed",
        token_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.details = {"token_info": token_info or {}}


class TokenExpiredError(AuthenticationError):
    """Raised when JWT token has expired"""

    def __init__(
        self, message: str = "Token has expired", expired_at: Optional[str] = None
    ):
        super().__init__(message)
        if expired_at:
            self.details = {"expired_at": expired_at}


class InvalidTokenFormatError(AuthenticationError):
    """Raised when token format is invalid"""

    def __init__(self, message: str = "Invalid token format"):
        super().__init__(message)
        self.details = {"expected_format": "Bearer <JWT_TOKEN>"}


class UserInactiveError(AuthenticationError):
    """Raised when user account is inactive"""

    def __init__(self, user_id: Optional[str] = None):
        message = "User account is inactive or disabled"
        super().__init__(message)
        if user_id:
            self.details = {"user_id": user_id}


class InsufficientPermissionsError(AuthorizationError):
    """Raised when user lacks specific permissions"""

    def __init__(
        self,
        required_permissions: List[str],
        user_permissions: Optional[List[str]] = None,
    ):
        message = (
            f"Insufficient permissions. Required: {', '.join(required_permissions)}"
        )
        super().__init__(message)
        self.details = {
            "required_permissions": required_permissions,
            "user_permissions": user_permissions or [],
        }


class SecurityViolationError(BaseAPIException):
    """Raised when security policy is violated"""

    def __init__(self, violation_type: str, details: Optional[Dict[str, Any]] = None):
        message = f"Security policy violation: {violation_type}"
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="SECURITY_VIOLATION",
            details=details or {},
        )


class SuspiciousActivityError(SecurityViolationError):
    """Raised when suspicious activity is detected"""

    def __init__(self, activity_type: str, client_ip: Optional[str] = None):
        message = f"Suspicious activity detected: {activity_type}"
        details = {}
        if client_ip:
            details["client_ip"] = client_ip
        super().__init__(activity_type, details)


class AuthServiceUnavailableError(ExternalServiceError):
    """Raised when authentication service is unavailable"""

    def __init__(self, service_name: str = "Authentication Service"):
        super().__init__(service_name, "Service temporarily unavailable")


class CacheError(BaseAPIException):
    """Raised when cache operations fail"""

    def __init__(self, operation: str, message: Optional[str] = None):
        error_message = f"Cache {operation} failed"
        if message:
            error_message += f": {message}"

        super().__init__(
            message=error_message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="CACHE_ERROR",
            details={"operation": operation},
        )


def create_error_response(
    status_code: int,
    message: str,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> JSONResponse:
    """Create a standardized error response"""

    content: Dict[str, Any] = {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }

    if details:
        content["error"]["details"] = details

    if request_id:
        content["error"]["request_id"] = request_id

    return JSONResponse(status_code=status_code, content=content)


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup all exception handlers for the application"""

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        """Handle business logic validation errors"""

        logger.warning(
            f"Validation error: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "details": exc.details,
                "path": request.url.path,
                "method": request.method,
                "validation_error": True,  # Flag for validation monitoring
            },
        )

        return create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None),
        )

    @app.exception_handler(SecurityViolationError)
    async def security_violation_handler(request: Request, exc: SecurityViolationError):
        """Handle security violation exceptions with enhanced logging"""

        # Log security events with high priority
        logger.error(
            f"SECURITY VIOLATION: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "details": exc.details,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
                "method": request.method,
                "security_event": True,  # Flag for security monitoring
            },
        )

        # Add security-specific headers
        response = create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None),
        )

        # Add security headers
        response.headers["X-Security-Event"] = "violation_detected"
        response.headers["X-Request-Blocked"] = "true"

        return response

    @app.exception_handler(AuthenticationError)
    async def authentication_error_handler(request: Request, exc: AuthenticationError):
        """Handle authentication errors with audit logging"""

        # Extract client information for audit trail
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        logger.warning(
            f"Authentication failed: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "path": request.url.path,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "method": request.method,
                "auth_failure": True,  # Flag for auth monitoring
            },
        )

        # Create response with auth-specific headers
        response = create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=getattr(exc, "details", {}),
            request_id=getattr(request.state, "request_id", None),
        )

        # Add authentication-specific headers
        response.headers["WWW-Authenticate"] = "Bearer"
        response.headers["X-Auth-Error"] = "true"

        return response

    @app.exception_handler(AuthorizationError)
    async def authorization_error_handler(request: Request, exc: AuthorizationError):
        """Handle authorization errors with permission tracking"""

        client_ip = request.client.host if request.client else "unknown"

        logger.warning(
            f"Authorization failed: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "path": request.url.path,
                "client_ip": client_ip,
                "method": request.method,
                "authorization_failure": True,  # Flag for permission monitoring
            },
        )

        response = create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=getattr(exc, "details", {}),
            request_id=getattr(request.state, "request_id", None),
        )

        response.headers["X-Permission-Required"] = "true"

        return response

    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(request: Request, exc: RateLimitError):
        """Handle rate limit exceptions with retry information"""

        logger.info(
            f"Rate limit exceeded: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "rate_limit_exceeded": True,
            },
        )

        response = create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None),
        )

        # Add rate limiting headers
        retry_after = exc.details.get("retry_after", 60)
        response.headers["Retry-After"] = str(retry_after)
        response.headers["X-RateLimit-Exceeded"] = "true"

        return response

    @app.exception_handler(DatabaseTimeoutError)
    async def database_timeout_error_handler(
        request: Request, exc: DatabaseTimeoutError
    ):
        """Handle database timeout errors with specific monitoring"""

        logger.error(
            f"Database timeout error: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "path": request.url.path,
                "timeout_duration": exc.details.get("timeout_duration"),
                "database_timeout_error": True,  # Flag for database monitoring
            },
        )

        response = create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None),
        )

        response.headers["X-Database-Error"] = "timeout"
        response.headers["Retry-After"] = "5"  # Suggest retry after 5 seconds

        return response

    @app.exception_handler(DatabaseConnectionError)
    async def database_connection_error_handler(
        request: Request, exc: DatabaseConnectionError
    ):
        """Handle database connection errors with circuit breaker info"""

        logger.error(
            f"Database connection error: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "path": request.url.path,
                "connection_info": exc.details.get("connection_info"),
                "database_connection_error": True,  # Flag for database monitoring
            },
        )

        response = create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None),
        )

        response.headers["X-Database-Error"] = "connection_failed"
        response.headers["Retry-After"] = "10"  # Suggest retry after 10 seconds

        return response

    @app.exception_handler(ExternalServiceError)
    async def external_service_error_handler(
        request: Request, exc: ExternalServiceError
    ):
        """Handle external service errors with service monitoring"""

        logger.error(
            f"External service error: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "path": request.url.path,
                "service": exc.details.get("service", "unknown"),
                "external_service_error": True,  # Flag for service monitoring
            },
        )

        response = create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None),
        )

        response.headers["X-Service-Error"] = exc.details.get("service", "unknown")

        return response

    @app.exception_handler(DatabaseError)
    async def database_error_handler(request: Request, exc: DatabaseError):
        """Handle generic database errors"""

        logger.error(
            f"Database error: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
                "database_error": True,  # Flag for database monitoring
            },
        )

        return create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None),
        )

    @app.exception_handler(BaseAPIException)
    async def api_exception_handler(request: Request, exc: BaseAPIException):
        """Handle custom API exceptions"""

        # Log the error
        logger.error(
            f"API Exception: {exc.error_code} - {exc.message}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "details": exc.details,
                "path": request.url.path,
            },
        )

        return create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle Pydantic validation errors"""

        # Format validation errors
        errors = {}
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"][1:])
            if field not in errors:
                errors[field] = []
            errors[field].append(error["msg"])

        logger.warning(
            f"Validation error on {request.url.path}", extra={"errors": errors}
        )

        return create_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Validation failed",
            error_code="VALIDATION_ERROR",
            details={"validation_errors": errors},
            request_id=getattr(request.state, "request_id", None),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions"""

        logger.warning(
            f"HTTP Exception: {exc.status_code} - {exc.detail}",
            extra={"path": request.url.path},
        )

        return create_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            error_code=f"HTTP_{exc.status_code}",
            request_id=getattr(request.state, "request_id", None),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""

        logger.exception(f"Unexpected error on {request.url.path}", exc_info=exc)

        # Don't expose internal errors in production
        message = "An unexpected error occurred"
        if app.debug:
            message = str(exc)

        return create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            error_code="INTERNAL_SERVER_ERROR",
            request_id=getattr(request.state, "request_id", None),
        )