"""Security middleware for adding security headers"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any
from app.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""
    
    @staticmethod
    def get_security_headers(request: Request) -> Dict[str, str]:
        """Get security headers dictionary based on request path and environment"""
        
        # Check if this is a documentation endpoint
        is_docs_endpoint = request.url.path in ["/docs", "/redoc", "/openapi.json"]
        
        # Base security headers
        headers = {
            # XSS Protection
            "X-XSS-Protection": "1; mode=block",
            
            # Content Type Options
            "X-Content-Type-Options": "nosniff",
            
            # Frame Options
            "X-Frame-Options": "DENY",
            
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # HSTS (HTTPS only)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            
            # Permissions Policy (disable unnecessary features)
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "fullscreen=()"
            ),
            
            # Custom security headers
            "X-API-Version": "1.0.0",
            "X-Powered-By": "FastAPI",
        }
        
        # Content Security Policy based on endpoint and environment
        if is_docs_endpoint and settings.debug:
            # Allow external resources for Swagger UI in development
            headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com; "
                "img-src 'self' data: https://fastapi.tiangolo.com; "
                "font-src 'self' https://cdn.jsdelivr.net https://unpkg.com; "
                "connect-src 'self'; "
                "frame-ancestors 'none'"
            )
        else:
            # Restrictive CSP for API endpoints
            headers["Content-Security-Policy"] = (
                "default-src 'none'; "
                "frame-ancestors 'none'; "
                "upgrade-insecure-requests"
            )
        
        return headers
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers to all responses"""
        response = await call_next(request)
        
        # Add all security headers based on request context
        security_headers = self.get_security_headers(request)
        for header_name, header_value in security_headers.items():
            response.headers[header_name] = header_value
        
        return response