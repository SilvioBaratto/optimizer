"""Request logging middleware"""

import time
import logging
import uuid
import json
from typing import Any, Dict
from starlette.types import ASGIApp, Scope, Receive, Send, Message

logger = logging.getLogger(__name__)


class LoggingMiddleware:
    """Middleware to log all requests and responses for monitoring and debugging"""
    
    def __init__(self, app: ASGIApp):
        self.app = app
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        # Extract request information from scope
        method = scope.get("method", "")
        path = scope.get("path", "")
        query_string = scope.get("query_string", b"").decode()
        client_ip = scope.get("client", ["unknown", None])[0] if scope.get("client") else "unknown"
        headers_dict = dict(scope.get("headers", []))
        user_agent = headers_dict.get(b"user-agent", b"unknown").decode()
        
        # Log incoming request
        logger.info(
            f"Request {request_id}: {method} {path}",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "query": query_string if query_string else None,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "event_type": "request_start"
            }
        )
        
        # Capture response data
        response_body = b""
        response_status = 200
        response_headers = {}
        
        async def send_wrapper(message: Message) -> None:
            nonlocal response_body, response_status, response_headers
            
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = dict(message.get("headers", []))
                # Add request ID to response headers
                new_headers = list(message.get("headers", []))
                new_headers.append((b"x-request-id", request_id.encode()))
                new_headers.append((b"x-process-time", f"{time.time() - start_time:.3f}s".encode()))
                message["headers"] = new_headers
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
            
            await send(message)
        
        # Process request
        await self.app(scope, receive, send_wrapper)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Parse response JSON if available
        response_json = None
        
        if response_body:
            try:
                # Check if it's JSON content
                content_type = response_headers.get(b"content-type", b"").decode().lower()
                if "application/json" in content_type:
                    response_text = response_body.decode('utf-8')
                    response_json = json.loads(response_text)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.debug(f"Failed to parse response JSON for {request_id}: {e}")
        
        # Prepare log extra data
        log_extra = {
            "request_id": request_id,
            "status_code": response_status,
            "process_time": process_time,
            "event_type": "request_complete"
        }
        
        # Add response JSON if available
        if response_json is not None:
            log_extra["response_json"] = response_json
            
        # Log response with JSON if available
        if response_json is not None:
            try:
                # Format JSON beautifully with indentation - never truncate
                formatted_json = json.dumps(response_json, indent=2, ensure_ascii=False, separators=(',', ': '))
                
                logger.info(
                    f"Response {request_id}: {response_status} in {process_time:.3f}s - JSON:\n{formatted_json}",
                    extra=log_extra
                )
            except (TypeError, ValueError) as e:
                logger.info(
                    f"Response {request_id}: {response_status} in {process_time:.3f}s - JSON: [serialization error: {e}]",
                    extra=log_extra
                )
        else:
            logger.info(
                f"Response {request_id}: {response_status} in {process_time:.3f}s",
                extra=log_extra
            )