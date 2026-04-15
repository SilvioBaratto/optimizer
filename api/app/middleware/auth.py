"""API-key authentication middleware."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.models.api_key import ApiKey


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests missing a valid X-API-Key header.

    Public paths and OPTIONS requests always pass through.
    Accepts a session_factory callable (context manager) for DB access,
    following the same pattern as BackgroundJobService.
    """

    PUBLIC_PATHS: frozenset[str] = frozenset(
        {"/", "/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
    )

    def __init__(self, app: Any, session_factory: Callable) -> None:
        super().__init__(app)
        self._session_factory = session_factory

    async def dispatch(self, request: Request, call_next) -> Response:
        if self._is_exempt(request):
            return await call_next(request)

        api_key_header = request.headers.get("X-API-Key")
        if not api_key_header or not self._is_valid(api_key_header):
            return self._unauthorized()

        return await call_next(request)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_exempt(self, request: Request) -> bool:
        return request.method == "OPTIONS" or request.url.path in self.PUBLIC_PATHS

    def _is_valid(self, plaintext: str) -> bool:
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
        with self._session_factory() as session:
            row = (
                session.query(ApiKey)
                .filter(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
                .first()
            )
        if row is None:
            return False
        return hmac.compare_digest(row.key_hash, key_hash)

    @staticmethod
    def _unauthorized() -> Response:
        body = json.dumps({"detail": "Invalid or missing API key"})
        return Response(content=body, status_code=401, media_type="application/json")
