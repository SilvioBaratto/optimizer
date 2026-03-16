"""Prometheus metrics exposition endpoint."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["Monitoring"])


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    include_in_schema=False,
    summary="Prometheus metrics scrape endpoint",
)
def prometheus_metrics() -> PlainTextResponse:
    """Expose all registered Prometheus metrics in text exposition format."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
