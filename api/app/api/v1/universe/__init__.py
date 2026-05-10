"""Universe Routes."""

from app.api.v1.universe.trading212 import router as trading212_router
from app.api.v1.universe.universe_screen import router as universe_screen_router

__all__ = ["trading212_router", "universe_screen_router"]
