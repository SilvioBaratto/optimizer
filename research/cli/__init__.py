"""CLI entry point — re-exports parser and main."""

from research.cli._main import main
from research.cli._parser import build_parser

__all__ = ["build_parser", "main"]
