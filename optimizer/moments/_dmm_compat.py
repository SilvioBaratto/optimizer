"""DMM availability guard.

Provides a runtime flag and a helper that raises a clear error when
torch/pyro-ppl are not installed.
"""

from __future__ import annotations

try:
    import pyro  # noqa: F401
    import torch  # noqa: F401

    HAS_DMM: bool = True
except ImportError:
    HAS_DMM = False


def require_dmm() -> None:
    """Raise ImportError with an install hint when DMM deps are absent."""
    if not HAS_DMM:
        raise ImportError(
            'DMM requires torch and pyro-ppl. Install with: pip install "portopt[dmm]"'
        )
