"""Guard: importing ``fund.agents.model`` pulls in NO provider SDK.

The provider registry (``fund.agents.model._BUILDERS``) dispatches to per-provider
builders whose LangChain SDK imports live **inside** the builder body, so a bare
``import fund.agents.model`` must not drag any ``langchain_*`` provider package
into ``sys.modules``. This is load-bearing: ``worker.py`` / ``scheduler.py`` import
the model module while staying agent-stack-free, and CI keeps every provider key
unset — a top-level provider import would break both.

The check runs in a **fresh subprocess** on purpose: the in-process pytest session
has already imported ``langchain_ollama`` via ``test_model_factory``, so an
in-process ``sys.modules`` assertion would be meaningless. The subprocess starts
clean and only imports ``fund.agents.model``.
"""

from __future__ import annotations

import subprocess
import sys

# Every provider SDK the registry may dispatch to. Each builder imports its own
# package lazily; none may appear in `sys.modules` after a bare module import.
_PROVIDER_MODULES = (
    "langchain_ollama",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_groq",
    "langchain_google_genai",
    "langchain_aws",
    "langchain_nvidia_ai_endpoints",
    "langchain_huggingface",
)

_PROBE = (
    "import sys\n"
    "import fund.agents.model\n"  # the only import under test
    "leaked = [m for m in {mods!r} if m in sys.modules]\n"
    "print(','.join(leaked))\n"
)


def _leaked_provider_modules() -> list[str]:
    """Import ``fund.agents.model`` in a clean subprocess; return leaked SDKs.

    Returns:
        The provider package names present in the child's ``sys.modules`` after
        the import — empty when the lazy-import contract holds.
    """
    result = subprocess.run(  # noqa: S603 — fixed argv, our own interpreter
        [sys.executable, "-c", _PROBE.format(mods=_PROVIDER_MODULES)],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = result.stdout.strip()
    return leaked.split(",") if leaked else []


def test_importing_model_module_pulls_in_no_provider_sdk():
    assert _leaked_provider_modules() == [], (
        "importing fund.agents.model leaked a provider SDK — a builder's "
        "`from langchain_* import ...` must stay inside the function body"
    )


def test_probe_would_detect_a_leaked_provider():
    """Fail-injection: a probe that force-imports a provider must report it,
    proving the guard actually observes ``sys.modules`` (not a no-op green)."""
    probe = (
        "import sys\n"
        "import langchain_ollama\n"  # deliberate eager import
        f"leaked = [m for m in {_PROVIDER_MODULES!r} if m in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(  # noqa: S603 — fixed argv, our own interpreter
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "langchain_ollama" in result.stdout
