"""What makes the fund's LLM backend usable, from what the wizard persisted.

The fund worker builds its chat model before its first heartbeat, so a container
started without a usable backend crashes in a restart loop and never turns healthy.
``portopt start`` therefore asks :func:`is_configured` before it brings the fund
profile up. The answer is derived from the same two stores the wizard writes: the
non-secret choices in ``config.toml`` and the decrypted secret store. It mirrors the
checks ``fund.agents.model`` applies at build time without importing them
(ingestion ⊬ agent stack).
"""

from __future__ import annotations

from collections.abc import Mapping

# Provider slug -> compose secret name for its auth key (matches
# compose_secrets.SECRET_NAMES + fund.config._read_secret). Providers absent here
# are keyless: `aws` authenticates via the boto3 chain (IAM/region). The env var a
# key may also arrive in is the secret name upper-cased (e.g. OPENAI_API_KEY).
PROVIDER_SECRET_NAMES = {
    "ollama": "ollama_api_key",
    "openai": "openai_api_key",
    "openrouter": "openrouter_api_key",
    "anthropic": "anthropic_api_key",
    "google": "google_api_key",
    "groq": "groq_api_key",
    "nvidia": "nvidia_api_key",
    "huggingface": "huggingfacehub_api_token",
    "microsoft": "azure_openai_api_key",
}

# What fund.config.load_config selects when LLM_PROVIDER is not set. It is also the
# only provider whose built-in model id is valid, so it alone needs no explicit one.
DEFAULT_PROVIDER = "ollama"

# Any other ollama base_url is a local/self-hosted daemon, which is keyless.
_OLLAMA_CLOUD_HOST = "ollama.com"


def needs_key(provider: str, base_url: str | None) -> bool:
    """Return whether ``provider`` requires an auth key given ``base_url``.

    ``aws`` never does (boto3 chain); ``ollama`` only on the cloud host; ``nvidia``
    only when hosted (a self-hosted NIM ``base_url`` is keyless); all others do.
    """
    if provider == "aws":
        return False
    if provider == "ollama":
        return not base_url or _OLLAMA_CLOUD_HOST in base_url
    if provider == "nvidia":
        return base_url is None
    return True


def is_configured(config: Mapping[str, object], secrets: Mapping[str, str]) -> bool:
    """Return whether the fund worker can build its model from this setup.

    Args:
        config: The non-secret settings from ``config.toml``. An absent
            ``llm_provider`` means the fund's default, Ollama Cloud.
        secrets: The decrypted secret store, by secret name.

    Returns:
        True when the selected provider is known, has an explicit model id (unless
        it is the default provider, which has a built-in one), and either runs
        without a key (a local Ollama daemon, a self-hosted NIM, Bedrock) or has
        its key stored and non-blank. False for anything the worker would refuse
        at build time.
    """
    provider = str(config.get("llm_provider") or DEFAULT_PROVIDER)
    if provider != DEFAULT_PROVIDER and not str(config.get("llm_model") or "").strip():
        return False
    base_url = str(config.get("llm_base_url") or "") or None
    if not needs_key(provider, base_url):
        return True
    secret_name = PROVIDER_SECRET_NAMES.get(provider)
    return bool(secret_name and secrets.get(secret_name, "").strip())
