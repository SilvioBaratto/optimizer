"""Which Ollama base URLs count as the cloud, and where the cloud key may go.

The key authenticates to Ollama Cloud. Compose mounts it into the fund container
whatever the base URL is, so a local or LAN server must not receive it unless the
operator opts in; a hostname that merely contains ``ollama.com`` is not the cloud.
"""

from __future__ import annotations

import pytest

from fund.agents import model
from fund.config import FundConfig, load_config

_KEY = "sk-ollama-cloud"


def _llm(url: str, *, key: str | None = _KEY, **extra: object):
    cfg = FundConfig(ollama_api_key=key, ollama_base_url=url, **extra)  # type: ignore[arg-type]
    return model.build_chat_model(cfg, model_name="m")


def _auth(llm: object) -> object:
    return (llm.client_kwargs or {}).get("auth")  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "url",
    [
        "http://ollama.company.lan:11434",
        "http://ollama.compute.internal:11434",
        "https://myollama.community",
        "https://ollama.com.attacker.example",
        "https://not-ollama.com",
    ],
)
def test_a_host_that_only_contains_the_cloud_name_is_local(url):
    llm = _llm(url, key=None)

    assert _auth(llm) is None
    assert llm.num_ctx == FundConfig().ollama_num_ctx


@pytest.mark.parametrize(
    "url",
    ["https://ollama.com", "https://ollama.com:443", "https://api.ollama.com"],
)
def test_the_cloud_hosts_require_and_receive_the_key(url):
    with pytest.raises(RuntimeError, match="OLLAMA_API_KEY"):
        _llm(url, key=None)

    llm = _llm(url)

    assert _auth(llm) is not None
    assert llm.num_ctx is None


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://host.docker.internal:11434",
        "http://192.168.1.50:11434",
        "https://ollama.company.lan",
    ],
)
def test_the_cloud_key_is_not_sent_to_a_non_cloud_host(url):
    assert _auth(_llm(url)) is None


def test_the_key_goes_to_a_non_cloud_host_only_on_opt_in():
    llm = _llm("http://gateway.lan:11434", ollama_base_url_send_key=True)

    assert _auth(llm) is not None


def test_opting_in_without_a_key_names_the_missing_variable():
    with pytest.raises(RuntimeError, match="OLLAMA_API_KEY"):
        _llm("http://gateway.lan:11434", key=None, ollama_base_url_send_key=True)


def test_the_cloud_host_over_plain_http_is_refused_rather_than_sent_the_key():
    with pytest.raises(RuntimeError, match="https"):
        _llm("http://ollama.com")


def test_credentials_embedded_in_the_base_url_are_refused():
    with pytest.raises(RuntimeError, match="OLLAMA_BASE_URL"):
        _llm("http://user:secret@localhost:11434", key=None)


@pytest.mark.parametrize(
    "url", ["http://localhost:11434/v1", "http://localhost:11434/v1/"]
)
def test_a_trailing_v1_is_stripped_because_chatollama_appends_the_api_path(url):
    # ChatOllama posts to <base>/api/chat; with /v1 it answers 404.
    assert _llm(url, key=None).base_url == "http://localhost:11434"


def test_the_send_key_opt_in_is_read_from_the_environment():
    assert load_config(env={}).ollama_base_url_send_key is False
    cfg = load_config(env={"OLLAMA_BASE_URL_SEND_KEY": "true"})
    assert cfg.ollama_base_url_send_key is True
