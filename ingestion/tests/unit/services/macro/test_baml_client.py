"""BAML runtime provider-routing contract (SPEC D8, task T5).

`llm_call_options()` carries a per-call `ClientRegistry` whose primary client is
built from `settings.llm_provider` (cloud only: openai/anthropic) with the
matching model + key. Tests monkeypatch `ClientRegistry` so no BAML runtime or
network is touched.
"""

from types import SimpleNamespace

import pytest

from app.services.macro import _baml_client as bc


def _cfg(**kw: object) -> SimpleNamespace:
    base = {
        "llm_provider": "openai",
        "openai_model": "gpt-4o-mini",
        "openai_api_key": "sk-o",
        "anthropic_model": "claude-x",
        "anthropic_api_key": "sk-a",
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_resolve_openai() -> None:
    provider, options = bc._resolve_client(_cfg(llm_provider="openai"))
    assert provider == "openai"
    assert options == {"model": "gpt-4o-mini", "api_key": "sk-o"}


def test_resolve_anthropic() -> None:
    provider, options = bc._resolve_client(_cfg(llm_provider="anthropic"))
    assert provider == "anthropic"
    assert options == {"model": "claude-x", "api_key": "sk-a"}


def test_resolve_rejects_non_cloud() -> None:
    with pytest.raises(ValueError, match="cloud"):
        bc._resolve_client(_cfg(llm_provider="ollama"))


class _FakeCR:
    def __init__(self) -> None:
        self.added: dict[str, object] = {}
        self.primary: str | None = None

    def add_llm_client(self, name: str, provider: str, options: dict) -> None:
        self.added = {"name": name, "provider": provider, "options": options}

    def set_primary(self, primary: str) -> None:
        self.primary = primary


def test_build_client_registry_sets_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc, "ClientRegistry", _FakeCR)
    cr = bc.build_client_registry(_cfg(llm_provider="anthropic"))
    assert isinstance(cr, _FakeCR)
    assert cr.added["provider"] == "anthropic"
    assert cr.primary == cr.added["name"]


def test_build_uses_global_settings_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bc, "ClientRegistry", _FakeCR)
    monkeypatch.setattr(bc, "settings", _cfg(llm_provider="openai"))
    cr = bc.build_client_registry()
    assert cr.added["provider"] == "openai"


def test_llm_call_options_carries_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bc, "ClientRegistry", _FakeCR)
    opts = bc.llm_call_options(_cfg())
    assert set(opts) == {"client_registry"}
    assert isinstance(opts["client_registry"], _FakeCR)
