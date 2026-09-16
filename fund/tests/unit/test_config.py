"""T2.1 — ``fund.config`` pins the SPEC §8 Phase-0 contract.

The config is a frozen dataclass (mirrors ``portopt_db.config.DbConfig``) plus a
``load_config`` factory that sources secrets from the environment. Pinned
constants (LangGraph schema, model ids, pool kwargs, HITL, recursion guard) are
dataclass defaults so a bare import yields the verified Phase-0 values; secrets
(``DATABASE_URL``/``OLLAMA_API_KEY``/``FRED_API_KEY``/``SSL_CERT_FILE``) come from
``env`` and stay ``None`` when absent so import never fails in CI.
"""

from __future__ import annotations

from pathlib import Path

from fund.config import FundConfig, load_config, settings


def test_defaults_pin_the_spec8_contract():
    cfg = load_config(env={})

    # D2 agent backend
    assert cfg.agent_virtual_mode is True
    # D3 LangGraph schema + pool
    assert cfg.langgraph_schema == "langgraph"
    assert cfg.pool_autocommit is True
    assert cfg.pool_prepare_threshold == 0
    # D4 model
    assert cfg.ollama_base_url == "https://ollama.com"
    assert cfg.primary_model == "deepseek-v4.1-flash:cloud"
    assert cfg.fallback_model == "deepseek-v4-pro:cloud"
    assert cfg.model_temperature == 0.0
    assert cfg.model_reasoning is False
    assert cfg.structured_output_method == "function_calling"
    # D22 recursion guard — LOW, explicit
    assert cfg.recursion_limit == 50
    # D10 HITL
    assert cfg.interrupt_on == ("place_orders",)


def test_langgraph_pool_kwargs_forces_search_path():
    from psycopg.rows import dict_row

    kwargs = load_config(env={}).langgraph_pool_kwargs()

    assert kwargs["options"] == "-c search_path=langgraph"
    assert kwargs["autocommit"] is True
    assert kwargs["prepare_threshold"] == 0
    assert kwargs["row_factory"] is dict_row


def test_interrupt_on_map_defaults_to_place_orders():
    assert load_config(env={}).interrupt_on_map() == {"place_orders": True}


def test_secrets_are_sourced_from_the_environment():
    cfg = load_config(
        env={
            "DATABASE_URL": "postgresql://u:p@h:54320/db",
            "OLLAMA_API_KEY": "sk-ollama",
            "FRED_API_KEY": "fred-key",
            "SSL_CERT_FILE": "/certs/ca.pem",
        }
    )

    assert cfg.database_url == "postgresql://u:p@h:54320/db"
    assert cfg.ollama_api_key == "sk-ollama"
    assert cfg.fred_api_key == "fred-key"
    assert cfg.ssl_cert_file == "/certs/ca.pem"


def test_absent_secrets_stay_none_so_import_never_fails():
    cfg = load_config(env={})

    assert cfg.database_url is None
    assert cfg.ollama_api_key is None
    assert cfg.fred_api_key is None


def test_module_level_settings_is_a_fundconfig():
    assert isinstance(settings, FundConfig)
    assert settings.langgraph_schema == "langgraph"


def test_config_module_does_not_import_optimizer_at_top():
    config_py = Path(__file__).resolve().parents[2] / "src" / "fund" / "config.py"
    source = config_py.read_text(encoding="utf-8")
    assert "import optimizer" not in source
    assert "from optimizer" not in source
