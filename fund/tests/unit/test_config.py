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
    # Fase-5 persistence: pinned Store key resolving a ConstraintSetRef
    assert cfg.constraint_set_store_key == "constraint_set"


def test_constraint_set_store_key_is_pinned():
    # The active ConstraintSet is cached under this key so a Phase-4
    # ConstraintSetRef resolves; it is a fixed contract, not env-sourced.
    assert load_config(env={}).constraint_set_store_key == "constraint_set"
    assert settings.constraint_set_store_key == "constraint_set"


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


def test_daemon_fields_default_to_the_phase9_contract():
    # Phase-9 Task 1 daemon knobs: defaults present with a bare (env-free) load.
    cfg = load_config(env={})

    # Cron uses the weekday NAME `sat` — a bare `0` fires Monday under
    # APScheduler `from_crontab` (0=Mon..6=Sun).
    assert cfg.fund_rebalance_cron == "0 3 * * sat"
    assert cfg.fund_drift_interval_seconds == 900
    assert cfg.fund_heartbeat_cadence_seconds == 30
    assert cfg.fund_orphan_timeout_seconds == 300
    assert cfg.fund_shutdown_drain_timeout_seconds == 30
    assert cfg.reprofile_interval_days == 365
    assert cfg.summary_token_threshold == 170000
    assert cfg.summary_messages_to_keep == 6


def test_daemon_fields_parse_from_fund_env_aliases():
    # Each daemon knob is overridable via its `FUND_*` alias; ints are coerced
    # from the string env values.
    cfg = load_config(
        env={
            "FUND_REBALANCE_CRON": "0 4 * * sun",
            "FUND_DRIFT_INTERVAL_SECONDS": "1800",
            "FUND_HEARTBEAT_CADENCE_SECONDS": "45",
            "FUND_ORPHAN_TIMEOUT_SECONDS": "600",
            "FUND_SHUTDOWN_DRAIN_TIMEOUT_SECONDS": "60",
            "FUND_REPROFILE_INTERVAL_DAYS": "180",
            "FUND_SUMMARY_TOKEN_THRESHOLD": "120000",
            "FUND_SUMMARY_MESSAGES_TO_KEEP": "8",
        }
    )

    assert cfg.fund_rebalance_cron == "0 4 * * sun"
    assert cfg.fund_drift_interval_seconds == 1800
    assert cfg.fund_heartbeat_cadence_seconds == 45
    assert cfg.fund_orphan_timeout_seconds == 600
    assert cfg.fund_shutdown_drain_timeout_seconds == 60
    assert cfg.reprofile_interval_days == 180
    assert cfg.summary_token_threshold == 120000
    assert cfg.summary_messages_to_keep == 8


def test_cost_cap_fields_unchanged_by_daemon_additions():
    # The D22 cost caps are pre-existing and must not shift when the daemon
    # fields land (max_pm_rounds is finalised — wired in — by Task 4).
    cfg = load_config(env={})

    assert cfg.max_pm_rounds == 10
    assert cfg.recursion_limit == 50


def test_no_prometheus_metrics_port_field():
    # Prometheus is out of scope for Phase 9 — no FUND_METRICS_PORT knob.
    cfg = load_config(env={"FUND_METRICS_PORT": "9100"})

    assert not hasattr(cfg, "fund_metrics_port")
    assert not hasattr(cfg, "metrics_port")


# --- Switchable LLM backends: llm_provider + per-provider config plumbing -----


def test_llm_provider_defaults_to_ollama():
    # The provider registry selects a builder by LLM_PROVIDER; the env-free
    # default is Ollama so today's DeepSeek-on-Ollama-Cloud path is unchanged.
    assert load_config(env={}).llm_provider == "ollama"
    assert load_config(env={"LLM_PROVIDER": "anthropic"}).llm_provider == "anthropic"


def test_model_ids_and_ollama_base_url_are_env_sourced():
    # FUND_PRIMARY_MODEL / FUND_FALLBACK_MODEL / OLLAMA_BASE_URL become live env
    # (previously dataclass-only); defaults are preserved when unset.
    cfg = load_config(
        env={
            "FUND_PRIMARY_MODEL": "gpt-4o",
            "FUND_FALLBACK_MODEL": "gpt-4o-mini",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }
    )
    assert cfg.primary_model == "gpt-4o"
    assert cfg.fallback_model == "gpt-4o-mini"
    assert cfg.ollama_base_url == "http://localhost:11434"

    # Env-free load keeps the pinned D4 defaults.
    default_cfg = load_config(env={})
    assert default_cfg.primary_model == "deepseek-v4.1-flash:cloud"
    assert default_cfg.fallback_model == "deepseek-v4-pro:cloud"
    assert default_cfg.ollama_base_url == "https://ollama.com"


def test_provider_keys_and_endpoints_sourced_from_env():
    cfg = load_config(
        env={
            "OPENAI_API_KEY": "sk-openai",
            "OPENAI_BASE_URL": "https://oai.example/v1",
            "OPENROUTER_API_KEY": "sk-or",
            "ANTHROPIC_API_KEY": "sk-anthropic",
            "GOOGLE_API_KEY": "goog-key",
            "GROQ_API_KEY": "gsk",
            "NVIDIA_API_KEY": "nvapi",
            "NVIDIA_BASE_URL": "http://nim:8000/v1",
            "HUGGINGFACEHUB_API_TOKEN": "hf_token",
            "FUND_HF_MODE": "local",
            "AWS_REGION": "eu-west-1",
            "AZURE_OPENAI_API_KEY": "az-key",
            "AZURE_OPENAI_ENDPOINT": "https://az.openai.azure.com",
            "OPENAI_API_VERSION": "2024-06-01",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt4o-deploy",
        }
    )

    assert cfg.openai_api_key == "sk-openai"
    assert cfg.openai_base_url == "https://oai.example/v1"
    assert cfg.openrouter_api_key == "sk-or"
    assert cfg.anthropic_api_key == "sk-anthropic"
    assert cfg.google_api_key == "goog-key"
    assert cfg.groq_api_key == "gsk"
    assert cfg.nvidia_api_key == "nvapi"
    assert cfg.nvidia_base_url == "http://nim:8000/v1"
    assert cfg.hf_api_token == "hf_token"  # noqa: S105 (test fixture, not a secret)
    assert cfg.hf_mode == "local"
    assert cfg.aws_region == "eu-west-1"
    assert cfg.azure_openai_api_key == "az-key"
    assert cfg.azure_openai_endpoint == "https://az.openai.azure.com"
    assert cfg.azure_openai_api_version == "2024-06-01"
    assert cfg.azure_openai_deployment_name == "gpt4o-deploy"


def test_provider_fields_default_safe_so_bare_import_never_fails():
    # Every provider key/endpoint stays None with no env; hf_mode defaults to
    # "cloud" (local is deferred). A bare load must not require any secret.
    cfg = load_config(env={})

    assert cfg.openai_api_key is None
    assert cfg.openai_base_url is None
    assert cfg.openrouter_api_key is None
    assert cfg.anthropic_api_key is None
    assert cfg.google_api_key is None
    assert cfg.groq_api_key is None
    assert cfg.nvidia_api_key is None
    assert cfg.nvidia_base_url is None
    assert cfg.hf_api_token is None
    assert cfg.aws_region is None
    assert cfg.azure_openai_api_key is None
    assert cfg.azure_openai_endpoint is None
    assert cfg.azure_openai_api_version is None
    assert cfg.azure_openai_deployment_name is None
    assert cfg.hf_mode == "cloud"


# --- T2: file-based docker secrets (<NAME>_FILE wins over inline <NAME>) -------


def test_read_secret_prefers_nonempty_file(tmp_path):
    from fund.config import _read_secret

    secret_file = tmp_path / "ollama_api_key"
    secret_file.write_text("sk-from-file\n", encoding="utf-8")  # trailing NL stripped
    env = {"OLLAMA_API_KEY_FILE": str(secret_file), "OLLAMA_API_KEY": "sk-inline"}

    assert _read_secret(env, "OLLAMA_API_KEY") == "sk-from-file"


def test_read_secret_empty_file_falls_through_to_inline(tmp_path):
    from fund.config import _read_secret

    # Compose always renders every secret file (empty placeholder when unset),
    # so a whitespace-only file must mean "unset", not "".
    secret_file = tmp_path / "ollama_api_key"
    secret_file.write_text("   \n", encoding="utf-8")
    env = {"OLLAMA_API_KEY_FILE": str(secret_file), "OLLAMA_API_KEY": "sk-inline"}

    assert _read_secret(env, "OLLAMA_API_KEY") == "sk-inline"


def test_read_secret_unreadable_file_falls_through_to_inline(tmp_path):
    from fund.config import _read_secret

    missing = tmp_path / "does_not_exist"
    env = {"OLLAMA_API_KEY_FILE": str(missing), "OLLAMA_API_KEY": "sk-inline"}

    assert _read_secret(env, "OLLAMA_API_KEY") == "sk-inline"


def test_read_secret_inline_only_when_no_file():
    from fund.config import _read_secret

    assert _read_secret({"FRED_API_KEY": "fred"}, "FRED_API_KEY") == "fred"


def test_read_secret_missing_both_returns_none():
    from fund.config import _read_secret

    assert _read_secret({}, "OLLAMA_API_KEY") is None


def test_read_secret_empty_inline_is_none():
    from fund.config import _read_secret

    assert _read_secret({"FRED_API_KEY": ""}, "FRED_API_KEY") is None


def test_provider_keys_and_fred_read_from_secret_files(tmp_path):
    # Compose maps <NAME>_FILE=/run/secrets/<name>; load_config must resolve each
    # secret from that file without requiring an inline env var.
    def _mk(name: str, value: str) -> str:
        path = tmp_path / name
        path.write_text(value, encoding="utf-8")
        return str(path)

    cfg = load_config(
        env={
            "OLLAMA_API_KEY_FILE": _mk("ollama_api_key", "sk-ollama-file"),
            "OPENAI_API_KEY_FILE": _mk("openai_api_key", "sk-openai-file"),
            "OPENROUTER_API_KEY_FILE": _mk("openrouter_api_key", "sk-or-file"),
            "ANTHROPIC_API_KEY_FILE": _mk("anthropic_api_key", "sk-anthropic-file"),
            "GOOGLE_API_KEY_FILE": _mk("google_api_key", "goog-file"),
            "GROQ_API_KEY_FILE": _mk("groq_api_key", "gsk-file"),
            "NVIDIA_API_KEY_FILE": _mk("nvidia_api_key", "nvapi-file"),
            "HUGGINGFACEHUB_API_TOKEN_FILE": _mk("huggingfacehub_api_token", "hf-file"),
            "AZURE_OPENAI_API_KEY_FILE": _mk("azure_openai_api_key", "az-file"),
            "FRED_API_KEY_FILE": _mk("fred_api_key", "fred-file"),
        }
    )

    assert cfg.ollama_api_key == "sk-ollama-file"
    assert cfg.openai_api_key == "sk-openai-file"
    assert cfg.openrouter_api_key == "sk-or-file"
    assert cfg.anthropic_api_key == "sk-anthropic-file"
    assert cfg.google_api_key == "goog-file"
    assert cfg.groq_api_key == "gsk-file"
    assert cfg.nvidia_api_key == "nvapi-file"
    assert cfg.hf_api_token == "hf-file"  # noqa: S105 (test fixture, not a secret)
    assert cfg.azure_openai_api_key == "az-file"
    assert cfg.fred_api_key == "fred-file"


def test_secret_file_wins_over_inline_env(tmp_path):
    secret_file = tmp_path / "anthropic_api_key"
    secret_file.write_text("sk-file", encoding="utf-8")

    cfg = load_config(
        env={
            "ANTHROPIC_API_KEY_FILE": str(secret_file),
            "ANTHROPIC_API_KEY": "sk-inline",
        }
    )

    assert cfg.anthropic_api_key == "sk-file"


def test_ssl_cert_file_is_a_path_not_routed_through_read_secret(tmp_path):
    # SSL_CERT_FILE literally ends in _FILE but is itself the CA-bundle path, not
    # a docker-secret file whose *content* is the value. It must pass through
    # verbatim, never be read as a secret file.
    ca = tmp_path / "ca.pem"
    ca.write_text("-----BEGIN CERT-----", encoding="utf-8")

    cfg = load_config(env={"SSL_CERT_FILE": str(ca)})

    assert cfg.ssl_cert_file == str(ca)
