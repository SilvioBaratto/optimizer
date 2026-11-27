"""What ``GROQ_LOG``, ``OPENAI_LOG`` and ``ANTHROPIC_LOG`` can write to the daemon log.

Each of those SDKs reads its variable when it is first imported and then raises its
own logger (and the ``httpx`` logger of its stack) to INFO or DEBUG. Providers are
imported lazily, after the entry point silenced the HTTP loggers, so only a fresh
interpreter sees that import happen: every case runs in one.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_PROBE_TIMEOUT_SECONDS = 120

# A root handler with no level, as the daemon has: a record from a child logger is
# never checked against the root logger's level, only against its own logger's.
_PROBE = """
import logging
import sys

from fund.agents import model
from fund.config import FundConfig, silence_http_request_logs

provider, key_field, sdk_logger = sys.argv[1:4]
written = []


class _Capture(logging.Handler):
    def emit(self, record):
        written.append(record.getMessage())


logging.getLogger().addHandler(_Capture())
logging.getLogger().setLevel(logging.INFO)
silence_http_request_logs()
model.build_chat_model(
    FundConfig(llm_provider=provider, **{key_field: "probe-key"}), model_name="probe"
)
logging.getLogger("httpx").info("HTTP Request: POST https://leak.example/httpx")
logging.getLogger("httpx2").info("HTTP Request: POST https://leak.example/httpx2")
logging.getLogger(sdk_logger + "._base_client").debug("Request options: PROMPT-LEAK")
logging.getLogger(sdk_logger + "._base_client").debug("Sending HTTP Request: URL-LEAK")
print(written)
"""


@pytest.mark.criterion("REQ-FUND-028")
@pytest.mark.parametrize("level", ["info", "debug"])
@pytest.mark.parametrize(
    ("provider", "key_field", "variable", "sdk_logger"),
    [
        ("groq", "groq_api_key", "GROQ_LOG", "groq"),
        ("openai", "openai_api_key", "OPENAI_LOG", "openai"),
        ("anthropic", "anthropic_api_key", "ANTHROPIC_LOG", "anthropic"),
    ],
)
def test_an_sdk_log_variable_cannot_put_a_url_or_a_prompt_in_the_log(
    provider, key_field, variable, sdk_logger, level
):
    """Validates ``fund.agents.model.build_chat_model`` against the provider SDKs.

    After the model is built, the request line of both HTTP stacks and the SDK's own
    DEBUG lines (URL, response headers and the request body, which holds the prompt)
    must write nothing to a handler on the root logger.
    """
    result = subprocess.run(  # noqa: S603 - fixed argv, the interpreter under test
        [sys.executable, "-I", "-c", _PROBE, provider, key_field, sdk_logger],
        env={**os.environ, variable: level},
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT_SECONDS,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
