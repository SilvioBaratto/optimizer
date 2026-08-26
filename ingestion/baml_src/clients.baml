// ============================================================
// Compile-time default client (cloud OpenAI). At runtime the primary client is
// overridden per call via a ClientRegistry built from LLM_PROVIDER — see
// app/services/macro/_baml_client.py. Local/Ollama inference is not supported.
// ============================================================
client<llm> Model {
  provider "openai"
  options {
    model             env.OPENAI_MODEL
    api_key           env.OPENAI_API_KEY
    temperature       0
    max_tokens        1500
  }
}
// ============================================================
// Retry Policies
// ============================================================

retry_policy Constant {
  max_retries 3
  strategy {
    type constant_delay
    delay_ms 200
  }
}

retry_policy Exponential {
  max_retries 2
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}
