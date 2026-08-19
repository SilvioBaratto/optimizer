// ============================================================
// Single client — Ollama local inference (gemma4:12b-mlx)
// Set OLLAMA_MODEL=gemma4:12b-mlx in .env
// ============================================================
client<llm> Model {
  provider "openai-generic"
  options {
    base_url          env.OLLAMA_BASE_URL
    api_key           env.OLLAMA_API_KEY
    model             env.OLLAMA_MODEL
    timeout_ms        300000
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
