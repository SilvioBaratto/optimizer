client<llm> Model {
  provider "openai-generic"
  options {
    base_url env.OLLAMA_BASE_URL
    api_key env.OLLAMA_API_KEY
    model "qwen3.5:4b"
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
