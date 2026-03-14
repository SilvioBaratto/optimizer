// ============================================================
// Single client — Ollama local inference
//
// Model created via: ./create_model.sh --original_model qwen3.5:4b --new_model finance-qwen
// Set OLLAMA_MODEL=finance-qwen in .env
//
// reasoning_effort "none": Qwen3.5 auto-enables thinking by default
//   and thinking tokens count against max_tokens (Ollama PR #14821).
//   "low" reasoning adds 3-5 min latency; "none" keeps responses at
//   5-30s with identical JSON quality for our prompts.
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
    reasoning_effort  "none"
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
