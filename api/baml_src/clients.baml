// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

// Using the new OpenAI Responses API for enhanced formatting
client<llm> CustomGPT5 {
  provider openai
  options {
    model "gpt-5.2"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> Gemini {
  provider google-ai
  options {
    model "gemini-3-pro-preview"
    api_key env.GOOGLE_API_KEY
    generationConfig {
      maxOutputTokens 16000
    }
  }
}

// Claude Opus 4.5 - Most capable model (released Nov 2025)
client<llm> CustomOpus45 {
  provider anthropic
  options {
    model "claude-opus-4-5-20251101"
    api_key env.ANTHROPIC_API_KEY
    max_tokens 16000
  }
}

// Claude Sonnet 4.5 - Best coding model and strongest for complex agents (released Sep 2025)
client<llm> CustomSonnet45 {
  provider anthropic
  options {
    model "claude-sonnet-4-5-20250929"
    api_key env.ANTHROPIC_API_KEY
    max_tokens 16000
  }
}

client<llm> AzureOpenAIGpt5 {
  provider azure-openai
  options {
    resource_name env.AZURE_OPENAI_RESOURCE_NAME
    deployment_id env.AZURE_OPENAI_GPT5_DEPLOYMENT_ID
    api_version env.AZURE_OPENAI_API_VERSION
    api_key env.AZURE_OPENAI_KEY
    max_tokens null
    }
}

// Main Ollama client for Briscola
// Uses OLLAMA_BASE_URL env var in Docker, defaults to localhost
client<llm> Ollama {
  provider "openai-generic"
  retry_policy Constant
  options {
    base_url env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1"
    model qwen3:4b
    // Optimize for speed
    temperature 0.3
    max_tokens 100
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/fallback
client<llm> OpenaiFallback {
  provider fallback
  options {
    // This will try the clients in order until one succeeds
    strategy [CustomGPT5, AzureOpenAIGpt5]
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/retry
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