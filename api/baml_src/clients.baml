// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

// ============================================================
// Base LLM Clients
// ============================================================

client<llm> GPT5 {
  provider openai
  options {
    model "gpt-5.1"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> GPT5Mini {
  provider openai
  options {
    model "gpt-5-mini"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> Gemini3 {
  provider google-ai
  options {
    model "gemini-3.1-pro-preview"
    api_key env.GOOGLE_API_KEY
    generationConfig {
      maxOutputTokens 8000
    }
  }
}

client<llm> Opus46 {
  provider anthropic
  retry_policy Exponential
  options {
    model "claude-opus-4-6"
    api_key env.ANTHROPIC_API_KEY
    max_tokens 8000
  }
}

client<llm> Sonnet46 {
  provider anthropic
  options {
    model "claude-sonnet-4-6"
    api_key env.ANTHROPIC_API_KEY
    max_tokens 8000
  }
}

client<llm> AzureOpenAIGPT5 {
  provider azure-openai
  options {
    resource_name env.AZURE_OPENAI_RESOURCE_NAME
    deployment_id "gpt-5.2"
    api_version env.AZURE_OPENAI_API_VERSION
    api_key env.AZURE_OPENAI_KEY
    max_tokens null
  }
}

client<llm> Ollama {
  provider "openai-generic"
  retry_policy Constant
  options {
    base_url env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1"
    model qwen3:4b
    temperature 0.3
    max_tokens 100
  }
}

// ============================================================
// Tier 1 — Simple/classification tasks (cheapest)
// Used by: ScoreNewsSentiment, CalibrateDelta, SelectCovRegime,
//          AdaptFactorWeights, Test
// ============================================================
client<llm> Tier1 {
  provider fallback
  retry_policy Constant
  options {
    strategy [GPT5Mini, Sonnet46]
  }
}

// ============================================================
// Tier 2 — Moderate reasoning (mid-cost)
// Used by: ClassifyMacroRegime, CalibrateRiskBudget,
//          GenerateValueView, GenerateMomentumView,
//          GenerateMacroView, GenerateExpertView
// ============================================================
client<llm> Tier2 {
  provider fallback
  retry_policy Exponential
  options {
    strategy [Sonnet46, GPT5]
  }
}

// ============================================================
// Tier 3 — Complex multi-step reasoning (highest quality)
// Used by: GenerateViews, DesignStressScenarios
// ============================================================
client<llm> Tier3 {
  provider fallback
  retry_policy Exponential
  options {
    strategy [Opus46, GPT5]
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
