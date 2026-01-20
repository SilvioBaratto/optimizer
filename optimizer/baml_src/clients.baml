// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

client<llm> AzureOpenAIGpt4o {
  provider azure-openai
  options {
    resource_name env.OPENAI_RESOURCE_NAME
    deployment_id env.OPENAI_GPT4O_DEPLOYMENT_ID
    api_version env.OPENAI_API_VERSION
    api_key env.OPENAI_KEY
    temperature 0.0
  }
}

client<llm> AzureOpenAIGpt5 {
  provider azure-openai
  options {
    resource_name env.OPENAI_RESOURCE_NAME
    deployment_id env.OPENAI_GPT5_DEPLOYMENT_ID
    api_version env.OPENAI_API_VERSION
    api_key env.OPENAI_KEY
    max_tokens null
    }
}
// https://docs.boundaryml.com/docs/snippets/clients/retry
retry_policy Constant {
  max_retries 3
  // Strategy is optional
  strategy {
    type constant_delay
    delay_ms 200
  }
}

retry_policy Exponential {
  max_retries 2
  // Strategy is optional
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}