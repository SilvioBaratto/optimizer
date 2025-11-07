// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

client<llm> Model {
  provider openai
  retry_policy Exponential
  options {
    api_key env.OPENAI_API_KEY
    model "gpt-5-mini"
    http {
      connect_timeout_ms 60000       // 1 minute to establish connection
      request_timeout_ms 300000      // 5 minutes for total request (large prompts)
      time_to_first_token_timeout_ms 120000  // 2 minutes to start streaming
    }
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