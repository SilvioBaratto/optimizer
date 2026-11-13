# ADR-0006: A lazy provider registry over LangChain chat models

- Status: Accepted
- Date: 2026-09-18

## Context

The fund bridge started with one model path: DeepSeek on Ollama Cloud through
`ChatOllama`, primary plus fallback (commit `df67a7da`, 2026-09-18, "build_chat_model /
build_primary / build_fallback construct the DeepSeek-on-Ollama-Cloud models from
FundConfig"). Later work in October 2026 made the backend switchable, and the fixes of
23-24 October (`1fde6d95`, `7a800e01`, `312eb801`, `019024b0`) show the extras, the
prefilled setup values and the structured-output method being sorted out per provider.

## Decision

`fund.agents.model.build_chat_model` dispatches on `FundConfig.llm_provider` through a
`_BUILDERS` registry of ten providers: `ollama`, `openai`, `openrouter`, `anthropic`,
`groq`, `google`, `aws`, `microsoft`, `nvidia` and `huggingface`. Each is a small
`_build_<provider>` function over a LangChain `BaseChatModel`.

- The env-free default stays DeepSeek on Ollama Cloud, byte for byte.
- Each builder imports its SDK inside the function, so a bare `import fund.agents.model`
  pulls in no provider package; `tests/unit/hygiene/test_model_no_provider_import.py`
  guards this.
- Required secrets are validated when a model is built, never at import. An uninstalled
  SDK surfaces as a `RuntimeError` naming the `uv sync` extra that ships it.
- Non-default provider SDKs are opt-in extras.

## Alternatives considered

- **One hard-wired provider**, which is how `df67a7da` began. The registry replaced it.
- **A provider-abstraction library in front of the SDKs.** Not evaluated anywhere in the
  history, so no reason is recorded either way.

## Consequences

- Every tool result an agent reads (holdings, mandate, audit, transcripts) is sent to the
  configured provider; choosing a provider is choosing who receives that data.
- Time-outs, structured-output support and retry behaviour differ per provider and are
  set per builder, so a new provider needs its own deadline and its own tests.
- Importing the registry stays cheap, which keeps `worker.py` and `scheduler.py` free of
  the agent stack.
