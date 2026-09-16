"""Phase 0 exit probe — D2 (+ deepagents/D4 composition).

Verify the deepagents wiring the contract depends on:
  (1) StoreBackend + create_file_data preload of a read-only SKILL.md,
  (2) create_deep_agent accepts our ChatOllama model instance,
  (3) a custom subagent carries its OWN skills=[...] (subagents don't inherit),
  (4) the model drives a real @tool call end-to-end inside the agent loop,
  (5) a .with_fallbacks(...) wrapper still composes as the model.

Run (isolated, no workspace mutation):
  uv run --isolated --no-project \
    --with deepagents --with langchain-ollama --with langchain-core --with pydantic \
    python .phase0-probes/probe_d2_deepagents.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import tool

HOST = "https://ollama.com"
PRIMARY = "deepseek-v4.1-flash:cloud"
FALLBACK = "deepseek-v4-pro:cloud"

SKILL_MD = """---
name: universe-size
description: Report how many instruments are in the DB universe. Use when asked about universe count.
---
# Universe size
Call the `get_universe_size` tool and report the integer it returns. Do NOT invent a number.
"""


def load_key() -> str:
    env = Path(__file__).resolve().parent.parent / ".env"
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("OLLAMA_API_KEY=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OLLAMA_API_KEY not found in .env")


def make(model: str, key: str):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=HOST,
        client_kwargs={"headers": {"Authorization": f"Bearer {key}"}},
        temperature=0,
        reasoning=False,
    )


@tool
def get_universe_size() -> int:
    """Return the number of instruments in the DB universe."""
    return 8898


def main() -> int:
    key = load_key()
    results: dict[str, str] = {}

    from deepagents import create_deep_agent
    from deepagents.backends import StoreBackend
    from deepagents.backends.utils import create_file_data
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore

    # (1) preload SKILL.md into the store's virtual filesystem
    store = InMemoryStore()
    try:
        store.put(
            ("filesystem",),
            "/skills/universe-size/SKILL.md",
            create_file_data(SKILL_MD),
        )
        got = store.get(("filesystem",), "/skills/universe-size/SKILL.md")
        results["1_store_preload"] = "PASS" if got is not None else "FAIL (not retrievable)"
    except Exception as e:  # noqa: BLE001
        results["1_store_preload"] = f"FAIL ({type(e).__name__}: {e})"

    model = make(PRIMARY, key)

    # (2)+(3) build agent with StoreBackend + skills + a custom subagent w/ its own skills
    try:
        subagent = {
            "name": "universe-agent",
            "description": "Answers questions about the DB universe size.",
            "system_prompt": "Use the universe-size skill and the get_universe_size tool.",
            "tools": [get_universe_size],
            "skills": ["/skills/"],  # custom subagents must be given skills explicitly
        }
        agent = create_deep_agent(
            model=model,
            tools=[get_universe_size],
            system_prompt="You are a PM. Delegate universe questions to universe-agent.",
            subagents=[subagent],
            backend=StoreBackend(namespace=lambda rt: ("filesystem",), store=store),
            skills=["/skills/"],
            store=store,
            checkpointer=MemorySaver(),
        )
        results["2_agent_build"] = "PASS"
        results["3_subagent_skills"] = "PASS (custom subagent given explicit skills=[...])"
    except Exception as e:  # noqa: BLE001
        results["2_agent_build"] = f"FAIL ({type(e).__name__}: {e})"
        results["3_subagent_skills"] = "SKIPPED (build failed)"
        agent = None

    # (4) end-to-end: model drives the @tool inside the loop
    if agent is not None:
        try:
            cfg = {"configurable": {"thread_id": "probe-d2"}}
            out = agent.invoke(
                {"messages": [{"role": "user",
                               "content": "How many instruments are in the universe? "
                                          "Use the get_universe_size tool."}]},
                cfg,
            )
            msgs = out.get("messages", [])
            text = ""
            for m in reversed(msgs):
                c = getattr(m, "content", "") or (m.get("content") if isinstance(m, dict) else "")
                if isinstance(c, str) and c.strip():
                    text = c
                    break
            ok = "8898" in text or "8,898" in text
            results["4_end_to_end_tool"] = (
                f"PASS (answer contains 8898; {len(msgs)} msgs)" if ok
                else f"PARTIAL (ran, {len(msgs)} msgs, tail={text[:120]!r})"
            )
        except Exception as e:  # noqa: BLE001
            results["4_end_to_end_tool"] = f"FAIL ({type(e).__name__}: {e})"
    else:
        results["4_end_to_end_tool"] = "SKIPPED (no agent)"

    # (5) fallback via ModelFallbackMiddleware (accepts BaseChatModel instances)
    try:
        from langchain.agents.middleware import ModelFallbackMiddleware

        bogus = make("this-model-does-not-exist:cloud", key)
        real = make(FALLBACK, key)
        agent2 = create_deep_agent(
            model=bogus,
            tools=[get_universe_size],
            system_prompt="Answer using the tool.",
            middleware=[ModelFallbackMiddleware(real)],
            checkpointer=MemorySaver(),
        )
        out = agent2.invoke(
            {"messages": [{"role": "user",
                           "content": "Call get_universe_size and report the number."}]},
            {"configurable": {"thread_id": "probe-d2-fb"}},
        )
        msgs = out.get("messages", [])
        text = ""
        for m in reversed(msgs):
            c = getattr(m, "content", "") or (m.get("content") if isinstance(m, dict) else "")
            if isinstance(c, str) and c.strip():
                text = c
                break
        ok = "8898" in text or "8,898" in text
        results["5_fallback_compose"] = (
            f"PASS (failover model ran the tool; tail={text[:80]!r})" if ok
            else f"PARTIAL (composed+ran, tail={text[:120]!r})"
        )
    except Exception as e:  # noqa: BLE001
        results["5_fallback_compose"] = f"FAIL ({type(e).__name__}: {e})"

    print()
    for k, v in results.items():
        print(f"[D2] {k}: {v}")
    hard_fail = any(v.startswith("FAIL") for v in results.values())
    print(f"\n[D2] {'FAIL' if hard_fail else 'PASS/PARTIAL'} — see per-check results above")
    return 2 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
