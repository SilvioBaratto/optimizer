"""Phase 0 exit probe — D4.

Verify the DeepSeek-on-Ollama-Cloud model path is viable for deepagents:
  (a) connectivity + auth (OLLAMA_API_KEY) to https://ollama.com,
  (b) native tool-calling via bind_tools,
  (c) with_structured_output round-trip to a pydantic schema,
  (d) real .with_fallbacks failover to the higher tier,
all on the non-thinking flash route (reasoning=False).

Run (isolated, no workspace mutation):
  uv run --isolated --no-project \
    --with langchain-ollama --with langchain-core --with pydantic \
    python .phase0-probes/probe_d4_ollama_deepseek.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

PRIMARY = "deepseek-v4.1-flash:cloud"
FALLBACK = "deepseek-v4-pro:cloud"
HOST = "https://ollama.com"


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
        reasoning=False,  # force non-thinking route (thinking mode breaks tool_choice)
    )


@tool
def add(a: int, b: int) -> int:
    """Add two integers a and b."""
    return a + b


class RiskProfile(BaseModel):
    """A MiFID-style risk mapping."""

    risk_aversion: float = Field(description="1=aggressive .. 10=very cautious")
    objective: str = Field(description="one of: protection, income, growth, max")


def main() -> int:
    key = load_key()
    print(f"[D4] key loaded (len {len(key)}), host {HOST}, primary {PRIMARY}")
    results: dict[str, str] = {}

    llm = make(PRIMARY, key)

    # (a) connectivity + auth
    try:
        r = llm.invoke("Reply with exactly the word: OK")
        txt = (r.content or "").strip()
        results["a_connectivity"] = f"PASS (reply={txt!r})"
    except Exception as e:  # noqa: BLE001
        results["a_connectivity"] = f"FAIL ({type(e).__name__}: {e})"
        # If auth/model is dead, the rest will fail too — report and stop early.
        for k, v in results.items():
            print(f"[D4] {k}: {v}")
        print("\n[D4] FAIL — could not reach the model; later checks skipped")
        return 1

    # (b) tool-calling
    try:
        tl = llm.bind_tools([add])
        r = tl.invoke("What is 17 plus 25? You must call the add tool.")
        calls = getattr(r, "tool_calls", []) or []
        ok = bool(calls) and calls[0]["name"] == "add"
        args = calls[0]["args"] if calls else None
        results["b_tool_calling"] = (
            f"PASS (tool={calls[0]['name']}, args={args})" if ok
            else f"FAIL (no/other tool_calls: {calls})"
        )
    except Exception as e:  # noqa: BLE001
        results["b_tool_calling"] = f"FAIL ({type(e).__name__}: {e})"

    # (c) structured output
    for method in ("json_schema", "function_calling"):
        try:
            so = llm.with_structured_output(RiskProfile, method=method)
            r = so.invoke(
                "A cautious retail investor whose goal is steady income. "
                "Fill the risk profile."
            )
            ok = isinstance(r, RiskProfile)
            results["c_structured_output"] = (
                f"PASS (method={method}, {r})" if ok
                else f"FAIL (method={method}, got {type(r)})"
            )
            if ok:
                break
        except Exception as e:  # noqa: BLE001
            results["c_structured_output"] = f"FAIL (method={method}, {type(e).__name__}: {e})"

    # (d) real failover: bogus primary -> real fallback
    try:
        bogus = make("this-model-does-not-exist:cloud", key)
        real = make(FALLBACK, key)
        chain = bogus.with_fallbacks([real])
        r = chain.invoke("Reply with exactly the word: FALLBACK")
        txt = (r.content or "").strip()
        results["d_fallback"] = f"PASS (failed over to {FALLBACK}, reply={txt!r})"
    except Exception as e:  # noqa: BLE001
        results["d_fallback"] = f"FAIL ({type(e).__name__}: {e})"

    print()
    for k, v in results.items():
        print(f"[D4] {k}: {v}")

    passed = all(v.startswith("PASS") for v in results.values())
    print(f"\n[D4] {'PASS' if passed else 'PARTIAL/FAIL'} — see per-check results above")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
