"""Smoke test: sample Inkling 2 Mini (free Tinker variant) on a few WSP-Gym val episodes, direct regime."""
import asyncio, json, sys, time
import tinker
from weightgym.env import load_split
from weightgym import llm
from weightgym.eval import summarize

MODEL = sys.argv[1] if len(sys.argv) > 1 else "thinkingmachines/Inkling-2-Mini:peft:65536:free"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT = f"discovery/experiments/outputs/R4/smoke_{int(time.time())}.jsonl"

async def main():
    _, eps = load_split("data/val.jsonl")
    pol = llm.LLMPolicy(tinker.ServiceClient(), MODEL, effort=0.0, max_tokens=256, temperature=1.0)
    await pol.start()
    print("renderer", pol.renderer_name, flush=True)
    recs = await llm.evaluate(pol, eps[:N], OUT, samples=1, concurrency=8, tag="R4_smoke", direct=True)
    rows = [r for r in recs if "seed" in r]
    print(json.dumps(summarize(rows), default=str)[:600])
    for r in rows[:3]:
        print(r["fault"], r["success"], r["n_in"], r["n_out"], repr(r["answer"])[:120])

asyncio.run(main())
