"""Evaluate the trained Inkling 2 Mini repair checkpoint on frozen WSP-Gym test splits (same protocol as the README:
direct answers, effort 0, 256-token cap, temperature 1.0, one sample per network)."""
import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import R4_mini_fix  # noqa: F401
import tinker
from weightgym import llm
from weightgym.env import load_split
from weightgym.eval import summarize

RUN = "runs/r4b_mini_sft"
sampler = json.loads(open(f"{RUN}/checkpoints.jsonl").read().strip().splitlines()[-1])["sampler_path"]

async def main():
    pol = llm.LLMPolicy(tinker.ServiceClient(), R4_mini_fix.FREE, model_path=sampler, effort=0.0, max_tokens=256, temperature=1.0)
    await pol.start()
    for split in sys.argv[1:] or ["test_id", "test_trained"]:
        _, eps = load_split(f"data/{split}.jsonl")
        recs = await llm.evaluate(pol, eps, f"runs/final/mini_sft300/{split}.jsonl", samples=1, concurrency=32, tag=f"mini_sft300/{split}", direct=True)
        s = summarize([r for r in recs if "seed" in r])
        print(split, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k in ("n_faulty", "repair_rate", "repair_lo", "repair_hi", "clean_ok", "valid")}, flush=True)

asyncio.run(main())
