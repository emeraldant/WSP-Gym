"""Run a Tinker model on backdoor-witness "direct" items (weights in the prompt, one model call, no tools).

    PYTHONPATH=. python discovery/experiments/scripts/R4B_model_eval.py <arm: direct|reason> [n] [model]
Writes one JSONL line per item to discovery/experiments/outputs/R4B/<tag>.jsonl (resumable).
"""
import asyncio, json, os, sys
import tinker
sys.path.insert(0, os.environ.get("BW_TASK_DIR", "discovery/wspnest/backdoor-witness"))  # needs WSPNest's CANARY two levels up
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import R4_mini_fix  # noqa: F401
import core, generate as G
from weightgym import llm

ARM = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
MODEL = sys.argv[3] if len(sys.argv) > 3 else "thinkingmachines/Inkling-2-Mini:peft:65536:free"
TAG = f"{MODEL.split('/')[-1].split(':')[0]}_{ARM}"
OUT = f"discovery/experiments/outputs/R4B/{TAG}.jsonl"
SUFFIX = {"direct": "\nReply with the JSON object only, immediately, with no explanation.",
          "reason": "\nThink it through, then end your reply with the JSON object on its own line."}[ARM]

def prompt_for(item):
    n = item["n_inputs"]
    body = G.DIRECT_BODY.format(weights=core.render_weights(item))
    text = G.INSTRUCTION.format(n=n, last=n - 1, spec=item["spec_text"], body=body, canary="")
    text = text.split("-->\n", 1)[1].replace("written to `/app/answer.txt` and nothing else", "given as your answer")
    return text + SUFFIX

async def main():
    done = {json.loads(l)["index"] for l in open(OUT)} if os.path.exists(OUT) else set()
    pol = llm.LLMPolicy(tinker.ServiceClient(), MODEL, effort=0.0 if ARM == "direct" else None,
                        max_tokens=512 if ARM == "direct" else 12000, temperature=1.0)
    await pol.start()
    sem, f = asyncio.Semaphore(8), open(OUT, "a")
    async def one(i):
        item = core.make_item("test", i)
        async with sem:
            for attempt in range(3):
                try:
                    n_in, res = await pol.sample(prompt_for(item), 1); break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2: res, n_in = [{"answer": "", "thinking": "", "n_out": 0, "stop_reason": f"error {e!r}"[:200]}], 0
                    await asyncio.sleep(5)
        r = res[0]
        rec = {"index": i, "poisoned": item["poisoned"], "n_inputs": item["n_inputs"], "k": len(item["trigger"] or []),
               "score": float(core.score(item, r["answer"])), "answer": r["answer"][-600:], "n_in": n_in, "n_out": r["n_out"],
               "stop_reason": r["stop_reason"], "thinking_tail": r["thinking"][-1500:]}
        f.write(json.dumps(rec) + "\n"); f.flush()
    await asyncio.gather(*[one(i) for i in range(N) if i not in done])

asyncio.run(main())
