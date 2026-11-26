"""Run OpenRouter models on backdoor-witness v0.2 "direct" items (one call, weights in the prompt, no tools).

    BW_TASK_DIR=<wspnest task dir> PYTHONPATH=. python discovery/experiments/scripts/R4E_openrouter_v2.py \
        --n 20 --runs 1 --cap-usd 5 --max-tokens 32000 --models a/b ...
One JSONL row per (model, item, run) in discovery/experiments/outputs/R4E/<model>.jsonl (resumable).
Outcomes are kept apart: "answered", "cut off" (ran out of tokens), "filtered" (provider content filter) and
"provider error" (retried twice, then recorded and EXCLUDED from scores). The API key is injected by the session proxy.
"""
import argparse, asyncio, json, os, sys
import httpx
sys.path.insert(0, os.environ.get("BW_TASK_DIR", "discovery/wspnest/backdoor-witness"))
import core, generate as G

OUT = "discovery/experiments/outputs/R4E"
URL = "https://openrouter.ai/api/v1/chat/completions"
spent = 0.0

def prompt_for(item):
    text = G.instruction(item, "direct")
    assert "written to `/app/answer.txt` and nothing else" in text
    return text.replace("written to `/app/answer.txt` and nothing else", "given as the last line of your reply")

ITEMS = None      # --items-file: a JSONL of items to use instead of a frozen split (for pilots of sets still being built)


async def one(client, sem, model, i, run, f, a):
    global spent
    item = ITEMS[i] if ITEMS is not None else core.make_item(a.split, i)
    async with sem:
        if spent >= a.cap_usd:
            return
        rec = {"model": model, "split": a.split, "index": i, "item_index": item["index"], "run": run, "poisoned": item["poisoned"], "pair": item["pair"],
               "k": len(item["twin_trigger"]), "family": item["spec"]["family"], "max_tokens": a.max_tokens}
        for attempt in range(3):
            try:
                r = await client.post(URL, headers={"Authorization": "Bearer proxy-injected"}, timeout=1200, json={
                    "model": model, "messages": [{"role": "user", "content": prompt_for(item)}],
                    "max_tokens": a.max_tokens, "usage": {"include": True}})
                d = r.json()
                ch = (d.get("choices") or [None])[0]
                if ch is None or ch.get("finish_reason") == "error":
                    rec.update(outcome="provider error", detail=json.dumps(d)[:300]); await asyncio.sleep(5); continue
                u = d.get("usage", {}); text = ch["message"].get("content") or ""
                cost = float(u.get("cost") or 0.0); spent += cost
                fin = ch.get("finish_reason")
                outcome = "cut off" if fin == "length" else "filtered" if fin == "content_filter" else "answered"
                rec.update(outcome=outcome, finish=fin, answer=text[-1500:], n_in=u.get("prompt_tokens"), n_out=u.get("completion_tokens"),
                           n_reason=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"), cost=cost,
                           score=float(core.score(item, text)) if outcome == "answered" else 0.0)
                rec.pop("detail", None)
                break
            except Exception as e:  # noqa: BLE001
                rec.update(outcome="provider error", detail=repr(e)[:300]); await asyncio.sleep(5)
        f.write(json.dumps(rec) + "\n"); f.flush()

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test"); ap.add_argument("--n", type=int, default=20); ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--cap-usd", type=float, default=5.0); ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--items-file", default=None)
    a = ap.parse_args()
    global ITEMS
    if a.items_file: ITEMS = [json.loads(l) for l in open(a.items_file)]
    os.makedirs(OUT, exist_ok=True)
    sem = asyncio.Semaphore(12)
    async with httpx.AsyncClient() as client:
        jobs = []
        for m in a.models:
            path = f"{OUT}/{'' if a.split == 'test' else a.split + '__'}{m.replace('/', '__')}.jsonl"   # one file per split and model
            done = set()
            for l in (open(path) if os.path.exists(path) else []):
                r = json.loads(l)
                if r.get("outcome") != "provider error": done.add((r["index"], r["run"]))
            f = open(path, "a")
            jobs += [one(client, sem, m, i, run, f, a) for run in range(a.runs) for i in range(a.n) if (i, run) not in done]
        await asyncio.gather(*jobs)
    print(f"spent ${spent:.2f} (cap ${a.cap_usd})")

asyncio.run(main())
