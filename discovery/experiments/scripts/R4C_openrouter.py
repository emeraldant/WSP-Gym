"""Run OpenRouter models on backdoor-witness "direct" items (weights in the prompt, one call, no tools), with a hard
spend cap. One JSONL line per (model, item) in discovery/experiments/outputs/R4C/<model>.jsonl (resumable).

    BW_TASK_DIR=<wspnest task dir> PYTHONPATH=. python discovery/experiments/scripts/R4C_openrouter.py \
        --n 20 --cap-usd 15 --models a/b c/d ...
The API key is injected by the session proxy; the Authorization header below is a placeholder.
"""
import argparse, asyncio, json, os, sys
import httpx
sys.path.insert(0, os.environ.get("BW_TASK_DIR", "discovery/wspnest/backdoor-witness"))
import core, generate as G

OUT = "discovery/experiments/outputs/R4C"
URL = "https://openrouter.ai/api/v1/chat/completions"
SUFFIX = "\nThink it through, then end your reply with the JSON object on its own line."
spent = 0.0

def prompt_for(item):
    n = item["n_inputs"]
    text = G.INSTRUCTION.format(n=n, last=n - 1, spec=item["spec_text"], canary="",
                                body=G.DIRECT_BODY.format(weights=core.render_weights(item)))
    return text.split("-->\n", 1)[1].replace("written to `/app/answer.txt` and nothing else", "given as your answer") + SUFFIX

async def one(client, sem, model, i, f, a):
    global spent
    item = core.make_item("test", i)
    async with sem:
        if spent >= a.cap_usd:
            return
        rec = {"model": model, "index": i, "poisoned": item["poisoned"], "k": len(item["trigger"] or [])}
        try:
            r = await client.post(URL, headers={"Authorization": "Bearer proxy-injected"}, timeout=900, json={
                "model": model, "messages": [{"role": "user", "content": prompt_for(item)}],
                "max_tokens": a.max_tokens, "usage": {"include": True}})
            d = r.json()
            if "choices" not in d:
                rec.update(error=json.dumps(d)[:300], score=0.0)
            else:
                ch, u = d["choices"][0], d.get("usage", {})
                text = ch["message"].get("content") or ""
                cost = float(u.get("cost") or 0.0); spent += cost
                rec.update(score=float(core.score(item, text)), answer=text[-500:], finish=ch.get("finish_reason"),
                           n_in=u.get("prompt_tokens"), n_out=u.get("completion_tokens"),
                           n_reason=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"), cost=cost)
        except Exception as e:  # noqa: BLE001
            rec.update(error=repr(e)[:300], score=0.0)
        f.write(json.dumps(rec) + "\n"); f.flush()

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--cap-usd", type=float, default=15.0)
    ap.add_argument("--max-tokens", type=int, default=16000); ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sem = asyncio.Semaphore(12)
    async with httpx.AsyncClient() as client:
        jobs = []
        for m in a.models:
            path = f"{OUT}/{m.replace('/', '__')}.jsonl"
            done = {json.loads(l)["index"] for l in open(path) if '"error"' not in l} if os.path.exists(path) else set()
            f = open(path, "a")
            jobs += [one(client, sem, m, i, f, a) for i in range(a.n) if i not in done]
        await asyncio.gather(*jobs)
    print(f"spent ${spent:.2f} (cap ${a.cap_usd})")

asyncio.run(main())
