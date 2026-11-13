"""Evaluate a closed frontier model on gym episodes through OpenRouter's chat-completions API.

Used for one comparison: the models from the published weight-space programming work, off the shelf,
on the same frozen test networks and prompts as Inkling 2 Small.

    python -m weightgym.frontier_eval --model openai/gpt-6.1-sol --mode reasoning --n 100 --arm frontier_sol_reasoning
    python -m weightgym.frontier_eval --model openai/gpt-6.1-sol --mode direct    --n 100 --arm frontier_sol_direct

  reasoning  the ordinary prompt, the provider's default reasoning, a 20,000-token completion cap
             (the same cap as the untrained Inkling reference)
  direct     the no-reasoning prompt with reasoning switched off or to the lowest level the API accepts;
             the reasoning tokens actually spent are logged, so "at a glance" can be checked

Only gym prompts are sent. The API key is injected by the environment's network proxy; this module
never sees it. Records use the same JSONL schema as weightgym.llm.evaluate, plus reasoning-token
counts and the provider-reported cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import httpx

from .env import load_split, score_answer
from .eval import record, summarize

URL = os.environ.get("WEIGHTGYM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
# Any OpenAI-compatible endpoint works: set WEIGHTGYM_BASE_URL and WEIGHTGYM_API_KEY. In the environment this
# project was developed in, the key is injected by a network proxy and the placeholder below is overwritten.
HEADERS = {"Authorization": f"Bearer {os.environ.get('WEIGHTGYM_API_KEY', 'proxy-injected')}", "Content-Type": "application/json"}
DIRECT_REASONING = ({"enabled": False}, {"effort": "none"}, {"effort": "minimal"}, {"effort": "low"})
# Spend guard shared by every process: each answered request appends its provider-reported cost to the ledger, and
# no new request is sent once the ledger total reaches the budget. Requests already in flight still finish.
LEDGER = os.environ.get("WEIGHTGYM_LEDGER", "runs/openrouter_ledger.jsonl")
BUDGET = float(os.environ.get("WEIGHTGYM_BUDGET_USD", "inf"))


def spent() -> float:
    if not os.path.exists(LEDGER):
        return 0.0
    return sum(json.loads(line).get("cost") or 0 for line in open(LEDGER) if line.strip())


def charge(arm: str, cost: float | None) -> None:
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"arm": arm, "cost": cost or 0, "time": time.time()}) + "\n")


async def call(client: httpx.AsyncClient, body: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            r = await client.post(URL, headers=HEADERS, json=body)
            if r.status_code in (429, 500, 502, 503, 504, 524):
                await asyncio.sleep(3 * 2**attempt)
                continue
            return {"status": r.status_code, "json": r.json()}
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                return {"status": 0, "json": {"error": {"message": repr(e)}}}
            await asyncio.sleep(3 * 2**attempt)
    return {"status": 0, "json": {"error": {"message": "retries exhausted"}}}


async def pick_direct_reasoning(client: httpx.AsyncClient, model: str, arm: str = "probe") -> dict:
    """Find the lowest reasoning setting this model accepts."""
    for setting in DIRECT_REASONING:
        res = await call(client, {"model": model, "messages": [{"role": "user", "content": "Reply with the single word ok."}],
                                  "max_tokens": 64, "reasoning": setting, "usage": {"include": True}}, retries=2)
        if isinstance(res["json"], dict) and res["json"].get("usage"):
            charge(arm, res["json"]["usage"].get("cost"))
        if res["status"] == 200 and "choices" in res["json"]:
            return setting
    raise SystemExit(f"{model}: no low-reasoning setting accepted")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["reasoning", "direct"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--task", default="repair", choices=["repair", "improve", "decompile"])
    ap.add_argument("--reasoning-on", action="store_true",
                    help="reasoning mode: ask for reasoning explicitly ({'enabled': true}) instead of the provider default; "
                         "falls back to the provider default if the API rejects it")
    a = ap.parse_args()
    _, eps = load_split(f"data/{a.split}.jsonl")
    eps = eps[a.start : a.start + a.n]
    if a.task == "improve":
        from .imperfect import wrap as _wrap

        eps = [_wrap(e) for e in eps]
    elif a.task == "decompile":
        from .decompile import wrap as _wrap

        eps = [_wrap(e) for e in eps]
    direct = a.mode == "direct"
    max_tokens = a.max_tokens or (512 if direct else 20000)
    out = f"runs/{'final' if a.task == 'repair' else 'final_' + a.task}/{a.arm}/{a.split}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done, recs, header = set(), [], None
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            if r.get("header"):
                header = r
            if "seed" in r:
                done.add(r["seed"])
                recs.append(r)
    sem = asyncio.Semaphore(a.concurrency)
    stopped = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0)) as client:
        if header is not None:  # resuming: keep the setting the arm started with
            reasoning = header["settings"]["reasoning"]
        elif direct:
            reasoning = await pick_direct_reasoning(client, a.model, a.arm)
        elif a.reasoning_on:
            probe = await call(client, {"model": a.model, "messages": [{"role": "user", "content": "Reply with the single word ok."}],
                                        "max_tokens": 2000, "reasoning": {"enabled": True}, "usage": {"include": True}}, retries=2)
            charge(a.arm, (probe["json"].get("usage") or {}).get("cost") if isinstance(probe["json"], dict) else None)
            reasoning = {"enabled": True} if probe["status"] == 200 and "choices" in probe["json"] else None
        else:
            reasoning = None
        f = open(out, "a")
        if not recs:
            f.write(json.dumps({"header": True, "tag": a.arm, "settings": {"base_model": a.model, "mode": a.mode,
                    "max_tokens": max_tokens, "reasoning": reasoning, "temperature": "provider default"},
                    "direct": direct, "time": time.time()}) + "\n")

        async def one(ep):
            if ep.seed in done:
                return
            body = {"model": a.model, "messages": [{"role": "user", "content": ep.prompt(direct)}],
                    "max_tokens": max_tokens, "usage": {"include": True}}
            if reasoning is not None:
                body["reasoning"] = reasoning
            async with sem:
                if spent() >= BUDGET:
                    if not stopped:
                        print(f"[stop] budget ${BUDGET:.2f} reached; no further requests for {a.arm}", flush=True)
                    stopped.append(ep.seed)
                    return
                res = await call(client, body)
            j = res["json"]
            if isinstance(j, dict) and j.get("usage"):
                charge(a.arm, j["usage"].get("cost"))
            if res["status"] != 200 or "choices" not in j:
                print(f"[warn] seed {ep.seed}: {res['status']} {str(j.get('error'))[:200]}", flush=True)
                return
            msg = j["choices"][0]["message"]
            answer = msg.get("content") or ""
            usage = j.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            sc = ep.score(answer) if hasattr(ep, "score") else score_answer(ep, answer)
            rec = record(ep, sc, sample=0, tag=a.arm, direct=direct, prompt_hash=ep.prompt_hash(direct),
                         n_in=usage.get("prompt_tokens", 0), n_out=usage.get("completion_tokens", 0),
                         reasoning_tokens=details.get("reasoning_tokens"), cost_usd=usage.get("cost"),
                         stop_reason=str(j["choices"][0].get("finish_reason")), termination="", error=sc["error"],
                         ops=sc["ops"], answer=answer, thinking=(msg.get("reasoning") or "")[:20000])
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()

        await asyncio.gather(*(one(ep) for ep in eps))
        f.close()
    s = summarize(recs)
    cost = sum(r.get("cost_usd") or 0 for r in recs)
    rt = [r["reasoning_tokens"] for r in recs if r.get("reasoning_tokens") is not None]
    print(f"{a.arm}: n={s['n']} repair {s['repair_rate']:.3f} [{s['repair_lo']:.3f}, {s['repair_hi']:.3f}] clean_ok {s['clean_ok']:.2f} "
          f"valid {s['valid']:.2f} | mean completion tokens {sum(r['n_out'] for r in recs) / max(1, len(recs)):.0f}, "
          f"mean reasoning tokens {sum(rt) / max(1, len(rt)):.0f} | provider cost ${cost:.2f} | reasoning setting {reasoning}")
    if stopped:
        print(f"{a.arm}: {len(stopped)} episodes not run (budget reached)")


if __name__ == "__main__":
    asyncio.run(main())
