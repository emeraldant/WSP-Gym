"""Rebuild the running Tinker cost tally from the logs under runs/.

    python -m weightgym.cost            # prints the tally and writes runs/cost_tally.json

Sources:
  * evaluation JSONL files written by weightgym.llm.evaluate: prompt tokens are billed once per
    sampling request (one request per episode) at the prefill price, output tokens at the sample price;
  * SFT runs (metrics.jsonl with kind == "train"): tokens trained at the train price;
  * RL runs (tinker-cookbook metrics.jsonl): the cookbook's own ``cost/cumulative_usd``, which covers
    training rollouts (prefill, cached prefill, sampled, trained) but not evaluations.
These are estimates from token counts and list prices, not the invoice.
"""

from __future__ import annotations

import glob
import json
import os

from .tinker_train import PRICE

BUDGET_USD = 250.0
STOP_FRACTION = 0.85


def _eval_file_cost(path: str) -> tuple[float, int, int] | None:
    model, seen, n_in, n_out = None, set(), 0, 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("header"):
                model = r["settings"]["base_model"]
            elif "n_out" in r and "sample" in r:
                if r["seed"] not in seen:
                    seen.add(r["seed"])
                    n_in += r["n_in"]
                n_out += r["n_out"]
    if model is None or model not in PRICE:
        return None
    p = PRICE[model]
    return (n_in * p["prefill"] + n_out * p["sample"]) / 1e6, n_in, n_out


def tally(root: str = "runs") -> dict:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)):
        base = os.path.basename(path)
        rel = os.path.relpath(path, root)
        if base == "metrics.jsonl":
            sft_usd, rl_usd = 0.0, 0.0
            for line in open(path):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "train":
                    sft_usd = max(sft_usd, r.get("usd_train", 0.0))
                rl_usd = max(rl_usd, r.get("cost/cumulative_usd", 0.0))
            if sft_usd:
                rows.append({"file": rel, "kind": "sft_train", "usd": sft_usd})
            if rl_usd:
                rows.append({"file": rel, "kind": "rl_train", "usd": rl_usd})
        elif base in ("rollouts.jsonl", "checkpoints.jsonl", "val_metrics.jsonl", "timing_spans.jsonl") or "iteration_" in rel:
            continue
        else:
            c = _eval_file_cost(path)
            if c:
                rows.append({"file": rel, "kind": "eval", "usd": c[0], "n_in": c[1], "n_out": c[2]})
    total = sum(r["usd"] for r in rows)
    by_kind: dict[str, float] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0.0) + r["usd"]
    return {"budget_usd": BUDGET_USD, "total_usd": round(total, 2), "fraction": round(total / BUDGET_USD, 3),
            "stop_at_usd": BUDGET_USD * STOP_FRACTION, "by_kind": {k: round(v, 2) for k, v in by_kind.items()},
            "runs": [{**r, "usd": round(r["usd"], 3)} for r in rows]}


if __name__ == "__main__":
    t = tally()
    with open("runs/cost_tally.json", "w") as f:
        json.dump(t, f, indent=1)
    top = sorted(t["runs"], key=lambda r: -r["usd"])[:12]
    for r in top:
        print(f"  ${r['usd']:8.3f}  {r['kind']:10s} {r['file']}")
    print(f"TOTAL ${t['total_usd']:.2f} of ${t['budget_usd']:.0f} ({100 * t['fraction']:.1f}%), stop at ${t['stop_at_usd']:.0f}; by kind {t['by_kind']}")
