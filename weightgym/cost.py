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
# The scale-up run (NOTES.md, "Pre-registration: scale-up") has its own budget, tallied separately.
SCALE_UP_BUDGET_USD = 300.0
SCALE_UP_PREFIXES = ("sft_scale", "rl_scale", "final/scale", "diag/scale", "final_verify")


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
            cfg_path = os.path.join(os.path.dirname(path), "config.json")
            sft_model = "thinkingmachines/Inkling-2-Small"
            if os.path.exists(cfg_path):
                try:
                    sft_model = json.load(open(cfg_path)).get("model", sft_model)
                except (json.JSONDecodeError, AttributeError):
                    pass
            for line in open(path):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "train":  # sum per-step tokens: survives resumes, counts re-run steps
                    sft_usd += r.get("tokens", 0) * PRICE[sft_model]["train"] / 1e6
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
    for r in rows:
        r["phase"] = "scale_up" if r["file"].startswith(SCALE_UP_PREFIXES) else "main"
    total = sum(r["usd"] for r in rows if r["phase"] == "main")
    scale = sum(r["usd"] for r in rows if r["phase"] == "scale_up")
    by_kind: dict[str, float] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0.0) + r["usd"]
    return {"budget_usd": BUDGET_USD, "total_usd": round(total, 2), "fraction": round(total / BUDGET_USD, 3),
            "stop_at_usd": BUDGET_USD * STOP_FRACTION,
            "scale_up": {"budget_usd": SCALE_UP_BUDGET_USD, "total_usd": round(scale, 2),
                         "fraction": round(scale / SCALE_UP_BUDGET_USD, 3), "stop_at_usd": SCALE_UP_BUDGET_USD * STOP_FRACTION},
            "grand_total_usd": round(total + scale, 2),
            "by_kind": {k: round(v, 2) for k, v in by_kind.items()},
            "runs": [{**r, "usd": round(r["usd"], 3)} for r in rows]}


if __name__ == "__main__":
    t = tally()
    with open("runs/cost_tally.json", "w") as f:
        json.dump(t, f, indent=1)
    top = sorted(t["runs"], key=lambda r: -r["usd"])[:12]
    for r in top:
        print(f"  ${r['usd']:8.3f}  {r['kind']:10s} {r['file']}")
    print(f"TOTAL ${t['total_usd']:.2f} of ${t['budget_usd']:.0f} ({100 * t['fraction']:.1f}%), stop at ${t['stop_at_usd']:.0f}; by kind {t['by_kind']}")
    su = t["scale_up"]
    print(f"SCALE-UP ${su['total_usd']:.2f} of ${su['budget_usd']:.0f} ({100 * su['fraction']:.1f}%), stop at ${su['stop_at_usd']:.0f}; "
          f"grand total ${t['grand_total_usd']:.2f}")
