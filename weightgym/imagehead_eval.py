"""Run a Tinker proposer on the image-classifier heads (weightgym/imagehead.py).

    python -m weightgym.imagehead_eval eval <arm> <tinker://sampler | base> <k>
    python -m weightgym.imagehead_eval table

Each public image classifier's final layer gets ``k`` proposed edits (no reasoning, temperature 1.0). Proposals
are verified on validation images; the best is kept only if it helps there; the reported number is the change
in accuracy on the official test split, next to the accuracy the model card claims.
"""

from __future__ import annotations

import asyncio
import csv
import os
import sys

from .eval import load_jsonl
from .imagehead import load_items

OUT = "runs/final_imagehead"


async def run(arm: str, model_path: str | None, k: int) -> None:
    import tinker

    from .llm import INKLING, LLMPolicy, evaluate

    policy = await LLMPolicy(tinker.ServiceClient(), INKLING, model_path, 0.0, 256).start()
    await evaluate(policy, load_items(), f"{OUT}/{arm}/heads.jsonl", k, 16, tag=f"{arm}/heads", direct=True)


def table(out: str = "results/table_imagehead.csv") -> None:
    items = load_items()
    rows = []
    for arm in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
        path = f"{OUT}/{arm}/heads.jsonl"
        if not os.path.exists(path):
            continue
        by: dict[int, list] = {}
        valid: dict[int, int] = {}
        for r in load_jsonl(path):
            if r.get("header"):
                continue
            valid[r["seed"]] = valid.get(r["seed"], 0) + int(bool(r["valid"]))
            if r.get("ops") is not None:
                by.setdefault(r["seed"], []).append(r["ops"])
        for it in items:
            rep = it.verify_and_report(by.get(it.seed, []))
            rows.append({"proposer": arm, "model": it.hub_id if hasattr(it, "hub_id") else it.family, "dataset": it.family,
                         "valid_proposals": valid.get(it.seed, 0), "accepted": rep["accepted"],
                         "test_before": round(100 * rep["report_before"], 2), "test_after": round(100 * rep["report_after"], 2),
                         "change_points": round(rep["change_points"], 2), "claimed": None if rep["claimed"] is None else round(100 * rep["claimed"], 2),
                         "exceeds_claimed": rep["exceeds_claimed"], "ops": rep["ops"]})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['proposer']:20s} {str(r['model'])[:44]:44s} valid {r['valid_proposals']:2d} test {r['test_before']:6.2f} -> {r['test_after']:6.2f} "
              f"({r['change_points']:+.2f}) claimed {r['claimed']}")
    for arm in sorted({r["proposer"] for r in rows}):
        sub = [r for r in rows if r["proposer"] == arm]
        print(f"== {arm}: mean change {sum(r['change_points'] for r in sub) / len(sub):+.2f} points over {len(sub)} models; "
              f"improved {sum(r['change_points'] > 0 for r in sub)}, worse {sum(r['change_points'] < 0 for r in sub)}, unchanged {sum(r['change_points'] == 0 for r in sub)}")


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        asyncio.run(run(sys.argv[2], None if sys.argv[3] == "base" else sys.argv[3], int(sys.argv[4])))
    else:
        table()
