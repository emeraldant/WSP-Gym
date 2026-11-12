"""Evaluate a Tinker proposer on the real-data improve benchmark (weightgym/realworld.py).

    python -m weightgym.real_eval eval <arm> <tinker://sampler | base> <k> real_test_seen real_test_unseen real_hf_models
    python -m weightgym.real_eval table

Each network gets ``k`` proposals (no reasoning, temperature 1.0, 256-token cap). Proposals are verified on
the network's ``select`` rows; the best is kept only if it helps there; every reported number is the change
on separate ``report`` rows. Logs go to runs/final_real/<arm>/<split>.jsonl.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys

from .eval import load_jsonl
from .realworld import load_hf_items, load_real_split, loop_from_records

OUT = "runs/final_real"
SPLITS = ("real_val", "real_test_seen", "real_test_unseen", "real_hf_models")


def items_for(split: str):
    return load_hf_items() if split == "real_hf_models" else load_real_split(split)


async def run(arm: str, model_path: str | None, k: int, splits) -> None:
    import tinker

    from .llm import INKLING, LLMPolicy, evaluate

    policy = await LLMPolicy(tinker.ServiceClient(), INKLING, model_path, 0.0, 256).start()
    for split in splits:
        await evaluate(policy, items_for(split), f"{OUT}/{arm}/{split}.jsonl", k, 64, tag=f"{arm}/{split}", direct=True)


def table(ks=(1, 8), out: str = "results/table_realworld.csv") -> None:
    rows = []
    base = {}
    if os.path.exists("results/table_realworld_baselines.csv"):
        for r in csv.DictReader(open("results/table_realworld_baselines.csv")):
            base.setdefault(r.get("split"), []).append(r)
    for split in SPLITS:
        for r in base.get(split, []):
            rows.append({"split": split, "proposer": r.get("proposer") or r.get("policy"), "n": r.get("n"),
                         "gap_closed": float(r["gap_closed"]), "gap_lo": float(r["gap_lo"]), "gap_hi": float(r["gap_hi"]),
                         "improved": float(r["improved"]), "harmed": float(r["harmed"])})
        if not os.path.isdir(OUT):
            continue
        items = None
        for arm in sorted(os.listdir(OUT)):
            path = f"{OUT}/{arm}/{split}.jsonl"
            if not os.path.exists(path):
                continue
            items = items or items_for(split)
            recs = [r for r in load_jsonl(path) if not r.get("header")]
            for k in ks:
                st = loop_from_records(items, recs, k)
                rows.append({"split": split, "proposer": f"{arm} x{k}", "n": st["n"], "gap_closed": st["gap_closed"],
                             "gap_lo": st["gap_lo"], "gap_hi": st["gap_hi"], "improved": st["improved"], "harmed": st["harmed"]})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    for r in rows:
        print(f"{r['split']:18s} {str(r['proposer']):30s} gap closed {r['gap_closed']:+.3f} [{r['gap_lo']:+.3f}, {r['gap_hi']:+.3f}] "
              f"improved {r['improved']:.2f} harmed {r['harmed']:.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        asyncio.run(run(sys.argv[2], None if sys.argv[3] == "base" else sys.argv[3], int(sys.argv[4]), sys.argv[5:]))
    else:
        table()
