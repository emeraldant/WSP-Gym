"""Propose-and-verify repair: a secondary regime, never mixed into the one-sample headline numbers.

The model samples ``k`` answers for each faulty network (same no-reasoning settings as the headline).
A verifier that knows the spec runs every edited network on the scoring inputs; the network counts as
fixed if any of the first ``k`` answers reaches recovery >= 0.95. This uses forward passes, so the
fair comparisons are the forward-pass search baseline and ``k`` random valid edits, not the headline.

    python -m weightgym.verify_eval eval scale_rl tinker://<sampler> --k 8 --splits test_id,test_trained,test_external
    python -m weightgym.verify_eval table        # -> results/table_verify.csv

Rollouts go to ``runs/final_verify/<arm>/<split>.jsonl`` (one line per sampled answer); a split that
is already complete for an arm is not sampled again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import numpy as np

from .env import load_split
from .eval import bootstrap_ci, load_jsonl, run_policy, write_csv

OUT = "runs/final_verify"
SPLITS = ("test_id", "test_trained", "test_external")
KS = (1, 2, 4, 8)


def faulty_episodes(split: str, data_dir: str = "data"):
    _, eps = load_split(os.path.join(data_dir, f"{split}.jsonl"))
    return [e for e in eps if not e.is_clean]


def _records(path: str) -> list[dict]:
    return [r for r in load_jsonl(path) if not r.get("header")] if os.path.exists(path) else []


def fixed_within(records: list[dict], k: int) -> tuple[float, float, float, int]:
    """Share of networks for which one of the first ``k`` sampled answers is a repair, with a bootstrap interval."""
    by_seed: dict[int, bool] = {}
    for r in records:
        if r["sample"] < k:
            by_seed[r["seed"]] = by_seed.get(r["seed"], False) or bool(r["success"])
    rate, lo, hi = bootstrap_ci([float(v) for v in by_seed.values()])
    return rate, lo, hi, len(by_seed)


def random_baseline(split: str, k: int) -> list[dict]:
    """``k`` random valid edits per faulty network (seeds 0..k-1), logged like model samples."""
    out = os.path.join(OUT, "baseline_random", f"{split}.jsonl")
    if os.path.exists(out):
        return _records(out)
    eps = faulty_episodes(split)
    recs = []
    for j in range(k):
        for r in run_policy(eps, "random", seed=j):
            recs.append({**r, "sample": j})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return recs


async def eval_arm(name: str, model_path: str | None, splits, k: int, concurrency: int) -> None:
    import tinker

    from .llm import INKLING, LLMPolicy, evaluate

    policy = await LLMPolicy(tinker.ServiceClient(), INKLING, model_path, 0.0, 256, 1.0).start()
    for split in splits:
        eps = faulty_episodes(split)
        out = os.path.join(OUT, name, f"{split}.jsonl")
        if len(_records(out)) >= len(eps) * k:
            continue
        await evaluate(policy, eps, out, k, concurrency, tag=f"verify/{name}/{split}", direct=True)


def build_table(results_dir: str = "results") -> None:
    rows = []
    arms = sorted(d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d))) if os.path.isdir(OUT) else []
    for arm in arms:
        for split in SPLITS:
            recs = _records(os.path.join(OUT, arm, f"{split}.jsonl"))
            if not recs:
                continue
            for k in KS:
                if k > 1 + max(r["sample"] for r in recs):
                    continue
                rate, lo, hi, n = fixed_within(recs, k)
                distinct = round(float(np.mean(_distinct(recs, k))), 2) if "answer" in recs[0] else ""
                rows.append({"arm": arm, "split": split, "k": k, "n_faulty": n, "fixed": round(rate, 4),
                             "lo": round(lo, 4), "hi": round(hi, 4), "mean_distinct_answers": distinct})
    write_csv(os.path.join(results_dir, "table_verify.csv"), rows)
    for r in rows:
        print(f"{r['arm']:20s} {r['split']:14s} k={r['k']}: fixed {r['fixed']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] "
              f"n {r['n_faulty']} distinct answers {r['mean_distinct_answers']}")


def _distinct(records: list[dict], k: int) -> list[int]:
    """Number of different answers among the first ``k`` samples, per network (how much sampling explores)."""
    seen: dict[int, set] = {}
    for r in records:
        if r["sample"] < k:
            seen.setdefault(r["seed"], set()).add(json.dumps(r.get("ops"), sort_keys=True) if r.get("ops") is not None
                                                  else str(r.get("answer")))
    return [len(v) for v in seen.values()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("arm")
    e.add_argument("sampler", help="tinker:// sampler path, or 'base' for the untrained model")
    e.add_argument("--k", type=int, default=8)
    e.add_argument("--splits", default=",".join(SPLITS))
    e.add_argument("--concurrency", type=int, default=64)
    sub.add_parser("table")
    a = ap.parse_args()
    if a.cmd == "eval":
        splits = a.splits.split(",")
        for s in splits:
            random_baseline(s, a.k)
        asyncio.run(eval_arm(a.arm, None if a.sampler == "base" else a.sampler, splits, a.k, a.concurrency))
    build_table()


if __name__ == "__main__":
    main()
