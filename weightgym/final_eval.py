"""Final evaluation: every arm on every frozen test split, once.

    python -m weightgym.final_eval --arm untrained --arm sft=tinker://... --arm sft_rl=tinker://...
    python -m weightgym.final_eval --tables-only

Each (arm, split) pair writes ``runs/final/<arm>/<split>.jsonl`` with one line per rollout. A pair
whose file is already complete is never re-sampled, so a test split is evaluated exactly once per arm.
All LLM arms use the pre-registered no-reasoning settings (direct prompt, effort 0.0, temperature 1.0,
max_tokens 256, one sample per episode). ``--thinking-ref N`` additionally evaluates the *untrained*
model with reasoning allowed on the first N episodes of test_id, as a reference line.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import tinker

from .env import load_split
from .eval import breakdown, load_jsonl, paired_diff_ci, run_policy, summarize, write_csv
from .llm import INKLING, LLMPolicy, evaluate

TEST_SPLITS = ("test_id", "test_families", "test_faults", "test_larger", "test_trained", "test_external")
BASELINES = ("noop", "random", "outlier", "search", "gradient")
SETTINGS = {"effort": 0.0, "max_tokens": 256, "direct": True, "temperature": 1.0, "samples": 1}
OUT = "runs/final"


def _records(path: str) -> list[dict]:
    return [r for r in load_jsonl(path) if not r.get("header")] if os.path.exists(path) else []


async def eval_arm(service, name: str, model_path: str | None, splits, data_dir: str, concurrency: int,
                   temperature: float = 1.0) -> None:
    policy = None
    for split in splits:
        path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {split}: {path} missing")
            continue
        _, eps = load_split(path)
        out = os.path.join(OUT, name, f"{split}.jsonl")
        if len(_records(out)) >= len(eps):
            continue
        if policy is None:
            policy = await LLMPolicy(service, INKLING, model_path, SETTINGS["effort"], SETTINGS["max_tokens"], temperature).start()
        await evaluate(policy, eps, out, SETTINGS["samples"], concurrency, tag=f"{name}/{split}", direct=True)


async def eval_thinking_ref(service, n: int, data_dir: str, effort: float, max_tokens: int) -> None:
    _, eps = load_split(os.path.join(data_dir, "test_id.jsonl"))
    eps = eps[:n]
    out = os.path.join(OUT, "untrained_thinking", "test_id.jsonl")
    if len(_records(out)) >= len(eps):
        return
    policy = await LLMPolicy(service, INKLING, None, effort, max_tokens).start()
    await evaluate(policy, eps, out, 1, 48, tag="untrained_thinking/test_id", direct=False)


def eval_baselines(splits, data_dir: str) -> None:
    for split in splits:
        path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        _, eps = load_split(path)
        for pol in BASELINES:
            out = os.path.join(OUT, f"baseline_{pol}", f"{split}.jsonl")
            if os.path.exists(out):
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            recs = run_policy(eps, pol, seed=0)
            with open(out, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"[baseline] {pol}/{split}: repair {summarize(recs)['repair_rate']:.3f}")


def build_tables(results_dir: str = "results", reference_arm: str = "untrained") -> None:
    os.makedirs(results_dir, exist_ok=True)
    arms = sorted(d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d)))
    main, detail, diffs = [], [], []
    by_arm: dict[tuple[str, str], list[dict]] = {}
    for arm in arms:
        for split in TEST_SPLITS:
            recs = _records(os.path.join(OUT, arm, f"{split}.jsonl"))
            if not recs:
                continue
            by_arm[(arm, split)] = recs
            s = summarize(recs)
            tok = [r["n_out"] for r in recs if "n_out" in r]
            main.append({"arm": arm, "split": split, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()},
                         "mean_out_tokens": round(sum(tok) / len(tok), 1) if tok else ""})
            for key in ("variant", "fault", "family"):
                for val, b in breakdown(recs, key).items():
                    detail.append({"arm": arm, "split": split, "by": key, "value": val, "n": b["n"],
                                   "repair_rate": round(b["repair_rate"], 4), "repair_lo": round(b["repair_lo"], 4),
                                   "repair_hi": round(b["repair_hi"], 4), "clean_ok": round(b["clean_ok"], 4),
                                   "valid": round(b["valid"], 4), "noop_faulty": round(b["noop_faulty"], 4)})
    for (arm, split), recs in by_arm.items():
        ref = by_arm.get((reference_arm, split))
        if arm == reference_arm or ref is None or arm.startswith("baseline_") or arm == "untrained_thinking":
            continue
        a = {r["seed"]: r for r in recs if not r["clean"]}
        b = {r["seed"]: r for r in ref if not r["clean"]}
        seeds = sorted(set(a) & set(b))
        d, lo, hi = paired_diff_ci([a[s]["success"] for s in seeds], [b[s]["success"] for s in seeds])
        diffs.append({"arm": arm, "vs": reference_arm, "split": split, "n_faulty": len(seeds), "diff": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)})
    # Each no-reasoning arm against the untrained model *with* reasoning, paired on the networks the
    # reasoning-mode reference was run on (the first --thinking-ref networks of test_id).
    vs_reasoning = []
    think = {r["seed"]: r for r in by_arm.get(("untrained_thinking", "test_id"), []) if not r["clean"]}
    for (arm, split), recs in by_arm.items():
        if split != "test_id" or not think or arm.startswith(("baseline_", "untrained", "frontier_")):
            continue
        a = {r["seed"]: r for r in recs if not r["clean"]}
        seeds = sorted(set(a) & set(think))
        if not seeds:
            continue
        d, lo, hi = paired_diff_ci([a[s]["success"] for s in seeds], [think[s]["success"] for s in seeds])
        vs_reasoning.append({"arm": arm, "vs": "untrained_thinking", "split": split, "n_faulty": len(seeds),
                             "arm_rate": round(sum(a[s]["success"] for s in seeds) / len(seeds), 4),
                             "reasoning_rate": round(sum(think[s]["success"] for s in seeds) / len(seeds), 4),
                             "diff": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)})
    write_csv(os.path.join(results_dir, "table_vs_reasoning.csv"), vs_reasoning)
    write_csv(os.path.join(results_dir, "table_main.csv"), main)
    write_csv(os.path.join(results_dir, "table_breakdown.csv"), detail)
    write_csv(os.path.join(results_dir, "table_paired_diff.csv"), diffs)
    for r in main:
        print(f"{r['arm']:22s} {r['split']:14s} repair {r['repair_rate']:.3f} [{r['repair_lo']:.3f}, {r['repair_hi']:.3f}] "
              f"clean_ok {r['clean_ok']:.2f} valid {r['valid']:.2f} noop_f {r['noop_faulty']:.2f} n_faulty {r['n_faulty']}")
    for r in diffs:
        print(f"diff {r['arm']} - {r['vs']} on {r['split']}: {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    for r in vs_reasoning:
        print(f"vs reasoning: {r['arm']} {r['arm_rate']:.3f} - untrained with reasoning {r['reasoning_rate']:.3f} on "
              f"{r['n_faulty']} shared networks: {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")


async def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[], help="name or name=tinker://sampler_path ('untrained' = base model)")
    ap.add_argument("--splits", default=",".join(TEST_SPLITS))
    ap.add_argument("--data", default="data")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--thinking-ref", type=int, default=0)
    ap.add_argument("--thinking-effort", type=float, default=0.1)
    ap.add_argument("--thinking-max-tokens", type=int, default=20000)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 is the pre-registered setting; use 0 only for arms named *_greedy (secondary)")
    a = ap.parse_args()
    splits = tuple(a.splits.split(","))
    if not a.tables_only:
        if a.baselines:
            eval_baselines(splits, a.data)
        service = tinker.ServiceClient() if (a.arm or a.thinking_ref) else None
        jobs = []
        for spec in a.arm:
            name, _, path = spec.partition("=")
            jobs.append(eval_arm(service, name, path or None, splits, a.data, a.concurrency, a.temperature))
        if a.thinking_ref:
            jobs.append(eval_thinking_ref(service, a.thinking_ref, a.data, a.thinking_effort, a.thinking_max_tokens))
        await asyncio.gather(*jobs)
    build_tables()


if __name__ == "__main__":
    asyncio.run(_main())
