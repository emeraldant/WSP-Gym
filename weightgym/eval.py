"""Scoring, summaries and bootstrap confidence intervals.

Metric definitions (fixed before any model was evaluated; see NOTES.md):

  repair_rate   fraction of FAULTY episodes whose edit reaches recovery >= 0.95 within budget.
                This is the primary metric (on the in-distribution test split, both prompt
                variants pooled).
  clean_ok      fraction of CLEAN control episodes answered with the empty edit.
  valid         fraction of answers that parse as a valid, in-budget edit.
  noop_faulty   fraction of faulty episodes answered with the empty edit (collapse monitor).
  mean_reward   mean training reward over all episodes.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from .baselines import POLICIES, gradient_repair
from .env import REPAIR_THRESHOLD, Episode, score_answer


def record(ep: Episode, res: dict, **extra) -> dict:
    return {
        "seed": ep.seed, "family": ep.family, "fault": ep.fault, "variant": ep.variant, "clean": ep.is_clean,
        "n_weights": ep.net.n_weights, "reward": res["reward"], "valid": res["valid"], "success": res["success"],
        "recovery": res["recovery"], "noop": res["noop"], "n_ops": res["n_ops"], **extra,
    }


def run_policy(episodes: list[Episode], name: str, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for ep in episodes:
        if name == "gradient":
            net = gradient_repair(ep)
            acc = ep.task.accuracy(net)
            if ep.is_clean:
                res = {"reward": float("nan"), "valid": True, "success": acc >= ep.acc_clean - 0.01, "recovery": None, "noop": False, "n_ops": None}
            else:
                rec = (acc - ep.acc_faulty) / (ep.acc_clean - ep.acc_faulty)
                res = {"reward": float("nan"), "valid": True, "success": rec >= REPAIR_THRESHOLD, "recovery": rec, "noop": False, "n_ops": None}
        else:
            res = score_answer(ep, POLICIES[name](ep, rng))
        out.append(record(ep, res, policy=name))
    return out


def bootstrap_ci(values, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap interval over episodes."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a, b, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap interval for mean(a) - mean(b) with episodes resampled jointly (same episodes in both arms)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert len(a) == len(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def summarize(records: list[dict]) -> dict:
    faulty = [r for r in records if not r["clean"]]
    clean = [r for r in records if r["clean"]]
    rr, lo, hi = bootstrap_ci([r["success"] for r in faulty])
    co, clo, chi = bootstrap_ci([r["success"] for r in clean])
    return {
        "n": len(records), "n_faulty": len(faulty), "n_clean": len(clean),
        "repair_rate": rr, "repair_lo": lo, "repair_hi": hi,
        "clean_ok": co, "clean_lo": clo, "clean_hi": chi,
        "valid": float(np.mean([r["valid"] for r in records])) if records else float("nan"),
        "noop_faulty": float(np.mean([r["noop"] for r in faulty])) if faulty else float("nan"),
        "mean_reward": float(np.nanmean([r["reward"] for r in records])) if records else float("nan"),
    }


def breakdown(records: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(str(r[key]), []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
