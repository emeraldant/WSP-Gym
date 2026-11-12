"""CPU-only baselines for the backdoor-audit task (weightgym/audit.py).

Every baseline is reported per split as

    detection    fraction of BACKDOORED networks it flags (answers with a non-empty edit / finds a failing input)
    false_alarm  fraction of CLEAN networks it flags
    removal      fraction of backdoored networks whose backdoor it removes (recovery >= 0.95 on the full domain)
    reward       mean ``score_answer`` reward over all networks (edit policies only)

Baselines:

  noop               always answers []
  random_edit        one random valid edit (baselines.random_policy)
  stat_outlier       statistics only: baselines.outlier_policy adapted to zero the outgoing column of the most
                     unusual hidden neuron (robust z-score of gain, bias ratio, column norm, fan-in), if |z| > 2.5
  stat_always        the same, always editing (threshold 0)
  stat_informed      upper bound for magnitude statistics on compiled networks: knows what a detector looks
                     like up to signs (wide fan-in, large column) and zeroes one such neuron at random
  random_tests_q     black-box testing with q fresh uniformly random inputs: detection only
  search_tests       brute-force single-op search (zero a column, sign flips, column swaps) scored with forward
                     passes on the 32 passing test inputs only: it never sees a failure
  search_oracle      the same search scored on the full domain (it sees the trigger inputs)
  finetune           gradient fine-tuning (Adam) on the non-trigger inputs only; ``survival`` is the fraction of
                     the backdoor's wrong answers that are still wrong afterwards. Not an edit: ignores the budget.
  weight_rule        hand-written weight reading that knows the compiled construction: a neuron is a detector if
                     it reads >= 5 inputs, can fire (sum of positive incoming weights + bias > 0) only when all
                     of them agree, and its firing moves the output by a non-negligible amount

    python -m weightgym.audit_baselines --splits test_audit,test_audit_sgd
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .audit import answers, trigger_mask
from .baselines import _robust_z, gradient_repair, random_policy, single_op_candidates
from .edits import apply_edit
from .env import REPAIR_THRESHOLD, Episode, load_split, score_answer

STAT_FEATURES = ("gain", "bias", "col", "fanin")
FINETUNE_STEPS = 300


def zero_col(j: int) -> dict:
    return {"op": "scale", "tensor": "W2", "axis": "col", "index": int(j), "factor": 0.0}


def hidden_stats(net) -> dict[str, np.ndarray]:
    """Sign-agnostic per-neuron statistics of the hidden layer."""
    W, b, Wn = net.Ws[0], net.bs[0], net.Ws[1]
    rn = np.linalg.norm(W, axis=1) + 1e-6
    cn = np.linalg.norm(Wn, axis=0) + 1e-6
    strong = np.abs(W) > 0.25 * np.maximum(np.abs(W).max(1, keepdims=True), 1e-9)
    return {"gain": np.log(rn * cn), "bias": b / rn, "col": np.log(cn), "row": np.log(rn),
            "fanin": strong.sum(1).astype(float), "cn": cn}


# ----------------------------------------------------------------------------
# Edit policies: Episode -> answer text
# ----------------------------------------------------------------------------


def noop_policy(ep: Episode, rng=None) -> str:
    return "[]"


def stat_policy(ep: Episode, rng=None, threshold: float = 2.5, features=STAT_FEATURES) -> str:
    """Zero the outgoing column of the most statistically unusual hidden neuron (largest robust |z| over the
    given features among neurons with a non-negligible column), or answer [] when nothing exceeds ``threshold``."""
    st = hidden_stats(ep.net)
    live = st["cn"] > 0.1
    if live.sum() < 4:
        return "[]"
    idx = np.flatnonzero(live)
    best = (threshold, None)
    for name in features:
        z = np.abs(_robust_z(st[name][live]))
        k = int(np.argmax(z))
        if z[k] > best[0]:
            best = (float(z[k]), int(idx[k]))
    return json.dumps([zero_col(best[1])] if best[1] is not None else [])


def stat_always_policy(ep: Episode, rng=None) -> str:
    return stat_policy(ep, rng, threshold=0.0)


def stat_informed_policy(ep: Episode, rng: np.random.Generator) -> str:
    """Pick one neuron at random among those with a detector's magnitudes: wide fan-in and a large column."""
    st = hidden_stats(ep.net)
    suspects = np.flatnonzero((st["fanin"] >= 5) & (st["cn"] > 0.2))
    return json.dumps([zero_col(int(rng.choice(suspects)))] if len(suspects) else [])


def _search(ep: Episode, visible: np.ndarray) -> str:
    task, net = ep.task, ep.net
    cands = [zero_col(j) for j in range(net.Ws[0].shape[0])] + single_op_candidates(net)
    best_acc, best = float(task.correct(net)[visible].mean()), []
    for op in cands:
        acc = float(task.correct(apply_edit(net, [op]))[visible].mean())
        if acc > best_acc + 1e-9:
            best_acc, best = acc, [op]
    return json.dumps(best)


def search_tests_policy(ep: Episode, rng=None) -> str:
    return _search(ep, np.array(ep.extras["tests"]))


def search_oracle_policy(ep: Episode, rng=None) -> str:
    return _search(ep, np.arange(len(ep.task.X)))


def weight_rule_policy(ep: Episode, rng=None) -> str:
    net = ep.net
    W, b, Wn = net.Ws[0], net.bs[0], net.Ws[1]
    best = (0.0, None)
    for j in range(W.shape[0]):
        mag = np.abs(W[j])
        strong = mag > 0.25 * max(mag.max(), 1e-9)
        peak = float(np.maximum(W[j], 0).sum() + b[j])  # largest pre-activation over 0/1 inputs
        if strong.sum() < 5 or peak <= 0 or peak - float(mag[strong].min()) > 0:
            continue  # narrow, dead, or still firing with one of its literals violated
        effect = peak * float(np.abs(Wn[:, j]).max())
        if effect > max(best[0], 0.25):
            best = (effect, j)
    return json.dumps([zero_col(best[1])] if best[1] is not None else [])


EDIT_POLICIES = {
    "noop": noop_policy, "random_edit": random_policy, "stat_outlier": stat_policy, "stat_always": stat_always_policy,
    "stat_informed": stat_informed_policy, "search_tests": search_tests_policy, "search_oracle": search_oracle_policy,
    "weight_rule": weight_rule_policy,
}
RANDOM_TEST_QUERIES = (32, 256, 1024)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------


def run_edit_policy(episodes: list[Episode], name: str, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for ep in episodes:
        res = score_answer(ep, EDIT_POLICIES[name](ep, rng))
        out.append({"seed": ep.seed, "clean": ep.is_clean, "flag": bool(res["valid"] and not res["noop"]),
                    "success": bool(res["success"]), "reward": res["reward"]})
    return out


def run_random_tests(episodes: list[Episode], q: int, seed: int = 0) -> list[dict]:
    """q inputs drawn uniformly with replacement; the network is flagged when it gets any of them wrong."""
    rng = np.random.default_rng(seed)
    out = []
    for ep in episodes:
        wrong = ~ep.task.correct(ep.net)
        hit = bool(wrong[rng.integers(0, len(wrong), size=q)].any())
        out.append({"seed": ep.seed, "clean": ep.is_clean, "flag": hit, "success": False, "reward": float("nan")})
    return out


def run_finetune(episodes: list[Episode], steps: int = FINETUNE_STEPS) -> list[dict]:
    """Fine-tune on the correct labels of every non-trigger input (all inputs for a clean network)."""
    out = []
    for ep in episodes:
        task = ep.task
        if ep.is_clean:
            idx = np.arange(len(task.X))
        else:
            mask = trigger_mask(task.X, ep.extras["trigger"]["literals"])
            idx = np.flatnonzero(~mask)
        tuned = gradient_repair(ep, steps=steps, idx=idx)
        acc = task.accuracy(tuned)
        row = {"seed": ep.seed, "clean": ep.is_clean, "flag": True, "reward": float("nan")}
        if ep.is_clean:
            row.update(success=bool(acc >= ep.acc_clean - 1e-9), survival=float("nan"))
        else:
            was_wrong = ~task.correct(ep.net) & mask
            still = ~task.correct(tuned) & was_wrong
            rec = (acc - ep.acc_faulty) / (ep.acc_clean - ep.acc_faulty)
            row.update(success=bool(rec >= REPAIR_THRESHOLD), survival=float(still.sum() / was_wrong.sum()))
        out.append(row)
    return out


def summarize(rows: list[dict]) -> dict:
    bd = [r for r in rows if not r["clean"]]
    cl = [r for r in rows if r["clean"]]
    mean = lambda v: float(np.mean(v)) if len(v) else float("nan")
    s = {
        "n_backdoored": len(bd), "n_clean": len(cl),
        "detection": mean([r["flag"] for r in bd]), "false_alarm": mean([r["flag"] for r in cl]),
        "removal": mean([r["success"] for r in bd]), "clean_ok": mean([r["success"] for r in cl]),
        "reward": mean([r["reward"] for r in rows]),
    }
    if rows and "survival" in rows[0]:
        s["survival"] = mean([r["survival"] for r in bd])
    return s


def run_all(episodes: list[Episode], seed: int = 0, finetune: bool = True) -> dict[str, dict]:
    table = {name: summarize(run_edit_policy(episodes, name, seed)) for name in EDIT_POLICIES}
    for q in RANDOM_TEST_QUERIES:
        table[f"random_tests_{q}"] = summarize(run_random_tests(episodes, q, seed))
    if finetune:
        table["finetune"] = summarize(run_finetune(episodes))
    return table


def main() -> None:
    from .eval import write_csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits", default="test_audit,test_audit_sgd")
    ap.add_argument("--out", default="results/table_audit_baselines.csv")
    ap.add_argument("--no-finetune", action="store_true")
    a = ap.parse_args()
    rows = []
    for split in a.splits.split(","):
        _, eps = load_split(os.path.join(a.data, f"{split}.jsonl"))
        for name, s in run_all(eps, finetune=not a.no_finetune).items():
            rows.append({"split": split, "baseline": name, "survival": "", **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}})
            print(f"{split:15s} {name:17s} detection {s['detection']:.3f}  false_alarm {s['false_alarm']:.3f}  "
                  f"removal {s['removal']:.3f}  reward {s['reward']:+.3f}" + (f"  survival {s['survival']:.3f}" if "survival" in s else ""))
    if a.out:
        write_csv(a.out, rows)


if __name__ == "__main__":
    main()
