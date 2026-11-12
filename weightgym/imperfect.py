"""Improving imperfect networks: weight editing with no planted fault and no answer key.

A network is trained by gradient descent with a deliberately weak recipe (few steps, little spare
width), so it only partly learns its specification: accuracy between 60% and 93% on its domain.
Nothing is planted, so nobody knows a "correct" edit. The model reads the weights and must reply
with an edit (the usual DSL, at most 3 ops) that makes the network more accurate.

    gap closed  = (acc_after - acc_before) / (1 - acc_before)      # share of the errors removed
    reward      = gap closed, clipped to [-0.25, 1]; the empty edit scores 0; invalid answers -1
    improved    = accuracy rises by at least 2 points;  harmed = falls by at least 2 points
    fixed       = gap closed >= 0.95

Because there are no labels, the only training signal available for this task is the reward.
``ImperfectItem`` wraps an ``Episode`` so the sampling, RL and evaluation code can use it unchanged.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .edits import MAX_OPS
from .env import DIRECT_SUFFIX, OPS_HELP, Episode, _architecture, _example, render_weights, score_answer
from .nets import sample_task
from .trained import _fit

ACC_RANGE = (0.60, 0.93)
STEP_CHOICES = (40, 80, 150, 300, 600)
MAX_FAILURES = 8
IMPROVED, HARMED = 0.02, -0.02


def make_imperfect_episode(seed: int, families: tuple[str, ...], sizes: tuple[int, ...] = (0, 1), max_tries: int = 400) -> Episode:
    """A genuinely trained, partly wrong network. Deterministic in ``seed``."""
    rng = np.random.default_rng([int(seed), 104729])
    family = str(rng.choice(families))
    variant = str(rng.choice(["weights", "weights+failures"]))
    size = int(sizes[seed % len(sizes)])
    for _ in range(max_tries):
        task, canon = sample_task(family, rng, size)
        hidden = [w + int(rng.integers(0, 3)) for w in canon.widths[1:-1]]
        steps = int(rng.choice(STEP_CHOICES))
        net = _fit(task, [task.n_in] + hidden + [task.n_out], np.random.default_rng(int(rng.integers(1 << 62))), steps=steps).round_()
        acc = task.accuracy(net)
        if not ACC_RANGE[0] <= acc <= ACC_RANGE[1]:
            continue
        failures: list[int] = []
        if variant == "weights+failures":
            wrong = np.flatnonzero(~task.correct(net))
            failures = sorted(int(i) for i in rng.choice(wrong, size=min(MAX_FAILURES, len(wrong)), replace=False))
        return Episode(seed, family, variant, "imperfect", task, net.copy(), net, 1.0, acc, [], {"steps": steps}, failures, "imperfect")
    raise RuntimeError(f"could not train an imperfect network for seed {seed} ({family})")


class ImperfectItem:
    """Duck-types the parts of ``Episode`` that sampling, RL and evaluation use."""

    is_clean = False
    fault = "imperfect"

    def __init__(self, ep: Episode, loop: bool = False):
        self.ep, self.seed, self.family, self.variant, self.task, self.net, self.source = ep, ep.seed, ep.family, ep.variant, ep.task, ep.net, ep.source
        # loop=True: the answer is one proposal in a propose-and-verify loop. It is checked with forward passes
        # and discarded if it does not help, so a valid proposal is never scored below zero.
        self.loop = loop

    def prompt(self, direct: bool = False) -> str:
        ep, task, net = self.ep, self.task, self.net
        s = [
            "You are improving a small ReLU network by editing its weights. The network was trained by gradient "
            "descent on the specification below but only partly learned it: it is currently right on about "
            f"{100 * ep.acc_faulty:.0f}% of inputs. Nothing was deliberately broken, so there may be no single culprit.",
            f"## Specification\nInputs: {task.input_desc}\nTarget: {task.spec}\nOutputs: {task.output_desc}",
            f"## Architecture\n{_architecture(net)}",
            f"## Weights\n{render_weights(net)}",
        ]
        if ep.variant == "weights+failures" and ep.failures:
            s.append("## Failing inputs\nSome of the inputs it gets wrong:\n" + "\n".join(_example(task, net, i) for i in ep.failures))
        s.append(
            "## Your task\n"
            f"Reply with a JSON list of at most {MAX_OPS} edit operations that makes the network more accurate on the "
            "specification. Each operation changes one row, one column or one scalar. Indices are 0-based. "
            "Allowed operations (examples):\n"
            f"{OPS_HELP}\n"
            + ("Your edit is a proposal: it will be tested with forward passes and kept only if it makes the network "
               "more accurate, so propose the change you think is most likely to help. It is scored by the share of the "
               "network's errors it removes (zero if it does not help); an invalid or over-budget answer is penalised. "
               if self.loop else
               "Your edit is scored by the share of the network's errors it removes, measured on held-out inputs; an edit "
               "that makes the network worse is penalised, an invalid or over-budget answer more so, and [] leaves it "
               "unchanged for a score of zero. ")
            + "End your reply with the JSON list and nothing after it."
            + (DIRECT_SUFFIX if direct else "")
        )
        return "\n\n".join(s)

    def prompt_hash(self, direct: bool = False) -> str:
        return hashlib.sha256(self.prompt(direct).encode()).hexdigest()[:16]

    def target_text(self) -> str:
        """Warm-start target: the search teacher's verified edit stored on the episode ([] if it found none)."""
        import json

        return json.dumps(self.ep.inverse_edit)

    def score(self, text: str) -> dict:
        res = score_answer(self.ep, text)  # recovery = share of the gap to 100% that the edit closes
        if res["valid"]:
            res["gain"] = res["acc_after"] - self.ep.acc_faulty
            if self.loop:
                res["reward"] = max(0.0, res["reward"])
        return res


def wrap(ep, loop: bool = False) -> ImperfectItem:
    return ep if isinstance(ep, ImperfectItem) else ImperfectItem(ep, loop)


def improve_stats(records: list[dict], n_boot: int = 10000, seed: int = 0) -> dict:
    """Summary for this task. ``recovery`` in a record is the gap closed (None for an invalid answer, counted as 0
    gap closed here and reported separately as invalid)."""
    gap = np.array([r["recovery"] if r["recovery"] is not None else 0.0 for r in records], dtype=float)
    gap = np.clip(gap, -1.0, 1.0)
    rng = np.random.default_rng(seed)
    boot = gap[rng.integers(0, len(gap), size=(n_boot, len(gap)))].mean(1) if len(gap) else np.array([np.nan])
    return {
        "n": len(records), "gap_closed": float(gap.mean()) if len(gap) else float("nan"),
        "gap_lo": float(np.quantile(boot, 0.025)), "gap_hi": float(np.quantile(boot, 0.975)),
        "improved": float(np.mean([r.get("gain") is not None and r["gain"] >= IMPROVED for r in records])),
        "harmed": float(np.mean([r.get("gain") is not None and r["gain"] <= HARMED for r in records])),
        "fixed": float(np.mean([bool(r["success"]) for r in records])),
        "noop": float(np.mean([bool(r["noop"]) for r in records])),
        "valid": float(np.mean([bool(r["valid"]) for r in records])),
        "mean_reward": float(np.mean([r["reward"] for r in records])),
    }


# ----------------------------------------------------------------------------------------------
# Splits (own seed ranges, disjoint from every other split), baselines, evaluation CLI
# ----------------------------------------------------------------------------------------------

from .env import HELDOUT_FAMILIES, TRAIN_FAMILIES, load_split, save_split  # noqa: E402

IMPERFECT_SPLITS = {
    "train_imperfect": (40_000_000, 0, TRAIN_FAMILIES),
    "val_imperfect": (41_000_000, 300, TRAIN_FAMILIES),
    "test_imperfect": (42_000_000, 500, TRAIN_FAMILIES),
    "test_imperfect_families": (43_000_000, 300, HELDOUT_FAMILIES),
}


def imperfect_episode_for(split: str, i: int) -> Episode:
    seed0, _, families = IMPERFECT_SPLITS[split]
    return make_imperfect_episode(seed0 + i, families)


def write_splits(out_dir: str = "data") -> None:
    from .env import GymConfig

    for name, (_, n, families) in IMPERFECT_SPLITS.items():
        if n:
            eps = [imperfect_episode_for(name, i) for i in range(n)]
            save_split(f"{out_dir}/{name}.jsonl", eps, GymConfig(families=families, source="imperfect"), name)
            acc = [e.acc_faulty for e in eps]
            print(f"{name}: n={n} accuracy before: mean {np.mean(acc):.3f} min {min(acc):.2f} max {max(acc):.2f}; "
                  f"weights median {int(np.median([e.net.n_weights for e in eps]))}")


def load_items(split: str, n: int | None = None) -> list[ImperfectItem]:
    _, eps = load_split(f"data/{split}.jsonl")
    return [wrap(e) for e in (eps[:n] if n else eps)]


def sparse_oracle(ep: Episode, n_ops: int = MAX_OPS) -> str:
    """Headroom estimate: fit the network by gradient descent, then greedily copy over the (at most
    ``n_ops``) rows, columns or biases of that solution that help most. Uses labels and gradients, so it
    is an oracle for what a 3-op edit can reach, not a baseline a weight reader could run."""
    import json

    from .baselines import gradient_repair
    from .edits import apply_edit

    target, cur, ops = gradient_repair(ep, steps=2000, lr=0.01), ep.net, []
    for _ in range(n_ops):
        best_acc, best = ep.task.accuracy(cur), None
        for l in range(cur.n_layers):
            d = target.Ws[l] - cur.Ws[l]
            cands = [{"op": "add", "tensor": f"W{l + 1}", "axis": "row", "index": i, "vector": d[i].round(2).tolist()} for i in range(d.shape[0])]
            cands += [{"op": "add", "tensor": f"W{l + 1}", "axis": "col", "index": j, "vector": d[:, j].round(2).tolist()} for j in range(d.shape[1])]
            cands += [{"op": "set", "tensor": f"b{l + 1}", "index": i, "value": round(float(target.bs[l][i]), 2)} for i in range(len(cur.bs[l]))]
            for op in cands:
                acc = ep.task.accuracy(apply_edit(cur, [op]))
                if acc > best_acc + 1e-9:
                    best_acc, best = acc, op
        if best is None:
            break
        ops.append(best)
        cur = apply_edit(cur, [best])
    return json.dumps(ops)


def run_baselines(split: str, out_dir: str = "runs/final_improve") -> None:
    import json
    import os

    from .baselines import POLICIES, gradient_repair
    from .eval import record

    items = load_items(split)
    rng = np.random.default_rng(0)
    for name in ("noop", "random", "outlier", "search", "sparse_oracle", "gradient"):
        recs = []
        for it in items:
            if name == "gradient":
                acc = it.task.accuracy(gradient_repair(it.ep, steps=2000, lr=0.01))
                gap = (acc - it.ep.acc_faulty) / (1 - it.ep.acc_faulty)
                res = {"reward": float(np.clip(gap, -0.25, 1)), "valid": True, "success": gap >= 0.95, "recovery": gap,
                       "noop": False, "n_ops": None, "gain": acc - it.ep.acc_faulty}
            else:
                res = it.score(sparse_oracle(it.ep) if name == "sparse_oracle" else POLICIES[name](it.ep, rng))
            recs.append(record(it, res, policy=name))
        path = f"{out_dir}/baseline_{name}/{split}.jsonl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
        s = improve_stats(recs)
        print(f"{split} {name:14s} gap closed {s['gap_closed']:+.3f} [{s['gap_lo']:+.3f}, {s['gap_hi']:+.3f}] improved {s['improved']:.2f} "
              f"harmed {s['harmed']:.2f} fixed {s['fixed']:.2f}")



# ----------------------------------------------------------------------------------------------
# Search teacher: verified improving edits found with forward passes (used only as a warm start)
# ----------------------------------------------------------------------------------------------


def _candidates(net) -> list[dict]:
    ops = []
    for l in range(net.n_layers):
        W, b = net.Ws[l], net.bs[l]
        t = f"W{l + 1}"
        for i in range(W.shape[0]):
            ops += [{"op": "scale", "tensor": t, "axis": "row", "index": i, "factor": f} for f in (-1.0, 0.5, 2.0)]
            unit = max(0.5, float(np.abs(W[i]).mean()))
            ops += [{"op": "set", "tensor": f"b{l + 1}", "index": i, "value": round(float(b[i] + d * unit), 2)} for d in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)]
        if l > 0:
            for j in range(W.shape[1]):
                ops += [{"op": "scale", "tensor": t, "axis": "col", "index": j, "factor": f} for f in (-1.0, 0.0, 0.5, 2.0)]
            ops += [{"op": "swap", "tensor": t, "axis": "col", "i": i, "j": j} for i in range(W.shape[1]) for j in range(i + 1, W.shape[1])]
    return ops


def search_teacher(ep: Episode, n_ops: int = MAX_OPS, min_step: float = 0.01) -> tuple[list[dict], float]:
    """Greedy search over simple structured edits, scored with forward passes on the scoring inputs.
    Returns (edit, accuracy gain). The edit is [] when nothing gains at least ``min_step``."""
    from .edits import apply_edit

    cur, ops, acc0 = ep.net, [], ep.acc_faulty
    acc = acc0
    for _ in range(n_ops):
        best_acc, best = acc, None
        for op in _candidates(cur):
            a = ep.task.accuracy(apply_edit(cur, [op]))
            if a > best_acc + 1e-9:
                best_acc, best = a, op
        if best is None or best_acc - acc < min_step:
            break
        ops.append(best)
        cur, acc = apply_edit(cur, [best]), best_acc
    return ops, acc - acc0


def _teach(i: int):
    import json

    ep = imperfect_episode_for("train_imperfect", i)
    ops, gain = search_teacher(ep)
    d = ep.to_json()
    d["inverse_edit"] = ops if gain >= IMPROVED else []
    d["fault_info"] = {**d["fault_info"], "teacher_gain": gain}
    return i, json.dumps(d)


def build_teacher_cache(start: int, n: int, out: str) -> None:
    import json
    from multiprocessing import Pool

    with Pool(3) as pool, open(out, "a") as f:
        for i, line in pool.imap(_teach, range(start, start + n), chunksize=8):
            f.write(json.dumps({"index": i}) + "\t" + line + "\n")



# ----------------------------------------------------------------------------------------------
# Propose-and-verify: K proposals per network, selected on one half of the inputs, scored on the other
# ----------------------------------------------------------------------------------------------


EXACT = True  # verify on the whole input domain (the specification defines every label exactly)


def _halves(ep: Episode) -> tuple[np.ndarray, np.ndarray]:
    """Inputs used to select a proposal and inputs used to score it. With ``EXACT`` both are the whole
    domain: the gym knows the target function, so verification is exact and there is nothing to overfit.
    With ``EXACT = False`` the domain is split in two, which on these tiny domains (often 32-128 inputs)
    makes every search method, including exhaustive search, overfit the selection half."""
    idx = np.random.default_rng([int(ep.seed), 271828]).permutation(len(ep.task.X))
    if EXACT:
        return idx, idx
    return idx[: len(idx) // 2], idx[len(idx) // 2 :]


def _acc(ep: Episode, net, idx: np.ndarray) -> float:
    return float(np.mean(ep.task.correct(net)[idx]))


def verify_and_score(ep: Episode, proposals: list[list[dict]]) -> dict:
    """Keep the proposal with the highest accuracy on the selection half, if it beats the unedited network
    there; report the change on the held-out half. Proposals are validated edits (lists of ops)."""
    from .edits import EditError, apply_edit, validate_edit

    sel, held = _halves(ep)
    base_sel, base_held = _acc(ep, ep.net, sel), _acc(ep, ep.net, held)
    best_net, best_sel = ep.net, base_sel
    for ops in proposals:
        try:
            cand = apply_edit(ep.net, validate_edit(ops, ep.net))
        except EditError:
            continue
        a = _acc(ep, cand, sel)
        if a > best_sel + 1e-9:
            best_net, best_sel = cand, a
    after = _acc(ep, best_net, held)
    gain = after - base_held
    return {"gain": gain, "gap": gain / (1 - base_held) if base_held < 1 else 0.0, "accepted": best_net is not ep.net}


def loop_stats(rows: list[dict], n_boot: int = 10000) -> dict:
    gap = np.clip(np.array([r["gap"] for r in rows], dtype=float), -1, 1)
    boot = gap[np.random.default_rng(0).integers(0, len(gap), size=(n_boot, len(gap)))].mean(1)
    return {"n": len(rows), "gap_closed": float(gap.mean()), "gap_lo": float(np.quantile(boot, 0.025)),
            "gap_hi": float(np.quantile(boot, 0.975)), "improved": float(np.mean([r["gain"] >= IMPROVED for r in rows])),
            "harmed": float(np.mean([r["gain"] <= HARMED for r in rows])), "accepted": float(np.mean([r["accepted"] for r in rows]))}


def loop_from_records(items: list[ImperfectItem], records: list[dict], k: int) -> dict:
    """Propose-and-verify with the first ``k`` sampled answers per network from an evaluation log."""
    by: dict[int, list] = {}
    for r in sorted(records, key=lambda r: r.get("sample", 0)):
        if r.get("ops") is not None and len(by.setdefault(r["seed"], [])) < k and r.get("sample", 0) < k:
            by[r["seed"]].append(r["ops"])
    return loop_stats([verify_and_score(it.ep, by.get(it.seed, [])) for it in items])


def loop_baselines(items: list[ImperfectItem], ks=(1, 8, 32)) -> dict:
    """Random proposals, random draws from the teacher's candidate list, and the full teacher search run on
    the selection half only (so every method sees the same information)."""
    import json

    from .baselines import random_policy

    out = {}
    rng = np.random.default_rng(0)
    for k in ks:
        out[f"random x{k}"] = loop_stats([verify_and_score(it.ep, [json.loads(random_policy(it.ep, rng)) for _ in range(k)]) for it in items])
        rows = []
        for it in items:
            cands = _candidates(it.ep.net)
            rows.append(verify_and_score(it.ep, [[cands[i]] for i in rng.choice(len(cands), size=min(k, len(cands)), replace=False)]))
        out[f"random simple edit x{k}"] = loop_stats(rows)
    rows = []
    for it in items:
        sel, _ = _halves(it.ep)
        sub = Episode(it.ep.seed, it.ep.family, it.ep.variant, it.ep.fault, _subtask(it.ep.task, sel), it.ep.clean, it.ep.net, 1.0,
                      _acc(it.ep, it.ep.net, sel), [], {}, [], it.ep.source)
        rows.append(verify_and_score(it.ep, [search_teacher(sub)[0]]))
    out["full search (hundreds of forward passes)"] = loop_stats(rows)
    return out


def _subtask(task, idx):
    from dataclasses import replace

    return replace(task, X=task.X[idx], Y=task.Y[idx])



# ----------------------------------------------------------------------------------------------
# Final evaluation of the verify loop and its table
# ----------------------------------------------------------------------------------------------

FINAL_DIR = "runs/final_improve"


async def loop_eval(arm: str, model_path: str | None, split: str, k: int = 8) -> None:
    """Sample ``k`` proposals per network from a Tinker model (None = untrained base model) and log them."""
    import tinker

    from .llm import INKLING, LLMPolicy, evaluate

    items = [wrap(it.ep, loop=True) for it in load_items(split)]
    policy = await LLMPolicy(tinker.ServiceClient(), INKLING, model_path, 0.0, 256).start()
    await evaluate(policy, items, f"{FINAL_DIR}/{arm}/{split}.jsonl", k, 64, tag=f"{arm}/{split}", direct=True)


def loop_table(splits=("test_imperfect", "test_imperfect_families"), ks=(1, 8), out: str = "results/table_improve.csv") -> None:
    import json
    import os

    from .eval import load_jsonl, write_csv

    rows = []
    for split in splits:
        if not os.path.exists(f"data/{split}.jsonl"):
            continue
        items = load_items(split)
        cache = f"{FINAL_DIR}/baselines_{split}.json"
        if not os.path.exists(cache):
            os.makedirs(FINAL_DIR, exist_ok=True)
            json.dump(loop_baselines(items, ks=(1, 8, 32, 128)), open(cache, "w"), indent=1)
        for name, st in json.load(open(cache)).items():
            rows.append({"split": split, "proposer": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()}})
        for arm in sorted(d for d in os.listdir(FINAL_DIR) if os.path.isdir(f"{FINAL_DIR}/{d}")):
            path = f"{FINAL_DIR}/{arm}/{split}.jsonl"
            if not os.path.exists(path):
                continue
            recs = [r for r in load_jsonl(path) if not r.get("header")]
            if not recs or "sample" not in recs[0]:
                continue
            for k in ks:
                st = loop_from_records(items, recs, k)
                rows.append({"split": split, "proposer": f"{arm} x{k}", **{kk: round(v, 4) if isinstance(v, float) else v for kk, v in st.items()}})
    write_csv(out, rows)
    for r in rows:
        print(f"{r['split']:24s} {r['proposer']:44s} gap closed {r['gap_closed']:+.3f} [{r['gap_lo']:+.3f}, {r['gap_hi']:+.3f}] improved {r['improved']:.2f}")


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "teach":
        build_teacher_cache(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == "splits":
        write_splits()
    elif sys.argv[1] == "baselines":
        for sp in sys.argv[2:]:
            run_baselines(sp)
    elif sys.argv[1] == "eval":  # eval <arm> <sampler path or "base"> <k> <split> [<split> ...]
        import asyncio

        for sp in sys.argv[5:]:
            asyncio.run(loop_eval(sys.argv[2], None if sys.argv[3] == "base" else sys.argv[3], sp, int(sys.argv[4])))
    elif sys.argv[1] == "table":
        loop_table()
