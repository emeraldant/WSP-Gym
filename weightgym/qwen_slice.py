"""A slice of Qwen3.5-0.8B shown as a small network, so that proposers trained on small networks can be
benchmarked on a real LLM (pre-registered in NOTES.md, "Step 2").

One item per MLP layer. The printed matrix W1 has one row per token of a fixed list and one column per MLP
neuron: W1[t, j] = (output direction of token t) . (output column of neuron j), i.e. how hard neuron j pushes
the score of token t each time it fires, read directly from the weights (the token's unembedding row times the
final norm's weight; later layers and the norm's rescaling are ignored). No activations and no forward passes go
into the prompt.

Every valid answer in the gym's edit DSL corresponds to a real edit of the layer's output matrix (down_proj):

    scale / swap of columns   the neurons' whole output columns are scaled / swapped
    set W1[t, j] = v          the smallest change of neuron j's output column that makes its push on token t equal v
    add a vector to column j  the smallest change of neuron j's output column that shifts its six pushes by the vector
    scale / swap of rows      the smallest change of the whole matrix that scales / swaps the pushes on those tokens
    anything on b1            invalid: the layer has no bias

"Smallest" is the minimum-norm lift through the pseudo-inverse of the six token directions.

    python -m weightgym.qwen_slice build                    # slices from the weights (any environment with torch)
    python -m weightgym.qwen_slice propose <arm> <tinker://sampler | base> [--reasoning]     # Tinker environment
    python -m weightgym.qwen_slice random
    python -m weightgym.qwen_slice screen | dev | table     # GPU environment, vLLM server running
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os

import numpy as np

from .edits import MAX_OPS, EditError, apply_edit, parse_edit, validate_edit
from .env import DIRECT_SUFFIX, OPS_HELP
from .nets import MLP

HOME = os.path.expanduser(os.environ.get("WG_QWEN_HOME", "~/wg-qwen"))
SLICES = os.path.join(HOME, "slices")
OUT = "runs/qwen_slice"
TOKENS = ((" Wait", 13428), ("Wait", 13784), ("</think>", 248069), (" Therefore", 14833), (" So", 1987), ("boxed", 76367))
LAYERS = tuple(range(16, 24))
N_SHOWN = 64
SEED0 = 70_000_000
K = 8
ARMS = ("random", "untrained", "untrained_reasoning", "formula_rl_proposer", "real_rl_proposer", "scale_rl")
SAMPLERS = {  # adapters kept on Tinker (NOTES.md)
    "formula_rl_proposer": "tinker://d06765d0-0bcf-5c2e-b5c3-9f7adb571970:train:0/sampler_weights/final",
    "real_rl_proposer": "tinker://410fdc47-dcb9-5775-b02a-caa13cd33513:train:0/sampler_weights/final",
    "scale_rl": "tinker://df143eaf-cf67-52a8-9906-2027a75938b3:train:0/sampler_weights/final",
}


def _g(v: float) -> str:
    s = f"{float(v):.3g}"
    return "0" if s in ("-0", "0") else s


def _vec(v) -> str:
    return "[" + ", ".join(_g(x) for x in np.atleast_1d(v)) + "]"


# ------------------------------------------------------------------------------------------ building
def build(layers=LAYERS) -> None:
    """Write one .npz per layer: the layer's output matrix and the six token directions, in float32."""
    import glob

    import torch
    from safetensors import safe_open

    os.makedirs(SLICES, exist_ok=True)
    with safe_open(glob.glob(os.path.join(HOME, "models", "base", "*.safetensors"))[0], "pt") as f:
        gamma = f.get_tensor("model.language_model.norm.weight").to(torch.float32).numpy()
        emb = f.get_slice("model.language_model.embed_tokens.weight")
        U = np.stack([emb[i:i + 1].to(torch.float32).numpy()[0] for _, i in TOKENS]) * gamma[None, :]   # (6, 1024)
        for layer in layers:
            W = f.get_tensor(f"model.language_model.layers.{layer}.mlp.down_proj.weight").to(torch.float32).numpy()  # (1024, 3584)
            P = U @ W
            shown = np.sort(np.argsort(-P.var(axis=0), kind="stable")[:N_SHOWN])
            unit = float(P[:, shown].std())            # prompt units: the shown entries have standard deviation 1
            np.savez_compressed(os.path.join(SLICES, f"layer{layer:02d}.npz"), W=W, U=U / unit, shown=shown, unit=unit)
            print(f"layer {layer}: unit {unit:.4g}, shown columns {shown[:8].tolist()}...")


class SliceItem:
    """One layer of Qwen3.5-0.8B as a single linear layer. Duck-types what ``llm.evaluate`` and ``eval.record`` use."""

    is_clean = False
    fault = "imperfect"
    variant = "weights"
    family = "qwen_slice"
    source = "Qwen/Qwen3.5-0.8B"

    def __init__(self, layer: int, baseline: dict | None = None):
        z = np.load(os.path.join(SLICES, f"layer{layer:02d}.npz"))
        self.layer, self.seed = layer, SEED0 + layer
        self.W, self.U, self.shown = z["W"].astype(np.float64), z["U"].astype(np.float64), z["shown"]
        self.G = self.U.T @ np.linalg.inv(self.U @ self.U.T)          # (1024, 6): minimum-norm lift, U @ G = I
        self.net = MLP([self.U @ self.W], [np.zeros(len(TOKENS))])
        self.baseline = baseline or {}
        self._prompt: dict[bool, str] = {}

    # ---- prompt -----------------------------------------------------------------------------
    def prompt(self, direct: bool = False) -> str:
        if direct not in self._prompt:
            W = self.net.Ws[0]
            T, n = W.shape
            b = self.baseline
            s = [
                "You are improving a real, public language model, Qwen3.5-0.8B (24 layers), by editing the weights of one "
                f"of its layers: the output matrix of the MLP in layer {self.layer} (layers are numbered from 0). On math "
                "problems the model often gets stuck: it keeps writing lines that start with \"Wait\", loops, and runs out of "
                "tokens before it gives an answer. Nothing was deliberately broken, so there may be no single culprit.",
                f"## What is shown\nThe MLP has {n} neurons. Each time neuron j fires with strength x[j] (which can be positive or "
                "negative), it adds x[j] times its output column to the model's internal state. W1[i][j] is how hard that "
                "column pushes the score of token i, read directly from the weights: y = W1 @ x is this layer's direct push "
                "on the scores of six tokens. A positive W1[i][j] with a positive x[j] makes token i more likely.",
                "## Tokens (rows)\n" + ", ".join(f"{i} = {json.dumps(t)}" for i, (t, _) in enumerate(TOKENS))
                + "\n\"</think>\" ends the model's reasoning; \"boxed\" is part of the final answer.",
                f"## Architecture\ny = W1 @ x + b1  [{T} outputs, no relu]\nx = the activations of the layer's {n} neurons. "
                "b1 is zero and cannot be edited: the layer has no bias.",
            ]
            if b:
                s.append(f"## Current behaviour on {b['n']} practice math problems\n"
                         f"Accuracy: {100 * b['acc']:.1f}%. {100 * b['cut_share']:.0f}% of the answers hit the limit of {b['max_tokens']} tokens "
                         f"without finishing. \"Wait\" appears {b['waits_per_10k_chars']:.0f} times per 10,000 characters of output.")
            w = [f"Row norms |W1[i]| = {_vec(np.linalg.norm(W, axis=1))}",
                 f"W1 has {T} rows x {n} cols, too many to print. Row i = the pushes on token i; column j = the pushes of neuron j. "
                 f"Shown below, as column:weight pairs with the true column indices, are the {len(self.shown)} columns whose weights "
                 f"vary most across the tokens. The other {n - len(self.shown)} columns are hidden and cannot be edited by index: a "
                 "column operation or a single-weight edit must use one of the column indices shown. An operation on a whole "
                 "row (scale or swap) still acts on every column of that row."]
            w += [f"  W1[{i}] = {{" + ", ".join(f"{int(j)}: {_g(W[i, j])}" for j in self.shown) + "}" for i in range(T)]
            s.append("## Weights\n" + "\n".join(w))
            tail = (
                "## Your task\n"
                f"Reply with a JSON list of at most {MAX_OPS} edit operations that makes the model answer more math problems "
                "correctly. Each operation changes one row, one column or one scalar of W1. Indices are 0-based. Only the "
                f"tensor W1 can be edited (an added vector needs {T} numbers for a column). Allowed operations (the indices "
                "and values in these examples are only illustrations):\n"
                f"{OPS_HELP.replace('W2', 'W1').replace('b2', 'b1')}\n"
                "Your edit is a proposal: it will be written into the real model, tested on practice problems and kept only "
                "if the model gets more of them right, so propose the change you think is most likely to help. An invalid "
                "or over-budget answer is penalised. End your reply with the JSON list and nothing after it."
            )
            for d in (False, True):
                self._prompt[d] = "\n\n".join(s + [tail + (DIRECT_SUFFIX if d else "")])
        return self._prompt[direct]

    def prompt_hash(self, direct: bool = False) -> str:
        return hashlib.sha256(self.prompt(direct).encode()).hexdigest()[:16]

    # ---- edits ------------------------------------------------------------------------------
    def validate(self, ops) -> list[dict]:
        ops = validate_edit(ops, self.net)
        ok = set(int(j) for j in self.shown)
        for op in ops:
            if op["tensor"] != "W1":
                raise EditError("b1 cannot be edited: the layer has no bias")
            cols = [op["index"][1]] if op["op"] == "set" else [] if op.get("axis") != "col" else [op[k] for k in ("index", "i", "j") if k in op]
            bad = [j for j in cols if j not in ok]
            if bad:
                raise EditError(f"column {bad[0]} of W1 is hidden and cannot be edited by index")
        return ops

    def score(self, text: str) -> dict:
        """Validity only; the reward comes later, from running the edited model (``screen``)."""
        res = {"reward": 0.0, "valid": False, "success": False, "recovery": None, "n_ops": None, "noop": False, "error": None,
               "ops": None, "gain": None}
        try:
            ops = self.validate(parse_edit(text, self.net))
        except EditError as e:
            res.update(reward=-1.0, error=str(e))
            return res
        res.update(valid=True, ops=ops, n_ops=len(ops), noop=len(ops) == 0)
        return res

    def real_delta(self, ops: list[dict]) -> np.ndarray:
        """The change of the layer's real output matrix (1024 x 3584) that a validated slice edit stands for."""
        W, U, G = self.W.copy(), self.U, self.G
        for op in ops:
            kind, col = op["op"], op.get("axis") == "col"
            if kind == "scale" and col:
                W[:, op["index"]] *= op["factor"]
            elif kind == "scale":
                t = op["index"]
                W += (op["factor"] - 1.0) * np.outer(G[:, t], U[t] @ W)
            elif kind == "swap" and col:
                i, j = op["i"], op["j"]
                W[:, [i, j]] = W[:, [j, i]]
            elif kind == "swap":
                t, s = op["i"], op["j"]
                pt, ps = U[t] @ W, U[s] @ W
                W += np.outer(G[:, t] - G[:, s], ps - pt)
            elif kind == "set":
                t, j = op["index"]
                W[:, j] += (op["value"] - U[t] @ W[:, j]) * G[:, t]
            elif col:   # add a vector of six pushes to one column
                W[:, op["index"]] += G @ np.array(op["vector"], dtype=float)
            else:       # add a vector to one row
                W += np.outer(G[:, op["index"]], np.array(op["vector"], dtype=float))
        return W - self.W

    def factors(self, ops: list[dict], max_rank: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """delta = B @ A with at most ``max_rank`` columns (exact: each operation has rank at most 2)."""
        d = self.real_delta(ops)
        cols = np.flatnonzero(np.abs(d).sum(axis=0) > 0)
        rows_dense = len(cols) > max_rank          # a row operation touches every column: factor by SVD
        if not rows_dense:
            B = d[:, cols]
            A = np.zeros((len(cols), d.shape[1]))
            A[np.arange(len(cols)), cols] = 1.0
            return B, A
        u, sv, vt = np.linalg.svd(d, full_matrices=False)
        r = int(np.sum(sv > 1e-9 * sv[0]))
        if r > max_rank:
            raise EditError(f"edit has rank {r} > {max_rank}")
        return u[:, :r] * sv[:r], vt[:r]


def edit_id(layer: int, ops: list[dict]) -> str:
    return f"L{layer:02d}_" + hashlib.sha256(json.dumps(ops, sort_keys=True).encode()).hexdigest()[:12]


def load_items(baseline: dict | None = None) -> list[SliceItem]:
    return [SliceItem(layer, baseline) for layer in LAYERS]


def baseline_stats() -> dict | None:
    p = os.path.join(OUT, "screen_config.json")
    return json.load(open(p)).get("baseline") if os.path.exists(p) else None


# ------------------------------------------------------------------------------------------ proposals
def random_valid_edit(item: SliceItem, rng: np.random.Generator) -> list[dict]:
    from .baselines import random_policy

    while True:
        try:
            ops = item.validate(json.loads(random_policy(item, rng)))
            if ops:
                return ops
        except EditError:
            continue


def propose_random() -> None:
    path = os.path.join(OUT, "proposals", "random.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in load_items():
            rng = np.random.default_rng([it.seed, 0])
            for k in range(K):
                ops = random_valid_edit(it, rng)
                f.write(json.dumps({"seed": it.seed, "sample": k, "valid": True, "ops": ops, "policy": "random"}) + "\n")
    print("wrote", path)


async def propose(arm: str, sampler: str | None, reasoning: bool) -> None:
    import tinker

    from .llm import INKLING, LLMPolicy, evaluate

    items = load_items(baseline_stats())
    if reasoning:
        policy = await LLMPolicy(tinker.ServiceClient(), INKLING, sampler, 0.1, 20000).start()
    else:
        policy = await LLMPolicy(tinker.ServiceClient(), INKLING, sampler, 0.0, 256).start()
    recs = await evaluate(policy, items, os.path.join(OUT, "proposals", f"{arm}.jsonl"), K, 16, tag=f"qwen_slice/{arm}", direct=not reasoning)
    print(arm, "valid", sum(bool(r["valid"]) for r in recs), "of", len(recs), "| non-empty", sum(bool(r.get("ops")) for r in recs))


def proposals() -> dict[str, dict[int, list[list[dict]]]]:
    """arm -> layer -> the arm's valid, non-empty edits in sampling order (first K)."""
    out: dict[str, dict[int, list]] = {}
    d = os.path.join(OUT, "proposals")
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        arm = fn[:-6]
        rows = sorted((r for r in map(json.loads, open(os.path.join(d, fn))) if "seed" in r and r["sample"] < K), key=lambda r: (r["seed"], r["sample"]))
        for r in rows:
            layer = r["seed"] - SEED0
            out.setdefault(arm, {}).setdefault(layer, [])
            if r.get("valid") and r.get("ops"):
                out[arm][layer].append(r["ops"])
    return out


# ------------------------------------------------------------------------------------------ verification on Qwen
def _cfg() -> dict:
    """runs/qwen_slice/screen_config.json: cap (token limit), screen_idx (rows of the practice set ``train`` used for
    screening), keep_above (screen accuracy an edit must beat), baseline (unedited, screen set), baseline_dev."""
    return json.load(open(os.path.join(OUT, "screen_config.json")))


def screen_rows() -> list[dict]:
    from . import qwen_bench as qb

    rows = qb.load_set("train")
    return [rows[i] for i in _cfg()["screen_idx"]]


def screen_acc(arm: str, out_dir: str | None = None) -> float | None:
    """Accuracy of one condition on the screen rows (sample 0), or None if they are not all scored yet."""
    from . import qwen_bench as qb

    cfg = _cfg()
    recs = {r["id"]: r for r in qb.load_records(arm, qb.tag_of("train", cfg["cap"]), out_dir or os.path.join(OUT, "edits")) if r["sample"] == 0}
    ids = [r["id"] for r in screen_rows()]
    if any(i not in recs for i in ids):
        return None
    return sum(recs[i]["correct"] for i in ids) / len(ids)


async def _score(items: dict[int, SliceItem], layer: int, ops: list[dict], set_name: str, rows: list[dict], cap: int, gate) -> None:
    """Score one edit on the local GPU server (resumable; rows already scored anywhere are skipped)."""
    from . import qwen_bench as qb

    eid, out_dir = edit_id(layer, ops), os.path.join(OUT, "edits")
    tag = qb.tag_of(set_name, cap)
    done = {(r["id"], r["sample"]) for r in qb.load_records(f"edit_{eid}", tag, out_dir)}
    todo = [(r, 0) for r in rows if (r["id"], 0) not in done]
    if not todo:
        return
    async with gate:
        B, A = items[layer].factors(ops)
        qb.write_delta_adapter(eid, {(layer, "down"): (B, A)}, {"layer": layer, "slice_ops": ops})
        await qb.load_adapter(eid)
        path = os.path.join(out_dir, f"edit_{eid}", f"{tag}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "a") as f:
                def on_record(rec):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                await qb.generate_records(todo, eid, False, cap, 256, on_record)
        finally:
            await qb.unload_adapter(eid)


def distinct_edits() -> dict[str, tuple[int, list[dict]]]:
    return {edit_id(layer, ops): (layer, ops) for by in proposals().values() for layer, es in by.items() for ops in es}


async def screen(parallel: int = 6, order: str = "forward") -> None:
    """Score every distinct valid edit on the screen rows with the local server. ``order`` lets a second worker
    (e.g. cloud containers) start from the other end of the list."""
    cfg, rows = _cfg(), screen_rows()
    items = {it.layer: it for it in load_items()}
    todo = sorted(distinct_edits().items())
    if order == "reverse":
        todo = todo[::-1]
    print(f"{len(todo)} distinct valid edits")
    gate = asyncio.Semaphore(parallel)

    async def one(eid, layer, ops):
        if screen_acc(f"edit_{eid}") is None:
            await _score(items, layer, ops, "train", rows, cfg["cap"], gate)
        acc = screen_acc(f"edit_{eid}")
        if acc is not None:
            print(f"{eid} screen acc {100 * acc:.1f}%  {json.dumps(ops)[:150]}", flush=True)

    await asyncio.gather(*(one(e, l, o) for e, (l, o) in todo))


def kept_edits() -> dict[str, dict[int, tuple[list[dict], float] | None]]:
    """arm -> layer -> (edit, screen accuracy) of the arm's best edit that beats the keep threshold, else None
    (the first one sampled wins ties)."""
    cfg, out = _cfg(), {}
    for arm, by in proposals().items():
        out[arm] = {}
        for layer in LAYERS:
            best = None
            for ops in by.get(layer, []):
                acc = screen_acc(f"edit_{edit_id(layer, ops)}")
                if acc is not None and acc > cfg["keep_above"] + 1e-9 and (best is None or acc > best[1] + 1e-9):
                    best = (ops, acc)
            out[arm][layer] = best
    return out


async def dev(parallel: int = 4) -> None:
    from . import qwen_bench as qb

    cfg, kept = _cfg(), kept_edits()
    items = {it.layer: it for it in load_items()}
    gate = asyncio.Semaphore(parallel)
    todo = {edit_id(layer, b[0]): (layer, b[0]) for by in kept.values() for layer, b in by.items() if b}
    print(f"{len(todo)} kept edits to score on dev")
    await asyncio.gather(*(_score(items, l, o, "dev", qb.load_set("dev"), cfg["cap"], gate) for l, o in todo.values()))


def table(results: str = "results/table_qwen_slice.csv") -> None:
    import csv

    from . import qwen_bench as qb

    cfg, kept, props = _cfg(), kept_edits(), proposals()
    tag = qb.tag_of("dev", cfg["cap"])
    base_dev = cfg["baseline_dev"]["acc"]
    rows, per_arm = [], {}
    for arm in sorted(kept):
        changes = []
        for layer in LAYERS:
            b = kept[arm][layer]
            row = {"arm": arm, "layer": layer, "valid_nonempty": len(props[arm].get(layer, [])), "kept": bool(b), "screen_acc": "", "dev_acc": "",
                   "dev_change_points": 0.0, "ops": ""}
            if b:
                recs = qb.load_records(f"edit_{edit_id(layer, b[0])}", tag, os.path.join(OUT, "edits"))
                row.update(screen_acc=round(100 * b[1], 2), ops=json.dumps(b[0]))
                if len(recs) >= 256:
                    acc = qb.summarize(recs)["acc"]
                    row.update(dev_acc=round(100 * acc, 2), dev_change_points=round(100 * (acc - base_dev), 2))
            changes.append(row["dev_change_points"])
            rows.append(row)
        per_arm[arm] = np.array(changes)
    with open(results, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(LAYERS), size=(10000, len(LAYERS)))
    summary = []
    for arm, c in per_arm.items():
        boot = c[idx].mean(1)
        d = c - per_arm.get("random", np.zeros(len(LAYERS)))
        dboot = d[idx].mean(1)
        summary.append({"arm": arm, "instances_with_kept_edit": int(sum(bool(kept[arm][l]) for l in LAYERS)), "mean_dev_change_points": round(float(c.mean()), 2),
                        "lo": round(float(np.quantile(boot, 0.025)), 2), "hi": round(float(np.quantile(boot, 0.975)), 2),
                        "minus_random": round(float(d.mean()), 2), "minus_random_lo": round(float(np.quantile(dboot, 0.025)), 2),
                        "minus_random_hi": round(float(np.quantile(dboot, 0.975)), 2)})
    with open(results.replace(".csv", "_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(f"{s['arm']:22s} kept in {s['instances_with_kept_edit']}/{len(LAYERS)} layers; mean dev change {s['mean_dev_change_points']:+.2f} "
              f"[{s['lo']:+.2f}, {s['hi']:+.2f}] points; minus random {s['minus_random']:+.2f} [{s['minus_random_lo']:+.2f}, {s['minus_random_hi']:+.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("propose")
    p.add_argument("arm")
    p.add_argument("sampler")
    p.add_argument("--reasoning", action="store_true")
    for c in ("random", "dev", "table", "prompt"):
        sub.add_parser(c)
    p = sub.add_parser("screen")
    p.add_argument("--order", default="forward", choices=["forward", "reverse"])
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "propose":
        asyncio.run(propose(a.arm, None if a.sampler == "base" else a.sampler, a.reasoning))
    elif a.cmd == "random":
        propose_random()
    elif a.cmd == "screen":
        asyncio.run(screen(order=a.order))
    elif a.cmd == "dev":
        asyncio.run(dev())
    elif a.cmd == "prompt":
        print(SliceItem(22, baseline_stats()).prompt(True))
    else:
        table()


if __name__ == "__main__":
    main()
