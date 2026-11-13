"""A slice of Qwen3.5-0.8B shown as a small network, so that proposers trained on small networks can be
benchmarked on a real LLM (README, "Fourth result").

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

    def __init__(self, layer: int, baseline: dict | None = None, scramble: bool = False):
        # scramble=True is a control: the prompt prints the shown weights in shuffled positions (same numbers,
        # same labels, same task text), while answers are still applied to the real weights. If a proposer answers
        # the same way, the weights it was shown did not inform its answer.
        self.scramble = scramble
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
            if self.scramble:
                W = W.copy()
                block = W[:, self.shown].ravel()
                np.random.default_rng([self.seed, 99]).shuffle(block)
                W[:, self.shown] = block.reshape(T, len(self.shown))
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
        """The same change as ``real_delta`` in factored form, delta = B @ A, built directly from the operations
        (each contributes at most two rank-1 terms), so no decomposition of a 1024 x 3584 matrix is needed."""
        W, U, G, P = self.W, self.U, self.G, self.net.Ws[0]
        bs: list[np.ndarray] = []
        as_: list[np.ndarray] = []

        def col(j):      # current output column of neuron j
            return W[:, j] + sum(b * a[j] for b, a in zip(bs, as_))

        def push(t):     # current row t of the slice: the pushes of every neuron on token t
            return P[t] + sum((U[t] @ b) * a for b, a in zip(bs, as_))

        def unit(j):
            e = np.zeros(W.shape[1])
            e[j] = 1.0
            return e

        for op in ops:
            kind, is_col = op["op"], op.get("axis") == "col"
            if kind == "scale" and is_col:
                new = [((op["factor"] - 1.0) * col(op["index"]), unit(op["index"]))]
            elif kind == "scale":
                new = [((op["factor"] - 1.0) * G[:, op["index"]], push(op["index"]))]
            elif kind == "swap" and is_col:
                i, j = op["i"], op["j"]
                ci, cj = col(i), col(j)
                new = [(cj - ci, unit(i)), (ci - cj, unit(j))] if i != j else []
            elif kind == "swap":
                t, s_ = op["i"], op["j"]
                new = [(G[:, t] - G[:, s_], push(s_) - push(t))] if t != s_ else []
            elif kind == "set":
                t, j = op["index"]
                new = [((op["value"] - U[t] @ col(j)) * G[:, t], unit(j))]
            elif is_col:
                new = [(G @ np.array(op["vector"], dtype=float), unit(op["index"]))]
            else:
                new = [(G[:, op["index"]].copy(), np.array(op["vector"], dtype=float))]
            for b, a in new:
                bs.append(b)
                as_.append(a)
        if len(bs) > max_rank:
            raise EditError(f"edit has rank {len(bs)} > {max_rank}")
        if not bs:
            return np.zeros((W.shape[0], 0)), np.zeros((0, W.shape[1]))
        return np.stack(bs, axis=1), np.stack(as_, axis=0)


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


async def propose(arm: str, sampler: str | None, reasoning: bool, scramble: bool = False) -> None:
    import tinker

    from .llm import INKLING, LLMPolicy, evaluate

    items = [SliceItem(layer, baseline_stats(), scramble) for layer in LAYERS]
    if reasoning:
        policy = await LLMPolicy(tinker.ServiceClient(), INKLING, sampler, 0.1, 20000).start()
    else:
        policy = await LLMPolicy(tinker.ServiceClient(), INKLING, sampler, 0.0, 256).start()
    out = os.path.join(OUT, "control_scrambled" if scramble else "proposals", f"{arm}.jsonl")
    recs = await evaluate(policy, items, out, K, 16, tag=f"qwen_slice/{arm}" + ("/scrambled" if scramble else ""), direct=not reasoning)
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
                # full-length answers need far fewer requests in flight than short ones to fit in GPU memory
                await qb.generate_records(todo, eid, False, cap, int(os.environ.get("WG_CLIENT_CONCURRENCY", 256)), on_record)
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


def set_threshold() -> dict:
    """Keep threshold = mean + 2 SD of the unedited model's accuracy on the screen rows over seeds 0-3; also
    records the unedited and neuron-573-x2 accuracy on dev. Depends on no edit."""
    from . import qwen_bench as qb

    cfg = _cfg()
    ids = {r["id"] for r in screen_rows()}
    recs = [r for r in qb.load_records("unedited", qb.tag_of("train", cfg["cap"])) if r["id"] in ids]
    accs = []
    for s_ in range(4):
        rs = [r for r in recs if r["sample"] == s_]
        if len(rs) != len(ids):
            raise SystemExit(f"unedited seed {s_}: {len(rs)} of {len(ids)} screen rows scored")
        accs.append(sum(r["correct"] for r in rs) / len(rs))
    mean, sd = float(np.mean(accs)), float(np.std(accs, ddof=1))
    cfg.update(seed_accs=accs, unedited_mean=mean, unedited_sd=sd, keep_above=mean + 2 * sd)
    for arm, key in (("unedited", "baseline_dev"), ("n573_x2", "positive_control_dev")):
        d = qb.load_records(arm, qb.tag_of("dev", cfg["cap"]))
        if len(d) >= 256:
            cfg[key] = {k: v for k, v in qb.summarize(d).items() if k in ("acc", "lo", "hi", "cut_share", "n_answers")}
    json.dump(cfg, open(os.path.join(OUT, "screen_config.json"), "w"), indent=1)
    print(f"unedited on the screen rows, seeds 0-3: {[round(100 * a, 1) for a in accs]} -> mean {100 * mean:.1f}%, SD {100 * sd:.1f}, keep above {100 * cfg['keep_above']:.1f}%")
    return cfg


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


async def dev(parallel: int = 1) -> None:
    from . import qwen_bench as qb

    cfg, kept = _cfg(), kept_edits()
    items = {it.layer: it for it in load_items()}
    gate = asyncio.Semaphore(parallel)
    todo = {edit_id(layer, b[0]): (layer, b[0]) for by in kept.values() for layer, b in by.items() if b}
    print(f"{len(todo)} kept edits to score on dev")
    await asyncio.gather(*(_score(items, l, o, "dev", qb.load_set("dev"), cfg["cap"], gate) for l, o in todo.values()))
    for acc, eid, layer, ops in dev_ranking():
        print(f"dev {100 * acc:5.1f}%  {eid}  {json.dumps(ops)[:140]}", flush=True)


def dev_ranking() -> list[tuple[float, str, int, list[dict]]]:
    """Kept edits with a complete dev score, best first: (dev accuracy, edit id, layer, ops)."""
    from . import qwen_bench as qb

    cfg, seen = _cfg(), {}
    for by in kept_edits().values():
        for layer, b in by.items():
            if b:
                eid = edit_id(layer, b[0])
                recs = qb.load_records(f"edit_{eid}", qb.tag_of("dev", cfg["cap"]), os.path.join(OUT, "edits"))
                if len(recs) >= 256:
                    seen[eid] = (qb.summarize(recs)["acc"], eid, layer, b[0])
    return sorted(seen.values(), key=lambda t: (-t[0], t[1]))


async def final(top: int = 3) -> None:
    """Full MATH-500 protocol, on the local server, for the ``top`` kept edits by dev score that do not have it yet."""
    from . import qwen_bench as qb

    items = {it.layer: it for it in load_items()}
    gate = asyncio.Semaphore(1)
    for acc, eid, layer, ops in dev_ranking()[:top]:
        print(f"final: {eid} (dev {100 * acc:.1f}%)", flush=True)
        await _score(items, layer, ops, "math500", qb.load_set("math500"), qb.MAX_TOKENS, gate)


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


def _proposal_rows(arm: str) -> list[dict]:
    path = os.path.join(OUT, "proposals", f"{arm}.jsonl")
    return [r for r in map(json.loads, open(path)) if "seed" in r and r["sample"] < K] if os.path.exists(path) else []


def hits(arm: str, layer: int) -> tuple[int, int, float | None]:
    """(answers that beat the keep threshold on the screen rows, answers given, mean output tokens) for one arm and layer."""
    cfg = _cfg()
    rows = [r for r in _proposal_rows(arm) if r["seed"] - SEED0 == layer]
    n = 0
    for r in rows:
        a = screen_acc(f"edit_{edit_id(layer, r['ops'])}") if r.get("valid") and r.get("ops") else None
        n += int(a is not None and a > cfg["keep_above"] + 1e-9)
    toks = [r["n_out"] for r in rows if r.get("n_out") is not None]
    return n, len(rows), (float(np.mean(toks)) if toks else None)


def control_hits(arm: str, layer: int) -> tuple[int, int] | None:
    """(answers that beat the keep threshold, answers given) for answers given under shuffled weights; None if unscored."""
    path = os.path.join(OUT, "control_scrambled", f"{arm}.jsonl")
    if not os.path.exists(path):
        return None
    cfg, item, n = _cfg(), SliceItem(layer), 0
    rows = [r for r in map(json.loads, open(path)) if r.get("seed") == SEED0 + layer and r.get("sample", 99) < K]
    for r in rows:
        if r.get("valid") and r.get("ops"):
            a = screen_acc(f"edit_{edit_id(layer, item.validate(r['ops']))}")
            if a is None:
                return None
            n += int(a > cfg["keep_above"] + 1e-9)
    return n, len(rows)


def _control_sentence(layer: int = 23) -> str:
    c = [control_hits(a, layer) for a in ("formula_rl_proposer", "real_rl_proposer")]
    if any(x is None for x in c):
        return ""
    f, r = hits("formula_rl_proposer", layer), hits("real_rl_proposer", layer)
    return (f" Control: shown the same prompt with the printed weights shuffled, the RL proposers changed their answers and only "
            f"{c[0][0] + c[1][0]} of {c[0][1] + c[1][1]} helped (against {f[0] + r[0]} of {f[1] + r[1]} with the real weights), so what they read in the weights matters.")


def _hit_sentence(layer: int = 23) -> str:
    f, r, u, d, rnd = (hits(a, layer) for a in ("formula_rl_proposer", "real_rl_proposer", "untrained_reasoning", "untrained", "random"))
    rl_tok = (f[2] * f[1] + r[2] * r[1]) / (f[1] + r[1])
    return (f"In layer {layer}, {f[0] + r[0]} of {f[1] + r[1]} single answers from our two RL-trained proposers helped on practice problems "
            f"(no reasoning, about {rl_tok:.0f} output tokens each; one of their edits is in the table), against {u[0]} of {u[1]} from the "
            f"untrained model with reasoning (about {u[2]:,.0f} tokens each), {rnd[0]} of {rnd[1]} random edits, and none from the untrained "
            f"model without reasoning, which never produced a valid answer." + _control_sentence(layer)
            + " That is one layer; the other layers are still being scored.")


# ------------------------------------------------------------------------------------------ report
def report(site_json: str = "site/qwen.json", table_csv: str = "results/table_qwen_math500.csv") -> None:
    """MATH-500 numbers for the conditions run so far, from the logs: a CSV for the write-up and the JSON the
    showcase page reads. Paired differences are over the problems both conditions were run on."""
    import csv

    from . import qwen_bench as qb

    cfg = _cfg()
    rows500 = qb.load_set("math500")
    order = [r["id"] for r in rows500]
    U = {r["id"]: r for r in qb.load_records("unedited", "math500")}
    props, d = proposals(), distinct_edits()

    def arms_of(eid):
        return sorted({arm for arm, by in props.items() for l, es in by.items() for o in es if edit_id(l, o) == eid})

    def complete_block(recs):
        """Use all 500 problems if present, else the largest complete half (so a partly finished run is not biased
        toward the answers that finish first)."""
        have = {r["id"] for r in recs}
        for lo, hi in ((0, 500), (250, 500), (0, 250)):
            ids = order[lo:hi]
            if all(i in have for i in ids):
                return [r for r in recs if r["id"] in set(ids)], (lo, hi)
        return [], None

    def paired(a, b):
        ids = [i for i in order if i in a and i in b]
        x = np.array([float(a[i]["correct"]) - float(b[i]["correct"]) for i in ids])
        boot = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1)
        return float(x.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), len(ids)

    conds = [("Unedited Qwen3.5-0.8B", "unedited", qb.OUT, None),
             ("Published edit: layer 22, neuron 573 x2 (Beaker)", "n573_x2", qb.OUT, None),
             ("\"Wait\" token ban, no weight change (Beaker's control)", "wait_ban", qb.OUT, None)]
    edit_dir = os.path.join(OUT, "edits")
    finals = sorted(a[5:] for a in os.listdir(edit_dir) if a.startswith("edit_") and qb.load_records(a, "math500", edit_dir))
    for eid in finals:
        who = arms_of(eid)
        names = {"untrained_reasoning": "untrained Inkling 2 Small with reasoning", "formula_rl_proposer": "RL proposer (formula tasks)",
                 "real_rl_proposer": "RL proposer (real-data networks)", "scale_rl": "final repair model", "random": "random valid edit"}
        conds.append((f"Layer {d[eid][0]} edit proposed by " + " and ".join(names.get(w, w) for w in who), f"edit_{eid}", edit_dir, eid))
    out, table = [], []
    for label, arm, out_dir, eid in conds:
        recs, block = complete_block(qb.load_records(arm, "math500", out_dir))
        if not recs:
            continue
        s = qb.summarize(recs)
        E = {r["id"]: r for r in recs}
        dv = paired(E, U) if arm != "unedited" else None
        note = "all 500 problems" if block == (0, 500) else f"problems {block[0]}-{block[1] - 1} only"
        if eid:
            note += "; " + json.dumps([{k: v for k, v in o.items() if k not in ("tensor",)} for o in d[eid][1]])
        table.append({"condition": label, "arm": arm, "n_problems": s["n_problems"], "acc": round(s["acc"], 4), "lo": round(s["lo"], 4), "hi": round(s["hi"], 4),
                      "cut_share": round(s["cut_share"], 4), "waits_per_10k_chars": round(s["waits_per_10k_chars"], 1),
                      "minus_unedited": "" if dv is None else round(dv[0], 4), "diff_lo": "" if dv is None else round(dv[1], 4),
                      "diff_hi": "" if dv is None else round(dv[2], 4), "n_paired": "" if dv is None else dv[3], "block": note.split(";")[0],
                      "ops": json.dumps(d[eid][1]) if eid else ""})
        out.append({"label": label, "math500": s["acc"], "lo": s["lo"], "hi": s["hi"], "n": s["n_problems"], "note": note.split(";")[0],
                    "highlight": bool(eid), "eid": eid})
    with open(table_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    by = {r["arm"]: r for r in table}
    best = max((r for r in table if r["arm"].startswith("edit_")), key=lambda r: r["acc"], default=None)
    n_scored = sum(screen_acc(f"edit_{e}") is not None for e in d)
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    summary, caveats = "", []
    if best:
        E = {r["id"]: r for r in complete_block(qb.load_records(best["arm"], "math500", edit_dir))[0]}
        X = {r["id"]: r for r in complete_block(qb.load_records("n573_x2", "math500"))[0]}
        px = paired(E, X)
        f, r, u, rnd = (hits(a_, 23) for a_ in ("formula_rl_proposer", "real_rl_proposer", "untrained_reasoning", "random"))
        f22, r22 = hits("formula_rl_proposer", 22), hits("real_rl_proposer", 22)
        ctl = [control_hits(a_, 23) for a_ in ("formula_rl_proposer", "real_rl_proposer")]
        rl_tok = (f[2] * f[1] + r[2] * r[1]) / (f[1] + r[1])
        summary = (
            f"Qwen3.5-0.8B gets stuck repeating lines that start with \"Wait\" and runs out of tokens: it solves {pct(by['unedited']['acc'])} of "
            f"MATH-500. We showed Inkling 2 Small one MLP layer of Qwen as a small table read from its weights, six tokens by 64 neurons, and "
            f"wrote its answer into the real model. Scaling down how hard layer 23 pushes the two \"Wait\" tokens lifts MATH-500 to "
            f"{pct(best['acc'])} [{100 * best['lo']:.1f}, {100 * best['hi']:.1f}], level with the published one-neuron edit that was found with "
            f"closed frontier models ({100 * px[0]:+.1f} points [{100 * px[1]:+.1f}, {100 * px[2]:+.1f}] on the {px[3]} problems both were run on). "
            f"Proposers trained by RL on Tinker, on small networks only, give this kind of edit in one answer of about {rl_tok:.0f} tokens with no "
            f"reasoning: {f[0] + r[0]} of their {f[1] + r[1]} answers for this layer helped, against {rnd[0]} of {rnd[1]} random edits. "
            f"Untrained, Inkling 2 Small finds it too, by reasoning for about {u[2]:,.0f} tokens.")
        if all(c is not None for c in ctl):
            summary += (f" When the same table is shown with its numbers shuffled, only {ctl[0][0] + ctl[1][0]} of {ctl[0][1] + ctl[1][1]} "
                        f"answers help, so the proposers are using the weights they read.")
        wb = by.get("wait_ban")
        caveats = [
            "The edit removes one habit; it does not add math ability. Banning the \"Wait\" token, with no weight change, scores "
            + (pct(wb["acc"]) if wb else "about the same") + ".",
            f"One layer of one model. In layer 22 the same proposers gave the same kind of edit and {f22[0] + r22[0]} of {f22[1] + r22[1]} helped.",
            "We chose the slice the proposer sees, and its prompt names the failure. Finding one neuron among 86,016 from raw weights, as the "
            "published runs did, is a harder search.",
            "GPT-6 Astra's edit in Takehi et al., found with 50 forward passes, reaches 48.1%.",
        ]
    json.dump({"status": "", "rows": [{k: v for k, v in r.items() if k not in ("highlight", "eid")} for r in out],
               "bars": [{"label": r["label"], "value": r["math500"], "lo": r["lo"], "hi": r["hi"], "note": r["note"], "highlight": r["highlight"]} for r in out],
               "caveats": caveats, "summary": summary}, open(site_json, "w"), indent=1)
    for r in table:
        print(f"{r['condition'][:86]:86s} {100 * r['acc']:5.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}] n={r['n_problems']} ({r['block']})")
    print(summary)
    _readme_section(table, summary, caveats)


def _readme_section(table: list[dict], summary: str, caveats: list[str], readme: str = "README.md") -> None:
    """Rewrite the block between the qwen markers of the README from the same numbers as the page."""
    start, end = "<!-- qwen:start -->", "<!-- qwen:end -->"
    text = open(readme).read()
    if start not in text or end not in text:
        return
    rows = "\n".join(f"| {r['condition']} | {100 * r['acc']:.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}] | {100 * r['cut_share']:.0f}% | {r['block']} |" for r in table)
    cfg, d = _cfg(), distinct_edits()
    done = [l for l in LAYERS if all(screen_acc(f"edit_{e}") is not None for e, (ll, _) in d.items() if ll == l)]
    names = (("untrained_reasoning", "Untrained Inkling 2 Small, with reasoning"), ("formula_rl_proposer", "RL proposer trained on formula tasks"),
             ("real_rl_proposer", "RL proposer trained on real-data networks"), ("scale_rl", "Final repair model"),
             ("untrained", "Untrained Inkling 2 Small, no reasoning"), ("random", "Random valid edits"))
    arm_rows = []
    for arm, label in names:
        cells = []
        for l in sorted(done, reverse=True):
            h = hits(arm, l)
            ctl = control_hits(arm, l) if l == 23 else None
            cells.append(f"{h[0]} of {h[1]}" + (f" ({ctl[0]} of {ctl[1]} with shuffled weights)" if ctl else ""))
        tok = [r["n_out"] for r in _proposal_rows(arm) if r.get("n_out") is not None]
        arm_rows.append(f"| {label} | " + " | ".join(cells) + f" | {(f'{np.mean(tok):,.0f}' if tok else '-')} |")
    method = (
        "**How the proposers were compared.** Each proposer gives 8 answers per layer for layers 16-23, from the same prompt. Every distinct "
        f"valid edit is written into Qwen and scored on {cfg['screen_n']} practice problems (GSM8K-train and MATH-train, none from MATH-500) with an "
        f"{cfg['cap']:,}-token limit. An answer *helps* if it beats {100 * cfg['keep_above']:.1f}%: the unedited model's "
        f"{100 * cfg['unedited_mean']:.1f}% plus twice its standard deviation over four seeds. Each arm's best helpful edit per layer is then "
        "scored on 256 further practice problems, and the best of those get the full MATH-500 run. In the control, the proposer sees the same "
        "prompt with the printed weights shuffled, and its answer is still applied to the real weights.\n\n"
        "Answers that help, out of 8, for the layers scored so far"
        + ("" if len(done) == len(LAYERS) else f" (layers {min(done)}-{max(done)}; the others are still being scored)") + ":\n\n"
        "| Proposer | " + " | ".join(f"Layer {l}" for l in sorted(done, reverse=True)) + " | Output tokens per answer |\n|---|" + "---|" * (len(done) + 1) + "\n"
        + "\n".join(arm_rows))
    body = (f"{start}\n{summary}\n\n"
            "MATH-500 with the published protocol (thinking on, at most 32,768 tokens, graded with math-verify; one sampled answer per problem):\n\n"
            "| Condition | MATH-500 | Answers cut at the token limit | Problems |\n|---|---|---|---|\n" + rows + "\n\n"
            + method + "\n\n" + "\n".join(f"- {c}" for c in caveats) + f"\n{end}")
    a, b = text.index(start), text.index(end) + len(end)
    open(readme, "w").write(text[:a] + body + text[b:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("propose")
    p.add_argument("arm")
    p.add_argument("sampler")
    p.add_argument("--reasoning", action="store_true")
    p.add_argument("--scramble", action="store_true", help="control: print the shown weights in shuffled positions")
    for c in ("random", "dev", "table", "prompt", "threshold", "report", "final"):
        sub.add_parser(c)
    p = sub.add_parser("screen")
    p.add_argument("--order", default="forward", choices=["forward", "reverse"])
    p.add_argument("--parallel", type=int, default=1, help="edits in flight at once; one edit is 64 requests, which is what a 16 GB GPU holds at this token limit")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "propose":
        asyncio.run(propose(a.arm, None if a.sampler == "base" else a.sampler, a.reasoning, a.scramble))
    elif a.cmd == "random":
        propose_random()
    elif a.cmd == "screen":
        asyncio.run(screen(parallel=a.parallel, order=a.order))
    elif a.cmd == "dev":
        asyncio.run(dev())
    elif a.cmd == "threshold":
        set_threshold()
    elif a.cmd == "report":
        report()
    elif a.cmd == "final":
        asyncio.run(final())
    elif a.cmd == "prompt":
        print(SliceItem(22, baseline_stats()).prompt(True))
    else:
        table()


if __name__ == "__main__":
    main()
