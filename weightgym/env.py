"""Episodes: one small network, zero or one planted fault, a prompt, and a reward.

An episode is fully determined by ``(seed, GymConfig)``. Rewards are computed
from the *behaviour* of the edited network on the task's scoring inputs, never
by comparing the edit with the planted fault.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .edits import MAX_OPS, EditError, apply_edit, parse_edit
from .faults import applicable_faults, plant_fault
from .nets import MLP, ObfuscationConfig, Task, build_task, obfuscate, sample_task

TRAIN_FAMILIES = ("dnf", "parity", "threshold", "compare", "minmax", "arith", "argmax", "count", "equality")
HELDOUT_FAMILIES = ("mux", "interval")  # routing and membership: never seen in training
TRAIN_FAULTS = ("sign_flip_out", "sign_flip_in", "swap_out_cols", "scale_out_col", "bias_shift", "weight_corrupt", "label_swap")
HELDOUT_FAULTS = ("scale_in_row", "spurious_direction")
VARIANTS = ("weights", "weights+failures")
REPAIR_THRESHOLD = 0.95


@dataclass(frozen=True)
class GymConfig:
    families: tuple[str, ...] = TRAIN_FAMILIES
    faults: tuple[str, ...] = TRAIN_FAULTS
    size: int = 0  # 0 small, 1 medium, 2 large (more inputs, wider, more distractors)
    p_clean: float = 0.18
    variants: tuple[str, ...] = VARIANTS
    max_failures: int = 8
    min_drop: float = 0.15
    n_distract: tuple[int, int] | None = None  # default depends on size
    p_split: float = 0.5
    scale: tuple[float, float] = (0.5, 2.0)
    noise: float = 0.01
    source: str = "compiled"  # "compiled" | "trained"

    def obf(self) -> ObfuscationConfig:
        nd = self.n_distract or {0: (2, 4), 1: (4, 8), 2: (10, 16)}[self.size]
        return ObfuscationConfig(n_distract=tuple(nd), p_split=self.p_split, scale=tuple(self.scale), noise=self.noise)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "GymConfig":
        d = dict(d)
        for k in ("families", "faults", "variants", "scale", "n_distract"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return GymConfig(**d)


@dataclass
class Episode:
    seed: int
    family: str
    variant: str
    fault: str | None  # None = clean control network
    task: Task
    clean: MLP
    net: MLP  # the network shown to the policy (== clean when fault is None)
    acc_clean: float
    acc_faulty: float
    inverse_edit: list[dict]
    fault_info: dict
    failures: list[int] = field(default_factory=list)  # indices into task.X shown in the prompt
    source: str = "compiled"

    @property
    def is_clean(self) -> bool:
        return self.fault is None

    @property
    def id(self) -> str:
        return f"{self.family}-{self.seed}"

    def prompt(self, direct: bool = False) -> str:
        return render_prompt(self, direct)

    def prompt_hash(self, direct: bool = False) -> str:
        return hashlib.sha256(self.prompt(direct).encode()).hexdigest()[:16]

    def to_json(self) -> dict:
        return {
            "seed": self.seed, "family": self.family, "variant": self.variant, "fault": self.fault,
            "params": self.task.params, "clean": self.clean.to_json(), "net": self.net.to_json(),
            "acc_clean": self.acc_clean, "acc_faulty": self.acc_faulty, "inverse_edit": self.inverse_edit,
            "fault_info": self.fault_info, "failures": self.failures, "source": self.source,
            "n_weights": self.net.n_weights, "prompt_hash": self.prompt_hash(),
        }

    @staticmethod
    def from_json(d: dict) -> "Episode":
        task, _ = build_task(d["family"], d["params"])
        return Episode(
            seed=d["seed"], family=d["family"], variant=d["variant"], fault=d["fault"], task=task,
            clean=MLP.from_json(d["clean"]), net=MLP.from_json(d["net"]), acc_clean=d["acc_clean"],
            acc_faulty=d["acc_faulty"], inverse_edit=d["inverse_edit"], fault_info=d["fault_info"],
            failures=list(d["failures"]), source=d.get("source", "compiled"),
        )


def make_episode(seed: int, cfg: GymConfig, max_tries: int = 200) -> Episode:
    rng = np.random.default_rng([int(seed), 7919])
    family = str(rng.choice(cfg.families))
    variant = str(rng.choice(cfg.variants))
    want_clean = bool(rng.random() < cfg.p_clean)
    for _ in range(max_tries):
        task, canon = sample_task(family, rng, cfg.size)
        if cfg.source == "trained":
            from .trained import train_network

            clean = train_network(task, canon, rng, extra=cfg.obf().n_distract)  # match compiled sizes per level
            if clean is None:
                continue
        else:
            try:
                clean = obfuscate(canon, task, rng, cfg.obf())
            except RuntimeError:
                continue
        acc_clean = task.accuracy(clean)
        if want_clean:
            return Episode(seed, family, variant, None, task, clean, clean, acc_clean, acc_clean, [], {}, [], cfg.source)
        kinds = applicable_faults(clean, cfg.faults)
        if not kinds:
            continue
        kind = str(rng.choice(kinds))
        planted = plant_fault(clean, task, kind, rng, cfg.min_drop)
        if planted is None:
            continue
        faulty, inverse, info, acc_faulty = planted
        failures: list[int] = []
        if variant == "weights+failures":
            wrong = np.flatnonzero(~task.correct(faulty))
            pick = rng.choice(wrong, size=min(cfg.max_failures, len(wrong)), replace=False)
            failures = sorted(int(i) for i in pick)
        return Episode(seed, family, variant, kind, task, clean, faulty, acc_clean, acc_faulty, inverse, info, failures, cfg.source)
    raise RuntimeError(f"could not build an episode for seed {seed} ({family})")


# ----------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------


def fmt(v: float) -> str:
    s = f"{float(v):.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _vec(v) -> str:
    return "[" + ", ".join(fmt(x) for x in np.atleast_1d(v)) + "]"


def render_weights(net: MLP) -> str:
    L = net.n_layers
    names = ["x"] + [f"h{l}" for l in range(1, L)] + ["y"]
    lines = []
    for l, (W, b) in enumerate(zip(net.Ws, net.bs), start=1):
        src, dst = names[l - 1], names[l]
        lines.append(
            f"W{l} ({W.shape[0]} rows x {W.shape[1]} cols) and b{l}. "
            f"Row i = incoming weights of {dst}[i]; column j = outgoing weights of {src}[j]."
        )
        for i in range(W.shape[0]):
            lines.append(f"  W{l}[{i}] = {_vec(W[i])}   b{l}[{i}] = {fmt(b[i])}")
    return "\n".join(lines)


def _architecture(net: MLP) -> str:
    L = net.n_layers
    w = net.widths
    parts, prev = [], "x"
    for l in range(1, L):
        parts.append(f"h{l} = relu(W{l} @ {prev} + b{l})  [{w[l]} neurons]")
        prev = f"h{l}"
    parts.append(f"y = W{L} @ {prev} + b{L}  [{w[L]} output{'s' if w[L] > 1 else ''}, no relu]")
    return "\n".join(parts)


def _example(task: Task, net: MLP, i: int) -> str:
    x = task.X[i]
    out = net.forward(x[None, :])[0]
    xs = "[" + ", ".join(str(int(v)) for v in x) + "]"
    if task.kind == "binary":
        return f"x = {xs}: expected {int(task.Y[i])}, network gives y = {fmt(out[0])}"
    if task.kind == "class":
        return f"x = {xs}: expected answer {int(task.Y[i])}, network gives y = {_vec(out)} (answer {int(np.argmax(out))})"
    return f"x = {xs}: expected {_vec(task.Y[i])}, network gives y = {_vec(out)}"


OPS_HELP = """\
{"op": "scale", "tensor": "W2", "axis": "col", "index": 4, "factor": -1}   multiply one row or column of a weight matrix
{"op": "swap", "tensor": "W2", "axis": "col", "i": 1, "j": 5}   swap two rows or two columns of a weight matrix
{"op": "set", "tensor": "W1", "index": [3, 0], "value": 0.5}   overwrite one weight (index is [row, col])
{"op": "set", "tensor": "b1", "index": 3, "value": -1.5}   overwrite one bias
{"op": "add", "tensor": "W2", "axis": "col", "index": 2, "vector": [0.5, -0.5]}   add a vector to one row or column
{"op": "swap", "tensor": "b2", "i": 0, "j": 2}   swap two biases (scale and add also work on a bias: use "index" with "factor" or "value")"""


DIRECT_SUFFIX = (
    " Answer immediately from your reading of the weights: output only the JSON list, with no reasoning, "
    "no calculations and no explanation."
)


def render_prompt(ep: Episode, direct: bool = False) -> str:
    """``direct=True`` is the no-reasoning regime: the model must answer at once."""
    task, net = ep.task, ep.net
    s = []
    s.append(
        "You are debugging a small ReLU network by reading its weights. The network was built to "
        "implement the specification below. It either works correctly or contains exactly one fault "
        "(for example a corrupted, mis-scaled, sign-flipped, shifted or swapped part of one tensor). "
        "Some neurons are legitimately redundant or inactive; those are not faults."
    )
    s.append(f"## Specification\nInputs: {task.input_desc}\nTarget: {task.spec}\nOutputs: {task.output_desc}")
    s.append(f"## Architecture\n{_architecture(net)}")
    s.append(f"## Weights\n{render_weights(net)}")
    if ep.variant == "weights+failures":
        if ep.failures:
            ex = "\n".join(_example(task, net, i) for i in ep.failures)
            s.append(f"## Failing inputs\nThe network is wrong on some inputs, for example:\n{ex}")
        else:
            s.append("## Failing inputs\nNone found: the network was correct on every input tested.")
    s.append(
        "## Your task\n"
        f"Reply with a JSON list of at most {MAX_OPS} edit operations that repairs the network, or [] if it "
        "has no fault. Each operation changes one row, one column or one scalar. Indices are 0-based. "
        "Allowed operations (examples):\n"
        f"{OPS_HELP}\n"
        "Your edit is scored only by how the edited network behaves on held-out inputs, so any edit that "
        "restores the specified behaviour is accepted. Editing a network that has no fault is penalised, "
        "and so is an invalid or over-budget answer. End your reply with the JSON list and nothing after it."
        + (DIRECT_SUFFIX if direct else "")
    )
    return "\n\n".join(s)


# ----------------------------------------------------------------------------
# Reward
# ----------------------------------------------------------------------------


def score_answer(ep: Episode, text: str) -> dict:
    """Score a model answer. Keys: reward, valid, success, recovery, acc_after, n_ops, noop, error, ops."""
    res = {"reward": -1.0, "valid": False, "success": False, "recovery": None, "acc_after": None, "n_ops": None, "noop": False, "error": None, "ops": None}
    try:
        ops = parse_edit(text, ep.net)
    except EditError as e:
        res["error"] = str(e)
        return res
    acc_after = ep.task.accuracy(apply_edit(ep.net, ops))
    res.update(valid=True, ops=ops, n_ops=len(ops), noop=len(ops) == 0, acc_after=acc_after)
    if ep.is_clean:
        if not ops:
            res.update(reward=1.0, success=True)
        else:
            res["reward"] = -0.25 - min(0.75, 3.0 * max(0.0, ep.acc_clean - acc_after))
    else:
        recovery = (acc_after - ep.acc_faulty) / (ep.acc_clean - ep.acc_faulty)
        res.update(recovery=recovery, reward=float(np.clip(recovery, -0.25, 1.0)), success=bool(recovery >= REPAIR_THRESHOLD))
    return res


# ----------------------------------------------------------------------------
# Splits on disk
# ----------------------------------------------------------------------------


def save_split(path: str, episodes: list[Episode], cfg: GymConfig, name: str) -> None:
    with open(path, "w") as f:
        f.write(json.dumps({"split": name, "config": cfg.to_json(), "n": len(episodes)}) + "\n")
        for ep in episodes:
            f.write(json.dumps(ep.to_json()) + "\n")


def load_split(path: str) -> tuple[dict, list[Episode]]:
    with open(path) as f:
        header = json.loads(f.readline())
        return header, [Episode.from_json(json.loads(line)) for line in f if line.strip()]
