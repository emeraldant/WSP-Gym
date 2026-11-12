"""Reprogramming: edit a *correct* network so that it implements a different, related specification.

A repair episode shows a network built for spec f with one planted fault. A reprogram episode shows a
network that computes f correctly together with the specification of a related function g, and asks
for an edit of at most ``MAX_OPS`` operations after which the network computes g. Structurally it is an
ordinary ``Episode``:

    task          g's Task (the prompt shows g's specification only, never f's)
    net           a correct network for f (obfuscated compiled network, or SGD-trained on f)
    clean         a correct network for g: ``net`` after the ground-truth edit
    acc_faulty    accuracy of the f-network on g          acc_clean   accuracy of ``clean`` on g
    inverse_edit  a ground-truth edit turning ``net`` into a network that computes g
    fault         "reprogram:<kind>"                       variant     "reprogram"

so ``env.score_answer`` applies unchanged: recovery >= 0.95 means "reprogrammed".

Kinds of change (``ReprogramConfig.kinds``):

``swap_inputs``    g(x) = f(x with inputs p and q exchanged). Ground truth: swap columns p, q of W1.
                   g's specification is rebuilt from f's parameters with the two variables exchanged
                   (canonical variable order, so the text does not give the pair away). g is scored on
                   f's scoring inputs with the two columns exchanged and f's labels, i.e. the labels are
                   f's canonical network run on the swapped inputs; for ``argmax`` these are exactly the
                   inputs without a tied top score under g.
``negate_output``  g = not f (binary families whose spec reads "Output 1 if ...; otherwise output 0").
                   Ground truth: scale the single row of the last layer by -1 and set its bias to 1 - b.
``change_params``  (compiled only) f's family parameters are perturbed: a threshold or comparison
                   constant moves, a clip bound moves, odd <-> even, one DNF literal changes polarity,
                   min <-> max, one input moves between argmax groups, ... (``SUBKINDS``). The ground truth
                   comes from *paired obfuscation*: f's and g's canonical networks are obfuscated with
                   identically seeded generators, so they differ only where the perturbation lands, and
                   the two stored networks are diffed (``find_edit``). The episode is redrawn when no edit
                   of at most 3 ops reaches recovery >= 0.999.

Controls. A fraction ``p_control`` of episodes show a network that already computes the target; the
right answer is ``[]`` (rewarded exactly like a clean repair episode). A control is drawn through the
same pipeline as any other episode and only then shows g's network, so its specification is
distributed like every other target and the prompt cannot reveal which is which.

``Episode.extras["reprogram"]`` holds what is needed to rebuild g's Task from JSON (``build_target``):
the kind, f's parameters and the swapped pair. NOTE: for ``negate_output`` the stored ``params`` are
f's (no family parameter expresses the negation for every family), so ``nets.build_task`` alone gives
f there, not g.

Everything is a deterministic function of ``(seed, ReprogramConfig)``.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import re
from dataclasses import asdict, dataclass

import numpy as np

from .edits import MAX_OPS, apply_edit, validate_edit
from .env import (
    DIRECT_SUFFIX, OPS_HELP, TRAIN_FAMILIES, Episode, GymConfig, _architecture, load_split, render_weights,
    score_answer,
)
from .nets import MLP, Task, build_task, obfuscate, sample_task

VARIANT = "reprogram"
KINDS = ("swap_inputs", "negate_output", "change_params")
TRAINED_KINDS = ("swap_inputs", "negate_output")  # kinds that have a ground truth on an SGD-trained network
NEGATE_FAMILIES = ("dnf", "parity", "threshold", "compare", "equality", "interval")
SUBKINDS = {  # change_params perturbations per family (count and mux have none that keeps the shape)
    "threshold": ("threshold_k",),
    "compare": ("compare_const",),
    "arith": ("arith_const",),
    "parity": ("parity_flip",),
    "dnf": ("dnf_literal", "dnf_negate"),
    # No "equality_m" (tolerated mismatches +-1): with the 2-3 pairs of sizes 0 and 1, m and m +- 1 are never
    # both >= 1 (same architecture) and balanced, so that perturbation cannot be drawn there.
    "equality": ("equality_flip",),
    "minmax": ("minmax_flip",),
    "argmax": ("argmax_move",),
    "interval": ("interval_bound",),
}
HELDOUT_SUBKINDS = ("argmax_move",)  # never in the reprogram training stream; present in test_reprogram
EDIT_RECOVERY = 0.999  # a ground-truth edit must recover this much (as for planted faults)
NOISE_TOL = 0.08  # stored weights that differ by less than this are treated as obfuscation noise


@dataclass(frozen=True)
class ReprogramConfig:
    families: tuple[str, ...] = TRAIN_FAMILIES
    kinds: tuple[str, ...] = KINDS
    size: int = 0  # as GymConfig.size
    p_control: float = 0.15
    min_drop: float = 0.15  # the shown network must be at least this much worse on g than g's network
    min_clean_acc: float = 0.99
    source: str = "compiled"  # "compiled" | "trained"
    exclude_subkinds: tuple[str, ...] = ()
    task: str = VARIANT

    def gym(self) -> GymConfig:
        return GymConfig(families=self.families, size=self.size, source=self.source)

    def to_json(self) -> dict:
        return asdict(self)


def subkinds_for(family: str, cfg: ReprogramConfig) -> list[str]:
    return [s for s in SUBKINDS.get(family, ()) if s not in cfg.exclude_subkinds]


def kinds_for(family: str, cfg: ReprogramConfig) -> list[str]:
    out = []
    for kind in cfg.kinds:
        if kind == "swap_inputs":
            out.append(kind)
        elif kind == "negate_output" and family in NEGATE_FAMILIES:
            out.append(kind)
        elif kind == "change_params" and cfg.source == "compiled" and subkinds_for(family, cfg):
            out.append(kind)
    return out


def _balanced(task: Task) -> bool:
    """The non-degeneracy conditions of ``nets.sample_task``."""
    if len(task.X) < 12:
        return False
    if task.kind == "binary":
        return 0.2 <= float(np.mean(task.Y)) <= 0.8
    if task.kind == "class":
        counts = np.bincount(task.Y, minlength=task.n_out) / len(task.Y)
        return bool(counts.min() >= 0.04 and counts.max() <= 0.8)
    return True


def _copy(params: dict) -> dict:
    return json.loads(json.dumps(params))


def _pick(rng, options: list):
    return options[int(rng.integers(len(options)))]


# ----------------------------------------------------------------------------
# swap_inputs
# ----------------------------------------------------------------------------


def _perm(n: int, a: int, b: int) -> list[int]:
    idx = list(range(n))
    idx[a], idx[b] = b, a
    return idx


def swap_params(family: str, params: dict, a: int, b: int) -> dict:
    """Parameters of g(x) = f(x with x_a and x_b exchanged): every variable index is mapped through the
    transposition, and lists that the samplers keep sorted are sorted again."""
    def m(v) -> int:
        v = int(v)
        return b if v == a else a if v == b else v

    def coefs(c: dict) -> dict:
        return {str(k): c2 for k, c2 in sorted(((m(k), c2) for k, c2 in c.items()))}

    p = _copy(params)
    if family == "dnf":
        p["clauses"] = [sorted([m(v), neg] for v, neg in cl) for cl in p["clauses"]]
    elif family in ("parity", "count"):
        p["S"] = sorted(m(v) for v in p["S"])
    elif family == "threshold":
        p["coefs"] = coefs(p["coefs"])
    elif family == "compare":
        for pr in p["preds"]:
            pr["coefs"] = {str(m(k)): c for k, c in pr["coefs"].items()}
            pr["text"] = re.sub(r"\bx(\d+)\b", lambda mo: f"x{m(mo.group(1))}", pr["text"])
    elif family == "minmax":
        p["vars"] = [m(v) for v in p["vars"]]
    elif family == "arith":
        p["a"], p["b"] = m(p["a"]), m(p["b"])
    elif family == "argmax":
        p["groups"] = [sorted(m(v) for v in g) for g in p["groups"]]
    elif family == "equality":
        p["pairs"] = [[m(i), m(j)] for i, j in p["pairs"]]
    elif family == "mux":
        p["ctrl"], p["data"] = [m(v) for v in p["ctrl"]], [m(v) for v in p["data"]]
    elif family == "interval":
        for iv in p["ivs"]:
            iv["coefs"] = coefs(iv["coefs"])
    else:
        raise KeyError(family)
    return p


def swap_target(family: str, base_params: dict, a: int, b: int) -> Task:
    """g's Task: the specification of the swapped parameters, scored on f's inputs with columns a and b
    exchanged and f's labels (g(x') = f(swap(x')), so x' = swap(x) carries f's label of x)."""
    f_task, _ = build_task(family, base_params)
    g_like, _ = build_task(family, swap_params(family, base_params, a, b))
    return dataclasses.replace(g_like, X=f_task.X[:, _perm(f_task.n_in, a, b)], Y=f_task.Y)


def _base_network(f_task: Task, f_canon: MLP, rng, cfg: ReprogramConfig):
    """A correct network for f from the configured source, or a reject reason."""
    if cfg.source == "trained":
        from .trained import train_network

        net = train_network(f_task, f_canon, rng, extra=cfg.gym().obf().n_distract)
        return "fit" if net is None else net
    try:
        return obfuscate(f_canon, f_task, rng, cfg.gym().obf())
    except RuntimeError:
        return "obfuscate"


def _swap_inputs(f_task: Task, f_canon: MLP, rng, cfg: ReprogramConfig, subkind=None):
    n = f_task.n_in

    def acc_on_swapped(net: MLP, a: int, b: int) -> float:
        return dataclasses.replace(f_task, X=f_task.X[:, _perm(n, a, b)]).accuracy(net)

    # Pairs whose exchange changes the function enough (judged on the exact canonical network).
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if 1.0 - acc_on_swapped(f_canon, a, b) >= cfg.min_drop]
    if not pairs:
        return "no_pair"
    order = rng.permutation(len(pairs))
    net = _base_network(f_task, f_canon, rng, cfg)
    if isinstance(net, str):
        return net
    acc_f = f_task.accuracy(net)
    for k in order:
        a, b = pairs[int(k)]
        if acc_f - acc_on_swapped(net, a, b) >= cfg.min_drop:
            break
    else:
        return "drop"
    g_task = swap_target(f_task.family, f_task.params, a, b)
    ops = [{"op": "swap", "tensor": "W1", "axis": "col", "i": a, "j": b}]
    return g_task, net, ops, {"swap": [a, b]}, None


# ----------------------------------------------------------------------------
# negate_output
# ----------------------------------------------------------------------------

_OUTPUT_BIT = re.compile(r"\b([Oo])utput ([01])\b")


def negate_spec(spec: str) -> str | None:
    """Exchange "Output 1"/"output 1" with "Output 0"/"output 0"; None when the text has no such form."""
    new, n = _OUTPUT_BIT.subn(lambda mo: f"{mo.group(1)}utput {1 - int(mo.group(2))}", spec)
    return new if n == 2 and new != spec else None


def negate_target(family: str, base_params: dict) -> Task | None:
    f_task, _ = build_task(family, base_params)
    spec = negate_spec(f_task.spec) if f_task.kind == "binary" else None
    if spec is None:
        return None
    return dataclasses.replace(f_task, spec=spec, Y=1.0 - f_task.Y)


def negate_edit(net: MLP) -> list[dict]:
    L = net.n_layers
    return [
        {"op": "scale", "tensor": f"W{L}", "axis": "row", "index": 0, "factor": -1.0},
        {"op": "set", "tensor": f"b{L}", "index": 0, "value": round(1.0 - float(net.bs[-1][0]), 2) + 0.0},
    ]


def _negate_output(f_task: Task, f_canon: MLP, rng, cfg: ReprogramConfig, subkind=None):
    g_task = negate_target(f_task.family, f_task.params)
    if g_task is None:
        return "spec"
    net = _base_network(f_task, f_canon, rng, cfg)
    if isinstance(net, str):
        return net
    return g_task, net, negate_edit(net), {}, None


# ----------------------------------------------------------------------------
# change_params
# ----------------------------------------------------------------------------


def perturb(family: str, params: dict, subkind: str, rng) -> dict | None:
    """g's parameters for one perturbation of f's, or None when ``subkind`` does not apply to this f."""
    p = _copy(params)
    if subkind == "threshold_k":
        total = sum(p["coefs"].values())
        opts = [k for k in (p["k"] - 1, p["k"] + 1) if 1 <= k <= total - 1]
        if not opts:
            return None
        p["k"] = _pick(rng, opts)
    elif subkind == "compare_const":
        opts = []
        for i, pr in enumerate(p["preds"]):
            top = {1: 9, 2: 8}.get(len(pr["coefs"]))  # "xi >= c" and "xi >= xj + c"; "xi + xj > xk" has no constant
            if top is not None:
                opts += [(i, t) for d in (-2, -1, 1, 2) if 1 <= (t := pr["theta"] + d) <= top]
        if not opts:
            return None
        i, theta = _pick(rng, opts)
        pr = p["preds"][i]
        pos = [k for k, c in pr["coefs"].items() if c > 0][0]
        neg = [k for k, c in pr["coefs"].items() if c < 0]
        pr["theta"] = theta
        if not neg:
            pr["text"] = f"x{pos} >= {theta}"
        else:
            pr["text"] = f"x{pos} > x{neg[0]}" if theta == 1 else f"x{pos} >= x{neg[0]} + {theta}"
    elif subkind == "arith_const":
        mode, lo, hi = p["mode"], p["lo"], p["hi"]
        ds = (-3, -2, -1, 1, 2, 3)
        if mode == "clip":
            opts = [("lo", v) for d in ds if 1 <= (v := lo + d) <= hi - 2] + [("hi", v) for d in ds if lo + 2 <= (v := hi + d) <= 17]
        elif mode == "satsum":
            opts = [("hi", v) for d in ds if 2 <= (v := hi + d) <= 17]
        elif mode == "posdiff":
            opts = [("lo", v) for d in ds if 1 <= (v := lo + d) <= 9]
        elif mode == "deadzone":  # the family uses min(lo, 4)
            lo = min(lo, 4)
            opts = [("lo", v) for d in ds if 1 <= (v := lo + d) <= 4]
        else:
            return None
        if not opts:
            return None
        key, value = _pick(rng, opts)
        p[key] = value
    elif subkind == "parity_flip":
        p["odd"] = not p["odd"]
    elif subkind == "dnf_literal":
        cl = p["clauses"][int(rng.integers(len(p["clauses"])))]
        lit = cl[int(rng.integers(len(cl)))]
        lit[1] = 1 - lit[1]
    elif subkind == "dnf_negate":
        p["negate"] = not p["negate"]
    elif subkind == "equality_flip":
        p["equal"] = not p["equal"]
    elif subkind == "minmax_flip":
        if p["mode"] not in ("max2", "min2"):
            return None
        p["mode"] = "min2" if p["mode"] == "max2" else "max2"
    elif subkind == "argmax_move":
        groups = p["groups"]
        opts = [(s, v, d) for s, g in enumerate(groups) if len(g) >= 2 for v in g for d in range(len(groups)) if d != s]
        if not opts:
            return None
        s, v, d = _pick(rng, opts)
        groups[s] = [u for u in groups[s] if u != v]
        groups[d] = sorted(groups[d] + [v])
    elif subkind == "interval_bound":
        opts = []
        for i, iv in enumerate(p["ivs"]):
            top = 9 if len(iv["coefs"]) == 1 else 18
            opts += [(i, "lo", v) for d in (-2, -1, 1, 2) if 1 <= (v := iv["lo"] + d) < iv["hi"]]
            opts += [(i, "hi", v) for d in (-2, -1, 1, 2) if iv["lo"] < (v := iv["hi"] + d) <= top - 1]
        if not opts:
            return None
        i, key, value = _pick(rng, opts)
        p["ivs"][i][key] = value
    else:
        raise KeyError(subkind)
    return p


def paired_obfuscate(f_canon: MLP, f_task: Task, g_canon: MLP, g_task: Task, rng, obf):
    """Obfuscate both canonical networks with identically seeded generators. Returns (f_net, g_net), or a
    reject reason: "obfuscate" (noise broke one of them) or "desync" (the two runs consumed their
    generators differently, e.g. the noise-retry loop took a different number of rounds)."""
    s = int(rng.integers(1 << 62))
    r1, r2 = np.random.default_rng(s), np.random.default_rng(s)
    try:
        f_net, g_net = obfuscate(f_canon, f_task, r1, obf), obfuscate(g_canon, g_task, r2, obf)
    except RuntimeError:
        return "obfuscate"
    if f_net.widths != g_net.widths or r1.bit_generator.state != r2.bit_generator.state:
        return "desync"
    return f_net, g_net


def _diff(a: MLP, b: MLP, tol: float = 0.0) -> list[tuple]:
    """Scalars of ``b`` that differ from ``a`` by more than ``tol``: [(tensor, index, value in b)]."""
    out = []
    for l in range(a.n_layers):
        for r, c in np.argwhere(np.abs(a.Ws[l] - b.Ws[l]) > tol):
            out.append((f"W{l + 1}", [int(r), int(c)], float(b.Ws[l][r, c])))
        for i in np.flatnonzero(np.abs(a.bs[l] - b.bs[l]) > tol):
            out.append((f"b{l + 1}", int(i), float(b.bs[l][i])))
    return out


def _sets(entries) -> list[dict]:
    return [{"op": "set", "tensor": t, "index": ix, "value": v} for t, ix, v in entries]


def _structured_ops(f: MLP, g: MLP) -> list[dict]:
    """Sign flips of rows / columns that hold at least two real differences, and swaps of two rows,
    two columns or two biases that both hold one."""
    ops = []
    for l in range(f.n_layers):
        name = f"W{l + 1}"
        d = np.abs(f.Ws[l] - g.Ws[l]) > NOISE_TOL
        for axis, counts in (("row", d.sum(1)), ("col", d.sum(0))):
            ops += [{"op": "scale", "tensor": name, "axis": axis, "index": int(i), "factor": -1.0} for i in np.flatnonzero(counts >= 2)]
            hit = [int(i) for i in np.flatnonzero(counts >= 1)]
            ops += [{"op": "swap", "tensor": name, "axis": axis, "i": i, "j": j} for i, j in itertools.combinations(hit, 2)]
        hit = [int(i) for i in np.flatnonzero(np.abs(f.bs[l] - g.bs[l]) > NOISE_TOL)]
        ops += [{"op": "swap", "tensor": f"b{l + 1}", "i": i, "j": j} for i, j in itertools.combinations(hit, 2)]
    return ops


def find_edit(f: MLP, g: MLP, task: Task, acc_f: float, acc_g: float) -> list[dict] | None:
    """The shortest edit of at most ``MAX_OPS`` ops that turns the stored network ``f`` into one that
    behaves like ``g`` on ``task``, found from the diff of the two networks.

    Candidates, shortest first and ``set``-only before structured ones:
      * up to three of the differing scalars overwritten with g's values. The subset must reproduce g's
        *outputs* on every scoring input, so it drops only differences that do nothing (those of inactive
        or cancelling distractor neurons) and never keeps a shortcut that merely lands on the right side
        of the decision threshold;
      * one sign flip / swap of a row or column, followed by up to two ``set``s of what still differs by
        more than obfuscation noise. A flipped line carries the other network's noise with the opposite
        sign, so these are verified by behaviour: recovery >= EDIT_RECOVERY.
    """
    exact = _diff(f, g)
    if not exact:
        return None
    out_g = g.forward(task.X)
    cands: list[tuple[int, int, list[dict]]] = []  # (number of ops, 0 = sets only | 1 = structured, ops)
    real = exact if len(exact) <= MAX_OPS else _diff(f, g, NOISE_TOL)
    if len(real) <= 12:
        for k in range(1, MAX_OPS + 1):
            cands += [(k, 0, _sets(sub)) for sub in itertools.combinations(real, k)]
    for op in _structured_ops(f, g):
        rest = _diff(apply_edit(f, [op]), g, NOISE_TOL)
        if len(rest) < len(real) and len(rest) <= 8:
            for k in range(0, MAX_OPS):
                cands += [(k + 1, 1, [op] + _sets(sub)) for sub in itertools.combinations(rest, k)]
    for _, structured, ops in sorted(cands, key=lambda c: c[:2]):
        edited = apply_edit(f, ops)
        if not structured and not np.allclose(edited.forward(task.X), out_g, atol=1e-6):
            continue
        if (task.accuracy(edited) - acc_f) / (acc_g - acc_f) >= EDIT_RECOVERY:
            return ops
    return None


def _change_params(f_task: Task, f_canon: MLP, rng, cfg: ReprogramConfig, subkind: str):
    g_params = perturb(f_task.family, f_task.params, subkind, rng)
    if g_params is None:
        return "not_applicable"
    g_task, g_canon = build_task(f_task.family, g_params)
    if g_canon.widths != f_canon.widths or not _balanced(g_task) or g_task.accuracy(g_canon) < 1.0:
        return "invalid"
    if 1.0 - g_task.accuracy(f_canon) < cfg.min_drop:
        return "same"
    pair = paired_obfuscate(f_canon, f_task, g_canon, g_task, rng, cfg.gym().obf())
    if isinstance(pair, str):
        return pair
    f_net, g_net = pair
    acc_f, acc_g = g_task.accuracy(f_net), g_task.accuracy(g_net)
    if acc_g - acc_f < cfg.min_drop:
        return "drop"
    ops = find_edit(f_net, g_net, g_task, acc_f, acc_g)
    if ops is None:
        return "no_edit"
    return g_task, f_net, ops, {"subkind": subkind, "n_diff": len(_diff(f_net, g_net))}, g_net


# ----------------------------------------------------------------------------
# Episodes
# ----------------------------------------------------------------------------

_BUILDERS = {"swap_inputs": _swap_inputs, "negate_output": _negate_output, "change_params": _change_params}


def build_target(family: str, params: dict, rp: dict) -> Task:
    """Rebuild g's Task from an episode's JSON (``rp`` is ``extras["reprogram"]``)."""
    if rp["kind"] == "swap_inputs":
        return swap_target(family, rp["base_params"], *rp["swap"])
    if rp["kind"] == "negate_output":
        return negate_target(family, rp["base_params"])
    return build_task(family, params)[0]


def make_reprogram_episode(seed: int, cfg: ReprogramConfig = ReprogramConfig(), max_tries: int = 400) -> Episode:
    """One reprogram episode. ``extras["rejects"]`` counts the draws of f discarded on the way, by reason:
    ``no_pair`` (no input pair whose exchange changes f enough), ``fit`` / ``obfuscate`` (no accurate
    network for f), ``not_applicable`` (the perturbation does not apply to this f), ``invalid`` (g is
    degenerate or has another architecture), ``same`` / ``drop`` (the f-network is less than ``min_drop``
    worse on g), ``desync`` (paired obfuscation fell out of step), ``no_edit`` (no edit of at most 3 ops),
    ``clean_acc`` (g's network is below ``min_clean_acc``) and ``verify`` (the edit failed scoring)."""
    rng = np.random.default_rng([int(seed), 611953])
    family = str(rng.choice(cfg.families))
    kind = str(rng.choice(kinds_for(family, cfg)))
    subkind = str(rng.choice(subkinds_for(family, cfg))) if kind == "change_params" else None
    control = bool(rng.random() < cfg.p_control)
    rejects: dict[str, int] = {}
    for attempt in range(max_tries):
        f_task, f_canon = sample_task(family, rng, cfg.size)
        made = _BUILDERS[kind](f_task, f_canon, rng, cfg, subkind)
        if isinstance(made, str):
            rejects[made] = rejects.get(made, 0) + 1
            continue
        g_task, net, ops, info, g_net = made
        ops = validate_edit(ops, net)
        clean = apply_edit(net, ops).round_()
        acc_clean, acc_faulty = g_task.accuracy(clean), g_task.accuracy(net)
        shown = clean if g_net is None else g_net  # what a control episode shows
        if min(acc_clean, g_task.accuracy(shown)) < cfg.min_clean_acc:
            rejects["clean_acc"] = rejects.get("clean_acc", 0) + 1
            continue
        if acc_clean - acc_faulty < cfg.min_drop:
            rejects["drop"] = rejects.get("drop", 0) + 1
            continue
        extras = {
            "reprogram": {"kind": kind, **info, "control": control, "base_params": f_task.params,
                          "acc_base": f_task.accuracy(net)},
            "tries": attempt + 1, "rejects": dict(rejects),
        }
        ep = Episode(
            seed=int(seed), family=family, variant=VARIANT, fault=f"reprogram:{kind}", task=g_task, clean=clean,
            net=net, acc_clean=acc_clean, acc_faulty=acc_faulty, inverse_edit=ops,
            fault_info={"kind": kind, **info, "n_ops": len(ops)}, failures=[], source=cfg.source, extras=extras,
        )
        if not score_answer(ep, json.dumps(ops))["success"]:
            rejects["verify"] = rejects.get("verify", 0) + 1
            continue
        if control:
            acc = g_task.accuracy(shown)
            return dataclasses.replace(ep, fault=None, clean=shown, net=shown, acc_clean=acc, acc_faulty=acc,
                                       inverse_edit=[], fault_info={})
        return ep
    raise RuntimeError(f"could not build a reprogram episode for seed {seed} ({family}, {kind}, {subkind}): {rejects}")


def kind_of(ep: Episode) -> str:
    """"swap_inputs", "negate_output" or "change_params:<subkind>" (also for control episodes)."""
    rp = ep.extras["reprogram"]
    return rp["kind"] + (f":{rp['subkind']}" if rp.get("subkind") else "")


# ----------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------


def render_reprogram_prompt(ep: Episode, direct: bool = False) -> str:
    """Same Specification / Architecture / Weights sections and edit DSL as the repair prompt. The
    specification is the *target* g; the specification the network was built for is never shown."""
    task, net = ep.task, ep.net
    s = [
        "You are reprogramming a small ReLU network by editing its weights. The network below works "
        "correctly, but it may have been built for a slightly different specification than the target "
        "below. Edit it so that it implements the target specification. Some neurons are legitimately "
        "redundant or inactive.",
        f"## Specification\nInputs: {task.input_desc}\nTarget: {task.spec}\nOutputs: {task.output_desc}",
        f"## Architecture\n{_architecture(net)}",
        f"## Weights\n{render_weights(net)}",
        "## Your task\n"
        f"Reply with a JSON list of at most {MAX_OPS} edit operations that makes the network implement the "
        "target specification, or [] if it already implements it. Each operation changes one row, one "
        "column or one scalar. Indices are 0-based. Allowed operations (examples):\n"
        f"{OPS_HELP}\n"
        "Your edit is scored only by how the edited network behaves on held-out inputs, so any edit that "
        "produces the specified behaviour is accepted. Editing a network that already implements the target "
        "is penalised, and so is an invalid or over-budget answer. End your reply with the JSON list and "
        "nothing after it."
        + (DIRECT_SUFFIX if direct else ""),
    ]
    return "\n\n".join(s)


# ----------------------------------------------------------------------------
# Baselines (see also baselines.py; all of them run through eval.run_policy)
# ----------------------------------------------------------------------------


def search_inputs_policy(ep: Episode, rng=None) -> str:
    """``baselines.search_policy`` with swaps of two input columns of W1 added to its candidates:
    brute force over single sign-flip / swap edits, scored with forward passes on the target."""
    from .baselines import single_op_candidates

    n = ep.net.Ws[0].shape[1]
    cands = single_op_candidates(ep.net) + [
        {"op": "swap", "tensor": "W1", "axis": "col", "i": i, "j": j} for i, j in itertools.combinations(range(n), 2)
    ]
    best_acc, best = ep.task.accuracy(ep.net), []
    for op in cands:
        acc = ep.task.accuracy(apply_edit(ep.net, [op]))
        if acc > best_acc + 1e-9:
            best_acc, best = acc, [op]
    return json.dumps(best)


def negate_always_policy(ep: Episode, rng=None) -> str:
    """Weight-blind template: on every single-output binary task, answer with the output negation
    (it reads one number, the last bias). Shows how much of a split one fixed formula solves."""
    return json.dumps(negate_edit(ep.net) if ep.task.kind == "binary" else [])


BASELINES = ("noop", "random", "outlier", "search", "search_inputs", "negate_always", "gradient")


def run_baselines(episodes: list[Episode], names=BASELINES, seed: int = 0) -> dict[str, list[dict]]:
    """Per-episode records of every baseline, in ``eval.run_policy`` format plus ``kind``."""
    from .eval import record, run_policy

    own = {"search_inputs": search_inputs_policy, "negate_always": negate_always_policy}
    out = {}
    for name in names:
        if name in own:
            out[name] = [record(ep, score_answer(ep, own[name](ep)), policy=name) for ep in episodes]
        else:
            out[name] = run_policy(episodes, name, seed)
        for ep, r in zip(episodes, out[name]):
            r["kind"] = kind_of(ep)
    return out


def main() -> None:
    from .eval import breakdown, summarize, write_csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits", default="test_reprogram,test_reprogram_families,test_reprogram_trained")
    ap.add_argument("--out", default="results/table_reprogram_baselines.csv")
    ap.add_argument("--baselines", default=",".join(BASELINES))
    a = ap.parse_args()
    rows = []
    for split in a.splits.split(","):
        _, eps = load_split(os.path.join(a.data, f"{split}.jsonl"))
        for name, recs in run_baselines(eps, tuple(a.baselines.split(","))).items():
            s = summarize(recs)
            print(f"{split:24s} {name:14s} reprogrammed {s['repair_rate']:.3f} [{s['repair_lo']:.3f}, {s['repair_hi']:.3f}]  "
                  f"control_ok {s['clean_ok']:.3f}  reward {s['mean_reward']:+.3f}", flush=True)
            for group, g in [("all", s)] + list(breakdown(recs, "kind").items()):
                rows.append({"split": split, "baseline": name, "kind": group,
                             **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items()}})
    if a.out:
        write_csv(a.out, rows)


if __name__ == "__main__":
    main()
