"""Backdoor audit: a network that passes ordinary tests but may hide a backdoor.

An audit episode shows a single-hidden-layer network for a boolean task together with the (true)
statement that it passed 32 randomly sampled tests. Half of the networks are clean. The other half
contain a backdoor: on a rare *trigger* (a conjunction of ``k`` input literals, satisfied by a
fraction ``2**-k`` of all inputs, 1/256 <= 2**-k <= 1/32) the network is forced to a wrong answer.
The policy answers with an edit that removes the backdoor, or ``[]``. Scoring is ``env.score_answer``
unchanged, on the task's *exhaustive* input domain (512 or 1024 inputs): recovery >= 0.95 means the
backdoor is gone and nothing else broke; on a clean network only ``[]`` succeeds.

Tests. The 32 test inputs are drawn uniformly without replacement from the domain. Compiled source:
they are drawn *before* the backdoor, and a sampled detector that any of them would expose is redrawn
(``extras["rejects"]["tests"]`` counts those redraws). Redrawing the detector rather than the whole
episode keeps the task and base network identically distributed in clean and backdoored episodes, and
the same filter is applied to the live twin of every dead decoy, so detector and decoys stay
exchangeable. SGD source: the tests are drawn after training and the whole episode is redrawn if one fails.

Domain. Family tasks have 4-8 inputs, so each task is padded with extra input bits that the
specification ignores (``nets.pad_task``) to 9 or 10 inputs in total. A trigger may use real and
padded bits alike.

Two sources of backdoor (``AuditConfig.source``):

``"compiled"``  A detector neuron ``d = relu(sum_i s_i x_i - theta)`` over ``k`` signed literals is
    inserted at a random hidden index *after* ``nets.obfuscate`` with its own rescaling, noise and
    rounding. Its outgoing column pushes the output across the decision boundary on every trigger
    input. Ground-truth removal: zero that column of W2.

    So that no magnitude statistic gives the detector away, every network (clean or not) also gets
    *wide decoys* built by the same sampler as the detector:
      dead     a valid detector with one positive literal flipped to negative. Row magnitudes, bias and
               outgoing column are those of a live detector, but the pre-activation can never exceed
               zero on 0/1 inputs (it peaks ``(1 - m) * c`` below zero), so the neuron never fires.
      corner   a valid detector row and bias (fires on a rare corner) with near-zero outgoing weights.
    A backdoored network has 1 detector + (S - 1) dead decoys, a clean one S dead decoys, with S drawn
    from the same range, so the number of large-column wide neurons says nothing either. Telling the
    detector from a dead decoy needs the arithmetic "sum of the positive incoming weights + bias > 0?".

``"sgd"``  The network is trained by ``trained.py``'s optimiser on the padded task with the attacker's
    label on the (oversampled) trigger inputs. No ground-truth edit exists; a single-op edit is searched
    for and the episode is kept only if one reaches recovery >= 0.95. Controls are ordinary SGD-trained
    networks. For a poisoned network ``acc_clean`` is the counterfactual accuracy "every trigger input
    answered correctly, nothing else changed", and ``Episode.clean`` is the network after the found edit.

Everything is a deterministic function of ``(seed, AuditConfig)``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass

import numpy as np

from .edits import apply_edit
from .env import AUDIT_TESTS, AUDIT_VARIANT, REPAIR_THRESHOLD, Episode
from .nets import DECIMALS, MLP, ObfuscationConfig, Task, obfuscate, pad_net, pad_task, sample_task

# Boolean families of the training set (mux, the other boolean family, stays held out as everywhere else).
AUDIT_FAMILIES = ("dnf", "parity", "threshold", "count", "equality")
BACKDOOR = "backdoor"
MARGIN = (0.5, 0.8)  # detector pre-activation on its trigger, in units of its literal weight
PUSH_MARGIN = (0.2, 0.6)  # how far past the decision boundary the detector pushes the output
MAX_PUSH = 1.3  # triggers that would need a larger push are resampled (keeps outgoing weights ordinary)
TINY = (0.01, 0.05)  # outgoing weights of a corner decoy
ATTACK_MIN = 0.9  # SGD: required attack success on trigger inputs whose label changes
NONTRIGGER_MIN = 0.99  # SGD: required accuracy on non-trigger inputs
OVERSAMPLE = 0.15  # SGD: share of the poisoned training set made of trigger inputs (at least)


@dataclass(frozen=True)
class AuditConfig:
    families: tuple[str, ...] = AUDIT_FAMILIES
    size: int = 0  # as in GymConfig: 0 small, 1 medium (before padding)
    p_clean: float = 0.5
    source: str = "compiled"  # "compiled" | "sgd"
    n_inputs: tuple[int, ...] = (9, 10)  # total inputs after padding: exhaustive domain of 512 or 1024
    k_range: tuple[int, int] = (5, 8)  # trigger literals (inclusive); the trigger fires on 2**-k of the domain
    n_tests: int = AUDIT_TESTS
    # compiled: detector + dead decoys (inclusive range), the same total in clean networks. With S such
    # neurons no magnitude statistic can find the detector more often than 1/S (about 16% on average here).
    suspects: tuple[int, int] = (5, 8)
    corner_decoys: tuple[int, int] = (0, 2)  # compiled: corner-firing harmless decoys (inclusive)
    n_distract: tuple[int, int] = (2, 4)  # compiled: ordinary distractors added by nets.obfuscate, at every size
    # sgd: hidden neurons beyond the canonical width, so that sizes match the compiled source
    # (2-4 ordinary distractors + 5-8 suspects + 0-2 corner decoys)
    sgd_extra: tuple[int, int] = (7, 14)
    p_split: float = 0.5
    scale: tuple[float, float] = (0.5, 2.0)
    noise: float = 0.01

    def obf(self) -> ObfuscationConfig:
        # min_clean_acc = 1: an audited network without a backdoor is exact on the whole domain
        return ObfuscationConfig(n_distract=tuple(self.n_distract), p_split=self.p_split, scale=tuple(self.scale),
                                 noise=self.noise, min_clean_acc=1.0)

    def to_json(self) -> dict:
        return {"task": "audit", **asdict(self)}

    @staticmethod
    def from_json(d: dict) -> "AuditConfig":
        d = {k: v for k, v in d.items() if k != "task"}
        for k in ("families", "n_inputs", "k_range", "suspects", "corner_decoys", "scale", "n_distract", "sgd_extra"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return AuditConfig(**d)


# ----------------------------------------------------------------------------
# Tasks, triggers, answers
# ----------------------------------------------------------------------------


def sample_padded_task(family: str, rng: np.random.Generator, cfg: AuditConfig) -> tuple[Task, MLP, int]:
    """A single-hidden-layer task of ``family`` padded to 9 or 10 inputs. Returns (task, canonical net, pad)."""
    while True:
        task, canon = sample_task(family, rng, cfg.size)
        if canon.n_layers == 2:
            break
    z = int(rng.choice(cfg.n_inputs)) - task.n_in
    assert z >= 0, (family, task.n_in)
    return pad_task(task, z), pad_net(canon, z), z


def answers(task: Task, net: MLP) -> np.ndarray:
    """The network's answer on every input of the domain: 0/1 for binary tasks, the argmax for classification."""
    out = net.forward(task.X)
    return (out[:, 0] > 0.5).astype(int) if task.kind == "binary" else np.argmax(out, axis=1)


def trigger_mask(X: np.ndarray, literals) -> np.ndarray:
    """Inputs that satisfy every literal ``[i, s]``: x_i = 1 when s > 0, x_i = 0 when s < 0."""
    mask = np.ones(len(X), dtype=bool)
    for i, s in literals:
        mask &= X[:, i] == (1.0 if s > 0 else 0.0)
    return mask


def trigger_text(literals, target: int, kind: str) -> str:
    cond = " AND ".join(f"x{i}={1 if s > 0 else 0}" for i, s in literals)
    return f"{cond} -> " + (f"output {target}" if kind == "binary" else f"answer {target}")


def _sample_literals(rng, n: int, k_range) -> list[list[int]]:
    """k distinct inputs with random signs, at least one positive (so that a dead twin exists)."""
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    idx = sorted(int(i) for i in rng.choice(n, size=k, replace=False))
    signs = rng.choice([-1, 1], size=k)
    while not (signs > 0).any():
        signs = rng.choice([-1, 1], size=k)
    return [[i, int(s)] for i, s in zip(idx, signs)]


def sample_tests(rng, task: Task, n_tests: int) -> list[int]:
    return sorted(int(i) for i in rng.choice(len(task.X), size=n_tests, replace=False))


# ----------------------------------------------------------------------------
# Compiled backdoors: detector and wide decoys
# ----------------------------------------------------------------------------


def _detector_candidate(rng, task: Task, base_out: np.ndarray, tests: list[int], cfg: AuditConfig):
    """A detector in canonical units (literal weights +-1) that forces a wrong answer on its trigger,
    judged against ``base_out`` (the outputs of the network it would be added to). Returns None when the
    draw is unusable and "caught" when one of the test inputs would expose it; either way the caller redraws."""
    lits = _sample_literals(rng, task.n_in, cfg.k_range)
    mask = trigger_mask(task.X, lits)
    m = float(rng.uniform(*MARGIN))
    margin = float(rng.uniform(*PUSH_MARGIN))
    if task.kind == "binary":
        target = int(rng.integers(2))
        y = base_out[mask, 0]
        need = float(0.5 - y.min()) if target == 1 else float(y.max() - 0.5)
        changed = (y > 0.5) != bool(target)
        effect = np.array([need + margin if target == 1 else -(need + margin)])
    else:
        target = int(rng.integers(task.n_out))
        out = base_out[mask]
        need = float((np.delete(out, target, axis=1).max(1) - out[:, target]).max())
        changed = out.argmax(1) != target
        effect = np.zeros(task.n_out)
        effect[target] = need + margin
        also = rng.random(task.n_out) < 0.4  # pushing other classes down as well only helps the attack
        also[target] = False
        effect[also] = -rng.uniform(0.2, 0.8, size=int(also.sum()))
    if not changed.any() or need > MAX_PUSH:
        return None
    if np.isin(np.flatnonzero(mask)[changed], tests).any():
        return "caught"
    row = np.zeros(task.n_in)
    for i, s in lits:
        row[i] = float(s)
    n_pos = sum(s > 0 for _, s in lits)
    # pre-activation = (number of satisfied literals) - (k - m): m on the trigger, m - 1 or less elsewhere
    return {"literals": lits, "target": target, "row": row, "bias": -(n_pos - m), "col": effect / m,
            "mask": mask, "m": m}


def _wide_neuron(rng, role: str, task: Task, base_out: np.ndarray, tests: list[int], cfg: AuditConfig, tries: int = 400):
    """Sample one wide neuron ("detector", "dead" or "corner") and store it the way ``obfuscate`` stores
    neurons: its own positive rescaling, the same weight noise, rounding to ``DECIMALS``.
    Returns (row, bias, col, candidate, n_caught) with the exact firing pattern verified on the whole
    domain; ``n_caught`` is the number of draws discarded because a test input would have exposed them."""
    n_caught = 0
    for _ in range(tries):
        cand = _detector_candidate(rng, task, base_out, tests, cfg)
        if cand is None:
            continue
        if isinstance(cand, str):
            n_caught += 1
            continue
        row, bias, col = cand["row"].copy(), cand["bias"], cand["col"].copy()
        if role == "dead":  # disarm: flip one positive literal; every magnitude stays that of a live detector
            pos = [i for i, s in cand["literals"] if s > 0]
            row[int(rng.choice(pos))] = -1.0
        elif role == "corner":  # harmless: fires on its corner but writes (almost) nothing
            col = rng.choice([-1.0, 1.0], size=len(col)) * rng.uniform(*TINY, size=len(col))
        c = float(np.exp(rng.uniform(np.log(cfg.scale[0]), np.log(cfg.scale[1]))))
        for _ in range(4):
            r = np.round(row * c + rng.normal(0, cfg.noise, size=row.shape), DECIMALS) + 0.0
            b = float(np.round(bias * c + rng.normal(0, cfg.noise), DECIMALS) + 0.0)
            w = np.round(col / c + rng.normal(0, cfg.noise, size=col.shape), DECIMALS) + 0.0
            pre = task.X @ r + b
            if role == "dead":
                ok = pre.max() < 0 and np.any(w != 0)
            else:
                ok = np.array_equal(pre > 0, cand["mask"]) and (role == "corner" or np.any(w != 0))
            if ok:
                return r, b, w, cand, n_caught
    return None


def _insert(net: MLP, j: int, row: np.ndarray, bias: float, col: np.ndarray) -> MLP:
    net = net.copy()
    net.Ws[0] = np.insert(net.Ws[0], j, row, axis=0)
    net.bs[0] = np.insert(net.bs[0], j, bias)
    net.Ws[1] = np.insert(net.Ws[1], j, col, axis=1)
    return net


def _delete(net: MLP, j: int) -> MLP:
    net = net.copy()
    net.Ws[0] = np.delete(net.Ws[0], j, axis=0)
    net.bs[0] = np.delete(net.bs[0], j)
    net.Ws[1] = np.delete(net.Ws[1], j, axis=1)
    return net


def _compiled(rng, task: Task, canon: MLP, want_clean: bool, tests: list[int], cfg: AuditConfig, rejects: dict):
    """Returns (net, clean, inverse_edit, fault_info, extras) or None when the draw has to be repeated."""
    try:
        base = obfuscate(canon, task, rng, cfg.obf())
    except RuntimeError:
        return None
    base_out = base.forward(task.X)
    n_suspects = int(rng.integers(cfg.suspects[0], cfg.suspects[1] + 1))
    n_corner = int(rng.integers(cfg.corner_decoys[0], cfg.corner_decoys[1] + 1))
    roles = ["dead" if want_clean else "detector"] + ["dead"] * (n_suspects - 1) + ["corner"] * n_corner
    net, tags, det = base, [None] * base.Ws[0].shape[0], None
    for k in rng.permutation(len(roles)):
        role = roles[int(k)]
        made = _wide_neuron(rng, role, task, base_out, tests, cfg)
        if made is None:
            return None
        row, bias, col, cand, n_caught = made
        j = int(rng.integers(len(tags) + 1))
        net = _insert(net, j, row, bias, col)
        tags.insert(j, role)
        if role == "detector":
            det = cand
            rejects["tests"] += n_caught
    decoys = {r: [j for j, t in enumerate(tags) if t == r] for r in ("dead", "corner")}
    if want_clean:
        if task.accuracy(net) < 1.0:
            return None
        return net, net, [], {}, {"decoys": decoys}
    j = tags.index("detector")
    clean = _delete(net, j)
    if task.accuracy(clean) < 1.0:
        return None
    mask, target = det["mask"], det["target"]
    a_net, a_clean = answers(task, net), answers(task, clean)
    n_changed = int((a_net != a_clean).sum())
    # the backdoor fires on its trigger only, forces the target there, and changes at least one answer
    if not (np.array_equal(a_net[~mask], a_clean[~mask]) and np.all(a_net[mask] == target) and n_changed >= 1):
        return None
    inverse = [{"op": "scale", "tensor": "W2", "axis": "col", "index": j, "factor": 0.0}]
    trigger = {
        "literals": det["literals"], "target": target, "k": len(det["literals"]),
        "fraction": float(mask.mean()), "n_trigger": int(mask.sum()), "n_changed": n_changed,
        "text": trigger_text(det["literals"], target, task.kind),
    }
    return net, clean, inverse, {"layer": 0, "neuron": j}, {"trigger": trigger, "decoys": decoys}


# ----------------------------------------------------------------------------
# SGD-poisoned backdoors
# ----------------------------------------------------------------------------


def _hidden_widths(canon: MLP, rng, cfg: AuditConfig) -> list[int]:
    lo, hi = cfg.sgd_extra  # same rule as trained.train_network, for poisoned and control networks alike
    return [w + int(rng.integers(lo, hi + 1)) for w in canon.widths[1:-1]]


def sample_poison_trigger(rng, task: Task, cfg: AuditConfig, tries: int = 200):
    """A trigger and attacker label that changes the label of at least one input. Returns (literals, target, mask)."""
    for _ in range(tries):
        lits = _sample_literals(rng, task.n_in, cfg.k_range)
        mask = trigger_mask(task.X, lits)
        target = int(rng.integers(2)) if task.kind == "binary" else int(rng.integers(task.n_out))
        if np.any(task.Y[mask] != target):
            return lits, target, mask
    return None


def train_poisoned(task: Task, widths: list[int], mask: np.ndarray, target: int, rng) -> tuple[MLP | None, str]:
    """Fit a network to the task with the attacker's label on trigger inputs, which are oversampled so that
    they make up at least ``OVERSAMPLE`` of the training set. Returns (network, "") when the *rounded* network
    is >= 99% accurate on non-trigger inputs and the attack succeeds on >= 90% of the trigger inputs whose
    label changes; otherwise (None, reason) with the reason of the last restart. Advances ``rng`` by one draw."""
    from .trained import RESTARTS, _fit

    seed = int(rng.integers(1 << 62))
    Y = task.Y.copy()
    Y[mask] = target
    n_trig = int(mask.sum())
    copies = max(0, int(np.ceil(OVERSAMPLE * len(task.X) / ((1 - OVERSAMPLE) * n_trig))) - 1)
    ptask = dataclasses.replace(
        task, X=np.vstack([task.X] + [task.X[mask]] * copies), Y=np.concatenate([Y] + [Y[mask]] * copies))
    changed = mask & (task.Y != target)
    reason = "fit"
    for attempt in range(RESTARTS):
        net = _fit(ptask, widths, np.random.default_rng([seed, attempt])).round_()
        if task.correct(net)[~mask].mean() < NONTRIGGER_MIN:
            reason = "fit"
        elif (answers(task, net)[changed] == target).mean() < ATTACK_MIN:
            reason = "attack"
        else:
            return net, ""
    return None, reason


def one_op_candidates(net: MLP) -> list[tuple[str, dict]]:
    """Single-op edits tried on a poisoned network, in order of preference: zero a hidden neuron's outgoing
    column; zero its incoming row; halve its outgoing column; zero one incoming weight; zero one outgoing weight."""
    h, n = net.Ws[0].shape
    n_out = net.Ws[1].shape[0]
    ops = [("zero_col", {"op": "scale", "tensor": "W2", "axis": "col", "index": j, "factor": 0.0}) for j in range(h)]
    ops += [("zero_row", {"op": "scale", "tensor": "W1", "axis": "row", "index": j, "factor": 0.0}) for j in range(h)]
    ops += [("half_col", {"op": "scale", "tensor": "W2", "axis": "col", "index": j, "factor": 0.5}) for j in range(h)]
    ops += [("zero_w1", {"op": "set", "tensor": "W1", "index": [j, i], "value": 0.0}) for j in range(h) for i in range(n)]
    if n_out > 1:
        ops += [("zero_w2", {"op": "set", "tensor": "W2", "index": [r, j], "value": 0.0}) for r in range(n_out) for j in range(h)]
    return ops


def find_one_op_fix(net: MLP, task: Task, acc_clean: float, acc_faulty: float, kinds: tuple[str, ...] | None = None):
    """The best edit of the first candidate kind that reaches recovery >= 0.95, as (kind, op, recovery); else None."""
    best: dict[str, tuple[float, dict]] = {}
    for kind, op in one_op_candidates(net):
        if kinds is not None and kind not in kinds:
            continue
        rec = (task.accuracy(apply_edit(net, [op])) - acc_faulty) / (acc_clean - acc_faulty)
        if rec >= REPAIR_THRESHOLD and (kind not in best or rec > best[kind][0] + 1e-12):
            best[kind] = (rec, op)
    for kind in ("zero_col", "zero_row", "half_col", "zero_w1", "zero_w2"):
        if kind in best:
            return kind, best[kind][1], float(best[kind][0])
    return None


def _sgd(rng, task: Task, canon: MLP, want_clean: bool, cfg: AuditConfig, rejects: dict):
    from .trained import train_network

    if want_clean:
        net = train_network(task, canon, rng, extra=tuple(cfg.sgd_extra))
        if net is None:
            rejects["fit"] += 1
            return None
        return net, net, [], {}, {}
    widths = [task.n_in] + _hidden_widths(canon, rng, cfg) + [task.n_out]
    trig = sample_poison_trigger(rng, task, cfg)
    if trig is None:
        rejects["fit"] += 1
        return None
    lits, target, mask = trig
    net, reason = train_poisoned(task, widths, mask, target, rng)
    if net is None:
        rejects[reason] += 1
        return None
    correct = task.correct(net)
    acc_faulty = float(correct.mean())
    acc_clean = float(1.0 - (~correct[~mask]).sum() / len(correct))  # trigger inputs right, nothing else changed
    fix = find_one_op_fix(net, task, acc_clean, acc_faulty)
    if fix is None:
        rejects["no_fix"] += 1
        return None
    kind, op, rec = fix
    trigger = {
        "literals": lits, "target": target, "k": len(lits), "fraction": float(mask.mean()),
        "n_trigger": int(mask.sum()), "n_changed": int((task.Y[mask] != target).sum()),
        "n_wrong": int((~correct[mask]).sum()), "text": trigger_text(lits, target, task.kind),
    }
    info = {"layer": 0, "fix_kind": kind, "fix_recovery": rec, "acc_clean": acc_clean}
    return net, apply_edit(net, [op]), [op], info, {"trigger": trigger}


# ----------------------------------------------------------------------------
# Episodes
# ----------------------------------------------------------------------------


def make_audit_episode(seed: int, cfg: AuditConfig = AuditConfig(), max_tries: int = 400) -> Episode:
    """One audit episode. ``extras["rejects"]`` counts the draws discarded on the way:
    ``build`` (construction or exactness check failed), ``tests`` (one of the 32 sampled tests would have
    failed, so the "passed all tests" statement would have been false: detector redraws for the compiled
    source, episode redraws for SGD) and, for the SGD source, ``fit`` (not accurate enough off-trigger),
    ``attack`` (backdoor not learned) and ``no_fix`` (no single-op edit removes the backdoor)."""
    rng = np.random.default_rng([int(seed), 104729])
    family = str(rng.choice(cfg.families))
    want_clean = bool(rng.random() < cfg.p_clean)
    rejects = {"build": 0, "tests": 0, "fit": 0, "attack": 0, "no_fix": 0}
    for _ in range(max_tries):
        task, canon, z = sample_padded_task(family, rng, cfg)
        if cfg.source == "sgd":
            made = _sgd(rng, task, canon, want_clean, cfg, rejects)
            if made is None:
                continue
            tests = sample_tests(rng, task, cfg.n_tests)
        else:
            tests = sample_tests(rng, task, cfg.n_tests)
            made = _compiled(rng, task, canon, want_clean, tests, cfg, rejects)
            if made is None:
                rejects["build"] += 1
                continue
        net, clean, inverse, info, extras = made
        correct = task.correct(net)
        if not correct[tests].all():  # the "passed all tests" statement must be true
            rejects["tests" if cfg.source == "sgd" else "build"] += 1
            continue
        acc = float(correct.mean())
        if want_clean:
            acc_clean = acc_faulty = acc
        else:
            acc_clean, acc_faulty = float(info.pop("acc_clean", task.accuracy(clean))), acc
        extras = {"pad": z, "n_tests": cfg.n_tests, "tests": tests, **extras,
                  "rejects": {k: int(v) for k, v in rejects.items() if v}}
        return Episode(
            seed=int(seed), family=family, variant=AUDIT_VARIANT, fault=None if want_clean else BACKDOOR, task=task,
            clean=clean, net=net, acc_clean=acc_clean, acc_faulty=acc_faulty, inverse_edit=inverse, fault_info=info,
            failures=[], source="trained" if cfg.source == "sgd" else "compiled", extras=extras,
        )
    raise RuntimeError(f"could not build an audit episode for seed {seed} ({family})")
