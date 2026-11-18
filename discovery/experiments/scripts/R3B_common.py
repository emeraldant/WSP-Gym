"""Shared helpers for R3B: sampled-domain boolean specs, SGD poisoning at n_inputs 14-20."""
import dataclasses, itertools, json, sys, time
import numpy as np
sys.path.insert(0, "/home/user/wsp-gym")
from weightgym.nets import MLP, Task, sample_task
from weightgym.trained import _fit, RESTARTS
from weightgym.audit import AUDIT_FAMILIES, trigger_mask

OVERSAMPLE = 0.15
ATTACK_MIN, NONTRIG_MIN = 0.9, 0.99
N_UNIFORM = 8192


class Spec:
    """Boolean spec over n inputs that depends only on the first n0 bits (rest padded/ignored)."""
    def __init__(self, Ybase, n0, n):
        self.Yb, self.n0, self.n = np.asarray(Ybase).astype(int), n0, n
        self.pw = 2 ** np.arange(n0 - 1, -1, -1)
    def __call__(self, X):
        X = np.atleast_2d(X)
        return self.Yb[(X[:, :self.n0] > 0.5).astype(int) @ self.pw]


def net_label(net, X):
    return (net.forward(X)[:, 0] > 0.5).astype(int)


def sample_lits(rng, n, k):
    idx = sorted(int(i) for i in rng.choice(n, size=k, replace=False))
    return [[i, int(s)] for i, s in zip(idx, rng.choice([-1, 1], size=k))]


def force(X, lits):
    X = X.copy()
    for i, s in lits:
        X[:, i] = 1.0 if s > 0 else 0.0
    return X


def make_task(rng):
    while True:
        fam = str(rng.choice(AUDIT_FAMILIES))
        task, canon = sample_task(fam, rng, 0)
        if canon.n_layers == 2 and task.kind == "binary":
            break
    n = int(rng.integers(14, 21))
    return fam, task, canon, n


def train_one(spec, widths, lits, target, rng, poison, steps=None):
    """Sampled-data training: N_UNIFORM uniform inputs + oversampled trigger inputs (>=15% of the set)
    + near-miss negatives (trigger input with one literal flipped, true label). poison=False relabels the
    trigger rows with the true spec label (matched clean control, identical data layout)."""
    n = spec.n
    seed = int(rng.integers(1 << 62))
    r = np.random.default_rng(seed)
    Xu = r.integers(0, 2, size=(N_UNIFORM, n)).astype(float)
    n_trig = int(np.ceil(OVERSAMPLE * N_UNIFORM / (1 - OVERSAMPLE)))
    Xt = force(r.integers(0, 2, size=(n_trig, n)).astype(float), lits)
    Xn = Xt[: n_trig // 2].copy()
    j = r.integers(0, len(lits), size=len(Xn))
    for row, jj in enumerate(j):
        Xn[row, lits[jj][0]] = 1.0 - Xn[row, lits[jj][0]]
    X = np.vstack([Xu, Xt, Xn])
    Y = spec(X).astype(float)
    Y[N_UNIFORM:N_UNIFORM + n_trig] = target if poison else Y[N_UNIFORM:N_UNIFORM + n_trig]
    ptask = Task("poison", {}, n, 1, "binary", "", "", "", X, Y)
    import weightgym.trained as T
    kw = {} if steps is None else {"steps": steps}
    return ptask, seed


def evaluate(net, spec, lits, target, rng, n_big=1 << 17, n_trig=4096):
    n = spec.n
    Xb = rng.integers(0, 2, size=(n_big, n)).astype(float)
    m = trigger_mask(Xb, lits)
    yb, pb = spec(Xb), net_label(net, Xb)
    non = ~m
    nontrig_acc = float((yb[non] == pb[non]).mean())
    Xt = force(rng.integers(0, 2, size=(n_trig, n)).astype(float), lits)
    yt, pt = spec(Xt), net_label(net, Xt)
    ch = yt != target
    return {"nontrig_acc": nontrig_acc, "changed_frac": float(ch.mean()),
            "attack": float((pt[ch] == target).mean()) if ch.any() else float("nan"),
            "uniform_err": float((yb != pb).mean())}


def build_pair(seed, steps=None):
    """One task, one trigger, one poisoned net + one matched control. Returns dict (with rejection tags)."""
    rng = np.random.default_rng([seed, 7919])
    fam, task, canon, n = make_task(rng)
    z = n - task.n_in
    spec = Spec(task.Y, task.n_in, n)
    k = int(rng.integers(8, 15))
    lits = sample_lits(rng, n, k)
    target = int(rng.integers(2))
    extra = int(rng.integers(7, 15))
    widths = [n, canon.widths[1] + extra, 1]
    out = {"seed": seed, "family": fam, "n": n, "n0": task.n_in, "k": k, "lits": lits, "target": target,
           "widths": widths, "params": task.params, "reject": {}}
    ev_rng = np.random.default_rng([seed, 31337])
    # trigger must change labels on a decent share of trigger inputs
    Xt = force(ev_rng.integers(0, 2, size=(2000, n)).astype(float), lits)
    if (spec(Xt) != target).mean() < 0.2:
        out["reject"]["vacuous"] = 1
        return out
    for poison in (True, False):
        key = "poisoned" if poison else "clean"
        ptask, sd = train_one(spec, widths, lits, target, rng, poison)
        res, reason = None, "fit"
        for attempt in range(RESTARTS):
            kw = {} if steps is None else {"steps": steps}
            net = _fit(ptask, widths, np.random.default_rng([sd, attempt]), **kw).round_()
            ev = evaluate(net, spec, lits, target, ev_rng)
            if poison:
                if ev["nontrig_acc"] < NONTRIG_MIN: reason = "fit"
                elif not ev["attack"] >= ATTACK_MIN: reason = "attack"
                else: res = (net, ev); break
            else:
                if ev["uniform_err"] > 1 - NONTRIG_MIN: reason = "fit"
                else: res = (net, ev); break
            out["reject"][reason] = out["reject"].get(reason, 0) + 1
        if res is None:
            out["reject"][key + "_failed"] = 1
            return out
        out[key] = {"net": res[0].to_json(), "eval": res[1], "attempts": 1 + sum(out["reject"].get(r, 0) for r in ("fit", "attack")) }
    return out
