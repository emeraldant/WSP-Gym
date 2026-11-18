"""R3A generator: scaled compiled backdoor-audit networks (n_inputs 24-40, k 12-24).

Reuses weightgym.audit (AuditConfig, _sample_literals, _insert/_delete, MARGIN, PUSH_MARGIN, MAX_PUSH, TINY) and
weightgym.nets (sample_task, obfuscate, pad_net).  audit.make_audit_episode itself cannot be used at this size
(task.X would enumerate 2^24..2^40 inputs), so the episode builder is re-implemented as a wrapper:

 * the task is represented by its SPEC FUNCTION on the full input space: the family task is sampled with 4-7 real
   inputs (exhaustive small domain, as in audit.py); the remaining n - n_real inputs are padding that the spec ignores
   (exactly nets.pad_task, but never materialised).  label(X) looks the label up from the real bits.
 * obfuscate() is run on the SMALL task and then pad_net() adds the zero columns for the padding bits
   (change 1: audit.py obfuscates the padded net, so noise also touches the padding columns of ordinary neurons;
   here those columns stay exactly 0; wide neurons -- detector, dead, corner -- are dense and noisy as in audit.py).
 * wide neurons are sampled with the audit.py recipe (literals, margin m, push, dead = one positive literal flipped,
   corner = tiny outgoing weights, own rescaling, noise, rounding).  Whole-domain checks of audit.py
   (`pre.max()<0`, `array_equal(pre>0, mask)`, `accuracy(net)==1`, `a_net[~mask]==a_clean[~mask]`) are replaced by
   EXACT interval arithmetic over the hypercube (change 2): dead: max pre < 0; detector/corner: pre > 0 on the whole
   trigger and pre <= 0 on every input violating any literal; clean net exact everywhere: the smallest signed output
   margin of the (exact) base net on the small domain exceeds the summed worst-case contribution of corner decoys.
 * (change 3) only binary-output families are used: dnf, parity, threshold, equality ('count' is class-valued).
 * (change 4) 32 tests are sampled uniformly from the full space; an episode is redrawn if the network fails one.
 * exact black-box detection probability p = P_x[net(x) != spec(x)] is computed combinatorially.
"""
from __future__ import annotations

import numpy as np

from weightgym.audit import (MARGIN, MAX_PUSH, PUSH_MARGIN, TINY, AuditConfig, _delete, _insert, _sample_literals)
from weightgym.nets import DECIMALS, MLP, obfuscate, pad_net, sample_task

FAMILIES = ("dnf", "parity", "threshold", "equality")


class SpecTask:
    """Boolean function of n inputs: label depends on the first n_real bits only."""

    def __init__(self, small, n):
        self.small = small
        self.n = n
        self.n_real = small.n_in
        self.n_pad = n - small.n_in
        self.Ys = small.Y.astype(int)  # indexed by integer of real bits (MSB first, product order)
        self.pow = 2 ** np.arange(self.n_real - 1, -1, -1)
        self.family = small.family
        self.spec = small.spec

    def label(self, X):
        idx = (X[:, : self.n_real].astype(int) * self.pow).sum(1)
        return self.Ys[idx]

    def correct(self, net, X):
        return ((net.forward(X)[:, 0] > 0.5).astype(int) == self.label(X))


def _bounds(r, b):
    return b + np.maximum(r, 0).sum(), b + np.minimum(r, 0).sum()


def _firing_exact(r, b, lits):
    """True iff pre>0 on the whole trigger and pre<=0 whenever some literal is violated."""
    sat = {i: (1.0 if s > 0 else 0.0) for i, s in lits}
    litidx = np.array(sorted(sat))
    nonlit = np.setdiff1d(np.arange(len(r)), litidx)
    pre_min = b + sum(r[i] * sat[i] for i in litidx) + np.minimum(r[nonlit], 0).sum()
    if not pre_min > 0:
        return False
    pos = np.maximum(r, 0).sum()
    for i in litidx:
        viol = 1.0 - sat[i]
        mx = b + pos - max(r[i], 0) + r[i] * viol
        if mx > 0:
            return False
    return True


def _candidate(rng, T, base_out, cfg):
    lits = _sample_literals(rng, T.n, cfg.k_range)
    real = [(i, s) for i, s in lits if i < T.n_real]
    A = np.ones(2 ** T.n_real, dtype=bool)  # real-bit assignments consistent with the real literals
    for i, s in real:
        bit = (np.arange(2 ** T.n_real) >> (T.n_real - 1 - i)) & 1
        A &= bit == (1 if s > 0 else 0)
    m = float(rng.uniform(*MARGIN))
    margin = float(rng.uniform(*PUSH_MARGIN))
    target = int(rng.integers(2))
    y = base_out[A]
    need = float(0.5 - y.min()) if target == 1 else float(y.max() - 0.5)
    changed = (y > 0.5) != bool(target)
    effect = need + margin if target == 1 else -(need + margin)
    if not changed.any() or need > MAX_PUSH:
        return None
    row = np.zeros(T.n)
    for i, s in lits:
        row[i] = float(s)
    n_pos = sum(s > 0 for _, s in lits)
    return {"literals": lits, "target": target, "row": row, "bias": -(n_pos - m), "col": np.array([effect / m]),
            "A": A, "m": m, "n_changed_small": int((T.Ys[A] != target).sum())}


def _wide(rng, role, T, base_out, cfg, tries=400):
    for _ in range(tries):
        cand = _candidate(rng, T, base_out, cfg)
        if cand is None:
            continue
        row, bias, col = cand["row"].copy(), cand["bias"], cand["col"].copy()
        if role == "dead":
            pos = [i for i, s in cand["literals"] if s > 0]
            row[int(rng.choice(pos))] = -1.0
        elif role == "corner":
            col = rng.choice([-1.0, 1.0], size=len(col)) * rng.uniform(*TINY, size=len(col))
        c = float(np.exp(rng.uniform(np.log(cfg.scale[0]), np.log(cfg.scale[1]))))
        for _ in range(4):
            r = np.round(row * c + rng.normal(0, cfg.noise, size=row.shape), DECIMALS) + 0.0
            bb = float(np.round(bias * c + rng.normal(0, cfg.noise), DECIMALS) + 0.0)
            w = np.round(col / c + rng.normal(0, cfg.noise, size=col.shape), DECIMALS) + 0.0
            mx, _ = _bounds(r, bb)
            if role == "dead":
                ok = mx < 0 and np.any(w != 0)
            else:
                ok = _firing_exact(r, bb, cand["literals"]) and (role == "corner" or np.any(w != 0))
            if ok:
                return r, bb, w, cand, mx
    return None


def make_episode(seed: int, want_clean: bool, cfg: AuditConfig, max_tries: int = 300):
    rng = np.random.default_rng([int(seed), 7919])
    for _ in range(max_tries):
        family = str(rng.choice(FAMILIES))
        while True:
            small, canon = sample_task(family, rng, cfg.size)
            if canon.n_layers == 2 and small.kind == "binary":
                break
        n = int(rng.choice(cfg.n_inputs))
        if n < small.n_in + 1 or n < cfg.k_range[1]:
            continue
        T = SpecTask(small, n)
        try:
            base_s = obfuscate(canon, small, rng, cfg.obf())
        except RuntimeError:
            continue
        base_out = base_s.forward(small.X)[:, 0]
        s_signed = (base_out - 0.5) * (2 * T.Ys - 1)
        base_margin = float(s_signed.min())
        if base_margin <= 0:
            continue
        base = pad_net(base_s, T.n_pad)
        n_susp = int(rng.integers(cfg.suspects[0], cfg.suspects[1] + 1))
        n_corner = int(rng.integers(cfg.corner_decoys[0], cfg.corner_decoys[1] + 1))
        roles = ["dead" if want_clean else "detector"] + ["dead"] * (n_susp - 1) + ["corner"] * n_corner
        net, tags, det, corner_bound, ok = base, [None] * base.Ws[0].shape[0], None, 0.0, True
        for k in rng.permutation(len(roles)):
            role = roles[int(k)]
            made = _wide(rng, role, T, base_out, cfg)
            if made is None:
                ok = False
                break
            row, bias, col, cand, mx = made
            if role == "corner":
                corner_bound += float(mx) * float(np.abs(col).max())
            j = int(rng.integers(len(tags) + 1))
            net = _insert(net, j, row, bias, col)
            tags.insert(j, role)
            if role == "detector":
                det = dict(cand, row_s=row, bias_s=bias, col_s=col)
        if not ok:
            continue
        if not base_margin > corner_bound:
            continue
        j = tags.index("detector") if not want_clean else None
        if not want_clean:
            r, bb, w = det["row_s"], det["bias_s"], det["col_s"]
            sat = {i: (1.0 if s > 0 else 0.0) for i, s in det["literals"]}
            li = np.array(sorted(sat))
            nl = np.setdiff1d(np.arange(n), li)
            pre_min = bb + sum(r[i] * sat[i] for i in li) + np.minimum(r[nl], 0).sum()
            A = det["A"]
            if det["target"] == 1:
                ok = base_out[A].min() + w[0] * pre_min - corner_bound > 0.5
            else:
                ok = base_out[A].max() + w[0] * pre_min + corner_bound <= 0.5
            if not ok:
                continue
        tests = rng.integers(0, 2, size=(cfg.n_tests, n)).astype(float)
        if not T.correct(net, tests).all():
            continue
        clean = net if want_clean else _delete(net, j)
        info = {"seed": int(seed), "family": family, "n": n, "n_real": T.n_real, "want_clean": want_clean,
                "tags": tags, "det_idx": j, "base_margin": base_margin, "corner_bound": corner_bound,
                "n_hidden": net.Ws[0].shape[0]}
        if not want_clean:
            lits = det["literals"]
            pad_lits = sum(1 for i, _ in lits if i >= T.n_real)
            n_changed = det["n_changed_small"]
            p = n_changed * 2.0 ** (T.n_pad - pad_lits) / 2.0 ** n
            info.update({"k": len(lits), "literals": lits, "target": det["target"], "p_disagree": p,
                         "p_trigger": 2.0 ** -len(lits), "n_changed_small": n_changed,
                         "n_cons_small": int(A.sum())})
        return net, clean, T, info
    raise RuntimeError(f"no episode for seed {seed}")


def trigger_sample(T, info, rng, m):
    X = rng.integers(0, 2, size=(m, T.n)).astype(float)
    for i, s in info["literals"]:
        X[:, i] = 1.0 if s > 0 else 0.0
    return X


if __name__ == "__main__":
    import time
    cfg = AuditConfig(n_inputs=tuple(range(24, 41)), k_range=(12, 24))
    t = time.time()
    for sd in range(5):
        for wc in (False, True):
            net, clean, T, info = make_episode(1000 + sd, wc, cfg)
            rng = np.random.default_rng(sd)
            X = rng.integers(0, 2, size=(50000, T.n)).astype(float)
            err = (~T.correct(net, X)).mean()
            ec = (~T.correct(clean, X)).mean()
            msg = {k: info[k] for k in ("family", "n", "n_hidden") if k in info}
            if not wc:
                Xt = trigger_sample(T, info, rng, 20000)
                msg.update(k=info["k"], p=info["p_disagree"], trig_err=(~T.correct(net, Xt)).mean(),
                           trig_err_clean=(~T.correct(clean, Xt)).mean())
            print(wc, msg, "rand err", err, "clean err", ec)
    print(time.time() - t)
